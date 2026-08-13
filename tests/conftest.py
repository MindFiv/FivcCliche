"""Shared pytest fixtures for FivcCliche tests."""

import os
import tempfile
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel

from fivccliche.modules.agent_chats.models import UserChat, UserChatMessage  # noqa: F401
from fivccliche.modules.agent_configs.models import (  # noqa: F401
    UserAgent,
    UserEmbedding,
    UserLLM,
    UserQuestion,
    UserSkill,
    UserTool,
)
from fivccliche.modules.users.models import User  # noqa: F401
from fivccliche.services.implements.modules import ModuleSiteImpl
from fivccliche.utils import deps
from fivccliche.utils.deps import get_db_session_async
from fivcglue.implements.utils import load_component_site


def make_api_client(
    modules: list[str],
    *,
    extra_users: list[dict] | None = None,
) -> Iterator[TestClient]:
    """Yield a TestClient with a temp SQLite DB, admin user, and session overrides.

    Each extra_users dict is passed to ``create_user_async`` after the admin user.
    The client exposes ``admin_user``, ``async_session``, and ``loop`` for tests
    that talk to the DB directly.
    """
    import asyncio

    from fivccliche.modules.users import methods

    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db.close()
    database_url = f"sqlite+aiosqlite:///{temp_db.name}"

    async def create_tables():
        engine = create_async_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=NullPool,
        )
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        return engine

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        engine = loop.run_until_complete(create_tables())
        async_session = AsyncSession(engine, expire_on_commit=False)

        components_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "fivccliche",
            "settings",
            "services.yml",
        )
        component_site = load_component_site(filename=components_path, fmt="yaml")
        module_site = ModuleSiteImpl(component_site, modules=modules)
        app = module_site.create_application()

        async def override_get_db_session_async():
            yield async_session

        app.dependency_overrides[get_db_session_async] = override_get_db_session_async

        admin_user = loop.run_until_complete(
            methods.create_user_async(
                async_session,
                username="admin",
                email="admin@example.com",
                password="admin123",
                is_superuser=True,
            )
        )
        for extra in extra_users or []:
            loop.run_until_complete(methods.create_user_async(async_session, **extra))

        @asynccontextmanager
        async def override_get_db_session_context_async(session=None):
            yield async_session

        with patch.object(
            deps, "get_db_session_context_async", override_get_db_session_context_async
        ):
            with TestClient(app) as test_client:
                test_client.admin_user = admin_user  # type: ignore[attr-defined]
                test_client.async_session = async_session  # type: ignore[attr-defined]
                test_client.loop = loop  # type: ignore[attr-defined]
                yield test_client

        async def cleanup():
            await async_session.close()
            await engine.dispose()

        loop.run_until_complete(cleanup())
    finally:
        loop.close()
        try:
            Path(temp_db.name).unlink()
        except OSError:
            pass
