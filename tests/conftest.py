"""Shared pytest fixtures for FivcCliche tests."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch
from urllib.parse import urlparse, urlunparse

import pytest
from fastapi.testclient import TestClient
from pg0 import Pg0, Pg0AlreadyRunningError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
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

_TEST_PG0_NAME = "fivccliche-test"
_pg0_for_tests: Pg0 | None = None


def to_asyncpg_url(uri: str) -> str:
    if uri.startswith("postgresql://"):
        return "postgresql+asyncpg://" + uri[len("postgresql://") :]
    if uri.startswith("postgres://"):
        return "postgresql+asyncpg://" + uri[len("postgres://") :]
    return uri


def with_database(url: str, database: str) -> str:
    parsed = urlparse(url)
    return urlunparse(parsed._replace(path=f"/{database}"))


def _start_pg0(name: str) -> Pg0:
    pg = Pg0(name=name)
    try:
        pg.start()
    except Pg0AlreadyRunningError:
        pass
    return pg


@pytest.fixture(scope="session", autouse=True)
def pg0_instance() -> Iterator[Pg0]:
    global _pg0_for_tests
    pg = _start_pg0(_TEST_PG0_NAME)
    _pg0_for_tests = pg
    try:
        yield pg
    finally:
        _pg0_for_tests = None
        pg.stop()


def create_test_database(pg: Pg0) -> tuple[str, str]:
    """Create an isolated database; return (async_url, database_name)."""
    db_name = f"t_{uuid.uuid4().hex}"
    pg.execute(f'CREATE DATABASE "{db_name}"')
    return with_database(to_asyncpg_url(pg.uri), db_name), db_name


def drop_test_database(pg: Pg0, db_name: str) -> None:
    pg.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')


async def open_test_engine(database_url: str) -> AsyncEngine:
    engine = create_async_engine(database_url, poolclass=NullPool, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    return engine


@pytest.fixture
async def database_url(pg0_instance: Pg0) -> AsyncIterator[str]:
    """Allocate an isolated Postgres database URL (schema created) for one test."""
    url, db_name = create_test_database(pg0_instance)
    engine = await open_test_engine(url)
    await engine.dispose()
    try:
        yield url
    finally:
        drop_test_database(pg0_instance, db_name)


@pytest.fixture
async def session(database_url: str) -> AsyncIterator[AsyncSession]:
    """Yield an AsyncSession against a fresh Postgres database."""
    engine = create_async_engine(database_url, poolclass=NullPool, echo=False)
    async_session = AsyncSession(engine, expire_on_commit=False)
    try:
        yield async_session
    finally:
        await async_session.close()
        await engine.dispose()


def make_api_client(
    modules: list[str],
    *,
    extra_users: list[dict[str, Any]] | None = None,
) -> Iterator[TestClient]:
    """Yield a TestClient with an isolated Postgres DB, admin user, and session overrides.

    Each extra_users dict is passed to ``create_user_async`` after the admin user.
    The client exposes ``admin_user``, ``async_session``, and ``loop`` for tests
    that talk to the DB directly via ``loop.run_until_complete``.

    HTTP handlers get a fresh AsyncSession per request (asyncpg is loop-bound).
    """
    import asyncio

    from fivccliche.modules.users import utils as methods

    if _pg0_for_tests is None:
        raise RuntimeError("pg0 test instance is not running")
    pg = _pg0_for_tests

    url, db_name = create_test_database(pg)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Create schema once; seed session is only for test-side DB access.
        seed_engine = loop.run_until_complete(open_test_engine(url))
        async_session = AsyncSession(seed_engine, expire_on_commit=False)

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
            engine = create_async_engine(url, poolclass=NullPool)
            session = AsyncSession(engine, expire_on_commit=False)
            try:
                yield session
                await session.commit()
            finally:
                await session.close()
                await engine.dispose()

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
        loop.run_until_complete(async_session.commit())

        @asynccontextmanager
        async def override_get_db_session_context_async(session=None):
            if session is not None:
                yield session
                return
            engine = create_async_engine(url, poolclass=NullPool)
            db_session = AsyncSession(engine, expire_on_commit=False)
            try:
                yield db_session
                await db_session.commit()
            finally:
                await db_session.close()
                await engine.dispose()

        with (
            patch.object(
                deps, "get_db_session_context_async", override_get_db_session_context_async
            ),
            patch(
                "fivccliche.modules.users.services.get_db_session_context_async",
                override_get_db_session_context_async,
            ),
        ):
            with TestClient(app) as test_client:
                test_client.admin_user = admin_user  # type: ignore[attr-defined]
                test_client.async_session = async_session  # type: ignore[attr-defined]
                test_client.loop = loop  # type: ignore[attr-defined]
                yield test_client

        async def cleanup():
            await async_session.close()
            await seed_engine.dispose()

        try:
            loop.run_until_complete(cleanup())
        except RuntimeError:
            try:
                seed_engine.sync_engine.dispose()
            except Exception:
                pass
        drop_test_database(pg, db_name)
    finally:
        loop.close()
