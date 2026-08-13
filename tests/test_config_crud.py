"""Parametrized tests for user-scoped config SQL helpers in methods."""

import tempfile
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from fivccliche.modules.agent_configs import methods, models
from fivccliche.modules.users.models import User  # noqa: F401


CONFIG_MODELS = [
    pytest.param(
        models.UserEmbedding,
        {
            "id": "emb-1",
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_key": "k",
        },
        id="embedding",
    ),
    pytest.param(
        models.UserLLM,
        {"id": "llm-1", "provider": "openai", "model": "gpt-4", "api_key": "k"},
        id="llm",
    ),
    pytest.param(
        models.UserAgent,
        {"id": "agent-1", "model_id": "llm-1"},
        id="agent",
    ),
    pytest.param(
        models.UserTool,
        {"id": "tool-1", "transport": "stdio"},
        id="tool",
    ),
    pytest.param(
        models.UserSkill,
        {"id": "skill-1", "description": "a skill", "instructions": "do things"},
        id="skill",
    ),
    pytest.param(
        models.UserQuestion,
        {"id": "q-1", "question": "why?"},
        id="question",
    ),
]


@pytest.fixture
async def session():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{Path(tmpdir) / 'test.db'}",
            connect_args={"check_same_thread": False},
            poolclass=NullPool,
        )
        async with engine.begin() as conn:
            await conn.execute(text("PRAGMA foreign_keys=OFF"))
            from sqlmodel import SQLModel

            await conn.run_sync(SQLModel.metadata.create_all)
        async with AsyncSession(engine, expire_on_commit=False) as db_session:
            yield db_session
        await engine.dispose()


@pytest.mark.parametrize(("model", "fields"), CONFIG_MODELS)
class TestUserScopedConfigHelpers:
    async def test_get_list_count(self, session: AsyncSession, model, fields):
        created = model(user_uuid="user123", **fields)
        session.add(created)
        await session.commit()
        await session.refresh(created)

        fetched = await methods._get_user_scoped_async(
            session, model, "user123", config_id=fields["id"]
        )
        assert fetched is not None
        assert fetched.uuid == created.uuid

        by_uuid = await methods._get_user_scoped_async(
            session, model, "user123", config_uuid=created.uuid
        )
        assert by_uuid is not None
        assert by_uuid.id == fields["id"]

        listed = await methods._list_user_scoped_async(session, model, "user123")
        assert [item.id for item in listed] == [fields["id"]]
        assert await methods._count_user_scoped_async(session, model, "user123") == 1

        assert (
            await methods._get_user_scoped_async(
                session, model, "other-user", config_id=fields["id"]
            )
            is None
        )

    async def test_get_requires_exactly_one_identity(self, session: AsyncSession, model, fields):
        with pytest.raises(ValueError, match="Exactly one"):
            await methods._get_user_scoped_async(session, model, "user123")
        with pytest.raises(ValueError, match="Exactly one"):
            await methods._get_user_scoped_async(
                session, model, "user123", config_uuid="x", config_id="y"
            )
