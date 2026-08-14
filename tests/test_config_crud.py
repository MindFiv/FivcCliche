"""Parametrized tests for user-scoped config SQL helpers."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from fivccliche.modules.agent_configs import models, utils as methods
from fivccliche.modules.users.models import User


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
async def owner(session: AsyncSession) -> User:
    user = User(
        uuid="user123",
        username="owner",
        email="owner@example.com",
        hashed_password="x",
    )
    session.add(user)
    await session.commit()
    return user


@pytest.mark.parametrize(("model", "fields"), CONFIG_MODELS)
class TestUserScopedConfigHelpers:
    async def test_get_list_count(self, session: AsyncSession, owner: User, model, fields):
        created = model(user_uuid=owner.uuid, **fields)
        session.add(created)
        await session.commit()
        await session.refresh(created)

        fetched = await methods.get_user_scoped_async(
            session, model, owner.uuid, config_id=fields["id"]
        )
        assert fetched is not None
        assert fetched.uuid == created.uuid

        by_uuid = await methods.get_user_scoped_async(
            session, model, owner.uuid, config_uuid=created.uuid
        )
        assert by_uuid is not None
        assert by_uuid.id == fields["id"]

        listed = await methods.list_user_scoped_async(session, model, owner.uuid)
        assert [item.id for item in listed] == [fields["id"]]
        assert await methods.count_user_scoped_async(session, model, owner.uuid) == 1

        assert (
            await methods.get_user_scoped_async(
                session, model, "other-user", config_id=fields["id"]
            )
            is None
        )

    async def test_get_requires_exactly_one_identity(
        self, session: AsyncSession, owner: User, model, fields
    ):
        with pytest.raises(ValueError, match="Exactly one"):
            await methods.get_user_scoped_async(session, model, owner.uuid)
        with pytest.raises(ValueError, match="Exactly one"):
            await methods.get_user_scoped_async(
                session, model, owner.uuid, config_uuid="x", config_id="y"
            )
