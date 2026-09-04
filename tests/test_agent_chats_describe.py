"""Tests for filling an empty chat description from the first query."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fivcplayground.agents import AgentRunContent
from sqlalchemy.ext.asyncio import AsyncSession

from fivccliche.modules.agent_chats.jobs.describe import (
    ChatDescribeJob,
    _fallback_description,
)
from fivccliche.modules.agent_chats.models import UserChat


@pytest.fixture
async def test_user(session: AsyncSession):
    from fivccliche.modules.users import utils as user_methods

    user = await user_methods.create_user_async(
        session,
        username="descuser",
        email="desc@example.com",
        password="password123",
    )
    await session.commit()
    await session.refresh(user)
    return user


async def _add_chat(session: AsyncSession, user_uuid: str, **kwargs) -> UserChat:
    chat = UserChat(user_uuid=user_uuid, agent_id="agent", **kwargs)
    session.add(chat)
    await session.commit()
    await session.refresh(chat)
    return chat


async def _run_describe(user, provider, chat_uuid: str, query_text: str) -> None:
    with patch(
        "fivccliche.modules.agent_chats.jobs.describe.get_config_provider_async",
        AsyncMock(return_value=provider),
    ):
        await ChatDescribeJob(MagicMock()).run_async(
            chat_uuid, user_uuid=user.uuid, query_text=query_text
        )


@asynccontextmanager
async def _mutex_ok(*args, **kwargs):
    yield MagicMock()


@asynccontextmanager
async def _mutex_missing(*args, **kwargs):
    yield None


class _SessionCtx:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *args):
        return None


def _session_patch(session: AsyncSession):
    return patch(
        "fivccliche.modules.agent_chats.jobs.describe.get_db_session_context_async",
        side_effect=lambda: _SessionCtx(session),
    )


def _mutex_patch(factory=_mutex_ok):
    return patch(
        "fivccliche.modules.agent_chats.jobs.describe.get_mutex_context_async",
        factory,
    )


class TestFallbackDescription:
    def test_short_query_unchanged(self):
        assert _fallback_description("hello world") == "hello world"

    def test_breaks_at_whitespace(self):
        query = "word " + ("x" * 90)
        result = _fallback_description(query)
        assert result == "word"
        assert len(result) <= 80


class TestFillChatDescription:
    @pytest.mark.asyncio
    async def test_uses_llm_title_when_describe_model_exists(
        self, session: AsyncSession, test_user
    ):
        chat = await _add_chat(session, test_user.uuid)
        agent = MagicMock()
        agent.run_async = AsyncMock(return_value=AgentRunContent(text="  短标题  "))
        backend = MagicMock()
        backend.create_agent_async = AsyncMock(return_value=agent)
        repo = MagicMock()
        repo.get_model_config_async = AsyncMock(return_value=True)
        provider = MagicMock()
        provider.get_agent_backend.return_value = backend
        provider.get_model_backend.return_value = MagicMock()
        provider.get_model_repository.return_value = repo

        with _session_patch(session), _mutex_patch():
            await _run_describe(
                test_user,
                provider,
                chat.uuid,
                "帮我看看这段代码为什么报错",
            )

        await session.refresh(chat)
        assert chat.description == "短标题"
        backend.create_agent_async.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_truncates_llm_title_to_20_chars(self, session: AsyncSession, test_user):
        long_title = "这是一段明显超过二十个字符限制的聊天标题内容"
        chat = await _add_chat(session, test_user.uuid)
        agent = MagicMock()
        agent.run_async = AsyncMock(return_value=AgentRunContent(text=long_title))
        backend = MagicMock()
        backend.create_agent_async = AsyncMock(return_value=agent)
        repo = MagicMock()
        repo.get_model_config_async = AsyncMock(return_value=True)
        provider = MagicMock()
        provider.get_agent_backend.return_value = backend
        provider.get_model_backend.return_value = MagicMock()
        provider.get_model_repository.return_value = repo

        with _session_patch(session), _mutex_patch():
            await _run_describe(test_user, provider, chat.uuid, "帮我看看这段代码为什么报错")

        await session.refresh(chat)
        assert chat.description == long_title[:20]
        assert len(chat.description) == 20

    @pytest.mark.asyncio
    async def test_truncates_query_when_describe_model_missing(
        self, session: AsyncSession, test_user
    ):
        chat = await _add_chat(session, test_user.uuid)
        repo = MagicMock()
        repo.get_model_config_async = AsyncMock(return_value=None)
        provider = MagicMock()
        provider.get_model_repository.return_value = repo

        with _session_patch(session), _mutex_patch():
            await _run_describe(test_user, provider, chat.uuid, "short question")

        await session.refresh(chat)
        assert chat.description == "short question"
        provider.get_agent_backend.assert_not_called()

    @pytest.mark.asyncio
    async def test_leaves_existing_description(self, session: AsyncSession, test_user):
        chat = await _add_chat(session, test_user.uuid, description="keep me")
        provider = MagicMock()

        with _session_patch(session), _mutex_patch():
            await _run_describe(test_user, provider, chat.uuid, "new question")

        await session.refresh(chat)
        assert chat.description == "keep me"
        provider.get_model_repository.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_empty_llm_title(self, session: AsyncSession, test_user):
        chat = await _add_chat(session, test_user.uuid)
        agent = MagicMock()
        agent.run_async = AsyncMock(return_value=AgentRunContent(text="   "))
        backend = MagicMock()
        backend.create_agent_async = AsyncMock(return_value=agent)
        repo = MagicMock()
        repo.get_model_config_async = AsyncMock(return_value=True)
        provider = MagicMock()
        provider.get_agent_backend.return_value = backend
        provider.get_model_backend.return_value = MagicMock()
        provider.get_model_repository.return_value = repo

        with _session_patch(session), _mutex_patch():
            await _run_describe(test_user, provider, chat.uuid, "hello")

        await session.refresh(chat)
        assert chat.description is None

    @pytest.mark.asyncio
    async def test_skips_when_mutex_not_acquired(self, session: AsyncSession, test_user):
        chat = await _add_chat(session, test_user.uuid)
        provider = MagicMock()

        with _session_patch(session), _mutex_patch(_mutex_missing):
            await _run_describe(test_user, provider, chat.uuid, "hello")

        await session.refresh(chat)
        assert chat.description is None
        provider.get_model_repository.assert_not_called()
