"""Tests for chat-level memorize job and related methods."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi.testclient import TestClient
from fivcglue.implements.utils import load_component_site
from sqlalchemy.ext.asyncio import AsyncSession

from fivccliche.modules.agent_chats import utils as methods
from fivccliche.modules.agent_chats.jobs import ChatMemorizeJob
from fivccliche.modules.agent_chats.jobs.memorize import (
    _MEMORIZE_EXTRACT_PROMPT,
    _MEMORIZE_JOB_ID,
    _ChatMemorizeParseResult,
    _ChatMemorizeParser,
)
from fivccliche.modules.agent_chats.models import UserChat, UserChatMessage
from fivccliche.modules.agent_chats.services import ModuleImpl
from fivccliche.services.implements.modules import ModuleSiteImpl
from fivccliche.services.interfaces.agent_memories import MemoryRetainResult


@pytest.fixture
async def test_user(session: AsyncSession):
    from fivccliche.modules.users import utils as user_methods

    user = await user_methods.create_user_async(
        session,
        username="memuser",
        email="mem@example.com",
        password="password123",
    )
    await session.commit()
    await session.refresh(user)
    return user


def _older(hours: int = 25) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours)


def _recent() -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=1)


async def _add_chat(session: AsyncSession, user_uuid: str, **kwargs) -> UserChat:
    kwargs.setdefault("is_memorable", True)
    chat = UserChat(user_uuid=user_uuid, agent_id="agent", **kwargs)
    session.add(chat)
    await session.commit()
    await session.refresh(chat)
    return chat


async def _add_message(
    session: AsyncSession,
    chat_uuid: str,
    *,
    query: dict | None,
    reply: dict | None = None,
    created_at: datetime | None = None,
    is_memorized: bool = False,
    status: str = "completed",
) -> UserChatMessage:
    msg = UserChatMessage(
        chat_uuid=chat_uuid,
        query=query,
        reply=reply,
        status=status,
        is_memorized=is_memorized,
        created_at=created_at or _older(),
        completed_at=created_at or _older(),
    )
    session.add(msg)
    await session.commit()
    await session.refresh(msg)
    return msg


def _parser() -> _ChatMemorizeParser:
    chat = UserChat(user_uuid="user-1", agent_id="agent")
    return _ChatMemorizeParser(chat, created_at_to=datetime.now(timezone.utc))


class TestGetMemorizeContent:
    @pytest.mark.asyncio
    async def test_builds_user_and_assistant_json(self):
        msg = UserChatMessage(
            chat_uuid="c",
            query={"text": "帮我看看这段代码为什么报错"},
            reply={"text": "你的第 12 行变量未定义,应该是..."},
        )
        repo = MagicMock()
        repo.get_model_config_async = AsyncMock(return_value=None)
        provider = MagicMock()
        provider.get_model_repository.return_value = repo
        with patch(
            "fivccliche.modules.agent_chats.jobs.memorize.get_config_provider_async",
            AsyncMock(return_value=provider),
        ):
            result = await _parser()._extract_memories_async([msg])
        assert len(result) == 1
        assert json.loads(result[0]) == [
            {"role": "user", "content": "帮我看看这段代码为什么报错"},
            {"role": "assistant", "content": "你的第 12 行变量未定义,应该是..."},
        ]

    @pytest.mark.asyncio
    async def test_slash_query_without_user_returns_empty(self):
        msg = UserChatMessage(
            chat_uuid="c",
            query={"text": "/help"},
            reply={"text": "这里是帮助"},
        )
        assert await _parser()._extract_memories_async([msg]) == []

    @pytest.mark.asyncio
    async def test_slash_query_without_reply_returns_empty(self):
        msg = UserChatMessage(
            chat_uuid="c",
            query={"text": "/status"},
            reply=None,
        )
        assert await _parser()._extract_memories_async([msg]) == []


class TestMemorizeMethods:
    async def test_list_chats_filters_by_age_and_memorized(self, session: AsyncSession, test_user):
        chat_old = await _add_chat(session, test_user.uuid)
        chat_recent = await _add_chat(session, test_user.uuid)
        chat_done = await _add_chat(session, test_user.uuid)

        await _add_message(
            session,
            chat_old.uuid,
            query={"text": "old"},
            reply={"text": "ok"},
            created_at=_older(),
        )
        await _add_message(
            session,
            chat_recent.uuid,
            query={"text": "new"},
            reply={"text": "ok"},
            created_at=_recent(),
        )
        await _add_message(
            session,
            chat_done.uuid,
            query={"text": "done"},
            reply={"text": "ok"},
            created_at=_older(),
            is_memorized=True,
        )

        created_at_to = datetime.now(timezone.utc) - timedelta(hours=24)
        chats = await methods.list_unmemorized_chats_async(
            session, created_at_to=created_at_to, limit=50
        )
        assert [c.uuid for c in chats] == [chat_old.uuid]

    async def test_list_chats_requires_is_memorable(self, session: AsyncSession, test_user):
        chat_memorable = await _add_chat(session, test_user.uuid, is_memorable=True)
        chat_not_memorable = await _add_chat(session, test_user.uuid, is_memorable=False)

        await _add_message(
            session,
            chat_memorable.uuid,
            query={"text": "keep"},
            reply={"text": "ok"},
            created_at=_older(),
        )
        await _add_message(
            session,
            chat_not_memorable.uuid,
            query={"text": "skip"},
            reply={"text": "ok"},
            created_at=_older(),
        )

        created_at_to = datetime.now(timezone.utc) - timedelta(hours=24)
        chats = await methods.list_unmemorized_chats_async(
            session, created_at_to=created_at_to, limit=50
        )
        assert [c.uuid for c in chats] == [chat_memorable.uuid]

    async def test_list_chats_dedupes_multiple_matching_messages(
        self, session: AsyncSession, test_user
    ):
        chat = await _add_chat(
            session, test_user.uuid, is_memorable=True, context={"topic": "code"}
        )
        await _add_message(
            session,
            chat.uuid,
            query={"text": "first"},
            reply={"text": "ok"},
            created_at=_older(hours=26),
        )
        await _add_message(
            session,
            chat.uuid,
            query={"text": "second"},
            reply={"text": "ok"},
            created_at=_older(hours=25),
        )

        created_at_to = datetime.now(timezone.utc) - timedelta(hours=24)
        chats = await methods.list_unmemorized_chats_async(
            session, created_at_to=created_at_to, limit=50
        )
        assert [c.uuid for c in chats] == [chat.uuid]

    def test_list_unmemorized_chats_sql_omits_distinct_on_postgresql(self):
        import inspect

        source = inspect.getsource(methods.list_unmemorized_chats_async)
        assert "exists(" in source
        assert ".distinct(" not in source

    async def test_create_chat_is_memorable_flag(self, session: AsyncSession, test_user):
        default_chat = await methods.create_chat_async(
            session,
            user_uuid=test_user.uuid,
            agent_id="agent",
        )
        memorable_chat = await methods.create_chat_async(
            session,
            user_uuid=test_user.uuid,
            agent_id="agent",
            is_memorable=True,
        )
        assert default_chat.is_memorable is False
        assert memorable_chat.is_memorable is True

    async def test_list_messages_for_chat_and_mark(self, session: AsyncSession, test_user):
        chat = await _add_chat(session, test_user.uuid)
        msg = await _add_message(
            session,
            chat.uuid,
            query={"text": "q"},
            reply={"text": "a"},
            created_at=_older(),
        )
        created_at_to = datetime.now(timezone.utc) - timedelta(hours=24)
        found = await methods.list_unmemorized_chat_messages_async(
            session, chat.uuid, created_at_to=created_at_to
        )
        assert [m.uuid for m in found] == [msg.uuid]

        await methods.delete_unmemorized_chat_messages_async(
            session, chat.uuid, created_at_to=created_at_to
        )
        await session.commit()
        await session.refresh(msg)
        assert msg.is_memorized is True

        found_after = await methods.list_unmemorized_chat_messages_async(
            session, chat.uuid, created_at_to=created_at_to
        )
        assert found_after == []

    async def test_update_chat_message_is_memorized(self, session: AsyncSession, test_user):
        chat = await _add_chat(session, test_user.uuid)
        msg = await _add_message(session, chat.uuid, query={"text": "q"}, reply={"text": "a"})
        await methods.update_chat_message_async(session, msg, is_memorized=True)
        await session.commit()
        await session.refresh(msg)
        assert msg.is_memorized is True


class TestExtractMemorableContent:
    def test_extract_prompt_only_keeps_user_stated_facts(self):
        prompt = _MEMORIZE_EXTRACT_PROMPT
        assert "user turns" in prompt
        assert "Do not store assistant" in prompt
        assert "context only" in prompt

    def _provider_with_extraction(self, extraction, *, model_config: object | None = True):
        agent = MagicMock()
        agent.run_async = AsyncMock(return_value=extraction)
        backend = MagicMock()
        backend.create_agent_async = AsyncMock(return_value=agent)
        repo = MagicMock()
        repo.get_model_config_async = AsyncMock(return_value=model_config)
        provider = MagicMock()
        provider.get_agent_backend.return_value = backend
        provider.get_model_backend.return_value = MagicMock()
        provider.get_model_repository.return_value = repo
        return provider, backend, agent

    def _user_message(self) -> UserChatMessage:
        return UserChatMessage(
            chat_uuid="c",
            query={"text": "hi"},
            reply={"text": "hello"},
        )

    async def _parse(
        self,
        messages: list[UserChatMessage],
        provider: MagicMock,
    ) -> list[str]:
        parser = _parser()

        class _SessionCtx:
            async def __aenter__(self):
                session = MagicMock()
                session.commit = AsyncMock()
                return session

            async def __aexit__(self, *args):
                return None

        with (
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_db_session_context_async",
                side_effect=lambda: _SessionCtx(),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.utils.list_unmemorized_chat_messages_async",
                AsyncMock(return_value=messages),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.utils.delete_unmemorized_chat_messages_async",
                AsyncMock(),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_config_provider_async",
                AsyncMock(return_value=provider),
            ),
        ):
            async with parser as memories:
                return list(memories)

    @pytest.mark.asyncio
    async def test_strips_memories_when_should_retain(self):
        provider, backend, agent = self._provider_with_extraction(
            _ChatMemorizeParseResult(should_retain=True, memories=["a", " b ", ""])
        )
        messages = [self._user_message()]
        result = await self._parse(messages, provider)

        assert result == ["a", "b"]
        provider.get_model_repository.assert_called_once_with(user_uuid="user-1")
        provider.get_model_repository.return_value.get_model_config_async.assert_awaited_once_with(
            "memorize"
        )
        judge_config = backend.create_agent_async.await_args.args[2]
        assert judge_config.model_id == "memorize"
        assert judge_config.id == "chat-memorize-judge"
        agent.run_async.assert_awaited_once()
        assert json.loads(agent.run_async.await_args.kwargs["query"]) == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert agent.run_async.await_args.kwargs["response_model"] is _ChatMemorizeParseResult

    @pytest.mark.asyncio
    async def test_returns_empty_when_not_worth_retaining(self):
        provider, _, _ = self._provider_with_extraction(
            _ChatMemorizeParseResult(should_retain=False, memories=["ignored"])
        )
        result = await self._parse([self._user_message()], provider)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_memories_empty(self):
        provider, _, _ = self._provider_with_extraction(
            _ChatMemorizeParseResult(should_retain=True, memories=["", "  "])
        )
        result = await self._parse([self._user_message()], provider)
        assert result == []

    @pytest.mark.asyncio
    async def test_raises_when_result_is_not_extraction(self):
        from fivcplayground.agents import AgentRunContent

        provider, _, _ = self._provider_with_extraction(AgentRunContent(text="ok"))
        with pytest.raises(RuntimeError, match="should_retain"):
            await self._parse([self._user_message()], provider)

    @pytest.mark.asyncio
    async def test_returns_raw_content_when_memorize_model_missing(self):
        provider, backend, _ = self._provider_with_extraction(
            _ChatMemorizeParseResult(should_retain=True, memories=["x"]),
            model_config=None,
        )
        messages = [self._user_message()]
        result = await self._parse(messages, provider)

        assert json.loads(result[0]) == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        backend.create_agent_async.assert_not_awaited()


class TestMemorizeJob:
    @pytest.mark.asyncio
    async def test_job_retains_extracted_summary_and_marks_messages(
        self, session: AsyncSession, test_user
    ):
        chat = await _add_chat(session, test_user.uuid)
        msg = await _add_message(
            session,
            chat.uuid,
            query={"text": "帮我看看这段代码为什么报错"},
            reply={"text": "你的第 12 行变量未定义,应该是..."},
            created_at=_older(),
        )

        memory = MagicMock()
        memory.retain_async = AsyncMock(return_value=MemoryRetainResult(success=True, count=1))
        provider = MagicMock()
        provider.get_memory.return_value = memory

        mutex = MagicMock()
        mutex.acquire_async = AsyncMock(return_value=True)
        mutex.release_async = AsyncMock(return_value=True)
        mutex_site = MagicMock()
        mutex_site.get_mutex.return_value = mutex

        job = ChatMemorizeJob(MagicMock())

        class _SessionCtx:
            async def __aenter__(self):
                return session

            async def __aexit__(self, *args):
                return None

        with (
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_memory_provider_async",
                AsyncMock(return_value=provider),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_mutex_site_async",
                AsyncMock(return_value=mutex_site),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_db_session_context_async",
                side_effect=lambda: _SessionCtx(),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize._ChatMemorizeParser.__aenter__",
                AsyncMock(return_value=["The user is named Charlie"]),
            ),
        ):
            await job.run_async()

        provider.get_memory.assert_called_once_with(space_id=test_user.uuid)
        memory.retain_async.assert_awaited_once_with("The user is named Charlie")
        await session.refresh(msg)
        assert msg.is_memorized is True
        mutex_site.get_mutex.assert_called_with(f"agent-chats:memorize:{chat.uuid}")
        mutex.release_async.assert_awaited()

    @pytest.mark.asyncio
    async def test_job_retains_raw_json_when_memorize_model_missing(
        self, session: AsyncSession, test_user
    ):
        chat = await _add_chat(session, test_user.uuid)
        msg = await _add_message(
            session,
            chat.uuid,
            query={"text": "帮我看看这段代码为什么报错"},
            reply={"text": "你的第 12 行变量未定义,应该是..."},
            created_at=_older(),
        )

        memory = MagicMock()
        memory.retain_async = AsyncMock(return_value=MemoryRetainResult(success=True, count=1))
        provider = MagicMock()
        provider.get_memory.return_value = memory

        mutex = MagicMock()
        mutex.acquire_async = AsyncMock(return_value=True)
        mutex.release_async = AsyncMock(return_value=True)
        mutex_site = MagicMock()
        mutex_site.get_mutex.return_value = mutex

        repo = MagicMock()
        repo.get_model_config_async = AsyncMock(return_value=None)
        config_provider = MagicMock()
        config_provider.get_model_repository.return_value = repo

        job = ChatMemorizeJob(MagicMock())

        class _SessionCtx:
            async def __aenter__(self):
                return session

            async def __aexit__(self, *args):
                return None

        with (
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_memory_provider_async",
                AsyncMock(return_value=provider),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_mutex_site_async",
                AsyncMock(return_value=mutex_site),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_db_session_context_async",
                side_effect=lambda: _SessionCtx(),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_config_provider_async",
                AsyncMock(return_value=config_provider),
            ),
        ):
            await job.run_async()

        retained = memory.retain_async.await_args.args[0]
        assert json.loads(retained) == [
            {"role": "user", "content": "帮我看看这段代码为什么报错"},
            {"role": "assistant", "content": "你的第 12 行变量未定义,应该是..."},
        ]
        await session.refresh(msg)
        assert msg.is_memorized is True
        repo.get_model_config_async.assert_awaited_once_with("memorize")

    @pytest.mark.asyncio
    async def test_job_skips_retain_when_extract_returns_none(
        self, session: AsyncSession, test_user
    ):
        chat = await _add_chat(session, test_user.uuid)
        msg = await _add_message(
            session,
            chat.uuid,
            query={"text": "你好"},
            reply={"text": "你好"},
            created_at=_older(),
        )

        memory = MagicMock()
        memory.retain_async = AsyncMock(return_value=MemoryRetainResult(success=True, count=1))
        provider = MagicMock()
        provider.get_memory.return_value = memory

        mutex = MagicMock()
        mutex.acquire_async = AsyncMock(return_value=True)
        mutex.release_async = AsyncMock(return_value=True)
        mutex_site = MagicMock()
        mutex_site.get_mutex.return_value = mutex

        job = ChatMemorizeJob(MagicMock())

        class _SessionCtx:
            async def __aenter__(self):
                return session

            async def __aexit__(self, *args):
                return None

        with (
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_memory_provider_async",
                AsyncMock(return_value=provider),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_mutex_site_async",
                AsyncMock(return_value=mutex_site),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_db_session_context_async",
                side_effect=lambda: _SessionCtx(),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize._ChatMemorizeParser.__aenter__",
                AsyncMock(return_value=[]),
            ),
        ):
            await job.run_async()

        memory.retain_async.assert_not_awaited()
        await session.refresh(msg)
        assert msg.is_memorized is True

    @pytest.mark.asyncio
    async def test_job_leaves_unmemorized_when_extract_raises(
        self, session: AsyncSession, test_user
    ):
        chat = await _add_chat(session, test_user.uuid)
        msg = await _add_message(
            session,
            chat.uuid,
            query={"text": "记住我叫 Charlie"},
            reply={"text": "好的"},
            created_at=_older(),
        )

        memory = MagicMock()
        memory.retain_async = AsyncMock(return_value=MemoryRetainResult(success=True, count=1))
        provider = MagicMock()
        provider.get_memory.return_value = memory

        mutex = MagicMock()
        mutex.acquire_async = AsyncMock(return_value=True)
        mutex.release_async = AsyncMock(return_value=True)
        mutex_site = MagicMock()
        mutex_site.get_mutex.return_value = mutex

        job = ChatMemorizeJob(MagicMock())

        class _SessionCtx:
            async def __aenter__(self):
                return session

            async def __aexit__(self, *args):
                return None

        with (
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_memory_provider_async",
                AsyncMock(return_value=provider),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_mutex_site_async",
                AsyncMock(return_value=mutex_site),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_db_session_context_async",
                side_effect=lambda: _SessionCtx(),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize._ChatMemorizeParser.__aenter__",
                AsyncMock(side_effect=RuntimeError("no memorize model")),
            ),
        ):
            await job.run_async()

        memory.retain_async.assert_not_awaited()
        await session.refresh(msg)
        assert msg.is_memorized is False

    @pytest.mark.asyncio
    async def test_job_leaves_unmemorized_when_retain_fails(self, session: AsyncSession, test_user):
        chat = await _add_chat(session, test_user.uuid)
        msg = await _add_message(
            session,
            chat.uuid,
            query={"text": "记住我叫 Charlie"},
            reply={"text": "好的"},
            created_at=_older(),
        )

        memory = MagicMock()
        memory.retain_async = AsyncMock(return_value=MemoryRetainResult(success=False, count=0))
        provider = MagicMock()
        provider.get_memory.return_value = memory

        mutex = MagicMock()
        mutex.acquire_async = AsyncMock(return_value=True)
        mutex.release_async = AsyncMock(return_value=True)
        mutex_site = MagicMock()
        mutex_site.get_mutex.return_value = mutex

        job = ChatMemorizeJob(MagicMock())

        class _SessionCtx:
            async def __aenter__(self):
                return session

            async def __aexit__(self, *args):
                return None

        with (
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_memory_provider_async",
                AsyncMock(return_value=provider),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_mutex_site_async",
                AsyncMock(return_value=mutex_site),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_db_session_context_async",
                side_effect=lambda: _SessionCtx(),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize._ChatMemorizeParser.__aenter__",
                AsyncMock(return_value=["The user is named Charlie"]),
            ),
        ):
            await job.run_async()

        memory.retain_async.assert_awaited_once_with("The user is named Charlie")
        await session.refresh(msg)
        assert msg.is_memorized is False

    @pytest.mark.asyncio
    async def test_job_skips_when_mutex_not_acquired(self, session: AsyncSession, test_user):
        chat = await _add_chat(session, test_user.uuid)
        msg = await _add_message(
            session,
            chat.uuid,
            query={"text": "q"},
            reply={"text": "a"},
            created_at=_older(),
        )

        memory = MagicMock()
        memory.retain_async = AsyncMock(return_value=MemoryRetainResult(success=True, count=1))
        provider = MagicMock()
        provider.get_memory.return_value = memory

        mutex = MagicMock()
        mutex.acquire_async = AsyncMock(return_value=False)
        mutex.release_async = AsyncMock(return_value=True)
        mutex_site = MagicMock()
        mutex_site.get_mutex.return_value = mutex

        job = ChatMemorizeJob(MagicMock())

        class _SessionCtx:
            async def __aenter__(self):
                return session

            async def __aexit__(self, *args):
                return None

        with (
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_memory_provider_async",
                AsyncMock(return_value=provider),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_mutex_site_async",
                AsyncMock(return_value=mutex_site),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_db_session_context_async",
                side_effect=lambda: _SessionCtx(),
            ),
        ):
            await job.run_async()

        memory.retain_async.assert_not_awaited()
        await session.refresh(msg)
        assert msg.is_memorized is False

    @pytest.mark.asyncio
    async def test_slash_command_marks_without_retain_when_no_reply(
        self, session: AsyncSession, test_user
    ):
        chat = await _add_chat(session, test_user.uuid)
        msg = await _add_message(
            session,
            chat.uuid,
            query={"text": "/ping"},
            reply=None,
            created_at=_older(),
        )

        memory = MagicMock()
        memory.retain_async = AsyncMock(return_value=MemoryRetainResult(success=True, count=1))
        provider = MagicMock()
        provider.get_memory.return_value = memory

        mutex = MagicMock()
        mutex.acquire_async = AsyncMock(return_value=True)
        mutex.release_async = AsyncMock(return_value=True)
        mutex_site = MagicMock()
        mutex_site.get_mutex.return_value = mutex

        job = ChatMemorizeJob(MagicMock())

        class _SessionCtx:
            async def __aenter__(self):
                return session

            async def __aexit__(self, *args):
                return None

        with (
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_memory_provider_async",
                AsyncMock(return_value=provider),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_mutex_site_async",
                AsyncMock(return_value=mutex_site),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_db_session_context_async",
                side_effect=lambda: _SessionCtx(),
            ),
        ):
            await job.run_async()

        memory.retain_async.assert_not_awaited()
        await session.refresh(msg)
        assert msg.is_memorized is True

    @pytest.mark.asyncio
    async def test_slash_command_with_reply_marks_without_retain(
        self, session: AsyncSession, test_user
    ):
        chat = await _add_chat(session, test_user.uuid)
        msg = await _add_message(
            session,
            chat.uuid,
            query={"text": "/help"},
            reply={"text": "这里是帮助"},
            created_at=_older(),
        )

        memory = MagicMock()
        memory.retain_async = AsyncMock(return_value=MemoryRetainResult(success=True, count=1))
        provider = MagicMock()
        provider.get_memory.return_value = memory

        mutex = MagicMock()
        mutex.acquire_async = AsyncMock(return_value=True)
        mutex.release_async = AsyncMock(return_value=True)
        mutex_site = MagicMock()
        mutex_site.get_mutex.return_value = mutex

        job = ChatMemorizeJob(MagicMock())

        class _SessionCtx:
            async def __aenter__(self):
                return session

            async def __aexit__(self, *args):
                return None

        with (
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_memory_provider_async",
                AsyncMock(return_value=provider),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_mutex_site_async",
                AsyncMock(return_value=mutex_site),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.memorize.get_db_session_context_async",
                side_effect=lambda: _SessionCtx(),
            ),
        ):
            await job.run_async()

        memory.retain_async.assert_not_awaited()
        await session.refresh(msg)
        assert msg.is_memorized is True

    def test_config_defaults_and_custom_interval(self):
        component_site = MagicMock()
        with patch(
            "fivccliche.modules.agent_chats.jobs.memorize.query_component",
            return_value=None,
        ):
            job = ChatMemorizeJob(component_site)
        assert job.name == _MEMORIZE_JOB_ID
        assert job._setting.interval_minutes == 5
        assert job._setting.batch_size == 50
        assert job._setting.min_age_minutes == 5
        assert job.config["trigger"] == "interval"
        assert job.config["minutes"] == 5
        assert job.config["max_instances"] == 1
        assert job.config["coalesce"] is True

        session = MagicMock()
        session.get_value.side_effect = lambda key: {
            "INTERVAL_MINUTES": "15",
            "BATCH_SIZE": "0",
            "MAX_BATCHES_PER_RUN": "bad",
            "MIN_AGE_MINUTES": "10",
        }.get(key)
        config = MagicMock()
        config.get_session.return_value = session
        with patch(
            "fivccliche.modules.agent_chats.jobs.memorize.query_component",
            return_value=config,
        ):
            job = ChatMemorizeJob(component_site)
        config.get_session.assert_called_with("CHAT_MEMORIZE")
        assert job._setting.interval_minutes == 15
        assert job._setting.batch_size == 50  # invalid 0 → default
        assert job._setting.max_batches_per_run == 20  # invalid → default
        assert job._setting.min_age_minutes == 10
        assert job.config["minutes"] == 15


def test_agent_chats_list_jobs_does_not_register_memorize_job():
    components_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "fivccliche",
        "settings",
        "services.yml",
    )
    component_site = load_component_site(filename=components_path, fmt="yaml")
    module_site = ModuleSiteImpl(component_site, modules=[])
    module = ModuleImpl(component_site)

    jobs = module.list_jobs()
    assert jobs == []
    assert module.get_job(_MEMORIZE_JOB_ID) is None
    assert module.get_job("agent-chats-query") is None
    assert module.get_job("agent-chats-describe") is None
    assert module.get_job("missing") is None

    module_site.register_module(module)
    app = module_site.create_application()

    scheduler: AsyncIOScheduler = app.state.scheduler
    with TestClient(app):
        assert scheduler.get_job(_MEMORIZE_JOB_ID) is None
