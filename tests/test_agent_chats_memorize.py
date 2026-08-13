"""Tests for chat-level memorize job and related methods."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi.testclient import TestClient
from fivcglue.implements.utils import load_component_site
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel

from fivccliche.modules.agent_chats import methods
from fivccliche.modules.agent_chats.jobs import (
    MEMORIZE_JOB_ID,
    ChatMemorizeJob,
    build_conversation_turns,
)
from fivccliche.modules.agent_chats.models import UserChat, UserChatMessage
from fivccliche.modules.agent_chats.services import ModuleImpl
from fivccliche.modules.users.models import User  # noqa: F401
from fivccliche.services.implements.modules import ModuleSiteImpl
from fivccliche.services.interfaces.agent_memories import MemoryRetainResult


@pytest.fixture
async def session():
    """Create a temporary SQLite database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        database_url = f"sqlite+aiosqlite:///{db_path}"

        engine = create_async_engine(
            database_url,
            connect_args={"check_same_thread": False, "timeout": 30},
            poolclass=NullPool,
            echo=False,
        )

        @event.listens_for(engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        async with engine.begin() as conn:
            await conn.execute(text("PRAGMA foreign_keys = ON"))
            await conn.run_sync(SQLModel.metadata.create_all)

        async_session = AsyncSession(engine, expire_on_commit=False)
        try:
            yield async_session
        finally:
            await async_session.close()
            await engine.dispose()


@pytest.fixture
async def test_user(session: AsyncSession):
    from fivccliche.modules.users import methods as user_methods

    return await user_methods.create_user_async(
        session,
        username="memuser",
        email="mem@example.com",
        password="password123",
    )


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


class TestBuildConversationTurns:
    def test_builds_user_and_assistant_turns(self):
        msg = UserChatMessage(
            chat_uuid="c",
            query={"text": "帮我看看这段代码为什么报错"},
            reply={"text": "你的第 12 行变量未定义,应该是..."},
        )
        turns = build_conversation_turns([msg])
        assert turns == [
            {"role": "user", "content": "帮我看看这段代码为什么报错"},
            {"role": "assistant", "content": "你的第 12 行变量未定义,应该是..."},
        ]

    def test_slash_query_skips_user_keeps_assistant(self):
        msg = UserChatMessage(
            chat_uuid="c",
            query={"text": "/help"},
            reply={"text": "这里是帮助"},
        )
        turns = build_conversation_turns([msg])
        assert turns == [{"role": "assistant", "content": "这里是帮助"}]

    def test_slash_query_without_reply_yields_empty(self):
        msg = UserChatMessage(
            chat_uuid="c",
            query={"text": "/status"},
            reply=None,
        )
        assert build_conversation_turns([msg]) == []


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
        from sqlalchemy.dialects import postgresql

        statement = methods._select_unmemorized_chats(
            created_at_to=datetime.now(timezone.utc),
            limit=50,
        )
        sql = str(statement.compile(dialect=postgresql.dialect())).upper()
        assert "DISTINCT" not in sql
        assert "EXISTS" in sql

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

        await methods.mark_unmemorized_chat_messages_async(session, [msg.uuid])
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
        await session.refresh(msg)
        assert msg.is_memorized is True


class TestMemorizeJob:
    @pytest.mark.asyncio
    async def test_job_retains_json_and_marks_messages(self, session: AsyncSession, test_user):
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
                "fivccliche.modules.agent_chats.jobs.get_memory_provider_async",
                AsyncMock(return_value=provider),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.get_mutex_site_async",
                AsyncMock(return_value=mutex_site),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.get_db_session_context_async",
                side_effect=lambda: _SessionCtx(),
            ),
        ):
            await job.run_async()

        provider.get_memory.assert_called_once_with(space_id=test_user.uuid)
        retained = memory.retain_async.await_args.args[0]
        assert json.loads(retained) == [
            {"role": "user", "content": "帮我看看这段代码为什么报错"},
            {"role": "assistant", "content": "你的第 12 行变量未定义,应该是..."},
        ]
        await session.refresh(msg)
        assert msg.is_memorized is True
        mutex_site.get_mutex.assert_called_with(f"agent-chats:memorize:{chat.uuid}")
        mutex.release_async.assert_awaited()

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
                "fivccliche.modules.agent_chats.jobs.get_memory_provider_async",
                AsyncMock(return_value=provider),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.get_mutex_site_async",
                AsyncMock(return_value=mutex_site),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.get_db_session_context_async",
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
                "fivccliche.modules.agent_chats.jobs.get_memory_provider_async",
                AsyncMock(return_value=provider),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.get_mutex_site_async",
                AsyncMock(return_value=mutex_site),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.get_db_session_context_async",
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
                "fivccliche.modules.agent_chats.jobs.get_memory_provider_async",
                AsyncMock(return_value=provider),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.get_mutex_site_async",
                AsyncMock(return_value=mutex_site),
            ),
            patch(
                "fivccliche.modules.agent_chats.jobs.get_db_session_context_async",
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
            "fivccliche.modules.agent_chats.jobs.query_component",
            return_value=None,
        ):
            job = ChatMemorizeJob(component_site)
        assert job.name == MEMORIZE_JOB_ID
        assert job.interval_minutes == 5
        assert job.batch_size == 50
        assert job.min_age_hours == 24
        assert job.config["trigger"] == "interval"
        assert job.config["minutes"] == 5
        assert job.config["max_instances"] == 1
        assert job.config["coalesce"] is True

        session = MagicMock()
        session.get_value.side_effect = lambda key: {
            "MEMORIZE_INTERVAL_MINUTES": "15",
            "MEMORIZE_BATCH_SIZE": "0",
            "MEMORIZE_MAX_BATCHES_PER_RUN": "bad",
            "MEMORIZE_MIN_AGE_HOURS": "48",
        }.get(key)
        config = MagicMock()
        config.get_session.return_value = session
        with patch(
            "fivccliche.modules.agent_chats.jobs.query_component",
            return_value=config,
        ):
            job = ChatMemorizeJob(component_site)
        assert job.interval_minutes == 15
        assert job.batch_size == 50  # invalid 0 → default
        assert job.max_batches_per_run == 20  # invalid → default
        assert job.min_age_hours == 48
        assert job.config["minutes"] == 15


def test_agent_chats_list_jobs_registers_memorize_job():
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
    assert len(jobs) == 1
    assert jobs[0].name == MEMORIZE_JOB_ID
    assert module.get_job(MEMORIZE_JOB_ID) is jobs[0]
    assert module.get_job("missing") is None

    module_site.register_module(module)
    app = module_site.create_application()

    scheduler: AsyncIOScheduler = app.state.scheduler
    # Stretch interval so the job cannot fire during the short TestClient window.
    scheduler.reschedule_job(MEMORIZE_JOB_ID, trigger="interval", minutes=60)
    with TestClient(app):
        job = scheduler.get_job(MEMORIZE_JOB_ID)
        assert job is not None
        assert job.id == MEMORIZE_JOB_ID
