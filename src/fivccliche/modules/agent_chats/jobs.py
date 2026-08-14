"""Scheduled jobs for the agent_chats module."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from fivcglue import IComponentSite, query_component
from fivcglue.interfaces import configs
from fivcglue.interfaces.mutexes import IMutexSite

from fivccliche.modules.agent_chats import models, utils
from fivccliche.services.interfaces.agent_memories import IUserMemoryProvider
from fivccliche.services.interfaces.modules import IModuleJob
from fivccliche.utils.deps import (
    get_db_session_context_async,
    get_memory_provider_async,
    get_mutex_context_async,
    get_mutex_site_async,
)
from fivccliche.utils.types import to_int

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_MINUTES = 5
_DEFAULT_BATCH_SIZE = 50
_DEFAULT_MAX_BATCHES_PER_RUN = 20
_DEFAULT_MIN_AGE_HOURS = 24
_MUTEX_EXPIRE = timedelta(minutes=30)
MEMORIZE_JOB_ID = "agent-chats-memorize"


def _get_memorize_settings(component_site: IComponentSite) -> dict[str, int]:
    config = query_component(component_site, configs.IConfig)
    session = config.get_session("CHAT_MEMORIZE") if config else None
    interval_minutes = to_int(
        session.get_value("INTERVAL_MINUTES") if session else None,
        _DEFAULT_INTERVAL_MINUTES,
    )
    batch_size = to_int(
        session.get_value("BATCH_SIZE") if session else None,
        _DEFAULT_BATCH_SIZE,
    )
    max_batches_per_run = to_int(
        session.get_value("MAX_BATCHES_PER_RUN") if session else None,
        _DEFAULT_MAX_BATCHES_PER_RUN,
    )
    min_age_hours = to_int(
        session.get_value("MIN_AGE_HOURS") if session else None,
        _DEFAULT_MIN_AGE_HOURS,
    )
    return {
        "interval_minutes": (
            interval_minutes if interval_minutes > 0 else _DEFAULT_INTERVAL_MINUTES
        ),
        "batch_size": batch_size if batch_size > 0 else _DEFAULT_BATCH_SIZE,
        "max_batches_per_run": (
            max_batches_per_run if max_batches_per_run > 0 else _DEFAULT_MAX_BATCHES_PER_RUN
        ),
        "min_age_hours": min_age_hours if min_age_hours > 0 else _DEFAULT_MIN_AGE_HOURS,
    }


def _get_memorize_content(messages: list[models.UserChatMessage]) -> str:
    """Build JSON retain payload; empty when there is no user turn."""
    turns: list[dict[str, str]] = []
    has_user = False
    for message in messages:
        query_text = ""
        if isinstance(message.query, dict) and message.query.get("text") is not None:
            query_text = str(message.query["text"]).strip()
        reply_text = ""
        if isinstance(message.reply, dict) and message.reply.get("text") is not None:
            reply_text = str(message.reply["text"]).strip()
        if query_text and not query_text.lstrip().startswith("/"):
            turns.append({"role": "user", "content": query_text})
            has_user = True
        if reply_text:
            turns.append({"role": "assistant", "content": reply_text})
    if not has_user:
        return ""
    return json.dumps(turns, ensure_ascii=False)


async def _memorize_async(
    chat: models.UserChat,
    *,
    memory_provider: IUserMemoryProvider,
    mutex_site: IMutexSite,
    created_at_to: datetime,
) -> None:
    if not chat.user_uuid:
        return

    async with get_mutex_context_async(
        f"agent-chats:memorize:{chat.uuid}",
        expire=_MUTEX_EXPIRE,
        timeout=None,
        mutex_site=mutex_site,
    ) as mutex:
        if mutex is None:
            logger.debug("Could not acquire memorize lock for chat %s", chat.uuid)
            return

        try:
            async with get_db_session_context_async() as session:
                messages = await utils.list_unmemorized_chat_messages_async(
                    session,
                    chat.uuid,
                    created_at_to=created_at_to,
                )
            if not messages:
                return

            content = _get_memorize_content(messages)
            if content:
                memory = memory_provider.get_memory(space_id=chat.user_uuid)
                result = await memory.retain_async(content)
                if not result.success:
                    logger.warning(
                        "retain_async failed for chat %s; leaving unmemorized",
                        chat.uuid,
                    )
                    return

            async with get_db_session_context_async() as session:
                await utils.delete_unmemorized_chat_messages_async(
                    session, chat.uuid, created_at_to=created_at_to
                )
                await session.commit()
        except Exception:
            logger.exception("Failed to memorize chat %s", chat.uuid)


class ChatMemorizeJob(IModuleJob):
    """Interval job that retains aged chat conversations into agent memory.

    Expose via ModuleImpl.list_jobs; ModuleSiteImpl registers it on the
    shared AsyncIOScheduler from ``config``.
    """

    def __init__(self, component_site: IComponentSite) -> None:
        settings = _get_memorize_settings(component_site)
        self.interval_minutes = settings["interval_minutes"]
        self.batch_size = settings["batch_size"]
        self.max_batches_per_run = settings["max_batches_per_run"]
        self.min_age_hours = settings["min_age_hours"]
        self._config = {
            "trigger": "interval",
            "minutes": self.interval_minutes,
            "max_instances": 1,
            "coalesce": True,
            "replace_existing": True,
        }
        logger.info(
            "Configured %s job (interval=%s minutes)",
            MEMORIZE_JOB_ID,
            self.interval_minutes,
        )

    @property
    def name(self) -> str:
        return MEMORIZE_JOB_ID

    @property
    def config(self) -> dict:
        return dict(self._config)

    async def run_async(self) -> None:
        """Drain aged unmemorized chats into agent memory (per-chat mutex)."""
        memory_provider = await get_memory_provider_async()
        if memory_provider is None:
            logger.debug("Memory provider not mounted; skip memorize job")
            return

        mutex_site = await get_mutex_site_async()
        if mutex_site is None:
            logger.debug("Mutex site not available; skip memorize job")
            return

        created_at_to = datetime.now(timezone.utc) - timedelta(hours=self.min_age_hours)

        for _ in range(self.max_batches_per_run):
            async with get_db_session_context_async() as session:
                chats = await utils.list_unmemorized_chats_async(
                    session,
                    created_at_to=created_at_to,
                    limit=self.batch_size,
                )
            if not chats:
                break

            for chat in chats:
                await _memorize_async(
                    chat,
                    memory_provider=memory_provider,
                    mutex_site=mutex_site,
                    created_at_to=created_at_to,
                )
            if len(chats) < self.batch_size:
                break
