"""Scheduled jobs for the agent_chats module."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fivcglue import IComponentSite, query_component
from fivcglue.interfaces import configs
from fivcglue.interfaces.mutexes import IMutexSite

from fivccliche.modules.agent_chats import methods, models
from fivccliche.services.interfaces.agent_memories import IUserMemoryProvider
from fivccliche.services.interfaces.modules import IModuleJob
from fivccliche.utils.deps import (
    get_db_session_context,
    get_memory_provider_async,
    get_mutex_site_async,
)

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_MINUTES = 5
_DEFAULT_BATCH_SIZE = 50
_DEFAULT_MAX_BATCHES_PER_RUN = 20
_DEFAULT_MIN_AGE_HOURS = 24
_MUTEX_EXPIRE = timedelta(minutes=30)
MEMORIZE_JOB_ID = "agent-chats-memorize"


@dataclass(frozen=True)
class _MemorizeSettings:
    interval_minutes: int
    batch_size: int
    max_batches_per_run: int
    min_age_hours: int


def _config_int(session: configs.IConfigSession | None, key: str, default: int) -> int:
    if session is None:
        return default
    raw = session.get_value(key)
    if raw is None:
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _load_memorize_settings(component_site: IComponentSite) -> _MemorizeSettings:
    config = query_component(component_site, configs.IConfig)
    session = config.get_session("agent_chats") if config else None
    return _MemorizeSettings(
        interval_minutes=_config_int(
            session, "MEMORIZE_INTERVAL_MINUTES", _DEFAULT_INTERVAL_MINUTES
        ),
        batch_size=_config_int(session, "MEMORIZE_BATCH_SIZE", _DEFAULT_BATCH_SIZE),
        max_batches_per_run=_config_int(
            session, "MEMORIZE_MAX_BATCHES_PER_RUN", _DEFAULT_MAX_BATCHES_PER_RUN
        ),
        min_age_hours=_config_int(session, "MEMORIZE_MIN_AGE_HOURS", _DEFAULT_MIN_AGE_HOURS),
    )


def build_conversation_turns(messages: list[models.UserChatMessage]) -> list[dict[str, str]]:
    """Build structured conversation turns for memory retain."""
    turns: list[dict[str, str]] = []
    for message in messages:
        query_text = ""
        if isinstance(message.query, dict) and message.query.get("text") is not None:
            query_text = str(message.query["text"]).strip()
        reply_text = ""
        if isinstance(message.reply, dict) and message.reply.get("text") is not None:
            reply_text = str(message.reply["text"]).strip()
        if query_text and not query_text.lstrip().startswith("/"):
            turns.append({"role": "user", "content": query_text})
        if reply_text:
            turns.append({"role": "assistant", "content": reply_text})
    return turns


class ChatMemorizeJob(IModuleJob):
    """Interval job that retains aged chat conversations into agent memory.

    Expose via ModuleImpl.list_jobs; ModuleSiteImpl registers it on the
    shared AsyncIOScheduler from ``config``.
    """

    def __init__(self, component_site: IComponentSite) -> None:
        settings = _load_memorize_settings(component_site)
        self.interval_minutes = settings.interval_minutes
        self.batch_size = settings.batch_size
        self.max_batches_per_run = settings.max_batches_per_run
        self.min_age_hours = settings.min_age_hours
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
            async with get_db_session_context() as session:
                chats = await methods.list_unmemorized_chats_async(
                    session,
                    created_at_to=created_at_to,
                    limit=self.batch_size,
                )
            if not chats:
                break

            for chat in chats:
                await self._memorize_one_chat_async(
                    chat,
                    memory_provider=memory_provider,
                    mutex_site=mutex_site,
                    created_at_to=created_at_to,
                )
            if len(chats) < self.batch_size:
                break

    async def _memorize_one_chat_async(
        self,
        chat: models.UserChat,
        *,
        memory_provider: IUserMemoryProvider,
        mutex_site: IMutexSite,
        created_at_to: datetime,
    ) -> None:
        if not chat.user_uuid:
            return

        mutex = mutex_site.get_mutex(f"agent-chats:memorize:{chat.uuid}")
        if mutex is None:
            logger.debug("No mutex for chat %s; skip", chat.uuid)
            return

        acquired = await mutex.acquire_async(expire=_MUTEX_EXPIRE, timeout=None)
        if not acquired:
            logger.debug("Could not acquire memorize lock for chat %s", chat.uuid)
            return

        try:
            async with get_db_session_context() as session:
                messages = await methods.list_unmemorized_chat_messages_async(
                    session,
                    chat.uuid,
                    created_at_to=created_at_to,
                )
            if not messages:
                return

            message_uuids = [message.uuid for message in messages]
            turns = build_conversation_turns(messages)
            has_user = any(turn["role"] == "user" for turn in turns)
            if has_user:
                memory = memory_provider.get_memory(space_id=chat.user_uuid)
                content = json.dumps(turns, ensure_ascii=False)
                result = await memory.retain_async(content)
                if not result.success:
                    logger.warning(
                        "retain_async failed for chat %s; leaving unmemorized",
                        chat.uuid,
                    )
                    return

            async with get_db_session_context() as session:
                await methods.mark_unmemorized_chat_messages_async(session, message_uuids)
        except Exception:
            logger.exception("Failed to memorize chat %s", chat.uuid)
        finally:
            await mutex.release_async()
