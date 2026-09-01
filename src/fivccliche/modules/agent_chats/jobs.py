"""Scheduled jobs for the agent_chats module."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from fivcglue import IComponentSite, query_component
from fivcglue.interfaces import configs
from fivcglue.interfaces.mutexes import IMutexSite
from fivcplayground.agents import AgentConfig
from pydantic import BaseModel, Field

from fivccliche.modules.agent_chats import models, utils
from fivccliche.services.interfaces.agent_memories import IUserMemoryProvider
from fivccliche.services.interfaces.modules import IModuleJob
from fivccliche.utils.deps import (
    get_config_provider_async,
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
_DEFAULT_MIN_AGE_MINUTES = 5
_MUTEX_EXPIRE = timedelta(minutes=30)
MEMORIZE_JOB_ID = "agent-chats-memorize"
_MEMORIZE_MODEL_ID = "memorize"
_MEMORIZE_EXTRACT_PROMPT = """\
You decide whether a chat transcript should be stored as long-term user memory,
and extract the information to store.

Extract memories only from user turns: durable facts the user stated about
themselves, such as:
- identity, preferences, constraints, decisions
- ongoing tasks, projects, or relationships the user described
- other facts they would reasonably want recalled later

Do not store assistant explanations, diagnoses, tutorials, advice, opinions,
or code analysis as memories. Assistant turns are context only; never rewrite
an assistant conclusion as "the user ...".

Do not retain greetings, thanks, small talk, slash-command noise, or generic
public-knowledge Q&A with no user-stated facts about themselves.

When worth retaining, extract short standalone facts (third person or
"the user ...") whose source is the user content. Do not copy the transcript
verbatim. Do not invent facts that are not in the user turns.

When the user only asked a public-knowledge question, or nothing user-stated
is worth retaining, set should_retain to false and memories to [].

The input is a JSON array of {role, content} turns.
"""


class _MemorizeExtraction(BaseModel):
    should_retain: bool = Field(
        description=("True if user turns contain durable facts the user stated about themselves.")
    )
    memories: list[str] = Field(
        default_factory=list,
        description=(
            "Short standalone facts from user turns only. Empty when nothing user-stated "
            "is worth retaining. Never include assistant explanations or advice."
        ),
    )


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
    min_age_minutes = to_int(
        session.get_value("MIN_AGE_MINUTES") if session else None,
        _DEFAULT_MIN_AGE_MINUTES,
    )
    return {
        "interval_minutes": (
            interval_minutes if interval_minutes > 0 else _DEFAULT_INTERVAL_MINUTES
        ),
        "batch_size": batch_size if batch_size > 0 else _DEFAULT_BATCH_SIZE,
        "max_batches_per_run": (
            max_batches_per_run if max_batches_per_run > 0 else _DEFAULT_MAX_BATCHES_PER_RUN
        ),
        "min_age_minutes": min_age_minutes if min_age_minutes > 0 else _DEFAULT_MIN_AGE_MINUTES,
    }


def _get_memorize_content(messages: list[models.UserChatMessage]) -> str:
    """Build JSON conversation payload for the memorize judge; empty when there is no user turn."""
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


async def _extract_memorable_content_async(content: str, *, user_uuid: str) -> str | None:
    """Ask the memorize LLM whether to retain, and extract the text to store.

    If the user has no visible ``id=memorize`` LLM, return ``content`` unchanged
    so the caller retains the raw transcript (legacy behavior).
    """
    config_provider = await get_config_provider_async()
    model_repo = config_provider.get_model_repository(user_uuid=user_uuid)
    if await model_repo.get_model_config_async(_MEMORIZE_MODEL_ID) is None:
        logger.info(
            "No %s LLM for user %s; retaining raw transcript",
            _MEMORIZE_MODEL_ID,
            user_uuid,
        )
        return content

    judge_config = AgentConfig(
        id="chat-memorize-judge",
        model_id=_MEMORIZE_MODEL_ID,
        system_prompt=_MEMORIZE_EXTRACT_PROMPT,
    )
    agent = await config_provider.get_agent_backend().create_agent_async(
        config_provider.get_model_backend(),
        model_repo,
        judge_config,
    )
    result = await agent.run_async(
        query=content,
        response_model=_MemorizeExtraction,
    )
    if not isinstance(result, _MemorizeExtraction):
        raise RuntimeError("memorize judge did not return should_retain")
    if not result.should_retain:
        return None
    memories = [item.strip() for item in result.memories if item.strip()]
    if not memories:
        return None
    return "\n".join(memories)


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
                extracted = await _extract_memorable_content_async(
                    content, user_uuid=chat.user_uuid
                )
                if extracted:
                    memory = memory_provider.get_memory(space_id=chat.user_uuid)
                    result = await memory.retain_async(extracted)
                    if not result.success:
                        logger.warning(
                            "retain_async failed for chat %s; leaving unmemorized",
                            chat.uuid,
                        )
                        return
                else:
                    logger.info(
                        "Skip retain for chat %s; nothing worth remembering",
                        chat.uuid,
                    )

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
        self.min_age_minutes = settings["min_age_minutes"]
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

        created_at_to = datetime.now(timezone.utc) - timedelta(minutes=self.min_age_minutes)

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
