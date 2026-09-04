"""Fill an empty chat description from the first user query."""

from __future__ import annotations

import logging
from datetime import timedelta

from fivcglue import IComponentSite
from fivcplayground.agents import AgentConfig

from fivccliche.modules.agent_chats import utils
from fivccliche.modules.agent_chats.filters import ChatEditableFilterSet
from fivccliche.services.interfaces.modules import IModuleJob
from fivccliche.utils.deps import (
    get_config_provider_async,
    get_db_session_context_async,
    get_mutex_context_async,
)

logger = logging.getLogger(__name__)

_DESCRIBE_JOB_ID = "agent-chats-describe"
_DESCRIBE_MODEL_ID = "describe"
_DESCRIBE_PROMPT = """\
Write a short chat title from the user's first message.
Use the same language as the message. One line only.
Do not use quotes. Do not add facts that are not in the message.
"""
_FALLBACK_MAX_CHARS = 80
_DESCRIPTION_MAX_CHARS = 1024
_MUTEX_EXPIRE = timedelta(minutes=5)


def _fallback_description(query_text: str) -> str:
    text = query_text.strip()
    if len(text) <= _FALLBACK_MAX_CHARS:
        return text
    cut = text[:_FALLBACK_MAX_CHARS].rsplit(None, 1)[0]
    return cut or text[:_FALLBACK_MAX_CHARS]


def _title_from_result(result: object) -> str:
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text.strip()
    return ""


class ChatDescribeJob(IModuleJob):
    """On-demand job that writes ``chat.description`` when it is empty.

    Not listed on ``ModuleImpl.list_jobs``; invoke ``run_async`` from the
    message-create handler. ``config`` is ``None`` so it is never scheduled.
    """

    def __init__(self, component_site: IComponentSite) -> None:
        self._component_site = component_site

    @property
    def name(self) -> str:
        return _DESCRIBE_JOB_ID

    @property
    def config(self) -> dict | None:
        return None

    async def run_async(self, chat_uuid: str, *, user_uuid: str, query_text: str, **kwargs) -> None:
        """Write a title onto ``chat.description`` when it is still empty."""
        try:
            async with get_mutex_context_async(
                f"agent-chats:describe:{chat_uuid}",
                expire=_MUTEX_EXPIRE,
                timeout=None,
            ) as mutex:
                if mutex is None:
                    logger.debug("Could not acquire describe lock for chat %s", chat_uuid)
                    return

                async with get_db_session_context_async() as session:
                    chat = await utils.get_chat_async(
                        session,
                        chat_uuid,
                        filters=ChatEditableFilterSet(user_uuid, is_superuser=False),
                    )
                    if chat is None or (chat.description or "").strip():
                        return

                title = await self._summarize_query_async(user_uuid, query_text)
                if not title:
                    return

                async with get_db_session_context_async() as session:
                    chat = await utils.get_chat_async(
                        session,
                        chat_uuid,
                        filters=ChatEditableFilterSet(user_uuid, is_superuser=False),
                    )
                    if chat is None or (chat.description or "").strip():
                        return
                    chat.description = title
                    session.add(chat)
                    await session.commit()
        except Exception:
            logger.exception("Failed to fill description for chat %s", chat_uuid)

    async def _summarize_query_async(self, user_uuid: str, query_text: str) -> str:
        config_provider = await get_config_provider_async()
        model_repo = config_provider.get_model_repository(user_uuid=user_uuid)
        if await model_repo.get_model_config_async(_DESCRIBE_MODEL_ID) is None:
            logger.info(
                "No %s LLM for user %s; using truncated query",
                _DESCRIBE_MODEL_ID,
                user_uuid,
            )
            return _fallback_description(query_text)

        agent = await config_provider.get_agent_backend().create_agent_async(
            config_provider.get_model_backend(),
            model_repo,
            AgentConfig(
                id="chat-describe",
                model_id=_DESCRIBE_MODEL_ID,
                system_prompt=_DESCRIBE_PROMPT,
            ),
        )
        result = await agent.run_async(query=query_text)
        return _title_from_result(result)[:_DESCRIPTION_MAX_CHARS]
