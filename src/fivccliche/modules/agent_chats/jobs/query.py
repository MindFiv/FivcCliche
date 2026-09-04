"""On-demand job that runs a chat agent turn."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable

from fivcglue import IComponentSite
from fivcglue.interfaces.mutexes import IMutex
from fivcplayground.agents import AgentRun, AgentRunEvent, create_agent_async
from fivcplayground.skills import create_skill_retriever_async
from fivcplayground.tools import Tool, create_tool_retriever_async

from fivccliche.services.interfaces.modules import IModuleJob
from fivccliche.utils.deps import get_chat_provider_async, get_config_provider_async

logger = logging.getLogger(__name__)

_QUERY_JOB_ID = "agent-chats-query"


class ChatQueryJob(IModuleJob):
    """Run one user query through the chat agent.

    Not listed on ``ModuleImpl.list_jobs``; invoke ``run_async`` from the
    message-create handler. ``config`` is ``None`` so it is never scheduled.
    """

    def __init__(self, component_site: IComponentSite) -> None:
        self._component_site = component_site

    @property
    def name(self) -> str:
        return _QUERY_JOB_ID

    @property
    def config(self) -> dict | None:
        return None

    async def run_async(
        self,
        chat_uuid: str,
        *,
        user_uuid: str,
        query: str,
        agent_id: str = "default",
        tools: list[Tool] | None = None,
        skills_enabled: bool = True,
        context: dict | None = None,
        chat_mutex: IMutex | None = None,
        run_timeout: float | None = None,
        event_callback: Callable | None = None,
        finish_callback: (
            Callable[[AgentRun], None] | Callable[[AgentRun], Awaitable[None]] | None
        ) = None,
        **kwargs,
    ) -> None:
        finish_run = None
        config_provider = await get_config_provider_async()
        chat_provider = await get_chat_provider_async()
        context_copy = chat_provider.get_chat_context(
            user_uuid=user_uuid,
            context=context,
            chat_uuid=chat_uuid,
        )
        tools_by_name = {tool.name: tool for tool in tools or []}
        resolved_tools = list(tools_by_name.values()) or None
        tool_ids = [tool.name for tool in resolved_tools] if resolved_tools else []
        try:
            async with asyncio.timeout(run_timeout):
                agent = await create_agent_async(
                    model_backend=config_provider.get_model_backend(),
                    model_config_repository=config_provider.get_model_repository(
                        user_uuid=user_uuid
                    ),
                    agent_backend=config_provider.get_agent_backend(),
                    agent_config_repository=config_provider.get_agent_repository(
                        user_uuid=user_uuid
                    ),
                    agent_config_id=agent_id,
                )
                agent_tools = await create_tool_retriever_async(
                    tool_backend=config_provider.get_tool_backend(),
                    tools=resolved_tools,
                    tool_config_repository=config_provider.get_tool_repository(user_uuid=user_uuid),
                    embedding_backend=config_provider.get_embedding_backend(),
                    embedding_config_repository=config_provider.get_embedding_repository(
                        user_uuid=user_uuid
                    ),
                    space_id=user_uuid,
                )
                agent_skills = (
                    await create_skill_retriever_async(
                        tool_backend=config_provider.get_tool_backend(),
                        skill_config_repository=config_provider.get_skill_repository(
                            user_uuid=user_uuid
                        ),
                        embedding_backend=config_provider.get_embedding_backend(),
                        embedding_config_repository=config_provider.get_embedding_repository(
                            user_uuid=user_uuid
                        ),
                        space_id=user_uuid,
                    )
                    if skills_enabled
                    else None
                )

                def _on_event(ev, run):
                    nonlocal finish_run
                    if ev == AgentRunEvent.FINISH:
                        finish_run = run
                    if event_callback is not None:
                        event_callback(ev, run)

                await agent.run_async(
                    query=query,
                    tool_retriever=agent_tools,
                    tool_ids=tool_ids,
                    skill_retriever=agent_skills,
                    agent_run_repository=chat_provider.get_chat_repository(user_uuid=user_uuid),
                    agent_run_session_id=chat_uuid,
                    context=context_copy,
                    event_callback=_on_event,
                )
        except TimeoutError:
            logger.warning("ChatQueryJob timed out chat_uuid=%s", chat_uuid)
            raise
        finally:
            try:
                if finish_callback and finish_run is not None:
                    callback_result = finish_callback(finish_run)
                    if inspect.iscoroutine(callback_result):
                        await callback_result
            finally:
                if chat_mutex:
                    await chat_mutex.release_async()
