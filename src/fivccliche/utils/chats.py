import asyncio
import inspect
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

from fivcglue.interfaces.mutexes import IMutex
from fivcplayground.agents import AgentRun, AgentRunEvent, create_agent_async
from fivcplayground.skills import create_skill_retriever_async
from fivcplayground.tools import Tool, create_tool_retriever_async
from sqlalchemy.ext.asyncio import AsyncSession

from fivccliche.services.interfaces.agent_chats import IUserChatProvider
from fivccliche.services.interfaces.agent_configs import IUserConfigProvider
from fivccliche.services.interfaces.auth import IUser
from fivccliche.services.interfaces.db import IDatabase
from fivccliche.utils.deps import default_db

logger = logging.getLogger(__name__)


class ChatTask:
    """Background agent runner with SSE streaming via get_stream_async()."""

    def __init__(
        self,
        user: IUser,
        user_config_provider: IUserConfigProvider,
        user_chat_provider: IUserChatProvider,
        chat_query: str = "",
        chat_uuid: str | None = None,
        chat_agent_id: str = "default",
        chat_tools: list[Tool] | None = None,
        chat_skills_enabled: bool = True,
        chat_context: dict | None = None,
        chat_finish_callback: (
            Callable[[AgentRun], None] | Callable[[AgentRun], Awaitable[None]] | None
        ) = None,
        chat_mutex: IMutex | None = None,
        chat_db: IDatabase | None = None,
        **kwargs,
    ):
        self._user = user
        self._user_config_provider = user_config_provider
        self._user_chat_provider = user_chat_provider
        self._chat_query = chat_query
        self._chat_uuid = chat_uuid
        self._chat_agent_id = chat_agent_id
        tools_by_name = {tool.name: tool for tool in chat_tools or []}
        self._resolved_chat_tools = list(tools_by_name.values()) or None
        self._chat_tool_ids = (
            [tool.name for tool in self._resolved_chat_tools] if self._resolved_chat_tools else []
        )
        self._chat_skills_enabled = chat_skills_enabled
        self._chat_context = chat_context
        self._chat_finish_callback = chat_finish_callback
        self._chat_mutex = chat_mutex
        self._chat_db = chat_db or default_db()
        self._chat_queue: asyncio.Queue = asyncio.Queue()
        self._asyncio_task: asyncio.Task | None = None

    def start(self) -> None:
        """Schedule this ChatTask on the running event loop."""
        if self._asyncio_task is not None:
            raise RuntimeError("ChatTask already started")

        def _log_unretrieved_exception(task: asyncio.Task) -> None:
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                logger.exception("ChatTask failed", exc_info=exc)

        self._asyncio_task = asyncio.create_task(self._run_async())
        self._asyncio_task.add_done_callback(_log_unretrieved_exception)

    async def join_async(self) -> None:
        """Wait for the started task to finish, swallowing its errors."""
        if self._asyncio_task is None:
            return
        try:
            await self._asyncio_task
        except Exception:
            pass

    async def get_stream_async(self) -> AsyncIterator[str]:
        """Yield SSE chunks from the agent event queue."""
        try:
            while True:
                if (
                    self._asyncio_task is not None
                    and self._asyncio_task.done()
                    and self._chat_queue.empty()
                ):
                    self._asyncio_task.result()
                    break

                try:
                    ev, ev_run = await asyncio.wait_for(self._chat_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if self._asyncio_task is None or not self._asyncio_task.done():
                        print("⏱️  [QUEUE] Timeout waiting for event, task still running")
                    continue

                data_fields_basics = {
                    "id",
                    "agent_id",
                    "started_at",
                    "completed_at",
                }
                data_fields = {
                    "query",
                    "reply",
                    "tool_calls",
                    *data_fields_basics,
                }
                if ev == AgentRunEvent.START:
                    data = ev_run.model_dump(mode="json", include=data_fields)
                    data.update({"chat_uuid": self._chat_uuid})
                    data = {"event": "start", "info": data}
                    data_json = json.dumps(data)
                    yield f"data: {data_json}\n\n"

                elif ev == AgentRunEvent.FINISH:
                    data = ev_run.model_dump(mode="json", include=data_fields)
                    data.update({"chat_uuid": self._chat_uuid})
                    data = {"event": "finish", "info": data}
                    data_json = json.dumps(data)
                    yield f"data: {data_json}\n\n"

                elif ev == AgentRunEvent.STREAM:
                    data = ev_run.model_dump(mode="json", include=data_fields_basics)
                    data.update(
                        {
                            "chat_uuid": self._chat_uuid,
                            "delta": (
                                ev_run.delta.model_dump(mode="json") if ev_run.delta else None
                            ),
                        }
                    )
                    data = {"event": "stream", "info": data}
                    data = json.dumps(data)
                    yield f"data: {data}\n\n"

                elif ev == AgentRunEvent.TOOL:
                    data = ev_run.model_dump(mode="json", include=data_fields)
                    data.update({"chat_uuid": self._chat_uuid})
                    data = {"event": "tool", "info": data}
                    data = json.dumps(data)
                    yield f"data: {data}\n\n"

                self._chat_queue.task_done()

        except Exception as e:
            data = {"event": "error", "info": {"message": str(e)}}
            data = json.dumps(data)
            print(f"❌ [QUEUE] Error in chat queue: {e}")
            yield f"data: {data}\n\n"

    async def _run_async(self) -> None:
        owned_session = self._chat_db.create_session()
        finish_run = None
        base_context = {
            **(self._chat_context or {}),
            "user_uuid": self._user.uuid,
            "chat_uuid": self._chat_uuid,
        }
        try:
            context_copy = {**base_context, "session": owned_session}
            agent = await create_agent_async(
                model_backend=self._user_config_provider.get_model_backend(),
                model_config_repository=self._user_config_provider.get_model_repository(
                    user_uuid=self._user.uuid, session=owned_session
                ),
                agent_backend=self._user_config_provider.get_agent_backend(),
                agent_config_repository=self._user_config_provider.get_agent_repository(
                    user_uuid=self._user.uuid, session=owned_session
                ),
                agent_config_id=self._chat_agent_id,
            )
            agent_tools = await create_tool_retriever_async(
                tool_backend=self._user_config_provider.get_tool_backend(),
                tools=self._resolved_chat_tools,
                tool_config_repository=self._user_config_provider.get_tool_repository(
                    user_uuid=self._user.uuid, session=owned_session
                ),
                embedding_backend=self._user_config_provider.get_embedding_backend(),
                embedding_config_repository=self._user_config_provider.get_embedding_repository(
                    user_uuid=self._user.uuid, session=owned_session
                ),
                space_id=self._user.uuid,
            )
            agent_skills = (
                await create_skill_retriever_async(
                    tool_backend=self._user_config_provider.get_tool_backend(),
                    skill_config_repository=self._user_config_provider.get_skill_repository(
                        user_uuid=self._user.uuid, session=owned_session
                    ),
                    embedding_backend=self._user_config_provider.get_embedding_backend(),
                    embedding_config_repository=self._user_config_provider.get_embedding_repository(
                        user_uuid=self._user.uuid, session=owned_session
                    ),
                    space_id=self._user.uuid,
                )
                if self._chat_skills_enabled
                else None
            )

            def _event_callback(ev, run):
                nonlocal finish_run
                if ev == AgentRunEvent.FINISH:
                    finish_run = run
                self._chat_queue.put_nowait((ev, run))

            await agent.run_async(
                query=self._chat_query,
                tool_retriever=agent_tools,
                tool_ids=self._chat_tool_ids,
                skill_retriever=agent_skills,
                agent_run_repository=self._user_chat_provider.get_chat_repository(
                    user_uuid=self._user.uuid, session=owned_session
                ),
                agent_run_session_id=self._chat_uuid,
                context=context_copy,
                event_callback=_event_callback,
            )
        finally:
            await self._finish_async(owned_session, finish_run)

    async def _finish_async(
        self,
        owned_session: AsyncSession,
        finish_run: AgentRun | None,
    ) -> None:
        try:
            if self._chat_finish_callback and finish_run is not None:
                callback_result = self._chat_finish_callback(finish_run)
                if inspect.iscoroutine(callback_result):
                    await callback_result
        finally:
            try:
                await owned_session.close()
            finally:
                if self._chat_mutex:
                    await self._chat_mutex.release_async()
