import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable

from fivcglue.interfaces.mutexes import IMutex
from fivcplayground.agents import AgentRun, AgentRunEvent, create_agent_async
from fivcplayground.skills import create_skill_retriever_async
from fivcplayground.tools import Tool, create_tool_retriever_async

from fivccliche.services.interfaces.agent_chats import IUserChatProvider
from fivccliche.services.interfaces.agent_configs import IUserConfigProvider
from fivccliche.services.interfaces.auth import IUser
from fivccliche.utils.deps import default_db


class _ChatStreamingGenerator:
    """Generator for streaming agent runs."""

    def __init__(
        self,
        chat_task: asyncio.Task,
        chat_queue: asyncio.Queue,
        chat_uuid: str | None = None,
    ):
        self._chat_task = chat_task
        self._chat_queue = chat_queue
        self._chat_uuid = chat_uuid

    async def wait_async(self) -> None:
        """Wait for the background chat task to finish, swallowing its errors."""
        try:
            await self._chat_task
        except Exception:
            pass

    async def __call__(self, *args, **kwargs):
        try:
            while True:
                # Check if task is done and queue is empty
                if self._chat_task.done() and self._chat_queue.empty():
                    # Make sure to get any exception from the task
                    self._chat_task.result()
                    break

                # Try to get an event from the queue with timeout
                try:
                    ev, ev_run = await asyncio.wait_for(self._chat_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    # No event available, continue checking
                    if not self._chat_task.done():
                        print("⏱️  [QUEUE] Timeout waiting for event, task still running")
                    continue

                # Process the event
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
                    # Add chat_uuid from the router context (for new chats)
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
            # Ensure any exception is properly handled
            data = {"event": "error", "info": {"message": str(e)}}
            data = json.dumps(data)
            print(f"❌ [QUEUE] Error in chat queue: {e}")
            yield f"data: {data}\n\n"
        finally:
            await self.wait_async()


async def create_chat_streaming_generator_async(
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
    **kwargs,
):
    try:
        tools_by_name = {tool.name: tool for tool in chat_tools or []}
        resolved_chat_tools = list(tools_by_name.values()) or None
        chat_tool_ids = [tool.name for tool in resolved_chat_tools] if resolved_chat_tools else []
        chat_queue = asyncio.Queue()
        base_context = {**(chat_context or {}), "user_uuid": user.uuid, "chat_uuid": chat_uuid}

        async def _run_chat_task():
            owned_session = default_db().create_session()
            finish_run = None
            try:
                context_copy = {**base_context, "session": owned_session}
                agent = await create_agent_async(
                    model_backend=user_config_provider.get_model_backend(),
                    model_config_repository=user_config_provider.get_model_repository(
                        user_uuid=user.uuid, session=owned_session
                    ),
                    agent_backend=user_config_provider.get_agent_backend(),
                    agent_config_repository=user_config_provider.get_agent_repository(
                        user_uuid=user.uuid, session=owned_session
                    ),
                    agent_config_id=chat_agent_id,
                )
                agent_tools = await create_tool_retriever_async(
                    tool_backend=user_config_provider.get_tool_backend(),
                    tools=resolved_chat_tools,
                    tool_config_repository=user_config_provider.get_tool_repository(
                        user_uuid=user.uuid, session=owned_session
                    ),
                    embedding_backend=user_config_provider.get_embedding_backend(),
                    embedding_config_repository=user_config_provider.get_embedding_repository(
                        user_uuid=user.uuid, session=owned_session
                    ),
                    space_id=user.uuid,
                )
                agent_skills = (
                    await create_skill_retriever_async(
                        tool_backend=user_config_provider.get_tool_backend(),
                        skill_config_repository=user_config_provider.get_skill_repository(
                            user_uuid=user.uuid, session=owned_session
                        ),
                        embedding_backend=user_config_provider.get_embedding_backend(),
                        embedding_config_repository=user_config_provider.get_embedding_repository(
                            user_uuid=user.uuid, session=owned_session
                        ),
                        space_id=user.uuid,
                    )
                    if chat_skills_enabled
                    else None
                )

                def _event_callback(ev, run):
                    nonlocal finish_run
                    if ev == AgentRunEvent.FINISH:
                        finish_run = run
                    chat_queue.put_nowait((ev, run))

                await agent.run_async(
                    query=chat_query,
                    tool_retriever=agent_tools,
                    tool_ids=chat_tool_ids,
                    skill_retriever=agent_skills,
                    agent_run_repository=user_chat_provider.get_chat_repository(
                        user_uuid=user.uuid, session=owned_session
                    ),
                    agent_run_session_id=chat_uuid,
                    context=context_copy,
                    event_callback=_event_callback,
                )
            finally:
                try:
                    if chat_finish_callback and finish_run is not None:
                        callback_result = chat_finish_callback(finish_run)
                        if inspect.iscoroutine(callback_result):
                            await callback_result
                finally:
                    try:
                        await owned_session.close()
                    finally:
                        if chat_mutex:
                            chat_mutex.release()

        chat_task = asyncio.create_task(_run_chat_task())
        return _ChatStreamingGenerator(
            chat_task,
            chat_queue,
            chat_uuid=chat_uuid,
        )
    except Exception:
        if chat_mutex:
            chat_mutex.release()
        raise
