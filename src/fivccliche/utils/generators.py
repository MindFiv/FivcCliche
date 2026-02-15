import asyncio
import inspect
import json
from collections.abc import Callable
from collections.abc import Awaitable

from fivcplayground.agents import AgentRunEvent, AgentRun, create_agent_async
from fivcplayground.tools import create_tool_retriever_async, Tool
from sqlalchemy.ext.asyncio.session import AsyncSession

from fivccliche.services.interfaces.agent_chats import IUserChatProvider
from fivccliche.services.interfaces.agent_configs import IUserConfigProvider
from fivccliche.services.interfaces.auth import IUser


class _ChatStreamingGenerator:
    """Generator for streaming agent runs."""

    def __init__(
        self,
        chat_task: asyncio.Task,
        chat_queue: asyncio.Queue,
        chat_uuid: str | None = None,
        chat_finish_callback: (
            Callable[[AgentRun], None] | Callable[[AgentRun], Awaitable[None]] | None
        ) = None,
    ):
        self.chat_task = chat_task
        self.chat_queue = chat_queue
        self.chat_uuid = chat_uuid
        self.chat_finish_callback = chat_finish_callback

    async def __call__(self, *args, **kwargs):
        try:
            while True:
                # Check if task is done and queue is empty
                if self.chat_task.done() and self.chat_queue.empty():
                    # Make sure to get any exception from the task
                    self.chat_task.result()
                    break

                # Try to get an event from the queue with timeout
                try:
                    ev, ev_run = await asyncio.wait_for(self.chat_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    # No event available, continue checking
                    if not self.chat_task.done():
                        print("⏱️  [QUEUE] Timeout waiting for event, task still running")
                    continue

                # Process the event
                data_fields = {
                    "id",
                    "agent_id",
                    "started_at",
                    "completed_at",
                    "query",
                    "reply",
                    "tool_calls",
                }
                if ev == AgentRunEvent.START:
                    data = ev_run.model_dump(mode="json", include=data_fields)
                    # Add chat_uuid from the router context (for new chats)
                    data.update({"chat_uuid": self.chat_uuid})
                    data = {"event": "start", "info": data}
                    data_json = json.dumps(data)
                    yield f"data: {data_json}\n\n"

                elif ev == AgentRunEvent.FINISH:
                    data = ev_run.model_dump(mode="json", include=data_fields)
                    data.update({"chat_uuid": self.chat_uuid})
                    data = {"event": "finish", "info": data}
                    data_json = json.dumps(data)
                    yield f"data: {data_json}\n\n"

                    if self.chat_finish_callback:
                        result = self.chat_finish_callback(ev_run)
                        if inspect.iscoroutine(result):
                            await result

                elif ev == AgentRunEvent.STREAM:
                    data = ev_run.model_dump(mode="json", include=data_fields)
                    data.update(
                        {
                            "chat_uuid": self.chat_uuid,
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
                    data.update({"chat_uuid": self.chat_uuid})
                    data = {"event": "tool", "info": data}
                    data = json.dumps(data)
                    yield f"data: {data}\n\n"

                self.chat_queue.task_done()

        except Exception as e:
            # Ensure any exception is properly handled
            data = {"event": "error", "info": {"message": str(e)}}
            data = json.dumps(data)
            print(f"❌ [QUEUE] Error in chat queue: {e}")
            yield f"data: {data}\n\n"


async def create_chat_streaming_generator_async(
    user: IUser,
    user_config_provider: IUserConfigProvider,
    user_chat_provider: IUserChatProvider,
    chat_query: str = "",
    chat_uuid: str | None = None,
    chat_agent_id: str = "default",
    chat_tools: list[Tool] | None = None,
    chat_finish_callback: (
        Callable[[AgentRun], None] | Callable[[AgentRun], Awaitable[None]] | None
    ) = None,
    session: AsyncSession | None = None,
    **kwargs,
):
    chat_tool_ids = [tool.name for tool in chat_tools] if chat_tools else []
    agent = await create_agent_async(
        model_backend=user_config_provider.get_model_backend(),
        model_config_repository=user_config_provider.get_model_repository(
            user_uuid=user.uuid, session=session
        ),
        agent_backend=user_config_provider.get_agent_backend(),
        agent_config_repository=user_config_provider.get_agent_repository(
            user_uuid=user.uuid, session=session
        ),
        agent_config_id=chat_agent_id,
    )
    agent_tools = await create_tool_retriever_async(
        tool_backend=user_config_provider.get_tool_backend(),
        tools=chat_tools,
        tool_config_repository=user_config_provider.get_tool_repository(
            user_uuid=user.uuid, session=session
        ),
        embedding_backend=user_config_provider.get_embedding_backend(),
        embedding_config_repository=user_config_provider.get_embedding_repository(
            user_uuid=user.uuid, session=session
        ),
        space_id=user.uuid,
    )
    chat_queue = asyncio.Queue()

    # chat_uuid = chat.uuid if chat else str(uuid.uuid4())

    # Debug: Event callback wrapper
    def _event_callback(ev, run):
        chat_queue.put_nowait((ev, run))

    chat_task = asyncio.create_task(
        agent.run_async(
            query=chat_query,
            tool_retriever=agent_tools,
            tool_ids=chat_tool_ids,
            agent_run_repository=user_chat_provider.get_chat_repository(
                user_uuid=user.uuid, session=session
            ),
            agent_run_session_id=chat_uuid,
            event_callback=_event_callback,
        )
    )
    return _ChatStreamingGenerator(
        chat_task,
        chat_queue,
        chat_uuid=chat_uuid,
        chat_finish_callback=chat_finish_callback,
    )
