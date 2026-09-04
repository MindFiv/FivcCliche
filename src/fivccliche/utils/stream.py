"""SSE adapter for chat agent events."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fivcplayground.agents import AgentRunEvent

logger = logging.getLogger(__name__)


class ChatStream:
    """Yield SSE chunks from an attached agent task's event queue."""

    def __init__(self, chat_uuid: str | None = None) -> None:
        self._chat_uuid = chat_uuid
        self._chat_queue: asyncio.Queue = asyncio.Queue()
        self._asyncio_task: asyncio.Task | None = None

    def attach(self, task: asyncio.Task) -> None:
        """Watch ``task`` so ``__call__`` can detect completion."""
        self._asyncio_task = task

    def on_event(self, ev, run) -> None:
        """Enqueue an agent event for SSE formatting."""
        self._chat_queue.put_nowait((ev, run))

    async def __call__(self) -> AsyncIterator[str]:
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
                except TimeoutError:
                    if self._asyncio_task is None or not self._asyncio_task.done():
                        logger.debug("Timeout waiting for chat event, task still running")
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
            message = "Chat message processing timed out" if isinstance(e, TimeoutError) else str(e)
            data = {"event": "error", "info": {"message": message}}
            data = json.dumps(data)
            logger.exception("Error in chat queue")
            yield f"data: {data}\n\n"
