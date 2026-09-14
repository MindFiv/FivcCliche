"""Transport-independent chat event adapter."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fivcplayground.agents import AgentRunEvent

logger = logging.getLogger(__name__)


class ChatStream:
    """Yield events from an attached agent task's event queue."""

    def __init__(self, chat_uuid: str | None = None) -> None:
        self._chat_uuid = chat_uuid
        self._chat_queue: asyncio.Queue = asyncio.Queue()
        self._asyncio_task: asyncio.Task | None = None

    def attach(self, task: asyncio.Task) -> None:
        """Watch ``task`` so ``events`` can detect completion."""
        self._asyncio_task = task

    def on_event(self, ev, run) -> None:
        """Enqueue an agent event for formatting."""
        self._chat_queue.put_nowait((ev, run))

    async def events(self) -> AsyncIterator[dict]:
        """Yield JSON-compatible event payloads from the agent event queue."""
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
                    info = ev_run.model_dump(mode="json", include=data_fields)
                    info.update({"chat_uuid": self._chat_uuid})
                    yield {"event": "start", "info": info}

                elif ev == AgentRunEvent.FINISH:
                    info = ev_run.model_dump(mode="json", include=data_fields)
                    info.update({"chat_uuid": self._chat_uuid})
                    yield {"event": "finish", "info": info}

                elif ev == AgentRunEvent.STREAM:
                    info = ev_run.model_dump(mode="json", include=data_fields_basics)
                    info.update(
                        {
                            "chat_uuid": self._chat_uuid,
                            "delta": (
                                ev_run.delta.model_dump(mode="json") if ev_run.delta else None
                            ),
                        }
                    )
                    yield {"event": "stream", "info": info}

                elif ev == AgentRunEvent.TOOL:
                    info = ev_run.model_dump(mode="json", include=data_fields)
                    info.update({"chat_uuid": self._chat_uuid})
                    yield {"event": "tool", "info": info}

                self._chat_queue.task_done()

        except Exception as exception:
            message = (
                "Chat message processing timed out"
                if isinstance(exception, TimeoutError)
                else str(exception)
            )
            logger.exception("Error in chat queue")
            yield {"event": "error", "info": {"message": message}}

    async def __call__(self) -> AsyncIterator[str]:
        """Yield the event queue as SSE chunks."""
        async for event in self.events():
            yield f"data: {json.dumps(event)}\n\n"
