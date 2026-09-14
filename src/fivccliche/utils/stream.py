"""Transport-independent chat event adapter."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import WebSocket, WebSocketDisconnect
from fivcplayground.agents import AgentRunEvent

from fivccliche.services.interfaces.auth import IUser, IUserAuthenticator

logger = logging.getLogger(__name__)


class ChatEventHandler:
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


ChatStream = ChatEventHandler


class ChatWSHandler:
    """One-shot WebSocket protocol for first-frame JWT auth and event send."""

    def __init__(self, websocket: WebSocket, *, auth_timeout: float = 5.0) -> None:
        self._ws = websocket
        self._auth_timeout = auth_timeout

    async def fail(self, *, code: str, message: str, close_code: int) -> None:
        """Send an error envelope and close the socket."""
        await self._ws.send_json({"event": "error", "info": {"code": code, "message": message}})
        await self._ws.close(code=close_code)

    async def authenticate(self, authenticator: IUserAuthenticator) -> IUser | None:
        """Accept the socket and verify a first-frame JWT.

        Returns ``None`` when the socket has already been closed with an error.
        """
        await self._ws.accept()
        try:
            auth_frame = await asyncio.wait_for(
                self._ws.receive_json(),
                timeout=self._auth_timeout,
            )
        except TimeoutError:
            await self.fail(
                code="unauthorized",
                message="Authentication timed out",
                close_code=1008,
            )
            return None
        except WebSocketDisconnect:
            await self._ws.close(code=1008)
            return None
        except (json.JSONDecodeError, ValueError):
            await self.fail(
                code="invalid_frame",
                message="Invalid JSON frame",
                close_code=1003,
            )
            return None

        if (
            not isinstance(auth_frame, dict)
            or auth_frame.get("type") != "auth"
            or not isinstance(auth_frame.get("access_token"), str)
        ):
            await self.fail(
                code="unauthorized",
                message="Invalid token",
                close_code=1008,
            )
            return None

        user = await authenticator.verify_credential_async(auth_frame["access_token"])
        if user is None:
            await self.fail(
                code="unauthorized",
                message="Invalid token",
                close_code=1008,
            )
            return None
        return user

    async def receive(self, *, expected_type: str, invalid_code: str) -> dict | None:
        """Read one JSON object of ``expected_type``.

        Disconnect closes ``1000``. Invalid JSON closes ``1003``. A type
        mismatch closes ``1003`` with ``invalid_code``.
        """
        try:
            frame = await self._ws.receive_json()
        except WebSocketDisconnect:
            await self._ws.close(code=1000)
            return None
        except (json.JSONDecodeError, ValueError):
            await self.fail(
                code="invalid_frame",
                message="Invalid JSON frame",
                close_code=1003,
            )
            return None

        if not isinstance(frame, dict) or frame.get("type") != expected_type:
            await self.fail(
                code=invalid_code,
                message=f"A {expected_type} frame is required",
                close_code=1003,
            )
            return None
        return frame

    async def send(self, events: AsyncIterator[dict]) -> bool:
        """Push JSON events. Return True if the client disconnected."""
        try:
            async for event in events:
                await self._ws.send_json(event)
        except WebSocketDisconnect:
            return True
        return False
