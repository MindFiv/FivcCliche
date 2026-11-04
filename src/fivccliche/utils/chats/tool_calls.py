"""Per-call client tool RPC and the session that pumps ``tool_result`` frames."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import timedelta
from typing import Any

from fivccliche.utils.chats.channels import ChatChannel, ChatChannelError

logger = logging.getLogger(__name__)

MAX_TOOL_RESULT_CHARS = 256_000
MAX_TOOL_ERROR_CHARS = 4_096
DEFAULT_TOOL_TIMEOUT = timedelta(seconds=30)
MAX_TOOL_TIMEOUT = timedelta(seconds=120)


class ChatToolCall:
    """One in-flight client tool invoke waiting for a matching ``tool_result``."""

    def __init__(self, call_id: str, name: str, future: asyncio.Future[Any]) -> None:
        self.id = call_id
        self.name = name
        self._future = future

    def resolve(self, frame: dict[str, Any]) -> bool:
        """Apply a ``tool_result`` envelope to this call. Returns True if consumed."""
        if self._future.done():
            return False
        if frame.get("type") != "tool_result":
            return False
        if frame.get("id") != self.id:
            return False

        payload = frame.get("payload")
        if not isinstance(payload, dict):
            payload = frame
        ok = payload.get("ok", True)
        if ok is False:
            error = str(payload.get("error", "Client tool failed"))
            self._future.set_exception(RuntimeError(error[:MAX_TOOL_ERROR_CHARS]))
            return True

        result = payload.get("result", payload.get("data"))
        encoded = json.dumps(result, ensure_ascii=False, default=str)
        if len(encoded) > MAX_TOOL_RESULT_CHARS:
            self._future.set_exception(RuntimeError("Client tool result too large"))
            return True
        self._future.set_result(result)
        return True

    def cancel(self, message: str) -> None:
        if not self._future.done():
            self._future.set_exception(TimeoutError(message))

    async def wait(self, timeout: timedelta) -> Any:
        return await asyncio.wait_for(self._future, timeout=timeout.total_seconds())


class ChatToolSession:
    """Owns pending :class:`ChatToolCall` objects and the inbound ``tool_result`` pump."""

    def __init__(
        self,
        channel: ChatChannel,
        *,
        tool_timeout: timedelta = DEFAULT_TOOL_TIMEOUT,
    ) -> None:
        self._channel = channel
        self._tool_timeout = min(tool_timeout, MAX_TOOL_TIMEOUT)
        self._pending: dict[str, ChatToolCall] = {}
        self._listen_task: asyncio.Task[None] | None = None
        channel.bind_tool_session(self)

    @property
    def channel(self) -> ChatChannel:
        return self._channel

    def cancel_all(self, message: str) -> None:
        pending = list(self._pending.values())
        self._pending.clear()
        for call in pending:
            call.cancel(message)

    def handle_result(self, frame: dict[str, Any]) -> bool:
        """Resolve a pending call from a ``tool_result`` envelope."""
        if frame.get("type") != "tool_result":
            return False
        call_id = frame.get("id")
        if not isinstance(call_id, str):
            return False
        call = self._pending.get(call_id)
        if call is None:
            return False
        return call.resolve(frame)

    async def invoke_async(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: timedelta | None = None,
    ) -> Any:
        """Send ``tool_call`` and wait for matching ``tool_result``."""
        if self._channel.closed:
            raise TimeoutError("Chat channel closed")
        if name not in self._channel.tool_names:
            raise KeyError(f"Unknown client tool: {name}")

        call_id = uuid.uuid4().hex
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        call = ChatToolCall(call_id, name, future)
        self._pending[call_id] = call
        wait = min(timeout or self._tool_timeout, MAX_TOOL_TIMEOUT)
        await self._channel.send_json_async(
            {
                "v": 1,
                "type": "tool_call",
                "id": call_id,
                "payload": {"name": name, "arguments": arguments or {}},
            }
        )
        try:
            return await call.wait(wait)
        except TimeoutError:
            self._pending.pop(call_id, None)
            raise TimeoutError(f"Client tool timed out: {name}") from None
        finally:
            self._pending.pop(call_id, None)

    async def start_listening_async(self) -> None:
        """Start the background pump that reads client frames during streaming."""
        if self._listen_task is not None and not self._listen_task.done():
            return
        self._listen_task = asyncio.create_task(
            self._listen_loop_async(),
            name="chat-tool-session-listen",
        )

    async def stop_listening_async(self) -> ChatChannelError | None:
        """Stop the pump. A protocol failure from the pump is returned to the caller."""
        task = self._listen_task
        self._listen_task = None
        if task is None:
            return None
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return None
        except ChatChannelError as exc:
            return exc
        except Exception:
            logger.debug("ChatToolSession listen task ended with error", exc_info=True)
        return None

    async def _listen_loop_async(self) -> None:
        while not self._channel.closed:
            frame = await self._channel.receive_json_async(raise_exception=True)
            if frame is None:
                self.cancel_all("WebSocket disconnected")
                return
            if frame.get("type") == "tool_result":
                self.handle_result(frame)
                continue
            if self._pending:
                continue
            await self._channel.fail_async(
                code="invalid_message",
                message="Unexpected frame while listening for tool results",
                close_code=1003,
            )
