"""WebSocket protocol channel for chat transports."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import timedelta
from enum import Enum
from typing import Any, NoReturn, Protocol

from fastapi import WebSocket, WebSocketDisconnect

from fivccliche.services.interfaces.auth import IUser, IUserAuthenticator

TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
MAX_TOOL_DESCRIPTION_LEN = 4_096
MAX_PARAMETERS_CHARS = 32_768


class _ToolSession(Protocol):
    def cancel_all(self, message: str) -> None: ...


class ChatChannelError(Exception):
    """A controlled WebSocket failure carrying its protocol close code."""

    def __init__(self, code: int, msg: str) -> None:
        super().__init__(msg)
        self.code = code
        self.msg = msg


class ChatChannelState(str, Enum):
    """Lifecycle for a chat session WebSocket."""

    CONNECTING = "connecting"
    AUTHENTICATED = "authenticated"
    PREPARED = "prepared"
    STREAMING = "streaming"
    CLOSED = "closed"


class ChatChannel:
    """Own WebSocket frames, session tools, and lifecycle after JWT auth."""

    def __init__(
        self,
        ws: WebSocket,
        auth: IUserAuthenticator,
        auth_timeout: timedelta = timedelta(seconds=5),
    ) -> None:
        self._ws = ws
        self._auth = auth
        self._auth_user: IUser | None = None
        self._auth_timeout = auth_timeout
        self._state = ChatChannelState.CONNECTING
        self._tool_specs: dict[str, Any] = {}
        self._tool_session: _ToolSession | None = None
        self._failure: ChatChannelError | None = None

    def bind_tool_session(self, session: _ToolSession) -> None:
        """Attach the client-tool RPC session cancelled on channel close/fail."""
        self._tool_session = session

    @property
    def socket(self) -> WebSocket:
        return self._ws

    @property
    def state(self) -> ChatChannelState:
        return self._state

    @property
    def closed(self) -> bool:
        return self._state is ChatChannelState.CLOSED

    @property
    def tool_specs(self) -> list[Any]:
        return list(self._tool_specs.values())

    @property
    def tool_names(self) -> list[str]:
        return list(self._tool_specs.keys())

    def begin_streaming(self) -> None:
        """Mark the channel as running an agent session (Pipecat or text run)."""
        if self._state is ChatChannelState.CLOSED:
            return
        if self._state not in (
            ChatChannelState.PREPARED,
            ChatChannelState.AUTHENTICATED,
            ChatChannelState.STREAMING,
        ):
            raise ChatChannelError(code=1008, msg="Channel is not prepared to stream")
        self._state = ChatChannelState.STREAMING

    def mark_prepared(self) -> None:
        """Enter PREPARED without registering tools (voice with no client tools)."""
        if self._state is ChatChannelState.CLOSED:
            return
        if self._state in (ChatChannelState.AUTHENTICATED, ChatChannelState.PREPARED):
            self._state = ChatChannelState.PREPARED

    async def __aenter__(self) -> ChatChannel:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        failure = exc_val if isinstance(exc_val, ChatChannelError) else self._failure
        close_code = 1000
        reason = None
        if failure is not None:
            close_code = failure.code
            reason = failure.msg
        await self._close_async(close_code=close_code, reason=reason)
        return isinstance(exc_val, ChatChannelError)

    async def _close_async(self, *, close_code: int = 1000, reason: str | None = None) -> None:
        if self._state is ChatChannelState.CLOSED:
            if self._ws.client_state.name == "CONNECTED":
                await self._ws.close(code=close_code, reason=reason)
            return
        if self._tool_session is not None:
            self._tool_session.cancel_all("Chat channel closed")
        self._tool_specs.clear()
        self._state = ChatChannelState.CLOSED
        if self._ws.client_state.name == "CONNECTED":
            await self._ws.close(code=close_code, reason=reason)

    def _cancel_pending_tools(self, message: str) -> None:
        if self._tool_session is not None:
            self._tool_session.cancel_all(message)

    async def authenticate_async(self) -> IUser:
        """Accept the socket and verify the first-frame JWT."""
        if self._auth_user is not None:
            return self._auth_user

        await self._ws.accept()
        try:
            auth_frame = await asyncio.wait_for(
                self.receive_json_async(raise_exception=True),
                timeout=self._auth_timeout.total_seconds(),
            )
        except TimeoutError:
            await self.fail_async(
                code="unauthorized",
                message="Authentication timed out",
                close_code=1008,
            )
        if auth_frame is None:
            raise ChatChannelError(code=1008, msg="WebSocket disconnected")
        if (
            not isinstance(auth_frame, dict)
            or auth_frame.get("type") != "auth"
            or not isinstance(auth_frame.get("access_token"), str)
        ):
            await self.fail_async(
                code="unauthorized",
                message="Invalid token",
                close_code=1003,
            )

        self._auth_user = await self._auth.verify_credential_async(auth_frame["access_token"])
        if self._auth_user is None:
            await self.fail_async(
                code="unauthorized",
                message="Invalid token",
                close_code=1008,
            )
        self._state = ChatChannelState.AUTHENTICATED
        return self._auth_user

    async def fail_async(
        self,
        *,
        code: str,
        message: str,
        close_code: int,
    ) -> NoReturn:
        """Send the protocol error envelope and raise a controlled error."""
        error = ChatChannelError(code=close_code, msg=message)
        if self._failure is None:
            self._failure = error
        await self.send_json_async({"event": "error", "info": {"code": code, "message": message}})
        self._cancel_pending_tools(message)
        self._state = ChatChannelState.CLOSED
        raise error

    async def register_tools_async(self, frame: dict[str, Any]) -> list[str]:
        """Replace session tools from a ``register_tools`` frame; enter PREPARED."""
        if self._state not in (
            ChatChannelState.AUTHENTICATED,
            ChatChannelState.PREPARED,
        ):
            await self.fail_async(
                code="invalid_state",
                message="register_tools is not allowed in the current state",
                close_code=1008,
            )

        # Import here to avoid a circular import with tools.py at module load.
        from fivccliche.utils.chats.tools import ChatToolSpec

        payload = frame.get("payload")
        raw_tools = payload.get("tools") if isinstance(payload, dict) else frame.get("tools")
        if raw_tools is None:
            raw_tools = []
        if not isinstance(raw_tools, list):
            await self.fail_async(
                code="invalid_message",
                message="register_tools requires a tools list",
                close_code=1003,
            )

        accepted: dict[str, ChatToolSpec] = {}
        warnings: list[str] = []
        for item in raw_tools:
            if not isinstance(item, dict):
                warnings.append("skipped non-object tool entry")
                continue
            name = item.get("name")
            description = item.get("description") or ""
            parameters = item.get("parameters")
            if not isinstance(name, str) or not TOOL_NAME_RE.fullmatch(name):
                warnings.append(f"skipped invalid tool name: {name!r}")
                continue
            if not isinstance(description, str) or len(description) > MAX_TOOL_DESCRIPTION_LEN:
                warnings.append(f"skipped invalid description for {name}")
                continue
            if parameters is None:
                parameters = {"type": "object", "properties": {}}
            if not isinstance(parameters, dict):
                warnings.append(f"skipped invalid parameters for {name}")
                continue
            encoded = json.dumps(parameters, ensure_ascii=False)
            if len(encoded) > MAX_PARAMETERS_CHARS:
                warnings.append(f"skipped oversized parameters for {name}")
                continue
            accepted[name] = ChatToolSpec(
                name=name,
                description=description,
                parameters=parameters,
            )

        self._tool_specs = accepted
        self._state = ChatChannelState.PREPARED
        response: dict[str, Any] = {
            "v": 1,
            "type": "tools_ready",
            "payload": {"tools": list(accepted.keys()), "warnings": warnings},
        }
        await self.send_json_async(response)
        return list(accepted.keys())

    async def receive_async(self, raise_exception: bool = False) -> str | bytes | None:
        """Read the next text or binary frame; disconnect reads as ``None``."""
        try:
            message = await self._ws.receive()
        except WebSocketDisconnect:
            self._cancel_pending_tools("WebSocket disconnected")
            self._state = ChatChannelState.CLOSED
            return None
        except ValueError:
            if not raise_exception:
                return None
            await self.fail_async(
                code="invalid_frame",
                message="Invalid JSON frame",
                close_code=1003,
            )
        if message.get("type") == "websocket.disconnect":
            self._cancel_pending_tools("WebSocket disconnected")
            self._state = ChatChannelState.CLOSED
            return None

        data = message.get("bytes")
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        text = message.get("text")
        if isinstance(text, str):
            return text
        if not raise_exception:
            return None
        await self.fail_async(
            code="invalid_frame",
            message="Invalid WebSocket frame",
            close_code=1003,
        )

    async def receive_text_async(self, raise_exception: bool = False) -> str | None:
        frame = await self.receive_async(raise_exception=raise_exception)
        if frame is None or isinstance(frame, str):
            return frame
        if not raise_exception:
            return None
        await self.fail_async(
            code="invalid_frame",
            message="Text frame required",
            close_code=1003,
        )

    async def receive_json_async(self, raise_exception: bool = False) -> dict[str, Any] | None:
        frame = await self.receive_async(raise_exception=raise_exception)
        if frame is None:
            return None
        if not isinstance(frame, str):
            if not raise_exception:
                return None
            await self.fail_async(
                code="invalid_frame",
                message="Invalid JSON frame",
                close_code=1003,
            )
        try:
            payload = json.loads(frame)
        except json.JSONDecodeError:
            if not raise_exception:
                return None
            await self.fail_async(
                code="invalid_frame",
                message="Invalid JSON frame",
                close_code=1003,
            )
        if not isinstance(payload, dict):
            if not raise_exception:
                return None
            await self.fail_async(
                code="invalid_frame",
                message="Invalid JSON frame",
                close_code=1003,
            )
        return payload

    async def receive_data_async(self, raise_exception: bool = False) -> bytes | None:
        frame = await self.receive_async(raise_exception=raise_exception)
        if frame is None or isinstance(frame, bytes):
            return frame
        if not raise_exception:
            return None
        await self.fail_async(
            code="invalid_frame",
            message="Binary frame required",
            close_code=1003,
        )

    async def send_json_async(self, payload: dict[str, Any], raise_exception: bool = False) -> None:
        if self._state is ChatChannelState.CLOSED:
            return
        try:
            await self._ws.send_json(payload)
        except WebSocketDisconnect:
            self._cancel_pending_tools("WebSocket disconnected")
            self._state = ChatChannelState.CLOSED
            if raise_exception:
                raise ChatChannelError(code=1000, msg="WebSocket disconnected") from None

    async def send_data_async(self, data: bytes, raise_exception: bool = False) -> None:
        if self._state is ChatChannelState.CLOSED:
            return
        try:
            await self._ws.send_bytes(data)
        except WebSocketDisconnect:
            self._cancel_pending_tools("WebSocket disconnected")
            self._state = ChatChannelState.CLOSED
            if raise_exception:
                raise ChatChannelError(code=1000, msg="WebSocket disconnected") from None
