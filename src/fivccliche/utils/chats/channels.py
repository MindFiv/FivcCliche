"""WebSocket protocol channel for chat transports."""

from __future__ import annotations

import asyncio
import json

from datetime import timedelta
from typing import Any, NoReturn

from fastapi import WebSocket, WebSocketDisconnect

from fivccliche.services.interfaces.auth import IUser, IUserAuthenticator


class ChatChannelError(Exception):
    """A controlled WebSocket failure carrying its protocol close code."""

    def __init__(self, code: int, msg: str) -> None:
        super().__init__(msg)
        self.code = code
        self.msg = msg


class ChatChannelClosedError(Exception):
    """The peer disconnected while a frame was being sent."""


class ChatChannel:
    """Own WebSocket frames after first-frame JWT authentication."""

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
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def shutdown(self) -> None:
        """Stop further sends without closing the socket."""
        self._closed = True

    async def __aenter__(self) -> ChatChannel:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        close_code = 1000
        reason = None
        if isinstance(exc_val, ChatChannelError):
            close_code = exc_val.code
            reason = exc_val.msg
        self._closed = True
        if self._ws.client_state.name == "CONNECTED":
            await self._ws.close(code=close_code, reason=reason)
        return isinstance(exc_val, ChatChannelError)

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
        return self._auth_user

    async def fail_async(
        self,
        *,
        code: str,
        message: str,
        close_code: int,
    ) -> NoReturn:
        """Send the protocol error envelope and raise a controlled error."""
        await self.send_json_async({"event": "error", "info": {"code": code, "message": message}})
        self._closed = True
        raise ChatChannelError(code=close_code, msg=message)

    async def receive_async(self, raise_exception: bool = False) -> str | bytes | None:
        """Read the next text or binary frame; disconnect reads as ``None``."""
        try:
            message = await self._ws.receive()
        except WebSocketDisconnect:
            self._closed = True
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
            self._closed = True
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
        if self._closed:
            return
        try:
            await self._ws.send_json(payload)
        except WebSocketDisconnect:
            self._closed = True
            if raise_exception:
                raise ChatChannelClosedError() from None

    async def send_data_async(self, data: bytes, raise_exception: bool = False) -> None:
        if self._closed:
            return
        try:
            await self._ws.send_bytes(data)
        except WebSocketDisconnect:
            self._closed = True
            if raise_exception:
                raise ChatChannelClosedError() from None
