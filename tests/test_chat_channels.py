"""Unit tests for the WebSocket channel boundary."""

import asyncio
import json
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import WebSocketDisconnect
from json import JSONDecodeError

from fivccliche.utils.chats import ChatChannel, ChatChannelError


class TestChatChannel:
    """Test first-frame auth, typed receive, sends, and context-owned close."""

    @staticmethod
    def _fake_websocket(frames=None, *, hang_on_empty=False):
        class FakeWebSocket:
            def __init__(self):
                self.frames = list(frames or [])
                self.sent = []
                self.closed = []
                self.accepted = False
                self.client_state = MagicMock()
                self.client_state.name = "CONNECTED"

            async def accept(self):
                self.accepted = True

            async def receive(self):
                if not self.frames:
                    if hang_on_empty:
                        await asyncio.sleep(10)
                    raise WebSocketDisconnect
                frame = self.frames.pop(0)
                if isinstance(frame, Exception):
                    raise frame
                if isinstance(frame, (bytes, bytearray)):
                    return {"type": "websocket.receive", "bytes": bytes(frame)}
                if isinstance(frame, str):
                    return {"type": "websocket.receive", "text": frame}
                return {"type": "websocket.receive", "text": json.dumps(frame)}

            async def send_json(self, message):
                self.sent.append(message)

            async def send_bytes(self, message):
                self.sent.append(message)

            async def close(self, code=1000, reason=None):
                self.closed.append((code, reason))

        return FakeWebSocket()

    @staticmethod
    def _auth(user=None, *, valid=True):
        auth = MagicMock()
        auth.verify_credential_async = AsyncMock(return_value=user if valid else None)
        return auth

    @pytest.mark.asyncio
    async def test_authenticate_returns_verified_user_and_closes_on_exit(self):
        websocket = self._fake_websocket([{"type": "auth", "access_token": "valid-token"}])
        verified = MagicMock()
        auth = self._auth(verified)

        async with ChatChannel(websocket, auth) as channel:
            assert await channel.authenticate_async() is verified

        auth.verify_credential_async.assert_awaited_once_with("valid-token")
        assert websocket.accepted is True
        assert websocket.sent == []
        assert websocket.closed == [(1000, None)]

    @pytest.mark.asyncio
    async def test_authenticate_times_out(self):
        websocket = self._fake_websocket(hang_on_empty=True)
        auth = MagicMock()
        channel = ChatChannel(websocket, auth, timedelta(seconds=0.01))

        with pytest.raises(ChatChannelError, match="Authentication timed out") as exc_info:
            await channel.authenticate_async()
        exc = exc_info.value
        await channel.__aexit__(type(exc), exc, exc.__traceback__)

        auth.verify_credential_async.assert_not_called()
        assert websocket.sent == [
            {
                "event": "error",
                "info": {"code": "unauthorized", "message": "Authentication timed out"},
            }
        ]
        assert websocket.closed == [(1008, "Authentication timed out")]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "auth_frame", [JSONDecodeError("Invalid JSON", "", 0), ValueError("bad"), b"pcm"]
    )
    async def test_authenticate_rejects_malformed_frame(self, auth_frame):
        websocket = self._fake_websocket([auth_frame])
        channel = ChatChannel(websocket, self._auth())
        await websocket.accept()

        with pytest.raises(ChatChannelError, match="Invalid JSON frame") as exc_info:
            await channel.authenticate_async()
        exc = exc_info.value
        await channel.__aexit__(type(exc), exc, exc.__traceback__)

        assert websocket.sent[-1]["info"]["code"] == "invalid_frame"
        assert websocket.closed == [(1003, "Invalid JSON frame")]

    @pytest.mark.asyncio
    async def test_authenticate_rejects_invalid_token(self):
        websocket = self._fake_websocket([{"type": "auth", "access_token": "invalid-token"}])
        channel = ChatChannel(websocket, self._auth(valid=False))
        await websocket.accept()

        with pytest.raises(ChatChannelError, match="Invalid token") as exc_info:
            await channel.authenticate_async()
        exc = exc_info.value
        await channel.__aexit__(type(exc), exc, exc.__traceback__)

        assert websocket.sent == [
            {"event": "error", "info": {"code": "unauthorized", "message": "Invalid token"}}
        ]
        assert websocket.closed == [(1008, "Invalid token")]

    @pytest.mark.asyncio
    async def test_authenticate_raises_on_disconnect(self):
        websocket = self._fake_websocket()
        channel = ChatChannel(websocket, self._auth())

        with pytest.raises(ChatChannelError, match="WebSocket disconnected") as exc_info:
            await channel.authenticate_async()
        exc = exc_info.value
        await channel.__aexit__(type(exc), exc, exc.__traceback__)

        assert websocket.sent == []
        assert websocket.closed == [(1008, "WebSocket disconnected")]

    @pytest.mark.asyncio
    async def test_receive_returns_mixed_frames(self):
        websocket = self._fake_websocket(["plain", b"pcm"])
        async with ChatChannel(websocket, self._auth()) as channel:
            await websocket.accept()
            assert await channel.receive_async() == "plain"
            assert await channel.receive_async() == b"pcm"

    @pytest.mark.asyncio
    async def test_receive_json_decodes_objects(self):
        frame = {"type": "start"}
        websocket = self._fake_websocket([frame])
        async with ChatChannel(websocket, self._auth()) as channel:
            await websocket.accept()
            assert await channel.receive_json_async() == frame

    @pytest.mark.asyncio
    async def test_receive_json_rejects_malformed_and_non_objects(self):
        websocket = self._fake_websocket(["not-json", "[]"])
        channel = ChatChannel(websocket, self._auth())
        await websocket.accept()

        with pytest.raises(ChatChannelError, match="Invalid JSON frame") as exc_info:
            await channel.receive_json_async(raise_exception=True)
        exc = exc_info.value
        await channel.__aexit__(type(exc), exc, exc.__traceback__)

        assert websocket.closed == [(1003, "Invalid JSON frame")]

    @pytest.mark.asyncio
    async def test_receive_json_rejects_binary(self):
        websocket = self._fake_websocket([b"pcm"])
        channel = ChatChannel(websocket, self._auth())
        await websocket.accept()

        with pytest.raises(ChatChannelError, match="Invalid JSON frame") as exc_info:
            await channel.receive_json_async(raise_exception=True)
        exc = exc_info.value
        await channel.__aexit__(type(exc), exc, exc.__traceback__)

        assert websocket.closed == [(1003, "Invalid JSON frame")]

    @pytest.mark.asyncio
    async def test_receive_text_rejects_binary(self):
        websocket = self._fake_websocket([b"pcm"])
        channel = ChatChannel(websocket, self._auth())
        await websocket.accept()

        with pytest.raises(ChatChannelError, match="Text frame required") as exc_info:
            await channel.receive_text_async(raise_exception=True)
        exc = exc_info.value
        await channel.__aexit__(type(exc), exc, exc.__traceback__)

        assert websocket.closed == [(1003, "Text frame required")]

    @pytest.mark.asyncio
    async def test_receive_data_rejects_text(self):
        websocket = self._fake_websocket(["plain"])
        channel = ChatChannel(websocket, self._auth())
        await websocket.accept()

        with pytest.raises(ChatChannelError, match="Binary frame required") as exc_info:
            await channel.receive_data_async(raise_exception=True)
        exc = exc_info.value
        await channel.__aexit__(type(exc), exc, exc.__traceback__)

        assert websocket.closed == [(1003, "Binary frame required")]

    @pytest.mark.asyncio
    async def test_receive_returns_none_on_disconnect(self):
        websocket = self._fake_websocket()
        async with ChatChannel(websocket, self._auth()) as channel:
            await websocket.accept()
            websocket.client_state.name = "DISCONNECTED"
            assert await channel.receive_async() is None

        assert websocket.closed == []

    @pytest.mark.asyncio
    async def test_direct_send_primitives(self):
        websocket = self._fake_websocket()
        async with ChatChannel(websocket, self._auth()) as channel:
            await websocket.accept()
            await channel.send_json_async({"event": "finish"})
            await channel.send_data_async(b"pcm")

        assert websocket.sent == [{"event": "finish"}, b"pcm"]

    @pytest.mark.asyncio
    async def test_send_disconnect_marks_closed_without_raising(self):
        websocket = self._fake_websocket()
        channel = ChatChannel(websocket, self._auth())
        await websocket.accept()

        async def fail_send(_message):
            raise WebSocketDisconnect

        websocket.send_json = fail_send
        await channel.send_json_async({"event": "finish"})

        assert channel.closed is True
        await channel.send_data_async(b"pcm")

    @pytest.mark.asyncio
    async def test_fail_sends_error_and_raises_without_closing(self):
        websocket = self._fake_websocket()
        channel = ChatChannel(websocket, self._auth())
        await websocket.accept()

        with pytest.raises(ChatChannelError, match="Chat not found"):
            await channel.fail_async(
                code="chat_not_found", message="Chat not found", close_code=4004
            )

        assert websocket.sent == [
            {"event": "error", "info": {"code": "chat_not_found", "message": "Chat not found"}}
        ]
        assert websocket.closed == []

    @pytest.mark.asyncio
    async def test_aexit_closes_and_suppresses_controlled_error(self):
        websocket = self._fake_websocket()
        async with ChatChannel(websocket, self._auth()):
            raise ChatChannelError(code=4004, msg="Chat not found")

        assert websocket.closed == [(4004, "Chat not found")]

    @pytest.mark.asyncio
    async def test_aexit_closes_after_unexpected_error_and_propagates(self):
        websocket = self._fake_websocket()
        with pytest.raises(RuntimeError, match="unexpected"):
            async with ChatChannel(websocket, self._auth()):
                raise RuntimeError("unexpected")

        assert websocket.closed == [(1000, None)]
