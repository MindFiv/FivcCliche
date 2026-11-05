"""Unit tests for chat frontend tool registration and bridging."""

import asyncio
import inspect
import json
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import WebSocketDisconnect

from fivccliche.utils.chats import (
    ChatChannel,
    ChatChannelError,
    ChatChannelState,
    ChatToolBridge,
    ChatToolCall,
    ChatToolSession,
)
from fivccliche.utils.chats.tools import ChatToolSpec, make_client_tool_callable


class TestChatChannelTools:
    @staticmethod
    def _fake_websocket(frames=None, *, blocking: bool = False):
        class FakeWebSocket:
            def __init__(self):
                self.sent = []
                self.closed = []
                self.accepted = False
                self.client_state = MagicMock()
                self.client_state.name = "CONNECTED"
                self._blocking = blocking
                if blocking:
                    self._inbound: asyncio.Queue = asyncio.Queue()
                    for frame in frames or []:
                        self._inbound.put_nowait(frame)
                else:
                    self.frames = list(frames or [])

            async def accept(self):
                self.accepted = True

            def push(self, frame):
                if not self._blocking:
                    raise RuntimeError("push requires blocking=True")
                self._inbound.put_nowait(frame)

            def disconnect(self):
                if not self._blocking:
                    raise RuntimeError("disconnect requires blocking=True")
                self._inbound.put_nowait(None)

            async def receive(self):
                if self._blocking:
                    frame = await self._inbound.get()
                    if frame is None:
                        raise WebSocketDisconnect
                else:
                    if not self.frames:
                        raise WebSocketDisconnect
                    frame = self.frames.pop(0)
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
    def _auth(user=None):
        auth = MagicMock()
        auth.verify_credential_async = AsyncMock(return_value=user or MagicMock())
        return auth

    @pytest.mark.asyncio
    async def test_register_tools_enters_prepared_and_echoes_ready(self):
        websocket = self._fake_websocket([{"type": "auth", "access_token": "t"}])
        async with ChatChannel(websocket, self._auth()) as channel:
            await channel.authenticate_async()
            assert channel.state is ChatChannelState.AUTHENTICATED
            names = await channel.register_tools_async(
                {
                    "type": "register_tools",
                    "payload": {
                        "tools": [
                            {
                                "name": "pick_file",
                                "description": "Pick a file",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"accept": {"type": "string"}},
                                },
                            },
                            {"name": "bad name", "description": "nope"},
                        ]
                    },
                }
            )
            assert names == ["pick_file"]
            assert channel.state is ChatChannelState.PREPARED
            assert channel.tool_names == ["pick_file"]

        ready = websocket.sent[0]
        assert ready["type"] == "tools_ready"
        assert ready["payload"]["tools"] == ["pick_file"]
        assert ready["payload"]["warnings"]

    @pytest.mark.asyncio
    async def test_register_tools_rejects_over_limit(self):
        websocket = self._fake_websocket([{"type": "auth", "access_token": "t"}])
        async with ChatChannel(websocket, self._auth()) as channel:
            await channel.authenticate_async()
            tools = [
                {
                    "name": f"tool_{index}",
                    "description": "Valid",
                    "parameters": {"type": "object", "properties": {}},
                }
                for index in range(65)
            ]
            with pytest.raises(ChatChannelError, match="at most 64 tools"):
                await channel.register_tools_async(
                    {"type": "register_tools", "payload": {"tools": tools}}
                )

    @pytest.mark.asyncio
    async def test_register_tools_skips_invalid_parameter_schemas(self):
        websocket = self._fake_websocket([{"type": "auth", "access_token": "t"}])
        async with ChatChannel(websocket, self._auth()) as channel:
            await channel.authenticate_async()
            names = await channel.register_tools_async(
                {
                    "type": "register_tools",
                    "payload": {
                        "tools": [
                            {
                                "name": "not_object",
                                "description": "Invalid schema",
                                "parameters": {"type": "string"},
                            },
                            {
                                "name": "bad_property",
                                "description": "Invalid name",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"bad-name": {"type": "string"}},
                                },
                            },
                            {
                                "name": "keyword_property",
                                "description": "Reserved name",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"class": {"type": "string"}},
                                },
                            },
                            {
                                "name": "unknown_required",
                                "description": "Required points outside properties",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"accept": {"type": "string"}},
                                    "required": ["missing"],
                                },
                            },
                            {
                                "name": "invalid_property_schema",
                                "description": "Property schema is not an object",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"accept": "string"},
                                },
                            },
                        ]
                    },
                }
            )
            assert names == []
            assert channel.tool_names == []

        ready = websocket.sent[0]
        assert ready["payload"]["tools"] == []
        assert len(ready["payload"]["warnings"]) == 5

    @pytest.mark.asyncio
    async def test_invoke_tool_pairs_with_tool_result_via_pump(self):
        websocket = self._fake_websocket(
            [{"type": "auth", "access_token": "t"}],
            blocking=True,
        )
        async with ChatChannel(websocket, self._auth()) as channel:
            await channel.authenticate_async()
            await channel.register_tools_async(
                {
                    "type": "register_tools",
                    "tools": [{"name": "ui_open", "description": "Open UI", "parameters": {}}],
                }
            )
            tool_session = ChatToolSession(channel)
            channel.begin_streaming()
            await tool_session.start_listening_async()
            try:

                async def respond():
                    for _ in range(50):
                        await asyncio.sleep(0)
                        calls = [m for m in websocket.sent if m.get("type") == "tool_call"]
                        if calls:
                            call = calls[-1]
                            websocket.push(
                                {
                                    "type": "tool_result",
                                    "id": call["id"],
                                    "payload": {"ok": True, "result": {"opened": True}},
                                }
                            )
                            return
                    raise AssertionError("tool_call was not sent")

                task = asyncio.create_task(respond())
                result = await tool_session.invoke_async("ui_open", {"panel": "settings"})
                await task
                assert result == {"opened": True}
            finally:
                await tool_session.stop_listening_async()

    @pytest.mark.asyncio
    async def test_invoke_tool_times_out(self):
        websocket = self._fake_websocket([{"type": "auth", "access_token": "t"}])
        channel = ChatChannel(websocket, self._auth())
        await channel.authenticate_async()
        await channel.register_tools_async(
            {
                "type": "register_tools",
                "tools": [{"name": "slow", "description": "Slow", "parameters": {}}],
            }
        )
        tool_session = ChatToolSession(channel, tool_timeout=timedelta(milliseconds=20))
        with pytest.raises(TimeoutError, match="timed out"):
            await tool_session.invoke_async("slow", {})
        await channel.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_invoke_tool_rejects_pending_limit(self):
        websocket = self._fake_websocket([{"type": "auth", "access_token": "t"}])
        channel = ChatChannel(websocket, self._auth())
        await channel.authenticate_async()
        await channel.register_tools_async(
            {
                "type": "register_tools",
                "tools": [{"name": "busy", "description": "Busy", "parameters": {}}],
            }
        )
        tool_session = ChatToolSession(channel, max_pending_tools=0)
        with pytest.raises(RuntimeError, match="Too many pending client tools"):
            await tool_session.invoke_async("busy", {})
        await channel.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_listen_fails_on_unexpected_frame_without_pending(self):
        websocket = self._fake_websocket(
            [{"type": "auth", "access_token": "t"}],
            blocking=True,
        )
        async with ChatChannel(websocket, self._auth()) as channel:
            await channel.authenticate_async()
            await channel.register_tools_async({"type": "register_tools", "tools": []})
            tool_session = ChatToolSession(channel)
            channel.begin_streaming()
            await tool_session.start_listening_async()
            websocket.push({"type": "user_message", "payload": {"query": "nope"}})
            for _ in range(50):
                await asyncio.sleep(0.01)
                if channel.closed:
                    break
            assert channel.closed
            assert any(
                isinstance(frame, dict)
                and frame.get("event") == "error"
                and frame.get("info", {}).get("code") == "invalid_message"
                for frame in websocket.sent
            )
            listen_error = await tool_session.stop_listening_async()
            assert isinstance(listen_error, ChatChannelError)
            assert listen_error.code == 1003
        assert websocket.closed[-1][0] == 1003

    @pytest.mark.asyncio
    async def test_late_tool_result_after_timeout_does_not_close(self):
        websocket = self._fake_websocket(
            [{"type": "auth", "access_token": "t"}],
            blocking=True,
        )
        async with ChatChannel(websocket, self._auth()) as channel:
            await channel.authenticate_async()
            await channel.register_tools_async(
                {
                    "type": "register_tools",
                    "tools": [{"name": "slow", "description": "Slow", "parameters": {}}],
                }
            )
            tool_session = ChatToolSession(channel, tool_timeout=timedelta(milliseconds=20))
            channel.begin_streaming()
            await tool_session.start_listening_async()
            try:
                with pytest.raises(TimeoutError, match="timed out"):
                    await tool_session.invoke_async("slow", {})
                websocket.push(
                    {
                        "type": "tool_result",
                        "id": "late",
                        "payload": {"ok": True, "result": {"late": True}},
                    }
                )
                await asyncio.sleep(0.05)
                assert channel.state is ChatChannelState.STREAMING
            finally:
                listen_error = await tool_session.stop_listening_async()
            assert listen_error is None
        assert websocket.closed[-1][0] == 1000

    @pytest.mark.asyncio
    async def test_listen_ignores_unexpected_frame_while_pending(self):
        websocket = self._fake_websocket(
            [{"type": "auth", "access_token": "t"}],
            blocking=True,
        )
        async with ChatChannel(websocket, self._auth()) as channel:
            await channel.authenticate_async()
            await channel.register_tools_async(
                {
                    "type": "register_tools",
                    "tools": [{"name": "ui_open", "description": "Open UI", "parameters": {}}],
                }
            )
            tool_session = ChatToolSession(channel)
            channel.begin_streaming()
            await tool_session.start_listening_async()
            try:

                async def respond():
                    for _ in range(50):
                        await asyncio.sleep(0)
                        calls = [m for m in websocket.sent if m.get("type") == "tool_call"]
                        if calls:
                            call = calls[-1]
                            websocket.push({"type": "noise", "payload": {}})
                            await asyncio.sleep(0)
                            websocket.push(
                                {
                                    "type": "tool_result",
                                    "id": call["id"],
                                    "payload": {"ok": True, "result": {"opened": True}},
                                }
                            )
                            return
                    raise AssertionError("tool_call was not sent")

                task = asyncio.create_task(respond())
                result = await tool_session.invoke_async("ui_open", {})
                await task
                assert result == {"opened": True}
            finally:
                await tool_session.stop_listening_async()

    @pytest.mark.asyncio
    async def test_register_tools_rejected_while_streaming(self):
        websocket = self._fake_websocket([{"type": "auth", "access_token": "t"}])
        async with ChatChannel(websocket, self._auth()) as channel:
            await channel.authenticate_async()
            await channel.register_tools_async({"type": "register_tools", "tools": []})
            channel.begin_streaming()
            with pytest.raises(ChatChannelError, match="not allowed"):
                await channel.register_tools_async({"type": "register_tools", "tools": []})

    @pytest.mark.asyncio
    async def test_begin_streaming_still_allows_json_sends(self):
        websocket = self._fake_websocket([{"type": "auth", "access_token": "t"}])
        async with ChatChannel(websocket, self._auth()) as channel:
            await channel.authenticate_async()
            await channel.register_tools_async({"type": "register_tools", "tools": []})
            channel.begin_streaming()
            assert channel.state is ChatChannelState.STREAMING
            await channel.send_json_async({"event": "finish"})
        assert {"event": "finish"} in websocket.sent


class TestChatToolCall:
    @pytest.mark.asyncio
    async def test_error_string_is_truncated(self):
        future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        call = ChatToolCall("id", "ui_open", future)
        assert call.resolve(
            {
                "type": "tool_result",
                "id": "id",
                "payload": {"ok": False, "error": "x" * 5_000},
            }
        )
        with pytest.raises(RuntimeError) as exc_info:
            await future
        assert str(exc_info.value) == "x" * 4_096


class TestMakeClientToolCallable:
    @pytest.mark.asyncio
    async def test_signature_mirrors_flat_object_parameters(self):
        session = MagicMock()
        session.invoke_async = AsyncMock(return_value={"ok": True})
        spec = ChatToolSpec(
            name="pick_file",
            description="Pick a file",
            parameters={
                "type": "object",
                "properties": {
                    "accept": {"type": "string"},
                    "multiple": {"type": "boolean"},
                    "bad-name": {"type": "string"},
                },
                "required": ["accept"],
            },
        )
        fn = make_client_tool_callable(session, spec)
        signature = inspect.signature(fn)
        assert list(signature.parameters) == ["accept", "multiple"]
        assert signature.parameters["accept"].annotation is str
        assert signature.parameters["accept"].default is inspect.Parameter.empty
        assert signature.parameters["multiple"].annotation is bool
        assert signature.parameters["multiple"].default is None

        result = await fn(accept="image/*", multiple=True)
        assert result == {"ok": True}
        session.invoke_async.assert_awaited_once_with(
            "pick_file", {"accept": "image/*", "multiple": True}
        )


class TestChatToolBridge:
    @pytest.mark.asyncio
    async def test_inject_uses_create_tool_with_signed_callable(self):
        session = MagicMock()
        session.channel = MagicMock()
        session.channel.tool_specs = [
            ChatToolSpec(
                name="pick_file",
                description="Pick a file",
                parameters={
                    "type": "object",
                    "properties": {"accept": {"type": "string"}},
                    "required": ["accept"],
                },
            )
        ]
        session.invoke_async = AsyncMock(return_value={"path": "/tmp/a"})

        backend = MagicMock()
        created = MagicMock(name="pick_file")
        created.name = "pick_file"
        backend.create_tool.return_value = created

        retriever = MagicMock()
        retriever.tools = {}
        bridge = ChatToolBridge(session)
        ids = bridge.inject(retriever, backend, tool_ids=["existing"])
        assert ids == ["existing", "pick_file"]
        assert retriever.tools["pick_file"] is created
        backend.create_tool.assert_called_once()
        call_kwargs = backend.create_tool.call_args.kwargs
        assert call_kwargs["tool_name"] == "pick_file"
        assert call_kwargs["tool_description"] == "Pick a file"
        signed = backend.create_tool.call_args.args[0]
        assert list(inspect.signature(signed).parameters) == ["accept"]

    @pytest.mark.asyncio
    async def test_inject_rejects_configured_tool_name_collision(self):
        session = MagicMock()
        session.channel = MagicMock()
        session.channel.tool_specs = [
            ChatToolSpec(
                name="pick_file",
                description="Pick a file",
                parameters={"type": "object", "properties": {}},
            )
        ]
        backend = MagicMock()
        configured = MagicMock(name="pick_file")
        retriever = MagicMock()
        retriever.tools = {"pick_file": configured}
        bridge = ChatToolBridge(session)

        with pytest.raises(ValueError, match="pick_file"):
            bridge.inject(retriever, backend)

        backend.create_tool.assert_not_called()
        assert list(retriever.tools) == ["pick_file"]
        assert retriever.tools["pick_file"] is configured
