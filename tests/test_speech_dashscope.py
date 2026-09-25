"""Unit tests for DashScope Qwen3-ASR adapters (HTTP mocked, WS injected)."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

import pytest

from fivccliche.services.implements.speech.dashscope import (
    _DEFAULT_GENERATION_URL,
    _DashScopeMultimodalRecognizer,
    DashScopeSpeechProvider,
)
from fivccliche.services.implements.speech.dashscope_realtime import (
    _connect_websockets,
    _tts_start_frame,
    _websocket_url,
    _websocket_headers,
    _DashScopeRealtimeRecognizer,
    DashScopeRealtimeSpeechProvider,
)
from fivccliche.services.implements.speech.dashscope_realtime import (
    _DashScopeTTSSynthesizer,
)
from fivccliche.services.interfaces.speech import (
    SpeechAudioInput,
    SpeechEvent,
    SpeechRecognizeOptions,
    SpeechSynthesisOptions,
    SpeechRequestError,
)


def _flash_response(
    *, text: str = "欢迎使用阿里云。", language: str = "zh", emotion: str = "neutral"
):
    response = MagicMock()
    response.is_success = True
    response.status_code = 200
    response.json.return_value = {
        "output": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "annotations": [
                            {"language": language, "type": "audio_info", "emotion": emotion}
                        ],
                        "content": [{"text": text}],
                        "role": "assistant",
                    },
                }
            ]
        },
        "usage": {"seconds": 1},
        "request_id": "req-1",
    }
    return response


def _qwen_audio_response(output: dict[str, Any]):
    response = MagicMock()
    response.is_success = True
    response.status_code = 200
    response.json.return_value = {"output": output, "request_id": "req-qwen-audio"}
    return response


async def _collect(recognizer, audio) -> list[SpeechEvent]:
    return [event async for event in recognizer.stream_async(audio)]


async def _collect_bytes(synthesizer, text) -> list[bytes]:
    return [chunk async for chunk in synthesizer.stream_async(text)]


class TestDashScopeMultimodalRecognizer:
    @pytest.mark.asyncio
    async def test_recognize_posts_url_and_maps_transcript(self):
        http_client = MagicMock()
        http_client.post = AsyncMock(return_value=_flash_response())
        recognizer = _DashScopeMultimodalRecognizer(
            api_key="sk-test",
            model="qwen3-asr-flash",
            http_client=http_client,
            options=SpeechRecognizeOptions(language="zh", enable_itn=True, context="fivc"),
        )
        async with recognizer:
            events = await _collect(
                recognizer,
                SpeechAudioInput(url="https://example.com/a.mp3", format="mp3"),
            )

        http_client.post.assert_awaited_once()
        args, kwargs = http_client.post.call_args
        assert args[0] == _DEFAULT_GENERATION_URL
        assert kwargs["headers"]["Authorization"] == "Bearer sk-test"
        assert kwargs["headers"]["Content-Type"] == "application/json"
        payload = kwargs["json"]
        assert payload["model"] == "qwen3-asr-flash"
        messages = payload["input"]["messages"]
        assert messages[0] == {"role": "system", "content": [{"text": "fivc"}]}
        assert messages[1]["content"][0]["audio"] == "https://example.com/a.mp3"
        assert payload["parameters"]["asr_options"]["language"] == "zh"
        assert payload["parameters"]["asr_options"]["enable_itn"] is True
        assert events == [
            SpeechEvent(type="final", text="欢迎使用阿里云。", language="zh"),
        ]

    @pytest.mark.asyncio
    async def test_recognize_wraps_base64_as_data_uri(self):
        http_client = MagicMock()
        http_client.post = AsyncMock(return_value=_flash_response(text="hi"))
        recognizer = _DashScopeMultimodalRecognizer(api_key="sk-test", http_client=http_client)
        async with recognizer:
            await _collect(recognizer, SpeechAudioInput(data_b64="abc123", format="wav"))

        audio = http_client.post.call_args.kwargs["json"]["input"]["messages"][0]["content"][0][
            "audio"
        ]
        assert audio == "data:audio/wav;base64,abc123"

    @pytest.mark.asyncio
    async def test_custom_model_uses_qwen3_payload(self):
        http_client = MagicMock()
        http_client.post = AsyncMock(return_value=_flash_response(text="hi"))
        recognizer = _DashScopeMultimodalRecognizer(
            api_key="sk-test",
            model="custom-flash",
            http_client=http_client,
            options=SpeechRecognizeOptions(language="zh", enable_itn=True),
        )

        async with recognizer:
            events = await _collect(
                recognizer,
                SpeechAudioInput(url="https://example.com/a.wav", format="wav"),
            )

        kwargs = http_client.post.call_args.kwargs
        assert kwargs["headers"]["Authorization"] == "Bearer sk-test"
        assert "X-DashScope-SSE" not in kwargs["headers"]
        assert kwargs["json"]["input"]["messages"][-1]["content"] == [
            {"audio": "https://example.com/a.wav"}
        ]
        assert kwargs["json"]["parameters"] == {
            "asr_options": {"enable_itn": True, "language": "zh"}
        }
        assert events[-1].text == "hi"

    @pytest.mark.asyncio
    async def test_recognize_buffers_byte_stream(self):
        http_client = MagicMock()
        http_client.post = AsyncMock(return_value=_flash_response(text="hi"))
        recognizer = _DashScopeMultimodalRecognizer(
            api_key="sk-test",
            http_client=http_client,
            options=SpeechRecognizeOptions(format="pcm"),
        )

        async def chunks():
            yield b"abc"
            yield b"123"

        async with recognizer:
            events = await _collect(recognizer, chunks())
        audio = http_client.post.call_args.kwargs["json"]["input"]["messages"][0]["content"][0][
            "audio"
        ]
        assert audio == "data:audio/pcm;base64," + base64.b64encode(b"abc123").decode("ascii")
        assert events[0].text == "hi"

    @pytest.mark.asyncio
    async def test_recognize_http_error_raises(self):
        response = MagicMock()
        response.is_success = False
        response.status_code = 401
        response.text = "unauthorized"
        http_client = MagicMock()
        http_client.post = AsyncMock(return_value=response)
        recognizer = _DashScopeMultimodalRecognizer(api_key="bad", http_client=http_client)
        async with recognizer:
            with pytest.raises(SpeechRequestError, match="401"):
                await _collect(recognizer, SpeechAudioInput(url="https://example.com/a.wav"))

    @pytest.mark.asyncio
    async def test_recognize_requires_http_client(self):
        recognizer = _DashScopeMultimodalRecognizer(api_key="sk-test")
        with pytest.raises(SpeechRequestError, match="not started"):
            await _collect(recognizer, SpeechAudioInput(data_b64="abc", format="wav"))

    @pytest.mark.asyncio
    async def test_owned_http_client_closes_on_exit(self):
        recognizer = _DashScopeMultimodalRecognizer(api_key="sk-test")
        async with recognizer:
            assert recognizer._http_client is not None
            owned_client = recognizer._http_client
        assert owned_client.is_closed


class TestDashScopeQwenAudioRecognizer:
    @pytest.mark.asyncio
    async def test_recognize_posts_qwen_audio_payload(self):
        http_client = MagicMock()
        http_client.post = AsyncMock(
            return_value=_qwen_audio_response({"text": "欢迎使用阿里云。"})
        )
        recognizer = _DashScopeMultimodalRecognizer(
            api_key="sk-test",
            model="qwen-audio-3.0-asr-flash",
            http_client=http_client,
            options=SpeechRecognizeOptions(language="zh", enable_itn=True, context="fivc"),
        )

        async with recognizer:
            events = await _collect(
                recognizer,
                SpeechAudioInput(url="https://example.com/a.mp3", format="mp3"),
            )

        http_client.post.assert_awaited_once()
        args, kwargs = http_client.post.call_args
        assert args[0] == _DEFAULT_GENERATION_URL
        assert kwargs["headers"]["Authorization"] == "Bearer sk-test"
        assert kwargs["headers"]["X-DashScope-SSE"] == "disable"
        assert kwargs["json"] == {
            "model": "qwen-audio-3.0-asr-flash",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": "https://example.com/a.mp3",
                                },
                            }
                        ],
                    }
                ]
            },
            "parameters": {"format": "mp3"},
        }
        assert events == [SpeechEvent(type="final", text="欢迎使用阿里云。", language=None)]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("output", "expected_text"),
        [
            ({"text": "hello"}, "hello"),
            ({"sentence": {"text": "world"}}, "world"),
        ],
    )
    async def test_recognize_parses_qwen_audio_outputs(self, output, expected_text):
        http_client = MagicMock()
        http_client.post = AsyncMock(return_value=_qwen_audio_response(output))
        recognizer = _DashScopeMultimodalRecognizer(
            api_key="sk-test",
            model="qwen-audio-3.0-asr-flash",
            http_client=http_client,
        )

        async with recognizer:
            events = await _collect(
                recognizer,
                SpeechAudioInput(data_b64="abc123", format="wav"),
            )

        payload = http_client.post.call_args.kwargs["json"]
        content = payload["input"]["messages"][0]["content"][0]
        assert content["input_audio"]["data"].startswith("data:audio/wav;base64,")
        assert payload["parameters"] == {"format": "wav"}
        assert events == [SpeechEvent(type="final", text=expected_text, language=None)]


class _FakeDashScopeWs:
    def __init__(self, *, result_sentences: list[dict] | None = None) -> None:
        self.result_sentences = result_sentences or []
        self.messages: list[dict] = []
        self.audio_frames: list[bytes] = []
        self._incoming: asyncio.Queue[str | bytes] = asyncio.Queue()
        self.closed = False

    async def send(self, data: str) -> None:
        message = json.loads(data)
        self.messages.append(message)
        action = message["header"]["action"]
        if action == "run-task":
            await self._incoming.put(
                json.dumps(
                    {
                        "header": {
                            "event": "task-started",
                            "task_id": message["header"]["task_id"],
                        },
                        "payload": {"input": {}},
                    }
                )
            )
            return
        if action == "finish-task":
            await self._emit_results()

    async def send_bytes(self, data: bytes) -> None:
        self.audio_frames.append(data)

    async def _emit_results(self) -> None:
        task_id = self.messages[0]["header"]["task_id"]
        for sentence in self.result_sentences:
            await self._incoming.put(
                json.dumps(
                    {
                        "header": {
                            "event": "result-generated",
                            "task_id": task_id,
                        },
                        "payload": {"output": {"sentence": sentence}},
                    }
                )
            )
        await self._incoming.put(
            json.dumps(
                {
                    "header": {"event": "task-finished", "task_id": task_id},
                    "payload": {"output": {}},
                }
            )
        )

    async def fail(self, code: str, message: str) -> None:
        task_id = self.messages[0]["header"]["task_id"]
        await self._incoming.put(
            json.dumps(
                {
                    "header": {
                        "event": "task-failed",
                        "task_id": task_id,
                        "error_code": code,
                        "error_message": message,
                    },
                    "payload": {"input": {}},
                }
            )
        )

    async def recv(self) -> str:
        return await self._incoming.get()

    async def close(self) -> None:
        self.closed = True


class TestWebSocketUrl:
    @pytest.mark.parametrize(
        ("base_url", "expected_url"),
        [
            (
                "https://dashscope.aliyuncs.com",
                "wss://dashscope.aliyuncs.com/api-ws/v1/inference",
            ),
            ("https://host/compatible-mode/v1", "wss://host/api-ws/v1/inference"),
            ("wss://host", "wss://host/api-ws/v1/inference"),
            ("wss://host/", "wss://host/api-ws/v1/inference"),
            ("wss://host/api-ws/v1/inference", "wss://host/api-ws/v1/inference"),
            ("wss://host/custom/path?model=x", "wss://host/custom/path?model=x"),
        ],
    )
    def test_normalizes_dashscope_base_url(self, base_url: str, expected_url: str):
        assert _websocket_url(base_url) == expected_url


class TestWebSocketHandshake:
    @pytest.mark.asyncio
    async def test_realtime_sends_clean_api_key_only(self):
        fake_ws = _FakeDashScopeWs()
        connections: list[tuple[str, dict[str, str]]] = []

        async def connect(url: str, headers: dict[str, str]) -> _FakeDashScopeWs:
            connections.append((url, headers))
            return fake_ws

        recognizer = _DashScopeRealtimeRecognizer(
            api_key=" Bearer sk-test \n",
            ws_url=(
                "wss://llm-pbm19qg9671grxpg.cn-beijing.maas.aliyuncs.com" "/api-ws/v1/inference"
            ),
            connect=connect,
            options=SpeechRecognizeOptions(format="pcm"),
        )
        async with recognizer:
            pass

        assert connections == [
            (
                "wss://llm-pbm19qg9671grxpg.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference",
                {"Authorization": "Bearer sk-test"},
            )
        ]

    @pytest.mark.asyncio
    async def test_synthesizer_sends_clean_api_key_only(self):
        fake_ws = _FakeDashScopeTtsWs()
        connections: list[tuple[str, dict[str, str]]] = []

        async def connect(url: str, headers: dict[str, str]) -> _FakeDashScopeTtsWs:
            connections.append((url, headers))
            return fake_ws

        synthesizer = _DashScopeTTSSynthesizer(
            api_key=" Bearer sk-test \n",
            ws_url=(
                "wss://llm-pbm19qg9671grxpg.cn-beijing.maas.aliyuncs.com" "/api-ws/v1/inference"
            ),
            options=SpeechSynthesisOptions(voice="longanhuan_v3.1"),
            connect=connect,
        )
        async with synthesizer:
            pass

        assert connections == [
            (
                "wss://llm-pbm19qg9671grxpg.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference",
                {"Authorization": "Bearer sk-test"},
            )
        ]

    def test_headers_contain_authorization_only(self):
        headers = _websocket_headers("sk-test")
        assert headers == {"Authorization": "Bearer sk-test"}

    @pytest.mark.asyncio
    async def test_connection_failure_includes_dashscope_response_body(self):
        response = Response(
            403,
            "Forbidden",
            Headers(),
            body=(
                b'{"code":"AccessDenied","message":"The model is not authorized.",'
                b'"request_id":"req-403"}'
            ),
        )
        url = "wss://llm-pbm19qg9671grxpg.cn-beijing.maas.aliyuncs.com" "/api-ws/v1/inference"
        with (
            patch(
                "websockets.connect",
                new_callable=AsyncMock,
                side_effect=InvalidStatus(response),
            ),
            pytest.raises(
                SpeechRequestError,
                match=(
                    rf"Failed to connect to DashScope \({url}\): "
                    r"HTTP 403 AccessDenied: The model is not authorized."
                ),
            ),
        ):
            await _connect_websockets(url, {"Authorization": "Bearer sk-test"})


def _sentence(text: str, *, end_time: int | None = None) -> dict:
    sentence: dict[str, Any] = {"text": text}
    if end_time is not None:
        sentence["end_time"] = end_time
    return sentence


class _FakeDashScopeTtsWs:
    def __init__(self, audio_chunks: list[bytes] | None = None) -> None:
        self.audio_chunks = audio_chunks or []
        self.messages: list[dict] = []
        self.texts: list[str] = []
        self.closed = False
        self.finish_sent = False
        self.audio_sent = False
        self._incoming: asyncio.Queue[str | bytes] = asyncio.Queue()

    async def send(self, data: str) -> None:
        message = json.loads(data)
        self.messages.append(message)
        action = message["header"]["action"]
        if action == "run-task":
            await self._incoming.put(json.dumps({"header": {"event": "task-started"}}))
        elif action == "continue-task":
            self.texts.append(message["payload"]["input"]["text"])
            if not self.audio_sent:
                self.audio_sent = True
                for chunk in self.audio_chunks:
                    await self._incoming.put(chunk)
        elif action == "finish-task":
            self.finish_sent = True
            await self._incoming.put(json.dumps({"header": {"event": "task-finished"}}))

    async def recv(self) -> str | bytes:
        return await self._incoming.get()

    async def fail(self, code: str, message: str) -> None:
        await self._incoming.put(
            json.dumps(
                {
                    "header": {
                        "event": "task-failed",
                        "error_code": code,
                        "error_message": message,
                    }
                }
            )
        )

    async def close(self) -> None:
        self.closed = True


class TestDashScopeRealtimeRecognizer:
    @pytest.mark.asyncio
    async def test_starts_recognition_task_and_maps_sentences(self):
        fake_ws = _FakeDashScopeWs(
            result_sentences=[
                _sentence("北"),
                _sentence("北京", end_time=1234),
                _sentence("北京。", end_time=2345),
            ]
        )

        async def connect(url: str, headers: dict[str, str]) -> _FakeDashScopeWs:
            connect.last_url = url  # type: ignore[attr-defined]
            connect.last_headers = headers  # type: ignore[attr-defined]
            return fake_ws

        recognizer = _DashScopeRealtimeRecognizer(
            api_key="sk-test",
            model="qwen-audio-3.1-asr-flash-streaming",
            connect=connect,
            options=SpeechRecognizeOptions(
                format="pcm",
                sample_rate=16000,
                extra={"disfluency_removal_enabled": True},
            ),
        )

        async def chunks():
            yield b"x" * 12800
            yield b"end"

        async with recognizer:
            start = fake_ws.messages[0]
            assert start["header"]["action"] == "run-task"
            assert start["header"]["streaming"] == "duplex"
            assert start["header"]["task_id"]
            assert start["payload"]["model"] == "qwen-audio-3.1-asr-flash-streaming"
            assert start["payload"]["task_group"] == "audio"
            assert start["payload"]["task"] == "recognition"
            assert start["payload"]["function"] == "recognition"
            assert start["payload"]["parameters"] == {
                "format": "pcm",
                "sample_rate": 16000,
                "disfluency_removal_enabled": True,
            }
            assert start["payload"]["input"] == {}
            assert connect.last_headers["Authorization"] == "Bearer sk-test"  # type: ignore[attr-defined]
            assert connect.last_url == "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
            events = await asyncio.wait_for(_collect(recognizer, chunks()), timeout=2)

        assert fake_ws.audio_frames == [b"x" * 12800, b"end"]
        finish = fake_ws.messages[-1]
        assert finish["header"]["action"] == "finish-task"
        assert finish["payload"] == {"input": {}}
        assert [event.type for event in events] == ["partial", "final", "final"]
        assert events[0].text == "北"
        assert events[1].text == "北京"
        assert events[2].text == "北京。"
        assert fake_ws.closed is True

    @pytest.mark.asyncio
    async def test_finish_without_result_yields_empty_final(self):
        fake_ws = _FakeDashScopeWs(result_sentences=[])

        async def connect(url: str, headers: dict[str, str]) -> _FakeDashScopeWs:
            return fake_ws

        recognizer = _DashScopeRealtimeRecognizer(
            api_key="sk-test",
            connect=connect,
            options=SpeechRecognizeOptions(format="wav"),
        )
        payload = base64.b64encode(b"clip-bytes").decode("ascii")
        async with recognizer:
            events = await asyncio.wait_for(
                _collect(recognizer, SpeechAudioInput(data_b64=payload, format="wav")),
                timeout=2,
            )
        assert fake_ws.audio_frames == [b"clip-bytes"]
        assert [(event.type, event.text) for event in events] == [("final", "")]

    @pytest.mark.asyncio
    async def test_task_failed_yields_error_event(self):
        fake_ws = _FakeDashScopeWs(result_sentences=[])

        async def connect(url: str, headers: dict[str, str]) -> _FakeDashScopeWs:
            return fake_ws

        recognizer = _DashScopeRealtimeRecognizer(api_key="sk-test", connect=connect)
        async with recognizer:
            await fake_ws.fail("InvalidParameter", "format is invalid")
            events = await asyncio.wait_for(
                _collect(recognizer, SpeechAudioInput(data_b64="abc", format="wav")),
                timeout=2,
            )
        assert len(events) == 1
        assert events[0].type == "error"
        assert events[0].message == "InvalidParameter: format is invalid"

    @pytest.mark.asyncio
    async def test_clip_url_is_fetched(self):
        fake_ws = _FakeDashScopeWs()

        async def connect(url: str, headers: dict[str, str]) -> _FakeDashScopeWs:
            return fake_ws

        response = MagicMock()
        response.is_success = True
        response.content = b"from-url"
        http_client = MagicMock()
        http_client.get = AsyncMock(return_value=response)
        recognizer = _DashScopeRealtimeRecognizer(
            api_key="sk-test",
            connect=connect,
            http_client=http_client,
        )
        async with recognizer:
            await asyncio.wait_for(
                _collect(recognizer, SpeechAudioInput(url="https://example.com/a.wav")),
                timeout=2,
            )
        http_client.get.assert_awaited_once_with("https://example.com/a.wav")
        assert fake_ws.audio_frames == [b"from-url"]

    @pytest.mark.asyncio
    async def test_recognize_before_aenter_raises(self):
        recognizer = _DashScopeRealtimeRecognizer(api_key="sk-test", connect=AsyncMock())
        with pytest.raises(SpeechRequestError, match="not started"):
            await _collect(recognizer, SpeechAudioInput(data_b64="abc"))


def _speech_config(values: dict) -> MagicMock:
    session = MagicMock()
    session.get_value.side_effect = lambda key: values.get(key)
    config = MagicMock()
    config.get_session.return_value = session
    return config


class TestDashScopeSpeechProvider:
    @pytest.mark.asyncio
    async def test_get_recognizer_builds_flash(self):
        options = SpeechRecognizeOptions(language="zh")
        with (
            patch(
                "fivccliche.services.implements.speech.dashscope.query_component",
                return_value=_speech_config(
                    {"ASR_API_KEY": "sk-x", "ASR_MODEL": "qwen3-asr-flash"}
                ),
            ),
            patch(
                "fivccliche.services.implements.speech.dashscope._DashScopeMultimodalRecognizer",
            ) as clip_cls,
        ):
            provider = DashScopeSpeechProvider(MagicMock())
            recognizer = await provider.get_recognizer(options)

        clip_cls.assert_called_once()
        assert recognizer is clip_cls.return_value
        assert clip_cls.call_args.kwargs["api_key"] == "sk-x"
        assert clip_cls.call_args.kwargs["model"] == "qwen3-asr-flash"
        assert clip_cls.call_args.kwargs["options"] is options

    @pytest.mark.asyncio
    async def test_get_recognizer_overrides_credentials(self):
        options = SpeechRecognizeOptions(language="zh")
        with (
            patch(
                "fivccliche.services.implements.speech.dashscope.query_component",
                return_value=_speech_config(
                    {"ASR_API_KEY": "sk-x", "ASR_MODEL": "qwen3-asr-flash"}
                ),
            ),
            patch(
                "fivccliche.services.implements.speech.dashscope._DashScopeMultimodalRecognizer",
            ) as clip_cls,
        ):
            provider = DashScopeSpeechProvider(MagicMock())
            await provider.get_recognizer(
                options,
                api_key="sk-override",
                model="custom-flash",
                base_url="https://example.com/",
            )

        assert clip_cls.call_args.kwargs["api_key"] == "sk-override"
        assert clip_cls.call_args.kwargs["model"] == "custom-flash"
        assert clip_cls.call_args.kwargs["base_url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_get_recognizer_empty_api_key_is_explicit(self):
        with (
            patch(
                "fivccliche.services.implements.speech.dashscope.query_component",
                return_value=_speech_config({"ASR_API_KEY": "sk-x"}),
            ),
            patch(
                "fivccliche.services.implements.speech.dashscope._DashScopeMultimodalRecognizer",
            ) as clip_cls,
        ):
            provider = DashScopeSpeechProvider(MagicMock())
            await provider.get_recognizer(api_key="")

        assert clip_cls.call_args.kwargs["api_key"] == ""


class TestDashScopeRealtimeSpeechProvider:
    @pytest.mark.asyncio
    async def test_get_recognizer_only_constructs(self):
        stream = MagicMock()
        stream.start_async = AsyncMock()
        stream.__aenter__ = AsyncMock(return_value=stream)
        stream.__aexit__ = AsyncMock(return_value=None)
        options = SpeechRecognizeOptions(format="pcm")
        with (
            patch(
                "fivccliche.services.implements.speech.dashscope_realtime.query_component",
                return_value=_speech_config(
                    {
                        "ASR_API_KEY": "sk-x",
                        "ASR_MODEL": "qwen-audio-3.1-asr-flash-streaming",
                    }
                ),
            ),
            patch(
                "fivccliche.services.implements.speech.dashscope_realtime._DashScopeRealtimeRecognizer",
                return_value=stream,
            ) as stream_cls,
        ):
            provider = DashScopeRealtimeSpeechProvider(MagicMock())
            recognizer = await provider.get_recognizer(options)

        stream_cls.assert_called_once()
        assert recognizer is stream
        stream.start_async.assert_not_called()
        stream.__aenter__.assert_not_called()
        assert stream_cls.call_args.kwargs["model"] == ("qwen-audio-3.1-asr-flash-streaming")
        assert stream_cls.call_args.kwargs["ws_url"] == (
            "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
        )
        assert stream_cls.call_args.kwargs["options"] is options

        async with recognizer:
            pass
        stream.__aenter__.assert_awaited_once()
        stream.start_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_recognizer_overrides_credentials(self):
        stream = MagicMock()
        options = SpeechRecognizeOptions(format="pcm")
        with (
            patch(
                "fivccliche.services.implements.speech.dashscope_realtime.query_component",
                return_value=_speech_config(
                    {
                        "ASR_API_KEY": "sk-x",
                        "ASR_MODEL": "qwen-audio-3.1-asr-flash-streaming",
                    }
                ),
            ),
            patch(
                "fivccliche.services.implements.speech.dashscope_realtime._DashScopeRealtimeRecognizer",
                return_value=stream,
            ) as stream_cls,
        ):
            provider = DashScopeRealtimeSpeechProvider(MagicMock())
            await provider.get_recognizer(
                options,
                api_key="sk-override",
                model="custom-rt",
                base_url="https://example.com",
            )

        assert stream_cls.call_args.kwargs["api_key"] == "sk-override"
        assert stream_cls.call_args.kwargs["model"] == "custom-rt"
        assert stream_cls.call_args.kwargs["ws_url"] == ("wss://example.com/api-ws/v1/inference")
        assert stream_cls.call_args.kwargs["options"] is options

    @pytest.mark.asyncio
    async def test_get_recognizer_accepts_websocket_base_url(self):
        stream = MagicMock()
        with (
            patch(
                "fivccliche.services.implements.speech.dashscope_realtime.query_component",
                return_value=_speech_config({}),
            ),
            patch(
                "fivccliche.services.implements.speech.dashscope_realtime._DashScopeRealtimeRecognizer",
                return_value=stream,
            ) as stream_cls,
        ):
            provider = DashScopeRealtimeSpeechProvider(MagicMock())
            await provider.get_recognizer(
                base_url=(
                    "wss://llm-pbm19qg9671grxpg.cn-beijing.maas.aliyuncs.com" "/api-ws/v1/inference"
                )
            )

        assert stream_cls.call_args.kwargs["ws_url"] == (
            "wss://llm-pbm19qg9671grxpg.cn-beijing.maas.aliyuncs.com" "/api-ws/v1/inference"
        )

    @pytest.mark.asyncio
    async def test_recognizer_and_synthesizer_share_maas_origin_url(self):
        base_url = "wss://llm-pbm19qg9671grxpg.cn-beijing.maas.aliyuncs.com/"
        with (
            patch(
                "fivccliche.services.implements.speech.dashscope_realtime.query_component",
                return_value=_speech_config({}),
            ),
            patch(
                "fivccliche.services.implements.speech.dashscope_realtime._DashScopeRealtimeRecognizer"
            ) as recognizer_cls,
            patch(
                "fivccliche.services.implements.speech.dashscope_realtime._DashScopeTTSSynthesizer"
            ) as synthesizer_cls,
        ):
            provider = DashScopeRealtimeSpeechProvider(MagicMock())
            await provider.get_recognizer(base_url=base_url)
            await provider.get_synthesizer(
                SpeechSynthesisOptions(voice="longanhuan_v3.1"), base_url=base_url
            )

        expected_url = "wss://llm-pbm19qg9671grxpg.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference"
        assert recognizer_cls.call_args.kwargs["ws_url"] == expected_url
        assert synthesizer_cls.call_args.kwargs["ws_url"] == expected_url


class TestDashScopeTTS:
    @pytest.mark.asyncio
    async def test_get_synthesizer_defaults_without_options(self):
        with (
            patch(
                "fivccliche.services.implements.speech.dashscope_realtime.query_component",
                return_value=_speech_config({}),
            ),
            patch(
                "fivccliche.services.implements.speech.dashscope_realtime._DashScopeTTSSynthesizer",
            ) as synthesizer_cls,
        ):
            provider = DashScopeRealtimeSpeechProvider(MagicMock())
            await provider.get_synthesizer()

        assert synthesizer_cls.call_args.kwargs["options"] is None

    @pytest.mark.asyncio
    async def test_synthesizer_defaults_to_compatible_voice(self):
        fake_ws = _FakeDashScopeTtsWs()

        async def connect(url: str, headers: dict[str, str]) -> _FakeDashScopeTtsWs:
            return fake_ws

        synthesizer = _DashScopeTTSSynthesizer(
            api_key="sk-test",
            model="qwen-audio-3.1-tts-flash",
            options=None,
            connect=connect,
        )
        async with synthesizer:
            assert synthesizer._options.voice == "longanhuan_v3.1"

    def test_default_voice_is_compatible_with_qwen_audio_tts(self):
        options = SpeechSynthesisOptions(voice="longanhuan_v3.1")
        frame = _tts_start_frame("qwen-audio-3.1-tts-flash", options, "task-id")

        assert frame["payload"]["parameters"]["voice"] == "longanhuan_v3.1"

    @pytest.mark.asyncio
    async def test_rejects_cherry_for_qwen_audio_tts_without_connecting(self):
        async def connect(url: str, headers: dict[str, str]) -> None:
            raise AssertionError("incompatible voice must not open a WebSocket")

        synthesizer = _DashScopeTTSSynthesizer(
            api_key="sk-test",
            model="qwen-audio-3.1-tts-flash",
            options=SpeechSynthesisOptions(voice="Cherry"),
            connect=connect,
        )
        with pytest.raises(
            SpeechRequestError,
            match=(
                r"voice Cherry is not supported by qwen-audio-3\.1-tts-flash; "
                r"use longanhuan_v3\.1"
            ),
        ):
            await synthesizer.__aenter__()

    @pytest.mark.asyncio
    async def test_starts_native_websocket_tts_task(self):
        fake_ws = _FakeDashScopeTtsWs()

        async def connect(url: str, headers: dict[str, str]) -> _FakeDashScopeTtsWs:
            connect.last_url = url  # type: ignore[attr-defined]
            connect.last_headers = headers  # type: ignore[attr-defined]
            return fake_ws

        options = SpeechSynthesisOptions(
            voice="longanhuan_v3.1",
            format="wav",
            sample_rate=24000,
            volume=70,
            speech_rate=1.2,
            pitch_rate=0.9,
            extra={"enable_ssml": True},
        )

        synthesizer = _DashScopeTTSSynthesizer(
            api_key="sk-test",
            model="qwen-audio-3.1-tts-flash",
            ws_url="wss://dashscope.aliyuncs.com/api-ws/v1/inference",
            options=options,
            connect=connect,
        )
        async with synthesizer:
            start = fake_ws.messages[0]

        assert start["header"]["action"] == "run-task"
        assert start["header"]["streaming"] == "duplex"
        assert start["payload"]["model"] == "qwen-audio-3.1-tts-flash"
        assert start["payload"]["task_group"] == "audio"
        assert start["payload"]["task"] == "tts"
        assert start["payload"]["function"] == "SpeechSynthesizer"
        assert start["payload"]["input"] == {}
        assert start["payload"]["parameters"] == {
            "voice": "longanhuan_v3.1",
            "volume": 70,
            "text_type": "PlainText",
            "sample_rate": 24000,
            "rate": 1.2,
            "format": "wav",
            "pitch": 0.9,
            "seed": 0,
            "type": 0,
            "enable_ssml": True,
        }
        assert connect.last_headers["Authorization"] == "Bearer sk-test"  # type: ignore[attr-defined]
        assert connect.last_url == "wss://dashscope.aliyuncs.com/api-ws/v1/inference"  # type: ignore[attr-defined]
        assert fake_ws.messages[-1]["header"]["action"] == "finish-task"
        assert fake_ws.closed is True

    @pytest.mark.asyncio
    async def test_synthesizes_complete_text(self):
        fake_ws = _FakeDashScopeTtsWs([b"left ", b"right"])

        async def connect(url: str, headers: dict[str, str]) -> _FakeDashScopeTtsWs:
            return fake_ws

        synthesizer = _DashScopeTTSSynthesizer(
            api_key="sk-test",
            model="qwen-audio-3.1-tts-flash",
            ws_url="wss://dashscope.aliyuncs.com/api-ws/v1/inference",
            options=SpeechSynthesisOptions(voice="longanhuan_v3.1"),
            connect=connect,
        )

        async with synthesizer:
            audio = await asyncio.wait_for(_collect_bytes(synthesizer, "你好"), timeout=2)

        assert audio == [b"left ", b"right"]
        assert [message["header"]["action"] for message in fake_ws.messages] == [
            "run-task",
            "continue-task",
            "finish-task",
        ]
        assert fake_ws.messages[1]["payload"]["input"] == {"text": "你好"}
        assert fake_ws.texts == ["你好"]
        assert fake_ws.finish_sent is True
        assert fake_ws.closed is True

    @pytest.mark.asyncio
    async def test_synthesizes_streamed_text(self):
        fake_ws = _FakeDashScopeTtsWs([b"streamed"])

        async def connect(url: str, headers: dict[str, str]) -> _FakeDashScopeTtsWs:
            return fake_ws

        synthesizer = _DashScopeTTSSynthesizer(
            api_key="sk-test",
            model="qwen-audio-3.1-tts-flash",
            options=SpeechSynthesisOptions(voice="longanhuan_v3.1"),
            connect=connect,
        )

        async def text():
            yield "你好"
            yield "世界"

        async with synthesizer:
            audio = await asyncio.wait_for(_collect_bytes(synthesizer, text()), timeout=2)

        assert audio == [b"streamed"]
        assert fake_ws.texts == ["你好", "世界"]
        assert fake_ws.finish_sent is True

    @pytest.mark.asyncio
    async def test_wraps_task_failure(self):
        fake_ws = _FakeDashScopeTtsWs()

        async def connect(url: str, headers: dict[str, str]) -> _FakeDashScopeTtsWs:
            return fake_ws

        synthesizer = _DashScopeTTSSynthesizer(
            api_key="sk-test",
            model="qwen-audio-3.1-tts-flash",
            options=SpeechSynthesisOptions(voice="longanhuan_v3.1"),
            connect=connect,
        )
        async with synthesizer:
            await fake_ws.fail("BadTTS", "synthesis failed")
            with pytest.raises(SpeechRequestError, match="BadTTS: synthesis failed"):
                await _collect_bytes(synthesizer, "你好")
        assert fake_ws.closed is True

    @pytest.mark.asyncio
    async def test_wraps_connection_failure(self):
        synthesizer = _DashScopeTTSSynthesizer(
            api_key="sk-test",
            model="qwen-audio-3.1-tts-flash",
            options=SpeechSynthesisOptions(voice="longanhuan_v3.1"),
        )
        with (
            patch(
                "websockets.connect",
                new_callable=AsyncMock,
                side_effect=RuntimeError("connection refused"),
            ),
            pytest.raises(
                SpeechRequestError,
                match=r"Failed to connect to DashScope \(wss://[^)]+\): connection refused",
            ),
        ):
            await synthesizer.__aenter__()

    @pytest.mark.asyncio
    async def test_get_synthesizer_uses_tts_defaults_and_overrides(self):
        options = SpeechSynthesisOptions(voice="longanhuan_v3.1")
        with (
            patch(
                "fivccliche.services.implements.speech.dashscope_realtime.query_component",
                return_value=_speech_config({"TTS_API_KEY": "sk-x"}),
            ),
            patch(
                "fivccliche.services.implements.speech.dashscope_realtime._DashScopeTTSSynthesizer",
            ) as synthesizer_cls,
        ):
            provider = DashScopeRealtimeSpeechProvider(MagicMock())
            await provider.get_synthesizer(options)

        assert synthesizer_cls.call_args.kwargs["api_key"] == "sk-x"
        assert synthesizer_cls.call_args.kwargs["model"] == "qwen-audio-3.1-tts-flash"
        assert synthesizer_cls.call_args.kwargs["ws_url"] == (
            "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
        )
        assert synthesizer_cls.call_args.kwargs["options"] is options

        with (
            patch(
                "fivccliche.services.implements.speech.dashscope_realtime.query_component",
                return_value=_speech_config({}),
            ),
            patch(
                "fivccliche.services.implements.speech.dashscope_realtime._DashScopeTTSSynthesizer",
            ) as synthesizer_cls,
        ):
            provider = DashScopeRealtimeSpeechProvider(MagicMock())
            await provider.get_synthesizer(
                options,
                api_key="sk-override",
                model="custom-tts",
                base_url="https://example.com",
            )

        assert synthesizer_cls.call_args.kwargs["api_key"] == "sk-override"
        assert synthesizer_cls.call_args.kwargs["model"] == "custom-tts"
        assert synthesizer_cls.call_args.kwargs["ws_url"] == (
            "wss://example.com/api-ws/v1/inference"
        )


class TestDashScopeConcurrency:
    @pytest.mark.asyncio
    async def test_multimodal_recognizers_are_isolated(self):
        barrier = asyncio.Barrier(2)
        requests: list[dict[str, Any]] = []

        async def post(url: str, json: dict[str, Any], headers: dict[str, str]):
            requests.append({"url": url, "json": json, "headers": headers})
            await barrier.wait()
            return _flash_response(text=headers["Authorization"])

        recognizers = []
        for index in range(2):
            http_client = MagicMock()
            http_client.post = post
            recognizers.append(
                _DashScopeMultimodalRecognizer(
                    api_key=f"sk-{index}",
                    http_client=http_client,
                    options=SpeechRecognizeOptions(format="wav"),
                )
            )

        async def recognize(index: int) -> list[SpeechEvent]:
            async with recognizers[index]:
                return await _collect(
                    recognizers[index],
                    SpeechAudioInput(
                        data_b64=base64.b64encode(f"audio-{index}".encode()).decode("ascii"),
                        format="wav",
                    ),
                )

        results = await asyncio.gather(recognize(0), recognize(1))

        assert [result[0].text for result in results] == ["Bearer sk-0", "Bearer sk-1"]
        assert [request["url"] for request in requests] == [
            _DEFAULT_GENERATION_URL,
            _DEFAULT_GENERATION_URL,
        ]
        assert [request["headers"]["Authorization"] for request in requests] == [
            "Bearer sk-0",
            "Bearer sk-1",
        ]
        assert [request["json"]["model"] for request in requests] == [
            "qwen3-asr-flash",
            "qwen3-asr-flash",
        ]

    @pytest.mark.asyncio
    async def test_multimodal_http_request_failure_is_wrapped(self):
        http_client = MagicMock()
        http_client.post = AsyncMock(side_effect=RuntimeError("connection refused"))
        recognizer = _DashScopeMultimodalRecognizer(
            api_key="sk-test",
            http_client=http_client,
            options=SpeechRecognizeOptions(format="wav"),
        )

        async with recognizer:
            with pytest.raises(SpeechRequestError, match="connection refused"):
                await _collect(
                    recognizer,
                    SpeechAudioInput(
                        data_b64=base64.b64encode(b"audio").decode("ascii"),
                        format="wav",
                    ),
                )

        http_client.post.assert_awaited_once_with(
            _DEFAULT_GENERATION_URL,
            json={
                "model": "qwen3-asr-flash",
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"audio": "data:audio/wav;base64,YXVkaW8="}],
                        }
                    ]
                },
                "parameters": {"asr_options": {"enable_itn": False}},
            },
            headers={
                "Authorization": "Bearer sk-test",
                "Content-Type": "application/json",
            },
        )

    @pytest.mark.asyncio
    async def test_multimodal_recognizers_use_custom_base_url(self):
        http_client = MagicMock()
        http_client.post = AsyncMock(return_value=_flash_response(text="hi"))
        recognizer = _DashScopeMultimodalRecognizer(
            api_key="sk-test",
            base_url="https://example.com/",
            http_client=http_client,
            options=SpeechRecognizeOptions(format="wav"),
        )

        async with recognizer:
            await _collect(
                recognizer,
                SpeechAudioInput(data_b64="YXVkaW8=", format="wav"),
            )

        assert http_client.post.call_args.args[0] == (
            "https://example.com/api/v1/services/aigc/multimodal-generation/generation"
        )

    @pytest.mark.asyncio
    async def test_realtime_recognizers_are_isolated(self):
        sockets = [
            _FakeDashScopeWs(result_sentences=[_sentence("first", end_time=1)]),
            _FakeDashScopeWs(result_sentences=[_sentence("second", end_time=2)]),
        ]
        connections: list[tuple[str, dict[str, str]]] = []

        async def connect(url: str, headers: dict[str, str]) -> _FakeDashScopeWs:
            connection = (url, headers)
            connections.append(connection)
            return sockets[len(connections) - 1]

        recognizers = [
            _DashScopeRealtimeRecognizer(
                api_key=f"sk-{index}",
                connect=connect,
                options=SpeechRecognizeOptions(format="pcm"),
            )
            for index in range(2)
        ]

        async def recognize(index: int) -> list[SpeechEvent]:
            async with recognizers[index]:

                async def chunks():
                    yield f"audio-{index}".encode()

                return await asyncio.wait_for(_collect(recognizers[index], chunks()), timeout=2)

        results = await asyncio.gather(recognize(0), recognize(1))

        assert [result[0].text for result in results] == ["first", "second"]
        assert [headers["Authorization"] for _, headers in connections] == [
            "Bearer sk-0",
            "Bearer sk-1",
        ]
        assert sockets[0].audio_frames == [b"audio-0"]
        assert sockets[1].audio_frames == [b"audio-1"]
        assert all(socket.closed for socket in sockets)

    @pytest.mark.asyncio
    async def test_tts_synthesizers_are_isolated(self):
        fake_wss = [
            _FakeDashScopeTtsWs([b"first-0", b"first-1"]),
            _FakeDashScopeTtsWs([b"second-0", b"second-1"]),
        ]
        connections: list[dict[str, str]] = []

        def build_synthesizer(api_key: str, text: str, index: int) -> _DashScopeTTSSynthesizer:
            fake_ws = fake_wss[index]

            async def connect(url: str, headers: dict[str, str]) -> _FakeDashScopeTtsWs:
                connections.append(headers)
                return fake_ws

            return _DashScopeTTSSynthesizer(
                api_key=api_key,
                options=SpeechSynthesisOptions(voice="longanhuan_v3.1"),
                connect=connect,
            )

        async def synthesize(api_key: str, text: str, index: int) -> list[bytes]:
            synthesizer = build_synthesizer(api_key, text, index)
            async with synthesizer:
                return await asyncio.wait_for(_collect_bytes(synthesizer, text), timeout=2)

        results = await asyncio.gather(
            synthesize("sk-0", "first", 0),
            synthesize("sk-1", "second", 1),
        )

        assert results == [
            [b"first-0", b"first-1"],
            [b"second-0", b"second-1"],
        ]
        assert [headers["Authorization"] for headers in connections] == [
            "Bearer sk-0",
            "Bearer sk-1",
        ]
        assert fake_wss[0].texts == ["first"]
        assert fake_wss[1].texts == ["second"]
        assert all(fake_ws.finish_sent for fake_ws in fake_wss)
        assert all(fake_ws.closed for fake_ws in fake_wss)
