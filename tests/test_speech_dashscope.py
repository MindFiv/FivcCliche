"""Unit tests for DashScope Qwen3-ASR adapters (HTTP mocked, WS injected)."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fivccliche.services.implements.speech.dashscope_multimodal import (
    DEFAULT_GENERATION_URL,
    DashScopeMultimodalRecognizer,
    DashScopeMultimodalSpeechProvider,
)
from fivccliche.services.implements.speech.dashscope_realtime import (
    DashScopeRealtimeRecognizer,
    DashScopeRealtimeSpeechProvider,
)
from fivccliche.services.interfaces.speech import (
    SpeechAudioInput,
    SpeechEvent,
    SpeechRecognizeOptions,
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


class TestDashScopeMultimodalRecognizer:
    @pytest.mark.asyncio
    async def test_recognize_posts_url_and_maps_transcript(self):
        http_client = MagicMock()
        http_client.post = AsyncMock(return_value=_flash_response())
        recognizer = DashScopeMultimodalRecognizer(
            api_key="sk-test",
            model="qwen3-asr-flash",
            http_client=http_client,
            options=SpeechRecognizeOptions(language="zh", enable_itn=True, context="fivc"),
        )
        events = await _collect(
            recognizer,
            SpeechAudioInput(url="https://example.com/a.mp3", format="mp3"),
        )

        http_client.post.assert_awaited_once()
        args, kwargs = http_client.post.call_args
        assert args[0] == DEFAULT_GENERATION_URL
        assert kwargs["headers"]["Authorization"] == "Bearer sk-test"
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
        recognizer = DashScopeMultimodalRecognizer(api_key="sk-test", http_client=http_client)
        await _collect(recognizer, SpeechAudioInput(data_b64="abc123", format="wav"))

        audio = http_client.post.call_args.kwargs["json"]["input"]["messages"][0]["content"][0][
            "audio"
        ]
        assert audio == "data:audio/wav;base64,abc123"

    @pytest.mark.asyncio
    async def test_custom_model_uses_qwen3_payload(self):
        http_client = MagicMock()
        http_client.post = AsyncMock(return_value=_flash_response(text="hi"))
        recognizer = DashScopeMultimodalRecognizer(
            api_key="sk-test",
            model="custom-flash",
            http_client=http_client,
            options=SpeechRecognizeOptions(language="zh", enable_itn=True),
        )

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
        recognizer = DashScopeMultimodalRecognizer(
            api_key="sk-test",
            http_client=http_client,
            options=SpeechRecognizeOptions(format="pcm"),
        )

        async def chunks():
            yield b"abc"
            yield b"123"

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
        recognizer = DashScopeMultimodalRecognizer(api_key="bad", http_client=http_client)
        with pytest.raises(SpeechRequestError, match="401"):
            await _collect(recognizer, SpeechAudioInput(url="https://example.com/a.wav"))


class TestDashScopeQwenAudioRecognizer:
    @pytest.mark.asyncio
    async def test_recognize_posts_qwen_audio_payload(self):
        http_client = MagicMock()
        http_client.post = AsyncMock(
            return_value=_qwen_audio_response({"text": "欢迎使用阿里云。"})
        )
        recognizer = DashScopeMultimodalRecognizer(
            api_key="sk-test",
            model="qwen-audio-3.0-asr-flash",
            http_client=http_client,
            options=SpeechRecognizeOptions(language="zh", enable_itn=True, context="fivc"),
        )

        events = await _collect(
            recognizer,
            SpeechAudioInput(url="https://example.com/a.mp3", format="mp3"),
        )

        http_client.post.assert_awaited_once()
        args, kwargs = http_client.post.call_args
        assert args[0] == DEFAULT_GENERATION_URL
        assert kwargs["headers"]["Authorization"] == "Bearer sk-test"
        assert kwargs["headers"]["X-DashScope-SSE"] == "disable"
        payload = kwargs["json"]
        assert payload == {
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
        recognizer = DashScopeMultimodalRecognizer(
            api_key="sk-test",
            model="qwen-audio-3.0-asr-flash",
            http_client=http_client,
        )

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
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self._incoming: asyncio.Queue[str] = asyncio.Queue()
        self.closed = False

    async def send(self, data: str) -> None:
        message = json.loads(data)
        self.sent.append(message)
        if message.get("type") == "session.update":
            await self._incoming.put(json.dumps({"type": "session.updated"}))
        elif message.get("type") == "input_audio_buffer.commit":
            await self._incoming.put(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.text",
                        "text": "北",
                        "stash": "京",
                        "language": "zh",
                        "emotion": "neutral",
                    }
                )
            )
            await self._incoming.put(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "transcript": "北京",
                        "language": "zh",
                    }
                )
            )

    async def recv(self) -> str:
        return await self._incoming.get()

    async def close(self) -> None:
        self.closed = True


class TestDashScopeRealtimeRecognizer:
    @pytest.mark.asyncio
    async def test_aenter_opens_session_then_recognize_maps_events(self):
        fake_ws = _FakeDashScopeWs()

        async def connect(url: str, headers: dict[str, str]) -> _FakeDashScopeWs:
            connect.last_url = url  # type: ignore[attr-defined]
            connect.last_headers = headers  # type: ignore[attr-defined]
            return fake_ws

        recognizer = DashScopeRealtimeRecognizer(
            api_key="sk-test",
            model="qwen3-asr-flash-realtime",
            connect=connect,
            options=SpeechRecognizeOptions(format="pcm"),
        )

        async def chunks():
            yield b"pcm-bytes"

        async with recognizer:
            update = next(msg for msg in fake_ws.sent if msg.get("type") == "session.update")
            assert update["session"]["turn_detection"] is None
            assert connect.last_headers["Authorization"] == "Bearer sk-test"  # type: ignore[attr-defined]
            assert "qwen3-asr-flash-realtime" in connect.last_url  # type: ignore[attr-defined]
            events = await asyncio.wait_for(_collect(recognizer, chunks()), timeout=2)

        append = next(msg for msg in fake_ws.sent if msg.get("type") == "input_audio_buffer.append")
        assert append["audio"] == base64.b64encode(b"pcm-bytes").decode()
        assert [event.type for event in events] == ["partial", "final"]
        assert events[0].text == "北京"
        assert events[1].text == "北京"
        assert events[1].language == "zh"
        assert fake_ws.closed is True

    @pytest.mark.asyncio
    async def test_clip_base64_is_sent_as_bytes(self):
        fake_ws = _FakeDashScopeWs()

        async def connect(url: str, headers: dict[str, str]) -> _FakeDashScopeWs:
            return fake_ws

        recognizer = DashScopeRealtimeRecognizer(
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
        append = next(msg for msg in fake_ws.sent if msg.get("type") == "input_audio_buffer.append")
        assert append["audio"] == payload
        assert events[-1].type == "final"

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
        recognizer = DashScopeRealtimeRecognizer(
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
        append = next(msg for msg in fake_ws.sent if msg.get("type") == "input_audio_buffer.append")
        assert append["audio"] == base64.b64encode(b"from-url").decode()

    @pytest.mark.asyncio
    async def test_recognize_before_aenter_raises(self):
        recognizer = DashScopeRealtimeRecognizer(api_key="sk-test", connect=AsyncMock())
        with pytest.raises(SpeechRequestError, match="not started"):
            await _collect(recognizer, SpeechAudioInput(data_b64="abc"))


def _speech_config(values: dict) -> MagicMock:
    session = MagicMock()
    session.get_value.side_effect = lambda key: values.get(key)
    config = MagicMock()
    config.get_session.return_value = session
    return config


class TestDashScopeMultimodalSpeechProvider:
    @pytest.mark.asyncio
    async def test_get_recognizer_builds_flash(self):
        options = SpeechRecognizeOptions(language="zh")
        with (
            patch(
                "fivccliche.services.implements.speech.dashscope_multimodal.query_component",
                return_value=_speech_config(
                    {"ASR_API_KEY": "sk-x", "ASR_MODEL": "qwen3-asr-flash"}
                ),
            ),
            patch(
                "fivccliche.services.implements.speech.dashscope_multimodal.DashScopeMultimodalRecognizer",
            ) as clip_cls,
        ):
            provider = DashScopeMultimodalSpeechProvider(MagicMock())
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
                "fivccliche.services.implements.speech.dashscope_multimodal.query_component",
                return_value=_speech_config(
                    {"ASR_API_KEY": "sk-x", "ASR_MODEL": "qwen3-asr-flash"}
                ),
            ),
            patch(
                "fivccliche.services.implements.speech.dashscope_multimodal.DashScopeMultimodalRecognizer",
            ) as clip_cls,
        ):
            provider = DashScopeMultimodalSpeechProvider(MagicMock())
            await provider.get_recognizer(
                options,
                api_key="sk-override",
                model="custom-flash",
                base_url="https://example.com/",
            )

        assert clip_cls.call_args.kwargs["api_key"] == "sk-override"
        assert clip_cls.call_args.kwargs["model"] == "custom-flash"
        assert clip_cls.call_args.kwargs["generation_url"] == (
            "https://example.com/api/v1/services/aigc/multimodal-generation/generation"
        )

    @pytest.mark.asyncio
    async def test_get_recognizer_empty_api_key_is_explicit(self):
        with (
            patch(
                "fivccliche.services.implements.speech.dashscope_multimodal.query_component",
                return_value=_speech_config({"ASR_API_KEY": "sk-x"}),
            ),
            patch(
                "fivccliche.services.implements.speech.dashscope_multimodal.DashScopeMultimodalRecognizer",
            ) as clip_cls,
        ):
            provider = DashScopeMultimodalSpeechProvider(MagicMock())
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
                    {"ASR_API_KEY": "sk-x", "ASR_MODEL": "qwen3-asr-flash-realtime"}
                ),
            ),
            patch(
                "fivccliche.services.implements.speech.dashscope_realtime.DashScopeRealtimeRecognizer",
                return_value=stream,
            ) as stream_cls,
        ):
            provider = DashScopeRealtimeSpeechProvider(MagicMock())
            recognizer = await provider.get_recognizer(options)

        stream_cls.assert_called_once()
        assert recognizer is stream
        stream.start_async.assert_not_called()
        stream.__aenter__.assert_not_called()
        assert stream_cls.call_args.kwargs["model"] == "qwen3-asr-flash-realtime"
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
                    {"ASR_API_KEY": "sk-x", "ASR_MODEL": "qwen3-asr-flash-realtime"}
                ),
            ),
            patch(
                "fivccliche.services.implements.speech.dashscope_realtime.DashScopeRealtimeRecognizer",
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
        assert stream_cls.call_args.kwargs["ws_url"] == (
            "wss://example.com/api-ws/v1/realtime?model=custom-rt"
        )
        assert stream_cls.call_args.kwargs["options"] is options
