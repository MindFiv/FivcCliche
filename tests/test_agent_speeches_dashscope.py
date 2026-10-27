"""Unit tests for DashScope Qwen3-ASR adapters (HTTP mocked, WS injected)."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fivcplayground.speeches import (
    SpeechError,
    SpeechRecognizeEvent,
    SpeechRecognizeOption,
    SpeechSynthesisOption,
)

from fivccliche.services.implements.agent_speeches.dashscope import (
    _DEFAULT_GENERATION_URL,
    _QWEN_AUDIO_TTS_MODELS,
    _QWEN_AUDIO_TTS_PATH,
    _DashScopeMultimodalRecognizer,
    _DashScopeQwenAudioTTSSynthesizer,
    _qwen_audio_tts_request,
    _qwen_audio_tts_url,
    _qwen_audio_tts_voice,
    _TTS_DEFAULT_MODEL,
    DashScopeSpeechProvider,
)
from fivccliche.services.implements.agent_speeches.dashscope_realtime import (
    DashScopeRealtimeSpeechProvider,
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


async def _collect(recognizer, audio) -> list[SpeechRecognizeEvent]:
    return [event async for event in recognizer.stream_async(audio)]


async def _collect_bytes(synthesizer, text) -> list[bytes]:
    return [chunk async for chunk in synthesizer.stream_async(text)]


class TestDashScopeMultimodalRecognizer:
    @pytest.mark.asyncio
    async def test_recognize_posts_bytes_as_data_uri(self):
        http_client = MagicMock()
        http_client.post = AsyncMock(return_value=_flash_response())
        recognizer = _DashScopeMultimodalRecognizer(
            api_key="sk-test",
            model="qwen3-asr-flash",
            http_client=http_client,
            options=SpeechRecognizeOption(
                language="zh", enable_itn=True, context="fivc", format="mp3"
            ),
        )
        async with recognizer:
            events = await _collect(recognizer, b"clip-bytes")

        http_client.post.assert_awaited_once()
        args, kwargs = http_client.post.call_args
        assert args[0] == _DEFAULT_GENERATION_URL
        assert kwargs["headers"]["Authorization"] == "Bearer sk-test"
        assert kwargs["headers"]["Content-Type"] == "application/json"
        payload = kwargs["json"]
        assert payload["model"] == "qwen3-asr-flash"
        messages = payload["input"]["messages"]
        assert messages[0] == {"role": "system", "content": [{"text": "fivc"}]}
        assert messages[1]["content"][0]["audio"] == (
            "data:audio/mpeg;base64," + base64.b64encode(b"clip-bytes").decode("ascii")
        )
        assert payload["parameters"]["asr_options"]["language"] == "zh"
        assert payload["parameters"]["asr_options"]["enable_itn"] is True
        assert events == [
            SpeechRecognizeEvent(type="final", text="欢迎使用阿里云。", language="zh"),
        ]

    @pytest.mark.asyncio
    async def test_recognize_uses_option_format_for_data_uri(self):
        http_client = MagicMock()
        http_client.post = AsyncMock(return_value=_flash_response(text="hi"))
        recognizer = _DashScopeMultimodalRecognizer(
            api_key="sk-test",
            http_client=http_client,
            options=SpeechRecognizeOption(format="wav"),
        )
        async with recognizer:
            await _collect(recognizer, b"abc123")

        audio = http_client.post.call_args.kwargs["json"]["input"]["messages"][0]["content"][0][
            "audio"
        ]
        assert audio == "data:audio/wav;base64," + base64.b64encode(b"abc123").decode("ascii")

    @pytest.mark.asyncio
    async def test_custom_model_uses_qwen3_payload(self):
        http_client = MagicMock()
        http_client.post = AsyncMock(return_value=_flash_response(text="hi"))
        recognizer = _DashScopeMultimodalRecognizer(
            api_key="sk-test",
            model="custom-flash",
            http_client=http_client,
            options=SpeechRecognizeOption(language="zh", enable_itn=True, format="wav"),
        )

        async with recognizer:
            events = await _collect(recognizer, b"wav-bytes")

        kwargs = http_client.post.call_args.kwargs
        assert kwargs["headers"]["Authorization"] == "Bearer sk-test"
        assert "X-DashScope-SSE" not in kwargs["headers"]
        assert kwargs["json"]["input"]["messages"][-1]["content"] == [
            {"audio": "data:audio/wav;base64," + base64.b64encode(b"wav-bytes").decode("ascii")}
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
            options=SpeechRecognizeOption(format="pcm"),
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
            with pytest.raises(SpeechError, match="401"):
                await _collect(recognizer, b"clip")

    @pytest.mark.asyncio
    async def test_recognize_requires_http_client(self):
        recognizer = _DashScopeMultimodalRecognizer(api_key="sk-test")
        with pytest.raises(SpeechError, match="not started"):
            await _collect(recognizer, b"abc")

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
            options=SpeechRecognizeOption(
                language="zh", enable_itn=True, context="fivc", format="mp3"
            ),
        )

        async with recognizer:
            events = await _collect(recognizer, b"mp3-bytes")

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
                                    "data": "data:audio/mpeg;base64,"
                                    + base64.b64encode(b"mp3-bytes").decode("ascii"),
                                },
                            }
                        ],
                    }
                ]
            },
            "parameters": {"format": "mp3", "sample_rate": 16000},
        }
        assert events == [
            SpeechRecognizeEvent(type="final", text="欢迎使用阿里云。", language=None)
        ]

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
            options=SpeechRecognizeOption(format="wav"),
        )

        async with recognizer:
            events = await _collect(recognizer, b"abc123")

        payload = http_client.post.call_args.kwargs["json"]
        content = payload["input"]["messages"][0]["content"][0]
        assert content["input_audio"]["data"] == (
            "data:audio/wav;base64," + base64.b64encode(b"abc123").decode("ascii")
        )
        assert payload["parameters"] == {"format": "wav", "sample_rate": 16000}
        assert events == [SpeechRecognizeEvent(type="final", text=expected_text, language=None)]

    @pytest.mark.asyncio
    async def test_recognize_streamed_pcm_uses_recognizer_sample_rate(self):
        http_client = MagicMock()
        http_client.post = AsyncMock(return_value=_qwen_audio_response({"text": "audio"}))
        recognizer = _DashScopeMultimodalRecognizer(
            api_key="sk-test",
            model="qwen-audio-3.0-asr-flash",
            http_client=http_client,
            options=SpeechRecognizeOption(format="pcm", sample_rate=24000),
        )

        async def audio() -> AsyncIterator[bytes]:
            yield b"pcm-audio"

        async with recognizer:
            events = await _collect(recognizer, audio())

        payload = http_client.post.call_args.kwargs["json"]
        assert payload["parameters"] == {"format": "pcm", "sample_rate": 24000}
        assert events == [SpeechRecognizeEvent(type="final", text="audio", language=None)]


def _json_response(payload: Any | None = None, *, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.is_success = status_code < 400
    response.status_code = status_code
    response.text = "vendor-error"
    if payload is None:
        response.json.side_effect = ValueError("invalid json")
    else:
        response.json.return_value = payload
    return response


class TestDashScopeQwenAudioTTS:
    @pytest.mark.parametrize(
        ("base_url", "expected_url"),
        [
            (
                "https://dashscope.aliyuncs.com",
                "https://dashscope.aliyuncs.com" + _QWEN_AUDIO_TTS_PATH,
            ),
            (
                "https://dashscope.aliyuncs.com" + _QWEN_AUDIO_TTS_PATH,
                "https://dashscope.aliyuncs.com" + _QWEN_AUDIO_TTS_PATH,
            ),
            (
                "wss://llm-workspace.cn-beijing.maas.aliyuncs.com/",
                "https://llm-workspace.cn-beijing.maas.aliyuncs.com" + _QWEN_AUDIO_TTS_PATH,
            ),
        ],
    )
    def test_normalizes_http_tts_url(self, base_url: str, expected_url: str):
        assert _qwen_audio_tts_url(base_url) == expected_url

    @pytest.mark.parametrize(
        ("model", "expected_voice"),
        [
            ("qwen-audio-3.1-tts-flash", "longanhuan_v3.1"),
            ("qwen-audio-3.0-tts-flash", "longanhuan_v3.6"),
            ("qwen-audio-3.0-tts-plus", "longanhuan_v3.6"),
        ],
    )
    def test_selects_model_family_voice(self, model: str, expected_voice: str):
        assert _qwen_audio_tts_voice(model, "") == expected_voice
        assert _qwen_audio_tts_voice(model, "custom-voice") == "custom-voice"

    def test_builds_qwen_audio_tts_request(self):
        options = SpeechSynthesisOption(
            voice="custom-voice",
            format="WAV",
            sample_rate=24000,
            volume=70,
            speech_rate=1.2,
            pitch_rate=0.9,
            extra={"seed": 7},
        )
        assert _qwen_audio_tts_request("qwen-audio-3.0-tts-flash", "你好", options) == {
            "model": "qwen-audio-3.0-tts-flash",
            "input": {
                "seed": 7,
                "text": "你好",
                "voice": "custom-voice",
                "format": "wav",
                "sample_rate": 24000,
                "volume": 70,
                "rate": 1.2,
                "pitch": 0.9,
            },
        }

    def test_rejects_models_outside_qwen_audio_http_family(self):
        for model in {
            "qwen-tts-realtime",
            "cosyvoice-v2",
            "qwen3-tts-flash-realtime",
        } - _QWEN_AUDIO_TTS_MODELS:
            with pytest.raises(SpeechError, match="not supported"):
                _DashScopeQwenAudioTTSSynthesizer(api_key="sk-test", model=model)

    @pytest.mark.asyncio
    async def test_synthesizes_and_downloads_audio_in_chunks(self):
        audio = b"a" * 12800 + b"end"
        http_client = MagicMock()
        http_client.post = AsyncMock(
            return_value=_json_response({"output": {"audio": {"url": "https://audio/a.wav"}}})
        )
        audio_response = MagicMock()
        audio_response.is_success = True
        audio_response.content = audio
        http_client.get = AsyncMock(return_value=audio_response)
        synthesizer = _DashScopeQwenAudioTTSSynthesizer(
            api_key="sk-test",
            model="qwen-audio-3.0-tts-flash",
            base_url=_qwen_audio_tts_url("https://example.com"),
            http_client=http_client,
            options=SpeechSynthesisOption(voice="", format="wav", sample_rate=24000),
        )

        async with synthesizer:
            chunks = await _collect_bytes(synthesizer, "你好")

        assert chunks == [b"a" * 12800, b"end"]
        args, kwargs = http_client.post.call_args
        assert args[0] == "https://example.com" + _QWEN_AUDIO_TTS_PATH
        assert kwargs["headers"]["Authorization"] == "Bearer sk-test"
        assert kwargs["json"]["input"]["voice"] == "longanhuan_v3.6"
        http_client.get.assert_awaited_once_with("https://audio/a.wav")

    @pytest.mark.asyncio
    async def test_synthesizes_streamed_text(self):
        http_client = MagicMock()
        http_client.post = AsyncMock(
            return_value=_json_response({"output": {"audio": {"url": "https://audio/a.wav"}}})
        )
        audio_response = MagicMock()
        audio_response.is_success = True
        audio_response.content = b"audio"
        http_client.get = AsyncMock(return_value=audio_response)
        synthesizer = _DashScopeQwenAudioTTSSynthesizer(
            api_key="sk-test",
            http_client=http_client,
            options=SpeechSynthesisOption(voice="voice"),
        )

        async def text():
            yield "你好"
            yield "世界"

        async with synthesizer:
            chunks = await _collect_bytes(synthesizer, text())

        assert chunks == [b"audio"]
        assert http_client.post.call_args.kwargs["json"]["input"]["text"] == "你好世界"

    @pytest.mark.asyncio
    async def test_wraps_http_json_and_download_failures(self):
        audio_response = MagicMock()
        audio_response.is_success = False
        audio_response.status_code = 403

        cases = [
            ("request failed", RuntimeError("connection refused"), None),
            ("TTS HTTP 401", _json_response(status_code=401), None),
            ("invalid JSON", _json_response(), None),
            ("missing audio URL", _json_response({"output": {}}), None),
            (
                "Failed to download DashScope TTS audio: dns",
                _json_response({"output": {"audio": {"url": "https://audio/a.wav"}}}),
                RuntimeError("dns"),
            ),
            (
                "download DashScope TTS audio HTTP 403",
                _json_response({"output": {"audio": {"url": "https://audio/a.wav"}}}),
                audio_response,
            ),
        ]
        for message, post_response, download_response in cases:
            http_client = MagicMock()
            http_client.post = (
                AsyncMock(side_effect=post_response)
                if isinstance(post_response, Exception)
                else AsyncMock(return_value=post_response)
            )
            if download_response is not None:
                http_client.get = (
                    AsyncMock(side_effect=download_response)
                    if isinstance(download_response, Exception)
                    else AsyncMock(return_value=download_response)
                )
            synthesizer = _DashScopeQwenAudioTTSSynthesizer(
                api_key="sk-test",
                http_client=http_client,
                options=SpeechSynthesisOption(voice="voice"),
            )
            async with synthesizer:
                with pytest.raises(SpeechError, match=message.replace("?", "\\?")):
                    await _collect_bytes(synthesizer, "hello")

    @pytest.mark.asyncio
    async def test_owned_http_client_closes_on_exit(self):
        synthesizer = _DashScopeQwenAudioTTSSynthesizer(
            api_key="sk-test",
            options=SpeechSynthesisOption(voice="voice"),
        )
        async with synthesizer:
            assert synthesizer._http_client is not None
            owned_client = synthesizer._http_client
        assert owned_client.is_closed


class TestDashScopeSpeechProvider:
    @pytest.mark.asyncio
    async def test_get_recognizer_ignores_global_speech_config(self):
        component_site = MagicMock()
        component_site.query_component.return_value = MagicMock()
        with patch(
            "fivccliche.services.implements.agent_speeches.dashscope._DashScopeMultimodalRecognizer",
        ) as clip_cls:
            provider = DashScopeSpeechProvider(component_site)
            await provider.get_recognizer()

        component_site.query_component.assert_not_called()
        assert clip_cls.call_args.kwargs["api_key"] == ""
        assert clip_cls.call_args.kwargs["model"] == "qwen3-asr-flash"
        assert clip_cls.call_args.kwargs["base_url"] == "https://dashscope.aliyuncs.com"

    @pytest.mark.asyncio
    async def test_get_synthesizer_ignores_global_speech_config(self):
        component_site = MagicMock()
        component_site.query_component.return_value = MagicMock()
        options = SpeechSynthesisOption(voice="", format="wav", sample_rate=24000)
        with patch(
            "fivccliche.services.implements.agent_speeches.dashscope._DashScopeQwenAudioTTSSynthesizer",
        ) as synthesizer_cls:
            provider = DashScopeSpeechProvider(component_site)
            await provider.get_synthesizer(options)

        component_site.query_component.assert_not_called()
        assert synthesizer_cls.call_args.kwargs["api_key"] == ""
        assert synthesizer_cls.call_args.kwargs["model"] == _TTS_DEFAULT_MODEL
        assert synthesizer_cls.call_args.kwargs["base_url"] == (
            "https://dashscope.aliyuncs.com" + _QWEN_AUDIO_TTS_PATH
        )

    @pytest.mark.asyncio
    async def test_get_recognizer_builds_flash(self):
        options = SpeechRecognizeOption(language="zh")
        with patch(
            "fivccliche.services.implements.agent_speeches.dashscope._DashScopeMultimodalRecognizer",
        ) as clip_cls:
            provider = DashScopeSpeechProvider(MagicMock())
            recognizer = await provider.get_recognizer(
                options,
                api_key="sk-x",
                model="qwen3-asr-flash",
            )

        clip_cls.assert_called_once()
        assert recognizer is clip_cls.return_value
        assert clip_cls.call_args.kwargs["api_key"] == "sk-x"
        assert clip_cls.call_args.kwargs["model"] == "qwen3-asr-flash"
        assert clip_cls.call_args.kwargs["options"] is options

    @pytest.mark.asyncio
    async def test_get_recognizer_overrides_credentials(self):
        options = SpeechRecognizeOption(language="zh")
        with patch(
            "fivccliche.services.implements.agent_speeches.dashscope._DashScopeMultimodalRecognizer",
        ) as clip_cls:
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
        with patch(
            "fivccliche.services.implements.agent_speeches.dashscope._DashScopeMultimodalRecognizer",
        ) as clip_cls:
            provider = DashScopeSpeechProvider(MagicMock())
            await provider.get_recognizer(api_key="")

        assert clip_cls.call_args.kwargs["api_key"] == ""

    @pytest.mark.asyncio
    async def test_get_synthesizer_uses_tts_defaults_and_overrides(self):
        options = SpeechSynthesisOption(voice="", format="wav", sample_rate=24000)
        with patch(
            "fivccliche.services.implements.agent_speeches.dashscope._DashScopeQwenAudioTTSSynthesizer",
        ) as synthesizer_cls:
            provider = DashScopeSpeechProvider(MagicMock())
            await provider.get_synthesizer(options)
            await provider.get_synthesizer(
                options,
                api_key="sk-override",
                model="qwen-audio-3.1-tts-flash",
                base_url="wss://example.com/",
            )

        default_call, override_call = synthesizer_cls.call_args_list

        assert default_call.kwargs["api_key"] == ""
        assert default_call.kwargs["model"] == _TTS_DEFAULT_MODEL
        assert default_call.kwargs["base_url"] == (
            "https://dashscope.aliyuncs.com" + _QWEN_AUDIO_TTS_PATH
        )
        assert default_call.kwargs["options"] is options

        assert override_call.kwargs["api_key"] == "sk-override"
        assert override_call.kwargs["model"] == "qwen-audio-3.1-tts-flash"
        assert override_call.kwargs["base_url"] == ("https://example.com" + _QWEN_AUDIO_TTS_PATH)


class _ScriptedRecognition:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.frames: list[bytes] = []
        self.callback = kwargs["callback"]

    def start(self) -> None:
        self.callback.on_open()

    def send_audio_frame(self, frame: bytes) -> None:
        self.frames.append(frame)
        if frame == b"final-frame":
            result = MagicMock()
            result.get_sentence.return_value = {"text": "你好", "end_time": 10}
            self.callback.on_event(result)

    def stop(self) -> None:
        self.callback.on_complete()
        self.callback.on_close()


class _ScriptedSynthesizer:
    last: _ScriptedSynthesizer | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _ScriptedSynthesizer.last = self

    def streaming_call(self, content: str) -> None:
        self.content = content
        self.kwargs["callback"].on_data(b"pcm-a")

    def streaming_complete(self) -> None:
        self.kwargs["callback"].on_complete()


class TestDashScopeRealtimeSDK:
    def test_websocket_url_from_http_origin(self):
        from fivccliche.services.implements.agent_speeches.dashscope_realtime import (
            _websocket_url,
        )

        assert (
            _websocket_url("https://dashscope.aliyuncs.com")
            == "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
        )
        assert (
            _websocket_url("wss://ws.example.com/api-ws/v1/inference")
            == "wss://ws.example.com/api-ws/v1/inference"
        )

    @pytest.mark.asyncio
    async def test_missing_api_key(self):
        provider = DashScopeRealtimeSpeechProvider(MagicMock())
        with pytest.raises(SpeechError, match="empty"):
            await provider.get_recognizer()
        with pytest.raises(SpeechError, match="empty"):
            await provider.get_synthesizer()

    @pytest.mark.asyncio
    async def test_stream_async_bytes_uses_recognition(self):
        from fivccliche.services.implements.agent_speeches.dashscope_realtime import (
            _DashScopeRealtimeRecognizer,
        )

        created: list[_ScriptedRecognition] = []

        def factory(**kwargs: Any) -> _ScriptedRecognition:
            recognition = _ScriptedRecognition(**kwargs)
            created.append(recognition)
            return recognition

        recognizer = _DashScopeRealtimeRecognizer(
            api_key="sk-test",
            ws_url="wss://dashscope.aliyuncs.com/api-ws/v1/inference",
            speech_id="asr-1",
        )
        with patch(
            "fivccliche.services.implements.agent_speeches.dashscope_realtime.Recognition",
            side_effect=factory,
        ):
            async with recognizer:
                await recognizer.begin_utterance()
                await recognizer.push_audio(b"final-frame")
                events = [event async for event in recognizer.end_utterance()]

        assert recognizer.id == "asr-1"
        assert created[0].kwargs["api_key"] == "sk-test"
        assert created[0].kwargs["base_address"].endswith("/api-ws/v1/inference")
        assert created[0].kwargs["model"] == "qwen-audio-3.1-asr-flash-streaming"
        assert created[0].frames == [b"final-frame"]
        assert events[-1].type == "final"
        assert events[-1].text == "你好"

    @pytest.mark.asyncio
    async def test_stream_async_bytes_clip(self):
        from fivccliche.services.implements.agent_speeches.dashscope_realtime import (
            _DashScopeRealtimeRecognizer,
        )

        created: list[_ScriptedRecognition] = []

        def factory(**kwargs: Any) -> _ScriptedRecognition:
            recognition = _ScriptedRecognition(**kwargs)
            created.append(recognition)
            return recognition

        clip = b"\x00\x01" * 80
        recognizer = _DashScopeRealtimeRecognizer(
            api_key="sk-test",
            ws_url="wss://example.invalid/api-ws/v1/inference",
        )
        with patch(
            "fivccliche.services.implements.agent_speeches.dashscope_realtime.Recognition",
            side_effect=factory,
        ):
            async with recognizer:
                events = [event async for event in recognizer.stream_async(clip)]
        assert created[0].frames
        assert events[-1].type == "final"

    @pytest.mark.asyncio
    async def test_synthesizer_uses_option_and_resolved_url(self):
        import dashscope

        from fivccliche.services.implements.agent_speeches.dashscope_realtime import (
            _DashScopeRealtimeSynthesizer,
        )

        previous = dashscope.api_key
        option = SpeechSynthesisOption(
            voice="",
            format="pcm",
            sample_rate=24000,
            volume=70,
            speech_rate=1.2,
            pitch_rate=0.8,
        )
        synthesizer = _DashScopeRealtimeSynthesizer(
            api_key="sk-tts",
            ws_url="wss://ws.example.com/api-ws/v1/inference",
            options=option,
            speech_id="tts-1",
        )
        with patch(
            "fivccliche.services.implements.agent_speeches.dashscope_realtime.DashScopeTTSV2Client",
            _ScriptedSynthesizer,
        ):
            async with synthesizer:
                audio = [chunk async for chunk in synthesizer.stream_async("你好")]
        assert synthesizer.id == "tts-1"
        assert synthesizer.get_option().voice == "longanhuan_v3.1"
        assert audio == [b"pcm-a"]
        client = _ScriptedSynthesizer.last
        assert client is not None
        assert client.kwargs["url"] == "wss://ws.example.com/api-ws/v1/inference"
        assert client.kwargs["voice"] == "longanhuan_v3.1"
        assert client.kwargs["volume"] == 70
        assert client.kwargs["speech_rate"] == 1.2
        assert client.kwargs["pitch_rate"] == 0.8
        assert client.content == "你好"
        assert dashscope.api_key == previous

    @pytest.mark.asyncio
    async def test_provider_defaults(self):
        from fivccliche.services.implements.agent_speeches.dashscope_realtime import (
            _DashScopeRealtimeRecognizer,
            _DashScopeRealtimeSynthesizer,
        )

        provider = DashScopeRealtimeSpeechProvider(MagicMock())
        with (
            patch(
                "fivccliche.services.implements.agent_speeches.dashscope_realtime._DashScopeRealtimeRecognizer",
                wraps=_DashScopeRealtimeRecognizer,
            ) as recognizer_cls,
            patch(
                "fivccliche.services.implements.agent_speeches.dashscope_realtime._DashScopeRealtimeSynthesizer",
                wraps=_DashScopeRealtimeSynthesizer,
            ) as synthesizer_cls,
        ):
            await provider.get_recognizer(api_key="sk", id="asr")
            await provider.get_synthesizer(api_key="sk", id="tts")

        assert recognizer_cls.call_args.kwargs["model"] == "qwen-audio-3.1-asr-flash-streaming"
        assert recognizer_cls.call_args.kwargs["ws_url"] == (
            "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
        )
        assert recognizer_cls.call_args.kwargs["speech_id"] == "asr"
        assert synthesizer_cls.call_args.kwargs["model"] == "qwen-audio-3.1-tts-flash"
        assert synthesizer_cls.call_args.kwargs["speech_id"] == "tts"
