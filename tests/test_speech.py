"""Unit tests for the speech recognition contract and Fake provider."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from fivccliche.services.implements.speech.fake import FakeSpeechProvider, FakeSpeechRecognizer
from fivccliche.services.interfaces.speech import (
    ISpeechProvider,
    SpeechAudioInput,
    SpeechEvent,
    SpeechRecognizeOptions,
)


class TestSpeechAudioInput:
    def test_requires_exactly_one_source(self):
        with pytest.raises(ValidationError):
            SpeechAudioInput()
        with pytest.raises(ValidationError):
            SpeechAudioInput(url="https://example.com/a.wav", data_b64="abc")

    def test_accepts_url_or_base64(self):
        by_url = SpeechAudioInput(url="https://example.com/a.wav")
        by_b64 = SpeechAudioInput(data_b64="abc", format="mp3")
        assert by_url.url == "https://example.com/a.wav"
        assert by_b64.data_b64 == "abc"
        assert by_b64.format == "mp3"


class TestFakeSpeechProvider:
    @pytest.mark.asyncio
    async def test_clip_yields_configured_final(self):
        provider = FakeSpeechProvider(transcript="recognized clip")
        async with await provider.get_recognizer(
            SpeechRecognizeOptions(language="zh", context="project fivc")
        ) as recognizer:
            assert isinstance(recognizer, FakeSpeechRecognizer)
            events = [
                event
                async for event in recognizer.stream_async(
                    SpeechAudioInput(url="https://example.com/a.wav"),
                )
            ]
        assert events == [
            SpeechEvent(type="final", text="recognized clip", language="zh"),
        ]
        assert recognizer.last_audio is not None
        assert recognizer.last_audio.url == "https://example.com/a.wav"
        assert recognizer.last_options is not None
        assert recognizer.last_options.context == "project fivc"

    @pytest.mark.asyncio
    async def test_byte_stream_yields_configured_final(self):
        provider = FakeSpeechProvider(transcript="streamed hello")

        async def chunks():
            yield b"\x00\x01"
            yield b"\x02"

        async with await provider.get_recognizer() as recognizer:
            events = [event async for event in recognizer.stream_async(chunks())]
        assert events == [
            SpeechEvent(type="final", text="streamed hello", language="zh"),
        ]
        assert recognizer.last_chunks == [b"\x00\x01", b"\x02"]


class TestGetSpeechProviderAsync:
    @pytest.mark.asyncio
    async def test_queries_default_dashscope_name(self):
        from fivccliche.utils import deps

        provider = MagicMock(spec=ISpeechProvider)
        with patch(
            "fivccliche.utils.deps.query_component",
            return_value=provider,
        ) as query:
            result = await deps.get_speech_provider_async()

        assert result is provider
        assert query.call_args.kwargs["name"] == "dashscope"

    @pytest.mark.asyncio
    async def test_queries_explicit_name(self):
        from fivccliche.utils import deps

        provider = MagicMock(spec=ISpeechProvider)
        with patch(
            "fivccliche.utils.deps.query_component",
            return_value=provider,
        ) as query:
            result = await deps.get_speech_provider_async("dashscope_realtime")

        assert result is provider
        assert query.call_args.kwargs["name"] == "dashscope_realtime"

    @pytest.mark.asyncio
    async def test_returns_none_when_provider_not_registered(self):
        from fivccliche.utils import deps

        with patch("fivccliche.utils.deps.query_component", return_value=None):
            result = await deps.get_speech_provider_async()

        assert result is None
