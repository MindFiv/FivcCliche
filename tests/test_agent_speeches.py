"""Unit tests for the speech recognition contract and Fake provider."""

import inspect
from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

from fivccliche.services.implements.agent_speeches.fake import (
    FakeSpeechProvider,
    FakeSpeechRecognizer,
)
from fivcplayground.speeches import (
    SpeechRecognizeEvent,
    SpeechRecognizeOption,
    SpeechRecognizer,
    SpeechSynthesisOption,
    SpeechSynthesizer,
)

from fivccliche.services.interfaces.agent_speeches import ISpeechProvider


class TestFakeSpeechProvider:
    @pytest.mark.asyncio
    async def test_clip_yields_configured_final(self):
        provider = FakeSpeechProvider(transcript="recognized clip")
        async with await provider.get_recognizer(
            SpeechRecognizeOption(language="zh", context="project fivc")
        ) as recognizer:
            assert isinstance(recognizer, FakeSpeechRecognizer)
            events = [event async for event in recognizer.stream_async(b"\x00\x01\x02")]
        assert events == [
            SpeechRecognizeEvent(type="final", text="recognized clip", language="zh"),
        ]
        assert recognizer.last_chunks == [b"\x00\x01\x02"]
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
            SpeechRecognizeEvent(type="final", text="streamed hello", language="zh"),
        ]
        assert recognizer.last_chunks == [b"\x00\x01", b"\x02"]

    @pytest.mark.asyncio
    async def test_synthesizer_options_default(self):
        provider = FakeSpeechProvider(audio=(b"audio",))

        async with await provider.get_synthesizer() as synthesizer:
            audio = [chunk async for chunk in synthesizer.stream_async("hello")]

        assert audio == [b"audio"]
        assert synthesizer.options is None

    @pytest.mark.asyncio
    async def test_synthesizer_yields_configured_audio(self):
        provider = FakeSpeechProvider(audio=(b"left ", b"right"))

        async with await provider.get_synthesizer(
            SpeechSynthesisOption(voice="longxiaochun")
        ) as synthesizer:
            audio = [chunk async for chunk in synthesizer.stream_async("hello")]

        assert audio == [b"left ", b"right"]

    @pytest.mark.asyncio
    async def test_synthesizer_consumes_text_chunks(self):
        provider = FakeSpeechProvider(audio=(b"audio",))

        async def text():
            yield "hello "
            yield "world"

        async with await provider.get_synthesizer(
            SpeechSynthesisOption(voice="longxiaochun")
        ) as synthesizer:
            audio = [chunk async for chunk in synthesizer.stream_async(text())]

        assert audio == [b"audio"]


class TestSpeechRecognizerInterface:
    def test_stream_is_async_generator_and_lifecycle_is_abstract(self):
        assert inspect.isasyncgenfunction(FakeSpeechRecognizer.stream_async)
        assert SpeechRecognizer.stream_async.__isabstractmethod__
        assert SpeechRecognizer.__aenter__.__isabstractmethod__
        assert SpeechRecognizer.__aexit__.__isabstractmethod__
        assert SpeechRecognizer.id.fget.__isabstractmethod__  # type: ignore[union-attr]
        assert SpeechRecognizer.get_option.__isabstractmethod__
        assert SpeechSynthesizer.id.fget.__isabstractmethod__  # type: ignore[union-attr]
        assert SpeechSynthesizer.get_option.__isabstractmethod__

        class MissingLifecycle(SpeechRecognizer):
            async def stream_async(
                self,
                audio: bytes | AsyncIterator[bytes],
            ) -> AsyncIterator[SpeechRecognizeEvent]:
                yield SpeechRecognizeEvent(type="final")

        with pytest.raises(TypeError, match="abstract"):
            MissingLifecycle()


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


class TestSpeechProviderSynthesisContract:
    def test_get_synthesizer_is_abstract(self):
        assert ISpeechProvider.get_synthesizer.__isabstractmethod__
