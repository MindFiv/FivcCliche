"""In-memory speech provider for tests and local development."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Self

from fivccliche.services.interfaces.agent_speeches import (
    ISpeechProvider,
    ISpeechRecognizer,
    ISpeechSynthesizer,
    SpeechAudioInput,
    SpeechEvent,
    SpeechRecognizeOptions,
    SpeechSynthesisOptions,
)


class FakeSpeechRecognizer(ISpeechRecognizer):
    """Yields one configured ``final`` event after consuming the audio input."""

    def __init__(
        self,
        transcript: str,
        options: SpeechRecognizeOptions | None = None,
        *,
        speech_id: str = "",
    ) -> None:
        self.transcript = transcript
        self.last_audio: SpeechAudioInput | None = None
        self.last_options = options
        self.last_chunks: list[bytes] = []
        self._id = speech_id

    @property
    def id(self) -> str:
        return self._id

    def get_option(self) -> SpeechRecognizeOptions:
        return self.last_options or SpeechRecognizeOptions()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return

    async def stream_async(
        self,
        audio: SpeechAudioInput | AsyncIterator[bytes],
    ) -> AsyncIterator[SpeechEvent]:
        if isinstance(audio, SpeechAudioInput):
            self.last_audio = audio
            self.last_chunks = []
        else:
            self.last_audio = None
            chunks: list[bytes] = []
            async for chunk in audio:
                chunks.append(chunk)
            self.last_chunks = chunks
        language = self.last_options.language if self.last_options is not None else None
        yield SpeechEvent(type="final", text=self.transcript, language=language or "zh")


class FakeSpeechProvider(ISpeechProvider):
    """ISpeechProvider that returns FakeSpeechRecognizer instances."""

    def __init__(
        self,
        component_site: Any | None = None,
        *,
        transcript: str = "hello from fake",
        audio: tuple[bytes, ...] = (b"fake audio",),
        **_kwargs: Any,
    ) -> None:
        self.transcript = transcript
        self.audio = audio

    async def get_recognizer(
        self,
        options: SpeechRecognizeOptions | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> ISpeechRecognizer:
        speech_id = kwargs.get("id")
        return FakeSpeechRecognizer(
            self.transcript,
            options,
            speech_id=speech_id if isinstance(speech_id, str) else "",
        )

    async def get_synthesizer(
        self,
        options: SpeechSynthesisOptions | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> ISpeechSynthesizer:
        speech_id = kwargs.get("id")
        return FakeSpeechSynthesizer(
            self.audio,
            options,
            speech_id=speech_id if isinstance(speech_id, str) else "",
        )


class FakeSpeechSynthesizer(ISpeechSynthesizer):
    """Yields configured audio after recording the synthesis input."""

    def __init__(
        self,
        audio: tuple[bytes, ...],
        options: SpeechSynthesisOptions | None,
        *,
        speech_id: str = "",
    ) -> None:
        self.audio = audio
        self.options = options
        self.text_chunks: list[str] = []
        self._id = speech_id

    @property
    def id(self) -> str:
        return self._id

    def get_option(self) -> SpeechSynthesisOptions:
        return self.options or SpeechSynthesisOptions(voice="")

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return

    async def stream_async(self, text: str | AsyncIterator[str]) -> AsyncIterator[bytes]:
        if isinstance(text, str):
            self.text_chunks = [text]
        else:
            self.text_chunks = [chunk async for chunk in text]
        for chunk in self.audio:
            yield chunk
