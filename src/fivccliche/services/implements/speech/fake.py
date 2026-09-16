"""In-memory speech provider for tests and local development."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fivccliche.services.interfaces.speech import (
    ISpeechProvider,
    ISpeechRecognizer,
    SpeechAudioInput,
    SpeechEvent,
    SpeechRecognizeOptions,
)


class FakeSpeechRecognizer(ISpeechRecognizer):
    """Yields one configured ``final`` event after consuming the audio input."""

    def __init__(self, transcript: str, options: SpeechRecognizeOptions | None = None) -> None:
        self.transcript = transcript
        self.last_audio: SpeechAudioInput | None = None
        self.last_options = options
        self.last_chunks: list[bytes] = []

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
        **_kwargs: Any,
    ) -> None:
        self.transcript = transcript

    async def get_recognizer(
        self,
        options: SpeechRecognizeOptions | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        **_kwargs: Any,
    ) -> ISpeechRecognizer:
        return FakeSpeechRecognizer(self.transcript, options)
