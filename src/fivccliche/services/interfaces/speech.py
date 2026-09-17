"""Speech recognition interfaces.

Implementation-agnostic ASR so chat and tools can transcribe audio without
depending on a vendor SDK. ``ISpeechProvider`` is a factory; recognition runs
on ``ISpeechRecognizer``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

from fivcglue import IComponent

SpeechEventType = Literal["partial", "final", "error"]


class SpeechError(Exception):
    """Base error for speech recognition."""


class SpeechRequestError(SpeechError):
    """Raised when a vendor request fails."""


class SpeechAudioInput(BaseModel):
    """One clip: a URL or Base64 payload."""

    url: str | None = None
    data_b64: str | None = None
    format: str = "wav"
    sample_rate: int = 16000
    channels: int = 1

    @model_validator(mode="after")
    def require_one_source(self) -> SpeechAudioInput:
        has_url = bool(self.url)
        has_data = bool(self.data_b64)
        if has_url == has_data:
            raise ValueError("Exactly one of url or data_b64 is required")
        return self


class SpeechRecognizeOptions(BaseModel):
    """Optional hints captured when the recognizer is created."""

    language: str | None = None
    enable_itn: bool = False
    hotwords: list[str] | None = None
    context: str | None = None
    format: str = "pcm"
    sample_rate: int = 16000
    channels: int = 1
    extra: dict[str, Any] = Field(default_factory=dict)


class SpeechEvent(BaseModel):
    """One recognition event. ``text`` is set for partial/final; ``message`` for error."""

    type: SpeechEventType
    text: str = ""
    language: str | None = None
    message: str | None = None


class ISpeechRecognizer(ABC):
    """One recognition turn. Created by ``ISpeechProvider.get_recognizer``."""

    @abstractmethod
    async def __aenter__(self) -> Self:
        raise NotImplementedError

    @abstractmethod
    async def __aexit__(self, *exc: object) -> None:
        raise NotImplementedError

    @abstractmethod
    async def stream_async(
        self,
        audio: SpeechAudioInput | AsyncIterator[bytes],
    ) -> AsyncIterator[SpeechEvent]:
        """Yield ``partial`` / ``final`` / ``error`` events.

        ``SpeechAudioInput`` is a complete clip (URL may be passed through or
        fetched by the adapter). An async byte iterator is one utterance;
        exhaustion is the commit.
        """
        yield SpeechEvent(type="final")


class ISpeechProvider(IComponent):
    """Factory for speech recognizers (and later synthesizers)."""

    @abstractmethod
    async def get_recognizer(
        self,
        options: SpeechRecognizeOptions | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> ISpeechRecognizer:
        """Create a recognizer.

        ``api_key``, ``model``, and ``base_url`` override SPEECH config when
        not ``None``. Realtime implementations open the vendor session in
        ``__aenter__``.
        """
