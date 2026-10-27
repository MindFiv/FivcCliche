"""Speech provider factory.

Recognition and synthesis types live in ``fivcplayground.speeches``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from fivcglue import IComponent
from fivcplayground.speeches import (
    SpeechRecognizeOption,
    SpeechRecognizer,
    SpeechSynthesisOption,
    SpeechSynthesizer,
)


class ISpeechProvider(IComponent):
    """Factory for speech recognizers and synthesizers."""

    @abstractmethod
    async def get_recognizer(
        self,
        options: SpeechRecognizeOption | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> SpeechRecognizer:
        """Create a recognizer.

        ``api_key``, ``model``, and ``base_url`` are caller-supplied values,
        typically loaded from a database config row. Realtime implementations
        open the vendor session in ``__aenter__``.
        """

    @abstractmethod
    async def get_synthesizer(
        self,
        options: SpeechSynthesisOption | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> SpeechSynthesizer:
        """Create a synthesizer when the provider supports TTS."""
