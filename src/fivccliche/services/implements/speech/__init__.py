"""Speech implementations (Fake, DashScope Flash, Realtime, and TTS)."""

from .dashscope import DashScopeSpeechProvider
from .dashscope_realtime import DashScopeRealtimeSpeechProvider
from .fake import FakeSpeechProvider

__all__ = [
    "DashScopeRealtimeSpeechProvider",
    "DashScopeSpeechProvider",
    "FakeSpeechProvider",
]
