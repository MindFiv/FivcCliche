"""Speech recognition implementations (Fake, DashScope Flash, DashScope Realtime)."""

from .dashscope_multimodal import DashScopeMultimodalSpeechProvider
from .dashscope_realtime import DashScopeRealtimeSpeechProvider
from .fake import FakeSpeechProvider

__all__ = [
    "DashScopeMultimodalSpeechProvider",
    "DashScopeRealtimeSpeechProvider",
    "FakeSpeechProvider",
]
