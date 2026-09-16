"""Unit tests for speech function tools."""

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fivccliche.modules.agent_chats.tools import SpeechTranscribe
from fivccliche.services.interfaces.speech import (
    SpeechAudioInput,
    SpeechEvent,
    SpeechRecognizeOptions,
)

USER_UUID = "user-alice"


def _mock_asr(
    *,
    model_type: str = "dashscope",
    model: str = "qwen3-asr-flash",
    api_key: str = "sk-x",
    base_url: str | None = None,
):
    asr = MagicMock()
    asr.model_type = model_type
    asr.model = model
    asr.api_key = api_key
    asr.base_url = base_url
    return asr


@asynccontextmanager
async def _fake_session():
    yield MagicMock()


class TestSpeechTranscribe:
    @pytest.mark.asyncio
    async def test_raises_without_user_uuid(self):
        tool = SpeechTranscribe()
        with pytest.raises(ValueError, match="No user_uuid specified"):
            await tool(url="https://example.com/a.wav")

    @pytest.mark.asyncio
    async def test_raises_when_asr_config_missing(self):
        tool = SpeechTranscribe(user_uuid=USER_UUID)
        with (
            patch(
                "fivccliche.modules.agent_chats.tools.get_db_session_context_async",
                _fake_session,
            ),
            patch(
                "fivccliche.modules.agent_chats.tools.get_user_scoped_async",
                new=AsyncMock(return_value=None),
            ) as get_asr,
        ):
            with pytest.raises(ValueError, match="No speech provider specified"):
                await tool(url="https://example.com/a.wav")
        assert get_asr.await_args.kwargs["config_id"] == "default"

    @pytest.mark.asyncio
    async def test_raises_when_provider_missing(self):
        tool = SpeechTranscribe(user_uuid=USER_UUID)
        with (
            patch(
                "fivccliche.modules.agent_chats.tools.get_db_session_context_async",
                _fake_session,
            ),
            patch(
                "fivccliche.modules.agent_chats.tools.get_user_scoped_async",
                new=AsyncMock(return_value=_mock_asr()),
            ),
            patch(
                "fivccliche.modules.agent_chats.tools.get_speech_provider_async",
                new=AsyncMock(return_value=None),
            ),
        ):
            with pytest.raises(ValueError, match="No speech provider specified"):
                await tool(url="https://example.com/a.wav")

    @pytest.mark.asyncio
    async def test_transcribes_last_final_event(self):
        captured: dict = {}

        async def stream_async(audio):
            captured["audio"] = audio
            yield SpeechEvent(type="final", text="hello", language="zh")

        recognizer = MagicMock()
        recognizer.stream_async = stream_async
        recognizer.__aenter__ = AsyncMock(return_value=recognizer)
        recognizer.__aexit__ = AsyncMock(return_value=None)

        async def get_recognizer(options=None, **kwargs):
            captured["options"] = options
            captured["recognizer_kwargs"] = kwargs
            return recognizer

        provider = MagicMock()
        provider.get_recognizer = get_recognizer
        get_provider = AsyncMock(return_value=provider)
        asr = _mock_asr(model_type="dashscope_realtime", model="rt-model", api_key="sk-rt")
        tool = SpeechTranscribe(user_uuid=USER_UUID, asr_id="voice")
        with (
            patch(
                "fivccliche.modules.agent_chats.tools.get_db_session_context_async",
                _fake_session,
            ),
            patch(
                "fivccliche.modules.agent_chats.tools.get_user_scoped_async",
                new=AsyncMock(return_value=asr),
            ) as get_asr,
            patch(
                "fivccliche.modules.agent_chats.tools.get_speech_provider_async",
                new=get_provider,
            ),
        ):
            result = await tool(url="https://example.com/a.wav", language="zh")

        payload = json.loads(result)
        assert payload == {"text": "hello", "language": "zh"}
        audio = captured["audio"]
        options = captured["options"]
        assert isinstance(audio, SpeechAudioInput)
        assert audio.url == "https://example.com/a.wav"
        assert isinstance(options, SpeechRecognizeOptions)
        assert options.language == "zh"
        assert captured["recognizer_kwargs"]["api_key"] == "sk-rt"
        assert captured["recognizer_kwargs"]["model"] == "rt-model"
        get_provider.assert_awaited_once_with("dashscope_realtime")
        assert get_asr.await_args.kwargs["config_id"] == "voice"
        recognizer.__aexit__.assert_awaited()
