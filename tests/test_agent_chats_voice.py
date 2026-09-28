"""Tests for full-duplex chat voice orchestration."""

from unittest.mock import AsyncMock, MagicMock, patch

import asyncio
import pytest
from fivcplayground.agents import AgentRunEvent
from fivccliche.services.interfaces.speech import SpeechRequestError

from fivccliche.utils.voice import (
    _INPUT_SAMPLE_RATE,
    _OUTPUT_SAMPLE_RATE,
    VoiceChatWSHandler,
    VoiceChatSession,
    VoiceChatTextBuffer,
)


def test_voice_chat_text_buffer_clean_for_tts():
    assert VoiceChatTextBuffer.clean("看[这里](https://example.com)重要") == "看这里重要"


def test_speech_text_buffer_flushes_sentence_and_remainder():
    buffer = VoiceChatTextBuffer(max_length=24)
    assert buffer.append("第一句。第二") == ["第一句。"]
    assert buffer.append("句。尾巴") == ["第二句。"]
    assert buffer.flush() == ["尾巴"]


@pytest.mark.asyncio
async def test_voice_chat_ws_handler_resolves_realtime_providers():
    asr = MagicMock(id="asr", model_type="dashscope_realtime")
    tts = MagicMock(id="tts", model_type="dashscope_realtime")
    provider = MagicMock()

    with (
        patch(
            "fivccliche.utils.voice.get_user_scoped_async",
            new=AsyncMock(side_effect=[asr, tts]),
        ),
        patch(
            "fivccliche.utils.voice.get_speech_provider_async",
            new=AsyncMock(return_value=provider),
        ),
    ):
        handler = VoiceChatWSHandler(
            MagicMock(),
            chat_uuid="chat-1",
            authenticator=MagicMock(),
            session=AsyncMock(),
            mutex=None,
        )
        resolved = await handler._resolve_voice_configs(
            user=MagicMock(),
            chat=MagicMock(context={"asr_id": "chat-asr", "tts_id": "chat-tts"}),
            start_frame={},
        )

    assert resolved == (asr, tts, provider, provider)


@pytest.mark.asyncio
async def test_voice_chat_ws_handler_defaults_to_realtime_config_ids():
    asr = MagicMock(id="realtime", model_type="dashscope_realtime")
    tts = MagicMock(id="realtime", model_type="dashscope_realtime")
    provider = MagicMock()

    with (
        patch(
            "fivccliche.utils.voice.get_user_scoped_async",
            new=AsyncMock(side_effect=[asr, tts]),
        ) as get_config,
        patch(
            "fivccliche.utils.voice.get_speech_provider_async",
            new=AsyncMock(return_value=provider),
        ),
    ):
        handler = VoiceChatWSHandler(
            MagicMock(),
            chat_uuid="chat-1",
            authenticator=MagicMock(),
            session=AsyncMock(),
            mutex=None,
        )
        resolved = await handler._resolve_voice_configs(
            user=MagicMock(),
            chat=MagicMock(context={}),
            start_frame={},
        )

    assert resolved == (asr, tts, provider, provider)
    assert [call.kwargs["config_id"] for call in get_config.await_args_list] == [
        "realtime",
        "realtime",
    ]


class FakeVoiceWebSocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []
        self.closed = None
        self.turn_completed = asyncio.Event()
        self.client_state = MagicMock()
        self.client_state.name = "CONNECTED"

    async def receive(self):
        if not self.messages:
            await self.turn_completed.wait()
            return {"type": "websocket.disconnect"}
        await asyncio.sleep(0)
        return self.messages.pop(0)

    async def send_json(self, value):
        self.sent.append(value)
        if value.get("event") == "turn_completed":
            self.turn_completed.set()

    async def send_bytes(self, value):
        self.sent.append(value)

    async def close(self, code=1000):
        self.closed = code


@pytest.mark.asyncio
async def test_voice_session_recognizes_streams_and_synthesizes():
    websocket = FakeVoiceWebSocket(
        [
            {"type": "websocket.receive", "bytes": b"pcm"},
            {"type": "websocket.receive", "text": '{"type":"vad","event":"speech_end"}'},
        ]
    )

    class FakeRecognizer:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def stream_async(self, audio):
            async for _ in audio:
                pass
            yield MagicMock(type="final", text="你好")

    class FakeSynthesizer:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def stream_async(self, text):
            async for chunk in text:
                assert chunk
            yield b"tts-pcm"

    asr = MagicMock(model="asr-model")
    tts = MagicMock(model="tts-model")

    start_run = MagicMock(id="run-1", delta=None)
    start_run.model_dump.return_value = {"id": "run-1"}
    stream_run = MagicMock(id="run-1", delta=MagicMock(text="你好"))
    stream_run.model_dump.return_value = {"id": "run-1", "delta": {"text": "你好"}}
    finish_run = MagicMock(id="run-1", delta=None, reply=MagicMock(text="你好"))
    finish_run.model_dump.return_value = {"id": "run-1", "reply": {"text": "你好"}}

    class FakeJob:
        async def run_async(self, *_args, **kwargs):
            callback = kwargs["event_callback"]
            callback(AgentRunEvent.START, start_run)
            callback(AgentRunEvent.STREAM, stream_run)
            callback(AgentRunEvent.FINISH, finish_run)

    with patch("fivccliche.utils.voice.ChatQueryJob", return_value=FakeJob()):
        session = VoiceChatSession(
            websocket,
            chat=MagicMock(),
            user=MagicMock(),
            asr=asr,
            tts=tts,
            asr_provider=MagicMock(get_recognizer=AsyncMock(return_value=FakeRecognizer())),
            tts_provider=MagicMock(get_synthesizer=AsyncMock(return_value=FakeSynthesizer())),
            mutex=None,
            start_frame={},
        )

        await session.run()

        events = [item["event"] for item in websocket.sent if isinstance(item, dict)]
    assert events[:2] == ["ready", "state"]
    assert "transcript" in events
    assert "stream" in events
    assert "finish" in events
    assert "audio_start" in events
    assert "audio_end" in events
    assert "turn_completed" in events
    assert b"tts-pcm" in websocket.sent
    assert websocket.closed == 1000
    assert _INPUT_SAMPLE_RATE == 16000
    assert _OUTPUT_SAMPLE_RATE == 24000


def test_voice_ws_handler_is_public_entrypoint():
    handler = VoiceChatWSHandler(
        MagicMock(),
        chat_uuid="chat-1",
        authenticator=MagicMock(),
        session=AsyncMock(),
        mutex=None,
    )

    assert isinstance(handler, VoiceChatWSHandler)


@pytest.mark.asyncio
async def test_voice_session_reports_tts_vendor_detail():
    websocket = FakeVoiceWebSocket(
        [
            {"type": "websocket.receive", "text": '{"type":"vad","event":"speech_end"}'},
        ]
    )

    class FakeRecognizer:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def stream_async(self, audio):
            async for _ in audio:
                pass
            yield MagicMock(type="final", text="你好")

    class FailingSynthesizer:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def stream_async(self, text):
            raise SpeechRequestError("InvalidParameter: Request voice is invalid!")
            yield b""

    finish_run = MagicMock(id="run-1", delta=None, reply=MagicMock(text="你好"))
    finish_run.model_dump.return_value = {"id": "run-1", "reply": {"text": "你好"}}

    class FakeJob:
        async def run_async(self, *_args, **kwargs):
            kwargs["event_callback"](AgentRunEvent.FINISH, finish_run)

    with patch("fivccliche.utils.voice.ChatQueryJob", return_value=FakeJob()):
        session = VoiceChatSession(
            websocket,
            chat=MagicMock(),
            user=MagicMock(),
            asr=MagicMock(),
            tts=MagicMock(),
            asr_provider=MagicMock(get_recognizer=AsyncMock(return_value=FakeRecognizer())),
            tts_provider=MagicMock(get_synthesizer=AsyncMock(return_value=FailingSynthesizer())),
            mutex=None,
            start_frame={},
        )
        await session.run()

    errors = [
        item["info"]
        for item in websocket.sent
        if isinstance(item, dict) and item.get("event") == "error"
    ]
    assert {
        "code": "tts_failed",
        "message": "Speech synthesis failed",
        "detail": "InvalidParameter: Request voice is invalid!",
    } in errors


@pytest.mark.asyncio
async def test_voice_session_streams_tts_while_agent_is_running():
    websocket = FakeVoiceWebSocket(
        [
            {"type": "websocket.receive", "text": '{"type":"vad","event":"speech_end"}'},
        ]
    )
    first_text_seen = asyncio.Event()
    tts_chunks: list[str] = []

    class FakeRecognizer:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def stream_async(self, audio):
            async for _ in audio:
                pass
            yield MagicMock(type="final", text="你好")

    class StreamingSynthesizer:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def stream_async(self, text):
            async for chunk in text:
                tts_chunks.append(chunk)
                if len(tts_chunks) == 1:
                    first_text_seen.set()
                    yield b"tts-pcm"
                else:
                    yield b"unexpected"

    stream_run = MagicMock(id="run-1", delta=MagicMock(text="你好。"))
    stream_run.model_dump.return_value = {"id": "run-1", "delta": {"text": "你好。"}}
    finish_run = MagicMock(id="run-1", delta=None, reply=MagicMock(text="你好"))
    finish_run.model_dump.return_value = {"id": "run-1", "reply": {"text": "你好"}}

    class FakeJob:
        async def run_async(self, *_args, **kwargs):
            callback = kwargs["event_callback"]
            callback(AgentRunEvent.START, MagicMock(id="run-1", delta=None))
            callback(AgentRunEvent.STREAM, stream_run)
            await first_text_seen.wait()
            callback(AgentRunEvent.FINISH, finish_run)

    with patch("fivccliche.utils.voice.ChatQueryJob", return_value=FakeJob()):
        session = VoiceChatSession(
            websocket,
            chat=MagicMock(),
            user=MagicMock(),
            asr=MagicMock(),
            tts=MagicMock(),
            asr_provider=MagicMock(get_recognizer=AsyncMock(return_value=FakeRecognizer())),
            tts_provider=MagicMock(get_synthesizer=AsyncMock(return_value=StreamingSynthesizer())),
            mutex=None,
            start_frame={},
        )
        await session.run()

    assert tts_chunks == ["你好。"]
    indexes = {
        item["event"]: index for index, item in enumerate(websocket.sent) if isinstance(item, dict)
    }
    audio_index = websocket.sent.index(b"tts-pcm")
    assert audio_index < indexes["finish"] < indexes["turn_completed"]
