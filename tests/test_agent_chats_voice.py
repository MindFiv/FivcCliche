"""Tests for full-duplex chat voice orchestration."""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import asyncio
import json
import pytest
from fivcplayground.agents import AgentRunEvent
from fivccliche.services.interfaces.agent_speeches import SpeechRequestError

from fivccliche.utils.chats import ChatChannel
from fivccliche.utils.chats.parsers import _chat_voice_clean
from fivccliche.utils.chats.processors import (
    _INPUT_SAMPLE_RATE,
    _OUTPUT_SAMPLE_RATE,
    ChatSnapshot,
    ChatVoiceProcessor,
)


class _SpeechOption:
    def __init__(self, sample_rate: int) -> None:
        self.format = "pcm"
        self.sample_rate = sample_rate


class _Recognized:
    id = "asr"

    def get_option(self) -> _SpeechOption:
        return _SpeechOption(16_000)


class _Spoken:
    id = "tts"

    def get_option(self) -> _SpeechOption:
        return _SpeechOption(24_000)


def _voice_authenticator():
    auth = MagicMock()
    auth.verify_credential_async = AsyncMock(return_value=MagicMock())
    return auth


async def _run_voice_process(websocket, *, chat_run, asr, tts, provider, mutex_site=None):
    asr.model_type = "dashscope_realtime"
    tts.model_type = "dashscope_realtime"
    frames = [
        {"type": "auth", "access_token": "valid-token"},
        {"type": "start"},
    ]
    websocket.messages[0:0] = [
        {"type": "websocket.receive", "text": json.dumps(frame)} for frame in frames
    ]
    async with ChatChannel(websocket, _voice_authenticator()) as channel:
        user = await channel.authenticate_async()
        start_frame = await channel.receive_json_async()
        assert start_frame is not None
        recognizer = await provider.get_recognizer()
        synthesizer = await provider.get_synthesizer()
        run_provider = MagicMock()
        run_provider.create_chat_run.return_value = chat_run
        mutex = mutex_site.get_mutex("chats:message:chat-1") if mutex_site else None
        async with recognizer, synthesizer:
            processor = ChatVoiceProcessor(
                channel,
                user=user,
                chat=ChatSnapshot("chat-1", "agent-1", {}, None),
                recognizer=recognizer,
                synthesizer=synthesizer,
                run_provider=run_provider,
                mutex=mutex,
                timeout=timedelta(minutes=5),
            )
            await processor.process_async()
    if mutex_site is not None:
        mutex_site.get_mutex.assert_called_once_with("chats:message:chat-1")


def test_chat_voice_clean_strips_markdown_for_tts():
    assert _chat_voice_clean("看[这里](https://example.com)重要") == "看这里重要"


@pytest.mark.asyncio
async def test_router_resolves_realtime_voice_values():
    asr = MagicMock(id="asr", model_type="dashscope_realtime", model="asr-model", api_key="k")
    tts = MagicMock(id="tts", model_type="dashscope_realtime", model="tts-model", api_key="k")
    recognizer = MagicMock()
    synthesizer = MagicMock()
    provider = MagicMock()
    provider.get_recognizer = AsyncMock(return_value=recognizer)
    provider.get_synthesizer = AsyncMock(return_value=synthesizer)
    session = AsyncMock()
    user = _voice_authenticator().verify_credential_async.return_value

    with (
        patch(
            "fivccliche.modules.agent_configs.utils.get_user_scoped_async",
            new=AsyncMock(side_effect=[asr, tts]),
        ),
        patch(
            "fivccliche.utils.deps.get_speech_provider_async",
            new=AsyncMock(return_value=provider),
        ),
    ):
        from fivccliche.modules.agent_chats.routers import _load_realtime_voice_setup

        opened = await _load_realtime_voice_setup(
            session,
            MagicMock(),
            user,
            ChatSnapshot("chat-1", "agent-1", {}, None),
            {"asr_id": "chat-asr", "tts_id": "chat-tts"},
        )

    assert opened == (recognizer, synthesizer)
    assert provider.get_recognizer.await_args.kwargs["id"] == "asr"
    assert provider.get_synthesizer.await_args.kwargs["id"] == "tts"
    session.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_router_defaults_to_realtime_config_ids():
    asr = MagicMock(id="realtime", model_type="dashscope_realtime", model="m", api_key="k")
    tts = MagicMock(id="realtime", model_type="dashscope_realtime", model="m", api_key="k")
    provider = MagicMock()
    provider.get_recognizer = AsyncMock(return_value=MagicMock())
    provider.get_synthesizer = AsyncMock(return_value=MagicMock())
    session = AsyncMock()
    user = _voice_authenticator().verify_credential_async.return_value

    with (
        patch(
            "fivccliche.modules.agent_configs.utils.get_user_scoped_async",
            new=AsyncMock(side_effect=[asr, tts]),
        ) as get_config,
        patch(
            "fivccliche.utils.deps.get_speech_provider_async",
            new=AsyncMock(return_value=provider),
        ),
    ):
        from fivccliche.modules.agent_chats.routers import _load_realtime_voice_setup

        await _load_realtime_voice_setup(
            session,
            MagicMock(),
            user,
            ChatSnapshot("chat-1", "agent-1", {}, None),
            {},
        )

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

    async def accept(self):
        return None

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

    async def close(self, code=1000, reason=None):
        self.closed = code


@pytest.mark.asyncio
async def test_voice_session_recognizes_streams_and_synthesizes():
    websocket = FakeVoiceWebSocket(
        [
            {"type": "websocket.receive", "bytes": b"pcm"},
            {"type": "websocket.receive", "text": '{"type":"vad","event":"speech_end"}'},
        ]
    )

    class FakeRecognizer(_Recognized):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def stream_async(self, audio):
            async for _ in audio:
                pass
            yield MagicMock(type="final", text="你好")

    class FakeSynthesizer(_Spoken):
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
        async def stream_async(self, *_args, **kwargs):
            yield AgentRunEvent.START, start_run
            yield AgentRunEvent.STREAM, stream_run
            yield AgentRunEvent.FINISH, finish_run

    await _run_voice_process(
        websocket,
        chat_run=FakeJob(),
        asr=asr,
        tts=tts,
        provider=MagicMock(
            get_recognizer=AsyncMock(return_value=FakeRecognizer()),
            get_synthesizer=AsyncMock(return_value=FakeSynthesizer()),
        ),
    )

    events = [item["event"] for item in websocket.sent if isinstance(item, dict)]
    assert events[:2] == ["ready", "state"]
    assert "transcript" in events
    assert "audio_start" in events
    assert "audio_end" in events
    assert "turn_completed" in events
    assert b"tts-pcm" in websocket.sent
    assert websocket.closed == 1000
    assert _INPUT_SAMPLE_RATE == 16000
    assert _OUTPUT_SAMPLE_RATE == 24000


def test_voice_processor_is_public_entrypoint():
    processor = ChatVoiceProcessor(
        MagicMock(),
        user=MagicMock(),
        chat=ChatSnapshot("chat-1", "agent-1", {}, None),
        recognizer=MagicMock(),
        synthesizer=MagicMock(),
        run_provider=MagicMock(),
        mutex=None,
        timeout=timedelta(minutes=5),
    )

    assert isinstance(processor, ChatVoiceProcessor)


@pytest.mark.asyncio
async def test_voice_session_reports_tts_vendor_detail():
    websocket = FakeVoiceWebSocket(
        [
            {"type": "websocket.receive", "text": '{"type":"vad","event":"speech_end"}'},
        ]
    )

    class FakeRecognizer(_Recognized):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def stream_async(self, audio):
            async for _ in audio:
                pass
            yield MagicMock(type="final", text="你好")

    class FailingSynthesizer(_Spoken):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def stream_async(self, text):
            raise SpeechRequestError("InvalidParameter: Request voice is invalid!")
            yield b""

    finish_run = MagicMock(id="run-1", delta=None, reply=MagicMock(text="你好"))
    finish_run.model_dump.return_value = {"id": "run-1", "reply": {"text": "你好"}}
    stream_run = MagicMock(id="run-1", delta=MagicMock(text="你好"))
    stream_run.model_dump.return_value = {"id": "run-1", "delta": {"text": "你好"}}

    class FakeJob:
        async def stream_async(self, *_args, **kwargs):
            yield AgentRunEvent.STREAM, stream_run
            yield AgentRunEvent.FINISH, finish_run

    await _run_voice_process(
        websocket,
        chat_run=FakeJob(),
        asr=MagicMock(),
        tts=MagicMock(),
        provider=MagicMock(
            get_recognizer=AsyncMock(return_value=FakeRecognizer()),
            get_synthesizer=AsyncMock(return_value=FailingSynthesizer()),
        ),
    )

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

    class FakeRecognizer(_Recognized):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def stream_async(self, audio):
            async for _ in audio:
                pass
            yield MagicMock(type="final", text="你好")

    class StreamingSynthesizer(_Spoken):
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
        async def stream_async(self, *_args, **kwargs):
            yield AgentRunEvent.START, MagicMock(id="run-1", delta=None)
            yield AgentRunEvent.STREAM, stream_run
            await first_text_seen.wait()
            yield AgentRunEvent.FINISH, finish_run

    await _run_voice_process(
        websocket,
        chat_run=FakeJob(),
        asr=MagicMock(),
        tts=MagicMock(),
        provider=MagicMock(
            get_recognizer=AsyncMock(return_value=FakeRecognizer()),
            get_synthesizer=AsyncMock(return_value=StreamingSynthesizer()),
        ),
    )

    assert tts_chunks == ["你好。"]
    indexes = {
        item["event"]: index for index, item in enumerate(websocket.sent) if isinstance(item, dict)
    }
    audio_index = websocket.sent.index(b"tts-pcm")
    assert audio_index < indexes["turn_completed"]


@pytest.mark.asyncio
async def test_voice_session_ends_asr_when_partials_go_idle():
    websocket = FakeVoiceWebSocket(
        [
            {"type": "websocket.receive", "bytes": b"pcm"},
        ]
    )

    class IdleRecognizer(_Recognized):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def stream_async(self, audio):
            async for _chunk in audio:
                yield MagicMock(type="final", text="你好")
                for _ in range(5):
                    yield MagicMock(type="partial", text="")
                return

    class FakeSynthesizer(_Spoken):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def stream_async(self, text):
            async for _chunk in text:
                pass
            yield b"tts-pcm"

    finish_run = MagicMock(id="run-1", delta=None, reply=MagicMock(text="你好"))
    finish_run.model_dump.return_value = {"id": "run-1", "reply": {"text": "你好"}}
    stream_run = MagicMock(id="run-1", delta=MagicMock(text="你好"))
    stream_run.model_dump.return_value = {"id": "run-1", "delta": {"text": "你好"}}

    class FakeJob:
        async def stream_async(self, *_args, **kwargs):
            yield AgentRunEvent.START, MagicMock(id="run-1", delta=None)
            yield AgentRunEvent.STREAM, stream_run
            yield AgentRunEvent.FINISH, finish_run

    await _run_voice_process(
        websocket,
        chat_run=FakeJob(),
        asr=MagicMock(),
        tts=MagicMock(),
        provider=MagicMock(
            get_recognizer=AsyncMock(return_value=IdleRecognizer()),
            get_synthesizer=AsyncMock(return_value=FakeSynthesizer()),
        ),
    )

    events = [item["event"] for item in websocket.sent if isinstance(item, dict)]
    assert "transcript" in events
    assert {"event": "state", "info": {"state": "responding"}} in websocket.sent
    assert b"tts-pcm" in websocket.sent
    assert "turn_completed" in events
    assert websocket.closed == 1000
