"""Unit tests for ISpeech* ↔ Pipecat STT/TTS adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from fivccliche.services.implements.agent_speeches.fake import (
    FakeSpeechRecognizer,
    FakeSpeechSynthesizer,
)
from fivccliche.services.interfaces.agent_speeches import (
    SpeechEvent,
    SpeechRecognizeOptions,
    SpeechSynthesisOptions,
)
from fivccliche.utils.chats.speeches import (
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    SpeechSTTService,
    SpeechTTSService,
)


class _ScriptedVAD:
    def __init__(self, states: list[VADState]) -> None:
        self._states = list(states)
        self._i = 0

    def set_sample_rate(self, sample_rate: int) -> None:
        return None

    async def analyze_audio(self, buffer: bytes) -> VADState:
        if self._i < len(self._states):
            state = self._states[self._i]
            self._i += 1
            return state
        return VADState.QUIET


class _StreamingRecognizer(FakeSpeechRecognizer):
    async def stream_async(self, audio) -> AsyncIterator[SpeechEvent]:
        chunks: list[bytes] = []
        async for chunk in audio:
            chunks.append(chunk)
        self.last_chunks = chunks
        yield SpeechEvent(type="partial", text="你")
        yield SpeechEvent(type="final", text="你好", language="zh")


def _wire_stt(stt: SpeechSTTService) -> list[Any]:
    pushed: list[Any] = []

    async def capture(frame: Any, direction: Any = FrameDirection.DOWNSTREAM) -> None:
        pushed.append(frame)

    stt.push_frame = capture  # type: ignore[method-assign]
    stt._record_stt_audio_usage = lambda _audio: None  # type: ignore[method-assign]
    stt._muted = False
    return pushed


@pytest.mark.asyncio
async def test_speech_stt_emits_final_transcription_after_vad_stop():
    recognizer = FakeSpeechRecognizer("你好", SpeechRecognizeOptions(language="zh"))
    vad = _ScriptedVAD([VADState.SPEAKING, VADState.SPEAKING, VADState.QUIET])
    stt = SpeechSTTService(recognizer, vad_analyzer=vad, user_id="u1")  # type: ignore[arg-type]
    pushed = _wire_stt(stt)

    chunk = b"\x00\x01" * 80
    for _ in range(3):
        await stt.process_audio_frame(
            InputAudioRawFrame(audio=chunk, sample_rate=INPUT_SAMPLE_RATE, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )

    finals = [f for f in pushed if isinstance(f, TranscriptionFrame)]
    assert len(finals) == 1
    assert finals[0].text == "你好"
    assert finals[0].finalized is True
    assert recognizer.last_chunks == [chunk + chunk]


@pytest.mark.asyncio
async def test_speech_stt_emits_partial_then_final():
    recognizer = _StreamingRecognizer("unused")
    vad = _ScriptedVAD([VADState.SPEAKING, VADState.QUIET])
    stt = SpeechSTTService(recognizer, vad_analyzer=vad)  # type: ignore[arg-type]
    pushed = _wire_stt(stt)

    chunk = b"\x01\x02" * 40
    await stt.process_audio_frame(
        InputAudioRawFrame(audio=chunk, sample_rate=INPUT_SAMPLE_RATE, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )
    await stt.process_audio_frame(
        InputAudioRawFrame(audio=chunk, sample_rate=INPUT_SAMPLE_RATE, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )

    assert any(isinstance(f, InterimTranscriptionFrame) and f.text == "你" for f in pushed)
    assert any(isinstance(f, TranscriptionFrame) and f.text == "你好" for f in pushed)


@pytest.mark.asyncio
async def test_speech_tts_yields_audio_frames():
    synthesizer = FakeSpeechSynthesizer(
        (b"pcm-a", b"pcm-b"),
        SpeechSynthesisOptions(voice="Cherry", format="pcm", sample_rate=OUTPUT_SAMPLE_RATE),
    )
    tts = SpeechTTSService(synthesizer)
    frames = [frame async for frame in tts.run_tts("hello", "ctx-1") if frame is not None]
    assert len(frames) == 2
    assert all(isinstance(f, TTSAudioRawFrame) for f in frames)
    assert frames[0].audio == b"pcm-a"
    assert frames[0].sample_rate == OUTPUT_SAMPLE_RATE
    assert frames[0].context_id == "ctx-1"
    assert synthesizer.text_chunks == ["hello"]
