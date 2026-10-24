"""Pipecat speech adapters and full-duplex chat voice sessions."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from datetime import timedelta
from typing import Any

from fivcglue.interfaces.mutexes import IMutex
from fivcplayground.agents import AgentRunSession, create_agent_async
from fivcplayground.skills import create_skill_retriever_async
from fivcplayground.tools import create_tool_retriever_async
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADState
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.utils.time import time_now_iso8601

from fivccliche.services.interfaces.agent_chats import IUserChatProvider
from fivccliche.services.interfaces.agent_configs import IUserConfigProvider
from fivccliche.services.interfaces.agent_speeches import (
    ISpeechRecognizer,
    ISpeechSynthesizer,
)
from fivccliche.services.interfaces.auth import IUser
from fivccliche.utils.chats.channels import ChatChannel
from fivccliche.utils.deps import get_chat_provider_async, get_config_provider_async

logger = logging.getLogger(__name__)

INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000


class SpeechSTTService(STTService):
    """Utterance-batch STT: Silero VAD segments audio, then ``ISpeechRecognizer``."""

    def __init__(
        self,
        recognizer: ISpeechRecognizer,
        *,
        sample_rate: int = INPUT_SAMPLE_RATE,
        vad_analyzer: VADAnalyzer | None = None,
        user_id: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._recognizer = recognizer
        self._vad = vad_analyzer or SileroVADAnalyzer(sample_rate=sample_rate)
        self._user_id = user_id
        self._buffer = bytearray()
        self._speaking = False
        self._pre_roll = bytearray()
        self._pre_roll_limit = sample_rate * 2  # ~1s of 16-bit mono PCM

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Unused: audio is segmented in ``process_audio_frame``."""
        yield None

    async def process_audio_frame(self, frame: Any, direction: Any) -> None:
        from pipecat.frames.frames import UserAudioRawFrame

        if self._muted:
            return
        if isinstance(frame, UserAudioRawFrame):
            self._user_id = frame.user_id

        audio = frame.audio
        if not audio:
            return

        self._record_stt_audio_usage(audio)
        state = await self._vad.analyze_audio(audio)

        if state in (VADState.STARTING, VADState.SPEAKING, VADState.STOPPING):
            if not self._speaking:
                self._speaking = True
                self._buffer = bytearray(self._pre_roll)
                self._pre_roll.clear()
            self._buffer.extend(audio)
            return

        # QUIET
        if self._speaking:
            self._speaking = False
            segment = bytes(self._buffer)
            self._buffer.clear()
            if segment:
                await self._emit_transcript_async(segment, direction)
            return

        self._pre_roll.extend(audio)
        overflow = len(self._pre_roll) - self._pre_roll_limit
        if overflow > 0:
            del self._pre_roll[:overflow]

    async def _emit_transcript_async(self, segment: bytes, direction: Any) -> None:
        timestamp = time_now_iso8601()

        async def one_chunk() -> AsyncGenerator[bytes, None]:
            yield segment

        try:
            async for event in self._recognizer.stream_async(one_chunk()):
                language: Language | None = None
                if event.language:
                    try:
                        language = Language(event.language.lower())
                    except ValueError:
                        language = None
                if event.type == "partial":
                    await self.push_frame(
                        InterimTranscriptionFrame(
                            text=event.text,
                            user_id=self._user_id,
                            timestamp=timestamp,
                            language=language,
                        ),
                        direction,
                    )
                elif event.type == "final":
                    await self.push_frame(
                        TranscriptionFrame(
                            text=event.text,
                            user_id=self._user_id,
                            timestamp=timestamp,
                            language=language,
                            finalized=True,
                        ),
                        direction,
                    )
                elif event.type == "error":
                    await self.push_error(
                        event.message or "Speech recognition failed",
                    )
        except Exception as exc:
            logger.exception("SpeechSTTService transcription failed")
            await self.push_error(f"{self}: transcription failed: {exc}", exception=exc)


class SpeechTTSService(TTSService):
    """TTS backed by an already-opened ``ISpeechSynthesizer``."""

    def __init__(
        self,
        synthesizer: ISpeechSynthesizer,
        *,
        sample_rate: int | None = None,
        **kwargs: Any,
    ) -> None:
        options = synthesizer.get_option()
        resolved_rate = sample_rate if sample_rate is not None else options.sample_rate
        super().__init__(sample_rate=resolved_rate, **kwargs)
        self._synthesizer = synthesizer
        self._num_channels = 1

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        sample_rate = self.sample_rate or OUTPUT_SAMPLE_RATE
        async for chunk in self._synthesizer.stream_async(text):
            if not chunk:
                continue
            yield TTSAudioRawFrame(
                audio=chunk,
                sample_rate=sample_rate,
                num_channels=self._num_channels,
                context_id=context_id,
            )


async def process_chat_speech_async(
    *,
    channel: ChatChannel,
    user: IUser,
    chat: AgentRunSession,
    recognizer: ISpeechRecognizer,
    synthesizer: ISpeechSynthesizer,
    mutex: IMutex | None = None,
    timeout: timedelta = timedelta(minutes=5),
    config_provider: IUserConfigProvider | None = None,
    chat_provider: IUserChatProvider | None = None,
) -> None:
    """Send ``ready``, then run a Pipecat transport pipeline until disconnect."""
    from fivcplayground.backends.pipecat.agents import PipecatAgentBackend
    from fivcplayground.backends.pipecat.models import PipecatModelBackend
    from fivcplayground.backends.pipecat.tools import PipecatToolBackend

    if config_provider is None:
        config_provider = await get_config_provider_async()
    if chat_provider is None:
        chat_provider = await get_chat_provider_async()

    model_backend = PipecatModelBackend()
    tool_backend = PipecatToolBackend()
    agent_backend = PipecatAgentBackend()

    model_repository = config_provider.get_model_repository(user_uuid=user.uuid)
    agent_repository = config_provider.get_agent_repository(user_uuid=user.uuid)
    agent_config = await agent_repository.get_agent_config_async(chat.agent_id)
    if agent_config is None:
        await channel.fail_async(
            code="agent_not_found",
            message="Agent config not found",
            close_code=4404,
        )
    model_config = await model_repository.get_model_config_async(agent_config.model_id)
    if model_config is None:
        await channel.fail_async(
            code="model_not_found",
            message="LLM config not found",
            close_code=4404,
        )

    embedding_repository = config_provider.get_embedding_repository(user_uuid=user.uuid)
    embedding_backend = config_provider.get_embedding_backend()
    tool_repository = config_provider.get_tool_repository(user_uuid=user.uuid)
    skill_repository = config_provider.get_skill_repository(user_uuid=user.uuid)

    agent = await create_agent_async(
        model_backend=model_backend,
        model_config_repository=model_repository,
        agent_backend=agent_backend,
        agent_config_repository=agent_repository,
        agent_config_id=chat.agent_id,
    )
    tool_retriever = await create_tool_retriever_async(
        tool_backend=tool_backend,
        tools=None,
        tool_config_repository=tool_repository,
        embedding_backend=embedding_backend,
        embedding_config_repository=embedding_repository,
        space_id=user.uuid,
    )
    skill_retriever = await create_skill_retriever_async(
        tool_backend=tool_backend,
        skill_config_repository=skill_repository,
        embedding_backend=embedding_backend,
        embedding_config_repository=embedding_repository,
        space_id=user.uuid,
    )
    context = chat_provider.get_chat_context(
        user_uuid=user.uuid,
        context=chat.context,
        chat_uuid=chat.id,
    )
    chat_repository = chat_provider.get_chat_repository(user_uuid=user.uuid)

    recognize_opts = recognizer.get_option()
    synthesis_opts = synthesizer.get_option()
    await channel.send_json_async(
        {
            "event": "ready",
            "info": {
                "asr_id": recognizer.id,
                "tts_id": synthesizer.id,
                "input": {
                    "format": recognize_opts.format,
                    "sample_rate": recognize_opts.sample_rate,
                },
                "output": {
                    "format": synthesis_opts.format,
                    "sample_rate": synthesis_opts.sample_rate,
                },
            },
        }
    )

    mutex_acquired = False
    if mutex is not None:
        # Mutex TTL is slightly longer than the session timeout so the lock does not
        # expire while we are still in cleanup (release / finally) at the timeout edge.
        mutex_acquired = await mutex.acquire_async(
            expire=timeout + timedelta(seconds=10),
            timeout=None,
        )
        if not mutex_acquired:
            await channel.fail_async(
                code="chat_busy",
                message="Chat is already processing",
                close_code=1008,
            )

    stt = SpeechSTTService(recognizer, sample_rate=INPUT_SAMPLE_RATE)
    tts = SpeechTTSService(synthesizer, sample_rate=OUTPUT_SAMPLE_RATE)
    transport = FastAPIWebsocketTransport(
        websocket=channel.socket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=ProtobufFrameSerializer(),
            audio_in_sample_rate=INPUT_SAMPLE_RATE,
            audio_out_sample_rate=OUTPUT_SAMPLE_RATE,
        ),
    )

    # ChatChannel must not keep reading; Pipecat owns the socket after ready.
    channel.shutdown()
    try:
        await agent.run_async(
            transport=transport,
            stt=stt,
            tts=tts,
            agent_run_repository=chat_repository,
            agent_run_session_id=chat.id,
            tool_retriever=tool_retriever,
            tool_ids=[],
            skill_retriever=skill_retriever,
            skill_ids=[],
            context=context,
        )
    finally:
        if mutex_acquired and mutex is not None:
            await mutex.release_async()
