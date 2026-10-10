"""Voice WebSocket orchestration for chat transports."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import abstractmethod, ABC
from collections.abc import AsyncIterator
from datetime import timedelta
from dataclasses import dataclass
from typing import Any

from fivcglue.interfaces.mutexes import IMutex
from fivccliche.services.interfaces.agent_chats import IUserChatRunProvider
from fivccliche.services.interfaces.agent_speeches import (
    ISpeechRecognizer,
    ISpeechSynthesizer,
    SpeechRequestError,
)
from fivccliche.services.interfaces.auth import IUser
from fivccliche.utils.chats.channels import ChatChannel
from fivccliche.utils.chats.parsers import ChatRecognizeParser, ChatRunParser
from fivccliche.utils.chats.warmup import SessionWarmup

logger = logging.getLogger(__name__)

_INPUT_SAMPLE_RATE = 16_000
_OUTPUT_SAMPLE_RATE = 24_000
_LOCK_SLACK = timedelta(seconds=10)
_UTTERANCE_FLUSH_PARTIALS = 1_000_000


@dataclass(frozen=True)
class ChatSnapshot:
    """The immutable chat values needed to run a WebSocket turn."""

    uuid: str
    agent_id: str
    context: dict[str, Any]
    description: str | None


async def _voice_frame_kind_async(proc: ChatVoiceProcessor, frame_data: str | bytes) -> str:
    if isinstance(frame_data, bytes):
        return "audio"
    try:
        frame = json.loads(frame_data)
    except json.JSONDecodeError:
        frame = None
    if not isinstance(frame, dict):
        await proc.get_channel().send_json_async(
            {
                "event": "error",
                "info": {
                    "code": "invalid_frame",
                    "message": "Invalid JSON frame",
                },
            }
        )
        return "ignore"
    frame_type = frame.get("type")
    if frame_type == "stop":
        return "end"
    if frame_type == "vad" and frame.get("event") == "speech_start":
        return "speech_start"
    if frame_type == "vad" and frame.get("event") == "speech_end":
        return "speech_end"
    return "ignore"


class ChatVoiceProcessorPhase(ABC):

    @abstractmethod
    async def parse_async(self) -> None: ...

    @abstractmethod
    async def __aenter__(self): ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb): ...


class ChatVoiceProcessorPhaseEnd(Exception):  # noqa: N818
    pass


class ChatVoiceProcessorPhaseChange(Exception):  # noqa: N818

    def __init__(self, phase: ChatVoiceProcessorPhase):
        self.phase = phase


class ChatVoiceProcessor:
    """Run a full-duplex voice session in listening or responding state."""

    class ListeningPhase(ChatVoiceProcessorPhase):
        """Feed PCM to ASR until one utterance is committed."""

        def __init__(self, proc: ChatVoiceProcessor) -> None:
            self._proc = proc
            self._asr_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
            self._asr_task: asyncio.Task[str] | None = None

        async def __aenter__(self) -> ChatVoiceProcessor.ListeningPhase:
            await self._proc.get_channel().send_json_async(
                {"event": "state", "info": {"state": "listening"}}
            )
            self._asr_queue = asyncio.Queue()
            synthesizer = self._proc.get_synthesizer()
            schedule = getattr(type(synthesizer), "schedule_preopen", None)
            if callable(schedule):
                schedule(synthesizer)
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
            await self._stop_asr_async()
            return False

        async def parse_async(self) -> None:
            proc = self._proc
            recognize_task = asyncio.create_task(self._recognize_async())
            self._asr_task = recognize_task
            receive_task: asyncio.Task[str | bytes | None] | None = None
            try:
                while not proc.get_channel().closed:
                    if receive_task is None:
                        receive_task = asyncio.create_task(proc.get_channel().receive_async())
                    done, _pending = await asyncio.wait(
                        {receive_task, recognize_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if receive_task in done:
                        frame = receive_task.result()
                        receive_task = None
                        if frame is None:
                            raise ChatVoiceProcessorPhaseEnd()
                        kind = await _voice_frame_kind_async(proc, frame)
                        if kind == "end":
                            raise ChatVoiceProcessorPhaseEnd()
                        if kind == "audio" and isinstance(frame, bytes):
                            self._asr_queue.put_nowait(frame)
                        elif kind == "speech_end":
                            self._asr_queue.put_nowait(None)
                    if recognize_task in done:
                        break
                if proc.get_channel().closed or not recognize_task.done():
                    raise ChatVoiceProcessorPhaseEnd()
                transcript = recognize_task.result()
                if not transcript:
                    await proc.get_channel().send_json_async(
                        {
                            "event": "error",
                            "info": {
                                "code": "empty_transcript",
                                "message": "Speech recognition returned empty text",
                            },
                        }
                    )
                    raise ChatVoiceProcessorPhaseChange(proc.ListeningPhase(proc))
                await proc.get_channel().send_json_async(
                    {
                        "event": "transcript",
                        "info": {"text": transcript, "is_final": True},
                    }
                )
                raise ChatVoiceProcessorPhaseChange(proc.RespondingPhase(proc, transcript))
            finally:
                if receive_task is not None and not receive_task.done():
                    receive_task.cancel()
                    await asyncio.gather(receive_task, return_exceptions=True)

        async def _recognize_async(self) -> str:
            """Collect recognition text until the caller ends the audio stream."""
            proc = self._proc
            parts: list[str] = []
            try:
                async for segment in ChatRecognizeParser(
                    empty_partials_to_flush=_UTTERANCE_FLUSH_PARTIALS,
                ).parse_async(proc.get_recognizer().stream_async(self._asr_audio_async())):
                    text = segment.strip()
                    if text:
                        parts.append(text)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Voice ASR failed chat_uuid=%s", proc._chat.uuid)
                await proc.get_channel().send_json_async(
                    {
                        "event": "error",
                        "info": {
                            "code": "asr_failed",
                            "message": "Speech recognition failed",
                        },
                    }
                )
            return "\n".join(parts)

        async def _stop_asr_async(self) -> None:
            task = self._asr_task
            self._asr_task = None
            self._asr_queue.put_nowait(None)
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)

        async def _asr_audio_async(self) -> AsyncIterator[bytes]:
            while True:
                item = await self._asr_queue.get()
                if item is None:
                    return
                yield item

    class RespondingPhase(ChatVoiceProcessorPhase):
        """Speak one agent turn, sending each parser item as it arrives."""

        def __init__(self, proc: ChatVoiceProcessor, query: str) -> None:
            self._proc = proc
            self._query = query
            self._active = True
            self._events = ChatRunParser(proc._chat.uuid, output_type="voice")
            self._turn: asyncio.Task[None] | None = None

        async def __aenter__(self) -> ChatVoiceProcessor.RespondingPhase:
            proc = self._proc
            await proc.get_channel().send_json_async(
                {"event": "state", "info": {"state": "responding"}}
            )
            if proc._mutex is not None:
                proc._mutex_acquired = await proc._mutex.acquire_async(
                    expire=proc._timeout + _LOCK_SLACK,
                    timeout=None,
                )
                if not proc._mutex_acquired:
                    proc._mutex_acquired = False
                    await proc.get_channel().send_json_async(
                        {
                            "event": "error",
                            "info": {
                                "code": "chat_busy",
                                "message": "Chat is already processing",
                            },
                        }
                    )
                    raise ChatVoiceProcessorPhaseChange(proc.ListeningPhase(proc))
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
            self._active = False
            turn = self._turn
            if turn is not None and not turn.done():
                turn.cancel()
                await asyncio.gather(turn, return_exceptions=True)
            await self._proc._release_mutex_async()
            return False

        async def parse_async(self) -> None:
            proc = self._proc
            turn = asyncio.create_task(self._speak_async())
            self._turn = turn
            receive_task: asyncio.Task[str | bytes | None] | None = None
            try:
                while not proc.get_channel().closed and not turn.done():
                    if receive_task is None:
                        receive_task = asyncio.create_task(proc.get_channel().receive_async())
                    done, _pending = await asyncio.wait(
                        {receive_task, turn},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if receive_task in done:
                        frame = receive_task.result()
                        receive_task = None
                        if frame is None:
                            await self._cancel_turn_async(turn)
                            raise ChatVoiceProcessorPhaseEnd()
                        kind = await _voice_frame_kind_async(proc, frame)
                        if kind == "end":
                            await self._cancel_turn_async(turn)
                            raise ChatVoiceProcessorPhaseEnd()
                        if kind == "speech_start":
                            self._active = False
                            await self._cancel_turn_async(turn)
                            if not proc.get_channel().closed:
                                await proc.get_channel().send_json_async(
                                    {"event": "interrupted", "info": {}}
                                )
                            raise ChatVoiceProcessorPhaseChange(proc.ListeningPhase(proc))
                if proc.get_channel().closed:
                    await self._cancel_turn_async(turn)
                    raise ChatVoiceProcessorPhaseEnd()
                await turn
                raise ChatVoiceProcessorPhaseChange(proc.ListeningPhase(proc))
            finally:
                if receive_task is not None and not receive_task.done():
                    receive_task.cancel()
                    await asyncio.gather(receive_task, return_exceptions=True)

        async def _cancel_turn_async(self, turn: asyncio.Task[None]) -> None:
            if not turn.done():
                turn.cancel()
            await asyncio.gather(turn, return_exceptions=True)

        async def _speak_async(self) -> None:
            proc = self._proc
            chat = proc._chat
            try:
                chat_run = proc.get_run_provider().create_chat_run(
                    chat.uuid,
                    user_uuid=proc.get_user().uuid,
                    agent_id=chat.agent_id,
                    context=chat.context,
                )
            except Exception:
                logger.exception("Voice agent turn failed chat_uuid=%s", chat.uuid)
                await proc.get_channel().send_json_async(
                    {
                        "event": "error",
                        "info": {
                            "code": "agent_failed",
                            "message": "Agent processing failed",
                        },
                    }
                )
                return

            started_audio = False
            try:
                async for audio in proc.get_synthesizer().stream_async(
                    self._events.parse_async(
                        chat_run.stream_async(self._query, timeout=proc._timeout)
                    )
                ):
                    if not self._active or proc.get_channel().closed:
                        return
                    if not started_audio:
                        started_audio = True
                        synthesis = proc.get_synthesizer().get_option()
                        await proc.get_channel().send_json_async(
                            {
                                "event": "audio_start",
                                "info": {
                                    "format": synthesis.format,
                                    "sample_rate": synthesis.sample_rate,
                                },
                            }
                        )
                    await proc.get_channel().send_data_async(audio)
            except asyncio.CancelledError:
                raise
            except SpeechRequestError as exc:
                logger.exception("Voice TTS failed chat_uuid=%s", chat.uuid)
                await proc.get_channel().send_json_async(
                    {
                        "event": "error",
                        "info": {
                            "code": "tts_failed",
                            "message": "Speech synthesis failed",
                            "detail": str(exc),
                        },
                    }
                )
            except Exception:
                logger.exception("Voice agent turn failed chat_uuid=%s", chat.uuid)
                await proc.get_channel().send_json_async(
                    {
                        "event": "error",
                        "info": {
                            "code": "agent_failed",
                            "message": "Agent processing failed",
                        },
                    }
                )
            if not self._active or proc.get_channel().closed:
                return
            if started_audio:
                await proc.get_channel().send_json_async({"event": "audio_end", "info": {}})
            await proc.get_channel().send_json_async(
                {
                    "event": "turn_completed",
                    "info": {"id": self._events.current_run_id},
                }
            )

    def __init__(
        self,
        channel: ChatChannel,
        *,
        user: IUser,
        chat: ChatSnapshot,
        recognizer: ISpeechRecognizer,
        synthesizer: ISpeechSynthesizer,
        run_provider: IUserChatRunProvider,
        mutex: IMutex | None,
        timeout: timedelta,
    ) -> None:
        self._channel = channel
        self._user = user
        self._chat = chat
        self._recognizer = recognizer
        self._synthesizer = synthesizer
        self._run_provider = run_provider
        self._mutex = mutex
        self._timeout = timeout

        self._phase: ChatVoiceProcessorPhase = self.ListeningPhase(self)

        self._mutex_acquired = False

    def get_recognizer(self):
        return self._recognizer

    def get_synthesizer(self):
        return self._synthesizer

    def get_run_provider(self):
        return self._run_provider

    def get_channel(self) -> ChatChannel:
        return self._channel

    def get_user(self) -> IUser:
        return self._user

    async def process_async(self) -> None:
        """Send ready, then alternate listening and responding until the socket closes."""
        from fivccliche.modules.agent_chats.services import build_chat_collaborators

        chat = self._chat
        run = self.get_run_provider().create_chat_run(
            chat.uuid,
            user_uuid=self.get_user().uuid,
            agent_id=chat.agent_id,
            context=chat.context,
        )
        try:
            async with SessionWarmup(
                chat.agent_id,
                lambda: build_chat_collaborators(run),
            ).open():
                await self._process_turns_async()
        finally:
            synthesizer = self.get_synthesizer()
            cancel = getattr(type(synthesizer), "cancel_preopen_async", None)
            if callable(cancel):
                await cancel(synthesizer)

    async def _process_turns_async(self) -> None:
        recognize = self._recognizer.get_option()
        synthesis = self._synthesizer.get_option()
        await self.get_channel().send_json_async(
            {
                "event": "ready",
                "info": {
                    "asr_id": self._recognizer.id,
                    "tts_id": self._synthesizer.id,
                    "input": {"format": recognize.format, "sample_rate": recognize.sample_rate},
                    "output": {"format": synthesis.format, "sample_rate": synthesis.sample_rate},
                },
            }
        )
        try:
            while not self.get_channel().closed:
                try:
                    async with self._phase:
                        await self._phase.parse_async()
                except ChatVoiceProcessorPhaseChange as exc:
                    self._phase = exc.phase
                except ChatVoiceProcessorPhaseEnd:
                    break
                except Exception:
                    logger.exception("Chat voice session failed chat_uuid=%s", self._chat.uuid)
                    await self.get_channel().send_json_async(
                        {
                            "event": "error",
                            "info": {
                                "code": "internal_error",
                                "message": "Voice session failed",
                            },
                        }
                    )
                    break
        finally:
            self.get_channel().shutdown()
            await self._release_mutex_async()

    async def _release_mutex_async(self) -> None:
        if self._mutex is not None and self._mutex_acquired:
            self._mutex_acquired = False
            try:
                await self._mutex.release_async()
            except Exception:
                logger.exception("Failed to release chat voice mutex")
