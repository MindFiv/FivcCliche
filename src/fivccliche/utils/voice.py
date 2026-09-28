"""Full-duplex chat voice orchestration.

The framework exposes realtime ASR/TTS sessions, but a conversation turn needs
to keep one browser socket alive across recognition, agent streaming, speech
synthesis, and barge-in.  This module owns that state machine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import timedelta
from collections.abc import AsyncIterator
from typing import Any, cast

from fastapi import WebSocket
from fivcglue.interfaces.mutexes import IMutex
from fivccliche.modules.agent_chats.filters import ChatEditableFilterSet
from fivccliche.modules.agent_chats.jobs import ChatQueryJob
from fivccliche.modules.agent_chats.models import UserChat
from fivccliche.modules.agent_chats.utils import get_chat_async
from fivccliche.modules.agent_configs.filters import UserScopedReadableFilterSet
from fivccliche.modules.agent_configs.models import UserASR, UserTTS
from fivccliche.modules.agent_configs.utils import get_user_scoped_async
from fivcglue import IComponentSite
from fivccliche.services.implements import service_site
from fivccliche.services.interfaces.auth import IUser, IUserAuthenticator
from fivccliche.services.interfaces.speech import (
    ISpeechProvider,
    SpeechRecognizeOptions,
    SpeechSynthesisOptions,
)
from fivccliche.utils.chats import ChatEventHandler, ChatWSHandler
from fivccliche.utils.deps import get_speech_provider_async
from fivcplayground.agents import AgentRunEvent
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

_REALTIME_PROVIDER = "dashscope_realtime"
_INPUT_SAMPLE_RATE = 16_000
_OUTPUT_SAMPLE_RATE = 24_000
_TURN_LOCK_EXPIRE = timedelta(seconds=900)
_BOUNDARY = re.compile(r"[。！？!?；;\n]")  # noqa: RUF001
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_EMPHASIS = re.compile(r"(`{1,3}|\*{1,3}|_{1,3}|~{1,2})")


class VoiceChatError(Exception):
    """A fatal startup error; ``code`` uses the WebSocket error contract."""

    def __init__(self, code: str, message: str, close_code: int = 1011) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.close_code = close_code


class VoiceChatEventHandler(ChatEventHandler):
    """Forward chat events and collect assistant text for speech synthesis."""

    def __init__(self, chat_uuid: str, tts_chunks: asyncio.Queue[str | None]) -> None:
        super().__init__(chat_uuid)
        self._tts_chunks = tts_chunks
        self.current_run_id: str | None = None

    def on_event(self, ev, run) -> None:
        if ev == AgentRunEvent.START:
            self.current_run_id = run.id
        if ev == AgentRunEvent.STREAM and run.delta and run.delta.text:
            clean = VoiceChatTextBuffer.clean(run.delta.text)
            if clean:
                self._tts_chunks.put_nowait(clean)
        super().on_event(ev, run)


class VoiceChatTextBuffer:
    """Accumulate streamed text into small natural synthesis chunks."""

    @staticmethod
    def clean(value: str) -> str:
        """Turn assistant display text into compact text suitable for TTS."""

        value = _MARKDOWN_IMAGE.sub("", value)
        value = _MARKDOWN_LINK.sub(r"\1", value)
        value = _MARKDOWN_EMPHASIS.sub("", value)
        value = re.sub(r"https?://\S+", "", value)
        value = re.sub(r"[|#>]", " ", value)
        return " ".join(value.split())

    def __init__(self, max_length: int = 24) -> None:
        self.max_length = max_length
        self._value = ""

    def append(self, value: str) -> list[str]:
        self._value += self.clean(value)
        return self.pop()

    def pop(self) -> list[str]:
        chunks: list[str] = []
        while self._value:
            match = _BOUNDARY.search(self._value)
            if match:
                end = match.end()
                chunks.append(self._value[:end].strip())
                self._value = self._value[end:].lstrip()
                continue
            if len(self._value) < self.max_length:
                break
            chunks.append(self._value[: self.max_length].strip())
            self._value = self._value[self.max_length :].lstrip()
        return [chunk for chunk in chunks if chunk]

    def flush(self) -> list[str]:
        chunk = self._value.strip()
        self._value = ""
        return [chunk] if chunk else []


class VoiceChatSession:
    """One WebSocket connection and its repeated voice turns."""

    def __init__(
        self,
        websocket: WebSocket,
        *,
        chat: UserChat,
        user: IUser,
        asr: UserASR,
        tts: UserTTS,
        asr_provider: ISpeechProvider,
        tts_provider: ISpeechProvider,
        mutex: IMutex | None,
        start_frame: dict[str, Any],
    ) -> None:
        self._websocket = websocket
        self._chat = chat
        self._user = user
        self._asr = asr
        self._tts = tts
        self._asr_provider = asr_provider
        self._tts_provider = tts_provider
        self._mutex = mutex
        self._start = start_frame

        self._connected = True
        self._closed = False
        self._generation = 0
        self._state = "listening"
        self._mutex_acquired = False
        self._asr_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._asr_task: asyncio.Task[None] | None = None
        self._turn_tasks: list[asyncio.Task[None]] = []
        self._tts_chunks: asyncio.Queue[str | None] = asyncio.Queue()
        self._tts_buffer = VoiceChatTextBuffer()
        self._events: VoiceChatEventHandler | None = None
        self._outbound: asyncio.Queue[tuple[int, str, Any] | None] = asyncio.Queue()
        self._sender: asyncio.Task[None] | None = None

    @property
    def state(self) -> str:
        return self._state

    async def run(self) -> None:
        self._sender = asyncio.create_task(self._send_loop())
        self._send_json({"event": "ready", "info": self._ready_info()})
        self._send_state()
        self._start_asr()

        try:
            await self._receive_loop()
        except Exception:
            logger.exception("Chat voice session failed chat_uuid=%s", self._chat.uuid)
            self._send_json(
                {
                    "event": "error",
                    "info": {
                        "code": "internal_error",
                        "message": "Voice session failed",
                    },
                }
            )
        finally:
            await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._connected = False
        self._generation += 1

        await self._cancel_turn(send_interrupted=False)
        await self._stop_asr()
        while not self._outbound.empty():
            self._outbound.get_nowait()
        self._outbound.put_nowait(None)
        if self._sender is not None:
            await asyncio.gather(self._sender, return_exceptions=True)
            self._sender = None
        if self._websocket.client_state.name == "CONNECTED":
            await self._websocket.close(code=1000)

    def _ready_info(self) -> dict[str, Any]:
        return {
            "asr_id": self._asr.id,
            "tts_id": self._tts.id,
            "input": {"format": "pcm", "sample_rate": _INPUT_SAMPLE_RATE},
            "output": {"format": "pcm", "sample_rate": _OUTPUT_SAMPLE_RATE},
        }

    def _send_json(self, payload: dict[str, Any]) -> None:
        if self._connected:
            self._outbound.put_nowait((self._generation, "json", payload))

    def _send_audio(self, audio: bytes) -> None:
        if self._connected:
            self._outbound.put_nowait((self._generation, "bytes", audio))

    def _send_state(self, state: str | None = None) -> None:
        if state is not None:
            self._state = state
        self._send_json({"event": "state", "info": {"state": self._state}})

    async def _send_loop(self) -> None:
        while True:
            item = await self._outbound.get()
            if item is None:
                return
            generation, kind, payload = item
            if generation != self._generation or not self._connected:
                continue
            if kind == "json":
                await self._websocket.send_json(payload)
            else:
                await self._websocket.send_bytes(payload)

    async def _receive_loop(self) -> None:
        while self._connected:
            message = await self._websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return

            data = message.get("bytes")
            if data is not None:
                if self._state == "listening" and self._asr_task is not None:
                    self._asr_queue.put_nowait(bytes(data))
                continue

            text = message.get("text")
            try:
                frame = json.loads(text) if isinstance(text, str) else None
            except json.JSONDecodeError:
                frame = None
            if not isinstance(frame, dict):
                self._send_json(
                    {
                        "event": "error",
                        "info": {
                            "code": "invalid_frame",
                            "message": "Invalid JSON frame",
                        },
                    }
                )
                continue

            frame_type = frame.get("type")
            if frame_type == "stop":
                return
            if frame_type == "vad" and frame.get("event") == "speech_start":
                if self._state in {"thinking", "speaking"}:
                    await self._cancel_turn()
                    self._send_state("listening")
                    self._start_asr()
                continue
            if frame_type == "vad" and frame.get("event") == "speech_end":
                self._asr_queue.put_nowait(None)

    def _start_asr(self) -> None:
        if not self._connected or self._asr_task is not None:
            return
        self._asr_queue = asyncio.Queue()
        self._asr_task = asyncio.create_task(self._recognize())

    async def _stop_asr(self) -> None:
        task = self._asr_task
        self._asr_task = None
        self._asr_queue.put_nowait(None)
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    async def _asr_audio(self) -> AsyncIterator[bytes]:
        while True:
            item = await self._asr_queue.get()
            if item is None:
                return
            yield item

    async def _recognize(self) -> None:
        transcript = ""
        try:
            options = SpeechRecognizeOptions(
                language=self._string_option("language"),
                format="pcm",
                sample_rate=_INPUT_SAMPLE_RATE,
            )
            async with await self._asr_provider.get_recognizer(
                options,
                api_key=self._asr.api_key,
                model=self._asr.model,
                base_url=self._asr.base_url,
            ) as recognizer:
                async for event in recognizer.stream_async(self._asr_audio()):
                    if event.type in {"partial", "final"}:
                        self._send_json(
                            {
                                "event": "transcript",
                                "info": {
                                    "text": event.text,
                                    "is_final": event.type == "final",
                                },
                            }
                        )
                        if event.type == "final":
                            transcript = event.text.strip()
                            break
                    elif event.type == "error":
                        raise RuntimeError(event.message or "Speech recognition failed")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Voice ASR failed chat_uuid=%s", self._chat.uuid)
            self._send_json(
                {
                    "event": "error",
                    "info": {
                        "code": "asr_failed",
                        "message": "Speech recognition failed",
                    },
                }
            )
        finally:
            self._asr_task = None

        if not self._connected:
            return
        if not transcript:
            self._send_json(
                {
                    "event": "error",
                    "info": {
                        "code": "empty_transcript",
                        "message": "Speech recognition returned empty text",
                    },
                }
            )
            self._send_state("listening")
            self._start_asr()
            return
        await self._run_turn(transcript)

    async def _run_turn(self, query: str) -> None:
        if self._mutex is not None:
            self._mutex_acquired = await self._mutex.acquire_async(
                expire=_TURN_LOCK_EXPIRE,
                timeout=None,
            )
            if not self._mutex_acquired:
                self._mutex_acquired = False
                self._send_json(
                    {
                        "event": "error",
                        "info": {
                            "code": "chat_busy",
                            "message": "Chat is already processing",
                        },
                    }
                )
                self._send_state("listening")
                self._start_asr()
                return

        generation = self._generation
        self._tts_chunks = asyncio.Queue()
        self._tts_buffer = VoiceChatTextBuffer()
        events = VoiceChatEventHandler(self._chat.uuid, self._tts_chunks)
        self._events = events
        self._send_state("thinking")

        async def run_agent() -> None:
            await ChatQueryJob(cast(IComponentSite, service_site)).run_async(
                self._chat.uuid,
                user_uuid=self._user.uuid,
                query=query,
                agent_id=self._chat.agent_id,
                context=self._chat.context,
                skills_enabled=True,
                run_timeout=300,
                event_callback=events.on_event,
            )

        async def send_agent_events() -> None:
            async for event in events.events():
                self._send_json(event)

        async def run_tts() -> None:
            started_audio = False
            try:
                options = SpeechSynthesisOptions(
                    voice=self._string_option("voice") or "",
                    format="pcm",
                    sample_rate=_OUTPUT_SAMPLE_RATE,
                )

                async def text_stream() -> AsyncIterator[str]:
                    while True:
                        item = await self._tts_chunks.get()
                        if item is None:
                            for chunk in self._tts_buffer.flush():
                                yield chunk
                            return
                        for chunk in self._tts_buffer.append(item):
                            yield chunk

                async with await self._tts_provider.get_synthesizer(
                    options,
                    api_key=self._tts.api_key,
                    model=self._tts.model,
                    base_url=self._tts.base_url,
                ) as synthesizer:
                    async for audio in synthesizer.stream_async(text_stream()):
                        if not started_audio:
                            started_audio = True
                            self._send_state("speaking")
                            self._send_json(
                                {
                                    "event": "audio_start",
                                    "info": {
                                        "format": "pcm",
                                        "sample_rate": _OUTPUT_SAMPLE_RATE,
                                    },
                                }
                            )
                        self._send_audio(audio)
                if started_audio:
                    self._send_json({"event": "audio_end", "info": {}})
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Voice TTS failed chat_uuid=%s", self._chat.uuid)
                self._send_json(
                    {
                        "event": "error",
                        "info": {
                            "code": "tts_failed",
                            "message": "Speech synthesis failed",
                        },
                    }
                )

        agent_task = asyncio.create_task(run_agent())
        events.attach(agent_task)
        events_task = asyncio.create_task(send_agent_events())
        tts_task = asyncio.create_task(run_tts())
        self._turn_tasks = [agent_task, events_task, tts_task]

        try:
            await asyncio.gather(agent_task, events_task)
            self._tts_chunks.put_nowait(None)
            await tts_task
        except Exception:
            logger.exception("Voice agent turn failed chat_uuid=%s", self._chat.uuid)
            self._send_json(
                {
                    "event": "error",
                    "info": {
                        "code": "agent_failed",
                        "message": "Agent processing failed",
                    },
                }
            )
        finally:
            for task in (agent_task, events_task, tts_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                agent_task,
                events_task,
                tts_task,
                return_exceptions=True,
            )
            await self._release_mutex()
            self._turn_tasks = []
            if self._connected and generation == self._generation:
                self._send_json(
                    {
                        "event": "turn_completed",
                        "info": {"id": events.current_run_id},
                    }
                )
                self._send_state("listening")
                self._start_asr()

    async def _cancel_turn(self, *, send_interrupted: bool = True) -> None:
        self._generation += 1
        self._state = "listening"
        tasks = list(self._turn_tasks)
        self._turn_tasks = []
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._release_mutex()
        while not self._tts_chunks.empty():
            self._tts_chunks.get_nowait()
        self._tts_chunks.put_nowait(None)
        while not self._outbound.empty():
            self._outbound.get_nowait()
        if send_interrupted and self._connected:
            self._send_json({"event": "interrupted", "info": {}})

    async def _release_mutex(self) -> None:
        if self._mutex is not None and self._mutex_acquired:
            self._mutex_acquired = False
            try:
                await self._mutex.release_async()
            except Exception:
                logger.exception("Failed to release chat voice mutex")

    def _string_option(self, key: str) -> str | None:
        value = self._start.get(key)
        return value if isinstance(value, str) and value else None


class VoiceChatWSHandler:
    """Own WebSocket startup negotiation for one full-duplex voice session."""

    def __init__(
        self,
        websocket: WebSocket,
        *,
        chat_uuid: str,
        authenticator: IUserAuthenticator,
        session: AsyncSession,
        mutex: IMutex | None,
    ) -> None:
        self._websocket = websocket
        self._chat_uuid = chat_uuid
        self._authenticator = authenticator
        self._session = session
        self._mutex = mutex

    async def run(self) -> None:
        """Authenticate, negotiate realtime providers, and run voice turns."""

        chat_ws = ChatWSHandler(self._websocket)
        user = await chat_ws.authenticate(self._authenticator)
        if user is None:
            return
        start_frame = await chat_ws.receive(
            expected_type="start",
            invalid_code="invalid_message",
        )
        if not isinstance(start_frame, dict):
            return

        chat = await get_chat_async(
            self._session,
            self._chat_uuid,
            filters=ChatEditableFilterSet(user.uuid, is_superuser=user.is_superuser),
        )
        if chat is None:
            await self._session.close()
            await chat_ws.fail(code="chat_not_found", message="Chat not found", close_code=4404)
            return

        try:
            asr, tts, asr_provider, tts_provider = await self._resolve_voice_configs(
                user=user,
                chat=chat,
                start_frame=start_frame,
            )
        except VoiceChatError as exc:
            await self._session.close()
            await chat_ws.fail(code=exc.code, message=exc.message, close_code=exc.close_code)
            return

        await self._session.close()
        voice = VoiceChatSession(
            self._websocket,
            chat=chat,
            user=user,
            asr=asr,
            tts=tts,
            asr_provider=asr_provider,
            tts_provider=tts_provider,
            mutex=self._mutex,
            start_frame=start_frame,
        )
        await voice.run()

    async def _resolve_voice_configs(
        self,
        *,
        user: IUser,
        chat: UserChat,
        start_frame: dict[str, Any],
    ) -> tuple[UserASR, UserTTS, ISpeechProvider, ISpeechProvider]:
        """Resolve user-visible realtime speech configs and mounted providers."""

        context = chat.context or {}
        asr_id = start_frame.get("asr_id") or context.get("asr_id") or "realtime"
        tts_id = start_frame.get("tts_id") or context.get("tts_id") or "realtime"
        if not isinstance(asr_id, str) or not isinstance(tts_id, str):
            raise VoiceChatError("invalid_frame", "asr_id and tts_id must be strings", 1003)

        asr_filters = UserScopedReadableFilterSet(
            UserASR.user_uuid, user.uuid, is_superuser=user.is_superuser
        )
        tts_filters = UserScopedReadableFilterSet(
            UserTTS.user_uuid, user.uuid, is_superuser=user.is_superuser
        )
        asr = await get_user_scoped_async(
            self._session, UserASR, filters=asr_filters, config_id=asr_id
        )
        tts = await get_user_scoped_async(
            self._session, UserTTS, filters=tts_filters, config_id=tts_id
        )
        if asr is None:
            raise VoiceChatError("asr_not_found", "ASR config not found", 4404)
        if tts is None:
            raise VoiceChatError("tts_not_found", "TTS config not found", 4404)
        if asr.model_type != _REALTIME_PROVIDER:
            raise VoiceChatError(
                "invalid_asr_model_type", "ASR config must use dashscope_realtime", 1003
            )
        if tts.model_type != _REALTIME_PROVIDER:
            raise VoiceChatError(
                "invalid_tts_model_type", "TTS config must use dashscope_realtime", 1003
            )

        asr_provider = await get_speech_provider_async(asr.model_type)
        tts_provider = await get_speech_provider_async(tts.model_type)
        if asr_provider is None or tts_provider is None:
            raise VoiceChatError("speech_unavailable", "Speech provider is not mounted", 1011)
        return asr, tts, asr_provider, tts_provider


__all__ = [
    "VoiceChatError",
    "VoiceChatEventHandler",
    "VoiceChatSession",
    "VoiceChatTextBuffer",
    "VoiceChatWSHandler",
]
