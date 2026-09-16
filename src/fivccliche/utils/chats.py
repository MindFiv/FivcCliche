"""Chat event adapter and WebSocket turn query (including ASR)."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import WebSocket, WebSocketDisconnect
from fivcplayground.agents import AgentRunEvent
from sqlalchemy.ext.asyncio import AsyncSession

from fivccliche.modules.agent_configs.filters import UserScopedReadableFilterSet
from fivccliche.modules.agent_configs.models import UserASR
from fivccliche.modules.agent_configs.utils import get_user_scoped_async
from fivccliche.services.interfaces.auth import IUser, IUserAuthenticator
from fivccliche.services.interfaces.speech import (
    ISpeechProvider,
    SpeechAudioInput,
    SpeechRecognizeOptions,
)
from fivccliche.utils.deps import get_speech_provider_async

logger = logging.getLogger(__name__)


class ChatAbortedError(Exception):
    """Stop ASR because the WebSocket turn already failed or disconnected."""


class ChatEventHandler:
    """Yield events from an attached agent task's event queue."""

    def __init__(self, chat_uuid: str | None = None) -> None:
        self._chat_uuid = chat_uuid
        self._chat_queue: asyncio.Queue = asyncio.Queue()
        self._asyncio_task: asyncio.Task | None = None

    def attach(self, task: asyncio.Task) -> None:
        """Watch ``task`` so ``events`` can detect completion."""
        self._asyncio_task = task

    def on_event(self, ev, run) -> None:
        """Enqueue an agent event for formatting."""
        self._chat_queue.put_nowait((ev, run))

    async def events(self) -> AsyncIterator[dict]:
        """Yield JSON-compatible event payloads from the agent event queue."""
        try:
            while True:
                if (
                    self._asyncio_task is not None
                    and self._asyncio_task.done()
                    and self._chat_queue.empty()
                ):
                    self._asyncio_task.result()
                    break

                try:
                    ev, ev_run = await asyncio.wait_for(self._chat_queue.get(), timeout=0.5)
                except TimeoutError:
                    if self._asyncio_task is None or not self._asyncio_task.done():
                        logger.debug("Timeout waiting for chat event, task still running")
                    continue

                data_fields_basics = {
                    "id",
                    "agent_id",
                    "started_at",
                    "completed_at",
                }
                data_fields = {
                    "query",
                    "reply",
                    "tool_calls",
                    *data_fields_basics,
                }
                if ev == AgentRunEvent.START:
                    info = ev_run.model_dump(mode="json", include=data_fields)
                    info.update({"chat_uuid": self._chat_uuid})
                    yield {"event": "start", "info": info}

                elif ev == AgentRunEvent.FINISH:
                    info = ev_run.model_dump(mode="json", include=data_fields)
                    info.update({"chat_uuid": self._chat_uuid})
                    yield {"event": "finish", "info": info}

                elif ev == AgentRunEvent.STREAM:
                    info = ev_run.model_dump(mode="json", include=data_fields_basics)
                    info.update(
                        {
                            "chat_uuid": self._chat_uuid,
                            "delta": (
                                ev_run.delta.model_dump(mode="json") if ev_run.delta else None
                            ),
                        }
                    )
                    yield {"event": "stream", "info": info}

                elif ev == AgentRunEvent.TOOL:
                    info = ev_run.model_dump(mode="json", include=data_fields)
                    info.update({"chat_uuid": self._chat_uuid})
                    yield {"event": "tool", "info": info}

                self._chat_queue.task_done()

        except Exception as exception:
            message = (
                "Chat message processing timed out"
                if isinstance(exception, TimeoutError)
                else str(exception)
            )
            logger.exception("Error in chat queue")
            yield {"event": "error", "info": {"message": message}}

    async def __call__(self) -> AsyncIterator[str]:
        """Yield the event queue as SSE chunks."""
        async for event in self.events():
            yield f"data: {json.dumps(event)}\n\n"


class ChatWSHandler:
    """One-shot WebSocket protocol for first-frame JWT auth and event send."""

    def __init__(self, websocket: WebSocket, *, auth_timeout: float = 5.0) -> None:
        self._ws = websocket
        self._auth_timeout = auth_timeout

    async def fail(self, *, code: str, message: str, close_code: int) -> None:
        """Send an error envelope and close the socket."""
        await self._ws.send_json({"event": "error", "info": {"code": code, "message": message}})
        await self._ws.close(code=close_code)

    async def authenticate(self, authenticator: IUserAuthenticator) -> IUser | None:
        """Accept the socket and verify a first-frame JWT.

        Returns ``None`` when the socket has already been closed with an error.
        """
        await self._ws.accept()
        try:
            auth_frame = await asyncio.wait_for(
                self._ws.receive_json(),
                timeout=self._auth_timeout,
            )
        except TimeoutError:
            await self.fail(
                code="unauthorized",
                message="Authentication timed out",
                close_code=1008,
            )
            return None
        except WebSocketDisconnect:
            await self._ws.close(code=1008)
            return None
        except (json.JSONDecodeError, ValueError):
            await self.fail(
                code="invalid_frame",
                message="Invalid JSON frame",
                close_code=1003,
            )
            return None

        if (
            not isinstance(auth_frame, dict)
            or auth_frame.get("type") != "auth"
            or not isinstance(auth_frame.get("access_token"), str)
        ):
            await self.fail(
                code="unauthorized",
                message="Invalid token",
                close_code=1008,
            )
            return None

        user = await authenticator.verify_credential_async(auth_frame["access_token"])
        if user is None:
            await self.fail(
                code="unauthorized",
                message="Invalid token",
                close_code=1008,
            )
            return None
        return user

    async def receive(
        self,
        *,
        expected_type: str | None = None,
        invalid_code: str | None = None,
    ) -> dict | bytes | None:
        """Read the next JSON object, or a binary frame when untyped.

        ``expected_type`` requires a matching JSON object and rejects bytes.
        Disconnect closes ``1000``. Invalid JSON closes ``1003``. A type
        mismatch closes ``1003`` with ``invalid_code``.
        """
        if expected_type is not None and invalid_code is None:
            raise TypeError("invalid_code is required when expected_type is set")

        try:
            message = await self._ws.receive()
        except WebSocketDisconnect:
            await self._ws.close(code=1000)
            return None

        if message.get("type") == "websocket.disconnect":
            await self._ws.close(code=1000)
            return None

        data = message.get("bytes")
        if isinstance(data, (bytes, bytearray)):
            if expected_type is not None:
                await self.fail(
                    code="invalid_frame",
                    message="Invalid JSON frame",
                    close_code=1003,
                )
                return None
            return bytes(data)

        text = message.get("text")
        if not isinstance(text, str):
            await self.fail(
                code="invalid_frame",
                message="Invalid JSON frame",
                close_code=1003,
            )
            return None
        try:
            frame = json.loads(text)
        except json.JSONDecodeError:
            await self.fail(
                code="invalid_frame",
                message="Invalid JSON frame",
                close_code=1003,
            )
            return None
        if not isinstance(frame, dict):
            await self.fail(
                code="invalid_frame",
                message="Invalid JSON frame",
                close_code=1003,
            )
            return None

        if expected_type is not None and frame.get("type") != expected_type:
            await self.fail(
                code=invalid_code or "invalid_frame",
                message=f"A {expected_type} frame is required",
                close_code=1003,
            )
            return None
        return frame

    async def send(self, events: AsyncIterator[dict]) -> bool:
        """Push JSON events. Return True if the client disconnected."""
        try:
            async for event in events:
                await self._ws.send_json(event)
        except WebSocketDisconnect:
            return True
        return False


class ChatWSQuery:
    """Resolve a text query from a WS message frame, running ASR when needed."""

    def __init__(
        self,
        chat_ws: ChatWSHandler,
        websocket: WebSocket,
        *,
        timeout: float = 30.0,
    ) -> None:
        self._chat_ws = chat_ws
        self._websocket = websocket
        self._timeout = timeout

    async def resolve(
        self,
        message_frame: dict,
        *,
        user: IUser,
        session: AsyncSession,
        chat_context: dict | None,
    ) -> str | None:
        query_value = message_frame.get("query")
        audio_value = message_frame.get("audio")
        audio_stream = message_frame.get("audio_stream")
        has_query = isinstance(query_value, str) and bool(query_value.strip())
        has_audio = isinstance(audio_value, str) and bool(audio_value.strip())
        has_stream = audio_stream is True
        if has_query + has_audio + has_stream > 1:
            await self._chat_ws.fail(
                code="invalid_message",
                message="query, audio, and audio_stream are mutually exclusive",
                close_code=1003,
            )
            return None
        if isinstance(query_value, str) and query_value.strip():
            return query_value.strip()
        if not has_audio and not has_stream:
            await self._chat_ws.fail(
                code="invalid_message",
                message="A non-empty query is required",
                close_code=1003,
            )
            return None

        asr_id = str((chat_context or {}).get("asr_id") or "default")
        asr = await get_user_scoped_async(
            session,
            UserASR,
            filters=UserScopedReadableFilterSet(
                UserASR.user_uuid, user.uuid, is_superuser=user.is_superuser
            ),
            config_id=asr_id,
        )
        speech_provider = (
            await get_speech_provider_async(asr.model_type) if asr is not None else None
        )
        if asr is None or speech_provider is None:
            await self._chat_ws.fail(
                code="speech_unavailable",
                message="Speech provider is not mounted",
                close_code=1011,
            )
            return None

        options = self._recognize_options_from_message(message_frame)
        source: SpeechAudioInput | AsyncIterator[bytes]
        if has_audio:
            source = self._audio_input_from_message(message_frame, default_format="wav")
        else:
            source = self._pcm_chunks()
        text = await self._transcribe(speech_provider, source, options, asr)
        if text is None:
            return None

        if not text or not text.strip():
            await self._chat_ws.fail(
                code="empty_transcript",
                message="Speech recognition returned empty text",
                close_code=1003,
            )
            return None
        return text.strip()

    def _audio_input_from_message(
        self, message_frame: dict, *, default_format: str
    ) -> SpeechAudioInput:
        audio_value = str(message_frame.get("audio") or "").strip()
        fmt = message_frame.get("format") or default_format
        sample_rate = message_frame.get("sample_rate") or 16000
        if not isinstance(sample_rate, int):
            try:
                sample_rate = int(sample_rate)
            except (TypeError, ValueError):
                sample_rate = 16000
        format_name = str(fmt)
        if audio_value.startswith(("http://", "https://", "oss://")):
            return SpeechAudioInput(url=audio_value, format=format_name, sample_rate=sample_rate)
        return SpeechAudioInput(data_b64=audio_value, format=format_name, sample_rate=sample_rate)

    def _recognize_options_from_message(self, message_frame: dict) -> SpeechRecognizeOptions:
        language = message_frame.get("language")
        context = message_frame.get("context")
        hotwords = message_frame.get("hotwords")
        fmt = message_frame.get("format")
        sample_rate = message_frame.get("sample_rate") or 16000
        if not isinstance(sample_rate, int):
            try:
                sample_rate = int(sample_rate)
            except (TypeError, ValueError):
                sample_rate = 16000
        return SpeechRecognizeOptions(
            language=language if isinstance(language, str) else None,
            context=context if isinstance(context, str) else None,
            hotwords=hotwords if isinstance(hotwords, list) else None,
            format=str(fmt) if fmt else "pcm",
            sample_rate=sample_rate,
        )

    async def _pcm_chunks(self) -> AsyncIterator[bytes]:
        while True:
            item = await self._chat_ws.receive()
            if item is None:
                raise ChatAbortedError
            if isinstance(item, bytes):
                yield item
                continue
            if item.get("type") == "audio_commit":
                return
            await self._chat_ws.fail(
                code="invalid_frame",
                message="An audio_commit frame is required",
                close_code=1003,
            )
            raise ChatAbortedError

    async def _transcribe(
        self,
        speech_provider: ISpeechProvider,
        source: SpeechAudioInput | AsyncIterator[bytes],
        options: SpeechRecognizeOptions,
        asr: UserASR,
    ) -> str | None:
        text = ""
        asr_error: str | None = None

        async def _consume() -> None:
            nonlocal text, asr_error
            async with await speech_provider.get_recognizer(
                options,
                api_key=asr.api_key,
                model=asr.model,
                base_url=asr.base_url,
            ) as recognizer:
                async for event in recognizer.stream_async(source):
                    if event.type in {"partial", "final"}:
                        await self._websocket.send_json(
                            {
                                "event": "transcript",
                                "info": {
                                    "text": event.text,
                                    "is_final": event.type == "final",
                                },
                            }
                        )
                        if event.type == "final":
                            text = event.text
                    elif event.type == "error":
                        asr_error = event.message or "Speech recognition failed"
                        return

        try:
            await asyncio.wait_for(_consume(), timeout=self._timeout)
        except TimeoutError:
            await self._chat_ws.fail(
                code="asr_failed",
                message="Speech recognition timed out",
                close_code=1011,
            )
            return None
        except ChatAbortedError:
            return None
        except Exception:
            logger.exception("Speech recognition failed")
            await self._chat_ws.fail(
                code="asr_failed",
                message="Speech recognition failed",
                close_code=1011,
            )
            return None

        if asr_error:
            await self._chat_ws.fail(code="asr_failed", message=asr_error, close_code=1011)
            return None
        return text


ChatStream = ChatEventHandler
