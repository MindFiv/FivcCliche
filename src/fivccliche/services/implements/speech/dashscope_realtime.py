"""Qwen3-ASR-Flash-Realtime via DashScope realtime WebSocket."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from typing import Any, Self

from fivcglue import IComponentSite, query_component
from fivcglue.interfaces import configs

from fivccliche.services.interfaces.speech import (
    ISpeechProvider,
    ISpeechRecognizer,
    SpeechAudioInput,
    SpeechEvent,
    SpeechRecognizeOptions,
    SpeechRequestError,
)
from fivccliche.utils.types import to_string

logger = logging.getLogger(__name__)

DEFAULT_WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com"
_DEFAULT_MODEL = "qwen3-asr-flash-realtime"

ConnectFn = Callable[[str, dict[str, str]], Awaitable[Any]]


def default_realtime_url(model: str) -> str:
    return f"{DEFAULT_WS_URL}?model={model}"


async def _connect_websockets(url: str, headers: dict[str, str]) -> Any:
    import websockets

    try:
        return await websockets.connect(url, additional_headers=headers)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers)


def _decode_b64_payload(data: str) -> bytes:
    if data.startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    return base64.b64decode(data)


class _DashScopeRealtimeSocket:
    """Private DashScope realtime websocket session (manual commit)."""

    def __init__(
        self,
        *,
        url: str,
        headers: dict[str, str],
        options: SpeechRecognizeOptions | None,
        connect: ConnectFn,
    ) -> None:
        self._url = url
        self._headers = headers
        self._options = options or SpeechRecognizeOptions()
        self._connect = connect
        self._ws: Any | None = None
        self._closed = False

    async def start_async(self) -> None:
        self._ws = await self._connect(self._url, self._headers)
        session: dict[str, Any] = {
            "input_audio_format": self._options.format,
            "turn_detection": None,
        }
        transcription: dict[str, Any] = {}
        if self._options.language:
            transcription["language"] = self._options.language
        context = self._options.context
        if self._options.hotwords:
            hotword_line = "热词: " + ", ".join(self._options.hotwords)
            context = f"{context}\n{hotword_line}" if context else hotword_line
        if context:
            transcription["prompt"] = context
        if transcription:
            session["input_audio_transcription"] = transcription
        await self._send({"type": "session.update", "session": session})
        while True:
            message = await asyncio.wait_for(self._recv_json(), timeout=10)
            kind = message.get("type")
            if kind == "session.updated":
                return
            if kind == "session.created":
                continue
            if kind == "error":
                raise SpeechRequestError(str(message.get("error") or message))

    async def send_audio_async(self, chunk: bytes) -> None:
        await self._send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("ascii"),
            }
        )

    async def commit_async(self) -> None:
        await self._send({"type": "input_audio_buffer.commit"})

    async def events(self) -> AsyncIterator[SpeechEvent]:
        while not self._closed:
            try:
                message = await self._recv_json()
            except Exception:
                if self._closed:
                    return
                raise
            event_type = message.get("type")
            if event_type == "conversation.item.input_audio_transcription.text":
                text = f"{message.get('text') or ''}{message.get('stash') or ''}"
                yield SpeechEvent(
                    type="partial",
                    text=text,
                    language=message.get("language"),
                )
            elif event_type == "conversation.item.input_audio_transcription.completed":
                text = message.get("transcript") or message.get("text") or ""
                yield SpeechEvent(
                    type="final",
                    text=text,
                    language=message.get("language"),
                )
            elif event_type == "error":
                error = message.get("error") or {}
                detail = error.get("message") if isinstance(error, dict) else str(message)
                yield SpeechEvent(type="error", message=str(detail))
            elif event_type in {"session.finished", "session.closed"}:
                return

    async def close_async(self) -> None:
        self._closed = True
        if self._ws is None:
            return
        try:
            await self._send({"type": "session.finish"})
        except Exception:
            logger.debug("Failed to send session.finish", exc_info=True)
        try:
            await self._ws.close()
        except Exception:
            logger.debug("Failed to close DashScope realtime websocket", exc_info=True)
        self._ws = None

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise SpeechRequestError("Realtime ASR session is not started")
        await self._ws.send(json.dumps(payload))

    async def _recv_json(self) -> dict[str, Any]:
        if self._ws is None:
            raise SpeechRequestError("Realtime ASR session is not started")
        raw = await self._ws.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        message = json.loads(raw)
        if not isinstance(message, dict):
            raise SpeechRequestError("Realtime ASR received a non-object frame")
        return message


class DashScopeRealtimeRecognizer(ISpeechRecognizer):
    """Streaming ASR using qwen3-asr-flash-realtime."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        ws_url: str | None = None,
        connect: ConnectFn | None = None,
        http_client: Any | None = None,
        options: SpeechRecognizeOptions | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._ws_url = ws_url or default_realtime_url(model)
        self._connect = connect or _connect_websockets
        self._http_client = http_client
        self._options = options
        self._socket: _DashScopeRealtimeSocket | None = None

    async def __aenter__(self) -> Self:
        self._socket = _DashScopeRealtimeSocket(
            url=self._ws_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
            options=self._options,
            connect=self._connect,
        )
        await self._socket.start_async()
        return self

    async def __aexit__(self, *exc: object) -> None:
        socket = self._socket
        self._socket = None
        if socket is not None:
            await socket.close_async()

    async def stream_async(
        self,
        audio: SpeechAudioInput | AsyncIterator[bytes],
    ) -> AsyncIterator[SpeechEvent]:
        socket = self._socket
        if socket is None:
            raise SpeechRequestError("Realtime ASR session is not started")
        send_task: asyncio.Task[None] | None = None
        try:

            async def _pump_in() -> None:
                async for chunk in self._chunks(audio):
                    await socket.send_audio_async(chunk)
                await socket.commit_async()

            send_task = asyncio.create_task(_pump_in())
            async for event in socket.events():
                yield event
                if event.type in {"final", "error"}:
                    return
        finally:
            if send_task is not None:
                send_task.cancel()
                with suppress(asyncio.CancelledError):
                    await send_task

    async def _chunks(
        self,
        audio: SpeechAudioInput | AsyncIterator[bytes],
    ) -> AsyncIterator[bytes]:
        if not isinstance(audio, SpeechAudioInput):
            async for chunk in audio:
                yield chunk
            return
        if audio.data_b64:
            yield _decode_b64_payload(audio.data_b64)
            return
        url = audio.url or ""
        response = await self._fetch(url)
        if not getattr(response, "is_success", False):
            status = getattr(response, "status_code", "?")
            raise SpeechRequestError(f"Failed to fetch audio URL HTTP {status}")
        content = getattr(response, "content", b"")
        if isinstance(content, bytes):
            yield content
        else:
            yield bytes(content)

    async def _fetch(self, url: str) -> Any:
        client = self._http_client
        if client is not None:
            return await client.get(url)
        import httpx

        async with httpx.AsyncClient(timeout=60.0) as http_client:
            return await http_client.get(url)


class DashScopeRealtimeSpeechProvider(ISpeechProvider):
    """ISpeechProvider that creates DashScope realtime recognizers."""

    def __init__(self, component_site: IComponentSite, **_kwargs: Any) -> None:
        self._component_site = component_site

    async def get_recognizer(
        self,
        options: SpeechRecognizeOptions | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        **_kwargs: Any,
    ) -> ISpeechRecognizer:
        api_key, model, base_url = self._credentials(
            api_key=api_key, model=model, base_url=base_url
        )
        if base_url == _DEFAULT_BASE_URL:
            ws_url = default_realtime_url(model)
        else:
            ws_base = base_url.replace("https://", "wss://").replace("http://", "ws://")
            ws_url = f"{ws_base}/api-ws/v1/realtime?model={model}"
        return DashScopeRealtimeRecognizer(
            api_key=api_key,
            model=model,
            ws_url=ws_url,
            options=options,
        )

    def _credentials(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> tuple[str, str, str]:
        config = query_component(self._component_site, configs.IConfig)
        session: configs.IConfigSession | None = None
        if config is None:
            logger.warning("IConfig component not registered; using speech defaults")
        else:
            session = config.get_session("SPEECH")
            if session is None:
                logger.warning("Config session 'SPEECH' not found; using speech defaults")
        resolved_api_key = (
            api_key
            if api_key is not None
            else to_string(session.get_value("ASR_API_KEY") if session else None, "")
        )
        resolved_model = (
            model
            if model is not None
            else to_string(session.get_value("ASR_MODEL") if session else None, _DEFAULT_MODEL)
        )
        resolved_base_url = (
            base_url
            if base_url is not None
            else to_string(
                session.get_value("ASR_BASE_URL") if session else None,
                _DEFAULT_BASE_URL,
            )
        ).rstrip("/")
        return resolved_api_key, resolved_model, resolved_base_url
