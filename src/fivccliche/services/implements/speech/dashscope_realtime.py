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


class _RealtimeEndpoint:
    """URL rules for the DashScope OpenAI-compatible realtime endpoint."""

    DEFAULT_HTTP_ORIGIN = "https://dashscope.aliyuncs.com"
    DEFAULT_WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"

    @classmethod
    def url(cls, model: str, base_url: str) -> str:
        if base_url == cls.DEFAULT_HTTP_ORIGIN:
            return f"{cls.DEFAULT_WS_URL}?model={model}"
        ws_base = base_url.replace("https://", "wss://").replace("http://", "ws://")
        return f"{ws_base}/api-ws/v1/realtime?model={model}"


DEFAULT_WS_URL = _RealtimeEndpoint.DEFAULT_WS_URL
_DEFAULT_BASE_URL = _RealtimeEndpoint.DEFAULT_HTTP_ORIGIN
_DEFAULT_MODEL = "qwen3-asr-flash-realtime"

ConnectFn = Callable[[str, dict[str, str]], Awaitable[Any]]


async def _connect_websockets(url: str, headers: dict[str, str]) -> Any:
    import websockets

    try:
        return await websockets.connect(url, additional_headers=headers)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers)


class _RealtimeFrames:
    """Qwen3 realtime request frames and response event mapping."""

    @staticmethod
    def headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

    @staticmethod
    def session_update(options: SpeechRecognizeOptions) -> dict[str, Any]:
        session: dict[str, Any] = {
            "input_audio_format": options.format,
            "turn_detection": None,
        }
        transcription: dict[str, Any] = {}
        if options.language:
            transcription["language"] = options.language

        context = options.context
        if options.hotwords:
            hotword_line = "热词: " + ", ".join(options.hotwords)
            context = f"{context}\n{hotword_line}" if context else hotword_line
        if context:
            transcription["prompt"] = context
        if transcription:
            session["input_audio_transcription"] = transcription
        return {"type": "session.update", "session": session}

    @staticmethod
    def append(chunk: bytes) -> dict[str, Any]:
        audio = base64.b64encode(chunk).decode("ascii")
        return {"type": "input_audio_buffer.append", "audio": audio}

    @staticmethod
    def commit() -> dict[str, Any]:
        return {"type": "input_audio_buffer.commit"}

    @staticmethod
    def finish() -> dict[str, Any]:
        return {"type": "session.finish"}

    @staticmethod
    def event(message: dict[str, Any]) -> SpeechEvent | None:
        event_type = message.get("type")
        if event_type == "conversation.item.input_audio_transcription.text":
            text = f"{message.get('text') or ''}{message.get('stash') or ''}"
            return SpeechEvent(
                type="partial",
                text=text,
                language=message.get("language"),
            )
        if event_type == "conversation.item.input_audio_transcription.completed":
            text = message.get("transcript") or message.get("text") or ""
            return SpeechEvent(
                type="final",
                text=text,
                language=message.get("language"),
            )
        if event_type == "error":
            error = message.get("error") or {}
            detail = error.get("message") if isinstance(error, dict) else str(message)
            return SpeechEvent(type="error", message=str(detail))
        return None

    @staticmethod
    def is_terminal(message: dict[str, Any]) -> bool:
        return message.get("type") in {"session.finished", "session.closed"}


class _RealtimeAudio:
    """Prepare clip or byte-stream input for a realtime websocket."""

    def __init__(self, http_client: Any | None = None) -> None:
        self._http_client = http_client

    async def chunks(
        self,
        audio: SpeechAudioInput | AsyncIterator[bytes],
    ) -> AsyncIterator[bytes]:
        if not isinstance(audio, SpeechAudioInput):
            async for chunk in audio:
                yield chunk
            return

        if audio.data_b64:
            data = audio.data_b64
            if data.startswith("data:") and "," in data:
                data = data.split(",", 1)[1]
            yield base64.b64decode(data)
            return

        response = await self._fetch(audio.url or "")
        if not getattr(response, "is_success", False):
            status = getattr(response, "status_code", "?")
            raise SpeechRequestError(f"Failed to fetch audio URL HTTP {status}")
        content = getattr(response, "content", b"")
        yield content if isinstance(content, bytes) else bytes(content)

    async def _fetch(self, url: str) -> Any:
        client = self._http_client
        if client is not None:
            return await client.get(url)
        import httpx

        async with httpx.AsyncClient(timeout=60.0) as http_client:
            return await http_client.get(url)


class _DashScopeRealtimeSocket:
    """DashScope realtime websocket session with manual commit."""

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
        await self._send(_RealtimeFrames.session_update(self._options))
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
        await self._send(_RealtimeFrames.append(chunk))

    async def commit_async(self) -> None:
        await self._send(_RealtimeFrames.commit())

    async def events(self) -> AsyncIterator[SpeechEvent]:
        while not self._closed:
            try:
                message = await self._recv_json()
            except Exception:
                if self._closed:
                    return
                raise

            event = _RealtimeFrames.event(message)
            if event is not None:
                yield event
            if _RealtimeFrames.is_terminal(message):
                return

    async def close_async(self) -> None:
        self._closed = True
        if self._ws is None:
            return
        try:
            await self._send(_RealtimeFrames.finish())
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
        self._ws_url = ws_url or _RealtimeEndpoint.url(model, _DEFAULT_BASE_URL)
        self._connect = connect or _connect_websockets
        self._audio = _RealtimeAudio(http_client)
        self._options = options
        self._socket: _DashScopeRealtimeSocket | None = None

    async def __aenter__(self) -> Self:
        self._socket = _DashScopeRealtimeSocket(
            url=self._ws_url,
            headers=_RealtimeFrames.headers(self._api_key),
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
                async for chunk in self._audio.chunks(audio):
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
        return DashScopeRealtimeRecognizer(
            api_key=api_key,
            model=model,
            ws_url=_RealtimeEndpoint.url(model, base_url),
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
