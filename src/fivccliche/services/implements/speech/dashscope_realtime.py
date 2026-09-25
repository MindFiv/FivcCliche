"""DashScope realtime ASR and TTS."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from typing import Any, Literal, Self
from urllib.parse import urlsplit, urlunsplit

from fivcglue import IComponentSite, query_component
from fivcglue.interfaces import configs

from fivccliche.services.interfaces.speech import (
    ISpeechProvider,
    ISpeechRecognizer,
    ISpeechSynthesizer,
    SpeechAudioInput,
    SpeechEvent,
    SpeechRecognizeOptions,
    SpeechRequestError,
    SpeechEventType,
    SpeechSynthesisOptions,
)
from fivccliche.utils.types import to_string

_logger = logging.getLogger(__name__)

_DEFAULT_HTTP_ORIGIN = "https://dashscope.aliyuncs.com"
_INFERENCE_PATH = "/api-ws/v1/inference"
_DEFAULT_WS_URL = "wss://dashscope.aliyuncs.com" + _INFERENCE_PATH
_DEFAULT_BASE_URL = _DEFAULT_HTTP_ORIGIN
_DEFAULT_MODEL = "qwen-audio-3.1-asr-flash-streaming"
_TTS_DEFAULT_MODEL = "qwen-audio-3.1-tts-flash"
_TTS_DEFAULT_VOICE = "Cherry"
_AUDIO_CHUNK_SIZE = 12800

_ConnectFn = Callable[[str, dict[str, str]], Awaitable[Any]]


def _websocket_url(base_url: str) -> str:
    """Resolve an HTTP or WebSocket origin to the DashScope inference URL."""
    parsed = urlsplit(base_url)
    if parsed.scheme in {"ws", "wss"}:
        if parsed.path in {"", "/"} and not parsed.query:
            return urlunsplit((parsed.scheme, parsed.netloc, _INFERENCE_PATH, "", ""))
        return base_url

    if parsed.scheme in {"http", "https"}:
        scheme = "ws" if parsed.scheme == "http" else "wss"
        if parsed.path.rstrip("/") == _INFERENCE_PATH:
            return urlunsplit((scheme, parsed.netloc, parsed.path, parsed.query, ""))
        return urlunsplit((scheme, parsed.netloc, _INFERENCE_PATH, "", ""))

    raise SpeechRequestError(f"Unsupported DashScope base URL: {base_url}")


async def _connect_websockets(url: str, headers: dict[str, str]) -> Any:
    import websockets

    try:
        try:
            return await websockets.connect(url, additional_headers=headers)
        except TypeError:
            return await websockets.connect(url, extra_headers=headers)
    except Exception as exc:
        raise SpeechRequestError(f"Failed to connect to DashScope ({url}): {exc}") from exc


def _tts_error_message(message: Any) -> str:
    if isinstance(message, str):
        try:
            parsed_message = json.loads(message)
        except json.JSONDecodeError:
            return message
        message = parsed_message
    if not isinstance(message, dict):
        return f"{message}"
    header = message.get("header") or {}
    code = (
        header.get("error_code")
        or message.get("code")
        or message.get("error_code")
        or "task-failed"
    )
    detail = header.get("error_message") or message.get("message") or message.get("error_message")
    return f"{code}: {detail}" if detail else str(code)


def _tts_start_frame(
    model: str,
    options: SpeechSynthesisOptions,
    task_id: str,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "voice": options.voice,
        "volume": options.volume,
        "text_type": "PlainText",
        "sample_rate": options.sample_rate,
        "rate": options.speech_rate,
        "format": options.format.lower(),
        "pitch": options.pitch_rate,
        "seed": 0,
        "type": 0,
    }
    parameters.update(options.extra)
    return {
        "header": {
            "action": "run-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {
            "model": model,
            "task_group": "audio",
            "task": "tts",
            "function": "SpeechSynthesizer",
            "input": {},
            "parameters": parameters,
        },
    }


def _tts_continue_frame(
    model: str,
    text: str,
    task_id: str,
) -> dict[str, Any]:
    return {
        "header": {
            "action": "continue-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {
            "model": model,
            "task_group": "audio",
            "task": "tts",
            "function": "SpeechSynthesizer",
            "input": {"text": text},
        },
    }


def _start_frame(
    model: str,
    options: SpeechRecognizeOptions,
    task_id: str,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "format": options.format.lower(),
        "sample_rate": options.sample_rate,
    }
    parameters.update(options.extra)
    return {
        "header": {
            "action": "run-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {
            "model": model,
            "task_group": "audio",
            "task": "recognition",
            "function": "recognition",
            "parameters": parameters,
            "input": {},
        },
    }


def _finish_frame(task_id: str) -> dict[str, Any]:
    return {
        "header": {
            "action": "finish-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {"input": {}},
    }


def _message_discriminator(message: dict[str, Any]) -> str | None:
    header = message.get("header")
    return header.get("event") if isinstance(header, dict) else None


def _message_event(message: dict[str, Any], discriminator: str | None) -> SpeechEvent | None:
    if discriminator == "result-generated":
        payload = message.get("payload") or {}
        sentence = (payload.get("output") or {}).get("sentence") or {}
        if not isinstance(sentence, dict):
            raise SpeechRequestError("DashScope ASR sent an invalid sentence")
        event_type: SpeechEventType = "final" if sentence.get("end_time") is not None else "partial"
        return SpeechEvent(type=event_type, text=str(sentence.get("text") or ""))

    if discriminator == "task-failed":
        header = message.get("header") or {}
        code = header.get("error_code") or "task-failed"
        detail = header.get("error_message")
        return SpeechEvent(
            type="error",
            message=f"{code}: {detail}" if detail else str(code),
        )
    return None


async def _audio_data(audio: SpeechAudioInput, http_client: Any | None) -> bytes:
    if audio.data_b64:
        data = audio.data_b64
        if data.startswith("data:") and "," in data:
            data = data.split(",", 1)[1]
        return base64.b64decode(data)

    if http_client is not None:
        response = await http_client.get(audio.url or "")
    else:
        import httpx

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(audio.url or "")

    if not getattr(response, "is_success", False):
        status = getattr(response, "status_code", "?")
        raise SpeechRequestError(f"Failed to fetch audio URL HTTP {status}")
    content = getattr(response, "content", b"")
    return content if isinstance(content, bytes) else bytes(content)


async def _audio_chunks(
    audio: SpeechAudioInput | AsyncIterator[bytes],
    http_client: Any | None,
) -> AsyncIterator[bytes]:
    if isinstance(audio, SpeechAudioInput):
        data = await _audio_data(audio, http_client)
    else:
        data = bytearray()
        async for chunk in audio:
            data.extend(chunk)

    for offset in range(0, len(data), _AUDIO_CHUNK_SIZE):
        yield bytes(data[offset : offset + _AUDIO_CHUNK_SIZE])


class _DashScopeRecognitionSocket:
    """DashScope Recognition websocket session with manual completion."""

    def __init__(
        self,
        *,
        url: str,
        model: str,
        headers: dict[str, str],
        options: SpeechRecognizeOptions | None,
        connect: _ConnectFn,
    ) -> None:
        self._url = url
        self._model = model
        self._headers = headers
        self._options = options or SpeechRecognizeOptions()
        self._connect = connect
        self._ws: Any | None = None
        self._task_id = uuid.uuid4().hex
        self._started = False
        self._finish_sent = False
        self._closed = False

    async def start_async(self) -> None:
        try:
            self._ws = await self._connect(self._url, self._headers)
            await self._send_json(_start_frame(self._model, self._options, self._task_id))
            while True:
                message = await asyncio.wait_for(self._recv_json(), timeout=10)
                discriminator = _message_discriminator(message)
                if discriminator == "task-started":
                    self._started = True
                    return
                if discriminator == "task-failed":
                    event = _message_event(message, discriminator)
                    raise SpeechRequestError(event.message if event else "ASR task failed")
                if discriminator != "result-generated":
                    raise SpeechRequestError(
                        f"Unexpected ASR frame: {discriminator or 'missing event'}"
                    )
        except Exception as exc:
            await self.close_async()
            if isinstance(exc, SpeechRequestError):
                raise
            if isinstance(exc, TimeoutError):
                raise SpeechRequestError("Timed out waiting for DashScope ASR task") from exc
            raise SpeechRequestError(f"Failed to start DashScope ASR task: {exc}") from exc

    async def send_audio_async(self, chunk: bytes) -> None:
        if self._ws is None:
            raise SpeechRequestError("Realtime ASR session is not started")
        await self._ws.send_bytes(chunk)

    async def commit_async(self) -> None:
        if not self._finish_sent:
            await self._send_json(_finish_frame(self._task_id))
            self._finish_sent = True

    async def events(self) -> AsyncIterator[SpeechEvent]:
        saw_final = False
        while not self._closed:
            try:
                message = await self._recv_json()
            except Exception:
                if self._closed:
                    return
                raise

            discriminator = _message_discriminator(message)
            if discriminator not in {"result-generated", "task-finished", "task-failed"}:
                raise SpeechRequestError(
                    f"Unexpected ASR frame: {discriminator or 'missing event'}"
                )

            event = _message_event(message, discriminator)
            if event is not None:
                saw_final = saw_final or event.type == "final"
                yield event
            if discriminator == "task-failed":
                return
            if discriminator == "task-finished":
                if not saw_final:
                    yield SpeechEvent(type="final")
                return

    async def close_async(self) -> None:
        self._closed = True
        if self._ws is None:
            return
        if self._started and not self._finish_sent:
            try:
                await self.commit_async()
            except Exception:
                _logger.debug("Failed to send DashScope finish-task", exc_info=True)
        try:
            await self._ws.close()
        except Exception:
            _logger.debug("Failed to close DashScope ASR websocket", exc_info=True)
        self._ws = None

    async def _send_json(self, payload: dict[str, Any]) -> None:
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
            raise SpeechRequestError("DashScope ASR received a non-object frame")
        return message


class _DashScopeTTSSocket:
    """DashScope TTS websocket session with manual text submission."""

    def __init__(
        self,
        *,
        url: str,
        model: str,
        headers: dict[str, str],
        options: SpeechSynthesisOptions,
        connect: _ConnectFn,
    ) -> None:
        self._url = url
        self._model = model
        self._headers = headers
        self._options = options
        self._connect = connect
        self._ws: Any | None = None
        self._task_id = uuid.uuid4().hex
        self._started = False
        self._finish_sent = False
        self._failed = False
        self._closed = False

    async def start_async(self) -> None:
        try:
            self._ws = await self._connect(self._url, self._headers)
            await self._send_json(_tts_start_frame(self._model, self._options, self._task_id))
            while True:
                message = await asyncio.wait_for(self._recv_json(), timeout=10)
                discriminator = _message_discriminator(message)
                if discriminator == "task-started":
                    self._started = True
                    return
                if discriminator == "task-failed":
                    self._failed = True
                    raise SpeechRequestError(_tts_error_message(message))
                raise SpeechRequestError(
                    f"Unexpected TTS frame: {discriminator or 'missing event'}"
                )
        except Exception as exc:
            await self.close_async()
            if isinstance(exc, SpeechRequestError):
                raise
            if isinstance(exc, TimeoutError):
                raise SpeechRequestError("Timed out waiting for DashScope TTS task") from exc
            raise SpeechRequestError(f"Failed to start DashScope TTS task: {exc}") from exc

    async def send_text_async(self, text: str) -> None:
        if self._ws is None:
            raise SpeechRequestError("TTS session is not started")
        await self._send_json(_tts_continue_frame(self._model, text, self._task_id))

    async def finish_async(self) -> None:
        if not self._finish_sent:
            await self._send_json(_finish_frame(self._task_id))
            self._finish_sent = True

    async def recv_async(self) -> str | bytes:
        if self._ws is None:
            raise SpeechRequestError("TTS session is not started")
        raw = await self._ws.recv()
        if isinstance(raw, bytearray):
            return bytes(raw)
        if isinstance(raw, str):
            return raw
        return bytes(raw)

    async def close_async(self, *, failed: bool = False) -> None:
        self._closed = True
        self._failed = self._failed or failed
        if self._ws is None:
            return
        if self._started and not self._finish_sent and not self._failed:
            try:
                await self.finish_async()
            except Exception:
                _logger.debug("Failed to send DashScope finish-task", exc_info=True)
        try:
            await self._ws.close()
        except Exception:
            _logger.debug("Failed to close DashScope TTS websocket", exc_info=True)
        self._ws = None

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise SpeechRequestError("TTS session is not started")
        await self._ws.send(json.dumps(payload))

    async def _recv_json(self) -> dict[str, Any]:
        raw = await self.recv_async()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        message = json.loads(raw)
        if not isinstance(message, dict):
            raise SpeechRequestError("DashScope TTS received a non-object frame")
        return message


class _DashScopeRealtimeRecognizer(ISpeechRecognizer):
    """Streaming ASR using the DashScope Recognition protocol."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        ws_url: str | None = None,
        connect: _ConnectFn | None = None,
        http_client: Any | None = None,
        options: SpeechRecognizeOptions | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._ws_url = ws_url or _websocket_url(_DEFAULT_BASE_URL)
        self._connect = connect or _connect_websockets
        self._http_client = http_client
        self._options = options
        self._socket: _DashScopeRecognitionSocket | None = None

    async def __aenter__(self) -> Self:
        self._socket = _DashScopeRecognitionSocket(
            url=self._ws_url,
            model=self._model,
            headers={"Authorization": f"Bearer {self._api_key}"},
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
                async for chunk in _audio_chunks(audio, self._http_client):
                    await socket.send_audio_async(chunk)
                await socket.commit_async()

            send_task = asyncio.create_task(_pump_in())
            async for event in socket.events():
                yield event
                if event.type == "error":
                    return
        finally:
            if send_task is not None:
                send_task.cancel()
                with suppress(asyncio.CancelledError):
                    await send_task


class _DashScopeTTSSynthesizer(ISpeechSynthesizer):
    """Streaming synthesizer using the DashScope TTS websocket protocol."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = _TTS_DEFAULT_MODEL,
        ws_url: str = _DEFAULT_WS_URL,
        options: SpeechSynthesisOptions | None,
        connect: _ConnectFn | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._ws_url = ws_url
        self._options = options or SpeechSynthesisOptions(voice=_TTS_DEFAULT_VOICE)
        self._connect = connect or _connect_websockets
        self._socket: _DashScopeTTSSocket | None = None
        self._stream_lock = asyncio.Lock()

    async def __aenter__(self) -> Self:
        self._socket = _DashScopeTTSSocket(
            url=self._ws_url,
            model=self._model,
            headers={"Authorization": f"Bearer {self._api_key}"},
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

    async def stream_async(self, text: str | AsyncIterator[str]) -> AsyncIterator[bytes]:
        socket = self._socket
        if socket is None:
            raise SpeechRequestError("TTS session is not started")
        if self._stream_lock.locked():
            raise SpeechRequestError("TTS session is already streaming")

        async with self._stream_lock:
            send_task = asyncio.create_task(self._submit_text(socket, text))
            try:
                while True:
                    raw = await socket.recv_async()
                    if isinstance(raw, bytes):
                        yield raw
                        continue

                    message = json.loads(raw)
                    if not isinstance(message, dict):
                        raise SpeechRequestError("DashScope TTS received a non-object frame")
                    discriminator = _message_discriminator(message)
                    if discriminator == "task-finished":
                        return
                    if discriminator == "task-failed":
                        await socket.close_async(failed=True)
                        raise SpeechRequestError(_tts_error_message(message))
                    if discriminator not in {"result-generated", "task-started"}:
                        raise SpeechRequestError(
                            f"Unexpected TTS frame: {discriminator or 'missing event'}"
                        )
            finally:
                if send_task is not None:
                    send_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await send_task

    async def _submit_text(
        self,
        socket: _DashScopeTTSSocket,
        text: str | AsyncIterator[str],
    ) -> None:
        if isinstance(text, str):
            await socket.send_text_async(text)
        else:
            async for chunk in text:
                await socket.send_text_async(chunk)
        await socket.finish_async()


class DashScopeRealtimeSpeechProvider(ISpeechProvider):
    """ISpeechProvider that creates DashScope realtime ASR and TTS sessions."""

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
            capability="asr", api_key=api_key, model=model, base_url=base_url
        )
        return _DashScopeRealtimeRecognizer(
            api_key=api_key,
            model=model,
            ws_url=_websocket_url(base_url),
            options=options,
        )

    async def get_synthesizer(
        self,
        options: SpeechSynthesisOptions | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        **_kwargs: Any,
    ) -> ISpeechSynthesizer:
        api_key, model, base_url = self._credentials(
            capability="tts", api_key=api_key, model=model, base_url=base_url
        )
        return _DashScopeTTSSynthesizer(
            api_key=api_key,
            model=model,
            ws_url=_websocket_url(base_url),
            options=options,
        )

    def _credentials(
        self,
        *,
        capability: Literal["asr", "tts"],
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> tuple[str, str, str]:
        config = query_component(self._component_site, configs.IConfig)
        session: configs.IConfigSession | None = None
        if config is None:
            _logger.warning("IConfig component not registered; using speech defaults")
        else:
            session = config.get_session("SPEECH")
            if session is None:
                _logger.warning("Config session 'SPEECH' not found; using speech defaults")

        is_tts = capability == "tts"
        api_key_name = "TTS_API_KEY" if is_tts else "ASR_API_KEY"
        model_name = "TTS_MODEL" if is_tts else "ASR_MODEL"
        base_url_name = "TTS_BASE_URL" if is_tts else "ASR_BASE_URL"
        default_model = _TTS_DEFAULT_MODEL if is_tts else _DEFAULT_MODEL
        default_base_url = _DEFAULT_HTTP_ORIGIN if is_tts else _DEFAULT_BASE_URL

        resolved_api_key = (
            api_key
            if api_key is not None
            else to_string(
                session.get_value(api_key_name) if session else None,
                "",
            )
        )
        resolved_model = (
            model
            if model is not None
            else to_string(
                session.get_value(model_name) if session else None,
                default_model,
            )
        )
        resolved_base_url = (
            base_url
            if base_url is not None
            else to_string(
                session.get_value(base_url_name) if session else None,
                default_base_url,
            )
        ).rstrip("/")
        return resolved_api_key, resolved_model, resolved_base_url
