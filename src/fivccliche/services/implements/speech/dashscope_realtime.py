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
from urllib.parse import quote, urlsplit, urlunsplit
from websockets.exceptions import ConnectionClosed, InvalidStatus

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
_QWEN_TTS_PATH = "/api-ws/v1/realtime"
_DEFAULT_WS_URL = "wss://dashscope.aliyuncs.com" + _INFERENCE_PATH
_DEFAULT_BASE_URL = _DEFAULT_HTTP_ORIGIN
_DEFAULT_MODEL = "qwen-audio-3.1-asr-flash-streaming"
_TTS_DEFAULT_MODEL = "qwen-tts-realtime"
_TTS_DEFAULT_VOICE = "Cherry"
_QWEN_AUDIO_TTS_VOICE = "longanhuan_v3.1"
_QWEN_TTS_SAMPLE_RATE = 24000
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


def _qwen_tts_websocket_url(base_url: str, model: str) -> str:
    """Resolve the fixed Qwen-TTS-Realtime endpoint and include the model."""
    parsed = urlsplit(base_url)
    if parsed.scheme in {"ws", "wss"}:
        scheme = parsed.scheme
    elif parsed.scheme in {"http", "https"}:
        scheme = "ws" if parsed.scheme == "http" else "wss"
    else:
        raise SpeechRequestError(f"Unsupported DashScope base URL: {base_url}")

    netloc = parsed.netloc
    is_maas_endpoint = netloc.endswith(".maas.aliyuncs.com")
    if is_maas_endpoint:
        netloc = "dashscope.aliyuncs.com"
    path = parsed.path if parsed.path.rstrip("/") == _QWEN_TTS_PATH else _QWEN_TTS_PATH
    query = "" if is_maas_endpoint else parsed.query
    if "model=" not in query:
        query = f"model={quote(model, safe='')}"
    return urlunsplit((scheme, netloc, path, query, ""))


def _is_qwen_tts_realtime_model(model: str) -> bool:
    return model.startswith(("qwen-tts-", "qwen3-tts-"))


def _websocket_headers(api_key: str) -> dict[str, str]:
    resolved_api_key = api_key.strip()
    if resolved_api_key.casefold().startswith("bearer "):
        resolved_api_key = resolved_api_key[7:].strip()
    if not resolved_api_key:
        raise SpeechRequestError("DashScope API key is empty")

    return {"Authorization": f"Bearer {resolved_api_key}"}


def _handshake_detail(exc: InvalidStatus) -> str:
    response = exc.response
    body = getattr(response, "body", b"")
    detail = bytes(body).decode("utf-8", errors="replace").strip() if body else ""
    parsed: Any = None
    if detail:
        try:
            parsed = json.loads(detail)
        except json.JSONDecodeError:
            pass

    if isinstance(parsed, dict):
        code = parsed.get("code") or parsed.get("error_code")
        message = parsed.get("message") or parsed.get("error_message")
        if code and message:
            return f"HTTP {response.status_code} {code}: {message}"
        if message:
            return f"HTTP {response.status_code}: {message}"

    reason = getattr(response, "reason_phrase", "")
    suffix = detail or reason or str(exc)
    return f"HTTP {response.status_code}: {suffix}"


async def _connect_websockets(url: str, headers: dict[str, str]) -> Any:
    import websockets

    try:
        try:
            return await websockets.connect(url, proxy=None, additional_headers=headers)
        except TypeError:
            return await websockets.connect(url, extra_headers=headers)
    except InvalidStatus as exc:
        raise SpeechRequestError(
            f"Failed to connect to DashScope ({url}): {_handshake_detail(exc)}"
        ) from exc
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


def _qwen_tts_session_frame(options: SpeechSynthesisOptions) -> dict[str, Any]:
    return {
        "event_id": uuid.uuid4().hex,
        "type": "session.update",
        "session": {
            "voice": options.voice,
            "mode": "server_commit",
            "response_format": options.format.lower(),
            "sample_rate": options.sample_rate,
            "speech_rate": options.speech_rate,
        },
    }


def _qwen_tts_event_error(message: dict[str, Any]) -> str:
    error = message.get("error") or message
    code = error.get("code") or error.get("type") or "qwen-tts-error"
    detail = error.get("message") or error.get("detail")
    return f"{code}: {detail}" if detail else str(code)


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
        try:
            raw = await self._ws.recv()
        except ConnectionClosed as exc:
            raise SpeechRequestError(f"DashScope Qwen TTS connection closed: {exc}") from exc
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


class _DashScopeQwenTTSSocket:
    """DashScope Qwen-TTS-Realtime websocket session."""

    def __init__(
        self,
        *,
        url: str,
        headers: dict[str, str],
        options: SpeechSynthesisOptions,
        connect: _ConnectFn,
    ) -> None:
        self._url = url
        self._headers = headers
        self._options = options
        self._connect = connect
        self._ws: Any | None = None
        self._session_updated = False
        self._finish_sent = False
        self._failed = False
        self._closed = False

    async def start_async(self) -> None:
        try:
            self._ws = await self._connect(self._url, self._headers)
            created = await self._recv_json()
            if created.get("type") != "session.created":
                raise SpeechRequestError("Timed out waiting for DashScope TTS session")
            await self._send_json(_qwen_tts_session_frame(self._options))
            while not self._session_updated:
                message = await asyncio.wait_for(self._recv_json(), timeout=10)
                event = message.get("type")
                if event == "session.updated":
                    self._session_updated = True
                elif event == "error":
                    self._failed = True
                    raise SpeechRequestError(_qwen_tts_event_error(message))
        except Exception as exc:
            await self.close_async()
            if isinstance(exc, SpeechRequestError):
                raise
            if isinstance(exc, TimeoutError):
                raise SpeechRequestError("Timed out waiting for DashScope TTS session") from exc
            raise SpeechRequestError(f"Failed to start DashScope TTS session: {exc}") from exc

    async def send_text_async(self, text: str) -> None:
        await self._send_json(
            {
                "event_id": uuid.uuid4().hex,
                "type": "input_text_buffer.append",
                "text": text,
            }
        )

    async def finish_async(self) -> None:
        if self._finish_sent:
            return
        await self._send_json({"event_id": uuid.uuid4().hex, "type": "input_text_buffer.commit"})
        await self._send_json({"event_id": uuid.uuid4().hex, "type": "session.finish"})
        self._finish_sent = True

    async def recv_async(self) -> str | bytes:
        if self._ws is None:
            raise SpeechRequestError("TTS session is not started")
        try:
            raw = await self._ws.recv()
        except ConnectionClosed as exc:
            raise SpeechRequestError(f"DashScope TTS connection closed: {exc}") from exc
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
        if self._session_updated and not self._finish_sent and not self._failed:
            try:
                await self.finish_async()
            except Exception:
                _logger.debug("Failed to finish DashScope Qwen TTS session", exc_info=True)
        try:
            await self._ws.close()
        except Exception:
            _logger.debug("Failed to close DashScope Qwen TTS websocket", exc_info=True)
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
            raise SpeechRequestError("DashScope Qwen TTS received a non-object frame")
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
            headers=_websocket_headers(self._api_key),
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
        if options is not None:
            self._options = options
        elif _is_qwen_tts_realtime_model(model):
            self._options = SpeechSynthesisOptions(
                voice=_TTS_DEFAULT_VOICE,
                format="pcm",
                sample_rate=_QWEN_TTS_SAMPLE_RATE,
            )
        else:
            self._options = SpeechSynthesisOptions(voice=_QWEN_AUDIO_TTS_VOICE)
        self._connect = connect or _connect_websockets
        self._socket: _DashScopeTTSSocket | _DashScopeQwenTTSSocket | None = None
        self._stream_lock = asyncio.Lock()

    async def __aenter__(self) -> Self:
        if _is_qwen_tts_realtime_model(self._model):
            self._validate_qwen_tts_options()
            self._socket = _DashScopeQwenTTSSocket(
                url=self._ws_url,
                headers=_websocket_headers(self._api_key),
                options=self._options,
                connect=self._connect,
            )
        else:
            voice = self._options.voice
            if self._model.startswith("qwen-audio-3.1-tts-") and voice == "Cherry":
                raise SpeechRequestError(
                    f"voice {voice} is not supported by {self._model}; "
                    f"use {_QWEN_AUDIO_TTS_VOICE}"
                )
            self._socket = _DashScopeTTSSocket(
                url=self._ws_url,
                model=self._model,
                headers=_websocket_headers(self._api_key),
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
            if isinstance(socket, _DashScopeQwenTTSSocket):
                async for chunk in self._stream_qwen_tts(socket, text):
                    yield chunk
            else:
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

    async def _stream_qwen_tts(
        self,
        socket: _DashScopeQwenTTSSocket,
        text: str | AsyncIterator[str],
    ) -> AsyncIterator[bytes]:
        send_task = asyncio.create_task(self._submit_text(socket, text))
        try:
            while True:
                raw = await socket.recv_async()
                if isinstance(raw, bytes):
                    raise SpeechRequestError(
                        "DashScope Qwen TTS received an unexpected binary frame"
                    )
                message = json.loads(raw)
                if not isinstance(message, dict):
                    raise SpeechRequestError("DashScope Qwen TTS received a non-object frame")
                event = message.get("type")
                if event == "response.audio.delta":
                    delta = message.get("delta")
                    if not isinstance(delta, str):
                        raise SpeechRequestError("DashScope Qwen TTS sent an invalid audio delta")
                    yield base64.b64decode(delta)
                elif event in {"response.done", "session.finished"}:
                    return
                elif event == "error":
                    await socket.close_async(failed=True)
                    raise SpeechRequestError(_qwen_tts_event_error(message))
        finally:
            send_task.cancel()
            with suppress(asyncio.CancelledError):
                await send_task

    def _validate_qwen_tts_options(self) -> None:
        format_name = self._options.format.lower()
        if format_name not in {"pcm", "wav", "mp3", "opus"}:
            raise SpeechRequestError(
                f"audio format {self._options.format} is not supported by {self._model}"
            )
        if self._options.sample_rate != _QWEN_TTS_SAMPLE_RATE:
            raise SpeechRequestError(
                f"sample_rate {self._options.sample_rate} is not supported by {self._model}; "
                f"use {_QWEN_TTS_SAMPLE_RATE}"
            )

    async def _submit_text(
        self,
        socket: _DashScopeTTSSocket | _DashScopeQwenTTSSocket,
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
        ws_url = (
            _qwen_tts_websocket_url(base_url, model)
            if _is_qwen_tts_realtime_model(model)
            else _websocket_url(base_url)
        )
        return _DashScopeTTSSynthesizer(
            api_key=api_key,
            model=model,
            ws_url=ws_url,
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
