"""DashScope streaming ASR and TTS via the DashScope SDK."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import AsyncIterator
from typing import Any, Literal, Self
from urllib.parse import urlsplit, urlunsplit

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
from dashscope.audio.tts_v2 import AudioFormat as TTSV2AudioFormat
from dashscope.audio.tts_v2 import ResultCallback
from dashscope.audio.tts_v2 import SpeechSynthesizer as DashScopeTTSV2Client
from fivcglue import IComponentSite
from fivcplayground.speeches import (
    SpeechError,
    SpeechRecognizeEvent,
    SpeechRecognizeOption,
    SpeechRecognizer,
    SpeechSynthesisOption,
    SpeechSynthesizer,
)

from fivccliche.services.interfaces.agent_speeches import ISpeechProvider

logger = logging.getLogger(__name__)

_DEFAULT_HTTP_ORIGIN = "https://dashscope.aliyuncs.com"
_INFERENCE_PATH = "/api-ws/v1/inference"
_DEFAULT_MODEL = "qwen-audio-3.1-asr-flash-streaming"
_TTS_DEFAULT_MODEL = "qwen-audio-3.1-tts-flash"
_TTS_DEFAULT_VOICE = "longanhuan_v3.1"
# 100ms of 16-bit mono PCM at 16 kHz.
_DEFAULT_PCM_FRAME_BYTES = 3200
_SDK_KEY_LOCK = threading.Lock()


def _websocket_url(base_url: str) -> str:
    """Resolve an HTTP origin or WebSocket URL to the DashScope inference socket."""
    parsed = urlsplit(base_url.strip())
    if parsed.scheme in {"ws", "wss"}:
        path = _INFERENCE_PATH if parsed.path in {"", "/"} else parsed.path
        return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))
    if parsed.scheme in {"http", "https"}:
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return urlunsplit((scheme, parsed.netloc, _INFERENCE_PATH, "", ""))
    raise SpeechError(f"Unsupported DashScope base URL: {base_url}")


def _strip_wav_header(payload: bytes) -> bytes:
    if len(payload) >= 44 and payload[:4] == b"RIFF" and payload[8:12] == b"WAVE":
        return payload[44:]
    return payload


def _extract_recognition_text(result: RecognitionResult) -> str:
    sentence = result.get_sentence()
    if isinstance(sentence, dict):
        return sentence.get("text", "") or ""
    if isinstance(sentence, list):
        return " ".join(item.get("text", "") for item in sentence if item.get("text"))
    return ""


def _iter_pcm_frames(payload: bytes, frame_bytes: int = _DEFAULT_PCM_FRAME_BYTES):
    data = _strip_wav_header(payload)
    if not data:
        return
    for offset in range(0, len(data), frame_bytes):
        yield data[offset : offset + frame_bytes]


def _dashscope_audio_format(sample_rate: int) -> TTSV2AudioFormat:
    mapping = {
        8000: TTSV2AudioFormat.PCM_8000HZ_MONO_16BIT,
        16000: TTSV2AudioFormat.PCM_16000HZ_MONO_16BIT,
        22050: TTSV2AudioFormat.PCM_22050HZ_MONO_16BIT,
        24000: TTSV2AudioFormat.PCM_24000HZ_MONO_16BIT,
        44100: TTSV2AudioFormat.PCM_44100HZ_MONO_16BIT,
        48000: TTSV2AudioFormat.PCM_48000HZ_MONO_16BIT,
    }
    try:
        return mapping[sample_rate]
    except KeyError as exc:
        raise SpeechError(f"DashScope TTS PCM does not support sample_rate={sample_rate}") from exc


def _with_default_voice(option: SpeechSynthesisOption | None) -> SpeechSynthesisOption:
    if option is None:
        return SpeechSynthesisOption(
            voice=_TTS_DEFAULT_VOICE,
            format="pcm",
            sample_rate=24000,
        )
    if option.voice:
        return option
    return option.model_copy(update={"voice": _TTS_DEFAULT_VOICE})


class _DashScopeRealtimeRecognizer(SpeechRecognizer):
    """DashScope streaming ASR via ``Recognition.send_audio_frame``."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        ws_url: str,
        options: SpeechRecognizeOption | None = None,
        speech_id: str = "",
    ) -> None:
        if not api_key:
            raise SpeechError("DashScope API key is empty")
        self._id = speech_id
        self._api_key = api_key
        self._model = model
        self._ws_url = ws_url
        self._option = options or SpeechRecognizeOption(format="pcm", sample_rate=16000)
        self._entered = False
        self._utt_audio_queue: queue.Queue[bytes | None] | None = None
        self._utt_event_queue: asyncio.Queue[SpeechRecognizeEvent | None | BaseException] | None = (
            None
        )
        self._utt_worker: threading.Thread | None = None
        self._utt_last_text_box: dict[str, str] = {"text": ""}

    @property
    def id(self) -> str:
        return self._id

    def get_option(self) -> SpeechRecognizeOption:
        return self._option

    async def __aenter__(self) -> Self:
        self._entered = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._utt_worker is not None:
            await self.cancel_utterance()
        self._entered = False

    def _require_entered(self) -> None:
        if not self._entered:
            raise SpeechError(f"{type(self).__name__} must be entered before recognition")

    async def begin_utterance(self) -> None:
        """Open a DashScope ASR session (call when the user starts speaking)."""
        self._require_entered()
        if self._utt_worker is not None:
            await self.cancel_utterance()

        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue[SpeechRecognizeEvent | None | BaseException] = asyncio.Queue()
        audio_queue: queue.Queue[bytes | None] = queue.Queue()
        opened = threading.Event()
        last_text_box = {"text": ""}
        closed = {"done": False}

        class _Callback(RecognitionCallback):
            def on_open(self) -> None:
                opened.set()

            def on_event(self, result: RecognitionResult) -> None:
                sentence = result.get_sentence()
                text = _extract_recognition_text(result).strip()
                if not text:
                    return
                last_text_box["text"] = text
                is_final = isinstance(sentence, dict) and bool(
                    RecognitionResult.is_sentence_end(sentence)
                )
                event = SpeechRecognizeEvent(
                    type="final" if is_final else "partial",
                    text=text,
                )
                loop.call_soon_threadsafe(event_queue.put_nowait, event)

            def on_error(self, result: RecognitionResult) -> None:
                loop.call_soon_threadsafe(
                    event_queue.put_nowait,
                    SpeechRecognizeEvent(type="error", message=str(result)),
                )
                loop.call_soon_threadsafe(event_queue.put_nowait, None)

            def on_complete(self) -> None:
                loop.call_soon_threadsafe(event_queue.put_nowait, None)

            def on_close(self) -> None:
                if not closed["done"]:
                    closed["done"] = True
                    loop.call_soon_threadsafe(event_queue.put_nowait, None)

        def _worker() -> None:
            try:
                recognition = Recognition(
                    model=self._model,
                    callback=_Callback(),
                    format=self._option.format,
                    sample_rate=self._option.sample_rate,
                    api_key=self._api_key,
                    base_address=self._ws_url,
                )
                recognition.start()
                if not opened.wait(timeout=10):
                    raise TimeoutError("DashScope ASR websocket open timed out")
                try:
                    while True:
                        frame = audio_queue.get()
                        if frame is None:
                            break
                        recognition.send_audio_frame(frame)
                finally:
                    recognition.stop()
            except BaseException as exc:
                loop.call_soon_threadsafe(event_queue.put_nowait, exc)
                loop.call_soon_threadsafe(event_queue.put_nowait, None)

        worker = threading.Thread(target=_worker, name="dashscope-asr", daemon=True)
        self._utt_audio_queue = audio_queue
        self._utt_event_queue = event_queue
        self._utt_worker = worker
        self._utt_last_text_box = last_text_box
        worker.start()
        await asyncio.to_thread(opened.wait, 10)
        if not opened.is_set():
            await self.cancel_utterance()
            raise SpeechError("DashScope ASR websocket open timed out")

    async def push_audio(self, chunk: bytes) -> None:
        """Send one PCM (or wav) chunk into the active utterance."""
        if self._utt_audio_queue is None:
            raise SpeechError("begin_utterance() must be called before push_audio()")
        if not chunk:
            return
        payload = _strip_wav_header(chunk) if self._option.format == "pcm" else chunk
        if payload:
            self._utt_audio_queue.put(payload)

    async def end_utterance(self) -> AsyncIterator[SpeechRecognizeEvent]:
        """Finish the utterance and yield recognition events."""
        if self._utt_audio_queue is None or self._utt_event_queue is None:
            raise SpeechError("begin_utterance() must be called before end_utterance()")

        self._utt_audio_queue.put(None)
        saw_final = False
        try:
            while True:
                item = await self._utt_event_queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    if isinstance(item, SpeechError):
                        raise item
                    yield SpeechRecognizeEvent(type="error", message=str(item))
                    return
                if item.type == "final":
                    saw_final = True
                yield item

            last_text = self._utt_last_text_box.get("text", "")
            if not saw_final:
                if last_text:
                    yield SpeechRecognizeEvent(type="final", text=last_text)
                else:
                    logger.warning("empty transcription from DashScope")
                    yield SpeechRecognizeEvent(type="final", text="")
        finally:
            if self._utt_worker is not None:
                await asyncio.to_thread(self._utt_worker.join, 30)
            self._utt_audio_queue = None
            self._utt_event_queue = None
            self._utt_worker = None

    async def cancel_utterance(self) -> None:
        """Abort an in-flight utterance without yielding events."""
        if self._utt_audio_queue is not None:
            self._utt_audio_queue.put(None)
        if self._utt_event_queue is not None:
            self._utt_event_queue.put_nowait(None)
        if self._utt_worker is not None:
            await asyncio.to_thread(self._utt_worker.join, 30)
        self._utt_audio_queue = None
        self._utt_event_queue = None
        self._utt_worker = None

    async def stream_async(
        self,
        audio: bytes | AsyncIterator[bytes],
    ) -> AsyncIterator[SpeechRecognizeEvent]:
        await self.begin_utterance()
        try:
            if isinstance(audio, (bytes, bytearray)):
                for frame in _iter_pcm_frames(bytes(audio)):
                    await self.push_audio(frame)
            else:
                async for chunk in audio:
                    await self.push_audio(chunk)
            async for event in self.end_utterance():
                yield event
        except Exception:
            await self.cancel_utterance()
            raise


class _DashScopeRealtimeSynthesizer(SpeechSynthesizer):
    """DashScope ``audio.tts_v2`` synthesizer with streaming PCM output."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = _TTS_DEFAULT_MODEL,
        ws_url: str,
        options: SpeechSynthesisOption | None,
        speech_id: str = "",
    ) -> None:
        if not api_key:
            raise SpeechError("DashScope API key is empty")
        self._id = speech_id
        self._api_key = api_key
        self._model = model
        self._ws_url = ws_url
        self._option = _with_default_voice(options)
        self._entered = False

    @property
    def id(self) -> str:
        return self._id

    def get_option(self) -> SpeechSynthesisOption:
        return self._option

    async def __aenter__(self) -> Self:
        self._entered = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._entered = False

    async def stream_async(self, text: str | AsyncIterator[str]) -> AsyncIterator[bytes]:
        if not self._entered:
            raise SpeechError(f"{type(self).__name__} must be entered before stream_async")
        if isinstance(text, str):
            content = text
        else:
            parts: list[str] = []
            async for part in text:
                parts.append(part)
            content = "".join(parts)
        if not content:
            return

        loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue[bytes | None | BaseException] = asyncio.Queue()
        audio_format = _dashscope_audio_format(self._option.sample_rate)

        class _Callback(ResultCallback):
            def on_data(self, data: bytes) -> None:
                if data:
                    loop.call_soon_threadsafe(audio_queue.put_nowait, bytes(data))

            def on_complete(self) -> None:
                loop.call_soon_threadsafe(audio_queue.put_nowait, None)

            def on_error(self, message: Any) -> None:
                loop.call_soon_threadsafe(
                    audio_queue.put_nowait,
                    SpeechError(f"DashScope TTS error: {message}"),
                )
                loop.call_soon_threadsafe(audio_queue.put_nowait, None)

        def _worker() -> None:
            try:
                with _SDK_KEY_LOCK:
                    previous = dashscope.api_key
                    dashscope.api_key = self._api_key
                    try:
                        synthesizer = DashScopeTTSV2Client(
                            model=self._model,
                            voice=self._option.voice,
                            format=audio_format,
                            volume=self._option.volume,
                            speech_rate=self._option.speech_rate,
                            pitch_rate=self._option.pitch_rate,
                            callback=_Callback(),
                            url=self._ws_url,
                        )
                    finally:
                        dashscope.api_key = previous
                synthesizer.streaming_call(content)
                synthesizer.streaming_complete()
            except BaseException as exc:
                loop.call_soon_threadsafe(audio_queue.put_nowait, exc)
                loop.call_soon_threadsafe(audio_queue.put_nowait, None)

        worker = threading.Thread(target=_worker, name="dashscope-tts", daemon=True)
        worker.start()
        try:
            while True:
                item = await audio_queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise SpeechError(f"DashScope TTS error: {item}") from item
                yield item
        finally:
            await asyncio.to_thread(worker.join, 30)


class DashScopeRealtimeSpeechProvider(ISpeechProvider):
    """ISpeechProvider that creates DashScope SDK ASR and TTS sessions."""

    def __init__(self, _component_site: IComponentSite, **_kwargs: Any) -> None:
        """Accept component-site construction while using caller-supplied credentials."""

    async def get_recognizer(
        self,
        options: SpeechRecognizeOption | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> SpeechRecognizer:
        api_key, model, base_url = self._credentials(
            capability="asr", api_key=api_key, model=model, base_url=base_url
        )
        speech_id = kwargs.get("id")
        return _DashScopeRealtimeRecognizer(
            api_key=api_key,
            model=model,
            ws_url=_websocket_url(base_url),
            options=options,
            speech_id=speech_id if isinstance(speech_id, str) else "",
        )

    async def get_synthesizer(
        self,
        options: SpeechSynthesisOption | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> SpeechSynthesizer:
        api_key, model, base_url = self._credentials(
            capability="tts", api_key=api_key, model=model, base_url=base_url
        )
        speech_id = kwargs.get("id")
        return _DashScopeRealtimeSynthesizer(
            api_key=api_key,
            model=model,
            ws_url=_websocket_url(base_url),
            options=options,
            speech_id=speech_id if isinstance(speech_id, str) else "",
        )

    def _credentials(
        self,
        *,
        capability: Literal["asr", "tts"],
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> tuple[str, str, str]:
        if not api_key:
            raise SpeechError("DashScope API key is empty")
        default_model = _TTS_DEFAULT_MODEL if capability == "tts" else _DEFAULT_MODEL
        resolved_model = model if model is not None else default_model
        resolved_base_url = (base_url if base_url is not None else _DEFAULT_HTTP_ORIGIN).rstrip("/")
        return api_key, resolved_model, resolved_base_url
