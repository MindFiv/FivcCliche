"""DashScope clip ASR for Qwen3-ASR and Qwen-Audio."""

from __future__ import annotations

import base64
import logging
from collections.abc import AsyncIterator
from typing import Any, Self

import httpx
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
    SpeechSynthesisOptions,
)
from fivccliche.utils.types import to_string

_logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com"
_DEFAULT_MODEL = "qwen3-asr-flash"
_QWEN_AUDIO_MODEL = "qwen-audio-3.0-asr-flash"
_GENERATION_PATH = "/api/v1/services/aigc/multimodal-generation/generation"
_DEFAULT_GENERATION_URL = _DEFAULT_BASE_URL + _GENERATION_PATH

_AUDIO_MIME_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "mpeg": "audio/mpeg",
    "pcm": "audio/pcm",
    "opus": "audio/opus",
    "ogg": "audio/ogg",
    "m4a": "audio/mp4",
    "aac": "audio/aac",
}


def _audio_content(audio: SpeechAudioInput) -> str:
    if audio.url:
        return audio.url

    data = audio.data_b64 or ""
    if data.startswith("data:"):
        return data

    mime = _AUDIO_MIME_TYPES.get(audio.format.lower(), "audio/wav")
    return f"data:{mime};base64,{data}"


def _context_text(options: SpeechRecognizeOptions | None) -> str | None:
    if options is None:
        return None

    parts: list[str] = []
    if options.context:
        parts.append(options.context)
    if options.hotwords:
        parts.append("热词: " + ", ".join(options.hotwords))
    return "\n".join(parts) if parts else None


def _qwen3_request(
    model: str,
    audio: SpeechAudioInput,
    options: SpeechRecognizeOptions | None,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    context = _context_text(options)
    if context:
        messages.append({"role": "system", "content": [{"text": context}]})
    messages.append({"role": "user", "content": [{"audio": _audio_content(audio)}]})

    asr_options: dict[str, Any] = {
        "enable_itn": bool(options.enable_itn) if options is not None else False,
    }
    if options is not None and options.language:
        asr_options["language"] = options.language
    return {
        "model": model,
        "input": {"messages": messages},
        "parameters": {"asr_options": asr_options},
    }


def _qwen3_event(output: Any) -> SpeechEvent:
    choices = (output or {}).get("choices") or []
    if not choices:
        raise SpeechRequestError("DashScope ASR response missing output.choices")

    message = (choices[0] or {}).get("message") or {}
    text = next(
        (
            str(item["text"])
            for item in message.get("content") or []
            if isinstance(item, dict) and item.get("text")
        ),
        "",
    )
    annotations = message.get("annotations") or []
    language = None
    if annotations and isinstance(annotations[0], dict):
        language = annotations[0].get("language")
    return SpeechEvent(type="final", text=text, language=language)


def _qwen_audio_request(model: str, audio: SpeechAudioInput) -> dict[str, Any]:
    return {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": _audio_content(audio)},
                        }
                    ],
                }
            ]
        },
        "parameters": {"format": audio.format.lower(), "sample_rate": audio.sample_rate},
    }


def _qwen_audio_event(output: Any) -> SpeechEvent:
    output = output or {}
    sentence = output.get("sentence")
    text = output.get("text")
    if not text and isinstance(sentence, dict):
        text = sentence.get("text")
    if not text:
        raise SpeechRequestError("DashScope ASR response missing transcript")
    return SpeechEvent(type="final", text=str(text), language=None)


class _DashScopeMultimodalRecognizer(ISpeechRecognizer):
    """Clip ASR using DashScope synchronous multimodal generation."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
        http_client: httpx.AsyncClient | None = None,
        options: SpeechRecognizeOptions | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._http_client = http_client
        self._owns_http_client = http_client is None
        self._options = options

    async def __aenter__(self) -> Self:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=60.0)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def stream_async(
        self,
        audio: SpeechAudioInput | AsyncIterator[bytes],
    ) -> AsyncIterator[SpeechEvent]:
        if isinstance(audio, SpeechAudioInput):
            yield await self._recognize_clip(audio)
            return

        buffer = bytearray()
        async for chunk in audio:
            buffer.extend(chunk)
        if not buffer:
            yield SpeechEvent(type="final", text="")
            return

        audio_format = self._options.format if self._options is not None else "pcm"
        yield await self._recognize_clip(
            SpeechAudioInput(
                data_b64=base64.b64encode(bytes(buffer)).decode("ascii"),
                format=audio_format,
                sample_rate=self._options.sample_rate if self._options is not None else 16000,
            )
        )

    async def _recognize_clip(self, audio: SpeechAudioInput) -> SpeechEvent:
        is_qwen_audio = self._model == _QWEN_AUDIO_MODEL
        payload = (
            _qwen_audio_request(self._model, audio)
            if is_qwen_audio
            else _qwen3_request(self._model, audio, self._options)
        )
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if is_qwen_audio:
            headers["X-DashScope-SSE"] = "disable"

        if self._http_client is None:
            raise SpeechRequestError("DashScope ASR client is not started")

        try:
            response = await self._http_client.post(
                self._base_url + _GENERATION_PATH,
                json=payload,
                headers=headers,
            )
        except Exception as exc:
            raise SpeechRequestError(f"DashScope ASR request failed: {exc}") from exc

        if not response.is_success:
            status = response.status_code
            detail = response.text
            raise SpeechRequestError(f"DashScope ASR HTTP {status}: {detail}")

        try:
            response_payload = response.json()
        except ValueError as exc:
            raise SpeechRequestError(f"DashScope ASR returned invalid JSON: {exc}") from exc

        event_parser = _qwen_audio_event if is_qwen_audio else _qwen3_event
        return event_parser(response_payload.get("output"))


class DashScopeSpeechProvider(ISpeechProvider):
    """ISpeechProvider that creates DashScope clip recognizers."""

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
        return _DashScopeMultimodalRecognizer(
            api_key=api_key,
            model=model,
            base_url=base_url,
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
        raise SpeechRequestError("DashScope ASR provider does not support TTS")

    def _credentials(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> tuple[str, str, str]:
        session = _speech_config_session(self._component_site)
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


def _speech_config_session(component_site: IComponentSite) -> configs.IConfigSession | None:
    config = query_component(component_site, configs.IConfig)
    if config is None:
        _logger.warning("IConfig component not registered; using speech defaults")
        return None

    session = config.get_session("SPEECH")
    if session is None:
        _logger.warning("Config session 'SPEECH' not found; using speech defaults")
    return session
