"""DashScope clip ASR for Qwen3-ASR and Qwen-Audio 3.0."""

from __future__ import annotations

import base64
import logging
from collections.abc import AsyncIterator
from typing import Any, ClassVar, Self

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

DEFAULT_GENERATION_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
)
_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com"
_DEFAULT_MODEL = "qwen3-asr-flash"


class _AudioContent:
    """Audio encoded in the representation expected by a request body."""

    _MIME_BY_FORMAT: ClassVar[dict[str, str]] = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "mpeg": "audio/mpeg",
        "pcm": "audio/pcm",
        "opus": "audio/opus",
        "ogg": "audio/ogg",
        "m4a": "audio/mp4",
        "aac": "audio/aac",
    }

    @classmethod
    def from_input(cls, audio: SpeechAudioInput) -> str:
        if audio.url:
            return audio.url
        data = audio.data_b64 or ""
        if data.startswith("data:"):
            return data
        mime = cls._MIME_BY_FORMAT.get(audio.format.lower(), "audio/wav")
        return f"data:{mime};base64,{data}"


class _Qwen3Protocol:
    """DashScope multimodal generation protocol used by Qwen3 ASR."""

    @staticmethod
    def payload(
        model: str,
        audio: SpeechAudioInput,
        options: SpeechRecognizeOptions | None,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        context = _context_text(options)
        if context:
            messages.append({"role": "system", "content": [{"text": context}]})
        messages.append({"role": "user", "content": [{"audio": _AudioContent.from_input(audio)}]})
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

    @staticmethod
    def headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def event(payload: dict[str, Any]) -> SpeechEvent:
        choices = ((payload.get("output") or {}).get("choices")) or []
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


class _QwenAudioProtocol:
    """Native synchronous protocol used by Qwen-Audio 3.0 ASR."""

    MODEL = "qwen-audio-3.0-asr-flash"

    @staticmethod
    def payload(
        model: str,
        audio: SpeechAudioInput,
        _options: SpeechRecognizeOptions | None,
    ) -> dict[str, Any]:
        return {
            "model": model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": _AudioContent.from_input(audio)},
                            }
                        ],
                    }
                ]
            },
            "parameters": {"format": audio.format.lower()},
        }

    @staticmethod
    def headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-SSE": "disable",
        }

    @staticmethod
    def event(payload: dict[str, Any]) -> SpeechEvent:
        output = payload.get("output") or {}
        sentence = output.get("sentence")
        text = output.get("text")
        if not text and isinstance(sentence, dict):
            text = sentence.get("text")
        if not text:
            raise SpeechRequestError("DashScope ASR response missing transcript")
        return SpeechEvent(type="final", text=str(text), language=None)


def _context_text(options: SpeechRecognizeOptions | None) -> str | None:
    if options is None:
        return None
    parts: list[str] = []
    if options.context:
        parts.append(options.context)
    if options.hotwords:
        parts.append("热词: " + ", ".join(options.hotwords))
    return "\n".join(parts) if parts else None


class DashScopeMultimodalRecognizer(ISpeechRecognizer):
    """Clip ASR using DashScope synchronous multimodal generation."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        generation_url: str = DEFAULT_GENERATION_URL,
        http_client: Any | None = None,
        options: SpeechRecognizeOptions | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._generation_url = generation_url
        self._http_client = http_client
        self._options = options

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return

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
        audio = SpeechAudioInput(
            data_b64=base64.b64encode(bytes(buffer)).decode("ascii"),
            format=audio_format,
        )
        yield await self._recognize_clip(audio)

    async def _recognize_clip(self, audio: SpeechAudioInput) -> SpeechEvent:
        protocol = _QwenAudioProtocol if self._model == _QwenAudioProtocol.MODEL else _Qwen3Protocol
        payload = protocol.payload(self._model, audio, self._options)
        headers = protocol.headers(self._api_key)
        response = await self._post(payload, headers)
        if not getattr(response, "is_success", False):
            status = getattr(response, "status_code", "?")
            body = getattr(response, "text", "")
            raise SpeechRequestError(f"DashScope ASR HTTP {status}: {body}")
        return protocol.event(response.json())

    async def _post(self, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        client = self._http_client
        if client is not None:
            return await client.post(self._generation_url, json=payload, headers=headers)
        import httpx

        async with httpx.AsyncClient(timeout=60.0) as http_client:
            return await http_client.post(self._generation_url, json=payload, headers=headers)


class DashScopeMultimodalSpeechProvider(ISpeechProvider):
    """ISpeechProvider that creates DashScope Flash recognizers."""

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
        generation_url = (
            f"{base_url}/api/v1/services/aigc/multimodal-generation/generation"
            if base_url != _DEFAULT_BASE_URL
            else DEFAULT_GENERATION_URL
        )
        return DashScopeMultimodalRecognizer(
            api_key=api_key,
            model=model,
            generation_url=generation_url,
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
