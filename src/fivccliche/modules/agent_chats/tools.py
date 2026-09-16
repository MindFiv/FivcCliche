import json

from fivccliche.modules.agent_configs.filters import UserScopedReadableFilterSet
from fivccliche.modules.agent_configs.models import UserASR
from fivccliche.modules.agent_configs.utils import get_user_scoped_async
from fivccliche.services.interfaces.speech import SpeechAudioInput, SpeechRecognizeOptions
from fivccliche.utils.deps import get_db_session_context_async, get_speech_provider_async


class SpeechTranscribe:
    """Transcribe a complete audio clip for the current user.

    Args:
        url: Publicly reachable audio URL.
        data_b64: Base64-encoded audio (without a URL).
        audio_format: Clip format, such as wav or mp3.
        language: Optional language hint.
        hotwords: Optional list of terms to bias recognition.
        context: Optional recognition context prompt.
    """

    def __init__(self, **context):
        self.user_uuid = context.get("user_uuid")
        self.asr_id = str(context.get("asr_id") or "default")
        self._is_superuser = bool(context.get("is_superuser"))

    async def __call__(
        self,
        url: str | None = None,
        data_b64: str | None = None,
        audio_format: str = "wav",
        language: str | None = None,
        hotwords: list[str] | None = None,
        context: str | None = None,
    ) -> str:
        if not self.user_uuid:
            raise ValueError("No user_uuid specified")

        async with get_db_session_context_async() as session:
            asr = await get_user_scoped_async(
                session,
                UserASR,
                filters=UserScopedReadableFilterSet(
                    UserASR.user_uuid, self.user_uuid, is_superuser=self._is_superuser
                ),
                config_id=self.asr_id,
            )
        if asr is None:
            raise ValueError("No speech provider specified")

        speech_provider = await get_speech_provider_async(asr.model_type)
        if speech_provider is None:
            raise ValueError("No speech provider specified")

        options = SpeechRecognizeOptions(
            language=language,
            hotwords=hotwords,
            context=context,
            format=audio_format,
        )
        text = ""
        detected_language = language
        async with await speech_provider.get_recognizer(
            options,
            api_key=asr.api_key,
            model=asr.model,
            base_url=asr.base_url,
        ) as recognizer:
            async for event in recognizer.stream_async(
                SpeechAudioInput(url=url, data_b64=data_b64, format=audio_format),
            ):
                if event.type == "final":
                    text = event.text
                    detected_language = event.language
                elif event.type == "error":
                    raise ValueError(event.message or "Speech recognition failed")
        return json.dumps({"text": text, "language": detected_language})
