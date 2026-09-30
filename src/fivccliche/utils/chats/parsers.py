"""Parsers for chat agent and speech streams."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any, Literal

from fivcplayground.agents import AgentRun, AgentRunEvent

from fivccliche.services.interfaces.agent_speeches import SpeechEvent, SpeechRequestError

logger = logging.getLogger(__name__)

ChatRunParserType = Literal["raw", "sse", "voice"]

_VOICE_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_VOICE_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_VOICE_MARKDOWN_EMPHASIS = re.compile(r"(`{1,3}|\*{1,3}|_{1,3}|~{1,2})")


_SPEECH_BOUNDARY = re.compile(r"[。！？!?；;\n]")  # noqa: RUF001
_SPEECH_MAX_LENGTH = 24


class _ChatVoiceBuffer:
    """Accumulate cleaned assistant text into short synthesis chunks."""

    def __init__(self, max_length: int = _SPEECH_MAX_LENGTH) -> None:
        self.max_length = max_length
        self._value = ""

    def append(self, value: str) -> list[str]:
        self._value += value
        return self.pop()

    def pop(self) -> list[str]:
        chunks: list[str] = []
        while self._value:
            match = _SPEECH_BOUNDARY.search(self._value)
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


def _chat_voice_clean(value: str) -> str:
    """Turn assistant display text into compact text suitable for TTS."""

    value = _VOICE_MARKDOWN_IMAGE.sub("", value)
    value = _VOICE_MARKDOWN_LINK.sub(r"\1", value)
    value = _VOICE_MARKDOWN_EMPHASIS.sub("", value)
    value = re.sub(r"https?://\S+", "", value)
    value = re.sub(r"[|#>]", " ", value)
    return " ".join(value.split())


class ChatRunParser:
    """Format a chat run stream for JSON or SSE consumers."""

    def __init__(
        self,
        chat_uuid: str | None = None,
        *,
        output_type: ChatRunParserType = "raw",
    ) -> None:
        if output_type not in ("raw", "sse", "voice"):
            raise ValueError(f"Invalid output_type: {output_type}")
        self._chat_uuid = chat_uuid
        self._output_type = output_type
        self.current_run_id: str | None = None

    async def _parse_raw_async(
        self,
        events: AsyncIterator[tuple[AgentRunEvent, AgentRun]],
    ) -> AsyncIterator[dict[str, Any]]:
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
        async for ev, ev_run in events:
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
                delta = ev_run.delta.model_dump(mode="json") if ev_run.delta else None
                raw_text = getattr(ev_run.delta, "text", None) if ev_run.delta else None
                if isinstance(raw_text, str):
                    delta = (
                        {**delta, "text": raw_text}
                        if isinstance(delta, dict)
                        else {"text": raw_text}
                    )
                info.update(
                    {
                        "chat_uuid": self._chat_uuid,
                        "delta": delta,
                    }
                )
                yield {"event": "stream", "info": info}

            elif ev == AgentRunEvent.TOOL:
                info = ev_run.model_dump(mode="json", include=data_fields)
                info.update({"chat_uuid": self._chat_uuid})
                yield {"event": "tool", "info": info}

    async def _parse_sse_async(
        self,
        events: AsyncIterator[tuple[AgentRunEvent, AgentRun]],
    ) -> AsyncIterator[str]:
        async for payload in self._parse_raw_async(events):
            yield f"data: {json.dumps(payload)}\n\n"

    async def _parse_voice_async(
        self,
        events: AsyncIterator[tuple[AgentRunEvent, AgentRun]],
    ) -> AsyncIterator[str]:
        """Yield cleaned sentence chunks for speech synthesis."""
        buffer = _ChatVoiceBuffer()
        async for payload in self._parse_raw_async(events):
            if payload["event"] == "start":
                run_id = payload["info"].get("id")
                if isinstance(run_id, str):
                    self.current_run_id = run_id
                continue
            if payload["event"] != "stream":
                continue
            delta = payload["info"].get("delta")
            text = delta.get("text") if isinstance(delta, dict) else None
            if not isinstance(text, str) or not text:
                continue
            cleaned = _chat_voice_clean(text)
            if not cleaned:
                continue
            for chunk in buffer.append(cleaned):
                yield chunk
        for chunk in buffer.flush():
            yield chunk

    async def parse_async(
        self,
        events: AsyncIterator[tuple[AgentRunEvent, AgentRun]],
    ) -> AsyncIterator[dict | str]:
        """Yield transport-ready payloads from a chat run stream."""
        try:
            if self._output_type == "sse":
                async for frame in self._parse_sse_async(events):
                    yield frame
            elif self._output_type == "voice":
                async for item in self._parse_voice_async(events):
                    yield item
            else:
                async for event in self._parse_raw_async(events):
                    yield event
        except Exception as exception:
            if self._output_type == "voice":
                logger.exception("Error in chat run stream")
                raise
            message = (
                "Chat message processing timed out"
                if isinstance(exception, TimeoutError)
                else str(exception)
            )
            logger.exception("Error in chat run stream")
            error_payload = {"event": "error", "info": {"message": message}}
            if self._output_type == "sse":
                yield f"data: {json.dumps(error_payload)}\n\n"
            else:
                yield error_payload


class ChatRecognizeParser:
    """Turn recognition events into completed transcript segments."""

    def __init__(
        self,
        *,
        empty_partials_to_flush: int = 5,
        max_sentences: int | None = None,
    ) -> None:
        if max_sentences is not None and max_sentences <= 0:
            raise ValueError("max_sentences must be greater than zero")
        self._empty_partials_to_flush = empty_partials_to_flush
        self._max_sentences = max_sentences

    async def parse_async(
        self,
        events: AsyncIterator[SpeechEvent],
    ) -> AsyncIterator[str]:
        """Yield finals after idle partials, or when recognition ends."""
        pending: list[str] = []
        empty_partials = 0
        async for event in events:
            if event.type == "final":
                text = event.text.strip()
                if text:
                    pending.append(text)
                    if self._max_sentences is not None and len(pending) >= self._max_sentences:
                        yield "\n".join(pending)
                        return
                empty_partials = 0
            elif event.type == "partial":
                if event.text.strip():
                    empty_partials = 0
                    continue
                empty_partials += 1
                if empty_partials >= self._empty_partials_to_flush and pending:
                    yield "\n".join(pending)
                    pending.clear()
                    empty_partials = 0
            elif event.type == "error":
                raise SpeechRequestError(event.message or "Speech recognition failed")

        if pending:
            yield "\n".join(pending)
