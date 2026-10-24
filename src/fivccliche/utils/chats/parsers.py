"""Parsers for chat agent run streams."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Literal

from fivcplayground.agents import AgentRun, AgentRunEvent

logger = logging.getLogger(__name__)

ChatRunParserType = Literal["raw", "sse"]


class ChatRunParser:
    """Format a chat run stream for JSON or SSE consumers."""

    def __init__(
        self,
        chat_uuid: str | None = None,
        *,
        output_type: ChatRunParserType = "raw",
    ) -> None:
        if output_type not in ("raw", "sse"):
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

    async def parse_async(
        self,
        events: AsyncIterator[tuple[AgentRunEvent, AgentRun]],
    ) -> AsyncIterator[dict | str]:
        """Yield transport-ready payloads from a chat run stream."""
        try:
            if self._output_type == "sse":
                async for frame in self._parse_sse_async(events):
                    yield frame
            else:
                async for event in self._parse_raw_async(events):
                    yield event
        except Exception as exception:
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
