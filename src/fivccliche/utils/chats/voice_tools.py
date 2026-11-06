"""Pipecat transport and function schemas for voice client tools."""

from __future__ import annotations

import json
import logging
from typing import Any

from fivcplayground.backends.pipecat.tools import PipecatTool
from fivcplayground.tools import ToolRetriever
from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import (
    InputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketInputTransport,
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from fivccliche.utils.chats.tools import ChatToolSpec
from fivccliche.utils.chats.tool_calls import ChatToolSession

logger = logging.getLogger(__name__)


class VoiceToolBridge:
    """Turn registered client tools into Pipecat function schemas."""

    def __init__(self, session: ChatToolSession) -> None:
        self._session = session

    def inject(
        self,
        tool_retriever: ToolRetriever | None,
        *,
        tool_ids: list[str] | None = None,
    ) -> list[str]:
        """Add client tools to a Pipecat tool retriever and return merged ids."""
        existing_names = set(tool_retriever.tools) if tool_retriever is not None else set()
        client_names = self._session.channel.tool_names
        reserved = {"generate_structured_output"}
        collisions = sorted((existing_names | reserved).intersection(client_names))
        if collisions:
            raise ValueError(
                "Client tool names conflict with reserved or configured tools: "
                + ", ".join(collisions)
            )

        if tool_retriever is not None:
            for tool in self.to_tools():
                tool_retriever.tools[tool.name] = tool
        return list(dict.fromkeys([*(tool_ids or []), *client_names]))

    def to_tools(self) -> list[PipecatTool]:
        """Build schemas that preserve the frontend JSON property definitions."""
        return [
            PipecatTool(
                FunctionSchema(
                    name=spec.name,
                    description=spec.description,
                    properties=dict(spec.parameters.get("properties") or {}),
                    required=list(dict.fromkeys(spec.parameters.get("required") or [])),
                    handler=self._make_handler(spec),
                )
            )
            for spec in self._session.channel.tool_specs
        ]

    def _make_handler(self, spec: ChatToolSpec) -> Any:
        async def handler(params: FunctionCallParams) -> None:
            try:
                result = await self._session.invoke_async(
                    spec.name,
                    dict(params.arguments),
                )
            except Exception as exc:
                logger.warning("Voice client tool %s failed: %s", spec.name, exc)
                result = f"Error: {exc}"
            await params.result_callback(result)

        tool_handler: Any = handler
        tool_handler._pipecat_cancel_on_interruption = False
        tool_handler._pipecat_timeout_secs = self._session.tool_timeout.total_seconds()
        return tool_handler


class VoiceToolInputTransport(FastAPIWebsocketInputTransport):
    """Consume Protobuf transport messages and route client tool results."""

    def __init__(self, transport, client, params, *, tool_session: ChatToolSession, **kwargs):
        super().__init__(transport, client, params, **kwargs)
        self._tool_session = tool_session

    async def process_frame(self, frame, direction: FrameDirection):
        if not isinstance(frame, InputTransportMessageFrame):
            await super().process_frame(frame, direction)
            return

        message = frame.message
        if not isinstance(message, dict) or message.get("v") != 1:
            logger.debug("Ignoring non-envelope voice transport message")
            return
        if message.get("type") == "tool_result":
            self._tool_session.handle_result(message)
            return
        logger.debug("Ignoring unsupported voice tool message type: %s", message.get("type"))


class VoiceToolTransport(FastAPIWebsocketTransport):
    """FastAPI voice transport with an in-band client tool control channel."""

    def __init__(
        self,
        websocket,
        params: FastAPIWebsocketParams,
        *,
        tool_session: ChatToolSession,
    ) -> None:
        super().__init__(websocket, params)
        self._input = VoiceToolInputTransport(
            self,
            self._client,
            params,
            tool_session=tool_session,
            input_name=self._input_name,
        )

    async def send_tool_call_async(self, payload: dict[str, Any]) -> None:
        """Send a client tool RPC as an urgent Protobuf message frame."""
        message = {"v": 1, **payload}
        json.dumps(message, ensure_ascii=False)
        await self.output().queue_frame(
            OutputTransportMessageUrgentFrame(message=message),
            FrameDirection.DOWNSTREAM,
        )
