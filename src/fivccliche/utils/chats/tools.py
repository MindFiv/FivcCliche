"""Frontend/session tools bridged into playground ToolRetriever."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from fivcplayground.tools import Tool, ToolBackend, ToolRetriever

from fivccliche.utils.chats.tool_calls import ChatToolSession

logger = logging.getLogger(__name__)

_JSON_TYPE_TO_ANNOTATION: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


@dataclass(frozen=True)
class ChatToolSpec:
    """A frontend-advertised tool for one chat WebSocket session."""

    name: str
    description: str
    parameters: dict[str, Any]


def _annotation_for_property(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return Any
    if any(key in schema for key in ("$ref", "anyOf", "oneOf", "allOf")):
        return Any
    json_type = schema.get("type")
    if isinstance(json_type, list):
        return Any
    if isinstance(json_type, str):
        return _JSON_TYPE_TO_ANNOTATION.get(json_type, Any)
    return Any


def make_client_tool_callable(
    session: ChatToolSession, spec: ChatToolSpec
) -> Callable[..., Awaitable[Any]]:
    """Build an async callable whose signature mirrors ``spec.parameters``.

    Only flat ``object`` schemas with identifier property names are expanded.
    Complex combinators (``$ref`` / ``anyOf`` / …) degrade to ``Any``.
    """
    parameters = spec.parameters if isinstance(spec.parameters, dict) else {}
    props = parameters.get("properties")
    if not isinstance(props, dict):
        props = {}
    required_raw = parameters.get("required")
    required = (
        {str(item) for item in required_raw if isinstance(item, str)}
        if isinstance(required_raw, list)
        else set()
    )

    signature_params: list[inspect.Parameter] = []
    for prop_name, prop_schema in props.items():
        if not isinstance(prop_name, str) or not prop_name.isidentifier():
            continue
        annotation = _annotation_for_property(prop_schema)
        default = inspect.Parameter.empty if prop_name in required else None
        signature_params.append(
            inspect.Parameter(
                prop_name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=annotation,
            )
        )
    signature = inspect.Signature(signature_params)

    async def _invoke(**kwargs: Any) -> Any:
        try:
            return await session.invoke_async(spec.name, dict(kwargs))
        except Exception as exc:
            logger.warning("Client tool %s failed: %s", spec.name, exc)
            return f"Error: {exc}"

    _invoke.__name__ = spec.name if spec.name.isidentifier() else "client_tool"
    _invoke.__qualname__ = _invoke.__name__
    _invoke.__signature__ = signature  # type: ignore[attr-defined]
    _invoke.__annotations__ = {
        **{param.name: param.annotation for param in signature_params},
        "return": Any,
    }
    return _invoke


class ChatToolBridge:
    """Turn :class:`ChatToolSpec` entries on a tool session into playground tools."""

    def __init__(self, session: ChatToolSession) -> None:
        self._session = session

    def to_tools(self, tool_backend: ToolBackend) -> list[Tool]:
        tools: list[Tool] = []
        for spec in self._session.channel.tool_specs:
            tools.append(self._create_tool(tool_backend, spec))
        return tools

    def inject(
        self,
        tool_retriever: ToolRetriever | None,
        tool_backend: ToolBackend,
        *,
        tool_ids: list[str] | None = None,
    ) -> list[str]:
        """Write client tools into ``tool_retriever.tools`` and return merged ids."""
        client_tools = self.to_tools(tool_backend)
        client_names = [tool.name for tool in client_tools]
        if tool_retriever is not None:
            for tool in client_tools:
                tool_retriever.tools[tool.name] = tool
        merged = list(dict.fromkeys([*(tool_ids or []), *client_names]))
        return merged

    def _create_tool(self, tool_backend: ToolBackend, spec: ChatToolSpec) -> Tool:
        return tool_backend.create_tool(
            make_client_tool_callable(self._session, spec),
            tool_name=spec.name,
            tool_description=spec.description,
        )
