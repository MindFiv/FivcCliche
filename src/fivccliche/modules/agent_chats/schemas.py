__all__ = [
    "AgentRunContent",
    "AgentRunStatus",
    "AgentRunToolCall",
    "ChatMessageSchema",
    "ChatSchema",
]

from pydantic import ConfigDict, Field

from fivcplayground.agents.types import (
    AgentRunSession,
    AgentRun,
    AgentRunStatus,
    AgentRunToolCall,
    AgentRunContent,
)


class ChatSchema(AgentRunSession):
    """Schema for reading chat session data (response).

    Extends AgentRunSession from fivcplayground with additional fields for
    database persistence.
    """

    uuid: str = Field(default=None, description="Chat UUID (globally unique)")

    model_config = ConfigDict(from_attributes=True)


class ChatMessageSchema(AgentRun):
    """Schema for reading chat message data (response).

    Extends AgentRun from fivcplayground with additional fields for
    message-specific data and database persistence.
    """

    uuid: str = Field(default=None, description="Chat message UUID (globally unique)")
    chat_uuid: str = Field(default=None, description="Chat UUID")

    model_config = ConfigDict(from_attributes=True)
