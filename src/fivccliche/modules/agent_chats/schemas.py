__all__ = [
    "AgentRunContent",
    "AgentRunStatus",
    "AgentRunToolCall",
    "UserChatMessageSchema",
    "UserChatSchema",
]

from pydantic import ConfigDict, Field, BaseModel

from fivcplayground.agents.types import (
    AgentRunSession,
    AgentRun,
    AgentRunStatus,
    AgentRunToolCall,
    AgentRunContent,
)


class UserChatSchema(AgentRunSession):
    """Schema for reading user chat session data (response).

    Extends AgentRunSession from fivcplayground with additional fields for
    database persistence.
    """

    uuid: str = Field(default=None, description="Chat UUID (globally unique)")

    model_config = ConfigDict(from_attributes=True)


class UserChatMessageSchema(AgentRun):
    """Schema for reading user chat message data (response).

    Extends AgentRun from fivcplayground with additional fields for
    message-specific data and database persistence.
    """

    uuid: str = Field(default=None, description="Chat message UUID (globally unique)")
    chat_uuid: str = Field(default=None, description="Chat UUID")

    model_config = ConfigDict(from_attributes=True)


class UserChatQuery(BaseModel):
    """Schema for querying in chat."""

    chat_uuid: str | None = Field(default=None, description="Chat UUID")
    agent_id: str | None = Field(default=None, description="Agent ID")
    query: str = Field(..., description="Chat query")
