__all__ = [
    "AgentRunContent",
    "AgentRunStatus",
    "AgentRunToolCall",
    "UserChatCreateSchema",
    "UserChatMessageCreateSchema",
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
    context: dict | None = Field(default=None, description="Chat context")

    model_config = ConfigDict(from_attributes=True)


class UserChatMessageSchema(AgentRun):
    """Schema for reading user chat message data (response).

    Extends AgentRun from fivcplayground with additional fields for
    message-specific data and database persistence.
    """

    uuid: str = Field(default=None, description="Chat message UUID (globally unique)")
    chat_uuid: str = Field(default=None, description="Chat UUID")

    model_config = ConfigDict(from_attributes=True)


class UserChatCreateSchema(BaseModel):
    """Schema for creating a new chat session.

    Only requires agent_id; chat_uuid is generated server-side.
    """

    agent_id: str = Field(default="default", description="Agent ID for the chat")
    context: dict | None = Field(default=None, description="Initial chat context")


class UserChatMessageCreateSchema(BaseModel):
    """Schema for sending a message to an existing chat.

    Simplified schema for messaging; agent_id is determined from the chat.
    """

    query: str = Field(..., description="Message query/content")
