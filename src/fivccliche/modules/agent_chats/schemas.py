__all__ = [
    "AgentRunContent",
    "AgentRunStatus",
    "AgentRunToolCall",
    "UserChatCreateSchema",
    "UserChatMessageCreateSchema",
    "UserChatMessageSchema",
    "UserChatSchema",
    "UserChatUpdateSchema",
]

from pydantic import BaseModel, ConfigDict, Field, field_validator

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

    uuid: str | None = Field(default=None, description="Chat UUID (globally unique)")
    context: dict | None = Field(default=None, description="Chat context")
    is_memorable: bool = Field(
        default=False, description="Whether this chat is eligible for memory retention"
    )

    model_config = ConfigDict(from_attributes=True)


class UserChatMessageSchema(AgentRun):
    """Schema for reading user chat message data (response).

    Extends AgentRun from fivcplayground with additional fields for
    message-specific data and database persistence.
    """

    uuid: str | None = Field(default=None, description="Chat message UUID (globally unique)")
    chat_uuid: str | None = Field(default=None, description="Chat UUID")
    is_memorized: bool = Field(
        default=False, description="Whether the message has been retained to memory"
    )

    model_config = ConfigDict(from_attributes=True)


class UserChatCreateSchema(BaseModel):
    """Schema for creating a new chat session.

    Only requires agent_id; chat_uuid is generated server-side.
    """

    agent_id: str = Field(default="default", description="Agent ID for the chat")
    context: dict | None = Field(default=None, description="Initial chat context")


class UserChatUpdateSchema(BaseModel):
    """Schema for updating a chat session description."""

    description: str = Field(..., min_length=1, max_length=1024, description="Chat description")

    @field_validator("description")
    @classmethod
    def description_not_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("description must not be empty")
        return text


class UserChatMessageCreateSchema(BaseModel):
    """Schema for sending a message to an existing chat.

    Simplified schema for messaging; agent_id is determined from the chat.
    """

    query: str = Field(..., description="Message query/content")
