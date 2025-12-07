from datetime import datetime
from uuid import uuid4

# from sqlalchemy import Index
from sqlmodel import SQLModel, Field, JSON

from . import schemas


class UserChat(SQLModel, table=True):
    """User chat model."""

    __tablename__ = "chat"

    uuid: str = Field(
        default_factory=lambda: str(uuid4()),
        primary_key=True,
        max_length=36,
        description="Chat UUID (globally unique).",
    )
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        max_length=36,
        description="Chat ID (for schema compatibility).",
    )
    user_uuid: str | None = Field(
        default=None,
        foreign_key="user.uuid",
        description="User ID.",
    )
    agent_id: str = Field(
        max_length=255,
        description="Agent config ID.",
    )
    description: str | None = Field(
        default=None,
        max_length=1024,
        description="Chat description.",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Chat creation time.",
    )

    def to_schema(self) -> schemas.UserChatSchema:
        """Convert UserChat to ChatSessionRead schema (AgentRunSession-based)."""
        return schemas.UserChatSchema(
            uuid=self.uuid,
            id=self.uuid,
            agent_id=self.agent_id,
            description=self.description,
            started_at=self.created_at,
        )


class UserChatMessage(SQLModel, table=True):
    """User chat message model."""

    __tablename__ = "chat_message"

    uuid: str = Field(
        default_factory=lambda: str(uuid4()),
        primary_key=True,
        max_length=36,
        description="Chat message UUID (globally unique).",
    )
    chat_uuid: str = Field(
        foreign_key="chat.uuid",
        description="Chat UUID.",
    )
    status: schemas.AgentRunStatus = Field(
        default=schemas.AgentRunStatus.PENDING,
        description="Message status.",
    )
    query: dict | None = Field(
        sa_type=JSON,
        default=None,
        max_length=1024,
        description="Query text (user message).",
    )
    reply: dict | None = Field(
        sa_type=JSON,
        default=None,
        max_length=1024,
        description="Reply text (agent response).",
    )
    tool_calls: dict | None = Field(
        sa_type=JSON,
        default=None,
        max_length=1024,
        description="Tool calls (agent response).",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Message creation time.",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="Message completion time.",
    )

    def to_schema(self) -> schemas.UserChatMessageSchema:
        """Convert UserChatMessage to ChatMessageRead schema (AgentRun-based)."""
        return schemas.UserChatMessageSchema(
            uuid=self.uuid,
            chat_uuid=self.chat_uuid,
            id=self.uuid,  # Use message uuid as the run id
            agent_id="",  # Agent ID is not available on ChatMessage, will be set by caller if needed
            status=self.status or schemas.AgentRunStatus.PENDING,
            started_at=self.created_at,
            completed_at=self.completed_at,
            query=self.query,
            reply=self.reply,
            tool_calls=self.tool_calls or {},  # Default to empty dict if None
        )
