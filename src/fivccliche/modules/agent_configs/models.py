from uuid import uuid4

from sqlalchemy import Index
from sqlmodel import SQLModel, Field, JSON

from . import schemas


class UserEmbedding(SQLModel, table=True):
    """Embedding configuration model."""

    __tablename__ = "user_embedding"
    __table_args__ = (Index("ix_user_embedding_id_user_uuid", "id", "user_uuid", unique=True),)

    uuid: str = Field(
        default_factory=lambda: str(uuid4()),
        primary_key=True,
        max_length=32,
        description="Embedding config UUID.",
    )
    id: str = Field(
        max_length=32,
        description="Embedding config ID (unique within user scope).",
        index=True,
    )
    description: str | None = Field(
        default=None, max_length=1024, description="Embedding description."
    )
    provider: str = Field(
        default="openai",
        max_length=255,
        description="Embedding provider.",
    )
    model: str = Field(
        max_length=255,
        description="Embedding model name.",
    )
    api_key: str = Field(
        max_length=255,
        description="Embedding API key.",
    )
    base_url: str | None = Field(
        default=None,
        max_length=255,
        description="Embedding base URL.",
    )
    dimension: int = Field(
        default=1024,
        description="Embedding dimension.",
    )
    user_uuid: str | None = Field(
        default=None,
        foreign_key="user.uuid",
        description="User ID.",
    )

    def to_schema(self) -> schemas.UserEmbeddingSchema:
        return schemas.UserEmbeddingSchema(
            uuid=self.uuid,
            id=self.id,
            description=self.description,
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            dimension=self.dimension,
        )


class UserLLM(SQLModel, table=True):
    """LLM configuration model."""

    __tablename__ = "user_llm"
    __table_args__ = (Index("ix_user_llm_id_user_uuid", "id", "user_uuid", unique=True),)

    uuid: str = Field(
        default_factory=lambda: str(uuid4()),
        primary_key=True,
        max_length=32,
        description="LLM config UUID.",
    )
    id: str = Field(
        max_length=32,
        description="LLM config ID (unique within user scope).",
        index=True,
    )
    description: str | None = Field(default=None, max_length=1024, description="LLM description.")
    provider: str = Field(
        default="openai",
        max_length=255,
        description="LLM provider.",
    )
    model: str = Field(
        max_length=255,
        description="LLM model name.",
    )
    api_key: str = Field(
        max_length=255,
        description="LLM API key.",
    )
    base_url: str | None = Field(
        default=None,
        max_length=255,
        description="LLM base URL.",
    )
    temperature: float = Field(
        default=0.5,
        description="LLM temperature.",
    )
    max_tokens: int = Field(
        default=4096,
        description="LLM max tokens.",
    )
    user_uuid: str | None = Field(
        default=None,
        foreign_key="user.uuid",
        description="User ID.",
    )

    def to_schema(self) -> schemas.UserLLMSchema:
        return schemas.UserLLMSchema(
            uuid=self.uuid,
            id=self.id,
            description=self.description,
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


class UserTool(SQLModel, table=True):
    """Tool configuration model."""

    __tablename__ = "user_tool"
    __table_args__ = (Index("ix_user_tool_id_user_uuid", "id", "user_uuid", unique=True),)

    uuid: str = Field(
        default_factory=lambda: str(uuid4()),
        primary_key=True,
        max_length=32,
        description="Tool config UUID.",
    )
    id: str = Field(
        max_length=32,
        description="Tool config ID (unique within user scope).",
        index=True,
    )
    description: str | None = Field(default=None, max_length=1024, description="Tool description.")
    transport: schemas.UserToolTransport = Field(
        default=schemas.UserToolTransport.STDIO,
        max_length=16,
        description="Transport protocol for the tool (stdio, sse, or streamable_http)",
    )
    command: str | None = Field(default=None, description="Command to run the tool")
    args: list | None = Field(
        sa_type=JSON,
        default=None,
        description="Arguments for the command",
    )
    env: dict | None = Field(
        sa_type=JSON,
        default=None,
        description="Environment variables",
    )
    url: str | None = Field(default=None, description="URL for the tool")
    user_uuid: str | None = Field(
        default=None,
        foreign_key="user.uuid",
        description="User ID.",
    )

    def to_schema(self) -> schemas.UserToolSchema:
        return schemas.UserToolSchema(
            uuid=self.uuid,
            id=self.id,
            description=self.description,
            transport=self.transport,
            command=self.command,
            args=self.args,
            env=self.env,
            url=self.url,
        )


class UserAgent(SQLModel, table=True):
    """Agent configuration model."""

    __tablename__ = "user_agent"
    __table_args__ = (Index("ix_user_agent_id_user_uuid", "id", "user_uuid", unique=True),)

    uuid: str = Field(
        default_factory=lambda: str(uuid4()),
        primary_key=True,
        max_length=32,
        description="Agent config UUID.",
    )
    id: str = Field(
        max_length=32,
        description="Agent config ID (unique within user scope).",
        index=True,
    )
    description: str | None = Field(default=None, max_length=1024, description="Agent description.")
    model_id: str = Field(
        foreign_key="user_llm.id",
        description="LLM config ID.",
    )
    tools_ids: list[str] | None = Field(
        sa_type=JSON,
        default=None,
        description="List of tool config IDs.",
    )
    system_prompt: str | None = Field(
        default=None,
        max_length=1024,
        description="Agent system prompt.",
    )
    user_uuid: str | None = Field(
        default=None,
        foreign_key="user.uuid",
        description="User ID.",
    )

    def to_schema(self) -> schemas.UserAgentSchema:
        return schemas.UserAgentSchema(
            uuid=self.uuid,
            id=self.id,
            description=self.description,
            model_id=self.model_id,
            tool_ids=self.tools_ids,
            system_prompt=self.system_prompt,
        )
