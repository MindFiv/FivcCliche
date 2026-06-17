__all__ = [
    "UserAgentSchema",
    "UserEmbeddingSchema",
    "UserLLMSchema",
    "UserQuestionSchema",
    "UserSkillSchema",
    "UserToolSchema",
    "UserToolTransport",
]

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from fivcplayground.embeddings.types import EmbeddingConfig
from fivcplayground.models.types import ModelConfig
from fivcplayground.tools.types import (
    ToolConfig,
    ToolConfigTransport as UserToolTransport,
)
from fivcplayground.agents.types import AgentConfig
from fivcplayground.skills.types import SkillConfig


# ============================================================================
# Read/Response Schemas (with uuid field)
# ============================================================================


class UserEmbeddingSchema(EmbeddingConfig):
    """Schema for reading embedding config data (response)."""

    uuid: str = Field(default=None, description="Embedding config UUID (globally unique)")
    user_uuid: str | None = Field(default=None, description="User UUID (read-only)")
    updated_at: datetime | None = Field(default=None, description="Last update time (read-only)")
    updated_user_uuid: str | None = Field(
        default=None, description="UUID of user who last updated (read-only)"
    )

    model_config = ConfigDict(from_attributes=True)


class UserLLMSchema(ModelConfig):
    """Schema for reading LLM config data (response)."""

    uuid: str = Field(default=None, description="LLM config UUID (globally unique)")
    user_uuid: str | None = Field(default=None, description="User UUID (read-only)")
    updated_at: datetime | None = Field(default=None, description="Last update time (read-only)")
    updated_user_uuid: str | None = Field(
        default=None, description="UUID of user who last updated (read-only)"
    )

    model_config = ConfigDict(from_attributes=True)


class UserToolSchema(ToolConfig):
    """Schema for reading tool config data (response)."""

    uuid: str = Field(default=None, description="Tool config UUID (globally unique)")
    description: str | None = Field(default=None, description="Tool description")
    is_active: bool = Field(default=True, description="Whether the tool is active")
    user_uuid: str | None = Field(default=None, description="User UUID (read-only)")
    updated_at: datetime | None = Field(default=None, description="Last update time (read-only)")
    updated_user_uuid: str | None = Field(
        default=None, description="UUID of user who last updated (read-only)"
    )

    model_config = ConfigDict(from_attributes=True)


class UserToolProbeSchema(BaseModel):
    """Schema for reading tool config data (response)."""

    tool_names: list[str] = Field(default=None, description="Tool names")


class UserSkillSchema(SkillConfig):
    """Schema for reading skill config data (response)."""

    uuid: str = Field(default=None, description="Skill config UUID (globally unique)")
    is_active: bool = Field(default=True, description="Whether the skill is active")
    user_uuid: str | None = Field(default=None, description="User UUID (read-only)")
    updated_at: datetime | None = Field(default=None, description="Last update time (read-only)")
    updated_user_uuid: str | None = Field(
        default=None, description="UUID of user who last updated (read-only)"
    )

    model_config = ConfigDict(from_attributes=True)


class UserAgentSchema(AgentConfig):
    """Schema for reading agent config data (response)."""

    uuid: str = Field(default=None, description="Agent config UUID (globally unique)")
    user_uuid: str | None = Field(default=None, description="User UUID (read-only)")
    updated_at: datetime | None = Field(default=None, description="Last update time (read-only)")
    updated_user_uuid: str | None = Field(
        default=None, description="UUID of user who last updated (read-only)"
    )

    model_config = ConfigDict(from_attributes=True)


class UserQuestionSchema(BaseModel):
    """Schema for reading user question data (response)."""

    uuid: str = Field(default=None, description="Question UUID (globally unique)")
    id: str = Field(..., description="Question ID (unique within user scope)")
    question: str = Field(..., description="User question text")
    answer: str | None = Field(default=None, description="User answer text")
    is_active: bool = Field(default=False, description="Whether the question is active")
    user_uuid: str | None = Field(default=None, description="User UUID (read-only)")
    updated_at: datetime | None = Field(default=None, description="Last update time (read-only)")
    updated_user_uuid: str | None = Field(
        default=None, description="UUID of user who last updated (read-only)"
    )

    model_config = ConfigDict(from_attributes=True)
