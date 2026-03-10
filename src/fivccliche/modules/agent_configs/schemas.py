__all__ = [
    "UserAgentSchema",
    "UserEmbeddingSchema",
    "UserLLMSchema",
    "UserSkillSchema",
    "UserToolSchema",
    "UserToolTransport",
]

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

    model_config = ConfigDict(from_attributes=True)


class UserLLMSchema(ModelConfig):
    """Schema for reading LLM config data (response)."""

    uuid: str = Field(default=None, description="LLM config UUID (globally unique)")
    user_uuid: str | None = Field(default=None, description="User UUID (read-only)")

    model_config = ConfigDict(from_attributes=True)


class UserToolSchema(ToolConfig):
    """Schema for reading tool config data (response)."""

    uuid: str = Field(default=None, description="Tool config UUID (globally unique)")
    is_active: bool = Field(default=True, description="Whether the tool is active")
    user_uuid: str | None = Field(default=None, description="User UUID (read-only)")

    model_config = ConfigDict(from_attributes=True)


class UserToolProbeSchema(BaseModel):
    """Schema for reading tool config data (response)."""

    tool_names: list[str] = Field(default=None, description="Tool names")


class UserSkillSchema(SkillConfig):
    """Schema for reading skill config data (response)."""

    uuid: str = Field(default=None, description="Skill config UUID (globally unique)")
    is_active: bool = Field(default=True, description="Whether the skill is active")
    user_uuid: str | None = Field(default=None, description="User UUID (read-only)")

    model_config = ConfigDict(from_attributes=True)


class UserAgentSchema(AgentConfig):
    """Schema for reading agent config data (response)."""

    uuid: str = Field(default=None, description="Agent config UUID (globally unique)")
    user_uuid: str | None = Field(default=None, description="User UUID (read-only)")

    model_config = ConfigDict(from_attributes=True)
