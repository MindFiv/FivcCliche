__all__ = [
    "UserAgentSchema",
    "UserEmbeddingSchema",
    "UserLLMSchema",
    "UserToolSchema",
    "UserToolTransport",
]

from pydantic import ConfigDict, Field

from fivcplayground.embeddings.types import EmbeddingConfig
from fivcplayground.models.types import ModelConfig
from fivcplayground.tools.types import (
    ToolConfig,
    ToolConfigTransport as UserToolTransport,
)
from fivcplayground.agents.types import AgentConfig


# ============================================================================
# Read/Response Schemas (with uuid field)
# ============================================================================


class UserEmbeddingSchema(EmbeddingConfig):
    """Schema for reading embedding config data (response)."""

    uuid: str = Field(default=None, description="Embedding config UUID (globally unique)")

    model_config = ConfigDict(from_attributes=True)


class UserLLMSchema(ModelConfig):
    """Schema for reading LLM config data (response)."""

    uuid: str = Field(default=None, description="LLM config UUID (globally unique)")

    model_config = ConfigDict(from_attributes=True)


class UserToolSchema(ToolConfig):
    """Schema for reading tool config data (response)."""

    uuid: str = Field(default=None, description="Tool config UUID (globally unique)")

    model_config = ConfigDict(from_attributes=True)


class UserAgentSchema(AgentConfig):
    """Schema for reading agent config data (response)."""

    uuid: str = Field(default=None, description="Agent config UUID (globally unique)")

    model_config = ConfigDict(from_attributes=True)
