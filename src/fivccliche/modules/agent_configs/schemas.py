__all__ = [
    "UserASRCreateSchema",
    "UserASRProbeRequest",
    "UserASRSchema",
    "UserAgentSchema",
    "UserEmbeddingCreateSchema",
    "UserEmbeddingSchema",
    "UserLLMCreateSchema",
    "UserLLMSchema",
    "UserQuestionSchema",
    "UserSkillSchema",
    "UserTTSCreateSchema",
    "UserTTSSchema",
    "UserToolSchema",
    "UserToolTransport",
]

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from fivcplayground.embeddings.types import EmbeddingConfig
from fivcplayground.models.types import ModelConfig
from fivcplayground.tools.types import (
    ToolConfig,
    ToolConfigTransport as UserToolTransport,
)
from fivcplayground.agents.types import AgentConfig
from fivcplayground.skills.types import SkillConfig

# Column limits of the agent_configs tables (models.py). Declaring them on the
# request schemas turns oversized values into 422 instead of a database error.
ID_MAX_LENGTH = 36
DESCRIPTION_MAX_LENGTH = 1024
STRING_MAX_LENGTH = 255

# ============================================================================
# Read/Response Schemas (with uuid field)
# ============================================================================


class UserEmbeddingSchema(EmbeddingConfig):
    """Schema for reading embedding config data (response)."""

    id: str = Field(..., max_length=ID_MAX_LENGTH, description="Embedding config ID")
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX_LENGTH)
    provider: str = Field(..., max_length=STRING_MAX_LENGTH, description="Embedding provider")
    model: str = Field(..., max_length=STRING_MAX_LENGTH, description="Embedding model name")
    api_key: str | None = Field(default=None, max_length=STRING_MAX_LENGTH, exclude=True)
    base_url: str | None = Field(default=None, max_length=STRING_MAX_LENGTH)
    uuid: str | None = Field(default=None, description="Embedding config UUID (globally unique)")
    user_uuid: str | None = Field(default=None, description="User UUID (read-only)")
    updated_at: datetime | None = Field(default=None, description="Last update time (read-only)")
    updated_user_uuid: str | None = Field(
        default=None, description="UUID of user who last updated (read-only)"
    )

    model_config = ConfigDict(from_attributes=True)


class UserLLMSchema(ModelConfig):
    """Schema for reading LLM config data (response)."""

    id: str = Field(..., max_length=ID_MAX_LENGTH, description="LLM config ID")
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX_LENGTH)
    provider: str = Field(..., max_length=STRING_MAX_LENGTH, description="LLM provider")
    model: str = Field(..., max_length=STRING_MAX_LENGTH, description="LLM model name")
    api_key: str | None = Field(default=None, max_length=STRING_MAX_LENGTH, exclude=True)
    base_url: str | None = Field(default=None, max_length=STRING_MAX_LENGTH)
    uuid: str | None = Field(default=None, description="LLM config UUID (globally unique)")
    user_uuid: str | None = Field(default=None, description="User UUID (read-only)")
    updated_at: datetime | None = Field(default=None, description="Last update time (read-only)")
    updated_user_uuid: str | None = Field(
        default=None, description="UUID of user who last updated (read-only)"
    )

    model_config = ConfigDict(from_attributes=True)


class UserASRSchema(BaseModel):
    """Schema for reading ASR config data (response)."""

    id: str = Field(
        ...,
        max_length=ID_MAX_LENGTH,
        description="ASR config ID (unique within user scope)",
    )
    description: str | None = Field(
        default=None, max_length=DESCRIPTION_MAX_LENGTH, description="ASR description"
    )
    model: str = Field(..., max_length=STRING_MAX_LENGTH, description="ASR model name")
    base_url: str | None = Field(
        default=None, max_length=STRING_MAX_LENGTH, description="ASR base URL"
    )
    api_key: str | None = Field(default=None, max_length=STRING_MAX_LENGTH, exclude=True)
    model_type: Literal["dashscope", "dashscope_realtime"] = Field(
        default="dashscope",
        description="ISpeechProvider name (dashscope or dashscope_realtime)",
    )
    uuid: str | None = Field(default=None, description="ASR config UUID (globally unique)")
    user_uuid: str | None = Field(default=None, description="User UUID (read-only)")
    updated_at: datetime | None = Field(default=None, description="Last update time (read-only)")
    updated_user_uuid: str | None = Field(
        default=None, description="UUID of user who last updated (read-only)"
    )

    model_config = ConfigDict(from_attributes=True)


class UserASRProbeRequest(BaseModel):
    """Request body for probing an ASR config with a clip."""

    data_b64: str = Field(..., description="Base64-encoded audio")
    format: str = Field(default="wav", description="Clip format, such as wav or mp3")
    language: str | None = Field(default=None, description="Optional language hint")
    hotwords: list[str] | None = Field(default=None, description="Optional hotwords")
    context: str | None = Field(default=None, description="Optional recognition context")


class UserTTSSchema(BaseModel):
    """Schema for reading TTS config data (response)."""

    id: str = Field(
        ...,
        max_length=ID_MAX_LENGTH,
        description="TTS config ID (unique within user scope)",
    )
    description: str | None = Field(
        default=None, max_length=DESCRIPTION_MAX_LENGTH, description="TTS description"
    )
    model: str = Field(..., max_length=STRING_MAX_LENGTH, description="TTS model name")
    base_url: str | None = Field(
        default=None, max_length=STRING_MAX_LENGTH, description="TTS base URL"
    )
    api_key: str | None = Field(default=None, max_length=STRING_MAX_LENGTH, exclude=True)
    model_type: Literal["dashscope", "dashscope_realtime"] = Field(
        default="dashscope",
        description="ISpeechProvider name (dashscope or dashscope_realtime)",
    )
    uuid: str | None = Field(default=None, description="TTS config UUID (globally unique)")
    user_uuid: str | None = Field(default=None, description="User UUID (read-only)")
    updated_at: datetime | None = Field(default=None, description="Last update time (read-only)")
    updated_user_uuid: str | None = Field(
        default=None, description="UUID of user who last updated (read-only)"
    )

    model_config = ConfigDict(from_attributes=True)


class UserTTSProbeRequest(BaseModel):
    """Request body for synthesizing a short probe with a TTS config."""

    text: str = Field(..., min_length=1, description="Text to synthesize")
    voice: str = Field(default="", description="Vendor voice name; empty selects a model default")
    format: str = Field(default="pcm", description="Audio encoding, such as pcm, wav, or mp3")
    sample_rate: int = Field(default=24000, ge=8000, le=48000, description="Audio sample rate")
    volume: int = Field(default=50, ge=0, le=100, description="Volume")
    speech_rate: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech rate")
    pitch_rate: float = Field(default=1.0, ge=0.5, le=2.0, description="Pitch rate")


class UserToolSchema(ToolConfig):
    """Schema for reading tool config data (response)."""

    id: str = Field(..., max_length=ID_MAX_LENGTH, description="Tool config ID")
    uuid: str | None = Field(default=None, description="Tool config UUID (globally unique)")
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

    tool_names: list[str] | None = Field(default=None, description="Tool names")


class UserSkillSchema(SkillConfig):
    """Schema for reading skill config data (response)."""

    id: str = Field(..., max_length=ID_MAX_LENGTH, description="Skill config ID")
    uuid: str | None = Field(default=None, description="Skill config UUID (globally unique)")
    is_active: bool = Field(default=True, description="Whether the skill is active")
    user_uuid: str | None = Field(default=None, description="User UUID (read-only)")
    updated_at: datetime | None = Field(default=None, description="Last update time (read-only)")
    updated_user_uuid: str | None = Field(
        default=None, description="UUID of user who last updated (read-only)"
    )

    model_config = ConfigDict(from_attributes=True)


class UserAgentSchema(AgentConfig):
    """Schema for reading agent config data (response)."""

    id: str = Field(..., max_length=ID_MAX_LENGTH, description="Agent config ID")
    uuid: str | None = Field(default=None, description="Agent config UUID (globally unique)")
    is_frozen: bool = Field(
        default=False, description="Whether the agent config is frozen (cannot be edited)"
    )
    user_uuid: str | None = Field(default=None, description="User UUID (read-only)")
    updated_at: datetime | None = Field(default=None, description="Last update time (read-only)")
    updated_user_uuid: str | None = Field(
        default=None, description="UUID of user who last updated (read-only)"
    )

    model_config = ConfigDict(from_attributes=True)


class UserQuestionSchema(BaseModel):
    """Schema for reading user question data (response)."""

    uuid: str | None = Field(default=None, description="Question UUID (globally unique)")
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


# ============================================================================
# Create Schemas (POST bodies)
# ============================================================================
# user_*.api_key is NOT NULL, so creating a config without one must be a 422,
# not a database error. Updates keep api_key optional: omitted means "keep".
# An empty string stays allowed for providers that need no key.


class UserEmbeddingCreateSchema(UserEmbeddingSchema):
    """Request body for creating an embedding config."""

    api_key: str = Field(..., max_length=STRING_MAX_LENGTH, exclude=True)


class UserLLMCreateSchema(UserLLMSchema):
    """Request body for creating an LLM config."""

    api_key: str = Field(..., max_length=STRING_MAX_LENGTH, exclude=True)


class UserASRCreateSchema(UserASRSchema):
    """Request body for creating an ASR config."""

    api_key: str = Field(..., max_length=STRING_MAX_LENGTH, exclude=True)


class UserTTSCreateSchema(UserTTSSchema):
    """Request body for creating a TTS config."""

    api_key: str = Field(..., max_length=STRING_MAX_LENGTH, exclude=True)
