__all__ = [
    "UserASRProbeRequest",
    "UserASRSchema",
    "UserAgentSchema",
    "UserEmbeddingSchema",
    "UserLLMSchema",
    "UserQuestionSchema",
    "UserSkillSchema",
    "UserTTSSchema",
    "UserToolSchema",
    "UserToolTransport",
]

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

    api_key: str | None = Field(default=None, exclude=True)
    uuid: str | None = Field(default=None, description="Embedding config UUID (globally unique)")
    user_uuid: str | None = Field(default=None, description="User UUID (read-only)")
    updated_at: datetime | None = Field(default=None, description="Last update time (read-only)")
    updated_user_uuid: str | None = Field(
        default=None, description="UUID of user who last updated (read-only)"
    )

    model_config = ConfigDict(from_attributes=True)


class UserLLMSchema(ModelConfig):
    """Schema for reading LLM config data (response)."""

    api_key: str | None = Field(default=None, exclude=True)
    uuid: str | None = Field(default=None, description="LLM config UUID (globally unique)")
    user_uuid: str | None = Field(default=None, description="User UUID (read-only)")
    updated_at: datetime | None = Field(default=None, description="Last update time (read-only)")
    updated_user_uuid: str | None = Field(
        default=None, description="UUID of user who last updated (read-only)"
    )

    model_config = ConfigDict(from_attributes=True)


class UserASRSchema(BaseModel):
    """Schema for reading ASR config data (response)."""

    id: str = Field(..., description="ASR config ID (unique within user scope)")
    description: str | None = Field(default=None, description="ASR description")
    model: str = Field(..., description="ASR model name")
    base_url: str | None = Field(default=None, description="ASR base URL")
    api_key: str | None = Field(default=None, exclude=True)
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

    url: str | None = Field(default=None, description="Publicly reachable audio URL")
    data_b64: str | None = Field(default=None, description="Base64-encoded audio")
    format: str = Field(default="wav", description="Clip format, such as wav or mp3")
    language: str | None = Field(default=None, description="Optional language hint")
    hotwords: list[str] | None = Field(default=None, description="Optional hotwords")
    context: str | None = Field(default=None, description="Optional recognition context")

    @model_validator(mode="after")
    def require_one_source(self) -> Self:
        has_url = bool(self.url)
        has_data = bool(self.data_b64)
        if has_url == has_data:
            raise ValueError("Exactly one of url or data_b64 is required")
        return self


class UserTTSSchema(BaseModel):
    """Schema for reading TTS config data (response)."""

    id: str = Field(..., description="TTS config ID (unique within user scope)")
    description: str | None = Field(default=None, description="TTS description")
    model: str = Field(..., description="TTS model name")
    base_url: str | None = Field(default=None, description="TTS base URL")
    api_key: str | None = Field(default=None, exclude=True)
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
