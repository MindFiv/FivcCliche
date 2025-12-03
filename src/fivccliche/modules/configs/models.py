from sqlmodel import SQLModel, Field

from . import schemas


class UserEmbedding(SQLModel, table=True):
    """Embedding configuration model."""

    __tablename__ = "user_embedding"

    id: str = Field(
        primary_key=True,
        max_length=32,
        description="Embedding config ID.",
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
    user_id: str = Field(
        foreign_key="user.id",
        description="User ID.",
    )

    def to_config(self) -> schemas.UserEmbeddingSchema:
        return schemas.UserEmbeddingSchema(
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

    id: str = Field(
        primary_key=True,
        max_length=32,
        description="LLM config ID.",
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
    user_id: str = Field(
        foreign_key="user.id",
        description="User ID.",
    )

    def to_config(self) -> schemas.UserLLMSchema:
        return schemas.UserLLMSchema(
            id=self.id,
            description=self.description,
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


class UserAgent(SQLModel, table=True):
    """Agent configuration model."""

    __tablename__ = "user_agent"

    id: str = Field(
        primary_key=True,
        max_length=32,
        description="Agent config ID.",
    )
    description: str | None = Field(default=None, max_length=1024, description="Agent description.")
    model_id: str = Field(
        foreign_key="user_llm.id",
        description="LLM config ID.",
    )
    system_prompt: str | None = Field(
        default=None,
        max_length=1024,
        description="Agent system prompt.",
    )
    user_id: str = Field(
        foreign_key="user.id",
        description="User ID.",
    )

    def to_config(self) -> schemas.UserAgentSchema:
        return schemas.UserAgentSchema(
            id=self.id,
            description=self.description,
            model_id=self.model_id,
            system_prompt=self.system_prompt,
        )
