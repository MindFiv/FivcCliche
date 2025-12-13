from abc import abstractmethod

from fivcglue import IComponent

from fivcplayground.embeddings.types.repositories import (
    EmbeddingConfigRepository as UserEmbeddingRepository,
)
from fivcplayground.models.types.repositories import (
    ModelConfigRepository as UserLLMRepository,
)
from fivcplayground.tools.types.repositories import (
    ToolConfigRepository as UserToolRepository,
)
from fivcplayground.agents.types.repositories import (
    AgentConfigRepository as UserAgentRepository,
)


class IUserConfigProvider(IComponent):
    """IUserConfigProvider is an interface for defining user config providers."""

    @abstractmethod
    def get_embedding_repository(
        self,
        user_uuid: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserEmbeddingRepository:
        """Get the embedding config repository."""

    @abstractmethod
    def get_model_repository(
        self,
        user_uuid: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserLLMRepository:
        """Get the model config repository."""

    @abstractmethod
    def get_tool_repository(
        self,
        user_uuid: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserToolRepository:
        """Get the tool config repository."""

    @abstractmethod
    def get_agent_repository(
        self,
        user_uuid: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserAgentRepository:
        """Get the agent config repository."""
