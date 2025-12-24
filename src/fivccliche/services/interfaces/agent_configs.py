from abc import abstractmethod

from fivcglue import IComponent

from fivcplayground.embeddings import (
    EmbeddingBackend as UserEmbeddingBackend,
    EmbeddingConfigRepository as UserEmbeddingRepository,
)
from fivcplayground.models import (
    ModelBackend as UserLLMBackend,
    ModelConfigRepository as UserLLMRepository,
)
from fivcplayground.tools import (
    ToolBackend as UserToolBackend,
    ToolConfigRepository as UserToolRepository,
)
from fivcplayground.agents import (
    AgentBackend as UserAgentBackend,
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
    def get_embedding_backend(
        self,
        user_uuid: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserEmbeddingBackend:
        """Get the embedding backend."""

    @abstractmethod
    def get_model_repository(
        self,
        user_uuid: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserLLMRepository:
        """Get the model config repository."""

    @abstractmethod
    def get_model_backend(
        self,
        user_uuid: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserLLMBackend:
        """Get the model backend."""

    @abstractmethod
    def get_tool_repository(
        self,
        user_uuid: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserToolRepository:
        """Get the tool config repository."""

    @abstractmethod
    def get_tool_backend(
        self,
        user_uuid: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserToolBackend:
        """Get the tool backend."""

    @abstractmethod
    def get_agent_repository(
        self,
        user_uuid: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserAgentRepository:
        """Get the agent config repository."""

    @abstractmethod
    def get_agent_backend(
        self,
        user_uuid: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserAgentBackend:
        """Get the agent backend."""
