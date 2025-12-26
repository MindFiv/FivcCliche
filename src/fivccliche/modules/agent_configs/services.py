import asyncio
from fastapi import FastAPI

from fivcglue import IComponentSite
from sqlalchemy.ext.asyncio.session import AsyncSession

from fivcplayground.embeddings.types import EmbeddingConfig
from fivcplayground.tools.types import ToolConfig
from fivcplayground.models.types import ModelConfig
from fivcplayground.agents.types import AgentConfig

from fivcplayground.backends.chroma import (
    ChromaEmbeddingBackend,
)
from fivcplayground.backends.strands import (
    StrandsModelBackend,
    StrandsToolBackend,
    StrandsAgentBackend,
)

from fivccliche.services.interfaces.modules import IModule
from fivccliche.services.interfaces.agent_configs import (
    UserEmbeddingRepository,
    UserEmbeddingBackend,
    UserToolRepository,
    UserToolBackend,
    UserLLMRepository,
    UserLLMBackend,
    UserAgentRepository,
    UserAgentBackend,
    IUserConfigProvider,
)

from . import methods, routers


class UserEmbeddingRepositoryImpl(UserEmbeddingRepository):
    """Embedding config repository implementation."""

    def __init__(self, user_uuid: str | None = None, session: AsyncSession | None = None):
        self.user_uuid = user_uuid
        self.session = session

    def update_embedding_config(self, embedding_config: EmbeddingConfig) -> None:
        """Create or update an embedding configuration."""
        asyncio.run(self.update_embedding_config_async(embedding_config))

    def get_embedding_config(self, embedding_id: str) -> EmbeddingConfig | None:
        """Retrieve an embedding configuration by ID."""
        return asyncio.run(self.get_embedding_config_async(embedding_id))

    def list_embedding_configs(self, **kwargs) -> list[EmbeddingConfig]:
        """List all embedding configurations in the repository."""
        return asyncio.run(self.list_embedding_configs_async(**kwargs))

    def delete_embedding_config(self, embedding_id: str) -> None:
        """Delete an embedding configuration."""
        asyncio.run(self.delete_embedding_config_async(embedding_id))

    # Abstract methods from fivcplayground.embeddings.types.repositories.EmbeddingConfigRepository
    async def update_embedding_config_async(self, embedding_config: EmbeddingConfig) -> None:
        """Create or update an embedding configuration."""
        if not self.session or not self.user_uuid:
            raise ValueError(
                "Session and user_uuid are required for update_embedding_config operation"
            )
        # Check if config exists by ID
        existing = await methods.get_embedding_config_async(
            self.session, self.user_uuid, config_id=embedding_config.id
        )
        if existing:
            # Update existing config
            await methods.update_embedding_config_async(self.session, existing, embedding_config)
        else:
            # Create new config
            await methods.create_embedding_config_async(
                self.session, self.user_uuid, embedding_config
            )

    async def get_embedding_config_async(self, embedding_id: str) -> EmbeddingConfig | None:
        """Retrieve an embedding configuration by ID."""
        if not self.session or not self.user_uuid:
            raise ValueError(
                "Session and user_uuid are required for get_embedding_config operation"
            )
        config = await methods.get_embedding_config_async(
            self.session, self.user_uuid, config_id=embedding_id
        )
        return config.to_schema() if config else None

    async def list_embedding_configs_async(self, **kwargs) -> list[EmbeddingConfig]:
        """List all embedding configurations in the repository."""
        if not self.session or not self.user_uuid:
            raise ValueError(
                "Session and user_uuid are required for list_embedding_configs operation"
            )
        skip = kwargs.get("skip", 0)
        limit = kwargs.get("limit", 100)
        configs = await methods.list_embedding_configs_async(
            self.session, self.user_uuid, skip=skip, limit=limit
        )
        return [config.to_schema() for config in configs]

    async def delete_embedding_config_async(self, embedding_id: str) -> None:
        """Delete an embedding configuration."""
        if not self.session or not self.user_uuid:
            raise ValueError(
                "Session and user_uuid are required for delete_embedding_config operation"
            )
        config = await methods.get_embedding_config_async(
            self.session, self.user_uuid, config_id=embedding_id
        )
        if config:
            await methods.delete_embedding_config_async(self.session, config)


class UserLLMRepositoryImpl(UserLLMRepository):
    """LLM config repository implementation."""

    def __init__(self, user_uuid: str | None = None, session: AsyncSession | None = None):
        self.user_uuid = user_uuid
        self.session = session

    def update_model_config(self, model_config: ModelConfig) -> None:
        """Create or update a model configuration."""
        asyncio.run(self.update_model_config_async(model_config))

    def get_model_config(self, model_id: str) -> ModelConfig | None:
        """Retrieve a model configuration by ID."""
        return asyncio.run(self.get_model_config_async(model_id))

    def list_model_configs(self, **kwargs) -> list[ModelConfig]:
        """List all model configurations in the repository."""
        return asyncio.run(self.list_model_configs_async(**kwargs))

    def delete_model_config(self, model_id: str) -> None:
        """Delete a model configuration."""
        asyncio.run(self.delete_model_config_async(model_id))

    # Abstract methods from fivcplayground.models.types.repositories.ModelConfigRepository
    async def update_model_config_async(self, model_config: ModelConfig) -> None:
        """Create or update a model configuration."""
        if not self.session or not self.user_uuid:
            raise ValueError("Session and user_uuid are required for update_model_config operation")
        # Check if config exists by ID
        existing = await methods.get_llm_config_async(
            self.session, self.user_uuid, config_id=model_config.id
        )
        if existing:
            # Update existing config
            await methods.update_llm_config_async(self.session, existing, model_config)
        else:
            # Create new config
            await methods.create_llm_config_async(self.session, self.user_uuid, model_config)

    async def get_model_config_async(self, model_id: str) -> ModelConfig | None:
        """Retrieve a model configuration by ID."""
        if not self.session or not self.user_uuid:
            raise ValueError("Session and user_uuid are required for get_model_config operation")
        config = await methods.get_llm_config_async(
            self.session, self.user_uuid, config_id=model_id
        )
        return config.to_schema() if config else None

    async def list_model_configs_async(self, **kwargs) -> list[ModelConfig]:
        """List all model configurations in the repository."""
        if not self.session or not self.user_uuid:
            raise ValueError("Session and user_uuid are required for list_model_configs operation")
        skip = kwargs.get("skip", 0)
        limit = kwargs.get("limit", 100)
        configs = await methods.list_llm_configs_async(
            self.session, self.user_uuid, skip=skip, limit=limit
        )
        return [config.to_schema() for config in configs]

    async def delete_model_config_async(self, model_id: str) -> None:
        """Delete a model configuration."""
        if not self.session or not self.user_uuid:
            raise ValueError("Session and user_uuid are required for delete_model_config operation")
        config = await methods.get_llm_config_async(
            self.session, self.user_uuid, config_id=model_id
        )
        if config:
            await methods.delete_llm_config_async(self.session, config)


class UserToolRepositoryImpl(UserToolRepository):
    """Tool config repository implementation."""

    def __init__(self, user_uuid: str | None = None, session: AsyncSession | None = None):
        self.user_uuid = user_uuid
        self.session = session

    def update_tool_config(self, tool_config: ToolConfig) -> None:
        """Create or update a tool configuration."""
        asyncio.run(self.update_tool_config_async(tool_config))

    def get_tool_config(self, tool_id: str):
        """Retrieve a tool configuration by ID."""
        return asyncio.run(self.get_tool_config_async(tool_id))

    def list_tool_configs(self, **kwargs) -> list:
        """List all tool configurations in the repository."""
        return asyncio.run(self.list_tool_configs_async(**kwargs))

    def delete_tool_config(self, tool_id: str) -> None:
        """Delete a tool configuration."""
        asyncio.run(self.delete_tool_config_async(tool_id))

    async def update_tool_config_async(self, tool_config: ToolConfig) -> None:
        """Create or update a tool configuration."""
        if not self.session or not self.user_uuid:
            raise ValueError("Session and user_uuid are required for update_tool_config operation")
        # Check if config exists by ID
        existing = await methods.get_tool_config_async(
            self.session, self.user_uuid, config_id=tool_config.id
        )
        if existing:
            # Update existing config
            await methods.update_tool_config_async(self.session, existing, tool_config)
        else:
            # Create new config
            await methods.create_tool_config_async(self.session, self.user_uuid, tool_config)

    async def get_tool_config_async(self, tool_id: str):
        """Retrieve a tool configuration by ID."""
        if not self.session or not self.user_uuid:
            raise ValueError("Session and user_uuid are required for get_tool_config operation")
        config = await methods.get_tool_config_async(
            self.session, self.user_uuid, config_id=tool_id
        )
        return config.to_schema() if config else None

    async def list_tool_configs_async(self, **kwargs) -> list:
        """List all tool configurations in the repository."""
        if not self.session or not self.user_uuid:
            raise ValueError("Session and user_uuid are required for list_tool_configs operation")
        skip = kwargs.get("skip", 0)
        limit = kwargs.get("limit", 100)
        configs = await methods.list_tool_configs_async(
            self.session, self.user_uuid, skip=skip, limit=limit
        )
        return [config.to_schema() for config in configs]

    async def delete_tool_config_async(self, tool_id: str) -> None:
        """Delete a tool configuration."""
        if not self.session or not self.user_uuid:
            raise ValueError("Session and user_uuid are required for delete_tool_config operation")
        config = await methods.get_tool_config_async(
            self.session, self.user_uuid, config_id=tool_id
        )
        if config:
            await methods.delete_tool_config_async(self.session, config)


class UserAgentRepositoryImpl(UserAgentRepository):
    """Agent config repository implementation."""

    def __init__(self, user_uuid: str | None = None, session: AsyncSession | None = None):
        self.user_uuid = user_uuid
        self.session = session

    def update_agent_config(self, agent_config: AgentConfig) -> None:
        """Create or update an agent configuration."""
        asyncio.run(self.update_agent_config_async(agent_config))

    def get_agent_config(self, agent_id: str) -> AgentConfig | None:
        """Retrieve an agent configuration by ID."""
        return asyncio.run(self.get_agent_config_async(agent_id))

    def list_agent_configs(self) -> list[AgentConfig]:
        """List all agent configurations in the repository."""
        return asyncio.run(self.list_agent_configs_async())

    def delete_agent_config(self, agent_id: str) -> None:
        """Delete an agent configuration."""
        asyncio.run(self.delete_agent_config_async(agent_id))

    # Abstract methods from fivcplayground.agents.types.repositories.AgentConfigRepository
    async def update_agent_config_async(self, agent_config: AgentConfig) -> None:
        """Create or update an agent configuration."""
        if not self.session or not self.user_uuid:
            raise ValueError("Session and user_uuid are required for update_agent_config operation")
        # Check if config exists by ID
        existing = await methods.get_agent_config_async(
            self.session, self.user_uuid, config_id=agent_config.id
        )
        if existing:
            # Update existing config
            await methods.update_agent_config_async(self.session, existing, agent_config)
        else:
            # Create new config
            await methods.create_agent_config_async(self.session, self.user_uuid, agent_config)

    async def get_agent_config_async(self, agent_id: str) -> AgentConfig | None:
        """Retrieve an agent configuration by ID."""
        if not self.session or not self.user_uuid:
            raise ValueError("Session and user_uuid are required for get_agent_config operation")
        config = await methods.get_agent_config_async(
            self.session, self.user_uuid, config_id=agent_id
        )
        return config.to_schema() if config else None

    async def list_agent_configs_async(self) -> list[AgentConfig]:
        """List all agent configurations in the repository."""
        if not self.session or not self.user_uuid:
            raise ValueError("Session and user_uuid are required for list_agent_configs operation")
        configs = await methods.list_agent_configs_async(self.session, self.user_uuid)
        return [config.to_schema() for config in configs]

    async def delete_agent_config_async(self, agent_id: str) -> None:
        """Delete an agent configuration."""
        if not self.session or not self.user_uuid:
            raise ValueError("Session and user_uuid are required for delete_agent_config operation")
        config = await methods.get_agent_config_async(
            self.session, self.user_uuid, config_id=agent_id
        )
        if config:
            await methods.delete_agent_config_async(self.session, config)


class UserConfigProviderImpl(IUserConfigProvider):
    """Config provider implementation."""

    def __init__(self, component_site: IComponentSite, **kwargs):
        print("configs provider initialized...")
        self.component_site = component_site

    def get_embedding_repository(
        self,
        user_uuid: str | None = None,
        session: AsyncSession | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserEmbeddingRepository:
        """Get the embedding config repository."""
        return UserEmbeddingRepositoryImpl(user_uuid=user_uuid, session=session)

    def get_embedding_backend(
        self,
        user_uuid: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserEmbeddingBackend:
        """Get the embedding backend."""
        return ChromaEmbeddingBackend()

    def get_model_repository(
        self,
        user_uuid: str | None = None,
        session: AsyncSession | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserLLMRepository:
        """Get the model config repository."""
        return UserLLMRepositoryImpl(user_uuid=user_uuid, session=session)

    def get_model_backend(
        self,
        user_uuid: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserLLMBackend:
        """Get the model backend."""
        return StrandsModelBackend()

    def get_tool_repository(
        self,
        user_uuid: str | None = None,
        session: AsyncSession | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserToolRepository:
        """Get the tool config repository."""
        return UserToolRepositoryImpl(user_uuid=user_uuid, session=session)

    def get_tool_backend(
        self,
        user_uuid: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserToolBackend:
        """Get the tool backend."""
        return StrandsToolBackend()

    def get_agent_repository(
        self,
        user_uuid: str | None = None,
        session: AsyncSession | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserAgentRepository:
        """Get the agent config repository."""
        return UserAgentRepositoryImpl(user_uuid=user_uuid, session=session)

    def get_agent_backend(
        self,
        user_uuid: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserAgentBackend:
        """Get the agent backend."""
        return StrandsAgentBackend()


class ModuleImpl(IModule):
    """User module implementation."""

    def __init__(self, _: IComponentSite, **kwargs):
        print("agent configs module initialized...")

    @property
    def name(self):
        return "agent_configs"

    @property
    def description(self):
        return "Agent Configs management module."

    def mount(self, app: FastAPI, **kwargs) -> None:
        print("agent_configs module mounted.")
        app.include_router(routers.router_embeddings, **kwargs)
        app.include_router(routers.router_models, **kwargs)
        app.include_router(routers.router_agents, **kwargs)
        app.include_router(routers.router_tools, **kwargs)
