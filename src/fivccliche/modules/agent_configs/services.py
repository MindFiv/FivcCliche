import asyncio
import importlib.util
from pathlib import Path
from typing import cast

from fastapi import FastAPI

from fivcglue import IComponentSite
from fivcplayground.skills import SkillConfigRepository as UserSkillRepository

from fivcplayground.embeddings.types import EmbeddingConfig
from fivcplayground.tools.types import ToolConfig
from fivcplayground.models.types import ModelConfig
from fivcplayground.agents.types import AgentConfig
from fivcplayground.skills.types import SkillConfig

from fivcplayground.backends.chroma import (
    ChromaEmbeddingBackend,
)

from fivccliche.services.interfaces.modules import IModule, IModuleJob
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

from fivccliche.utils.deps import get_db_session_context_async

from . import methods, routers


def _load_playground_backend(package: str, module_name: str, class_name: str):
    """Load a backend class without executing ``backends/<package>/__init__.py``.

    A normal ``from fivcplayground.backends.strands.tools import ...`` still runs
    that package ``__init__``, which re-exports models/tools/agents together.
    """
    import fivcplayground

    backend_path = Path(fivcplayground.__file__).parent / "backends" / package / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(
        f"_fivccliche_{package}_{module_name}", backend_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load backend {package}.{module_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)()


class _LazyPlaygroundBackend:
    """Instantiate a playground backend only when a method is first used."""

    def __init__(self, package: str, module_name: str, class_name: str):
        self._package = package
        self._module_name = module_name
        self._class_name = class_name
        self._backend = None

    def _load(self):
        if self._backend is None:
            self._backend = _load_playground_backend(
                self._package, self._module_name, self._class_name
            )
        return self._backend

    def __getattr__(self, name: str):
        return getattr(self._load(), name)


class _UserScopedConfigRepository:
    """User-scoped config repository; each DB call opens a short-lived session."""

    def __init__(self, user_uuid: str):
        self.user_uuid = user_uuid


class UserEmbeddingRepositoryImpl(_UserScopedConfigRepository, UserEmbeddingRepository):
    """Embedding config repository implementation."""

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
        async with get_db_session_context_async() as db_session:
            # Check if config exists by ID
            existing = await methods.get_embedding_config_async(
                db_session, self.user_uuid, config_id=embedding_config.id
            )
            if existing:
                # Update existing config
                await methods.update_embedding_config_async(db_session, existing, embedding_config)
            else:
                # Create new config
                await methods.create_embedding_config_async(
                    db_session, self.user_uuid, embedding_config
                )

    async def get_embedding_config_async(self, embedding_id: str) -> EmbeddingConfig | None:
        """Retrieve an embedding configuration by ID."""
        async with get_db_session_context_async() as db_session:
            config = await methods.get_embedding_config_async(
                db_session, self.user_uuid, config_id=embedding_id
            )
            return config.to_schema() if config else None

    async def list_embedding_configs_async(self, **kwargs) -> list[EmbeddingConfig]:
        """List all embedding configurations in the repository."""
        async with get_db_session_context_async() as db_session:
            skip = kwargs.get("skip", 0)
            limit = kwargs.get("limit", 100)
            configs = await methods.list_embedding_configs_async(
                db_session, self.user_uuid, skip=skip, limit=limit
            )
            return [config.to_schema() for config in configs]

    async def delete_embedding_config_async(self, embedding_id: str) -> None:
        """Delete an embedding configuration."""
        async with get_db_session_context_async() as db_session:
            config = await methods.get_embedding_config_async(
                db_session, self.user_uuid, config_id=embedding_id
            )
            if config:
                await methods.delete_embedding_config_async(db_session, config)


class UserLLMRepositoryImpl(_UserScopedConfigRepository, UserLLMRepository):
    """LLM config repository implementation."""

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
        async with get_db_session_context_async() as db_session:
            # Check if config exists by ID
            existing = await methods.get_llm_config_async(
                db_session, self.user_uuid, config_id=model_config.id
            )
            if existing:
                # Update existing config
                await methods.update_llm_config_async(db_session, existing, model_config)
            else:
                # Create new config
                await methods.create_llm_config_async(db_session, self.user_uuid, model_config)

    async def get_model_config_async(self, model_id: str) -> ModelConfig | None:
        """Retrieve a model configuration by ID."""
        async with get_db_session_context_async() as db_session:
            config = await methods.get_llm_config_async(
                db_session, self.user_uuid, config_id=model_id
            )
            return config.to_schema() if config else None

    async def list_model_configs_async(self, **kwargs) -> list[ModelConfig]:
        """List all model configurations in the repository."""
        async with get_db_session_context_async() as db_session:
            skip = kwargs.get("skip", 0)
            limit = kwargs.get("limit", 100)
            configs = await methods.list_llm_configs_async(
                db_session, self.user_uuid, skip=skip, limit=limit
            )
            return [config.to_schema() for config in configs]

    async def delete_model_config_async(self, model_id: str) -> None:
        """Delete a model configuration."""
        async with get_db_session_context_async() as db_session:
            config = await methods.get_llm_config_async(
                db_session, self.user_uuid, config_id=model_id
            )
            if config:
                await methods.delete_llm_config_async(db_session, config)


class UserToolRepositoryImpl(_UserScopedConfigRepository, UserToolRepository):
    """Tool config repository implementation."""

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
        async with get_db_session_context_async() as db_session:
            # Check if config exists by ID
            existing = await methods.get_tool_config_async(
                db_session, self.user_uuid, config_id=tool_config.id
            )
            if existing and not existing.is_active:
                raise RuntimeError("Cannot update inactive tool config")

            if existing:
                # Update existing config
                await methods.update_tool_config_async(db_session, existing, tool_config)
            else:
                # Create new config
                await methods.create_tool_config_async(db_session, self.user_uuid, tool_config)

    async def get_tool_config_async(self, tool_id: str):
        """Retrieve a tool configuration by ID."""
        async with get_db_session_context_async() as db_session:
            config = await methods.get_tool_config_async(
                db_session, self.user_uuid, config_id=tool_id
            )
            return config.to_schema() if config and config.is_active else None

    async def list_tool_configs_async(self, **kwargs) -> list:
        """List all tool configurations in the repository."""
        async with get_db_session_context_async() as db_session:
            skip = kwargs.get("skip", 0)
            limit = kwargs.get("limit", 1000)
            configs = await methods.list_tool_configs_async(
                db_session, self.user_uuid, skip=skip, limit=limit
            )
            return [config.to_schema() for config in configs if config.is_active]

    async def delete_tool_config_async(self, tool_id: str) -> None:
        """Delete a tool configuration."""
        async with get_db_session_context_async() as db_session:
            config = await methods.get_tool_config_async(
                db_session, self.user_uuid, config_id=tool_id
            )
            if config and not config.is_active:
                raise RuntimeError("Cannot delete inactive tool config")
            if config:
                await methods.delete_tool_config_async(db_session, config)


class UserSkillRepositoryImpl(_UserScopedConfigRepository, UserSkillRepository):
    """Skill config repository implementation."""

    def update_skill_config(self, skill_config: SkillConfig) -> None:
        """Create or update a skill configuration."""
        asyncio.run(self.update_skill_config_async(skill_config))

    def get_skill_config(self, skill_id: str) -> SkillConfig | None:
        """Retrieve a skill configuration by ID."""
        return asyncio.run(self.get_skill_config_async(skill_id))

    def list_skill_configs(self, **kwargs) -> list[SkillConfig]:
        """List all skill configurations in the repository."""
        return asyncio.run(self.list_skill_configs_async(**kwargs))

    def delete_skill_config(self, skill_id: str) -> None:
        """Delete a skill configuration."""
        asyncio.run(self.delete_skill_config_async(skill_id))

    async def update_skill_config_async(self, skill_config: SkillConfig) -> None:
        """Create or update a skill configuration."""
        async with get_db_session_context_async() as db_session:
            # Check if config exists by ID
            existing = await methods.get_skill_config_async(
                db_session, self.user_uuid, config_id=skill_config.id
            )
            if existing and not existing.is_active:
                raise RuntimeError("Cannot update inactive skill config")

            if existing:
                # Update existing config
                await methods.update_skill_config_async(db_session, existing, skill_config)
            else:
                # Create new config
                await methods.create_skill_config_async(db_session, self.user_uuid, skill_config)

    async def get_skill_config_async(self, skill_id: str) -> SkillConfig | None:
        """Retrieve a skill configuration by ID."""
        async with get_db_session_context_async() as db_session:
            config = await methods.get_skill_config_async(
                db_session, self.user_uuid, config_id=skill_id
            )
            return config.to_schema() if config and config.is_active else None

    async def list_skill_configs_async(self, **kwargs) -> list[SkillConfig]:
        """List all skill configurations in the repository."""
        async with get_db_session_context_async() as db_session:
            skip = kwargs.get("skip", 0)
            limit = kwargs.get("limit", 1000)
            configs = await methods.list_skill_configs_async(
                db_session, self.user_uuid, skip=skip, limit=limit
            )
            return [config.to_schema() for config in configs if config.is_active]

    async def delete_skill_config_async(self, skill_id: str) -> None:
        """Delete a skill configuration."""
        async with get_db_session_context_async() as db_session:
            config = await methods.get_skill_config_async(
                db_session, self.user_uuid, config_id=skill_id
            )
            if config and not config.is_active:
                raise RuntimeError("Cannot delete inactive skill config")
            if config:
                await methods.delete_skill_config_async(db_session, config)


class UserAgentRepositoryImpl(_UserScopedConfigRepository, UserAgentRepository):
    """Agent config repository implementation."""

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
        async with get_db_session_context_async() as db_session:
            # Check if config exists by ID
            existing = await methods.get_agent_config_async(
                db_session, self.user_uuid, config_id=agent_config.id
            )
            if existing:
                # Update existing config
                await methods.update_agent_config_async(db_session, existing, agent_config)
            else:
                # Create new config
                await methods.create_agent_config_async(db_session, self.user_uuid, agent_config)

    async def get_agent_config_async(self, agent_id: str) -> AgentConfig | None:
        """Retrieve an agent configuration by ID."""
        async with get_db_session_context_async() as db_session:
            config = await methods.get_agent_config_async(
                db_session, self.user_uuid, config_id=agent_id
            )
            return config.to_schema() if config else None

    async def list_agent_configs_async(self) -> list[AgentConfig]:
        """List all agent configurations in the repository."""
        async with get_db_session_context_async() as db_session:
            configs = await methods.list_agent_configs_async(db_session, self.user_uuid)
            return [config.to_schema() for config in configs]

    async def delete_agent_config_async(self, agent_id: str) -> None:
        """Delete an agent configuration."""
        async with get_db_session_context_async() as db_session:
            config = await methods.get_agent_config_async(
                db_session, self.user_uuid, config_id=agent_id
            )
            if config:
                await methods.delete_agent_config_async(db_session, config)


class UserConfigProviderImpl(IUserConfigProvider):
    """Config provider implementation."""

    def __init__(self, component_site: IComponentSite, **kwargs):
        print("configs provider initialized...")
        self.component_site = component_site

    def get_embedding_repository(
        self,
        user_uuid: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserEmbeddingRepository:
        """Get the embedding config repository without binding a DB session."""
        assert user_uuid is not None, "user_uuid is required"
        return UserEmbeddingRepositoryImpl(user_uuid=user_uuid)

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
        **kwargs,  # ignore additional arguments
    ) -> UserLLMRepository:
        """Get the model config repository without binding a DB session."""
        assert user_uuid is not None, "user_uuid is required"
        return UserLLMRepositoryImpl(user_uuid=user_uuid)

    def get_model_backend(
        self,
        user_uuid: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserLLMBackend:
        """Get the model backend."""
        return cast(
            UserLLMBackend,
            _LazyPlaygroundBackend("strands", "models", "StrandsModelBackend"),
        )

    def get_tool_repository(
        self,
        user_uuid: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserToolRepository:
        """Get the tool config repository without binding a DB session."""
        assert user_uuid is not None, "user_uuid is required"
        return UserToolRepositoryImpl(user_uuid=user_uuid)

    def get_tool_backend(
        self,
        user_uuid: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserToolBackend:
        """Get the tool backend."""
        return cast(
            UserToolBackend,
            _LazyPlaygroundBackend("strands", "tools", "StrandsToolBackend"),
        )

    def get_skill_repository(
        self,
        user_uuid: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserSkillRepository:
        """Get the skill config repository without binding a DB session."""
        assert user_uuid is not None, "user_uuid is required"
        return UserSkillRepositoryImpl(user_uuid=user_uuid)

    def get_agent_repository(
        self,
        user_uuid: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserAgentRepository:
        """Get the agent config repository without binding a DB session."""
        assert user_uuid is not None, "user_uuid is required"
        return UserAgentRepositoryImpl(user_uuid=user_uuid)

    def get_agent_backend(
        self,
        user_uuid: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserAgentBackend:
        """Get the agent backend."""
        return cast(
            UserAgentBackend,
            _LazyPlaygroundBackend("strands", "agents", "StrandsAgentBackend"),
        )


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

    def list_jobs(self) -> list[IModuleJob]:
        return []

    def get_job(self, job_name: str) -> IModuleJob | None:
        return None

    def mount(self, app: FastAPI, **kwargs) -> None:
        print("agent_configs module mounted.")
        app.include_router(routers.router_embeddings, **kwargs)
        app.include_router(routers.router_models, **kwargs)
        app.include_router(routers.router_agents, **kwargs)
        app.include_router(routers.router_tools, **kwargs)
        app.include_router(routers.router_skills, **kwargs)
        app.include_router(routers.router_questions, **kwargs)
