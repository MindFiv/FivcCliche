import asyncio
import importlib.util
import logging
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

from . import models, routers, utils

logger = logging.getLogger(__name__)


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
    """User-scoped config repository; each DB call opens a short-lived session.

    HTTP list/get returns inactive tool/skill rows. Playground adapters filter
    ``is_active`` here so agents never bind disabled configs.
    """

    def __init__(self, user_uuid: str):
        self.user_uuid = user_uuid

    async def _upsert_async(self, model, updater, creator, config, *, kind: str | None = None):
        async with get_db_session_context_async() as db_session:
            existing = await utils.get_user_scoped_async(
                db_session, model, self.user_uuid, config_id=config.id
            )
            if kind and existing and not existing.is_active:
                raise RuntimeError(f"Cannot update inactive {kind} config")
            if existing:
                await updater(db_session, existing, config)
            else:
                await creator(db_session, self.user_uuid, config)
            await db_session.commit()

    async def _get_schema_async(self, model, config_id: str, *, active_only: bool = False):
        async with get_db_session_context_async() as db_session:
            config = await utils.get_user_scoped_async(
                db_session, model, self.user_uuid, config_id=config_id
            )
            if not config:
                return None
            if active_only and not config.is_active:
                return None
            return config.to_schema()

    async def _list_schemas_async(
        self, model, *, skip: int = 0, limit: int = 100, active_only: bool = False
    ) -> list:
        async with get_db_session_context_async() as db_session:
            configs = await utils.list_user_scoped_async(
                db_session, model, self.user_uuid, skip=skip, limit=limit
            )
            if active_only:
                configs = [item for item in configs if item.is_active]
            return [item.to_schema() for item in configs]

    async def _delete_async(self, model, config_id: str, *, kind: str | None = None):
        async with get_db_session_context_async() as db_session:
            config = await utils.get_user_scoped_async(
                db_session, model, self.user_uuid, config_id=config_id
            )
            if kind and config and not config.is_active:
                raise RuntimeError(f"Cannot delete inactive {kind} config")
            if config:
                await db_session.delete(config)
                await db_session.commit()


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
        await self._upsert_async(
            models.UserEmbedding,
            utils.update_embedding_config_async,
            utils.create_embedding_config_async,
            embedding_config,
        )

    async def get_embedding_config_async(self, embedding_id: str) -> EmbeddingConfig | None:
        """Retrieve an embedding configuration by ID."""
        return await self._get_schema_async(models.UserEmbedding, embedding_id)

    async def list_embedding_configs_async(self, **kwargs) -> list[EmbeddingConfig]:
        """List all embedding configurations in the repository."""
        return cast(
            list[EmbeddingConfig],
            await self._list_schemas_async(
                models.UserEmbedding,
                skip=kwargs.get("skip", 0),
                limit=kwargs.get("limit", 100),
            ),
        )

    async def delete_embedding_config_async(self, embedding_id: str) -> None:
        """Delete an embedding configuration."""
        await self._delete_async(models.UserEmbedding, embedding_id)


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
        await self._upsert_async(
            models.UserLLM,
            utils.update_llm_config_async,
            utils.create_llm_config_async,
            model_config,
        )

    async def get_model_config_async(self, model_id: str) -> ModelConfig | None:
        """Retrieve a model configuration by ID."""
        return await self._get_schema_async(models.UserLLM, model_id)

    async def list_model_configs_async(self, **kwargs) -> list[ModelConfig]:
        """List all model configurations in the repository."""
        return cast(
            list[ModelConfig],
            await self._list_schemas_async(
                models.UserLLM,
                skip=kwargs.get("skip", 0),
                limit=kwargs.get("limit", 100),
            ),
        )

    async def delete_model_config_async(self, model_id: str) -> None:
        """Delete a model configuration."""
        await self._delete_async(models.UserLLM, model_id)


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
        await self._upsert_async(
            models.UserTool,
            utils.update_tool_config_async,
            utils.create_tool_config_async,
            tool_config,
            kind="tool",
        )

    async def get_tool_config_async(self, tool_id: str):
        """Retrieve a tool configuration by ID."""
        return await self._get_schema_async(models.UserTool, tool_id, active_only=True)

    async def list_tool_configs_async(self, **kwargs) -> list:
        """List all tool configurations in the repository."""
        return cast(
            list,
            await self._list_schemas_async(
                models.UserTool,
                skip=kwargs.get("skip", 0),
                limit=kwargs.get("limit", 1000),
                active_only=True,
            ),
        )

    async def delete_tool_config_async(self, tool_id: str) -> None:
        """Delete a tool configuration."""
        await self._delete_async(models.UserTool, tool_id, kind="tool")


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
        await self._upsert_async(
            models.UserSkill,
            utils.update_skill_config_async,
            utils.create_skill_config_async,
            skill_config,
            kind="skill",
        )

    async def get_skill_config_async(self, skill_id: str) -> SkillConfig | None:
        """Retrieve a skill configuration by ID."""
        return await self._get_schema_async(models.UserSkill, skill_id, active_only=True)

    async def list_skill_configs_async(self, **kwargs) -> list[SkillConfig]:
        """List all skill configurations in the repository."""
        return cast(
            list[SkillConfig],
            await self._list_schemas_async(
                models.UserSkill,
                skip=kwargs.get("skip", 0),
                limit=kwargs.get("limit", 1000),
                active_only=True,
            ),
        )

    async def delete_skill_config_async(self, skill_id: str) -> None:
        """Delete a skill configuration."""
        await self._delete_async(models.UserSkill, skill_id, kind="skill")


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
        await self._upsert_async(
            models.UserAgent,
            utils.update_agent_config_async,
            utils.create_agent_config_async,
            agent_config,
        )

    async def get_agent_config_async(self, agent_id: str) -> AgentConfig | None:
        """Retrieve an agent configuration by ID."""
        return await self._get_schema_async(models.UserAgent, agent_id)

    async def list_agent_configs_async(self) -> list[AgentConfig]:
        """List all agent configurations in the repository."""
        return cast(
            list[AgentConfig],
            await self._list_schemas_async(models.UserAgent),
        )

    async def delete_agent_config_async(self, agent_id: str) -> None:
        """Delete an agent configuration."""
        await self._delete_async(models.UserAgent, agent_id)


class UserConfigProviderImpl(IUserConfigProvider):
    """Config provider implementation."""

    def __init__(self, component_site: IComponentSite, **kwargs):
        logger.info("configs provider initialized")
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
        logger.info("agent configs module initialized")

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
        logger.info("agent_configs module mounted")
        app.include_router(routers.router_embeddings, **kwargs)
        app.include_router(routers.router_models, **kwargs)
        app.include_router(routers.router_agents, **kwargs)
        app.include_router(routers.router_tools, **kwargs)
        app.include_router(routers.router_skills, **kwargs)
        app.include_router(routers.router_questions, **kwargs)
