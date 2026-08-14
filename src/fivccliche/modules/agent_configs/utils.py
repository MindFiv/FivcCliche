"""Shared SQL for user-scoped configs (uuid/id lookup, create/update field mapping)."""

from datetime import datetime, timezone
from typing import Any, TypeVar, cast

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel, col, select

from fivccliche.utils.filters import FilterSet

from . import models, schemas

TConfig = TypeVar("TConfig", bound=SQLModel)


async def get_user_scoped_async(
    session: AsyncSession,
    model: type[TConfig],
    *,
    filters: FilterSet,
    config_uuid: str | None = None,
    config_id: str | None = None,
) -> TConfig | None:
    """Get a user-scoped config by uuid or id (visibility via ``filters``)."""
    if (config_uuid is None) == (config_id is None):
        raise ValueError("Exactly one of config_uuid or config_id must be provided")

    table = cast(Any, model)
    identity = table.uuid == config_uuid if config_uuid is not None else table.id == config_id
    statement = select(model).where(identity)
    statement = filters.filter(statement)
    result = await session.execute(statement)
    return result.scalars().first()


async def list_user_scoped_async(
    session: AsyncSession,
    model: type[TConfig],
    *,
    filters: FilterSet,
    skip: int = 0,
    limit: int = 100,
) -> list[TConfig]:
    """List user-scoped configs ordered by id (visibility via ``filters``)."""
    table = cast(Any, model)
    statement = select(model).order_by(col(table.id).asc()).offset(skip).limit(limit)
    statement = filters.filter(statement)
    result = await session.execute(statement)
    return list(result.scalars().all())


async def count_user_scoped_async(
    session: AsyncSession,
    model: type[TConfig],
    *,
    filters: FilterSet,
) -> int:
    """Count user-scoped configs (visibility via ``filters``)."""
    table = cast(Any, model)
    statement = select(func.count(col(table.uuid)))
    statement = filters.filter(statement)
    result = await session.execute(statement)
    return result.scalar() or 0


async def create_embedding_config_async(
    session: AsyncSession,
    user_uuid: str | None,
    config_create: schemas.UserEmbeddingSchema,
    updated_user_uuid: str | None = None,
) -> models.UserEmbedding:
    """Create a new embedding config."""
    config = models.UserEmbedding(
        id=config_create.id,
        user_uuid=user_uuid,
        description=config_create.description,
        provider=config_create.provider,
        model=config_create.model,
        api_key=config_create.api_key,
        base_url=config_create.base_url,
        dimension=config_create.dimension,
        updated_at=datetime.now(timezone.utc),
        updated_user_uuid=updated_user_uuid,
    )
    session.add(config)
    return config


async def update_embedding_config_async(
    session: AsyncSession,
    config: models.UserEmbedding,
    config_update: schemas.UserEmbeddingSchema,
    updated_user_uuid: str | None = None,
) -> models.UserEmbedding:
    """Update an embedding config."""
    fields_set: set[str] = getattr(config_update, "model_fields_set", set())
    if "description" in fields_set:
        config.description = config_update.description
    if config_update.provider is not None:
        config.provider = config_update.provider
    if config_update.model is not None:
        config.model = config_update.model
    if config_update.api_key is not None:
        config.api_key = config_update.api_key
    if "base_url" in fields_set:
        config.base_url = config_update.base_url
    if config_update.dimension is not None:
        config.dimension = config_update.dimension
    config.updated_at = datetime.now(timezone.utc)
    config.updated_user_uuid = updated_user_uuid
    session.add(config)
    return config


async def create_llm_config_async(
    session: AsyncSession,
    user_uuid: str | None,
    config_create: schemas.UserLLMSchema,
    updated_user_uuid: str | None = None,
) -> models.UserLLM:
    """Create a new LLM config."""
    config = models.UserLLM(
        id=config_create.id,
        user_uuid=user_uuid,
        description=config_create.description,
        provider=config_create.provider,
        model=config_create.model,
        api_key=config_create.api_key,
        base_url=config_create.base_url,
        temperature=config_create.temperature,
        max_tokens=config_create.max_tokens,
        enable_thinking=config_create.enable_thinking,
        updated_at=datetime.now(timezone.utc),
        updated_user_uuid=updated_user_uuid,
    )
    session.add(config)
    return config


async def update_llm_config_async(
    session: AsyncSession,
    config: models.UserLLM,
    config_update: schemas.UserLLMSchema,
    updated_user_uuid: str | None = None,
) -> models.UserLLM:
    """Update an LLM config."""
    fields_set: set[str] = getattr(config_update, "model_fields_set", set())
    if "description" in fields_set:
        config.description = config_update.description
    if config_update.provider is not None:
        config.provider = config_update.provider
    if config_update.model is not None:
        config.model = config_update.model
    if config_update.api_key is not None:
        config.api_key = config_update.api_key
    if "base_url" in fields_set:
        config.base_url = config_update.base_url
    if config_update.temperature is not None:
        config.temperature = config_update.temperature
    if config_update.max_tokens is not None:
        config.max_tokens = config_update.max_tokens
    if "enable_thinking" in fields_set:
        config.enable_thinking = config_update.enable_thinking
    config.updated_at = datetime.now(timezone.utc)
    config.updated_user_uuid = updated_user_uuid
    session.add(config)
    return config


async def create_agent_config_async(
    session: AsyncSession,
    user_uuid: str | None,
    config_create: schemas.UserAgentSchema,
    updated_user_uuid: str | None = None,
) -> models.UserAgent:
    """Create a new agent config."""
    config = models.UserAgent(
        id=config_create.id,
        user_uuid=user_uuid,
        description=config_create.description,
        model_id=config_create.model_id,
        tools_ids=config_create.tool_ids,
        skill_ids=config_create.skill_ids,
        system_prompt=config_create.system_prompt,
        response_format=config_create.response_format,
        is_frozen=config_create.is_frozen if hasattr(config_create, "is_frozen") else False,
        updated_at=datetime.now(timezone.utc),
        updated_user_uuid=updated_user_uuid,
    )
    session.add(config)
    return config


async def update_agent_config_async(
    session: AsyncSession,
    config: models.UserAgent,
    config_update: schemas.UserAgentSchema,
    updated_user_uuid: str | None = None,
) -> models.UserAgent:
    """Update an agent config."""
    fields_set: set[str] = getattr(config_update, "model_fields_set", set())
    if "description" in fields_set:
        config.description = config_update.description
    if config_update.model_id is not None:
        config.model_id = config_update.model_id
    if "tool_ids" in fields_set:
        config.tools_ids = config_update.tool_ids
    if "skill_ids" in fields_set:
        config.skill_ids = config_update.skill_ids
    if "system_prompt" in fields_set:
        config.system_prompt = config_update.system_prompt
    if "response_format" in fields_set:
        config.response_format = config_update.response_format
    if hasattr(config_update, "is_frozen") and config_update.is_frozen is not None:
        config.is_frozen = config_update.is_frozen
    config.updated_at = datetime.now(timezone.utc)
    config.updated_user_uuid = updated_user_uuid
    session.add(config)
    return config


async def create_tool_config_async(
    session: AsyncSession,
    user_uuid: str | None,
    config_create: schemas.UserToolSchema,
    updated_user_uuid: str | None = None,
) -> models.UserTool:
    """Create a new tool config."""
    config = models.UserTool(
        id=config_create.id,
        user_uuid=user_uuid,
        description=config_create.description,
        transport=config_create.transport,
        command=config_create.command,
        args=config_create.args,
        env=config_create.env,
        url=config_create.url,
        functions=config_create.functions,
        is_active=config_create.is_active if hasattr(config_create, "is_active") else True,
        updated_at=datetime.now(timezone.utc),
        updated_user_uuid=updated_user_uuid,
    )
    session.add(config)
    return config


async def update_tool_config_async(
    session: AsyncSession,
    config: models.UserTool,
    config_update: schemas.UserToolSchema,
    updated_user_uuid: str | None = None,
) -> models.UserTool:
    """Update a tool config."""
    fields_set: set[str] = getattr(config_update, "model_fields_set", set())
    if "description" in fields_set:
        config.description = config_update.description
    if config_update.transport is not None:
        config.transport = config_update.transport
    if "command" in fields_set:
        config.command = config_update.command
    if "args" in fields_set:
        config.args = config_update.args
    if "env" in fields_set:
        config.env = config_update.env
    if "url" in fields_set:
        config.url = config_update.url
    if "functions" in fields_set:
        config.functions = config_update.functions
    if hasattr(config_update, "is_active") and config_update.is_active is not None:
        config.is_active = config_update.is_active
    config.updated_at = datetime.now(timezone.utc)
    config.updated_user_uuid = updated_user_uuid
    session.add(config)
    return config


async def create_skill_config_async(
    session: AsyncSession,
    user_uuid: str | None,
    config_create: schemas.UserSkillSchema,
    updated_user_uuid: str | None = None,
) -> models.UserSkill:
    """Create a new skill config."""
    config = models.UserSkill(
        id=config_create.id,
        user_uuid=user_uuid,
        description=config_create.description,
        instructions=config_create.instructions,
        tool_ids=config_create.tool_ids,
        resources=config_create.resources,
        is_active=config_create.is_active if hasattr(config_create, "is_active") else True,
        updated_at=datetime.now(timezone.utc),
        updated_user_uuid=updated_user_uuid,
    )
    session.add(config)
    return config


async def update_skill_config_async(
    session: AsyncSession,
    config: models.UserSkill,
    config_update: schemas.UserSkillSchema,
    updated_user_uuid: str | None = None,
) -> models.UserSkill:
    """Update a skill config."""
    fields_set: set[str] = getattr(config_update, "model_fields_set", set())
    if config_update.description is not None:
        config.description = config_update.description
    if "instructions" in fields_set:
        config.instructions = config_update.instructions
    if "tool_ids" in fields_set:
        config.tool_ids = config_update.tool_ids
    if "resources" in fields_set:
        config.resources = config_update.resources
    if hasattr(config_update, "is_active") and config_update.is_active is not None:
        config.is_active = config_update.is_active
    config.updated_at = datetime.now(timezone.utc)
    config.updated_user_uuid = updated_user_uuid
    session.add(config)
    return config
