"""Config service module with functions for config operations."""

from datetime import datetime, timezone
from typing import Any, TypeVar, cast

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel, col, select

from . import models, schemas

TConfig = TypeVar("TConfig", bound=SQLModel)


async def _get_user_scoped_async(
    session: AsyncSession,
    model: type[TConfig],
    user_uuid: str,
    *,
    config_uuid: str | None = None,
    config_id: str | None = None,
) -> TConfig | None:
    """Get a user-scoped config by uuid or id (user-owned or global)."""
    if (config_uuid is None) == (config_id is None):
        raise ValueError("Exactly one of config_uuid or config_id must be provided")

    table = cast(Any, model)
    identity = table.uuid == config_uuid if config_uuid is not None else table.id == config_id
    statement = select(model).where(
        identity,
        (table.user_uuid == user_uuid) | (table.user_uuid == None),  # noqa: E711
    )
    result = await session.execute(statement)
    return result.scalars().first()


async def _list_user_scoped_async(
    session: AsyncSession,
    model: type[TConfig],
    user_uuid: str,
    *,
    skip: int = 0,
    limit: int = 100,
    extra_conditions: list | None = None,
) -> list[TConfig]:
    """List user-owned and global configs, ordered by id."""
    table = cast(Any, model)
    conditions = [(table.user_uuid == user_uuid) | (table.user_uuid == None)]  # noqa: E711
    if extra_conditions:
        conditions.extend(extra_conditions)
    statement = (
        select(model).where(*conditions).order_by(col(table.id).asc()).offset(skip).limit(limit)
    )
    result = await session.execute(statement)
    return list(result.scalars().all())


async def _count_user_scoped_async(
    session: AsyncSession,
    model: type[TConfig],
    user_uuid: str,
    *,
    extra_conditions: list | None = None,
) -> int:
    """Count user-owned and global configs."""
    table = cast(Any, model)
    conditions = [(table.user_uuid == user_uuid) | (table.user_uuid == None)]  # noqa: E711
    if extra_conditions:
        conditions.extend(extra_conditions)
    statement = select(func.count(col(table.uuid))).where(*conditions)
    result = await session.execute(statement)
    return result.scalar() or 0


# ============================================================================
# Embedding Config Operations
# ============================================================================


async def create_embedding_config_async(
    session: AsyncSession,
    user_uuid: str | None,
    config_create: schemas.UserEmbeddingSchema,
    updated_user_uuid: str | None = None,
    **kwargs,  # ignore additional arguments
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
    await session.commit()
    await session.refresh(config)
    return config


async def get_embedding_config_async(
    session: AsyncSession,
    user_uuid: str,
    config_uuid: str | None = None,
    config_id: str | None = None,
    **kwargs,  # ignore additional arguments
) -> models.UserEmbedding | None:
    """Get an embedding config by UUID or ID for a specific user."""
    return await _get_user_scoped_async(
        session,
        models.UserEmbedding,
        user_uuid,
        config_uuid=config_uuid,
        config_id=config_id,
    )


async def list_embedding_configs_async(
    session: AsyncSession,
    user_uuid: str,
    skip: int = 0,
    limit: int = 100,
    **kwargs,  # ignore additional arguments
) -> list[models.UserEmbedding]:
    """List all embedding configs for a user with pagination."""
    return await _list_user_scoped_async(
        session, models.UserEmbedding, user_uuid, skip=skip, limit=limit
    )


async def count_embedding_configs_async(
    session: AsyncSession, user_uuid: str, **kwargs  # ignore additional arguments
) -> int:
    """Count the number of embedding configs for a user."""
    return await _count_user_scoped_async(session, models.UserEmbedding, user_uuid)


async def update_embedding_config_async(
    session: AsyncSession,
    config: models.UserEmbedding,
    config_update: schemas.UserEmbeddingSchema,
    updated_user_uuid: str | None = None,
    **kwargs,  # ignore additional arguments
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
    await session.commit()
    await session.refresh(config)
    return config


async def delete_embedding_config_async(
    session: AsyncSession, config: models.UserEmbedding, **kwargs  # ignore additional arguments
) -> None:
    """Delete an embedding config."""
    await session.delete(config)
    await session.commit()


# ============================================================================
# LLM Config Operations
# ============================================================================


async def create_llm_config_async(
    session: AsyncSession,
    user_uuid: str | None,
    config_create: schemas.UserLLMSchema,
    updated_user_uuid: str | None = None,
    **kwargs,  # ignore additional arguments
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
    await session.commit()
    await session.refresh(config)
    return config


async def get_llm_config_async(
    session: AsyncSession,
    user_uuid: str,
    config_uuid: str | None = None,
    config_id: str | None = None,
    **kwargs,  # ignore additional arguments
) -> models.UserLLM | None:
    """Get an LLM config by UUID or ID for a specific user."""
    return await _get_user_scoped_async(
        session, models.UserLLM, user_uuid, config_uuid=config_uuid, config_id=config_id
    )


async def list_llm_configs_async(
    session: AsyncSession,
    user_uuid: str,
    skip: int = 0,
    limit: int = 100,
    **kwargs,  # ignore additional arguments
) -> list[models.UserLLM]:
    """List all LLM configs for a user with pagination."""
    return await _list_user_scoped_async(session, models.UserLLM, user_uuid, skip=skip, limit=limit)


async def count_llm_configs_async(
    session: AsyncSession, user_uuid: str, **kwargs  # ignore additional arguments
) -> int:
    """Count the number of LLM configs for a user."""
    return await _count_user_scoped_async(session, models.UserLLM, user_uuid)


async def update_llm_config_async(
    session: AsyncSession,
    config: models.UserLLM,
    config_update: schemas.UserLLMSchema,
    updated_user_uuid: str | None = None,
    **kwargs,  # ignore additional arguments
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
    await session.commit()
    await session.refresh(config)
    return config


async def delete_llm_config_async(
    session: AsyncSession, config: models.UserLLM, **kwargs  # ignore additional arguments
) -> None:
    """Delete an LLM config."""
    await session.delete(config)
    await session.commit()


# ============================================================================
# Agent Config Operations
# ============================================================================


async def create_agent_config_async(
    session: AsyncSession,
    user_uuid: str | None,
    config_create: schemas.UserAgentSchema,
    updated_user_uuid: str | None = None,
    **kwargs,  # ignore additional arguments
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
    await session.commit()
    await session.refresh(config)
    return config


async def get_agent_config_async(
    session: AsyncSession,
    user_uuid: str,
    config_uuid: str | None = None,
    config_id: str | None = None,
    **kwargs,  # ignore additional arguments
) -> models.UserAgent | None:
    """Get an agent config by UUID or ID for a specific user."""
    return await _get_user_scoped_async(
        session, models.UserAgent, user_uuid, config_uuid=config_uuid, config_id=config_id
    )


async def list_agent_configs_async(
    session: AsyncSession,
    user_uuid: str,
    skip: int = 0,
    limit: int = 100,
    **kwargs,  # ignore additional arguments
) -> list[models.UserAgent]:
    """List all agent configs for a user with pagination."""
    return await _list_user_scoped_async(
        session, models.UserAgent, user_uuid, skip=skip, limit=limit
    )


async def count_agent_configs_async(
    session: AsyncSession, user_uuid: str, **kwargs  # ignore additional arguments
) -> int:
    """Count the number of agent configs for a user."""
    return await _count_user_scoped_async(session, models.UserAgent, user_uuid)


async def update_agent_config_async(
    session: AsyncSession,
    config: models.UserAgent,
    config_update: schemas.UserAgentSchema,
    updated_user_uuid: str | None = None,
    **kwargs,  # ignore additional arguments
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
    await session.commit()
    await session.refresh(config)
    return config


async def delete_agent_config_async(
    session: AsyncSession, config: models.UserAgent, **kwargs  # ignore additional arguments
) -> None:
    """Delete an agent config."""
    await session.delete(config)
    await session.commit()


# ============================================================================
# Tool Config Operations
# ============================================================================


async def create_tool_config_async(
    session: AsyncSession,
    user_uuid: str | None,
    config_create: schemas.UserToolSchema,
    updated_user_uuid: str | None = None,
    **kwargs,  # ignore additional arguments
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
    await session.commit()
    await session.refresh(config)
    return config


async def get_tool_config_async(
    session: AsyncSession,
    user_uuid: str,
    config_uuid: str | None = None,
    config_id: str | None = None,
    **kwargs,  # ignore additional arguments
) -> models.UserTool | None:
    """Get a tool config by UUID or ID for a specific user."""
    return await _get_user_scoped_async(
        session, models.UserTool, user_uuid, config_uuid=config_uuid, config_id=config_id
    )


async def list_tool_configs_async(
    session: AsyncSession,
    user_uuid: str,
    skip: int = 0,
    limit: int = 100,
    **kwargs,  # ignore additional arguments
) -> list[models.UserTool]:
    """List all tool configs for a user with pagination."""
    return await _list_user_scoped_async(
        session, models.UserTool, user_uuid, skip=skip, limit=limit
    )


async def count_tool_configs_async(
    session: AsyncSession, user_uuid: str, **kwargs  # ignore additional arguments
) -> int:
    """Count the number of tool configs for a user."""
    return await _count_user_scoped_async(session, models.UserTool, user_uuid)


async def update_tool_config_async(
    session: AsyncSession,
    config: models.UserTool,
    config_update: schemas.UserToolSchema,
    updated_user_uuid: str | None = None,
    **kwargs,  # ignore additional arguments
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
    await session.commit()
    await session.refresh(config)
    return config


async def delete_tool_config_async(
    session: AsyncSession, config: models.UserTool, **kwargs  # ignore additional arguments
) -> None:
    """Delete a tool config."""
    await session.delete(config)
    await session.commit()


# ============================================================================
# Skill Config Operations
# ============================================================================


async def create_skill_config_async(
    session: AsyncSession,
    user_uuid: str | None,
    config_create: schemas.UserSkillSchema,
    updated_user_uuid: str | None = None,
    **kwargs,  # ignore additional arguments
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
    await session.commit()
    await session.refresh(config)
    return config


async def get_skill_config_async(
    session: AsyncSession,
    user_uuid: str,
    config_uuid: str | None = None,
    config_id: str | None = None,
    **kwargs,  # ignore additional arguments
) -> models.UserSkill | None:
    """Get a skill config by UUID or ID for a specific user."""
    return await _get_user_scoped_async(
        session, models.UserSkill, user_uuid, config_uuid=config_uuid, config_id=config_id
    )


async def list_skill_configs_async(
    session: AsyncSession,
    user_uuid: str,
    skip: int = 0,
    limit: int = 100,
    **kwargs,  # ignore additional arguments
) -> list[models.UserSkill]:
    """List all skill configs for a user with pagination."""
    return await _list_user_scoped_async(
        session, models.UserSkill, user_uuid, skip=skip, limit=limit
    )


async def count_skill_configs_async(
    session: AsyncSession, user_uuid: str, **kwargs  # ignore additional arguments
) -> int:
    """Count the number of skill configs for a user."""
    return await _count_user_scoped_async(session, models.UserSkill, user_uuid)


async def update_skill_config_async(
    session: AsyncSession,
    config: models.UserSkill,
    config_update: schemas.UserSkillSchema,
    updated_user_uuid: str | None = None,
    **kwargs,  # ignore additional arguments
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
    await session.commit()
    await session.refresh(config)
    return config


async def delete_skill_config_async(
    session: AsyncSession, config: models.UserSkill, **kwargs  # ignore additional arguments
) -> None:
    """Delete a skill config."""
    await session.delete(config)
    await session.commit()


# ============================================================================
# Question Config Operations
# ============================================================================


async def create_question_async(
    session: AsyncSession,
    user_uuid: str | None,
    config_create: schemas.UserQuestionSchema,
    updated_user_uuid: str | None = None,
    **kwargs,  # ignore additional arguments
) -> models.UserQuestion:
    """Create a new question config."""
    config = models.UserQuestion(
        id=config_create.id,
        user_uuid=user_uuid,
        question=config_create.question,
        answer=config_create.answer,
        is_active=config_create.is_active if hasattr(config_create, "is_active") else False,
        updated_at=datetime.now(timezone.utc),
        updated_user_uuid=updated_user_uuid,
    )
    session.add(config)
    await session.commit()
    await session.refresh(config)
    return config


async def get_question_async(
    session: AsyncSession,
    user_uuid: str,
    config_uuid: str | None = None,
    config_id: str | None = None,
    **kwargs,  # ignore additional arguments
) -> models.UserQuestion | None:
    """Get a question config by UUID or ID for a specific user."""
    return await _get_user_scoped_async(
        session, models.UserQuestion, user_uuid, config_uuid=config_uuid, config_id=config_id
    )


async def list_questions_async(
    session: AsyncSession,
    user_uuid: str,
    skip: int = 0,
    limit: int = 100,
    is_active: bool | None = None,
    **kwargs,  # ignore additional arguments
) -> list[models.UserQuestion]:
    """List all question configs for a user with pagination."""
    extra = [models.UserQuestion.is_active == is_active] if is_active is not None else None
    return await _list_user_scoped_async(
        session,
        models.UserQuestion,
        user_uuid,
        skip=skip,
        limit=limit,
        extra_conditions=extra,
    )


async def count_questions_async(
    session: AsyncSession,
    user_uuid: str,
    is_active: bool | None = None,
    **kwargs,  # ignore additional arguments
) -> int:
    """Count the number of question configs for a user."""
    extra = [models.UserQuestion.is_active == is_active] if is_active is not None else None
    return await _count_user_scoped_async(
        session, models.UserQuestion, user_uuid, extra_conditions=extra
    )


async def update_question_async(
    session: AsyncSession,
    config: models.UserQuestion,
    config_update: schemas.UserQuestionSchema,
    updated_user_uuid: str | None = None,
    **kwargs,  # ignore additional arguments
) -> models.UserQuestion:
    """Update a question config."""
    fields_set: set[str] = getattr(config_update, "model_fields_set", set())
    if "question" in fields_set and config_update.question is not None:
        config.question = config_update.question
    if "answer" in fields_set:
        config.answer = config_update.answer
    if "is_active" in fields_set and config_update.is_active is not None:
        config.is_active = config_update.is_active
    config.updated_at = datetime.now(timezone.utc)
    config.updated_user_uuid = updated_user_uuid
    session.add(config)
    await session.commit()
    await session.refresh(config)
    return config


async def delete_question_async(
    session: AsyncSession, config: models.UserQuestion, **kwargs  # ignore additional arguments
) -> None:
    """Delete a question config."""
    await session.delete(config)
    await session.commit()
