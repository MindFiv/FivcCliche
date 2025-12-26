"""Config service module with functions for config operations."""

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from . import models, schemas


# ============================================================================
# Embedding Config Operations
# ============================================================================


async def create_embedding_config_async(
    session: AsyncSession,
    user_uuid: str,
    config_create: schemas.UserEmbeddingSchema,
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
    """Get an embedding config by UUID or ID for a specific user.

    Args:
        session: Database session
        user_uuid: User UUID for filtering
        config_uuid: Global unique identifier (optional)
        config_id: User-scoped identifier (optional)

    Returns:
        UserEmbedding config or None if not found

    Raises:
        ValueError: If both or neither config_uuid and config_id are provided
    """
    if (config_uuid is None and config_id is None) or (
        config_uuid is not None and config_id is not None
    ):
        raise ValueError("Exactly one of config_uuid or config_id must be provided")

    if config_uuid is not None:
        # Query by global unique identifier
        statement = select(models.UserEmbedding).where(
            (models.UserEmbedding.uuid == config_uuid)
            & (
                (models.UserEmbedding.user_uuid == user_uuid)
                | (models.UserEmbedding.user_uuid == None)  # noqa E711
            )
        )
    else:
        # Query by user-scoped identifier
        statement = select(models.UserEmbedding).where(
            (models.UserEmbedding.id == config_id)
            & (
                (models.UserEmbedding.user_uuid == user_uuid)
                | (models.UserEmbedding.user_uuid == None)  # noqa E711
            )
        )

    result = await session.execute(statement)
    return result.scalars().first()


async def list_embedding_configs_async(
    session: AsyncSession,
    user_uuid: str,
    skip: int = 0,
    limit: int = 100,
    **kwargs,  # ignore additional arguments
) -> list[models.UserEmbedding]:
    """List all embedding configs for a user with pagination."""
    statement = (
        select(models.UserEmbedding)
        .where(
            (models.UserEmbedding.user_uuid == user_uuid)
            | (models.UserEmbedding.user_uuid == None)  # noqa E711
        )
        .offset(skip)
        .limit(limit)
    )
    result = await session.execute(statement)
    return list(result.scalars().all())


async def count_embedding_configs_async(
    session: AsyncSession, user_uuid: str, **kwargs  # ignore additional arguments
) -> int:
    """Count the number of embedding configs for a user."""
    statement = select(func.count(models.UserEmbedding.uuid)).where(
        (models.UserEmbedding.user_uuid == user_uuid)
        | (models.UserEmbedding.user_uuid == None)  # noqa E711
    )
    result = await session.execute(statement)
    return result.scalar() or 0


async def update_embedding_config_async(
    session: AsyncSession,
    config: models.UserEmbedding,
    config_update: schemas.UserEmbeddingSchema,
    **kwargs,  # ignore additional arguments
) -> models.UserEmbedding:
    """Update an embedding config."""
    if config_update.description is not None:
        config.description = config_update.description
    if config_update.provider is not None:
        config.provider = config_update.provider
    if config_update.model is not None:
        config.model = config_update.model
    if config_update.api_key is not None:
        config.api_key = config_update.api_key
    if config_update.base_url is not None:
        config.base_url = config_update.base_url
    if config_update.dimension is not None:
        config.dimension = config_update.dimension
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
    user_uuid: str,
    config_create: schemas.UserLLMSchema,
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
    """Get an LLM config by UUID or ID for a specific user.

    Args:
        session: Database session
        user_uuid: User UUID for filtering
        config_uuid: Global unique identifier (optional)
        config_id: User-scoped identifier (optional)

    Returns:
        UserLLM config or None if not found

    Raises:
        ValueError: If both or neither config_uuid and config_id are provided
    """
    if (config_uuid is None and config_id is None) or (
        config_uuid is not None and config_id is not None
    ):
        raise ValueError("Exactly one of config_uuid or config_id must be provided")

    if config_uuid is not None:
        # Query by global unique identifier
        statement = select(models.UserLLM).where(
            (models.UserLLM.uuid == config_uuid)
            & (
                (models.UserLLM.user_uuid == user_uuid)
                | (models.UserLLM.user_uuid == None)  # noqa E711
            )
        )
    else:
        # Query by user-scoped identifier
        statement = select(models.UserLLM).where(
            (models.UserLLM.id == config_id)
            & (
                (models.UserLLM.user_uuid == user_uuid)
                | (models.UserLLM.user_uuid == None)  # noqa E711
            )
        )

    result = await session.execute(statement)
    return result.scalars().first()


async def list_llm_configs_async(
    session: AsyncSession,
    user_uuid: str,
    skip: int = 0,
    limit: int = 100,
    **kwargs,  # ignore additional arguments
) -> list[models.UserLLM]:
    """List all LLM configs for a user with pagination."""
    statement = (
        select(models.UserLLM)
        .where(
            (models.UserLLM.user_uuid == user_uuid)
            | (models.UserLLM.user_uuid == None)  # noqa E711
        )
        .offset(skip)
        .limit(limit)
    )
    result = await session.execute(statement)
    return list(result.scalars().all())


async def count_llm_configs_async(
    session: AsyncSession, user_uuid: str, **kwargs  # ignore additional arguments
) -> int:
    """Count the number of LLM configs for a user."""
    statement = select(func.count(models.UserLLM.uuid)).where(
        (models.UserLLM.user_uuid == user_uuid) | (models.UserLLM.user_uuid == None)  # noqa E711
    )
    result = await session.execute(statement)
    return result.scalar() or 0


async def update_llm_config_async(
    session: AsyncSession,
    config: models.UserLLM,
    config_update: schemas.UserLLMSchema,
    **kwargs,  # ignore additional arguments
) -> models.UserLLM:
    """Update an LLM config."""
    if config_update.description is not None:
        config.description = config_update.description
    if config_update.provider is not None:
        config.provider = config_update.provider
    if config_update.model is not None:
        config.model = config_update.model
    if config_update.api_key is not None:
        config.api_key = config_update.api_key
    if config_update.base_url is not None:
        config.base_url = config_update.base_url
    if config_update.temperature is not None:
        config.temperature = config_update.temperature
    if config_update.max_tokens is not None:
        config.max_tokens = config_update.max_tokens
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
    user_uuid: str,
    config_create: schemas.UserAgentSchema,
    **kwargs,  # ignore additional arguments
) -> models.UserAgent:
    """Create a new agent config."""
    config = models.UserAgent(
        id=config_create.id,
        user_uuid=user_uuid,
        description=config_create.description,
        model_id=config_create.model_id,
        tools_ids=config_create.tool_ids,
        system_prompt=config_create.system_prompt,
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
    """Get an agent config by UUID or ID for a specific user.

    Args:
        session: Database session
        user_uuid: User UUID for filtering
        config_uuid: Global unique identifier (optional)
        config_id: User-scoped identifier (optional)

    Returns:
        UserAgent config or None if not found

    Raises:
        ValueError: If both or neither config_uuid and config_id are provided
    """
    if (config_uuid is None and config_id is None) or (
        config_uuid is not None and config_id is not None
    ):
        raise ValueError("Exactly one of config_uuid or config_id must be provided")

    if config_uuid is not None:
        # Query by global unique identifier
        statement = select(models.UserAgent).where(
            (models.UserAgent.uuid == config_uuid)
            & (
                (models.UserAgent.user_uuid == user_uuid)
                | (models.UserAgent.user_uuid == None)  # noqa E711
            )
        )
    else:
        # Query by user-scoped identifier
        statement = select(models.UserAgent).where(
            (models.UserAgent.id == config_id)
            & (
                (models.UserAgent.user_uuid == user_uuid)
                | (models.UserAgent.user_uuid == None)  # noqa E711
            )
        )

    result = await session.execute(statement)
    return result.scalars().first()


async def list_agent_configs_async(
    session: AsyncSession,
    user_uuid: str,
    skip: int = 0,
    limit: int = 100,
    **kwargs,  # ignore additional arguments
) -> list[models.UserAgent]:
    """List all agent configs for a user with pagination."""
    statement = (
        select(models.UserAgent)
        .where(
            (models.UserAgent.user_uuid == user_uuid)
            | (models.UserAgent.user_uuid == None)  # noqa E711
        )
        .offset(skip)
        .limit(limit)
    )
    result = await session.execute(statement)
    return list(result.scalars().all())


async def count_agent_configs_async(
    session: AsyncSession, user_uuid: str, **kwargs  # ignore additional arguments
) -> int:
    """Count the number of agent configs for a user."""
    statement = select(func.count(models.UserAgent.uuid)).where(
        (models.UserAgent.user_uuid == user_uuid)
        | (models.UserAgent.user_uuid == None)  # noqa E711
    )
    result = await session.execute(statement)
    return result.scalar() or 0


async def update_agent_config_async(
    session: AsyncSession,
    config: models.UserAgent,
    config_update: schemas.UserAgentSchema,
    **kwargs,  # ignore additional arguments
) -> models.UserAgent:
    """Update an agent config."""
    if config_update.description is not None:
        config.description = config_update.description
    if config_update.model_id is not None:
        config.model_id = config_update.model_id
    if config_update.tool_ids is not None:
        config.tools_ids = config_update.tool_ids
    if config_update.system_prompt is not None:
        config.system_prompt = config_update.system_prompt
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
    user_uuid: str,
    config_create: schemas.UserToolSchema,
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
    """Get a tool config by UUID or ID for a specific user.

    Args:
        session: Database session
        user_uuid: User UUID for filtering
        config_uuid: Global unique identifier (optional)
        config_id: User-scoped identifier (optional)

    Returns:
        UserTool config or None if not found

    Raises:
        ValueError: If both or neither config_uuid and config_id are provided
    """
    if (config_uuid is None and config_id is None) or (
        config_uuid is not None and config_id is not None
    ):
        raise ValueError("Exactly one of config_uuid or config_id must be provided")

    if config_uuid is not None:
        # Query by global unique identifier
        statement = select(models.UserTool).where(
            (models.UserTool.uuid == config_uuid)
            & (
                (models.UserTool.user_uuid == user_uuid)
                | (models.UserTool.user_uuid == None)  # noqa E711
            )
        )
    else:
        # Query by user-scoped identifier
        statement = select(models.UserTool).where(
            (models.UserTool.id == config_id)
            & (
                (models.UserTool.user_uuid == user_uuid)
                | (models.UserTool.user_uuid == None)  # noqa E711
            )
        )

    result = await session.execute(statement)
    return result.scalars().first()


async def list_tool_configs_async(
    session: AsyncSession,
    user_uuid: str,
    skip: int = 0,
    limit: int = 100,
    **kwargs,  # ignore additional arguments
) -> list[models.UserTool]:
    """List all tool configs for a user with pagination."""
    statement = (
        select(models.UserTool)
        .where(
            (models.UserTool.user_uuid == user_uuid)
            | (models.UserTool.user_uuid == None)  # noqa E711
        )
        .offset(skip)
        .limit(limit)
    )
    result = await session.execute(statement)
    return list(result.scalars().all())


async def count_tool_configs_async(
    session: AsyncSession, user_uuid: str, **kwargs  # ignore additional arguments
) -> int:
    """Count the number of tool configs for a user."""
    statement = select(func.count(models.UserTool.uuid)).where(
        (models.UserTool.user_uuid == user_uuid) | (models.UserTool.user_uuid == None)  # noqa E711
    )
    result = await session.execute(statement)
    return result.scalar() or 0


async def update_tool_config_async(
    session: AsyncSession,
    config: models.UserTool,
    config_update: schemas.UserToolSchema,
    **kwargs,  # ignore additional arguments
) -> models.UserTool:
    """Update a tool config."""
    if config_update.description is not None:
        config.description = config_update.description
    if config_update.transport is not None:
        config.transport = config_update.transport
    if config_update.command is not None:
        config.command = config_update.command
    if config_update.args is not None:
        config.args = config_update.args
    if config_update.env is not None:
        config.env = config_update.env
    if config_update.url is not None:
        config.url = config_update.url
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
