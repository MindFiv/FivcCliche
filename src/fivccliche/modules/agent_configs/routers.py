from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from fivcplayground.tools import create_tool_retriever

from fivccliche.utils.deps import (
    IUser,
    get_authenticated_user_async,
    get_db_session_async,
    get_config_provider_async,
)
from fivccliche.utils.schemas import PaginatedResponse

from . import methods, schemas
from ...services.interfaces.agent_configs import IUserConfigProvider

# ============================================================================
# Embedding Config Endpoints
# ============================================================================

router_embeddings = APIRouter(prefix="/configs/embeddings", tags=["embedding_configs"])


@router_embeddings.post(
    "/",
    summary="Create a new embedding config for the authenticated user.",
    response_model=schemas.UserEmbeddingSchema,
    status_code=status.HTTP_201_CREATED,
)
async def create_embedding_config_async(
    config_create: schemas.UserEmbeddingSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.UserEmbeddingSchema:
    """Create a new embedding config."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    config = await methods.create_embedding_config_async(
        session,
        user.uuid,
        config_create,
    )
    return config.to_schema()


@router_embeddings.get(
    "/",
    summary="List all embedding configs for the authenticated user.",
    response_model=PaginatedResponse[schemas.UserEmbeddingSchema],
)
async def list_embedding_configs_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse[schemas.UserEmbeddingSchema]:
    """List all embedding configs for the authenticated user."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    configs = await methods.list_embedding_configs_async(session, user.uuid, skip=skip, limit=limit)
    total = await methods.count_embedding_configs_async(session, user.uuid)
    return PaginatedResponse[schemas.UserEmbeddingSchema](
        total=total,
        results=[config.to_schema() for config in configs],
    )


@router_embeddings.get(
    "/{config_uuid}",
    summary="Get an embedding config by ID for the authenticated user.",
    response_model=schemas.UserEmbeddingSchema,
)
async def get_embedding_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.UserEmbeddingSchema:
    """Get an embedding config by ID."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    config = await methods.get_embedding_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Embedding config not found",
        )
    return config.to_schema()


@router_embeddings.patch(
    "/{config_uuid}",
    summary="Update an embedding config by ID for the authenticated user.",
    response_model=schemas.UserEmbeddingSchema,
)
async def update_embedding_config_async(
    config_uuid: str,
    config_update: schemas.UserEmbeddingSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.UserEmbeddingSchema:
    """Update an embedding config."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    config = await methods.get_embedding_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Embedding config not found",
        )
    config = await methods.update_embedding_config_async(session, config, config_update)
    return config.to_schema()


@router_embeddings.delete(
    "/{config_uuid}",
    summary="Delete an embedding config by ID for the authenticated user.",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_embedding_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    """Delete an embedding config."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    config = await methods.get_embedding_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Embedding config not found",
        )
    await methods.delete_embedding_config_async(session, config)


# ============================================================================
# LLM Config Endpoints
# ============================================================================

router_models = APIRouter(prefix="/configs/models", tags=["model_configs"])


@router_models.post(
    "/",
    summary="Create a new LLM config for the authenticated user.",
    response_model=schemas.UserLLMSchema,
    status_code=status.HTTP_201_CREATED,
)
async def create_llm_config_async(
    config_create: schemas.UserLLMSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.UserLLMSchema:
    """Create a new LLM config."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    config = await methods.create_llm_config_async(session, user.uuid, config_create)
    return config.to_schema()


@router_models.get(
    "/",
    summary="List all LLM configs for the authenticated user.",
    response_model=PaginatedResponse[schemas.UserLLMSchema],
)
async def list_llm_configs_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse[schemas.UserLLMSchema]:
    """List all LLM configs for the authenticated user."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    configs = await methods.list_llm_configs_async(session, user.uuid, skip=skip, limit=limit)
    total = await methods.count_llm_configs_async(session, user.uuid)
    return PaginatedResponse[schemas.UserLLMSchema](
        total=total,
        results=[config.to_schema() for config in configs],
    )


@router_models.get(
    "/{config_uuid}",
    summary="Get an LLM config by ID for the authenticated user.",
    response_model=schemas.UserLLMSchema,
)
async def get_llm_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.UserLLMSchema:
    """Get an LLM config by ID."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    config = await methods.get_llm_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="LLM config not found",
        )
    return config.to_schema()


@router_models.patch(
    "/{config_uuid}",
    summary="Update an LLM config by ID for the authenticated user.",
    response_model=schemas.UserLLMSchema,
)
async def update_llm_config_async(
    config_uuid: str,
    config_update: schemas.UserLLMSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.UserLLMSchema:
    """Update an LLM config."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    config = await methods.get_llm_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="LLM config not found",
        )
    config = await methods.update_llm_config_async(session, config, config_update)
    return config.to_schema()


@router_models.delete(
    "/{config_uuid}",
    summary="Delete an LLM config by ID for the authenticated user.",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_llm_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    """Delete an LLM config."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    config = await methods.get_llm_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="LLM config not found",
        )
    await methods.delete_llm_config_async(session, config)


# ============================================================================
# Agent Config Endpoints
# ============================================================================

router_agents = APIRouter(prefix="/configs/agents", tags=["agent_configs"])


@router_agents.post(
    "/",
    summary="Create a new agent config for the authenticated user.",
    response_model=schemas.UserAgentSchema,
    status_code=status.HTTP_201_CREATED,
)
async def create_agent_config_async(
    config_create: schemas.UserAgentSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.UserAgentSchema:
    """Create a new agent config."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    config = await methods.create_agent_config_async(session, user.uuid, config_create)
    return config.to_schema()


@router_agents.get(
    "/",
    summary="List all agent configs for the authenticated user.",
    response_model=PaginatedResponse[schemas.UserAgentSchema],
)
async def list_agent_configs_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse[schemas.UserAgentSchema]:
    """List all agent configs for the authenticated user."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    configs = await methods.list_agent_configs_async(session, user.uuid, skip=skip, limit=limit)
    total = await methods.count_agent_configs_async(session, user.uuid)
    return PaginatedResponse[schemas.UserAgentSchema](
        total=total,
        results=[config.to_schema() for config in configs],
    )


@router_agents.get(
    "/{config_uuid}",
    summary="Get an agent config by ID for the authenticated user.",
    response_model=schemas.UserAgentSchema,
)
async def get_agent_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.UserAgentSchema:
    """Get an agent config by ID."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    config = await methods.get_agent_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent config not found",
        )
    return config.to_schema()


@router_agents.patch(
    "/{config_uuid}",
    summary="Update an agent config by ID for the authenticated user.",
    response_model=schemas.UserAgentSchema,
)
async def update_agent_config_async(
    config_uuid: str,
    config_update: schemas.UserAgentSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.UserAgentSchema:
    """Update an agent config."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    config = await methods.get_agent_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent config not found",
        )
    config = await methods.update_agent_config_async(session, config, config_update)
    return config.to_schema()


@router_agents.delete("/{config_uuid}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    """Delete an agent config."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    config = await methods.get_agent_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent config not found",
        )
    await methods.delete_agent_config_async(session, config)


# ============================================================================
# Tool Config Endpoints
# ============================================================================

router_tools = APIRouter(prefix="/configs/tools", tags=["tool_configs"])


@router_tools.post(
    "/indexing",
    summary="Indexing tool for the authenticated user.",
    status_code=status.HTTP_200_OK,
)
async def indexing_tool_async(
    user: IUser = Depends(get_authenticated_user_async),
    # session: AsyncSession = Depends(get_db_session_async),
    config_provider: IUserConfigProvider = Depends(get_config_provider_async),
):
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    agent_tools = create_tool_retriever(
        tool_backend=config_provider.get_tool_backend(),
        tool_repository=config_provider.get_tool_repository(user_uuid=user.uuid),
        embedding_backend=config_provider.get_embedding_backend(),
        embedding_repository=config_provider.get_embedding_repository(user_uuid=user.uuid),
        space_id=user.uuid,
    )
    agent_tools.index_tools()


@router_tools.post(
    "/",
    summary="Create a new tool config for the authenticated user.",
    response_model=schemas.UserToolSchema,
    status_code=status.HTTP_201_CREATED,
)
async def create_tool_config_async(
    config_create: schemas.UserToolSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.UserToolSchema:
    """Create a new tool config."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    config = await methods.create_tool_config_async(session, user.uuid, config_create)
    return config.to_schema()


@router_tools.get(
    "/",
    summary="List all tool configs for the authenticated user.",
    response_model=PaginatedResponse[schemas.UserToolSchema],
)
async def list_tool_configs_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse[schemas.UserToolSchema]:
    """List all tool configs for the authenticated user."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    configs = await methods.list_tool_configs_async(session, user.uuid, skip=skip, limit=limit)
    total = await methods.count_tool_configs_async(session, user.uuid)
    return PaginatedResponse[schemas.UserToolSchema](
        total=total,
        results=[config.to_schema() for config in configs],
    )


@router_tools.get(
    "/{config_uuid}",
    summary="Get a tool config by ID for the authenticated user.",
    response_model=schemas.UserToolSchema,
)
async def get_tool_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.UserToolSchema:
    """Get a tool config by ID."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    config = await methods.get_tool_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool config not found",
        )
    return config.to_schema()


@router_tools.patch(
    "/{config_uuid}",
    summary="Update a tool config by ID for the authenticated user.",
    response_model=schemas.UserToolSchema,
)
async def update_tool_config_async(
    config_uuid: str,
    config_update: schemas.UserToolSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.UserToolSchema:
    """Update a tool config."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    config = await methods.get_tool_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool config not found",
        )
    config = await methods.update_tool_config_async(session, config, config_update)
    return config.to_schema()


@router_tools.delete(
    "/{config_uuid}",
    summary="Delete a tool config by ID for the authenticated user.",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_tool_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    """Delete a tool config."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    config = await methods.get_tool_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool config not found",
        )
    await methods.delete_tool_config_async(session, config)


# ============================================================================
# Main Router
# ============================================================================

# router = APIRouter(prefix="/configs", tags=["configs"])
# router.include_router(router_models)
# router.include_router(router_embeddings)
# router.include_router(router_agents)
