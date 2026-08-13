from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic_strict_partial import create_partial_model
from sqlalchemy.ext.asyncio import AsyncSession

from fivcplayground.tools import create_tool_retriever_async

from fivccliche.services.interfaces.agent_configs import IUserConfigProvider
from fivccliche.utils.asserts import assert_user_owns_resource
from fivccliche.utils.deps import (
    IUser,
    get_authenticated_user_async,
    get_config_provider_async,
    get_db_session_async,
)
from fivccliche.utils.schemas import PaginatedResponse

from . import methods, schemas


def _reject_frozen_agent_update(config, config_update, _user) -> None:
    if not config.is_frozen:
        return
    fields_set: set[str] = getattr(config_update, "model_fields_set", set())
    if fields_set - {"is_frozen"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Agent config is frozen and cannot be edited",
        )


def _reject_frozen_agent_delete(config, _user) -> None:
    if config.is_frozen:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Agent config is frozen and cannot be deleted",
        )


# ============================================================================
# Embedding Config Endpoints
# ============================================================================

router_embeddings = APIRouter(prefix="/configs/embeddings", tags=["embedding_configs"])


@router_embeddings.post(
    "/",
    summary="Create a new embedding config for the authenticated user.",
    response_model=schemas.UserEmbeddingSchema,
    status_code=status.HTTP_201_CREATED,
    operation_id="create_embedding_config",
)
async def create_embedding_config_async(
    config_create: schemas.UserEmbeddingSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await methods.create_embedding_config_async(
        session,
        None if user.is_superuser else user.uuid,
        config_create,
        updated_user_uuid=user.uuid,
    )
    return config.to_schema()


@router_embeddings.get(
    "/",
    summary="List all embedding configs for the authenticated user.",
    response_model=PaginatedResponse[schemas.UserEmbeddingSchema],
    operation_id="list_embedding_configs",
)
async def list_embedding_configs_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse:
    configs = await methods.list_embedding_configs_async(session, user.uuid, skip=skip, limit=limit)
    total = await methods.count_embedding_configs_async(session, user.uuid)
    return PaginatedResponse(
        total=total,
        results=[config.to_schema() for config in configs],
    )


@router_embeddings.get(
    "/{config_uuid}/",
    summary="Get a embedding config by ID for the authenticated user.",
    response_model=schemas.UserEmbeddingSchema,
    operation_id="get_embedding_config",
)
async def get_embedding_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await methods.get_embedding_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Embedding config not found"
        )
    return config.to_schema()


@router_embeddings.patch(
    "/{config_uuid}/",
    summary="Update a embedding config by ID for the authenticated user.",
    response_model=schemas.UserEmbeddingSchema,
    operation_id="update_embedding_config",
)
async def update_embedding_config_async(
    config_uuid: str,
    config_update: create_partial_model(schemas.UserEmbeddingSchema),  # type: ignore[valid-type]
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await methods.get_embedding_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Embedding config not found"
        )
    assert_user_owns_resource(
        user,
        config.user_uuid,
        global_detail="Cannot update global configs",
        other_detail="Cannot update configs belonging to other users",
    )
    config = await methods.update_embedding_config_async(
        session, config, config_update, updated_user_uuid=user.uuid
    )
    return config.to_schema()


@router_embeddings.delete(
    "/{config_uuid}/",
    summary="Delete a embedding config by ID for the authenticated user.",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="delete_embedding_config",
)
async def delete_embedding_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    config = await methods.get_embedding_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Embedding config not found"
        )
    assert_user_owns_resource(
        user,
        config.user_uuid,
        global_detail="Cannot delete global configs",
        other_detail="Cannot delete configs belonging to other users",
    )
    await methods.delete_embedding_config_async(session, config)


# ============================================================================
# LLM Config Endpoints
# ============================================================================

router_models = APIRouter(prefix="/configs/models", tags=["model_configs"])


@router_models.post(
    "/",
    summary="Create a new llm config for the authenticated user.",
    response_model=schemas.UserLLMSchema,
    status_code=status.HTTP_201_CREATED,
    operation_id="create_llm_config",
)
async def create_llm_config_async(
    config_create: schemas.UserLLMSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await methods.create_llm_config_async(
        session,
        None if user.is_superuser else user.uuid,
        config_create,
        updated_user_uuid=user.uuid,
    )
    return config.to_schema()


@router_models.get(
    "/",
    summary="List all llm configs for the authenticated user.",
    response_model=PaginatedResponse[schemas.UserLLMSchema],
    operation_id="list_llm_configs",
)
async def list_llm_configs_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse:
    configs = await methods.list_llm_configs_async(session, user.uuid, skip=skip, limit=limit)
    total = await methods.count_llm_configs_async(session, user.uuid)
    return PaginatedResponse(
        total=total,
        results=[config.to_schema() for config in configs],
    )


@router_models.get(
    "/{config_uuid}/",
    summary="Get a llm config by ID for the authenticated user.",
    response_model=schemas.UserLLMSchema,
    operation_id="get_llm_config",
)
async def get_llm_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await methods.get_llm_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="LLM config not found")
    return config.to_schema()


@router_models.patch(
    "/{config_uuid}/",
    summary="Update a llm config by ID for the authenticated user.",
    response_model=schemas.UserLLMSchema,
    operation_id="update_llm_config",
)
async def update_llm_config_async(
    config_uuid: str,
    config_update: create_partial_model(schemas.UserLLMSchema),  # type: ignore[valid-type]
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await methods.get_llm_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="LLM config not found")
    assert_user_owns_resource(
        user,
        config.user_uuid,
        global_detail="Cannot update global configs",
        other_detail="Cannot update configs belonging to other users",
    )
    config = await methods.update_llm_config_async(
        session, config, config_update, updated_user_uuid=user.uuid
    )
    return config.to_schema()


@router_models.delete(
    "/{config_uuid}/",
    summary="Delete a llm config by ID for the authenticated user.",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="delete_llm_config",
)
async def delete_llm_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    config = await methods.get_llm_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="LLM config not found")
    assert_user_owns_resource(
        user,
        config.user_uuid,
        global_detail="Cannot delete global configs",
        other_detail="Cannot delete configs belonging to other users",
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
    operation_id="create_agent_config",
)
async def create_agent_config_async(
    config_create: schemas.UserAgentSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await methods.create_agent_config_async(
        session,
        None if user.is_superuser else user.uuid,
        config_create,
        updated_user_uuid=user.uuid,
    )
    return config.to_schema()


@router_agents.get(
    "/",
    summary="List all agent configs for the authenticated user.",
    response_model=PaginatedResponse[schemas.UserAgentSchema],
    operation_id="list_agent_configs",
)
async def list_agent_configs_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse:
    configs = await methods.list_agent_configs_async(session, user.uuid, skip=skip, limit=limit)
    total = await methods.count_agent_configs_async(session, user.uuid)
    return PaginatedResponse(
        total=total,
        results=[config.to_schema() for config in configs],
    )


@router_agents.get(
    "/{config_uuid}/",
    summary="Get a agent config by ID for the authenticated user.",
    response_model=schemas.UserAgentSchema,
    operation_id="get_agent_config",
)
async def get_agent_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await methods.get_agent_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent config not found")
    return config.to_schema()


@router_agents.patch(
    "/{config_uuid}/",
    summary="Update a agent config by ID for the authenticated user.",
    response_model=schemas.UserAgentSchema,
    operation_id="update_agent_config",
)
async def update_agent_config_async(
    config_uuid: str,
    config_update: create_partial_model(schemas.UserAgentSchema),  # type: ignore[valid-type]
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await methods.get_agent_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent config not found")
    assert_user_owns_resource(
        user,
        config.user_uuid,
        global_detail="Cannot update global configs",
        other_detail="Cannot update configs belonging to other users",
    )
    _reject_frozen_agent_update(config, config_update, user)
    config = await methods.update_agent_config_async(
        session, config, config_update, updated_user_uuid=user.uuid
    )
    return config.to_schema()


@router_agents.delete(
    "/{config_uuid}/",
    summary="Delete a agent config by ID for the authenticated user.",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="delete_agent_config",
)
async def delete_agent_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    config = await methods.get_agent_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent config not found")
    assert_user_owns_resource(
        user,
        config.user_uuid,
        global_detail="Cannot delete global configs",
        other_detail="Cannot delete configs belonging to other users",
    )
    _reject_frozen_agent_delete(config, user)
    await methods.delete_agent_config_async(session, config)


# ============================================================================
# Tool Config Endpoints
# ============================================================================

router_tools = APIRouter(prefix="/configs/tools", tags=["tool_configs"])


@router_tools.post(
    "/index/",
    summary="Index tool for the authenticated user.",
    status_code=status.HTTP_200_OK,
)
async def index_tool_async(
    user: IUser = Depends(get_authenticated_user_async),
    config_provider: IUserConfigProvider = Depends(get_config_provider_async),
):
    agent_tools = await create_tool_retriever_async(
        tool_backend=config_provider.get_tool_backend(),
        tool_config_repository=config_provider.get_tool_repository(user_uuid=user.uuid),
        embedding_backend=config_provider.get_embedding_backend(),
        embedding_config_repository=config_provider.get_embedding_repository(user_uuid=user.uuid),
        space_id=user.uuid,
    )
    await agent_tools.index_tools_async()


@router_tools.post(
    "/{config_uuid}/probe/",
    summary="Probe tool for the authenticated user.",
    status_code=status.HTTP_200_OK,
)
async def probe_tool_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
    config_provider: IUserConfigProvider = Depends(get_config_provider_async),
) -> schemas.UserToolProbeSchema:
    config = await methods.get_tool_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool config not found",
        )
    tool_schema = config.to_schema()
    await session.close()

    tool_backend = config_provider.get_tool_backend()
    tool_bundle = tool_backend.create_tool_bundle(tool_schema)
    tool_context = tool_bundle.setup()
    async with tool_context as tools:
        tool_names = [tool.name for tool in tools]

    return schemas.UserToolProbeSchema(tool_names=tool_names)


@router_tools.post(
    "/",
    summary="Create a new tool config for the authenticated user.",
    response_model=schemas.UserToolSchema,
    status_code=status.HTTP_201_CREATED,
    operation_id="create_tool_config",
)
async def create_tool_config_async(
    config_create: schemas.UserToolSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await methods.create_tool_config_async(
        session,
        None if user.is_superuser else user.uuid,
        config_create,
        updated_user_uuid=user.uuid,
    )
    return config.to_schema()


@router_tools.get(
    "/",
    summary="List all tool configs for the authenticated user.",
    response_model=PaginatedResponse[schemas.UserToolSchema],
    operation_id="list_tool_configs",
)
async def list_tool_configs_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse:
    configs = await methods.list_tool_configs_async(session, user.uuid, skip=skip, limit=limit)
    total = await methods.count_tool_configs_async(session, user.uuid)
    return PaginatedResponse(
        total=total,
        results=[config.to_schema() for config in configs],
    )


@router_tools.get(
    "/{config_uuid}/",
    summary="Get a tool config by ID for the authenticated user.",
    response_model=schemas.UserToolSchema,
    operation_id="get_tool_config",
)
async def get_tool_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await methods.get_tool_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tool config not found")
    return config.to_schema()


@router_tools.patch(
    "/{config_uuid}/",
    summary="Update a tool config by ID for the authenticated user.",
    response_model=schemas.UserToolSchema,
    operation_id="update_tool_config",
)
async def update_tool_config_async(
    config_uuid: str,
    config_update: create_partial_model(schemas.UserToolSchema),  # type: ignore[valid-type]
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await methods.get_tool_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tool config not found")
    assert_user_owns_resource(
        user,
        config.user_uuid,
        global_detail="Cannot update global configs",
        other_detail="Cannot update configs belonging to other users",
    )
    config = await methods.update_tool_config_async(
        session, config, config_update, updated_user_uuid=user.uuid
    )
    return config.to_schema()


@router_tools.delete(
    "/{config_uuid}/",
    summary="Delete a tool config by ID for the authenticated user.",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="delete_tool_config",
)
async def delete_tool_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    config = await methods.get_tool_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tool config not found")
    assert_user_owns_resource(
        user,
        config.user_uuid,
        global_detail="Cannot delete global configs",
        other_detail="Cannot delete configs belonging to other users",
    )
    await methods.delete_tool_config_async(session, config)


# ============================================================================
# Skill Config Endpoints
# ============================================================================

router_skills = APIRouter(prefix="/configs/skills", tags=["skill_configs"])


@router_skills.post(
    "/",
    summary="Create a new skill config for the authenticated user.",
    response_model=schemas.UserSkillSchema,
    status_code=status.HTTP_201_CREATED,
    operation_id="create_skill_config",
)
async def create_skill_config_async(
    config_create: schemas.UserSkillSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await methods.create_skill_config_async(
        session,
        None if user.is_superuser else user.uuid,
        config_create,
        updated_user_uuid=user.uuid,
    )
    return config.to_schema()


@router_skills.get(
    "/",
    summary="List all skill configs for the authenticated user.",
    response_model=PaginatedResponse[schemas.UserSkillSchema],
    operation_id="list_skill_configs",
)
async def list_skill_configs_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse:
    configs = await methods.list_skill_configs_async(session, user.uuid, skip=skip, limit=limit)
    total = await methods.count_skill_configs_async(session, user.uuid)
    return PaginatedResponse(
        total=total,
        results=[config.to_schema() for config in configs],
    )


@router_skills.get(
    "/{config_uuid}/",
    summary="Get a skill config by ID for the authenticated user.",
    response_model=schemas.UserSkillSchema,
    operation_id="get_skill_config",
)
async def get_skill_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await methods.get_skill_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Skill config not found")
    return config.to_schema()


@router_skills.patch(
    "/{config_uuid}/",
    summary="Update a skill config by ID for the authenticated user.",
    response_model=schemas.UserSkillSchema,
    operation_id="update_skill_config",
)
async def update_skill_config_async(
    config_uuid: str,
    config_update: create_partial_model(schemas.UserSkillSchema),  # type: ignore[valid-type]
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await methods.get_skill_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Skill config not found")
    assert_user_owns_resource(
        user,
        config.user_uuid,
        global_detail="Cannot update global configs",
        other_detail="Cannot update configs belonging to other users",
    )
    config = await methods.update_skill_config_async(
        session, config, config_update, updated_user_uuid=user.uuid
    )
    return config.to_schema()


@router_skills.delete(
    "/{config_uuid}/",
    summary="Delete a skill config by ID for the authenticated user.",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="delete_skill_config",
)
async def delete_skill_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    config = await methods.get_skill_config_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Skill config not found")
    assert_user_owns_resource(
        user,
        config.user_uuid,
        global_detail="Cannot delete global configs",
        other_detail="Cannot delete configs belonging to other users",
    )
    await methods.delete_skill_config_async(session, config)


# ============================================================================
# Question Config Endpoints
# ============================================================================

router_questions = APIRouter(prefix="/configs/questions", tags=["question_configs"])


@router_questions.post(
    "/",
    summary="Create a new question config for the authenticated user.",
    response_model=schemas.UserQuestionSchema,
    status_code=status.HTTP_201_CREATED,
    operation_id="create_question_config",
)
async def create_question_config_async(
    config_create: schemas.UserQuestionSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await methods.create_question_async(
        session,
        None if user.is_superuser else user.uuid,
        config_create,
        updated_user_uuid=user.uuid,
    )
    return config.to_schema()


@router_questions.get(
    "/",
    summary="List all question configs for the authenticated user.",
    response_model=PaginatedResponse[schemas.UserQuestionSchema],
    operation_id="list_question_configs",
)
async def list_question_configs_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    is_active: bool | None = Query(None),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse:
    configs = await methods.list_questions_async(
        session, user.uuid, skip=skip, limit=limit, is_active=is_active
    )
    total = await methods.count_questions_async(session, user.uuid, is_active=is_active)
    return PaginatedResponse(
        total=total,
        results=[config.to_schema() for config in configs],
    )


@router_questions.get(
    "/{config_uuid}/",
    summary="Get a question config by ID for the authenticated user.",
    response_model=schemas.UserQuestionSchema,
    operation_id="get_question_config",
)
async def get_question_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await methods.get_question_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Question config not found"
        )
    return config.to_schema()


@router_questions.patch(
    "/{config_uuid}/",
    summary="Update a question config by ID for the authenticated user.",
    response_model=schemas.UserQuestionSchema,
    operation_id="update_question_config",
)
async def update_question_config_async(
    config_uuid: str,
    config_update: create_partial_model(schemas.UserQuestionSchema),  # type: ignore[valid-type]
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await methods.get_question_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Question config not found"
        )
    assert_user_owns_resource(
        user,
        config.user_uuid,
        global_detail="Cannot update global configs",
        other_detail="Cannot update configs belonging to other users",
    )
    config = await methods.update_question_async(
        session, config, config_update, updated_user_uuid=user.uuid
    )
    return config.to_schema()


@router_questions.delete(
    "/{config_uuid}/",
    summary="Delete a question config by ID for the authenticated user.",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="delete_question_config",
)
async def delete_question_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    config = await methods.get_question_async(session, user.uuid, config_uuid=config_uuid)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Question config not found"
        )
    assert_user_owns_resource(
        user,
        config.user_uuid,
        global_detail="Cannot delete global configs",
        other_detail="Cannot delete configs belonging to other users",
    )
    await methods.delete_question_async(session, config)
