from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from fivcplayground.tools import create_tool_retriever_async

from fivccliche.services.interfaces.agent_configs import IUserConfigProvider
from fivccliche.utils import crud
from fivccliche.utils.deps import (
    IUser,
    get_authenticated_user_async,
    get_config_provider_async,
    get_db_session_async,
)

from . import methods, queries, schemas


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
crud.register_routes(
    router_embeddings,
    crud.RouteConfig(
        slug="embedding_config",
        noun="Embedding config",
        schema=schemas.UserEmbeddingSchema,
        create_fn=methods.create_embedding_config_async,
        get_fn=methods.get_embedding_config_async,
        list_fn=methods.list_embedding_configs_async,
        count_fn=methods.count_embedding_configs_async,
        update_fn=methods.update_embedding_config_async,
        delete_fn=methods.delete_embedding_config_async,
    ),
)


# ============================================================================
# LLM Config Endpoints
# ============================================================================

router_models = APIRouter(prefix="/configs/models", tags=["model_configs"])
crud.register_routes(
    router_models,
    crud.RouteConfig(
        slug="llm_config",
        noun="LLM config",
        schema=schemas.UserLLMSchema,
        create_fn=methods.create_llm_config_async,
        get_fn=methods.get_llm_config_async,
        list_fn=methods.list_llm_configs_async,
        count_fn=methods.count_llm_configs_async,
        update_fn=methods.update_llm_config_async,
        delete_fn=methods.delete_llm_config_async,
    ),
)


# ============================================================================
# Agent Config Endpoints
# ============================================================================

router_agents = APIRouter(prefix="/configs/agents", tags=["agent_configs"])
crud.register_routes(
    router_agents,
    crud.RouteConfig(
        slug="agent_config",
        noun="Agent config",
        schema=schemas.UserAgentSchema,
        create_fn=methods.create_agent_config_async,
        get_fn=methods.get_agent_config_async,
        list_fn=methods.list_agent_configs_async,
        count_fn=methods.count_agent_configs_async,
        update_fn=methods.update_agent_config_async,
        delete_fn=methods.delete_agent_config_async,
        before_update=_reject_frozen_agent_update,
        before_delete=_reject_frozen_agent_delete,
    ),
)


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


crud.register_routes(
    router_tools,
    crud.RouteConfig(
        slug="tool_config",
        noun="Tool config",
        schema=schemas.UserToolSchema,
        create_fn=methods.create_tool_config_async,
        get_fn=methods.get_tool_config_async,
        list_fn=methods.list_tool_configs_async,
        count_fn=methods.count_tool_configs_async,
        update_fn=methods.update_tool_config_async,
        delete_fn=methods.delete_tool_config_async,
    ),
)


# ============================================================================
# Skill Config Endpoints
# ============================================================================

router_skills = APIRouter(prefix="/configs/skills", tags=["skill_configs"])
crud.register_routes(
    router_skills,
    crud.RouteConfig(
        slug="skill_config",
        noun="Skill config",
        schema=schemas.UserSkillSchema,
        create_fn=methods.create_skill_config_async,
        get_fn=methods.get_skill_config_async,
        list_fn=methods.list_skill_configs_async,
        count_fn=methods.count_skill_configs_async,
        update_fn=methods.update_skill_config_async,
        delete_fn=methods.delete_skill_config_async,
    ),
)


# ============================================================================
# Question Config Endpoints
# ============================================================================

router_questions = APIRouter(prefix="/configs/questions", tags=["question_configs"])
crud.register_routes(
    router_questions,
    crud.RouteConfig(
        slug="question_config",
        noun="Question config",
        schema=schemas.UserQuestionSchema,
        create_fn=methods.create_question_async,
        get_fn=methods.get_question_async,
        list_fn=methods.list_questions_async,
        count_fn=methods.count_questions_async,
        update_fn=methods.update_question_async,
        delete_fn=methods.delete_question_async,
        list_query=queries.UserQuestionListQuery,
    ),
)
