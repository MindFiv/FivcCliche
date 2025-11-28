from fastapi import APIRouter, Depends, Query

from fivccliche.utils.deps import (
    IUser,
    get_authenticated_user_async,
    AsyncSession,
    get_db_session_async,
)
from fivccliche.utils.schemas import PaginatedResponse

from . import models

# Agent Model Configs -----------------------------------------------
router_models = APIRouter(prefix="/models", tags=["agent_models"])


@router_models.get("/", response_model=PaginatedResponse[models.UserLLM])
async def list_models_async(
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    """List all models."""
    raise NotImplementedError("List models not implemented.")


@router_models.get("/{model_id}")
async def get_model_async(
    model_id: str = Query(..., min_length=1, max_length=32),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    """Get a model by ID."""
    raise NotImplementedError("Get model not implemented.")


router = APIRouter(prefix="/agent_configs", tags=["agent_configs"])
router.include_router(router_models)
