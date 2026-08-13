"""HTTP routes for viewing the authenticated user's agent memories."""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from fivccliche.services.interfaces.agent_memories import IUserMemoryProvider
from fivccliche.utils.deps import (
    IUser,
    get_authenticated_user_async,
    get_memory_provider_async,
)
from fivccliche.utils.schemas import PaginatedResponse

from . import schemas

router_memories = APIRouter(tags=["memories"], prefix="/memories")


async def get_required_memory_provider_async(
    memory_provider: IUserMemoryProvider | None = Depends(get_memory_provider_async),
) -> IUserMemoryProvider:
    """Return the memory provider, or 503 if the component is not mounted."""
    if memory_provider is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory provider is not mounted",
        )
    return memory_provider


@router_memories.get(
    "/",
    summary="List memories for the authenticated user.",
    response_model=PaginatedResponse[schemas.MemoryContentSchema],
)
async def list_memories_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user: IUser = Depends(get_authenticated_user_async),
    memory_provider: IUserMemoryProvider = Depends(get_required_memory_provider_async),
) -> PaginatedResponse[schemas.MemoryContentSchema]:
    memory = memory_provider.get_memory(space_id=user.uuid)
    result = await memory.list_async(skip=skip, limit=limit)
    return PaginatedResponse(
        total=result.total,
        results=[
            schemas.MemoryContentSchema.model_validate(item.model_dump()) for item in result.items
        ],
    )


@router_memories.get(
    "/recall/",
    summary="Recall memories by semantic similarity for the authenticated user.",
    response_model=schemas.MemoryRecallResponseSchema,
)
async def recall_memories_async(
    query: str = Query(..., min_length=1),
    user: IUser = Depends(get_authenticated_user_async),
    memory_provider: IUserMemoryProvider = Depends(get_required_memory_provider_async),
) -> schemas.MemoryRecallResponseSchema:
    memory = memory_provider.get_memory(space_id=user.uuid)
    result = await memory.recall_async(query)
    return schemas.MemoryRecallResponseSchema(
        results=[
            schemas.MemoryContentSchema.model_validate(item.model_dump()) for item in result.items
        ],
    )
