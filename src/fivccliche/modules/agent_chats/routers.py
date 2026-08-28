import uuid
from datetime import timedelta

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    Request,
    responses,
    status,
)
from fivcglue.interfaces.mutexes import IMutexSite
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from fivccliche.services.interfaces.agent_chats import IUserChatProvider
from fivccliche.services.interfaces.agent_configs import IUserConfigProvider
from fivccliche.utils.chats import ChatTask
from fivccliche.utils.deps import (
    IUser,
    get_authenticated_user_async,
    get_chat_provider_async,
    get_config_provider_async,
    get_db_session_async,
    get_mutex_site_async,
)
from fivccliche.utils.filters import FilterError
from fivccliche.utils.schemas import PaginatedResponse

from . import models, schemas, utils
from .filters import ChatEditableFilterSet, ChatFilterSet

# ============================================================================
# Chat Session Endpoints
# ============================================================================

router_chats = APIRouter(tags=["chats"], prefix="/chats")


@router_chats.post(
    "/",
    summary="Create a new chat session.",
    status_code=status.HTTP_201_CREATED,
    response_model=schemas.UserChatSchema,
)
async def create_chat_async(
    chat_create: schemas.UserChatCreateSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.UserChatSchema:
    """Create a new chat session without processing."""
    # Create new chat with specified agent_id
    chat = await utils.create_chat_async(
        session=session,
        chat_uuid=str(uuid.uuid4()),
        agent_id=chat_create.agent_id,
        user_uuid=user.uuid,
        context=chat_create.context,
        is_memorable=True,
    )
    await session.commit()
    await session.refresh(chat)
    return chat.to_schema()


@router_chats.get(
    "/",
    summary="List all chat sessions for the authenticated user.",
    response_model=PaginatedResponse[schemas.UserChatSchema],
)
async def list_chats_async(
    request: Request,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    agent_id: str | None = Query(None, description="Filter chats by agent ID"),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse[schemas.UserChatSchema]:
    """List all chat sessions for the authenticated user."""
    try:
        filters = ChatFilterSet(user.uuid, is_superuser=user.is_superuser)
        filters.parse(
            agent_id=agent_id,
            **{
                key: value
                for key, value in request.query_params.multi_items()
                if key.startswith("context.")
            },
        )
    except FilterError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    sessions = await utils.list_chats_async(
        session,
        filters=filters,
        skip=skip,
        limit=limit,
    )
    total = await utils.count_chats_async(
        session,
        filters=filters,
    )
    return PaginatedResponse[schemas.UserChatSchema](
        total=total,
        results=[s.to_schema() for s in sessions],
    )


@router_chats.get(
    "/{chat_uuid}/",
    summary="Get a chat session by ID for the authenticated user.",
    response_model=schemas.UserChatSchema,
)
async def get_chat_async(
    chat_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.UserChatSchema:
    """Get a chat session by ID."""
    chat = await utils.get_chat_async(
        session,
        chat_uuid,
        filters=ChatFilterSet(user.uuid, is_superuser=user.is_superuser),
    )
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found",
        )
    return chat.to_schema()


@router_chats.delete(
    "/{chat_uuid}/",
    summary="Delete a chat session by ID for the authenticated user.",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_chat_async(
    chat_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    """Delete a chat session."""
    chat = await utils.get_chat_async(
        session,
        chat_uuid,
        filters=ChatEditableFilterSet(user.uuid, is_superuser=user.is_superuser),
    )
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found",
        )

    await session.delete(chat)
    await session.commit()


# ============================================================================
# Chat Message Endpoints
# ============================================================================

router_messages = APIRouter(tags=["chat_messages"], prefix="/chats")
CHAT_MESSAGE_LOCK_EXPIRE = timedelta(minutes=15)
CHAT_MESSAGE_RUN_TIMEOUT = timedelta(minutes=5)


@router_messages.post(
    "/{chat_uuid}/messages/",
    summary="Send a new message to an existing chat.",
    status_code=status.HTTP_201_CREATED,
)
async def create_chat_messages_async(
    chat_uuid: str,
    chat_message: schemas.UserChatMessageCreateSchema,
    background_tasks: BackgroundTasks,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
    config_provider: IUserConfigProvider = Depends(get_config_provider_async),
    chat_provider: IUserChatProvider = Depends(get_chat_provider_async),
    mutex_site: IMutexSite | None = Depends(get_mutex_site_async),
) -> responses.StreamingResponse:
    """Send a new message to an existing chat session."""
    # Verify chat exists and user owns it
    chat = await utils.get_chat_async(
        session,
        chat_uuid,
        filters=ChatEditableFilterSet(user.uuid, is_superuser=user.is_superuser),
    )
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found",
        )

    # Use the chat's existing agent_id, not from query
    chat_agent_id = chat.agent_id

    chat_mutex = mutex_site.get_mutex(f"chats:message:{chat_uuid}") if mutex_site else None
    if chat_mutex and not await chat_mutex.acquire_async(
        expire=CHAT_MESSAGE_LOCK_EXPIRE,
        timeout=None,
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Chat message processing already running",
        )

    try:
        chat_task = ChatTask(
            user,
            config_provider,
            chat_provider,
            chat_uuid=chat_uuid,
            chat_query=chat_message.query,
            chat_agent_id=chat_agent_id,
            chat_context=chat.context,
            chat_skills_enabled=True,
            chat_mutex=chat_mutex,
            chat_run_timeout=CHAT_MESSAGE_RUN_TIMEOUT.total_seconds(),
        )
        chat_task.start()
        background_tasks.add_task(chat_task.join_async)
    except Exception:
        if chat_mutex:
            await chat_mutex.release_async()
        raise

    # Release the request-scoped session before SSE so the pool connection is
    # not held for the entire stream duration.
    await session.close()

    return responses.StreamingResponse(
        chat_task.get_stream_async(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router_messages.get(
    "/{chat_uuid}/messages/",
    summary="List all chat messages for a chat.",
    response_model=PaginatedResponse[schemas.UserChatMessageSchema],
)
async def list_chat_messages_async(
    chat_uuid: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse[schemas.UserChatMessageSchema]:
    """List all chat messages for a session."""
    # Verify the session belongs to the user
    chat = await utils.get_chat_async(
        session,
        chat_uuid,
        filters=ChatFilterSet(user.uuid, is_superuser=user.is_superuser),
    )
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found",
        )
    messages = await utils.list_chat_messages_async(session, chat.uuid, skip=skip, limit=limit)
    total_result = await session.execute(
        select(func.count(col(models.UserChatMessage.uuid))).where(
            models.UserChatMessage.chat_uuid == chat.uuid
        )
    )
    total = total_result.scalar() or 0
    return PaginatedResponse[schemas.UserChatMessageSchema](
        total=total,
        results=[m.to_schema() for m in messages],
    )


@router_messages.delete(
    "/{chat_uuid}/messages/{message_uuid}/",
    summary="Delete a chat message.",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_chat_message_async(
    message_uuid: str,
    chat_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    """Delete a chat message."""
    # First verify the chat exists and user has access
    chat = await utils.get_chat_async(
        session,
        chat_uuid,
        filters=ChatEditableFilterSet(user.uuid, is_superuser=user.is_superuser),
    )
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found",
        )

    # Now get and delete the message
    message = await utils.get_chat_message_async(
        session,
        message_uuid,
        chat_uuid,
    )
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat message not found",
        )
    if message.chat_uuid != chat_uuid:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat message not found",
        )
    await session.delete(message)
    await session.commit()
