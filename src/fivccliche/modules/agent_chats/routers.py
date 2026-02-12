import uuid

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    responses,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from fivccliche.services.interfaces.agent_chats import IUserChatProvider
from fivccliche.services.interfaces.agent_configs import IUserConfigProvider
from fivccliche.utils.deps import (
    IUser,
    get_authenticated_user_async,
    get_db_session_async,
    get_config_provider_async,
    get_chat_provider_async,
)
from fivccliche.utils.generators import create_chat_streaming_generator_async
from fivccliche.utils.schemas import PaginatedResponse

from . import methods, schemas


# ============================================================================
# Chat Session Endpoints
# ============================================================================

router_chats = APIRouter(tags=["chats"], prefix="/chats")


@router_chats.post(
    "/",
    summary="Query by the authenticated user.",
    status_code=status.HTTP_201_CREATED,
)
async def query_chat_async(
    chat_query: schemas.UserChatQuery,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
    config_provider: IUserConfigProvider = Depends(get_config_provider_async),
    chat_provider: IUserChatProvider = Depends(get_chat_provider_async),
) -> responses.StreamingResponse:
    """Create a new chat session."""
    if chat_query.chat_uuid and chat_query.agent_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot specify both chat_uuid and agent_id",
        )
    if not chat_query.chat_uuid and not chat_query.agent_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must specify either chat_uuid or agent_id",
        )

    chat = (
        await methods.get_chat_async(session, chat_query.chat_uuid, user.uuid)
        if chat_query.chat_uuid
        else None
    )
    chat_uuid = chat.uuid if chat else str(uuid.uuid4())
    chat_agent_id = chat.agent_id if chat else chat_query.agent_id
    print(f"🤖 [AGENT] Creating agent with ID: {chat_agent_id}")

    streaming_gen = await create_chat_streaming_generator_async(
        user,
        config_provider,
        chat_provider,
        chat_uuid=chat_uuid,
        chat_query=chat_query.query,
        chat_agent_id=chat_agent_id,
        session=session,
    )
    return responses.StreamingResponse(
        streaming_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router_chats.get(
    "/",
    summary="List all chat sessions for the authenticated user.",
    response_model=PaginatedResponse[schemas.UserChatSchema],
)
async def list_chats_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse[schemas.UserChatSchema]:
    """List all chat sessions for the authenticated user."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    sessions = await methods.list_chats_async(session, user.uuid, skip=skip, limit=limit)
    total = await methods.count_chats_async(session, user.uuid)
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
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    chat = await methods.get_chat_async(session, chat_uuid, user.uuid)
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
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    chat = await methods.get_chat_async(session, chat_uuid, user.uuid)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found",
        )

    # Authorization check: users can only delete their own chats
    # Only superusers can delete global chats (where user_uuid is None)
    if chat.user_uuid is None and not user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot delete global chats",
        )
    if chat.user_uuid is not None and chat.user_uuid != user.uuid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot delete other user's chats",
        )

    await methods.delete_chat_async(session, chat)


# ============================================================================
# Chat Message Endpoints
# ============================================================================

router_messages = APIRouter(tags=["chat_messages"], prefix="/chats")


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
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    # Verify the session belongs to the user
    chat = await methods.get_chat_async(session, chat_uuid, user.uuid)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found",
        )
    messages = await methods.list_chat_messages_async(session, chat.uuid, skip=skip, limit=limit)
    total = await methods.count_chat_messages_async(session, chat.uuid)
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
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    # First verify the chat exists and user has access
    chat = await methods.get_chat_async(session, chat_uuid, user.uuid)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found",
        )

    # Authorization check for chat ownership
    if chat.user_uuid is None and not user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot delete messages in global chats",
        )
    if chat.user_uuid is not None and chat.user_uuid != user.uuid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot delete messages in other user's chats",
        )

    # Now get and delete the message
    message = await methods.get_chat_message_async(
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
    await methods.delete_chat_message_async(session, message)
