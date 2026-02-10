import asyncio
import json
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

from fivcplayground.agents import create_agent_async, AgentRunEvent
from fivcplayground.tools import create_tool_retriever_async
from fivccliche.services.interfaces.agent_chats import IUserChatProvider
from fivccliche.services.interfaces.agent_configs import IUserConfigProvider
from fivccliche.utils.deps import (
    IUser,
    get_authenticated_user_async,
    get_db_session_async,
    get_config_provider_async,
    get_chat_provider_async,
)
from fivccliche.utils.schemas import PaginatedResponse

from . import methods, schemas


class ChatStreamingGenerator:
    """Generator for streaming agent runs."""

    def __init__(
        self,
        chat_task: asyncio.Task,
        chat_queue: asyncio.Queue,
        chat_uuid: str | None = None,
    ):
        self.chat_task = chat_task
        self.chat_queue = chat_queue
        self.chat_uuid = chat_uuid

    async def __call__(self, *args, **kwargs):
        try:
            while True:
                # Check if task is done and queue is empty
                if self.chat_task.done() and self.chat_queue.empty():
                    # Make sure to get any exception from the task
                    self.chat_task.result()
                    break

                # Try to get an event from the queue with timeout
                try:
                    ev, ev_run = await asyncio.wait_for(self.chat_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    # No event available, continue checking
                    if not self.chat_task.done():
                        print("⏱️  [QUEUE] Timeout waiting for event, task still running")
                    continue

                # Process the event
                data_fields = {
                    "id",
                    "agent_id",
                    "started_at",
                    "completed_at",
                    "query",
                    "reply",
                    "tool_calls",
                }
                if ev == AgentRunEvent.START:
                    data = ev_run.model_dump(mode="json", include=data_fields)
                    # Add chat_uuid from the router context (for new chats)
                    data.update({"chat_uuid": self.chat_uuid})
                    data = {"event": "start", "info": data}
                    data_json = json.dumps(data)
                    yield f"data: {data_json}\n\n"

                elif ev == AgentRunEvent.FINISH:
                    data = ev_run.model_dump(mode="json", include=data_fields)
                    data.update({"chat_uuid": self.chat_uuid})
                    data = {"event": "finish", "info": data}
                    data_json = json.dumps(data)
                    yield f"data: {data_json}\n\n"

                elif ev == AgentRunEvent.STREAM:
                    data = ev_run.model_dump(mode="json", include=data_fields)
                    data.update(
                        {
                            "chat_uuid": self.chat_uuid,
                            "delta": (
                                ev_run.delta.model_dump(mode="json") if ev_run.delta else None
                            ),
                        }
                    )
                    data = {"event": "stream", "info": data}
                    data = json.dumps(data)
                    yield f"data: {data}\n\n"

                elif ev == AgentRunEvent.TOOL:
                    data = ev_run.model_dump(mode="json", include=data_fields)
                    data.update({"chat_uuid": self.chat_uuid})
                    data = {"event": "tool", "info": data}
                    data = json.dumps(data)
                    yield f"data: {data}\n\n"

                self.chat_queue.task_done()

        except Exception as e:
            # Ensure any exception is properly handled
            data = {"event": "error", "info": {"message": str(e)}}
            data = json.dumps(data)
            print(f"❌ [QUEUE] Error in chat queue: {e}")
            yield f"data: {data}\n\n"


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
    agent_id = chat.agent_id if chat else chat_query.agent_id
    print(f"🤖 [AGENT] Creating agent with ID: {agent_id}")

    agent = await create_agent_async(
        model_backend=config_provider.get_model_backend(),
        model_config_repository=config_provider.get_model_repository(
            user_uuid=user.uuid, session=session
        ),
        agent_backend=config_provider.get_agent_backend(),
        agent_config_repository=config_provider.get_agent_repository(
            user_uuid=user.uuid, session=session
        ),
        agent_config_id=agent_id,
    )
    agent_tools = await create_tool_retriever_async(
        tool_backend=config_provider.get_tool_backend(),
        tool_config_repository=config_provider.get_tool_repository(
            user_uuid=user.uuid, session=session
        ),
        embedding_backend=config_provider.get_embedding_backend(),
        embedding_config_repository=config_provider.get_embedding_repository(
            user_uuid=user.uuid, session=session
        ),
        space_id=user.uuid,
    )
    chat_queue = asyncio.Queue()
    chat_uuid = chat.uuid if chat else str(uuid.uuid4())

    # Determine owner_uuid: None for superusers (global chats), user.uuid for regular users
    owner_uuid = None if user.is_superuser else user.uuid

    # Debug: Event callback wrapper
    def _event_callback(ev, run):
        chat_queue.put_nowait((ev, run))

    chat_task = asyncio.create_task(
        agent.run_async(
            query=chat_query.query,
            tool_retriever=agent_tools,
            agent_run_repository=chat_provider.get_chat_repository(
                user_uuid=owner_uuid, session=session
            ),
            agent_run_session_id=chat_uuid,
            event_callback=_event_callback,
        )
    )
    chat_streamer = ChatStreamingGenerator(chat_task, chat_queue, chat_uuid=chat_uuid)
    return responses.StreamingResponse(chat_streamer())


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
