import asyncio
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    responses,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from fivcplayground.agents import create_agent, AgentRunEvent
from fivcplayground.tools import create_tool_retriever
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


class TaskStreamingGenerator:
    """Generator for streaming agent runs."""

    def __init__(
        self,
        task: asyncio.Task,
        task_queue: asyncio.Queue,
    ):
        self.task = task
        self.task_queue = task_queue

    async def __call__(self, *args, **kwargs):
        while True:
            if self.task.done() and self.task_queue.empty():
                break

            ev, ev_run = await self.task_queue.get()
            if ev == AgentRunEvent.START:
                data = ev_run.model_dump(include={"id", "agent_id", "started_at"})
                data.update({"event": "start"})
                yield f"data: {data}\n\n"

            elif ev == AgentRunEvent.FINISH:
                data = ev_run.model_dump(include={"id", "agent_id", "completed_at"})
                data.update({"event": "finish"})
                yield f"data: {data}\n\n"

            elif ev == AgentRunEvent.STREAM:
                data = ev_run.model_dump(include={"id", "agent_id", "streaming_text"})
                data.update({"event": "stream"})
                yield f"data: {data}\n\n"

            elif ev == AgentRunEvent.TOOL:
                data = ev_run.model_dump(include={"id", "agent_id", "tool_calls"})
                data.update({"event": "tool"})
                yield f"data: {data}\n\n"

            self.task_queue.task_done()


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
    agent = create_agent(
        model_backend=config_provider.get_model_backend(),
        model_config_repository=config_provider.get_model_repository(user_uuid=user.uuid),
        agent_backend=config_provider.get_agent_backend(),
        agent_config_repository=config_provider.get_agent_repository(user_uuid=user.uuid),
        agent_config_id=agent_id,
    )
    agent_tools = create_tool_retriever(
        tool_backend=config_provider.get_tool_backend(),
        tool_repository=config_provider.get_tool_repository(user_uuid=user.uuid),
        embedding_backend=config_provider.get_embedding_backend(),
        embedding_repository=config_provider.get_embedding_repository(user_uuid=user.uuid),
        space_id=user.uuid,
    )
    task_queue = asyncio.Queue()
    task = asyncio.create_task(
        agent.run_async(
            query=chat_query.query,
            tool_retriever=agent_tools,
            agent_run_repository=chat_provider.get_chat_repository(user_uuid=user.uuid),
            callback_queue=lambda ev, run: task_queue.put_nowait((ev, run)),
        )
    )
    return responses.StreamingResponse(TaskStreamingGenerator(task, task_queue))


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
    "/{chat_uuid}",
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
    "/{chat_uuid}",
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
    "/{chat_uuid}/messages/{message_uuid}",
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
