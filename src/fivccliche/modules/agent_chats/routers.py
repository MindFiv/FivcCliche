import uuid
from datetime import datetime, timedelta, timezone
from collections.abc import AsyncGenerator
from typing import Any, cast

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    Request,
    responses,
    status,
    WebSocket,
)
from fivcglue import IComponentSite
from fivcglue.interfaces.mutexes import IMutexSite
from fivccliche.services.interfaces.auth import IUserAuthenticator
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from fivccliche.services.implements import service_site
from fivccliche.utils import deps
from fivcglue.interfaces import configs
from fivccliche.utils.chats import ChatChannel, ChatRunParser, ChatSnapshot
from fivccliche.utils.deps import (
    IUser,
    get_admin_user_async,
    get_authenticator_async,
    get_authenticated_user_async,
    get_chat_run_provider_async,
    get_config_async,
    get_db_session_async,
    get_mutex_site_async,
)
from fivccliche.utils.filters import FilterError
from fivccliche.utils.schemas import PaginatedResponse
from . import models, schemas, utils
from .filters import ChatEditableFilterSet, ChatFilterSet
from .jobs import ChatDescribeJob
from .jobs.memorize import memorize_created_at_to

# ============================================================================
# Chat Session Endpoints
# ============================================================================

router_chats = APIRouter(tags=["chats"], prefix="/chats")


async def _load_chat_snapshot(
    session: AsyncSession,
    user: IUser,
    chat_uuid: str,
) -> ChatSnapshot | None:
    chat = await utils.get_chat_async(
        session,
        chat_uuid,
        filters=ChatEditableFilterSet(user.uuid, is_superuser=user.is_superuser),
    )
    if chat is None:
        return None
    return ChatSnapshot(
        uuid=chat.uuid,
        agent_id=chat.agent_id,
        context=chat.context or {},
        description=chat.description,
    )


async def _load_realtime_voice_setup(
    session: AsyncSession,
    channel: ChatChannel,
    user: IUser,
    chat: ChatSnapshot,
    start_frame: dict[str, Any],
) -> tuple[Any, Any]:
    from fivccliche.modules.agent_configs.filters import UserScopedReadableFilterSet
    from fivccliche.modules.agent_configs.models import UserASR, UserTTS
    from fivccliche.modules.agent_configs.utils import get_user_scoped_async
    from fivccliche.services.interfaces.agent_speeches import (
        SpeechRecognizeOptions,
        SpeechSynthesisOptions,
    )
    from fivccliche.utils.chats.processors import _INPUT_SAMPLE_RATE, _OUTPUT_SAMPLE_RATE
    from fivccliche.utils.deps import get_speech_provider_async

    context = chat.context or {}
    asr_id = start_frame.get("asr_id") or context.get("asr_id") or "realtime"
    tts_id = start_frame.get("tts_id") or context.get("tts_id") or "realtime"
    if not isinstance(asr_id, str) or not isinstance(tts_id, str):
        await channel.fail_async(
            code="invalid_frame",
            message="asr_id and tts_id must be strings",
            close_code=1003,
        )

    asr = await get_user_scoped_async(
        session,
        UserASR,
        filters=UserScopedReadableFilterSet(
            UserASR.user_uuid, user.uuid, is_superuser=user.is_superuser
        ),
        config_id=asr_id,
    )
    tts = await get_user_scoped_async(
        session,
        UserTTS,
        filters=UserScopedReadableFilterSet(
            UserTTS.user_uuid, user.uuid, is_superuser=user.is_superuser
        ),
        config_id=tts_id,
    )
    if asr is None:
        await channel.fail_async(
            code="asr_not_found",
            message="ASR config not found",
            close_code=4404,
        )
    if tts is None:
        await channel.fail_async(
            code="tts_not_found",
            message="TTS config not found",
            close_code=4404,
        )
    if asr.model_type != "dashscope_realtime":
        await channel.fail_async(
            code="invalid_asr_model_type",
            message="ASR config must use dashscope_realtime",
            close_code=1003,
        )
    if tts.model_type != "dashscope_realtime":
        await channel.fail_async(
            code="invalid_tts_model_type",
            message="TTS config must use dashscope_realtime",
            close_code=1003,
        )

    resolved_asr_id = asr.id
    resolved_tts_id = tts.id
    asr_model = asr.model
    tts_model = tts.model
    asr_api_key = asr.api_key
    tts_api_key = tts.api_key
    asr_base_url = asr.base_url
    tts_base_url = tts.base_url
    asr_model_type = asr.model_type
    tts_model_type = tts.model_type
    language = start_frame.get("language")
    voice_name = start_frame.get("voice")

    asr_provider = await get_speech_provider_async(asr_model_type)
    tts_provider = await get_speech_provider_async(tts_model_type)
    if asr_provider is None or tts_provider is None:
        await channel.fail_async(
            code="speech_unavailable",
            message="Speech provider is not mounted",
            close_code=1011,
        )
    recognizer = await asr_provider.get_recognizer(
        SpeechRecognizeOptions(
            language=language if isinstance(language, str) and language else None,
            format="pcm",
            sample_rate=_INPUT_SAMPLE_RATE,
        ),
        api_key=asr_api_key,
        model=asr_model,
        base_url=asr_base_url,
        id=resolved_asr_id,
    )
    synthesizer = await tts_provider.get_synthesizer(
        SpeechSynthesisOptions(
            voice=voice_name if isinstance(voice_name, str) and voice_name else "",
            format="pcm",
            sample_rate=_OUTPUT_SAMPLE_RATE,
        ),
        api_key=tts_api_key,
        model=tts_model,
        base_url=tts_base_url,
        id=resolved_tts_id,
    )
    return recognizer, synthesizer


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
        is_memorable=chat_create.is_memorable,
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
    description_contains: str | None = Query(
        None, description="Case-insensitive substring match on chat description"
    ),
    created_at_from: datetime | None = Query(
        None, description="Inclusive lower bound on chat created_at"
    ),
    created_at_to: datetime | None = Query(
        None, description="Inclusive upper bound on chat created_at"
    ),
    updated_at_from: datetime | None = Query(
        None, description="Inclusive lower bound on chat updated_at"
    ),
    updated_at_to: datetime | None = Query(
        None, description="Inclusive upper bound on chat updated_at"
    ),
    order_by: schemas.ChatOrderBy = Query(schemas.ChatOrderBy.updated_at),
    order_dir: schemas.ChatOrderDir = Query(schemas.ChatOrderDir.desc),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse[schemas.UserChatSchema]:
    """List all chat sessions for the authenticated user."""
    try:
        filters = ChatFilterSet(user.uuid, is_superuser=user.is_superuser)
        filters.parse(
            agent_id=agent_id,
            description_contains=description_contains,
            created_at_from=created_at_from,
            created_at_to=created_at_to,
            updated_at_from=updated_at_from,
            updated_at_to=updated_at_to,
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
        order_by=order_by,
        order_dir=order_dir,
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
    "/stats/",
    summary="Get chat statistics (Admin only).",
    response_model=schemas.ChatStatsSchema,
)
async def get_chat_stats_async(
    session: AsyncSession = Depends(get_db_session_async),
    admin_user: IUser = Depends(get_admin_user_async),
) -> schemas.ChatStatsSchema:
    """Return aggregate chat and message counts."""
    stats = await utils.get_stats_async(session)
    return schemas.ChatStatsSchema(
        generated_at=datetime.now(timezone.utc),
        chats=stats,
    )


@router_chats.get(
    "/stats/messages/unmemorized/",
    summary="List unmemorized chat messages the memorize job would process (Admin only).",
    response_model=PaginatedResponse[schemas.ChatStatsUnmemorizedMessageSchema],
)
async def list_chat_stats_unmemorized_messages_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    session: AsyncSession = Depends(get_db_session_async),
    config: configs.IConfig = Depends(get_config_async),
    admin_user: IUser = Depends(get_admin_user_async),
) -> PaginatedResponse[schemas.ChatStatsUnmemorizedMessageSchema]:
    """Return the memorize backlog, including messages beyond one job batch."""
    created_at_to = memorize_created_at_to(config)
    messages = await utils.list_unmemorized_chat_messages_async(
        session,
        None,
        created_at_to=created_at_to,
        skip=skip,
        limit=limit,
    )
    total = await utils.count_unmemorized_chat_messages_async(
        session,
        None,
        created_at_to=created_at_to,
    )
    return PaginatedResponse(
        total=total,
        results=[
            schemas.ChatStatsUnmemorizedMessageSchema.model_validate(message)
            for message in messages
        ],
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


@router_chats.patch(
    "/{chat_uuid}/",
    summary="Update a chat session's description.",
    response_model=schemas.UserChatSchema,
)
async def update_chat_async(
    chat_uuid: str,
    chat_update: schemas.UserChatUpdateSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.UserChatSchema:
    """Update a chat session description."""
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
    chat.description = chat_update.description
    chat.updated_at = datetime.now(timezone.utc)
    session.add(chat)
    await session.commit()
    await session.refresh(chat)
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


CHAT_MESSAGE_LOCK_EXPIRE = timedelta(minutes=15)
CHAT_MESSAGE_RUN_TIMEOUT = timedelta(minutes=5)


@router_chats.websocket("/{chat_uuid}/")
async def create_chat_voice_ws_async(
    websocket: WebSocket,
    chat_uuid: str,
    auth: IUserAuthenticator = Depends(get_authenticator_async),
    mutex_site: IMutexSite | None = Depends(get_mutex_site_async),
) -> None:
    """Run a full-duplex realtime voice session for one chat."""
    from fivccliche.utils.chats.processors import ChatVoiceProcessor

    async with ChatChannel(websocket, auth) as channel:
        user = await channel.authenticate_async()
        start_frame = await channel.receive_json_async()
        if start_frame is None:
            return
        if start_frame.get("type") != "start":
            await channel.fail_async(
                code="invalid_message",
                message="A start frame is required",
                close_code=1003,
            )
        async with deps.get_db_session_context_async() as session:
            chat = await _load_chat_snapshot(session, user, chat_uuid)
        if chat is None:
            await channel.fail_async(
                code="chat_not_found",
                message="Chat not found",
                close_code=4404,
            )
        async with deps.get_db_session_context_async() as session:
            recognizer, synthesizer = await _load_realtime_voice_setup(
                session, channel, user, chat, start_frame
            )

        run_provider = await get_chat_run_provider_async()
        mutex = mutex_site.get_mutex(f"chats:message:{chat_uuid}") if mutex_site else None
        async with recognizer, synthesizer:
            await ChatVoiceProcessor(
                channel,
                user=user,
                chat=chat,
                recognizer=recognizer,
                synthesizer=synthesizer,
                run_provider=run_provider,
                mutex=mutex,
                timeout=CHAT_MESSAGE_RUN_TIMEOUT,
            ).process_async()


# ============================================================================
# Chat Message Endpoints
# ============================================================================

router_messages = APIRouter(tags=["chat_messages"], prefix="/chats")


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
        chat_stream = ChatRunParser(chat_uuid=chat_uuid, output_type="sse")
        run_provider = await get_chat_run_provider_async()
        chat_run = run_provider.create_chat_run(
            chat_uuid,
            user_uuid=user.uuid,
            agent_id=chat_agent_id,
            context=chat.context,
        )
        run_events = chat_stream.parse_async(
            chat_run.stream_async(
                chat_message.query,
                mutex=chat_mutex,
                timeout=CHAT_MESSAGE_RUN_TIMEOUT,
            )
        )
    except Exception:
        if chat_mutex:
            await chat_mutex.release_async()
        raise

    query_text = chat_message.query.strip()
    if not (chat.description or "").strip() and query_text and not query_text.startswith("/"):
        background_tasks.add_task(
            ChatDescribeJob(cast(IComponentSite, service_site)).run_async,
            chat.uuid,
            user_uuid=user.uuid,
            query_text=query_text,
        )

    # Release the request-scoped session before SSE so the pool connection is
    # not held for the entire stream duration.
    await session.close()

    return responses.StreamingResponse(
        cast(AsyncGenerator[str, None], run_events),
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
