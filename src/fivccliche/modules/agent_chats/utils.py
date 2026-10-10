"""Shared SQL for chats and messages."""

from datetime import datetime, timezone
from typing import cast

from sqlalchemy import exists, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from fivccliche.utils.filters import FilterSet
from fivccliche.utils.types import UNSET, UnsetType

from . import models, schemas
from .filters import ChatFilterSet


async def create_chat_async(
    session: AsyncSession,
    user_uuid: str,
    agent_id: str,
    chat_uuid: str | None = None,
    description: str | None = None,
    context: dict | None = None,
    is_memorable: bool = False,
) -> models.UserChat:
    """Create a new chat session asynchronously.

    Args:
        session: AsyncSession for database operations
        user_uuid: User UUID (required)
        agent_id: Agent config ID (required)
        chat_uuid: Optional chat UUID (will be auto-generated if not provided)
        description: Optional chat description
        context: Optional chat context (arbitrary JSON data)
        is_memorable: Whether the chat is eligible for memory retention

    Returns:
        Created UserChat instance

    Raises:
        ValueError: If required parameters are missing
    """
    if not agent_id:
        raise ValueError("agent_id is required to create a chat")

    now = datetime.now(timezone.utc)
    chat = models.UserChat(
        user_uuid=user_uuid,
        agent_id=agent_id,
        description=description,
        context=context,
        is_memorable=is_memorable,
        created_at=now,
        updated_at=now,
    )
    if chat_uuid:
        chat.uuid = chat_uuid
    session.add(chat)
    return chat


async def get_chat_async(
    session: AsyncSession,
    chat_uuid: str,
    *,
    filters: FilterSet,
    agent_id: str | None = None,
) -> models.UserChat | None:
    """Get a chat session by UUID (visibility via ``filters``)."""
    statement = select(models.UserChat).where(models.UserChat.uuid == chat_uuid)
    if agent_id is not None:
        statement = statement.where(models.UserChat.agent_id == agent_id)
    statement = filters.filter(statement)

    result = await session.execute(statement)
    return result.scalars().first()


async def list_chats_async(
    session: AsyncSession,
    *,
    filters: ChatFilterSet,
    skip: int = 0,
    limit: int = 100,
    order_by: schemas.ChatOrderBy = schemas.ChatOrderBy.updated_at,
    order_dir: schemas.ChatOrderDir = schemas.ChatOrderDir.desc,
) -> list[models.UserChat]:
    """List chat sessions with pagination (visibility via ``filters``)."""
    order_col = col(getattr(models.UserChat, order_by.value))
    order_expr = order_col.desc() if order_dir == schemas.ChatOrderDir.desc else order_col.asc()
    statement = select(models.UserChat).order_by(order_expr).offset(skip).limit(limit)
    statement = filters.filter(statement)

    result = await session.execute(statement)
    return list(result.scalars().all())


async def count_chats_async(
    session: AsyncSession,
    *,
    filters: ChatFilterSet,
) -> int:
    """Count chat sessions (visibility via ``filters``)."""
    statement = select(func.count(col(models.UserChat.uuid)))
    statement = filters.filter(statement)

    result = await session.execute(statement)
    return result.scalar() or 0


async def list_chat_messages_async(
    session: AsyncSession,
    chat_uuid: str,
    skip: int = 0,
    limit: int = 100,
) -> list[models.UserChatMessage]:
    """List all chat messages for a session with pagination."""
    statement = (
        select(models.UserChatMessage)
        .where(models.UserChatMessage.chat_uuid == chat_uuid)
        .order_by(col(models.UserChatMessage.created_at))
        .offset(skip)
        .limit(limit)
    )
    result = await session.execute(statement)
    return list(result.scalars().all())


async def get_chat_message_async(
    session: AsyncSession,
    message_uuid: str,
    chat_uuid: str,
) -> models.UserChatMessage | None:
    """Get a chat message by UUID."""
    statement = select(models.UserChatMessage).where(
        models.UserChatMessage.uuid == message_uuid,
        models.UserChatMessage.chat_uuid == chat_uuid,
    )
    result = await session.execute(statement)
    return result.scalars().first()


async def create_chat_message_async(
    session: AsyncSession,
    chat_uuid: str,
    query: dict | None = None,
    message_uuid: str | None = None,
    status: schemas.AgentRunStatus | None = None,
    reply: dict | None = None,
    tool_calls: dict | None = None,
    completed_at: datetime | None = None,
) -> models.UserChatMessage:
    """Create a new chat message."""
    if not chat_uuid:
        raise ValueError("Chat UUID is required to create a message")

    message = models.UserChatMessage(
        chat_uuid=chat_uuid,
        query=query,
        reply=reply,
        tool_calls=tool_calls,
        completed_at=completed_at,
    )
    if message_uuid:
        message.uuid = message_uuid
    if status:
        message.status = status
    session.add(message)
    chat = await session.get(models.UserChat, chat_uuid)
    if chat is not None:
        chat.updated_at = datetime.now(timezone.utc)
        session.add(chat)
    return message


async def update_chat_message_async(
    session: AsyncSession,
    message: models.UserChatMessage,
    status: schemas.AgentRunStatus | None = None,
    reply: dict | None | UnsetType = UNSET,
    query: dict | None | UnsetType = UNSET,
    tool_calls: dict | None | UnsetType = UNSET,
    completed_at: datetime | None | UnsetType = UNSET,
    is_memorized: bool | UnsetType = UNSET,
) -> models.UserChatMessage:
    """Update a chat message."""
    if status is not None:
        message.status = status
    if reply is not UNSET:
        message.reply = cast("dict | None", reply)
    if query is not UNSET:
        message.query = cast("dict | None", query)
    if tool_calls is not UNSET:
        message.tool_calls = cast("dict | None", tool_calls)
    if completed_at is not UNSET:
        message.completed_at = cast("datetime | None", completed_at)
    if is_memorized is not UNSET:
        message.is_memorized = cast("bool", is_memorized)
    session.add(message)
    return message


async def list_unmemorized_chats_async(
    session: AsyncSession,
    *,
    created_at_to: datetime,
    limit: int = 50,
) -> list[models.UserChat]:
    """List chats that have aged, completed, unmemorized messages.

    Only chats with a non-null ``user_uuid`` are returned. Ordering is by chat
    ``created_at`` ascending. Uses EXISTS rather than JOIN+DISTINCT so PostgreSQL
    does not compare the ``json`` ``context`` column (which has no equality operator).
    """
    statement = (
        select(models.UserChat)
        .where(
            col(models.UserChat.user_uuid).is_not(None),
            col(models.UserChat.is_memorable).is_(True),
            exists(
                select(1).where(
                    col(models.UserChatMessage.chat_uuid) == models.UserChat.uuid,
                    col(models.UserChatMessage.is_memorized).is_(False),
                    models.UserChatMessage.status == schemas.AgentRunStatus.COMPLETED,
                    models.UserChatMessage.created_at <= created_at_to,
                )
            ),
        )
        .order_by(col(models.UserChat.created_at).asc())
        .limit(limit)
    )
    result = await session.execute(statement)
    return list(result.scalars().all())


def _unmemorized_chat_messages_statement(
    chat_uuid: str | None,
    created_at_to: datetime,
):
    """Select completed unmemorized messages the memorize job would handle.

    ``chat_uuid is None`` lists every eligible chat, including messages waiting
    beyond one job batch. A specific chat keeps the per-chat filter and does not
    re-check ``is_memorable``.
    """
    statement = select(models.UserChatMessage)
    if chat_uuid is None:
        statement = statement.join(
            models.UserChat,
            col(models.UserChat.uuid) == models.UserChatMessage.chat_uuid,
        ).where(
            col(models.UserChat.user_uuid).is_not(None),
            col(models.UserChat.is_memorable).is_(True),
        )
    else:
        statement = statement.where(models.UserChatMessage.chat_uuid == chat_uuid)
    return statement.where(
        col(models.UserChatMessage.is_memorized).is_(False),
        models.UserChatMessage.status == schemas.AgentRunStatus.COMPLETED,
        col(models.UserChatMessage.created_at) <= created_at_to,
    )


async def list_unmemorized_chat_messages_async(
    session: AsyncSession,
    chat_uuid: str | None,
    *,
    created_at_to: datetime,
    skip: int = 0,
    limit: int | None = None,
) -> list[models.UserChatMessage]:
    """List aged, completed, unmemorized messages, oldest first.

    ``limit is None`` returns every match. The memorize job relies on that so a
    single chat is not truncated.
    """
    statement = (
        _unmemorized_chat_messages_statement(chat_uuid, created_at_to)
        .order_by(col(models.UserChatMessage.created_at).asc())
        .offset(skip)
    )
    if limit is not None:
        statement = statement.limit(limit)
    result = await session.execute(statement)
    return list(result.scalars().all())


async def count_unmemorized_chat_messages_async(
    session: AsyncSession,
    chat_uuid: str | None,
    *,
    created_at_to: datetime,
) -> int:
    """Count messages selected by ``list_unmemorized_chat_messages_async``."""
    filtered = _unmemorized_chat_messages_statement(chat_uuid, created_at_to)
    result = await session.execute(select(func.count()).select_from(filtered.subquery()))
    return result.scalar_one() or 0


async def get_stats_async(session: AsyncSession) -> schemas.ChatStats:
    """Aggregate chat and chat-message counts without modifying the session."""
    chat_result = await session.execute(
        select(
            func.count(models.UserChat.uuid),  # type: ignore[arg-type]
            func.count(models.UserChat.uuid).filter(  # type: ignore[arg-type]
                col(models.UserChat.is_memorable).is_(True)
            ),
        )
    )
    chat_total, memorable = chat_result.one()

    message_result = await session.execute(
        select(
            func.count(models.UserChatMessage.uuid),  # type: ignore[arg-type]
            func.count(models.UserChatMessage.uuid).filter(  # type: ignore[arg-type]
                models.UserChatMessage.status == schemas.AgentRunStatus.COMPLETED
            ),
            func.count(models.UserChatMessage.uuid).filter(  # type: ignore[arg-type]
                col(models.UserChatMessage.is_memorized).is_(True)
            ),
            func.count(models.UserChatMessage.uuid).filter(  # type: ignore[arg-type]
                col(models.UserChatMessage.is_memorized).is_(False),
                col(models.UserChat.is_memorable).is_(True),
            ),
        )
        .select_from(models.UserChatMessage)
        .join(
            models.UserChat,
            col(models.UserChat.uuid) == models.UserChatMessage.chat_uuid,
        )
    )
    message_total, completed, memorized, unmemorized = message_result.one()
    return schemas.ChatStats(
        total=chat_total or 0,
        memorable=memorable or 0,
        messages=schemas.ChatStatsMessages(
            total=message_total or 0,
            completed=completed or 0,
            memorized=memorized or 0,
            unmemorized=unmemorized or 0,
        ),
    )


async def delete_unmemorized_chat_messages_async(
    session: AsyncSession,
    chat_uuid: str,
    *,
    created_at_to: datetime,
) -> None:
    """Mark aged, completed, unmemorized messages for one chat as memorized."""
    await session.execute(
        update(models.UserChatMessage)
        .where(
            col(models.UserChatMessage.chat_uuid) == chat_uuid,
            col(models.UserChatMessage.is_memorized).is_(False),
            models.UserChatMessage.status == schemas.AgentRunStatus.COMPLETED,
            col(models.UserChatMessage.created_at) <= created_at_to,
        )
        .values(is_memorized=True)
        .execution_options(synchronize_session=False)
    )
