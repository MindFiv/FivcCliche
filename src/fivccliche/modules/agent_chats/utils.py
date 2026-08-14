"""Shared SQL for chats and messages."""

from datetime import datetime
from typing import cast

from sqlalchemy import exists, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

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

    chat = models.UserChat(
        user_uuid=user_uuid,
        agent_id=agent_id,
        description=description,
        context=context,
        is_memorable=is_memorable,
    )
    if chat_uuid:
        chat.uuid = chat_uuid
    session.add(chat)
    return chat


async def get_chat_async(
    session: AsyncSession,
    chat_uuid: str,
    user_uuid: str,
    agent_id: str | None = None,
) -> models.UserChat | None:
    """Get a chat session by UUID for a specific user."""
    statement = select(models.UserChat).where(
        (models.UserChat.uuid == chat_uuid)
        & (
            (models.UserChat.user_uuid == user_uuid)
            | (models.UserChat.user_uuid == None)  # noqa: E711
        )
    )
    if agent_id is not None:
        statement = statement.where(models.UserChat.agent_id == agent_id)

    result = await session.execute(statement)
    return result.scalars().first()


async def list_chats_async(
    session: AsyncSession,
    user_uuid: str,
    skip: int = 0,
    limit: int = 100,
    filters: ChatFilterSet | None = None,
) -> list[models.UserChat]:
    """List all chat sessions for a user with pagination."""
    statement = (
        select(models.UserChat)
        .where(
            (models.UserChat.user_uuid == user_uuid)
            | (models.UserChat.user_uuid == None)  # noqa: E711
        )
        .order_by(col(models.UserChat.created_at).desc())
        .offset(skip)
        .limit(limit)
    )
    if filters is not None:
        statement = filters.filter(statement)

    result = await session.execute(statement)
    return list(result.scalars().all())


async def count_chats_async(
    session: AsyncSession,
    user_uuid: str,
    filters: ChatFilterSet | None = None,
) -> int:
    """Count the number of chat sessions for a user."""
    statement = select(func.count(col(models.UserChat.uuid))).where(
        (models.UserChat.user_uuid == user_uuid) | (models.UserChat.user_uuid == None)  # noqa: E711
    )
    if filters is not None:
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


async def list_unmemorized_chat_messages_async(
    session: AsyncSession,
    chat_uuid: str,
    *,
    created_at_to: datetime,
) -> list[models.UserChatMessage]:
    """List aged, completed, unmemorized messages for one chat (oldest first)."""
    statement = (
        select(models.UserChatMessage)
        .where(
            models.UserChatMessage.chat_uuid == chat_uuid,
            col(models.UserChatMessage.is_memorized).is_(False),
            models.UserChatMessage.status == schemas.AgentRunStatus.COMPLETED,
            models.UserChatMessage.created_at <= created_at_to,
        )
        .order_by(col(models.UserChatMessage.created_at).asc())
    )
    result = await session.execute(statement)
    return list(result.scalars().all())


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
