"""Chat service module with functions for chat operations."""

from datetime import datetime

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from . import models, schemas


async def create_chat_async(
    session: AsyncSession,
    user_uuid: str,
    agent_id: str,
    chat_uuid: str | None = None,
    description: str | None = None,
    context: dict | None = None,
    **kwargs,
) -> models.UserChat:
    """Create a new chat session asynchronously.

    Args:
        session: AsyncSession for database operations
        user_uuid: User UUID (required)
        agent_id: Agent config ID (required)
        chat_uuid: Optional chat UUID (will be auto-generated if not provided)
        description: Optional chat description
        context: Optional chat context (arbitrary JSON data)
        **kwargs: Additional arguments (ignored)

    Returns:
        Created UserChat instance

    Raises:
        ValueError: If required parameters are missing
    """
    if not agent_id:
        raise ValueError("agent_id is required to create a chat")

    chat = models.UserChat(
        uuid=chat_uuid,  # Will use auto-generated UUID if None
        user_uuid=user_uuid,
        agent_id=agent_id,
        description=description,
        context=context,
    )
    session.add(chat)
    await session.commit()
    await session.refresh(chat)
    return chat


async def get_chat_async(
    session: AsyncSession,
    chat_uuid: str,
    user_uuid: str,
    agent_id: str | None = None,
    **kwargs,  # ignore additional arguments
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
    agent_id: str | None = None,
    **kwargs,
) -> list[models.UserChat]:
    """List all chat sessions for a user with pagination."""
    statement = (
        select(models.UserChat)
        .where(
            (models.UserChat.user_uuid == user_uuid)
            | (models.UserChat.user_uuid == None)  # noqa: E711
        )
        .offset(skip)
        .limit(limit)
    )
    if agent_id is not None:
        statement = statement.where(models.UserChat.agent_id == agent_id)

    result = await session.execute(statement)
    return list(result.scalars().all())


async def count_chats_async(
    session: AsyncSession,
    user_uuid: str,
    agent_id: str | None = None,
    **kwargs,
) -> int:
    """Count the number of chat sessions for a user."""
    statement = select(func.count(models.UserChat.uuid)).where(
        (models.UserChat.user_uuid == user_uuid) | (models.UserChat.user_uuid == None)  # noqa: E711
    )
    if agent_id is not None:
        statement = statement.where(models.UserChat.agent_id == agent_id)

    result = await session.execute(statement)
    return result.scalar() or 0


async def delete_chat_async(session: AsyncSession, chat: models.UserChat, **kwargs) -> None:
    """Delete a chat session."""
    await session.delete(chat)
    await session.commit()


async def list_chat_messages_async(
    session: AsyncSession,
    chat_uuid: str,
    skip: int = 0,
    limit: int = 100,
    **kwargs,  # ignore additional arguments
) -> list[models.UserChatMessage]:
    """List all chat messages for a session with pagination."""
    statement = (
        select(models.UserChatMessage)
        .where(models.UserChatMessage.chat_uuid == chat_uuid)
        .order_by(models.UserChatMessage.created_at)
        .offset(skip)
        .limit(limit)
    )
    result = await session.execute(statement)
    return list(result.scalars().all())


async def count_chat_messages_async(
    session: AsyncSession,
    chat_uuid: str,
    **kwargs,  # ignore additional arguments
) -> int:
    """Count the number of chat messages for a session."""
    statement = select(func.count(models.UserChatMessage.uuid)).where(
        models.UserChatMessage.chat_uuid == chat_uuid
    )
    result = await session.execute(statement)
    return result.scalar() or 0


async def get_chat_message_async(
    session: AsyncSession,
    message_uuid: str,
    chat_uuid: str,
    **kwargs,  # ignore additional arguments
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
    query: dict,
    reply: dict | None = None,
    tool_calls: dict | None = None,
    message_uuid: str | None = None,
    **kwargs,  # ignore additional arguments
) -> models.UserChatMessage:
    """Create a new chat message.

    Args:
        session: AsyncSession for database operations
        chat_uuid: Chat UUID (required)
        query: Query data (required)
        reply: Optional reply data
        tool_calls: Optional tool calls data
        message_uuid: Optional message UUID (will be auto-generated if not provided)
        **kwargs: Additional arguments (ignored)

    Returns:
        Created UserChatMessage instance

    Raises:
        ValueError: If chat_uuid is missing
    """
    if not chat_uuid:
        raise ValueError("Chat UUID is required to create a message")

    message = models.UserChatMessage(
        uuid=message_uuid,  # Will use auto-generated UUID if None
        chat_uuid=chat_uuid,
        query=query,
        reply=reply,
        tool_calls=tool_calls,
    )
    session.add(message)
    await session.commit()
    await session.refresh(message)
    return message


async def update_chat_message_async(
    session: AsyncSession,
    message: models.UserChatMessage,
    status: schemas.AgentRunStatus | None = None,
    reply: dict | None = None,
    query: dict | None = None,
    tool_calls: dict | None = None,
    completed_at: datetime | None = None,
    **kwargs,  # ignore additional arguments
) -> models.UserChatMessage:
    """Update a chat message."""
    if status is not None:
        message.status = status
    if reply is not None:
        message.reply = reply
    if query is not None:
        message.query = query
    if tool_calls is not None:
        message.tool_calls = tool_calls
    if completed_at is not None:
        message.completed_at = completed_at
    session.add(message)
    await session.commit()
    await session.refresh(message)
    return message


async def delete_chat_message_async(
    session: AsyncSession,
    message: models.UserChatMessage,
    **kwargs,  # ignore additional arguments
) -> None:
    """Delete a chat message."""
    await session.delete(message)
    await session.commit()
