"""Chat service module with functions for chat operations."""

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from . import models


async def get_chat_async(
    session: AsyncSession, chat_uuid: str, user_uuid: str, **kwargs
) -> models.Chat | None:
    """Get a chat session by UUID for a specific user."""
    statement = select(models.Chat).where(
        (models.Chat.uuid == chat_uuid) & (models.Chat.user_uuid == user_uuid)
    )
    result = await session.execute(statement)
    return result.scalars().first()


async def list_chats_async(
    session: AsyncSession,
    user_uuid: str,
    skip: int = 0,
    limit: int = 100,
    **kwargs,
) -> list[models.Chat]:
    """List all chat sessions for a user with pagination."""
    statement = (
        select(models.Chat).where(models.Chat.user_uuid == user_uuid).offset(skip).limit(limit)
    )
    result = await session.execute(statement)
    return list(result.scalars().all())


async def count_chats_async(
    session: AsyncSession,
    user_uuid: str,
    **kwargs,
) -> int:
    """Count the number of chat sessions for a user."""
    statement = select(func.count(models.Chat.uuid)).where(models.Chat.user_uuid == user_uuid)
    result = await session.execute(statement)
    return result.scalar() or 0


async def delete_chat_async(session: AsyncSession, chat: models.Chat, **kwargs) -> None:
    """Delete a chat session."""
    await session.delete(chat)
    await session.commit()


async def list_chat_messages_async(
    session: AsyncSession,
    chat_uuid: str,
    skip: int = 0,
    limit: int = 100,
    **kwargs,  # ignore additional arguments
) -> list[models.ChatMessage]:
    """List all chat messages for a session with pagination."""
    statement = (
        select(models.ChatMessage)
        .where(models.ChatMessage.chat_uuid == chat_uuid)
        .order_by(models.ChatMessage.created_at)
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
    statement = select(func.count(models.ChatMessage.uuid)).where(
        models.ChatMessage.chat_uuid == chat_uuid
    )
    result = await session.execute(statement)
    return result.scalar() or 0


async def get_chat_message_async(
    session: AsyncSession,
    message_uuid: str,
    chat_uuid: str,
    **kwargs,  # ignore additional arguments
) -> models.ChatMessage | None:
    """Get a chat message by UUID."""
    statement = select(models.ChatMessage).where(
        models.ChatMessage.uuid == message_uuid,
        models.ChatMessage.chat_uuid == chat_uuid,
    )
    result = await session.execute(statement)
    return result.scalars().first()


async def delete_chat_message_async(
    session: AsyncSession,
    message: models.ChatMessage,
    **kwargs,  # ignore additional arguments
) -> None:
    """Delete a chat message."""
    await session.delete(message)
    await session.commit()
