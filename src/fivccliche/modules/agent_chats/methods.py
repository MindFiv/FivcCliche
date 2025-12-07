"""Chat service module with functions for chat operations."""

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from . import models


async def get_chat_async(
    session: AsyncSession, chat_uuid: str, user_uuid: str, **kwargs
) -> models.UserChat | None:
    """Get a chat session by UUID for a specific user."""
    statement = select(models.UserChat).where(
        (models.UserChat.uuid == chat_uuid) & (models.UserChat.user_uuid == user_uuid)
    )
    result = await session.execute(statement)
    return result.scalars().first()


async def list_chats_async(
    session: AsyncSession,
    user_uuid: str,
    skip: int = 0,
    limit: int = 100,
    **kwargs,
) -> list[models.UserChat]:
    """List all chat sessions for a user with pagination."""
    statement = (
        select(models.UserChat)
        .where(models.UserChat.user_uuid == user_uuid)
        .offset(skip)
        .limit(limit)
    )
    result = await session.execute(statement)
    return list(result.scalars().all())


async def count_chats_async(
    session: AsyncSession,
    user_uuid: str,
    **kwargs,
) -> int:
    """Count the number of chat sessions for a user."""
    statement = select(func.count(models.UserChat.uuid)).where(
        models.UserChat.user_uuid == user_uuid
    )
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


async def delete_chat_message_async(
    session: AsyncSession,
    message: models.UserChatMessage,
    **kwargs,  # ignore additional arguments
) -> None:
    """Delete a chat message."""
    await session.delete(message)
    await session.commit()
