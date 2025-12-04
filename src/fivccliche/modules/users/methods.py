"""User service module with functions for user operations."""

import uuid
from datetime import datetime

from passlib.context import CryptContext

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from . import models

# Password hashing context - using argon2 for better security and no length limits
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


def hash_user_password(password: str) -> str:
    """Hash a password using argon2."""
    return pwd_context.hash(password)


def verify_user_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


async def create_user_async(
    session: AsyncSession,
    username: str,
    email: str | None = None,
    password: str | None = None,
    is_superuser: bool = False,
) -> models.User:
    """Create a new user.

    Args:
        session: Database session
        username: Username for the new user
        email: Email address for the new user (optional)
        password: Password for the new user (optional)
        is_superuser: Whether the user is a superuser (default: False)

    Returns:
        The created User object
    """
    user = models.User(
        uuid=str(uuid.uuid4()),
        username=username,
        email=email,
        hashed_password=hash_user_password(password) if password else None,
        created_at=datetime.now(),
        is_active=True,
        is_superuser=is_superuser,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def get_user_async(
    session: AsyncSession,
    user_uuid: str | None = None,
    username: str | None = None,
    email: str | None = None,
) -> models.User | None:
    """Get a user by ID, username, or email.

    Args:
        session: Database session
        user_uuid: User ID to search by
        username: Username to search by
        email: Email to search by

    Returns:
        User if found, None otherwise

    Raises:
        ValueError: If no search criteria are provided
    """
    if not any([user_uuid, username, email]):
        raise ValueError(
            "At least one search criterion (user_uuid, username, or email) must be provided"
        )

    statement = select(models.User)
    if user_uuid:
        statement = statement.where(models.User.uuid == user_uuid)
    if username:
        statement = statement.where(models.User.username == username)
    if email:
        statement = statement.where(models.User.email == email)
    result = await session.execute(statement)
    return result.scalars().first()


async def list_users_async(
    session: AsyncSession, skip: int = 0, limit: int = 100
) -> list[models.User]:
    """List all users with pagination.

    Args:
        session: Database session
        skip: Number of users to skip
        limit: Maximum number of users to return

    Returns:
        List of users
    """
    statement = select(models.User).offset(skip).limit(limit)
    result = await session.execute(statement)
    return list(result.scalars().all())


async def count_users_async(session: AsyncSession) -> int:
    """Count the number of users.

    Args:
        session: Database session

    Returns:
        Number of users
    """

    statement = select(func.count(models.User.uuid))
    result = await session.execute(statement)
    return result.scalar() or 0


async def update_user_async(
    session: AsyncSession,
    user: models.User,
    username: str | None = None,
    email: str | None = None,
    password: str | None = None,
    is_active: bool | None = None,
    is_superuser: bool | None = None,
) -> models.User:
    """Update a user.

    Args:
        session: Database session
        user: User object to update
        username: New username (optional)
        email: New email address (optional)
        password: New password (optional)
        is_active: New active status (optional)
        is_superuser: New superuser status (optional)

    Returns:
        The updated User object
    """
    if username is not None:
        user.username = username
    if email is not None:
        user.email = email
    if password is not None:
        user.hashed_password = hash_user_password(password)
    if is_active is not None:
        user.is_active = is_active
    if is_superuser is not None:
        user.is_superuser = is_superuser
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def delete_user_async(session: AsyncSession, user: models.User) -> None:
    """Delete a user."""
    await session.delete(user)
    await session.commit()


async def authenticate_user_async(
    session: AsyncSession, username: str, password: str
) -> models.User | None:
    """Authenticate a user by username and password.

    Args:
        session: Database session
        username: User's username
        password: User's password (plain text)

    Returns:
        User if authentication successful, None otherwise
    """
    user = await get_user_async(session, username=username)
    if not user:
        return None
    if not verify_user_password(password, user.hashed_password):
        return None
    return user
