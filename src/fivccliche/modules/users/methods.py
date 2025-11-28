"""User service module with functions for user operations."""

import uuid
from datetime import datetime

from passlib.context import CryptContext

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from fivccliche.modules.users.models import User
from fivccliche.modules.users.schemas import UserCreate, UserUpdate

# Password hashing context - using argon2 for better security and no length limits
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


def hash_user_password(password: str) -> str:
    """Hash a password using argon2."""
    return pwd_context.hash(password)


def verify_user_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


async def create_user_async(session: AsyncSession, user_create: UserCreate) -> User:
    """Create a new user."""
    user = User(
        id=str(uuid.uuid4()),
        username=user_create.username,
        email=user_create.email,
        hashed_password=hash_user_password(user_create.password),
        created_at=datetime.now(),
        is_active=True,
        is_superuser=False,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def get_user_async(
    session: AsyncSession,
    user_id: str | None = None,
    username: str | None = None,
    email: str | None = None,
) -> User | None:
    """Get a user by ID, username, or email.

    Args:
        session: Database session
        user_id: User ID to search by
        username: Username to search by
        email: Email to search by

    Returns:
        User if found, None otherwise

    Raises:
        ValueError: If no search criteria are provided
    """
    if not any([user_id, username, email]):
        raise ValueError(
            "At least one search criterion (user_id, username, or email) must be provided"
        )

    statement = select(User)
    if user_id:
        statement = statement.where(User.id == user_id)
    if username:
        statement = statement.where(User.username == username)
    if email:
        statement = statement.where(User.email == email)
    result = await session.execute(statement)
    return result.scalars().first()


async def list_users_async(session: AsyncSession, skip: int = 0, limit: int = 100) -> list[User]:
    """List all users with pagination.

    Args:
        session: Database session
        skip: Number of users to skip
        limit: Maximum number of users to return

    Returns:
        List of users
    """
    statement = select(User).offset(skip).limit(limit)
    result = await session.execute(statement)
    return list(result.scalars().all())


async def count_users_async(session: AsyncSession) -> int:
    """Count the number of users.

    Args:
        session: Database session

    Returns:
        Number of users
    """

    statement = select(func.count(User.id))
    result = await session.execute(statement)
    return result.scalar() or 0


async def update_user_async(session: AsyncSession, user: User, user_update: UserUpdate) -> User:
    """Update a user."""
    if user_update.username is not None:
        user.username = user_update.username
    if user_update.email is not None:
        user.email = user_update.email
    if user_update.password is not None:
        user.hashed_password = hash_user_password(user_update.password)
    if user_update.is_active is not None:
        user.is_active = user_update.is_active
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def delete_user_async(session: AsyncSession, user: User) -> None:
    """Delete a user."""
    await session.delete(user)
    await session.commit()


async def authenticate_user_async(
    session: AsyncSession, username: str, password: str
) -> User | None:
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
