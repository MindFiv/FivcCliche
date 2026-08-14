"""Shared SQL for users (create/get/update)."""

import uuid
from datetime import datetime, timezone
from typing import cast

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from fivccliche.utils import UNSET, UnsetType

from . import models


async def create_user_async(
    session: AsyncSession,
    username: str,
    email: str | None = None,
    full_name: str | None = None,
    password: str | None = None,
    is_superuser: bool = False,
    preferences: dict | None = None,
) -> models.User:
    """Create a new user."""
    user = models.User(
        uuid=str(uuid.uuid4()),
        username=username,
        email=email,
        full_name=full_name,
        preferences=preferences,
        created_at=datetime.now(timezone.utc),
        is_active=True,
        is_superuser=is_superuser,
    )
    if password:
        user.change_password(password)
    session.add(user)
    return user


async def get_user_async(
    session: AsyncSession,
    user_uuid: str | None = None,
    username: str | None = None,
    email: str | None = None,
) -> models.User | None:
    """Get a user by ID, username, or email."""
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


async def update_user_async(
    session: AsyncSession,
    user: models.User,
    username: str | None = None,
    email: str | None = None,
    full_name: str | None | UnsetType = UNSET,
    password: str | None = None,
    is_active: bool | None = None,
    is_superuser: bool | None = None,
    preferences: dict | None | UnsetType = UNSET,
) -> models.User:
    """Update a user."""
    if username is not None:
        user.username = username
    if email is not None:
        user.email = email
    if full_name is not UNSET:
        user.full_name = cast("str | None", full_name)
    if password is not None:
        user.change_password(password)
    if is_active is not None:
        user.is_active = is_active
    if is_superuser is not None:
        user.is_superuser = is_superuser
    if preferences is not UNSET:
        user.preferences = cast("dict | None", preferences)
    session.add(user)
    return user
