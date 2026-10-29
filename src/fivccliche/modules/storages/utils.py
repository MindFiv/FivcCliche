"""Shared state transitions and cleanup queries for storage files."""

from datetime import datetime, timezone
from collections.abc import Iterable
from typing import cast

from sqlalchemy import delete, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from .models import StorageFile
from .schemas import StorageStatus


async def transition_storage_file_status_async(
    session: AsyncSession,
    *,
    file_uuid: str,
    target: StorageStatus,
    current_statuses: Iterable[StorageStatus],
    user_uuid: str | None = None,
    values: dict[str, object] | None = None,
) -> bool:
    """Conditionally transition one row; caller owns commit and refresh."""
    statement = update(StorageFile).where(col(StorageFile.uuid) == file_uuid)
    if user_uuid is not None:
        statement = statement.where(col(StorageFile.user_uuid) == user_uuid)
    statement = statement.where(col(StorageFile.status).in_(tuple(current_statuses)))
    assigned: dict[str, object] = {
        "status": target,
        "updated_at": datetime.now(timezone.utc),
    }
    if values:
        assigned.update(values)
    result = cast(
        CursorResult,
        await session.execute(statement.values(**assigned)),
    )
    return bool(result.rowcount)


async def list_stale_pending_async(
    session: AsyncSession,
    *,
    created_before: datetime,
    limit: int,
) -> list[StorageFile]:
    statement = (
        select(StorageFile)
        .where(col(StorageFile.status) == StorageStatus.PENDING)
        .where(col(StorageFile.created_at) < created_before)
        .order_by(col(StorageFile.created_at).asc())
        .limit(limit)
    )
    return list((await session.execute(statement)).scalars().all())


async def list_deleting_async(
    session: AsyncSession,
    *,
    limit: int,
) -> list[StorageFile]:
    statement = (
        select(StorageFile)
        .where(col(StorageFile.status) == StorageStatus.DELETING)
        .order_by(col(StorageFile.updated_at).asc())
        .limit(limit)
    )
    return list((await session.execute(statement)).scalars().all())


async def delete_storage_file_row_async(
    session: AsyncSession,
    *,
    file_uuid: str,
) -> bool:
    result = cast(
        CursorResult,
        await session.execute(delete(StorageFile).where(col(StorageFile.uuid) == file_uuid)),
    )
    return bool(result.rowcount)
