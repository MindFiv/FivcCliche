"""Shared state transitions and cleanup queries for storage files."""

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import cast

from sqlalchemy import delete, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from fivccliche.services.interfaces.storages import IStorage, IStorageProvider

from .models import StorageFile, StorageFolder
from .schemas import StorageStatus


def resolve_storage_for_file(
    provider: IStorageProvider,
    row: StorageFile,
    folder: StorageFolder | None,
) -> IStorage | None:
    """Return IStorage for a file row; relative object_key stays without prefix."""
    if folder is not None:
        return provider.get_storage(folder.bucket_name, key_prefix=folder.bucket_prefix)
    key_prefix = f"users/{row.user_uuid}" if row.user_uuid else ""
    return provider.get_storage(None, key_prefix=key_prefix)


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
