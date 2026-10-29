"""Tests for storage deletion and pending-object cleanup."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from fivccliche.modules.storages import jobs, models
from fivccliche.modules.storages.schemas import StorageStatus
from fivccliche.modules.users.models import User
from tests.test_storages_api import FakeStorage, FakeStorageProvider


async def make_file(
    session: AsyncSession,
    *,
    status: StorageStatus,
    created_at: datetime,
) -> models.StorageFile:
    owner = User(username=f"owner-{created_at.isoformat()}-{status.value}")
    session.add(owner)
    await session.flush()
    row = models.StorageFile(
        user_uuid=owner.uuid,
        object_key=f"storages/{owner.uuid}",
        filename="demo.txt",
        content_type="text/plain",
        byte_size=12,
        is_public=False,
        status=status,
        created_at=created_at,
        updated_at=created_at,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


class FakeCleanupConfigSession:
    def __init__(self, values: dict[str, Any]):
        self.values = values

    def get_value(self, key: str) -> Any:
        return self.values.get(key)


class FakeCleanupConfig:
    def __init__(
        self,
        cleanup_values: dict[str, Any] | None = None,
        *,
        bucket: str = "example-bucket",
    ):
        self.cleanup_values = cleanup_values or {}
        self.bucket = bucket

    def get_session(self, name: str) -> FakeCleanupConfigSession:
        if name == "STORAGE_CLEANUP":
            return FakeCleanupConfigSession(self.cleanup_values)
        if name == "OSS":
            return FakeCleanupConfigSession({"BUCKET": self.bucket})
        return FakeCleanupConfigSession({})


@asynccontextmanager
async def session_context(session: AsyncSession) -> AsyncIterator[AsyncSession]:
    yield session


def make_job(provider: FakeStorageProvider, session: AsyncSession) -> jobs.StorageCleanupJob:
    job = jobs.StorageCleanupJob.__new__(jobs.StorageCleanupJob)
    job._setting = jobs.StorageCleanupSetting(FakeCleanupConfig())
    job._provider = provider
    job._session_factory = lambda: session_context(session)
    return job


def test_cleanup_setting_defaults() -> None:
    setting = jobs.StorageCleanupSetting(FakeCleanupConfig())
    assert (setting.interval_minutes, setting.pending_ttl_hours, setting.batch_size) == (
        60,
        24,
        100,
    )
    assert setting.bucket == "example-bucket"


async def test_cleanup_deletes_stale_pending(session: AsyncSession) -> None:
    storage = FakeStorage()
    provider = FakeStorageProvider(storage)
    old = await make_file(
        session,
        status=StorageStatus.PENDING,
        created_at=datetime.now(timezone.utc) - timedelta(hours=25),
    )
    fresh = await make_file(
        session,
        status=StorageStatus.PENDING,
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
    )

    await make_job(provider, session).run_async()

    assert old.object_key in storage.deleted
    assert fresh.object_key not in storage.deleted
    assert await session.get(models.StorageFile, old.uuid) is None
    assert await session.get(models.StorageFile, fresh.uuid) is not None


async def test_cleanup_retries_deleting_rows(session: AsyncSession) -> None:
    storage = FakeStorage()
    provider = FakeStorageProvider(storage)
    row = await make_file(
        session,
        status=StorageStatus.DELETING,
        created_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    storage.delete_failures.add(row.object_key)

    await make_job(provider, session).run_async()
    assert await session.get(models.StorageFile, row.uuid) is not None

    storage.delete_failures.clear()
    await make_job(provider, session).run_async()
    assert await session.get(models.StorageFile, row.uuid) is None
