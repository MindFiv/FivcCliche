"""Scheduled cleanup for storage upload and deletion state."""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import cast

from fivcglue import IComponentSite, query_component
from fivcglue.interfaces import configs
from sqlalchemy.ext.asyncio import AsyncSession

from fivccliche.modules.storages import utils
from fivccliche.modules.storages.schemas import StorageStatus
from fivccliche.services.interfaces.modules import IModuleJob
from fivccliche.services.interfaces.storages import (
    IStorage,
    IStorageProvider,
    StorageError,
)
from fivccliche.utils.types import to_int

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_MINUTES = 60
_DEFAULT_PENDING_TTL_HOURS = 24
_DEFAULT_BATCH_SIZE = 100


class StorageCleanupSetting:
    def __init__(self, config: configs.IConfig | None) -> None:
        session = config.get_session("STORAGE_CLEANUP") if config else None
        interval = to_int(
            session.get_value("INTERVAL_MINUTES") if session else None,
            _DEFAULT_INTERVAL_MINUTES,
        )
        ttl = to_int(
            session.get_value("PENDING_TTL_HOURS") if session else None,
            _DEFAULT_PENDING_TTL_HOURS,
        )
        batch_size = to_int(
            session.get_value("BATCH_SIZE") if session else None,
            _DEFAULT_BATCH_SIZE,
        )
        self.interval_minutes = interval if interval > 0 else _DEFAULT_INTERVAL_MINUTES
        self.pending_ttl_hours = ttl if ttl > 0 else _DEFAULT_PENDING_TTL_HOURS
        self.batch_size = batch_size if batch_size > 0 else _DEFAULT_BATCH_SIZE

    def job_config(self) -> dict:
        return {
            "trigger": "interval",
            "minutes": self.interval_minutes,
            "max_instances": 1,
            "coalesce": True,
            "replace_existing": True,
        }


class StorageCleanupJob(IModuleJob):
    """Reclaim expired pending uploads and retry deleting rows."""

    def __init__(self, component_site: IComponentSite) -> None:
        config = query_component(component_site, configs.IConfig)
        self._setting = StorageCleanupSetting(config)
        self._provider = query_component(component_site, IStorageProvider)
        self._session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]] = (
            _default_session_factory
        )

    def _get_storage(self) -> IStorage | None:
        if self._provider is None:
            return None
        return cast(IStorage | None, self._provider.get_storage())

    @property
    def name(self) -> str:
        return "storages-cleanup"

    @property
    def config(self) -> dict | None:
        if self._get_storage() is None:
            return None
        return self._setting.job_config()

    async def run_async(self) -> None:
        storage = self._get_storage()
        if storage is None:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._setting.pending_ttl_hours)
        async with self._session_factory() as session:
            stale = await utils.list_stale_pending_async(
                session,
                created_before=cutoff,
                limit=self._setting.batch_size,
            )
        for row in stale:
            async with self._session_factory() as session:
                await utils.transition_storage_file_status_async(
                    session,
                    file_uuid=row.uuid,
                    target=StorageStatus.DELETING,
                    current_statuses=(StorageStatus.PENDING,),
                )
                await session.commit()

        async with self._session_factory() as session:
            deleting = await utils.list_deleting_async(
                session,
                limit=self._setting.batch_size,
            )
        for row in deleting:
            try:
                await storage.delete_async(row.object_key)
            except StorageError:
                logger.exception("Could not delete storage object %s", row.uuid)
                continue
            async with self._session_factory() as session:
                await utils.delete_storage_file_row_async(session, file_uuid=row.uuid)
                await session.commit()


@asynccontextmanager
async def _default_session_factory() -> AsyncIterator[AsyncSession]:
    from fivccliche.utils.deps import get_db_session_context_async

    async with get_db_session_context_async() as session:
        yield session
