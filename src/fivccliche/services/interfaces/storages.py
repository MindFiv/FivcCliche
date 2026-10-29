"""Storage provider contracts for object access and metadata."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fivcglue import IComponent


@dataclass(frozen=True)
class StorageUpLink:
    method: str
    url: str
    headers: dict[str, str]
    expires_at: datetime


@dataclass(frozen=True)
class StorageDownLink:
    method: str
    url: str
    expires_at: datetime | None


@dataclass(frozen=True)
class StorageInfo:
    byte_size: int
    content_type: str
    etag: str | None


class StorageError(RuntimeError):
    """A storage backend rejected or failed an operation."""


class IStorage(ABC):
    """One configured storage bucket / container."""

    @abstractmethod
    async def upload_async(
        self,
        object_key: str,
        content_type: str,
        *,
        is_public: bool = False,
    ) -> StorageUpLink:
        """Create a short-lived client-direct upload URL."""

    @abstractmethod
    async def download_async(
        self,
        object_key: str,
        *,
        signed: bool = True,
        content_type: str,
        filename: str,
    ) -> StorageDownLink:
        """Create a download URL; unsigned when ``signed`` is False."""

    @abstractmethod
    async def peek_async(self, object_key: str) -> StorageInfo | None:
        """Return object metadata, or None when the key is missing."""

    @abstractmethod
    async def delete_async(self, object_key: str) -> None:
        """Delete a key; a backend missing-key response is success."""


class IStorageProvider(IComponent):
    """Factory for named storage backends (bucket name for OSS)."""

    @abstractmethod
    def get_storage(self, name: str, **kwargs: Any) -> IStorage | None:
        """Return storage for ``name``, or None when configuration is incomplete.

        Optional kwargs may override ``upload_expires_in`` and
        ``download_expires_in`` (seconds).
        """
