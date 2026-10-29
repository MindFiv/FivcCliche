# Storages OSS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build authenticated client-direct Aliyun OSS file uploads with verified completion, immutable application-level permissions, and orphan cleanup.

**Architecture:** A new `storages` module owns file metadata, HTTP routes, ownership filters, and state transitions. `storages_oss` implements a narrow storage provider boundary over the official `oss2` SDK, while API tests use a fake provider and never contact OSS. Cleanup runs as an `IModuleJob` and uses conditional SQL state transitions so completion, deletion, and cleanup can race safely.

**Tech Stack:** FastAPI, SQLModel/SQLAlchemy async, PostgreSQL via existing pg0 test fixtures, fivcglue component registration, Aliyun OSS SDK `oss2`, Pydantic v2, APScheduler.

---

## File Structure

- Create `src/fivccliche/modules/storages/__init__.py`: public module exports.
- Create `src/fivccliche/modules/storages/schemas.py`: permission/status enums, upload request validation, response models.
- Create `src/fivccliche/modules/storages/models.py`: `StorageFile` SQLModel table and `to_schema`.
- Create `src/fivccliche/modules/storages/filters.py`: owner, status, permission, type, and date filters.
- Create `src/fivccliche/modules/storages/utils.py`: shared conditional state transitions and cleanup queries; callers own commits.
- Create `src/fivccliche/modules/storages/routers.py`: HTTP upload, completion, list, access, and deletion routes.
- Create `src/fivccliche/modules/storages/jobs.py`: `storages-cleanup`.
- Create `src/fivccliche/modules/storages/services.py`: `IModule` registration.
- Create `src/fivccliche/services/interfaces/storages.py`: provider boundary and signed-result/head DTOs.
- Create `src/fivccliche/services/implements/storages_oss.py`: Aliyun OSS provider.
- Modify `src/fivccliche/utils/deps.py`: lazy provider dependency.
- Modify `src/fivccliche/settings/services.yml`: provider and module registration.
- Modify `pyproject.toml` and `uv.lock`: add `oss2`.
- Modify `tests/conftest.py`: import the model into SQLModel metadata.
- Create `tests/test_storages_models.py`.
- Create `tests/test_storages_oss.py`.
- Create `tests/test_storages_api.py`.
- Create `tests/test_storages_cleanup.py`.
- Modify `CLAUDE.md` and `docs/architecture.md`; create `docs/storages.md`.

Execution must start by invoking `superpowers:using-git-worktrees` and following its directory-selection process exactly. The worktree must branch from `main`, beginning with commit `86514b5` or newer. Run `uv sync` and `uv run pytest tests/test_cli_jobs.py tests/test_modules_scheduler.py -v` before Task 1. If those baseline tests fail, stop and report the failures instead of attributing them to this feature.

### Task 1: Storage Contracts, Model, And Filters

**Files:**
- Create: `src/fivccliche/modules/storages/__init__.py`
- Create: `src/fivccliche/modules/storages/schemas.py`
- Create: `src/fivccliche/modules/storages/models.py`
- Create: `src/fivccliche/modules/storages/filters.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_storages_models.py`

- [ ] **Step 1: Write the failing schema/model/filter tests**

Create `tests/test_storages_models.py` with this complete content:

```python
"""Storage file schema, model, and filter tests."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from fivccliche.modules.storages.filters import (
    StorageFileFilterSet,
    StorageFileOwnedFilterSet,
)
from fivccliche.modules.storages.models import StorageFile
from fivccliche.modules.storages.schemas import (
    StorageFileCreateSchema,
    StorageFileSchema,
    StoragePermission,
    StorageStatus,
)


def test_create_schema_accepts_allowed_type_pair() -> None:
    body = StorageFileCreateSchema(
        filename="demo.mp4",
        content_type="video/mp4",
        byte_size=20 * 1024 * 1024,
        permission=StoragePermission.PUBLIC_READ,
    )
    assert body.filename == "demo.mp4"
    assert body.permission == StoragePermission.PUBLIC_READ


@pytest.mark.parametrize(
    ("filename", "content_type", "byte_size"),
    [
        ("demo.txt", "text/plain", 0),
        ("demo.txt", "text/plain", 20 * 1024 * 1024 + 1),
        ("demo.exe", "application/octet-stream", 12),
        ("demo.txt", "application/pdf", 12),
        ("directory/demo.txt", "text/plain", 12),
        ("back\\slash.txt", "text/plain", 12),
    ],
)
def test_create_schema_rejects_invalid_uploads(
    filename: str, content_type: str, byte_size: int
) -> None:
    with pytest.raises(ValidationError):
        StorageFileCreateSchema(
            filename=filename,
            content_type=content_type,
            byte_size=byte_size,
            permission=StoragePermission.PRIVATE,
        )


async def test_storage_file_to_schema(session: AsyncSession) -> None:
    created_at = datetime(2026, 10, 28, 8, 0, tzinfo=UTC)
    row = StorageFile(
        user_uuid="user-uuid",
        object_key="storages/file-uuid",
        filename="demo.png",
        content_type="image/png",
        byte_size=12,
        permission=StoragePermission.PRIVATE,
        status=StorageStatus.PENDING,
        created_at=created_at,
        updated_at=created_at,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)

    result = row.to_schema()
    assert isinstance(result, StorageFileSchema)
    assert result.uuid == row.uuid
    assert result.permission == StoragePermission.PRIVATE
    assert result.status == StorageStatus.PENDING


async def test_filters_apply_owner_and_http_fields(session: AsyncSession) -> None:
    created_at = datetime(2026, 10, 28, 8, 0, tzinfo=UTC)
    rows = [
        StorageFile(
            user_uuid="owner",
            object_key=f"storages/{index}",
            filename="demo.txt",
            content_type="text/plain",
            byte_size=12,
            permission=StoragePermission.PRIVATE if index == 0 else StoragePermission.PUBLIC_READ,
            status=StorageStatus.PENDING if index == 0 else StorageStatus.READY,
            created_at=created_at,
            updated_at=created_at,
        )
        for index in range(2)
    ]
    rows[1].user_uuid = "other"
    session.add_all(rows)
    await session.commit()

    filters = StorageFileFilterSet("owner")
    filters.parse(
        status=StorageStatus.PENDING,
        permission=StoragePermission.PRIVATE,
        content_type="text/plain",
        created_at_from=created_at,
        created_at_to=created_at,
    )
    owned = (await session.execute(filters.filter(select(StorageFile)))).scalars().all()
    assert [row.user_uuid for row in owned] == ["owner"]
    assert [row.status for row in owned] == [StorageStatus.PENDING]

    owned_filter = StorageFileOwnedFilterSet("owner")
    result = (
        await session.execute(owned_filter.filter(select(StorageFile).where(StorageFile.uuid == rows[1].uuid)))
    ).scalar_one_or_none()
    assert result is None
```

Modify the model imports in `tests/conftest.py` to include:

```python
from fivccliche.modules.storages.models import StorageFile  # noqa: F401
```

- [ ] **Step 2: Run the tests and verify the import fails**

Run:

```bash
uv run pytest tests/test_storages_models.py -v
```

Expected: collection fails because `fivccliche.modules.storages` does not exist.

- [ ] **Step 3: Implement schemas**

Create `src/fivccliche/modules/storages/schemas.py`:

```python
"""Request and response schemas for user-owned storage files."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_FILE_BYTES = 20 * 1024 * 1024

STORAGE_TYPES: dict[str, str] = {
    ".gif": "image/gif",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
    ".csv": "text/csv",
    ".json": "application/json",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".weba": "audio/webm",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}


class StoragePermission(StrEnum):
    """Application-level access permission; OSS ACLs always remain private."""

    PRIVATE = "PRIVATE"
    PUBLIC_READ = "PUBLIC_READ"


class StorageStatus(StrEnum):
    """Lifecycle state for a storage file row."""

    PENDING = "PENDING"
    READY = "READY"
    DELETING = "DELETING"


class StorageFileCreateSchema(BaseModel):
    """Validated request used to create a presigned upload."""

    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(min_length=3, max_length=255)
    byte_size: int = Field(gt=0, le=MAX_FILE_BYTES)
    permission: StoragePermission

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        value = value.strip()
        if not value or "/" in value or "\\" in value or "\x00" in value:
            raise ValueError("filename must be one path component")
        return value

    @model_validator(mode="after")
    def validate_type(self) -> StorageFileCreateSchema:
        suffix = self.filename.rsplit(".", 1)[-1].lower() if "." in self.filename else ""
        suffix = f".{suffix}"
        expected = STORAGE_TYPES.get(suffix)
        if expected is None or self.content_type != expected:
            raise ValueError("content_type does not match an allowed filename suffix")
        return self


class StorageFileSchema(BaseModel):
    """Storage file metadata response."""

    uuid: str
    user_uuid: str
    object_key: str
    filename: str
    content_type: str
    byte_size: int
    permission: StoragePermission
    status: StorageStatus
    etag: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None


class StorageFileUploadSchema(StorageFileSchema):
    """Metadata plus the client-direct upload instructions."""

    upload_method: str = "PUT"
    upload_url: str
    upload_headers: dict[str, str]
    upload_expires_at: datetime


class StorageFileUrlSchema(BaseModel):
    """Short-lived owner download response."""

    url: str
    expires_at: datetime
```

- [ ] **Step 4: Implement model and filters**

Create `src/fivccliche/modules/storages/__init__.py`:

```python
"""User-owned storage module."""
```

Create `src/fivccliche/modules/storages/models.py`:

```python
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Column, DateTime, Index, String, UniqueConstraint
from sqlmodel import Field, SQLModel

from . import schemas


class StorageFile(SQLModel, table=True):
    """Metadata for one object uploaded through the storages module."""

    __tablename__ = "storage_file"
    __table_args__ = (
        Index("ix_storage_file_user_created", "user_uuid", "created_at"),
        Index("ix_storage_file_status_created", "status", "created_at"),
        UniqueConstraint("object_key", name="uq_storage_file_object_key"),
    )

    uuid: str = Field(
        default_factory=lambda: str(uuid4()),
        primary_key=True,
        max_length=36,
    )
    user_uuid: str = Field(foreign_key="user.uuid", index=True, max_length=36)
    object_key: str = Field(max_length=512)
    filename: str = Field(max_length=255)
    content_type: str = Field(max_length=255)
    byte_size: int = Field(gt=0)
    permission: schemas.StoragePermission = Field(
        sa_column=Column(String(16), nullable=False)
    )
    status: schemas.StorageStatus = Field(
        sa_column=Column(String(16), nullable=False)
    )
    etag: str | None = Field(default=None, max_length=128)
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False),
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )

    def to_schema(self) -> schemas.StorageFileSchema:
        return schemas.StorageFileSchema(
            uuid=self.uuid,
            user_uuid=self.user_uuid,
            object_key=self.object_key,
            filename=self.filename,
            content_type=self.content_type,
            byte_size=self.byte_size,
            permission=self.permission
            status=self.status
            etag=self.etag,
            created_at=self.created_at,
            updated_at=self.updated_at,
            completed_at=self.completed_at,
        )
```

Create `src/fivccliche/modules/storages/filters.py`:

```python
"""HTTP and ownership filters for storage files."""

from __future__ import annotations

import operator

from fivccliche.utils.filters import (
    FilterEditableField,
    FilterSimpleField,
    FilterSet,
)

from . import models


class StorageFileFilterSet(FilterSet):
    """Owned-file list filters; there are no global storage rows."""

    def __init__(self, user_uuid: str) -> None:
        super().__init__(
            [
                FilterEditableField(
                    "owner",
                    models.StorageFile.user_uuid,
                    user_uuid,
                    is_superuser=False,
                ),
                FilterSimpleField("status", models.StorageFile.status, operator.eq),
                FilterSimpleField("permission", models.StorageFile.permission, operator.eq),
                FilterSimpleField("content_type", models.StorageFile.content_type, operator.eq),
                FilterSimpleField("created_at_from", models.StorageFile.created_at, operator.ge),
                FilterSimpleField("created_at_to", models.StorageFile.created_at, operator.le),
            ]
        )


class StorageFileOwnedFilterSet(FilterSet):
    """Ownership predicate for one known storage file."""

    def __init__(self, user_uuid: str) -> None:
        super().__init__(
            [
                FilterEditableField(
                    "editable",
                    models.StorageFile.user_uuid,
                    user_uuid,
                    is_superuser=False,
                ),
            ]
        )

```

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run pytest tests/test_storages_models.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Run formatting and focused static checks**

Run:

```bash
uv run black src/fivccliche/modules/storages tests/test_storages_models.py
uv run ruff check src/fivccliche/modules/storages tests/test_storages_models.py
uv run mypy src/fivccliche/modules/storages
```

Expected: formatter changes files if needed; Ruff and MyPy exit 0.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/fivccliche/modules/storages tests/conftest.py tests/test_storages_models.py
git commit -m "feat: add storage file contracts"
```

Before committing, run `git status --short` and include only files changed by this task.

### Task 2: Aliyun OSS Provider Boundary

**Files:**
- Create: `src/fivccliche/services/interfaces/storages.py`
- Create: `src/fivccliche/services/implements/storages_oss.py`
- Modify: `src/fivccliche/utils/deps.py`
- Modify: `src/fivccliche/settings/services.yml`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Test: `tests/test_storages_oss.py`

- [ ] **Step 1: Add the SDK**

Run:

```bash
uv add "oss2>=2.18.1"
```

Expected: `pyproject.toml` gains the production dependency and `uv.lock` is updated.

- [ ] **Step 2: Write failing provider tests**

Create `tests/test_storages_oss.py`:

```python
"""Tests for the Aliyun OSS storage provider boundary."""

from typing import Any

import oss2
import pytest

from fivccliche.services.implements.storages_oss import OssStorageProvider
from fivccliche.services.interfaces.storages import StorageProviderError


class FakeConfigSession:
    def __init__(self, values: dict[str, str]):
        self.values = values

    def get_value(self, key: str) -> str | None:
        return self.values.get(key)


class FakeConfig:
    def __init__(self, values: dict[str, str]):
        self.values = values

    def get_session(self, name: str) -> FakeConfigSession:
        return FakeConfigSession(self.values if name == "OSS" else {})


class FakeBucket:
    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.delete_errors: list[str] = []

    def sign_url(
        self,
        method: str,
        key: str,
        expires: int,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> str:
        return f"https://oss.test/{key}?method={method}&expires={expires}&headers={headers}%26params={params}"

    def head_object(self, key: str) -> dict[str, Any]:
        if key not in self.objects:
            raise oss2.exceptions.NoSuchKey(None)
        return self.objects[key]

    def delete_object(self, key: str) -> None:
        if key in self.delete_errors:
            raise oss2.exceptions.InternalError(None)
        self.objects.pop(key, None)


def configured_values() -> dict[str, str]:
    return {
        "ACCESS_KEY_ID": "key-id",
        "ACCESS_KEY_SECRET": "key-secret",
        "ENDPOINT": "oss-cn-hangzhou.aliyuncs.com",
        "BUCKET": "example-bucket",
    }


def make_provider(values: dict[str, str], bucket: FakeBucket) -> OssStorageProvider:
    return OssStorageProvider(config=FakeConfig(values), bucket=bucket)


async def test_provider_reports_missing_configuration() -> None:
    provider = make_provider({}, FakeBucket())
    assert provider.configured is False
    with pytest.raises(StorageProviderError):
        await provider.presign_put_async("storages/a", "text/plain", 600)


async def test_provider_signs_put_and_get() -> None:
    provider = make_provider(configured_values(), FakeBucket())
    signed_put = await provider.presign_put_async("storages/a", "text/plain", 600)
    assert signed_put.method == "PUT"
    assert signed_put.headers == {"Content-Type": "text/plain"}
    assert "storages/a" in signed_put.url

    signed_get = await provider.presign_get_async(
        "storages/a",
        300,
        content_type="text/plain",
        filename="demo.txt",
    )
    assert signed_get.method == "GET"
    assert "response-content-disposition" in signed_get.url


async def test_provider_maps_head_and_delete() -> None:
    bucket = FakeBucket()
    bucket.objects["storages/a"] = {
        "content_length": 12,
        "headers": {"Content-Type": "text/plain", "ETag": '"etag"'},
    }
    provider = make_provider(configured_values(), bucket)
    head = await provider.head_async("storages/a")
    assert head is not None
    assert (head.byte_size, head.content_type, head.etag) == (12, "text/plain", '"etag"')

    assert await provider.head_async("storages/missing") is None
    await provider.delete_async("storages/a")
    assert "storages/a" not in bucket.objects
    await provider.delete_async("storages/missing")

    bucket.objects["storages/fail"] = {}
    bucket.delete_errors.append("storages/fail")
    with pytest.raises(StorageProviderError):
        await provider.delete_async("storages/fail")
```

Run:

```bash
uv run pytest tests/test_storages_oss.py -v
```

Expected: collection fails because the provider modules do not exist.

- [ ] **Step 3: Implement provider contracts**

Create `src/fivccliche/services/interfaces/storages.py`:

```python
"""Storage provider contracts for signed object access and metadata."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime

from fivcglue import IComponent


@dataclass(frozen=True)
class StorageSignedPut:
    method: str
    url: str
    headers: dict[str, str]
    expires_at: datetime


@dataclass(frozen=True)
class StorageSignedGet:
    method: str
    url: str
    expires_at: datetime


@dataclass(frozen=True)
class StorageHeadObject:
    byte_size: int
    content_type: str
    etag: str | None


class StorageProviderError(RuntimeError):
    """A storage backend rejected or failed an operation."""


class IStorageProvider(IComponent):
    """Signed URL and object metadata operations used by the storages module."""

    @property
    @abstractmethod
    def configured(self) -> bool:
        """Whether required backend configuration is present."""

    @abstractmethod
    async def presign_put_async(
        self,
        object_key: str,
        content_type: str,
        expires_in: int,
    ) -> StorageSignedPut:
        """Create a short-lived client-direct PUT URL."""

    @abstractmethod
    async def presign_get_async(
        self,
        object_key: str,
        expires_in: int,
        *,
        content_type: str,
        filename: str,
    ) -> StorageSignedGet:
        """Create a short-lived download URL with response metadata."""

    @abstractmethod
    async def head_async(self, object_key: str) -> StorageHeadObject | None:
        """Return object metadata, or None when the key is missing."""

    @abstractmethod
    async def delete_async(self, object_key: str) -> None:
        """Delete a key; a backend missing-key response is success."""
```

- [ ] **Step 4: Implement the OSS adapter**

Create `src/fivccliche/services/implements/storages_oss.py`:

```python
"""Aliyun OSS implementation of IStorageProvider."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

import oss2
from fivcglue import IComponentSite
from fivcglue import query_component
from fivcglue.interfaces import configs

from fivccliche.services.interfaces.storages import (
    IStorageProvider,
    StorageHeadObject,
    StorageProviderError,
    StorageSignedGet,
    StorageSignedPut,
)
from fivccliche.utils.types import to_optional_int, to_string

_DEFAULT_UPLOAD_EXPIRES = 600
_DEFAULT_DOWNLOAD_EXPIRES = 300


class OssStorageProvider(IStorageProvider):
    """Adapter over the official synchronous oss2 SDK."""

    def __init__(
        self,
        component_site: IComponentSite | None = None,
        *,
        config: configs.IConfig | None = None,
        bucket: Any | None = None,
        **_kwargs: Any,
    ) -> None:
        if config is None and component_site is not None:
            config = query_component(component_site, configs.IConfig)
        self._config = config
        self._injected_bucket = bucket
        self._bucket: Any | None = bucket
        values = self._read_config()
        self._access_key_id = values["ACCESS_KEY_ID"]
        self._access_key_secret = values["ACCESS_KEY_SECRET"]
        self._endpoint = values["ENDPOINT"]
        self._bucket_name = values["BUCKET"]
        self._upload_expires = values["UPLOAD_URL_EXPIRES_SECONDS"]
        self._download_expires = values["DOWNLOAD_URL_EXPIRES_SECONDS"]
        required = (
            self._access_key_id,
            self._access_key_secret,
            self._endpoint,
            self._bucket_name,
        )
        self._configured = all(required)

    def _read_config(self) -> dict[str, Any]:
        if self._config is None:
            return {
                "ACCESS_KEY_ID": "",
                "ACCESS_KEY_SECRET": "",
                "ENDPOINT": "",
                "BUCKET": "",
                "UPLOAD_URL_EXPIRES_SECONDS": _DEFAULT_UPLOAD_EXPIRES,
                "DOWNLOAD_URL_EXPIRES_SECONDS": _DEFAULT_DOWNLOAD_EXPIRES,
            }
        session = self._config.get_session("OSS")

        def text(key: str) -> str:
            return to_string(session.get_value(key), "")

        upload_expires = to_optional_int(session.get_value("UPLOAD_URL_EXPIRES_SECONDS"))
        download_expires = to_optional_int(session.get_value("DOWNLOAD_URL_EXPIRES_SECONDS"))
        return {
            "ACCESS_KEY_ID": text("ACCESS_KEY_ID"),
            "ACCESS_KEY_SECRET": text("ACCESS_KEY_SECRET"),
            "ENDPOINT": text("ENDPOINT"),
            "BUCKET": text("BUCKET"),
            "UPLOAD_URL_EXPIRES_SECONDS": (
                upload_expires
                if upload_expires is not None and upload_expires > 0
                else _DEFAULT_UPLOAD_EXPIRES
            ),
            "DOWNLOAD_URL_EXPIRES_SECONDS": (
                download_expires
                if download_expires is not None and download_expires > 0
                else _DEFAULT_DOWNLOAD_EXPIRES
            ),
        }

    @property
    def configured(self) -> bool:
        return bool(self._configured)

    def _get_bucket(self) -> Any:
        if not self.configured:
            raise StorageProviderError("OSS storage is not configured")
        if self._bucket is None:
            auth = oss2.Auth(self._access_key_id, self._access_key_secret)
            self._bucket = oss2.Bucket(auth, self._endpoint, self._bucket_name)
        return self._bucket

    def _sign_put(self, object_key: str, content_type: str, expires_in: int) -> StorageSignedPut:
        headers = {"Content-Type": content_type}
        url = self._get_bucket().sign_url(
            "PUT",
            object_key,
            expires_in,
            headers=headers,
            slash_safe=True,
        )
        return StorageSignedPut(
            method="PUT",
            url=url,
            headers=headers,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=expires_in),
        )

    def _sign_get(
        self,
        object_key: str,
        expires_in: int,
        content_type: str,
        filename: str,
    ) -> StorageSignedGet:
        quoted = quote(filename, safe="")
        params = {
            "response-content-type": content_type,
            "response-content-disposition": f"attachment; filename*=UTF-8''{quoted}",
        }
        url = self._get_bucket().sign_url(
            "GET",
            object_key,
            expires_in,
            params=params,
            slash_safe=True,
        )
        return StorageSignedGet(
            method="GET",
            url=url,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=expires_in),
        )

    def _head(self, object_key: str) -> StorageHeadObject | None:
        try:
            result = self._get_bucket().head_object(object_key)
        except oss2.exceptions.NoSuchKey:
            return None
        except oss2.exceptions.OssError as exc:
            raise StorageProviderError("OSS HEAD failed for storage object") from exc
        return StorageHeadObject(
            byte_size=int(result.content_length),
            content_type=str(result.headers.get("Content-Type", "")),
            etag=result.headers.get("ETag"),
        )

    def _delete(self, object_key: str) -> None:
        try:
            self._get_bucket().delete_object(object_key)
        except oss2.exceptions.NoSuchKey:
            return
        except oss2.exceptions.OssError as exc:
            raise StorageProviderError("OSS DELETE failed for storage object") from exc

    async def presign_put_async(
        self,
        object_key: str,
        content_type: str,
        expires_in: int,
    ) -> StorageSignedPut:
        return await asyncio.to_thread(self._sign_put, object_key, content_type, expires_in)

    async def presign_get_async(
        self,
        object_key: str,
        expires_in: int,
        *,
        content_type: str,
        filename: str,
    ) -> StorageSignedGet:
        return await asyncio.to_thread(
            self._sign_get,
            object_key,
            expires_in,
            content_type,
            filename,
        )

    async def head_async(self, object_key: str) -> StorageHeadObject | None:
        return await asyncio.to_thread(self._head, object_key)

    async def delete_async(self, object_key: str) -> None:
        await asyncio.to_thread(self._delete, object_key)
```

Add `oss2` and `oss2.*` to the MyPy override module list in `pyproject.toml`:

```toml
"oss2",
"oss2.*",
```

- [ ] **Step 5: Register dependency and component**

In `src/fivccliche/utils/deps.py`, import the interface:

```python
from fivccliche.services.interfaces.storages import IStorageProvider
```

Add near the other lazy providers:

```python
default_storage_provider: LazyValue[IStorageProvider | None] = LazyValue(
    lambda: query_component(cast(IComponentSite, service_site), IStorageProvider)
)
```

Add:

```python
async def get_storage_provider_async() -> IStorageProvider | None:
    """Return the mounted storage provider, if one is registered."""
    return default_storage_provider()
```

In `src/fivccliche/settings/services.yml`, add before the modules section:

```yaml
# storage ---------------------------------------------
- entries:
    - interface: fivccliche.services.interfaces.storages.IStorageProvider

  class: fivccliche.services.implements.storages_oss.OssStorageProvider
```

The `storages` module registration is added in Task 3, after its router exists.

- [ ] **Step 6: Run focused verification**

Run:

```bash
uv run pytest tests/test_storages_oss.py -v
uv run black src/fivccliche/services/interfaces/storages.py src/fivccliche/services/implements/storages_oss.py src/fivccliche/utils/deps.py tests/test_storages_oss.py
uv run ruff check src/fivccliche/services/interfaces/storages.py src/fivccliche/services/implements/storages_oss.py src/fivccliche/utils/deps.py tests/test_storages_oss.py
uv run mypy src/fivccliche/services/interfaces/storages.py src/fivccliche/services/implements/storages_oss.py src/fivccliche/utils/deps.py
```

Expected: tests pass and static checks exit 0.

- [ ] **Step 7: Commit**

Run:

```bash
git add pyproject.toml uv.lock src/fivccliche/services/interfaces/storages.py src/fivccliche/services/implements/storages_oss.py src/fivccliche/utils/deps.py src/fivccliche/settings/services.yml tests/test_storages_oss.py
git commit -m "feat: add Aliyun OSS storage provider"
```

### Task 3: Upload Creation And Completion

**Files:**
- Create: `src/fivccliche/modules/storages/utils.py`
- Create: `src/fivccliche/modules/storages/routers.py`
- Create: `src/fivccliche/modules/storages/services.py`
- Modify: `src/fivccliche/settings/services.yml`
- Test: `tests/test_storages_api.py`

- [ ] **Step 1: Write failing upload API tests**

Create `tests/test_storages_api.py` with the shared fixtures and initial tests:

```python
"""API tests for user-owned OSS storage files."""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fivccliche.services.interfaces.storages import (
    IStorageProvider,
    StorageHeadObject,
    StorageProviderError,
    StorageSignedGet,
    StorageSignedPut,
)
from fivccliche.utils.deps import get_storage_provider_async
from tests.conftest import make_api_client


class FakeStorageProvider(IStorageProvider):
    def __init__(self) -> None:
        self.objects: dict[str, StorageHeadObject] = {}
        self.deleted: list[str] = []
        self.delete_failures: set[str] = set()

    @property
    def configured(self) -> bool:
        return True

    async def presign_put_async(
        self, object_key: str, content_type: str, expires_in: int
    ) -> StorageSignedPut:
        return StorageSignedPut(
            method="PUT",
            url=f"https://oss.test/{object_key}",
            headers={"Content-Type": content_type},
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=expires_in),
        )

    async def presign_get_async(
        self,
        object_key: str,
        expires_in: int,
        *,
        content_type: str,
        filename: str,
    ) -> StorageSignedGet:
        return StorageSignedGet(
            method="GET",
            url=f"https://oss.test/{object_key}?download={filename}",
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=expires_in),
        )

    async def head_async(self, object_key: str) -> StorageHeadObject | None:
        return self.objects.get(object_key)

    async def delete_async(self, object_key: str) -> None:
        if object_key in self.delete_failures:
            raise StorageProviderError("delete failed")
        self.deleted.append(object_key)
        self.objects.pop(object_key, None)


@pytest.fixture
def storage_provider() -> FakeStorageProvider:
    return FakeStorageProvider()


@pytest.fixture
def client(storage_provider: FakeStorageProvider):
    test_client = make_api_client(["users", "storages"])
    test_client.app.dependency_overrides[get_storage_provider_async] = lambda: storage_provider
    yield test_client


@pytest.fixture
def auth_token(client: TestClient) -> str:
    response = client.post("/users/login", json={"username": "admin", "password": "admin123"})
    return response.json()["access_token"]


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def create_upload(
    client: TestClient,
    token: str,
    *,
    filename: str = "demo.txt",
    content_type: str = "text/plain",
    byte_size: int = 12,
    permission: str = "PRIVATE",
) -> Any:
    return client.post(
        "/files/uploads/",
        headers=auth_headers(token),
        json={
            "filename": filename,
            "content_type": content_type,
            "byte_size": byte_size,
            "permission": permission,
        },
    )


def test_create_upload_returns_pending_metadata(
    client: TestClient, auth_token: str, storage_provider: FakeStorageProvider
) -> None:
    response = create_upload(client, auth_token)
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "PENDING"
    assert data["permission"] == "PRIVATE"
    assert data["upload_method"] == "PUT"
    assert data["upload_headers"] == {"Content-Type": "text/plain"}
    assert data["object_key"].startswith("storages/")


def test_create_upload_rejects_disallowed_type(client: TestClient, auth_token: str) -> None:
    response = create_upload(
        client,
        auth_token,
        filename="demo.exe",
        content_type="application/octet-stream",
    )
    assert response.status_code == 422


def test_complete_upload_verifies_object(
    client: TestClient, auth_token: str, storage_provider: FakeStorageProvider
) -> None:
    created = create_upload(client, auth_token).json()
    storage_provider.objects[created["object_key"]] = StorageHeadObject(
        byte_size=12, content_type="text/plain", etag='"etag"'
    )
    response = client.post(f"/files/{created['uuid']}/complete/", headers=auth_headers(auth_token))
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "READY"
    assert data["etag"] == '"etag"'

    repeat = client.post(f"/files/{created['uuid']}/complete/", headers=auth_headers(auth_token))
    assert repeat.status_code == 200


def test_complete_upload_rejects_mismatch_and_reclaims_object(
    client: TestClient, auth_token: str, storage_provider: FakeStorageProvider
) -> None:
    created = create_upload(client, auth_token, byte_size=12).json()
    storage_provider.objects[created["object_key"]] = StorageHeadObject(
        byte_size=13, content_type="text/plain", etag=None
    )
    response = client.post(f"/files/{created['uuid']}/complete/", headers=auth_headers(auth_token))
    assert response.status_code == 400
    assert created["object_key"] in storage_provider.deleted
    missing = client.get(f"/files/{created['uuid']}/", headers=auth_headers(auth_token))
    assert missing.status_code == 404


def test_complete_upload_rejects_missing_object(
    client: TestClient, auth_token: str, storage_provider: FakeStorageProvider
) -> None:
    created = create_upload(client, auth_token).json()
    response = client.post(f"/files/{created['uuid']}/complete/", headers=auth_headers(auth_token))
    assert response.status_code == 400
    assert storage_provider.deleted == []
    assert client.get(f"/files/{created['uuid']}/", headers=auth_headers(auth_token)).status_code == 404
```

Run:

```bash
uv run pytest tests/test_storages_api.py -v
```

Expected: collection or app creation fails because the `storages` module is not registered.

- [ ] **Step 2: Implement shared state utilities**

Create `src/fivccliche/modules/storages/utils.py`:

```python
"""Shared state transitions and cleanup queries for storage files."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

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
    statement = update(StorageFile).where(StorageFile.uuid == file_uuid)
    if user_uuid is not None:
        statement = statement.where(StorageFile.user_uuid == user_uuid)
    statement = statement.where(StorageFile.status.in_(tuple(current_statuses)))
    assigned: dict[str, object] = {"status": target, "updated_at": datetime.now(timezone.utc)}
    if values:
        assigned.update(values)
    result = await session.execute(statement.values(**assigned))
    return bool(result.rowcount)


async def list_stale_pending_async(
    session: AsyncSession,
    *,
    created_before: datetime,
    limit: int,
) -> list[StorageFile]:
    statement = (
        select(StorageFile)
        .where(StorageFile.status == StorageStatus.PENDING)
        .where(StorageFile.created_at < created_before)
        .order_by(StorageFile.created_at.asc())
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
        .where(StorageFile.status == StorageStatus.DELETING)
        .order_by(StorageFile.updated_at.asc())
        .limit(limit)
    )
    return list((await session.execute(statement)).scalars().all())


async def delete_storage_file_row_async(
    session: AsyncSession,
    *,
    file_uuid: str,
) -> bool:
    result = await session.execute(
        delete(StorageFile).where(StorageFile.uuid == file_uuid)
    )
    return bool(result.rowcount)
```

- [ ] **Step 3: Implement upload and complete routes**

Create `src/fivccliche/modules/storages/routers.py`:

```python
"""HTTP routes for authenticated user-owned storage files."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from fivccliche.services.interfaces.storages import (
    IStorageProvider,
    StorageProviderError,
)
from fivccliche.utils.deps import (
    IUser,
    get_authenticated_user_async,
    get_db_session_async,
    get_storage_provider_async,
)

from . import models, schemas, utils
from .filters import StorageFileOwnedFilterSet

logger = logging.getLogger(__name__)
router_files = APIRouter(tags=["files"], prefix="/files")


async def get_required_storage_provider_async(
    provider: IStorageProvider | None = Depends(get_storage_provider_async),
) -> IStorageProvider:
    if provider is None or not provider.configured:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Storage provider is unavailable")
    return provider


@router_files.post(
    "/uploads/",
    response_model=schemas.StorageFileUploadSchema,
    status_code=status.HTTP_201_CREATED,
)
async def create_upload_async(
    body: schemas.StorageFileCreateSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
    provider: IStorageProvider = Depends(get_required_storage_provider_async),
) -> schemas.StorageFileUploadSchema:
    file_uuid = str(uuid4())
    row = models.StorageFile(
        uuid=file_uuid,
        user_uuid=user.uuid,
        object_key=f"storages/{file_uuid}",
        filename=body.filename,
        content_type=body.content_type,
        byte_size=body.byte_size,
        permission=body.permission,
        status=schemas.StorageStatus.PENDING,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    try:
        signed = await provider.presign_put_async(
            row.object_key,
            row.content_type,
            600,
        )
    except StorageProviderError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not sign storage upload") from exc
    metadata = row.to_schema().model_dump(mode="json")
    return schemas.StorageFileUploadSchema(
        **metadata,
        upload_method=signed.method,
        upload_url=signed.url,
        upload_headers=signed.headers,
        upload_expires_at=signed.expires_at,
    )


@router_files.post(
    "/{file_uuid}/complete/",
    response_model=schemas.StorageFileSchema,
)
async def complete_upload_async(
    file_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
    provider: IStorageProvider = Depends(get_required_storage_provider_async),
) -> schemas.StorageFileSchema:
    owned_filters = StorageFileOwnedFilterSet(user.uuid)
    owned_statement = owned_filters.filter(
        select(models.StorageFile).where(models.StorageFile.uuid == file_uuid)
    )
    row = (await session.execute(owned_statement)).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found")
    if row.status == schemas.StorageStatus.READY:
        return row.to_schema()
    if row.status == schemas.StorageStatus.DELETING:
        raise HTTPException(status.HTTP_410_GONE, "File is being deleted")

    try:
        head = await provider.head_async(row.object_key)
    except StorageProviderError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not verify storage object") from exc

    valid = (
        head is not None
        and head.byte_size == row.byte_size
        and head.content_type == row.content_type
    )
    if not valid:
        claimed = await utils.transition_storage_file_status_async(
            session,
            file_uuid=file_uuid,
            user_uuid=user.uuid,
            target=schemas.StorageStatus.DELETING,
            current_statuses=(schemas.StorageStatus.PENDING,),
        )
        await session.commit()
        if not claimed:
            raise HTTPException(status.HTTP_410_GONE, "File is being deleted")
        if head is not None:
            try:
                await provider.delete_async(row.object_key)
            except StorageProviderError as exc:
                logger.exception("Could not delete rejected storage object %s", file_uuid)
                raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not remove rejected file") from exc
        await utils.delete_storage_file_row_async(session, file_uuid=file_uuid)
        await session.commit()
        detail = "Storage object was not uploaded" if head is None else "Uploaded file does not match its declaration"
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail)

    now = datetime.now(timezone.utc)
    claimed = await utils.transition_storage_file_status_async(
        session,
        file_uuid=file_uuid,
        user_uuid=user.uuid,
        target=schemas.StorageStatus.READY,
        current_statuses=(schemas.StorageStatus.PENDING,),
        values={"etag": head.etag, "completed_at": now},
    )
    await session.commit()
    if not claimed:
        await session.refresh(row)
        if row.status == schemas.StorageStatus.READY:
            return row.to_schema()
        raise HTTPException(status.HTTP_410_GONE, "File is being deleted")
    await session.refresh(row)
    return row.to_schema()
```

The hard-coded upload expiry is replaced in Task 4 by a provider-owned expires setting. For now, preserve the spec default of 600 seconds.

- [ ] **Step 4: Register the module**

Create `src/fivccliche/modules/storages/services.py`:

```python
"""Storages module registration."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fivcglue import IComponentSite

from fivccliche.services.interfaces.modules import IModule, IModuleJob

from . import routers

logger = logging.getLogger(__name__)


class ModuleImpl(IModule):
    """User-owned storage file module."""

    def __init__(self, component_site: IComponentSite, **kwargs):
        self._component_site = component_site
        self._jobs: list[IModuleJob] = []
        logger.info("storages module initialized")

    @property
    def name(self) -> str:
        return "storages"

    @property
    def description(self) -> str:
        return "User file storage backed by Aliyun OSS."

    def list_jobs(self) -> list[IModuleJob]:
        return list(self._jobs)

    def get_job(self, job_name: str) -> IModuleJob | None:
        for job in self._jobs:
            if job.name == job_name:
                return job
        return None

    def mount(self, app: FastAPI, **kwargs) -> None:
        logger.info("storages module mounted")
        app.include_router(routers.router_files, **kwargs)
```

In `src/fivccliche/settings/services.yml`, add:

```yaml
- entries:
    - interface: fivccliche.services.interfaces.modules.IModule
      name: storages

  class: fivccliche.modules.storages.ModuleImpl
```

and append `'storages'` to `IModuleSite.modules` after `'agent_memories'`.

- [ ] **Step 5: Run focused API tests**

Run:

```bash
uv run pytest tests/test_storages_api.py -v
```

Expected: all current tests pass.

- [ ] **Step 6: Run focused static checks**

Run:

```bash
uv run black src/fivccliche/modules/storages tests/test_storages_api.py
uv run ruff check src/fivccliche/modules/storages tests/test_storages_api.py
uv run mypy src/fivccliche/modules/storages
```

Expected: checks exit 0.

- [ ] **Step 7: Commit**

```bash
git add src/fivccliche/modules/storages src/fivccliche/settings/services.yml tests/test_storages_api.py
git commit -m "feat: add storage upload workflow"
```

### Task 4: Listing, Ownership, Private URL, And Public Access

**Files:**
- Modify: `src/fivccliche/modules/storages/routers.py`
- Modify: `src/fivccliche/services/interfaces/storages.py`
- Modify: `src/fivccliche/services/implements/storages_oss.py`
- Test: `tests/test_storages_api.py`

- [ ] **Step 1: Add failing access tests**

Append these tests to `tests/test_storages_api.py`:

```python
def create_ready_file(
    client: TestClient,
    token: str,
    storage_provider: FakeStorageProvider,
    *,
    permission: str = "PRIVATE",
) -> dict[str, Any]:
    created = create_upload(client, token, permission=permission).json()
    storage_provider.objects[created["object_key"]] = StorageHeadObject(
        byte_size=12, content_type="text/plain", etag='"etag"'
    )
    completed = client.post(f"/files/{created['uuid']}/complete/", headers=auth_headers(token))
    assert completed.status_code == 200
    return completed.json()


def test_list_defaults_to_ready_files(
    client: TestClient, auth_token: str, storage_provider: FakeStorageProvider
) -> None:
    pending = create_upload(client, auth_token).json()
    ready = create_ready_file(client, auth_token, storage_provider)
    response = client.get("/files/", headers=auth_headers(auth_token))
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert [item["uuid"] for item in data["results"]] == [ready["uuid"]]

    pending_list = client.get(
        "/files/",
        headers=auth_headers(auth_token),
        params={"status": "PENDING"},
    )
    assert [item["uuid"] for item in pending_list.json()["results"]] == [pending["uuid"]]


def test_private_url_requires_owner_and_ready(
    client: TestClient, auth_token: str, storage_provider: FakeStorageProvider
) -> None:
    pending = create_upload(client, auth_token).json()
    conflict = client.get(f"/files/{pending['uuid']}/url/", headers=auth_headers(auth_token))
    assert conflict.status_code == 409

    ready = create_ready_file(client, auth_token, storage_provider)
    response = client.get(f"/files/{ready['uuid']}/url/", headers=auth_headers(auth_token))
    assert response.status_code == 200
    assert response.json()["url"].startswith("https://oss.test/")


def create_second_user_token(client: TestClient) -> str:
    admin_token = client.post(
        "/users/login",
        json={"username": "admin", "password": "admin123"},
    ).json()["access_token"]
    client.post(
        "/users/",
        headers=auth_headers(admin_token),
        json={
            "username": "otheruser",
            "email": "other@example.com",
            "password": "password123",
        },
    )
    return client.post(
        "/users/login",
        json={"username": "otheruser", "password": "password123"},
    ).json()["access_token"]


def test_private_url_rejects_another_user(
    client: TestClient, auth_token: str, storage_provider: FakeStorageProvider
) -> None:
    ready = create_ready_file(client, auth_token, storage_provider)
    other_token = create_second_user_token(client)
    response = client.get(f"/files/{ready['uuid']}/url/", headers=auth_headers(other_token))
    assert response.status_code == 404


def test_public_route_requires_public_permission_and_ready(
    client: TestClient, auth_token: str, storage_provider: FakeStorageProvider
) -> None:
    private = create_ready_file(client, auth_token, storage_provider)
    assert client.get(f"/files/public/{private['uuid']}/").status_code == 404

    public = create_ready_file(
        client,
        auth_token,
        storage_provider,
        permission="PUBLIC_READ",
    )
    response = client.get(f"/files/public/{public['uuid']}/")
    assert response.status_code == 307
    assert response.headers["Location"].startswith("https://oss.test/")
    assert response.headers["Cache-Control"] == "no-store"


def test_permission_has_no_patch_route(
    client: TestClient, auth_token: str, storage_provider: FakeStorageProvider
) -> None:
    ready = create_ready_file(client, auth_token, storage_provider)
    response = client.patch(
        f"/files/{ready['uuid']}/",
        headers=auth_headers(auth_token),
        json={"permission": "PUBLIC_READ"},
    )
    assert response.status_code == 405
```

Run:

```bash
uv run pytest tests/test_storages_api.py -v
```

Expected: the new access tests fail with 404 because routes do not exist; existing upload tests still pass.

- [ ] **Step 2: Expose configured URL lifetimes**

Add this method to `IStorageProvider`:

```python
    @abstractmethod
    def upload_expires_in(self) -> int:
        """Configured presigned PUT lifetime in seconds."""

    @abstractmethod
    def download_expires_in(self) -> int:
        """Configured presigned GET lifetime in seconds."""
```

Implement both in `OssStorageProvider` and in test fakes:

```python
    def upload_expires_in(self) -> int:
        return self._upload_expires

    def download_expires_in(self) -> int:
        return self._download_expires
```

In `FakeStorageProvider`, return `600` and `300`.

Replace the hard-coded `600` in `create_upload_async` with:

```python
provider.upload_expires_in()
```

- [ ] **Step 3: Implement list, get, URL, and public routes**

Append these handlers to `routers.py`. The public route must be declared before the parameterized single-file GET route in the final file:

```python
@router_files.get(
    "/public/{file_uuid}/",
    summary="Redirect to an internet-public storage object.",
)
async def get_public_file_async(
    file_uuid: str,
    session: AsyncSession = Depends(get_db_session_async),
    provider: IStorageProvider = Depends(get_required_storage_provider_async),
) -> responses.RedirectResponse:
    statement = select(models.StorageFile).where(models.StorageFile.uuid == file_uuid)
    row = (await session.execute(statement)).scalar_one_or_none()
    if (
        row is None
        or row.permission != schemas.StoragePermission.PUBLIC_READ
        or row.status != schemas.StorageStatus.READY
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found")
    try:
        signed = await provider.presign_get_async(
            row.object_key,
            provider.download_expires_in(),
            content_type=row.content_type,
            filename=row.filename,
        )
    except StorageProviderError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not sign storage download") from exc
    return responses.RedirectResponse(
        signed.url,
        status_code=status.HTTP_307_TEMPORARY_REDIRECT,
        headers={"Cache-Control": "no-store"},
    )
```

Use this final router import block after Task 4:

```python
import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    responses,
    status,
)
from fivccliche.services.interfaces.storages import (
    IStorageProvider,
    StorageProviderError,
)
from fivccliche.utils.deps import (
    IUser,
    get_authenticated_user_async,
    get_db_session_async,
    get_storage_provider_async,
)
from fivccliche.utils.schemas import PaginatedResponse
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from . import filters as filters_module
from . import models, schemas, utils
```

Then add:

```python
@router_files.get(
    "/",
    response_model=PaginatedResponse[schemas.StorageFileSchema],
)
async def list_files_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status_filter: schemas.StorageStatus | None = Query(
        schemas.StorageStatus.READY,
        alias="status",
        description="PENDING or READY; DELETING is hidden",
    ),
    permission: schemas.StoragePermission | None = Query(None),
    content_type: str | None = Query(None, min_length=3, max_length=255),
    created_at_from: datetime | None = Query(None),
    created_at_to: datetime | None = Query(None),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse[schemas.StorageFileSchema]:
    if status_filter == schemas.StorageStatus.DELETING:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "DELETING files are not listable")
    filters = filters_module.StorageFileFilterSet(user.uuid)
    filters.parse(
        status=status_filter,
        permission=permission,
        content_type=content_type,
        created_at_from=created_at_from,
        created_at_to=created_at_to,
    )
    statement = filters.filter(select(models.StorageFile))
    rows = (
        (await session.execute(statement.order_by(models.StorageFile.created_at.desc()).offset(skip).limit(limit)))
        .scalars()
        .all()
    )
    count_statement = filters.filter(select(func.count()).select_from(models.StorageFile))
    total = (await session.execute(count_statement)).scalar_one()
    return PaginatedResponse[schemas.StorageFileSchema](
        total=total,
        results=[row.to_schema() for row in rows],
    )
```

Add the owned metadata, private URL, and delete handlers:

```python
@router_files.get(
    "/{file_uuid}/",
    response_model=schemas.StorageFileSchema,
)
async def get_file_async(
    file_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.StorageFileSchema:
    owned_filters = filters_module.StorageFileOwnedFilterSet(user.uuid)
    owned_statement = owned_filters.filter(
        select(models.StorageFile).where(models.StorageFile.uuid == file_uuid)
    )
    row = (await session.execute(owned_statement)).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found")
    return row.to_schema()


@router_files.get(
    "/{file_uuid}/url/",
    response_model=schemas.StorageFileUrlSchema,
)
async def get_file_url_async(
    file_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
    provider: IStorageProvider = Depends(get_required_storage_provider_async),
) -> schemas.StorageFileUrlSchema:
    owned_filters = filters_module.StorageFileOwnedFilterSet(user.uuid)
    owned_statement = owned_filters.filter(
        select(models.StorageFile).where(models.StorageFile.uuid == file_uuid)
    )
    row = (await session.execute(owned_statement)).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found")
    if row.status != schemas.StorageStatus.READY:
        raise HTTPException(status.HTTP_409_CONFLICT, "File is not ready")
    try:
        signed = await provider.presign_get_async(
            row.object_key,
            provider.download_expires_in(),
            content_type=row.content_type,
            filename=row.filename,
        )
    except StorageProviderError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not sign storage download") from exc
    return schemas.StorageFileUrlSchema(url=signed.url, expires_at=signed.expires_at)
```

Task 5 adds the delete handler with cleanup semantics.

- [ ] **Step 4: Run access tests**

```bash
uv run pytest tests/test_storages_api.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run static checks**

```bash
uv run black src/fivccliche/modules/storages src/fivccliche/services/interfaces/storages.py src/fivccliche/services/implements/storages_oss.py tests/test_storages_api.py
uv run ruff check src/fivccliche/modules/storages src/fivccliche/services/interfaces/storages.py src/fivccliche/services/implements/storages_oss.py tests/test_storages_api.py
uv run mypy src/fivccliche/modules/storages src/fivccliche/services/interfaces/storages.py src/fivccliche/services/implements/storages_oss.py
```

Expected: checks exit 0.

- [ ] **Step 6: Commit**

```bash
git add src/fivccliche/modules/storages src/fivccliche/services/interfaces/storages.py src/fivccliche/services/implements/storages_oss.py tests/test_storages_api.py
git commit -m "feat: control storage file access"
```

### Task 5: Deletion And Pending Cleanup

**Files:**
- Modify: `src/fivccliche/modules/storages/routers.py`
- Create: `src/fivccliche/modules/storages/jobs.py`
- Modify: `src/fivccliche/modules/storages/services.py`
- Test: `tests/test_storages_cleanup.py`
- Test: `tests/test_storages_api.py`

- [ ] **Step 1: Write failing cleanup tests**

Create `tests/test_storages_cleanup.py`:

```python
"""Tests for storage deletion and pending-object cleanup."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from fivccliche.modules.storages import jobs, models, utils
from fivccliche.modules.storages.schemas import StoragePermission, StorageStatus
from fivccliche.services.interfaces.storages import StorageHeadObject
from tests.test_storages_api import FakeStorageProvider


async def make_file(
    session: AsyncSession,
    *,
    status: StorageStatus,
    created_at: datetime,
) -> models.StorageFile:
    row = models.StorageFile(
        user_uuid="owner",
        object_key=f"storages/{status}-{created_at.timestamp()}",
        filename="demo.txt",
        content_type="text/plain",
        byte_size=12,
        permission=StoragePermission.PRIVATE,
        status=status,
        created_at=created_at,
        updated_at=created_at,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


class FakeCleanupConfigSession:
    def __init__(self, values: dict[str, int]):
        self.values = values

    def get_value(self, key: str) -> int | None:
        return self.values.get(key)


class FakeCleanupConfig:
    def __init__(self, values: dict[str, int]):
        self.values = values

    def get_session(self, name: str) -> FakeCleanupConfigSession:
        return FakeCleanupConfigSession(self.values if name == "STORAGE_CLEANUP" else {})


def make_job(provider: FakeStorageProvider, session: AsyncSession) -> jobs.StorageCleanupJob:
    job = jobs.StorageCleanupJob.__new__(jobs.StorageCleanupJob)
    job._setting = jobs.StorageCleanupSetting(FakeCleanupConfig({}))
    job._provider = provider
    job._session_factory = lambda: _session_context(session)
    return job


def test_cleanup_setting_uses_safe_defaults_and_custom_values() -> None:
    defaults = jobs.StorageCleanupSetting(FakeCleanupConfig({}))
    assert (defaults.interval_minutes, defaults.pending_ttl_hours, defaults.batch_size) == (
        60,
        24,
        100,
    )

    custom = jobs.StorageCleanupSetting(
        FakeCleanupConfig(
            {"INTERVAL_MINUTES": 15, "PENDING_TTL_HOURS": 2, "BATCH_SIZE": 3}
        )
    )
    assert (custom.interval_minutes, custom.pending_ttl_hours, custom.batch_size) == (15, 2, 3)


@asynccontextmanager
async def _session_context(session: AsyncSession) -> AsyncIterator[AsyncSession]:
    yield session


async def test_cleanup_deletes_stale_pending_and_missing_objects(
    session: AsyncSession,
) -> None:
    provider = FakeStorageProvider()
    old = await make_file(
        session,
        status=StorageStatus.PENDING,
        created_at=datetime.now(timezone.utc) - timedelta(hours=25),
    )
    missing = await make_file(
        session,
        status=StorageStatus.PENDING,
        created_at=datetime.now(timezone.utc) - timedelta(hours=26),
    )
    fresh = await make_file(
        session,
        status=StorageStatus.PENDING,
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
    )
    provider.objects[old.object_key] = StorageHeadObject(
        byte_size=12,
        content_type="text/plain",
        etag='"etag"',
    )

    await make_job(provider, session).run_async()

    assert old.object_key in provider.deleted
    assert fresh.object_key not in provider.deleted
    remaining = await session.get(models.StorageFile, old.uuid)
    assert remaining is None
    assert await session.get(models.StorageFile, missing.uuid) is None
    assert await session.get(models.StorageFile, fresh.uuid) is not None
```

Add this second test:

```python
async def test_cleanup_retries_deleting_rows(session: AsyncSession) -> None:
    provider = FakeStorageProvider()
    row = await make_file(
        session,
        status=StorageStatus.DELETING,
        created_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    provider.objects[row.object_key] = StorageHeadObject(
        byte_size=12,
        content_type="text/plain",
        etag='"etag"',
    )
    provider.delete_failures.add(row.object_key)

    await make_job(provider, session).run_async()
    assert await session.get(models.StorageFile, row.uuid) is not None
    assert row.status == StorageStatus.DELETING

    provider.delete_failures.clear()
    await make_job(provider, session).run_async()
    assert await session.get(models.StorageFile, row.uuid) is None


async def test_cleanup_honors_batch_size(session: AsyncSession) -> None:
    provider = FakeStorageProvider()
    rows = [
        await make_file(
            session,
            status=StorageStatus.PENDING,
            created_at=datetime.now(timezone.utc) - timedelta(hours=25 + index),
        )
        for index in range(2)
    ]
    job = make_job(provider, session)
    job._setting = jobs.StorageCleanupSetting(FakeCleanupConfig({"BATCH_SIZE": 1}))
    await job.run_async()

    assert await session.get(models.StorageFile, rows[0].uuid) is None
    assert await session.get(models.StorageFile, rows[1].uuid) is not None


async def test_status_transition_is_conditional(session: AsyncSession) -> None:
    row = await make_file(
        session,
        status=StorageStatus.PENDING,
        created_at=datetime.now(timezone.utc),
    )
    first = await utils.transition_storage_file_status_async(
        session,
        file_uuid=row.uuid,
        target=StorageStatus.DELETING,
        current_statuses=(StorageStatus.PENDING,),
    )
    second = await utils.transition_storage_file_status_async(
        session,
        file_uuid=row.uuid,
        target=StorageStatus.DELETING,
        current_statuses=(StorageStatus.PENDING,),
    )
    await session.commit()

    assert first is True
    assert second is False
```

Run:

```bash
uv run pytest tests/test_storages_cleanup.py -v
```

Expected: collection fails because `jobs.StorageCleanupJob` does not exist.

- [ ] **Step 2: Add failing deletion API test**

Append to `tests/test_storages_api.py`:

```python
def test_delete_claims_file_and_removes_object(
    client: TestClient, auth_token: str, storage_provider: FakeStorageProvider
) -> None:
    ready = create_ready_file(client, auth_token, storage_provider)
    first = client.delete(f"/files/{ready['uuid']}/", headers=auth_headers(auth_token))
    assert first.status_code == 202
    assert ready["object_key"] in storage_provider.deleted

    second = client.delete(f"/files/{ready['uuid']}/", headers=auth_headers(auth_token))
    assert second.status_code == 404


def test_delete_keeps_row_for_cleanup_when_oss_fails(
    client: TestClient, auth_token: str, storage_provider: FakeStorageProvider
) -> None:
    ready = create_ready_file(client, auth_token, storage_provider)
    storage_provider.delete_failures.add(ready["object_key"])
    response = client.delete(f"/files/{ready['uuid']}/", headers=auth_headers(auth_token))
    assert response.status_code == 202
    repeat = client.delete(f"/files/{ready['uuid']}/", headers=auth_headers(auth_token))
    assert repeat.status_code == 202
    detail = client.get(f"/files/{ready['uuid']}/", headers=auth_headers(auth_token))
    assert detail.status_code == 200
    assert detail.json()["status"] == "DELETING"
```

Run:

```bash
uv run pytest tests/test_storages_api.py -v
```

Expected: deletion tests fail because the route does not exist.

- [ ] **Step 3: Implement deletion route**

Append to `routers.py`:

```python
@router_files.delete(
    "/{file_uuid}/",
    status_code=status.HTTP_202_ACCEPTED,
)
async def delete_file_async(
    file_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
    provider: IStorageProvider = Depends(get_required_storage_provider_async),
) -> dict[str, bool]:
    owned_filters = filters_module.StorageFileOwnedFilterSet(user.uuid)
    owned_statement = owned_filters.filter(
        select(models.StorageFile).where(models.StorageFile.uuid == file_uuid)
    )
    row = (await session.execute(owned_statement)).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found")
    if row.status == schemas.StorageStatus.DELETING:
        return {"success": True}

    claimed = await utils.transition_storage_file_status_async(
        session,
        file_uuid=file_uuid,
        user_uuid=user.uuid,
        target=schemas.StorageStatus.DELETING,
        current_statuses=(
            schemas.StorageStatus.PENDING,
            schemas.StorageStatus.READY,
        ),
    )
    await session.commit()
    if not claimed:
        raise HTTPException(status.HTTP_409_CONFLICT, "File state changed")

    try:
        await provider.delete_async(row.object_key)
    except StorageProviderError as exc:
        logger.exception("Storage object delete failed for file %s", file_uuid)
        return {"success": True}
    await utils.delete_storage_file_row_async(session, file_uuid=file_uuid)
    await session.commit()
    return {"success": True}
```

Change `FakeStorageProvider.delete_async` to raise `StorageProviderError` instead of `RuntimeError`, so production code catches only the provider contract exception.

- [ ] **Step 4: Implement cleanup job**

Create `src/fivccliche/modules/storages/jobs.py`:

```python
"""Scheduled cleanup for storage upload and deletion state."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import datetime, timedelta, timezone

from fivcglue import IComponentSite
from fivcglue import query_component
from fivcglue.interfaces import configs

from fivccliche.modules.storages import utils
from fivccliche.modules.storages.schemas import StorageStatus
from fivccliche.services.interfaces.modules import IModuleJob
from fivccliche.services.interfaces.storages import (
    IStorageProvider,
    StorageProviderError,
)
from fivccliche.utils.types import to_int
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_MINUTES = 60
_DEFAULT_PENDING_TTL_HOURS = 24
_DEFAULT_BATCH_SIZE = 100
SessionContextFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


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
        self._setting = StorageCleanupSetting(_component_config(component_site))
        self._provider = query_component(component_site, IStorageProvider)
        self._session_factory: SessionContextFactory = _default_session_factory

    @property
    def name(self) -> str:
        return "storages-cleanup"

    @property
    def config(self) -> dict | None:
        if self._provider is None or not self._provider.configured:
            return None
        return self._setting.job_config()

    async def run_async(self) -> None:
        if self._provider is None or not self._provider.configured:
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
                claimed_now = await utils.transition_storage_file_status_async(
                    session,
                    file_uuid=row.uuid,
                    target=StorageStatus.DELETING,
                    current_statuses=(StorageStatus.PENDING,),
                )
                await session.commit()
            if not claimed_now:
                continue

        async with self._session_factory() as session:
            deleting = await utils.list_deleting_async(
                session,
                limit=self._setting.batch_size,
            )
        for row in deleting:
            try:
                await self._provider.delete_async(row.object_key)
            except StorageProviderError:
                logger.exception("Could not delete storage object %s", row.uuid)
                continue
            async with self._session_factory() as session:
                await utils.delete_storage_file_row_async(session, file_uuid=row.uuid)
                await session.commit()
```

Add these module-private helpers:

```python
def _component_config(component_site: IComponentSite) -> configs.IConfig | None:
    return query_component(component_site, configs.IConfig)


@asynccontextmanager
async def _default_session_factory() -> AsyncIterator[AsyncSession]:
    from fivccliche.utils.deps import get_db_session_context_async

    async with get_db_session_context_async() as session:
        yield session
```

In `services.py`, import `jobs` and initialize:

```python
self._jobs = [jobs.StorageCleanupJob(component_site)]
```

The component site constructor argument must be retained in `ModuleImpl.__init__`.

- [ ] **Step 5: Run cleanup tests**

```bash
uv run pytest tests/test_storages_cleanup.py tests/test_storages_api.py -v
```

Expected: all storage tests pass.

- [ ] **Step 6: Run scheduler compatibility tests**

```bash
uv run pytest tests/test_cli_jobs.py tests/test_modules_scheduler.py -v
```

Expected: all tests pass and the scheduler accepts `storages-cleanup.config`.

- [ ] **Step 7: Commit**

```bash
git add src/fivccliche/modules/storages tests/test_storages_cleanup.py tests/test_storages_api.py
git commit -m "feat: reclaim storage uploads"
```

### Task 6: Repository Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/architecture.md`
- Create: `docs/storages.md`

- [ ] **Step 1: Create focused storage documentation**

Create `docs/storages.md` with this content:

````markdown
# Storages

The `storages` module stores metadata for user-owned files in PostgreSQL and
uploads file bytes directly to Aliyun OSS with presigned URLs.

Run `fivccliche migrate` after upgrading so the `storage_file` table is created
in an existing database.

## Configuration

```json
{
  "OSS": {
    "ACCESS_KEY_ID": "...",
    "ACCESS_KEY_SECRET": "...",
    "ENDPOINT": "oss-cn-hangzhou.aliyuncs.com",
    "BUCKET": "example-bucket",
    "UPLOAD_URL_EXPIRES_SECONDS": 600,
    "DOWNLOAD_URL_EXPIRES_SECONDS": 300
  },
  "STORAGE_CLEANUP": {
    "INTERVAL_MINUTES": 60,
    "PENDING_TTL_HOURS": 24,
    "BATCH_SIZE": 100
  }
}
```

The Bucket and every object remain private. `PUBLIC_READ` is an application
permission: the backend redirects known UUIDs only after the DB row is ready
and explicitly marked public.

## Upload flow

1. `POST /api/files/uploads/` validates filename, MIME type, size, and immutable
   permission, creates a `PENDING` row, and returns a presigned PUT URL.
2. The client uploads bytes directly to OSS with the returned method, URL, and
   `Content-Type` header.
3. `POST /api/files/{uuid}/complete/` calls OSS HEAD, verifies size and type,
   stores the ETag, and transitions the row to `READY`.
4. Private owners use `GET /api/files/{uuid}/url/`.
5. Public files use `GET /api/files/public/{uuid}/` without authentication.

Allowed files are images, PDF, text, audio, and video up to 20 MiB. The exact
suffix/MIME pairs are listed in the storage create schema.

## Cleanup

Rows move from `PENDING` to `READY` only after completion verification. Expired
pending rows and owner-deleted rows move to `DELETING`. The
`storages-cleanup` job deletes the OSS object and then the DB row; OSS missing
keys count as successful deletes. If OSS delete fails, the row remains
`DELETING` and the next run retries it.

The deployment must configure OSS CORS to allow client-origin `PUT` requests
with a `Content-Type` request header. Real OSS integration is verified at
deployment; automated tests use a fake provider.
````

- [ ] **Step 2: Update module lists**

In `CLAUDE.md`:

- Add `IStorageProvider -> OssStorageProvider -- presigned OSS PUT/GET, HEAD, delete` to key interfaces.
- Change the module list to include `storages`.
- Add `docs/storages.md` to the top document map.

In `docs/architecture.md`:

- Add `storages` to modules.
- Add a short paragraph naming its model, filters, state transitions, provider, and `/files` routes.

Use the existing prose style and do not reorganize unrelated sections.

- [ ] **Step 3: Verify documentation references**

```bash
rg -n "storages|IStorageProvider|/api/files" CLAUDE.md docs/architecture.md docs/storages.md
```

Expected: all three documents contain the new module references.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/architecture.md docs/storages.md
git commit -m "docs: describe storage uploads"
```

### Task 7: Full Verification And Branch Review

**Files:**
- All files changed by Tasks 1-6.

- [ ] **Step 1: Inspect the complete diff from `main`**

```bash
git diff --stat main...
git diff --check main...
git status --short
```

Expected: only files listed in this plan are changed; no generated caches are staged.

- [ ] **Step 2: Run all tests**

```bash
make test
```

Expected: pytest exits 0 with zero failed tests.

- [ ] **Step 3: Run static checks**

```bash
make lint
make typecheck
```

Expected: Ruff and MyPy both exit 0.

- [ ] **Step 4: Exercise OpenAPI**

```bash
uv run python -c "from fivccliche.cli import app; paths=app.openapi()['paths']; assert '/api/files/uploads/' in paths; assert '/api/files/public/{file_uuid}/' in paths; print(sorted(path for path in paths if path.startswith('/api/files/')))"
```

Expected: command exits 0 and prints the storage routes.

- [ ] **Step 5: Review requirements against the spec**

Read `docs/superpowers/specs/2026-10-28-storages-oss-design.md` and confirm each item:

- `storages` module and `storages_oss` provider names are used.
- OSS dependency and service registration are present.
- create validates filename, exact MIME/suffix pair, 20 MiB maximum, and immutable permission.
- upload response contains PUT URL, headers, and expiry.
- complete uses HEAD, verifies size/type, stores ETag, and is idempotent.
- list filters live in `filters.py`, use scalar `Query()` parameters, and default to `READY`.
- private URL and public redirect enforce the specified state/permission.
- no permission PATCH route exists.
- delete uses `DELETING`, and cleanup retries OSS failures.
- cleanup config, batch selection, conditional claims, and missing-key behavior are tested.
- fake-provider API tests do not contact OSS.
- `CLAUDE.md`, architecture, and storage docs are updated.

Fix any missing item, rerun the relevant focused test plus `make test`, and commit the fix with a precise message, for example:

```bash
git commit -m "fix: enforce storage public access state"
```

- [ ] **Step 6: Final branch status**

```bash
git status --short
git log --oneline main..HEAD
```

Expected: clean worktree and a short sequence of task-scoped commits.

## Self-Review Result

- The plan covers every spec section: module naming, provider boundary, model, API, immutable permission, complete verification, public/private access, deletion, pending cleanup, configuration, errors, tests, and documentation.
- No implementation step relies on an undefined type or route.
- The plan intentionally uses red-green transitions for each behavioral boundary and keeps OSS network access out of automated tests.
