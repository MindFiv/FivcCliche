"""Storage file schema, model, and filter tests."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from fivccliche.modules.storages.filters import (
    StorageFileEditableFilterSet,
    StorageFileFilterSet,
    StorageFileReadableFilterSet,
)
from fivccliche.modules.storages.models import StorageFile
from fivccliche.modules.storages.schemas import (
    StorageFileCreateSchema,
    StorageFileSchema,
    StorageStatus,
)
from fivccliche.modules.users.models import User


def test_create_schema_accepts_allowed_type_pair() -> None:
    body = StorageFileCreateSchema(
        name="demo.mp4",
        content_type="video/mp4",
        byte_size=20 * 1024 * 1024,
        is_public=True,
    )
    assert body.name == "demo.mp4"
    assert body.is_public is True


@pytest.mark.parametrize(
    ("name", "content_type", "byte_size"),
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
    name: str,
    content_type: str,
    byte_size: int,
) -> None:
    with pytest.raises(ValidationError):
        StorageFileCreateSchema(
            name=name,
            content_type=content_type,
            byte_size=byte_size,
            is_public=False,
        )


async def test_storage_file_to_schema(session: AsyncSession) -> None:
    created_at = datetime(2026, 10, 28, 8, 0, tzinfo=UTC)
    owner = User(username="owner")
    session.add(owner)
    await session.commit()
    await session.refresh(owner)
    row = StorageFile(
        user_uuid=owner.uuid,
        object_key="storages/file-uuid",
        name="demo.png",
        content_type="image/png",
        byte_size=12,
        is_public=False,
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
    assert result.user_uuid == owner.uuid
    assert result.is_public is False
    assert result.status == StorageStatus.PENDING


async def test_filters_apply_owner_global_and_http_fields(session: AsyncSession) -> None:
    created_at = datetime(2026, 10, 28, 8, 0, tzinfo=UTC)
    owner = User(username="owner")
    other = User(username="other")
    session.add_all([owner, other])
    await session.commit()
    await session.refresh(owner)
    await session.refresh(other)
    owned = StorageFile(
        user_uuid=owner.uuid,
        object_key="storages/owned",
        name="demo.txt",
        content_type="text/plain",
        byte_size=12,
        is_public=False,
        status=StorageStatus.PENDING,
        created_at=created_at,
        updated_at=created_at,
    )
    global_row = StorageFile(
        user_uuid=None,
        object_key="storages/global",
        name="demo.txt",
        content_type="text/plain",
        byte_size=12,
        is_public=False,
        status=StorageStatus.READY,
        created_at=created_at,
        updated_at=created_at,
    )
    others = StorageFile(
        user_uuid=other.uuid,
        object_key="storages/other",
        name="demo.txt",
        content_type="text/plain",
        byte_size=12,
        is_public=True,
        status=StorageStatus.READY,
        created_at=created_at,
        updated_at=created_at,
    )
    session.add_all([owned, global_row, others])
    await session.commit()

    filters = StorageFileFilterSet(owner.uuid, is_superuser=False)
    filters.parse(
        status=StorageStatus.PENDING,
        is_public=False,
        content_type="text/plain",
        created_at_from=created_at,
        created_at_to=created_at,
    )
    pending = (await session.execute(filters.filter(select(StorageFile)))).scalars().all()
    assert [row.uuid for row in pending] == [owned.uuid]

    list_filters = StorageFileFilterSet(owner.uuid, is_superuser=False)
    list_filters.parse(status=StorageStatus.READY)
    ready = (await session.execute(list_filters.filter(select(StorageFile)))).scalars().all()
    assert {row.uuid for row in ready} == {global_row.uuid}

    global_filters = StorageFileFilterSet(owner.uuid, is_superuser=False)
    global_filters.parse(status=StorageStatus.READY, is_global=True)
    globals_only = (
        (await session.execute(global_filters.filter(select(StorageFile)))).scalars().all()
    )
    assert [row.uuid for row in globals_only] == [global_row.uuid]

    owned_filters = StorageFileFilterSet(owner.uuid, is_superuser=False)
    owned_filters.parse(is_global=False)
    only_owned = (await session.execute(owned_filters.filter(select(StorageFile)))).scalars().all()
    assert [row.uuid for row in only_owned] == [owned.uuid]

    readable = StorageFileReadableFilterSet(owner.uuid, is_superuser=False)
    seen = (
        await session.execute(
            readable.filter(select(StorageFile).where(StorageFile.uuid == global_row.uuid))
        )
    ).scalar_one_or_none()
    assert seen is not None

    editable_other = StorageFileEditableFilterSet(owner.uuid, is_superuser=False)
    assert (
        await session.execute(
            editable_other.filter(select(StorageFile).where(StorageFile.uuid == global_row.uuid))
        )
    ).scalar_one_or_none() is None

    editable_super = StorageFileEditableFilterSet(owner.uuid, is_superuser=True)
    assert (
        await session.execute(
            editable_super.filter(select(StorageFile).where(StorageFile.uuid == global_row.uuid))
        )
    ).scalar_one_or_none() is not None
