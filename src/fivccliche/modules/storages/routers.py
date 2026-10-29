"""HTTP routes for authenticated user-owned storage files and folders."""

import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from fivccliche.services.interfaces.storages import (
    IStorage,
    IStorageProvider,
    StorageError,
)
from fivccliche.utils.deps import (
    IUser,
    get_authenticated_user_async,
    get_db_session_async,
    get_storage_provider_async,
)
from fivccliche.utils.schemas import PaginatedResponse

from . import filters as filters_module
from . import models, schemas, utils

logger = logging.getLogger(__name__)
router_files = APIRouter(tags=["files"], prefix="/files")
router_folders = APIRouter(tags=["folders"], prefix="/folders")


def _require_provider(
    provider: IStorageProvider | None = Depends(get_storage_provider_async),
) -> IStorageProvider:
    if provider is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Storage provider is unavailable",
        )
    return provider


def _require_storage(
    provider: IStorageProvider, row: models.StorageFile, folder: models.StorageFolder | None
) -> IStorage:
    storage = utils.resolve_storage_for_file(provider, row, folder)
    if storage is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Storage provider is unavailable",
        )
    return storage


async def _load_folder_for_file_async(
    session: AsyncSession,
    row: models.StorageFile,
) -> models.StorageFolder | None:
    if row.folder_uuid is None:
        return None
    folder = (
        await session.execute(
            select(models.StorageFolder).where(models.StorageFolder.uuid == row.folder_uuid)
        )
    ).scalar_one_or_none()
    return folder


# ---------------------------------------------------------------------------
# Folders (superuser write)
# ---------------------------------------------------------------------------


@router_folders.post(
    "/",
    response_model=schemas.StorageFolderSchema,
    status_code=status.HTTP_201_CREATED,
)
async def create_folder_async(
    body: schemas.StorageFolderCreateSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.StorageFolderSchema:
    if not user.is_superuser:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only superusers can create folders")
    now = datetime.now(timezone.utc)
    row = models.StorageFolder(
        user_uuid=None,
        name=body.name,
        bucket_name=body.bucket_name,
        bucket_prefix=body.bucket_prefix,
        created_at=now,
        updated_at=now,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row.to_schema()


@router_folders.get(
    "/",
    response_model=PaginatedResponse[schemas.StorageFolderSchema],
)
async def list_folders_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse[schemas.StorageFolderSchema]:
    filters = filters_module.StorageFolderReadableFilterSet(user.uuid, is_superuser=False)
    statement = filters.filter(select(models.StorageFolder))
    rows = (
        (
            await session.execute(
                statement.order_by(col(models.StorageFolder.created_at).desc())
                .offset(skip)
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    count_statement = filters.filter(select(func.count(col(models.StorageFolder.uuid))))
    total = (await session.execute(count_statement)).scalar_one()
    return PaginatedResponse[schemas.StorageFolderSchema](
        total=total,
        results=[row.to_schema() for row in rows],
    )


@router_folders.get(
    "/{folder_uuid}/",
    response_model=schemas.StorageFolderSchema,
)
async def get_folder_async(
    folder_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.StorageFolderSchema:
    readable = filters_module.StorageFolderReadableFilterSet(
        user.uuid, is_superuser=user.is_superuser
    )
    statement = readable.filter(
        select(models.StorageFolder).where(models.StorageFolder.uuid == folder_uuid)
    )
    row: models.StorageFolder | None = (await session.execute(statement)).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Folder not found")
    return row.to_schema()


@router_folders.patch(
    "/{folder_uuid}/",
    response_model=schemas.StorageFolderSchema,
)
async def update_folder_async(
    folder_uuid: str,
    body: schemas.StorageFolderUpdateSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.StorageFolderSchema:
    if not user.is_superuser:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only superusers can update folders")
    editable = filters_module.StorageFolderEditableFilterSet(user.uuid, is_superuser=True)
    statement = editable.filter(
        select(models.StorageFolder).where(models.StorageFolder.uuid == folder_uuid)
    )
    row: models.StorageFolder | None = (await session.execute(statement)).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Folder not found")
    fields_set = body.model_fields_set
    if "name" in fields_set and body.name is not None:
        row.name = body.name
    if "bucket_name" in fields_set and body.bucket_name is not None:
        row.bucket_name = body.bucket_name
    if "bucket_prefix" in fields_set and body.bucket_prefix is not None:
        row.bucket_prefix = body.bucket_prefix
    row.updated_at = datetime.now(timezone.utc)
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row.to_schema()


@router_folders.delete(
    "/{folder_uuid}/",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_folder_async(
    folder_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    if not user.is_superuser:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only superusers can delete folders")
    editable = filters_module.StorageFolderEditableFilterSet(user.uuid, is_superuser=True)
    statement = editable.filter(
        select(models.StorageFolder).where(models.StorageFolder.uuid == folder_uuid)
    )
    row: models.StorageFolder | None = (await session.execute(statement)).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Folder not found")
    await session.delete(row)
    await session.commit()


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


@router_files.post(
    "/uploads/",
    response_model=schemas.StorageFileUploadSchema,
    status_code=status.HTTP_201_CREATED,
)
async def create_upload_async(
    body: schemas.StorageFileCreateSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
    provider: IStorageProvider = Depends(_require_provider),
) -> schemas.StorageFileUploadSchema:
    folder: models.StorageFolder | None = None
    if body.folder_uuid is not None:
        readable = filters_module.StorageFolderReadableFilterSet(
            user.uuid, is_superuser=user.is_superuser
        )
        folder = (
            await session.execute(
                readable.filter(
                    select(models.StorageFolder).where(
                        models.StorageFolder.uuid == body.folder_uuid
                    )
                )
            )
        ).scalar_one_or_none()
        if folder is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Folder not found")

    file_uuid = str(uuid4())
    owner_uuid = None if user.is_superuser else user.uuid
    row = models.StorageFile(
        uuid=file_uuid,
        user_uuid=owner_uuid,
        folder_uuid=folder.uuid if folder is not None else None,
        object_key=file_uuid,
        name=body.name,
        content_type=body.content_type,
        byte_size=body.byte_size,
        is_public=body.is_public,
        status=schemas.StorageStatus.PENDING,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)

    storage = _require_storage(provider, row, folder)
    try:
        signed = await storage.upload_async(
            row.object_key,
            row.content_type,
            is_public=row.is_public,
        )
    except StorageError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "Could not sign storage upload",
        ) from exc

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
    provider: IStorageProvider = Depends(_require_provider),
) -> schemas.StorageFileSchema:
    editable = filters_module.StorageFileEditableFilterSet(
        user.uuid, is_superuser=user.is_superuser
    )
    owned_statement = editable.filter(
        select(models.StorageFile).where(models.StorageFile.uuid == file_uuid)
    )
    row: models.StorageFile | None = (await session.execute(owned_statement)).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found")
    if row.status == schemas.StorageStatus.READY:
        return row.to_schema()
    if row.status == schemas.StorageStatus.DELETING:
        raise HTTPException(status.HTTP_410_GONE, "File is being deleted")

    folder = await _load_folder_for_file_async(session, row)
    storage = _require_storage(provider, row, folder)
    try:
        head = await storage.peek_async(row.object_key)
    except StorageError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "Could not verify storage object",
        ) from exc

    if head is None or head.byte_size != row.byte_size or head.content_type != row.content_type:
        claimed = await utils.transition_storage_file_status_async(
            session,
            file_uuid=file_uuid,
            user_uuid=row.user_uuid,
            target=schemas.StorageStatus.DELETING,
            current_statuses=(schemas.StorageStatus.PENDING,),
        )
        await session.commit()
        if not claimed:
            raise HTTPException(status.HTTP_410_GONE, "File is being deleted")
        if head is not None:
            try:
                await storage.delete_async(row.object_key)
            except StorageError as exc:
                logger.exception(
                    "Could not delete rejected storage object %s",
                    file_uuid,
                )
                raise HTTPException(
                    status.HTTP_502_BAD_GATEWAY,
                    "Could not remove rejected file",
                ) from exc
        await utils.delete_storage_file_row_async(session, file_uuid=file_uuid)
        await session.commit()
        detail = (
            "Storage object was not uploaded"
            if head is None
            else "Uploaded file does not match its declaration"
        )
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail)

    claimed = await utils.transition_storage_file_status_async(
        session,
        file_uuid=file_uuid,
        user_uuid=row.user_uuid,
        target=schemas.StorageStatus.READY,
        current_statuses=(schemas.StorageStatus.PENDING,),
        values={
            "etag": head.etag,
            "completed_at": datetime.now(timezone.utc),
        },
    )
    await session.commit()
    if not claimed:
        await session.refresh(row)
        if row.status == schemas.StorageStatus.READY:
            return row.to_schema()
        raise HTTPException(status.HTTP_410_GONE, "File is being deleted")
    await session.refresh(row)
    return row.to_schema()


@router_files.get(
    "/",
    response_model=PaginatedResponse[schemas.StorageFileSchema],
)
async def list_files_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status_filter: schemas.StorageStatus = Query(
        schemas.StorageStatus.READY,
        alias="status",
        description="PENDING or READY; DELETING is hidden",
    ),
    is_public: bool | None = Query(None),
    is_global: bool | None = Query(None),
    content_type: str | None = Query(None, min_length=3, max_length=255),
    created_at_from: datetime | None = Query(None),
    created_at_to: datetime | None = Query(None),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse[schemas.StorageFileSchema]:
    if status_filter == schemas.StorageStatus.DELETING:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "DELETING files are not listable",
        )
    filters = filters_module.StorageFileFilterSet(user.uuid, is_superuser=False)
    filters.parse(
        status=status_filter,
        is_public=is_public,
        is_global=is_global,
        content_type=content_type,
        created_at_from=created_at_from,
        created_at_to=created_at_to,
    )
    statement = filters.filter(select(models.StorageFile))
    rows = (
        (
            await session.execute(
                statement.order_by(col(models.StorageFile.created_at).desc())
                .offset(skip)
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    count_statement = filters.filter(select(func.count(col(models.StorageFile.uuid))))
    total = (await session.execute(count_statement)).scalar_one()
    return PaginatedResponse[schemas.StorageFileSchema](
        total=total,
        results=[row.to_schema() for row in rows],
    )


@router_files.get(
    "/{file_uuid}/",
    response_model=schemas.StorageFileSchema,
)
async def get_file_async(
    file_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.StorageFileSchema:
    readable = filters_module.StorageFileReadableFilterSet(
        user.uuid, is_superuser=user.is_superuser
    )
    owned_statement = readable.filter(
        select(models.StorageFile).where(col(models.StorageFile.uuid) == file_uuid)
    )
    row: models.StorageFile | None = (await session.execute(owned_statement)).scalar_one_or_none()
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
    provider: IStorageProvider = Depends(_require_provider),
) -> schemas.StorageFileUrlSchema:
    readable = filters_module.StorageFileReadableFilterSet(
        user.uuid, is_superuser=user.is_superuser
    )
    owned_statement = readable.filter(
        select(models.StorageFile).where(col(models.StorageFile.uuid) == file_uuid)
    )
    row: models.StorageFile | None = (await session.execute(owned_statement)).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found")
    if row.status != schemas.StorageStatus.READY:
        raise HTTPException(status.HTTP_409_CONFLICT, "File is not ready")
    folder = await _load_folder_for_file_async(session, row)
    storage = _require_storage(provider, row, folder)
    try:
        link = await storage.download_async(
            row.object_key,
            signed=not row.is_public,
            content_type=row.content_type,
            filename=row.name,
        )
    except StorageError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "Could not sign storage download",
        ) from exc
    return schemas.StorageFileUrlSchema(
        url=link.url,
        expires_at=link.expires_at,
    )


@router_files.delete(
    "/{file_uuid}/",
    status_code=status.HTTP_202_ACCEPTED,
)
async def delete_file_async(
    file_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
    provider: IStorageProvider = Depends(_require_provider),
) -> dict[str, bool]:
    editable = filters_module.StorageFileEditableFilterSet(
        user.uuid, is_superuser=user.is_superuser
    )
    owned_statement = editable.filter(
        select(models.StorageFile).where(col(models.StorageFile.uuid) == file_uuid)
    )
    row: models.StorageFile | None = (await session.execute(owned_statement)).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found")
    if row.status == schemas.StorageStatus.DELETING:
        return {"success": True}

    folder = await _load_folder_for_file_async(session, row)
    storage = _require_storage(provider, row, folder)

    claimed = await utils.transition_storage_file_status_async(
        session,
        file_uuid=file_uuid,
        user_uuid=row.user_uuid,
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
        await storage.delete_async(row.object_key)
    except StorageError:
        logger.exception("Storage object delete failed for file %s", file_uuid)
        return {"success": True}
    await utils.delete_storage_file_row_async(session, file_uuid=file_uuid)
    await session.commit()
    return {"success": True}
