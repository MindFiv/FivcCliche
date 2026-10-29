"""HTTP routes for authenticated user-owned storage files."""

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


async def get_required_storage_async(
    provider: IStorageProvider | None = Depends(get_storage_provider_async),
) -> IStorage:
    if provider is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Storage provider is unavailable",
        )
    storage = provider.get_storage()
    if storage is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Storage provider is unavailable",
        )
    return storage


@router_files.post(
    "/uploads/",
    response_model=schemas.StorageFileUploadSchema,
    status_code=status.HTTP_201_CREATED,
)
async def create_upload_async(
    body: schemas.StorageFileCreateSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
    storage: IStorage = Depends(get_required_storage_async),
) -> schemas.StorageFileUploadSchema:
    file_uuid = str(uuid4())
    row = models.StorageFile(
        uuid=file_uuid,
        user_uuid=None if user.is_superuser else user.uuid,
        object_key=f"storages/{file_uuid}",
        filename=body.filename,
        content_type=body.content_type,
        byte_size=body.byte_size,
        is_public=body.is_public,
        status=schemas.StorageStatus.PENDING,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
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
    storage: IStorage = Depends(get_required_storage_async),
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
    storage: IStorage = Depends(get_required_storage_async),
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
    try:
        link = await storage.download_async(
            row.object_key,
            signed=not row.is_public,
            content_type=row.content_type,
            filename=row.filename,
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
    storage: IStorage = Depends(get_required_storage_async),
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
