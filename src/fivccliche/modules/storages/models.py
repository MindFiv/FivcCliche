from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Boolean, Column, DateTime, Index, String, UniqueConstraint
from sqlmodel import Field, SQLModel

from . import schemas


class StorageFolder(SQLModel, table=True):
    """Named bucket + key_prefix binding used by storage files."""

    __tablename__ = "storage_folder"
    __table_args__ = (Index("ix_storage_folder_name_user_uuid", "name", "user_uuid", unique=True),)

    uuid: str = Field(
        default_factory=lambda: str(uuid4()),
        primary_key=True,
        max_length=36,
    )
    user_uuid: str | None = Field(
        default=None,
        foreign_key="user.uuid",
        index=True,
        max_length=36,
    )
    name: str = Field(max_length=255, index=True)
    bucket_name: str = Field(max_length=255)
    bucket_prefix: str = Field(default="", max_length=512)
    created_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True),
            default=lambda: datetime.now(timezone.utc),
            nullable=False,
        ),
    )
    updated_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True),
            default=lambda: datetime.now(timezone.utc),
            nullable=False,
        ),
    )

    def to_schema(self) -> schemas.StorageFolderSchema:
        return schemas.StorageFolderSchema(
            uuid=self.uuid,
            user_uuid=self.user_uuid,
            name=self.name,
            bucket_name=self.bucket_name,
            bucket_prefix=self.bucket_prefix,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


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
    user_uuid: str | None = Field(
        default=None,
        foreign_key="user.uuid",
        index=True,
        max_length=36,
    )
    folder_uuid: str | None = Field(
        default=None,
        foreign_key="storage_folder.uuid",
        index=True,
        max_length=36,
    )
    object_key: str = Field(max_length=512)
    filename: str = Field(max_length=255)
    content_type: str = Field(max_length=255)
    byte_size: int = Field(gt=0)
    is_public: bool = Field(default=False, sa_column=Column(Boolean, nullable=False))
    status: schemas.StorageStatus = Field(sa_column=Column(String(16), nullable=False))
    etag: str | None = Field(default=None, max_length=128)
    created_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True),
            default=lambda: datetime.now(timezone.utc),
            nullable=False,
        ),
    )
    updated_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True),
            default=lambda: datetime.now(timezone.utc),
            nullable=False,
        ),
    )
    completed_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )

    def to_schema(self) -> schemas.StorageFileSchema:
        return schemas.StorageFileSchema(
            uuid=self.uuid,
            user_uuid=self.user_uuid,
            folder_uuid=self.folder_uuid,
            object_key=self.object_key,
            filename=self.filename,
            content_type=self.content_type,
            byte_size=self.byte_size,
            is_public=self.is_public,
            status=self.status,
            etag=self.etag,
            created_at=self.created_at,
            updated_at=self.updated_at,
            completed_at=self.completed_at,
        )
