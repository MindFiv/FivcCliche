"""Request and response schemas for user-owned storage files."""

from datetime import datetime
from enum import StrEnum
from typing import Self

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


class StorageStatus(StrEnum):
    """Lifecycle state for a storage file."""

    PENDING = "PENDING"
    READY = "READY"
    DELETING = "DELETING"


class StorageFolderCreateSchema(BaseModel):
    """Validated request used to create a storage folder."""

    name: str = Field(min_length=1, max_length=255)
    bucket_name: str = Field(min_length=1, max_length=255)
    bucket_prefix: str = Field(default="", max_length=512)

    @field_validator("name", "bucket_name")
    @classmethod
    def strip_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("bucket_prefix")
    @classmethod
    def normalize_prefix(cls, value: str) -> str:
        return value.strip().strip("/")


class StorageFolderUpdateSchema(BaseModel):
    """Partial update for a storage folder."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    bucket_name: str | None = Field(default=None, min_length=1, max_length=255)
    bucket_prefix: str | None = Field(default=None, max_length=512)

    @field_validator("name", "bucket_name")
    @classmethod
    def strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("bucket_prefix")
    @classmethod
    def normalize_optional_prefix(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip().strip("/")


class StorageFolderSchema(BaseModel):
    """Storage folder metadata response."""

    uuid: str
    user_uuid: str | None
    name: str
    bucket_name: str
    bucket_prefix: str
    created_at: datetime
    updated_at: datetime


class StorageFileCreateSchema(BaseModel):
    """Validated request used to create a presigned upload."""

    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(min_length=3, max_length=255)
    byte_size: int = Field(gt=0, le=MAX_FILE_BYTES)
    is_public: bool = False
    folder_uuid: str | None = None

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        value = value.strip()
        if not value or "/" in value or "\\" in value or "\x00" in value:
            raise ValueError("filename must be one path component")
        return value

    @model_validator(mode="after")
    def validate_type(self) -> Self:
        suffix = self.filename.rsplit(".", 1)[-1].lower() if "." in self.filename else ""
        expected = STORAGE_TYPES.get(f".{suffix}")
        if expected is None or self.content_type != expected:
            raise ValueError("content_type does not match an allowed filename suffix")
        return self


class StorageFileSchema(BaseModel):
    """Storage file metadata response."""

    uuid: str
    user_uuid: str | None
    folder_uuid: str | None = None
    object_key: str
    filename: str
    content_type: str
    byte_size: int
    is_public: bool
    status: StorageStatus
    etag: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None


class StorageFileUploadSchema(StorageFileSchema):
    """Storage metadata plus direct-upload instructions."""

    upload_method: str = "PUT"
    upload_url: str
    upload_headers: dict[str, str]
    upload_expires_at: datetime


class StorageFileUrlSchema(BaseModel):
    """Owner download response; ``expires_at`` is null for public objects."""

    url: str
    expires_at: datetime | None
