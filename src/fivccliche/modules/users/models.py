from datetime import datetime
from uuid import uuid1

from sqlmodel import Field, SQLModel
from pydantic import EmailStr


class User(SQLModel, table=True):
    """User model."""

    __tablename__ = "user"

    uuid: str = Field(
        default_factory=lambda: str(uuid1()),
        primary_key=True,
        max_length=32,
        description="User ID (UUID).",
    )
    username: str = Field(max_length=255, index=True, unique=True, description="User name.")
    email: EmailStr | None = Field(
        default=None, max_length=255, index=True, unique=True, description="User email."
    )
    full_name: str | None = Field(default=None, max_length=1024, description="User full name.")
    hashed_password: str | None = Field(default=None, max_length=255, description="User password.")
    created_at: datetime = Field(default_factory=datetime.now, description="User creation time.")
    signed_in_at: datetime | None = Field(default=None, description="User last sign in time.")
    is_active: bool = True
    is_superuser: bool = False
