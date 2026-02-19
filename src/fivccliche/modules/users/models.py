from datetime import datetime, timezone
from uuid import uuid1

from passlib.context import CryptContext
from sqlalchemy import DateTime
from sqlmodel import Field, SQLModel
from pydantic import EmailStr

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


class User(SQLModel, table=True):
    """User model."""

    __tablename__ = "user"

    uuid: str = Field(
        default_factory=lambda: str(uuid1()),
        primary_key=True,
        max_length=36,
        description="User ID (UUID).",
    )
    username: str = Field(max_length=255, index=True, unique=True, description="User name.")
    email: EmailStr | None = Field(
        default=None, max_length=255, index=True, unique=True, description="User email."
    )
    full_name: str | None = Field(default=None, max_length=1024, description="User full name.")
    hashed_password: str | None = Field(default=None, max_length=255, description="User password.")
    created_at: datetime = Field(
        sa_type=DateTime(timezone=True),
        default_factory=lambda: datetime.now(timezone.utc),
        description="User creation time.",
    )
    signed_in_at: datetime | None = Field(
        sa_type=DateTime(timezone=True),
        default=None,
        description="User last sign in time.",
    )
    is_active: bool = True
    is_superuser: bool = False

    def check_password(self, password: str) -> bool:
        result: bool = pwd_context.verify(password, self.hashed_password)
        return result

    def change_password(self, password: str) -> None:
        self.hashed_password = pwd_context.hash(password)
