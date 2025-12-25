from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserBase(BaseModel):
    """Base user schema with common fields."""

    username: str = Field(..., min_length=3, max_length=255, description="Username")
    email: EmailStr = Field(..., description="User email address")


class UserCreate(UserBase):
    """Schema for creating a new user."""

    password: str = Field(..., min_length=8, max_length=255, description="User password")


class UserUpdate(BaseModel):
    """Schema for updating a user."""

    username: str | None = Field(None, min_length=3, max_length=255, description="Username")
    email: EmailStr | None = Field(None, description="User email address")
    password: str | None = Field(None, min_length=8, max_length=255, description="User password")
    is_active: bool | None = Field(None, description="User active status")


class UserRead(UserBase):
    """Schema for reading user data (response)."""

    uuid: str = Field(..., description="User ID (UUID)")
    created_at: datetime = Field(..., description="User creation time")
    signed_in_at: datetime | None = Field(None, description="User last sign in time")
    is_active: bool = Field(..., description="User active status")
    is_superuser: bool = Field(..., description="User superuser status")

    model_config = ConfigDict(from_attributes=True)


class UserLogin(BaseModel):
    """Schema for user login."""

    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="User password")


class UserLoginResponse(BaseModel):
    """Schema for login response with user data and token."""

    # user: UserRead = Field(..., description="Authenticated user data")
    access_token: str = Field(..., description="JWT access token")
    expires_in: int = Field(..., description="Token expiration time in seconds")
