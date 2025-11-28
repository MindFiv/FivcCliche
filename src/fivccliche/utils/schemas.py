from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class PaginatedResponse(BaseModel, Generic[T]):
    """Base pagination response schema."""

    total: int = Field(..., description="Total number of items")
    results: list[T] = Field(..., description="List of items")
