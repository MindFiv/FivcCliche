"""Pydantic schemas for the agent memories HTTP API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class MemoryContentSchema(BaseModel):
    """A single memory item returned by list/recall endpoints."""

    id: str | None = None
    content: str
    score: float | None = None
    categories: list[str] | None = None
    metadata: dict[str, Any] | None = None
    created_at: datetime | None = None


class MemoryRecallResponseSchema(BaseModel):
    """Response body for semantic memory recall."""

    results: list[MemoryContentSchema] = Field(default_factory=list)


class MemoryRetainRequestSchema(BaseModel):
    """Request body for retaining memory content."""

    content: str = Field(..., min_length=1, description="Memory content to retain")


class MemoryRetainResponseSchema(BaseModel):
    """Response body for retaining memory content."""

    success: bool = True
    count: int = 0
    ids: list[str] | None = None
