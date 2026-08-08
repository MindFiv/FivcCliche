"""Agent memory interfaces.

Implementation-agnostic memory contracts so the project can swap backends
(Hindsight, mem0, ...) without touching business code. Business code should
only depend on the neutral return models (``MemoryContent`` /
``MemoryRetainResult`` / ``MemoryRecallResult`` / ``MemoryListResult``) and
never on a backend's native response types.
"""

from abc import abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from fivcglue import IComponent


class MemoryContent(BaseModel):
    """A single recalled memory, normalized across backends.

    Field mapping (illustrative, not exhaustive):
    - Hindsight: text -> content, type -> categories=[type], timestamp -> created_at
    - mem0:      memory -> content, categories -> categories, created_at -> created_at
    """

    id: str | None = None
    content: str
    score: float | None = None
    categories: list[str] | None = None
    metadata: dict[str, Any] | None = None
    created_at: datetime | None = None


class MemoryRetainResult(BaseModel):
    """Outcome of storing memory content."""

    success: bool = True
    count: int = 0
    ids: list[str] | None = None
    # Backend-native payload; escape hatch for advanced use. Business code
    # should prefer the normalized fields above and avoid relying on this.
    raw: Any | None = None


class MemoryRecallResult(BaseModel):
    """Outcome of recalling memories."""

    items: list[MemoryContent] = Field(default_factory=list)
    # Backend-native payload; escape hatch for advanced use. Business code
    # should prefer ``items`` and avoid relying on this.
    raw: Any | None = None


class MemoryListResult(BaseModel):
    """Outcome of listing memories with pagination."""

    items: list[MemoryContent] = Field(default_factory=list)
    total: int = 0
    # Backend-native payload; escape hatch for advanced use. Business code
    # should prefer the normalized fields above and avoid relying on this.
    raw: Any | None = None


class IUserMemory(IComponent):
    """IUserMemory is an interface for a single user memory space."""

    @abstractmethod
    async def retain_async(self, content: str) -> MemoryRetainResult:
        """Store a piece of memory content."""

    @abstractmethod
    async def recall_async(self, query: str) -> MemoryRecallResult:
        """Recall memories by semantic similarity to ``query``."""

    @abstractmethod
    async def list_async(
        self,
        *,
        skip: int = 0,
        limit: int = 100,
        **kwargs: Any,
    ) -> MemoryListResult:
        """List memories with pagination (``skip`` / ``limit``)."""


class IUserMemoryProvider(IComponent):
    """IUserMemoryProvider is an interface for defining user memory providers."""

    @abstractmethod
    def get_memory(
        self,
        space_id: str | None = None,
        **kwargs,  # ignore additional arguments
    ) -> IUserMemory:
        """Get user memory by space."""
