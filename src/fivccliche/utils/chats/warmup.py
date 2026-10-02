"""In-flight warmup scoped to the current asyncio task."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, ClassVar, Generic, TypeVar

logger = logging.getLogger(__name__)

K = TypeVar("K")
T = TypeVar("T")


async def _cancel_task(task: asyncio.Task[Any] | None) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return
    if not task.cancelled():
        task.exception()


class SessionWarmup(Generic[K, T]):
    """One background load for the current task, reused while its key matches."""

    _current: ClassVar[ContextVar[SessionWarmup[Any, Any] | None]] = ContextVar(
        "session_warmup",
        default=None,
    )

    def __init__(self, key: K, factory: Callable[[], Coroutine[Any, Any, T]]) -> None:
        self.key = key
        self._factory = factory
        self._task: asyncio.Task[T] | None = None

    @asynccontextmanager
    async def open(self) -> AsyncIterator[None]:
        """Start the factory and expose this warmup to ``take`` until exit."""
        self._task = asyncio.create_task(self._factory())
        token = self._current.set(self)
        try:
            yield
        finally:
            task = self._task
            self._current.reset(token)
            await _cancel_task(task)

    @classmethod
    async def take(cls, key: K) -> Any:
        """Return the warmed value, or None when this turn must build its own.

        The active warmup is stored as ``SessionWarmup[Any, Any]``, so the
        value type is not recovered here. Callers treat ``None`` as a miss.
        """
        warm = cls._current.get()
        if warm is None or warm._task is None or warm.key != key:
            return None
        task = warm._task
        try:
            return await task
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Session warmup failed; building this turn", exc_info=True)
            if warm._task is task:
                warm._task = None
            return None
