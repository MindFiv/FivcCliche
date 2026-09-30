from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Self

from fivcglue import IComponent
from fivcglue.interfaces.mutexes import IMutex
from fivcplayground.agents import (
    AgentRun,
    AgentRunEvent,
    AgentRunRepository as UserChatRepository,
)


class IUserChatProvider(IComponent):
    """IUserChatProvider is an interface for defining user chat providers."""

    @abstractmethod
    def get_chat_repository(
        self,
        user_uuid: str,
        **kwargs,  # ignore additional arguments
    ) -> UserChatRepository:
        """Get the chat repository.

        Implementations must not bind a long-lived DB session. Each repository
        operation should open a short-lived session when it needs the database.
        """

    @abstractmethod
    def get_chat_context(
        self,
        user_uuid: str,
        context: dict | None = None,
        **kwargs,
    ) -> dict:
        """Return a copy of ``context`` with ``user_uuid`` and ``**kwargs`` merged in."""


class IUserChatRun(ABC):
    """One agent turn bound to a user chat."""

    @abstractmethod
    async def __aenter__(self) -> Self:
        """Enter the run context."""

    @abstractmethod
    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit the run context."""

    @abstractmethod
    def stream_async(
        self,
        query: str,
        *,
        mutex: IMutex | None = None,
        timeout: timedelta | None = None,
    ) -> AsyncIterator[tuple[AgentRunEvent, AgentRun]]:
        """Run one query and yield agent events as they occur."""


class IUserChatRunProvider(IComponent):
    """Factory for user-scoped chat runs."""

    @abstractmethod
    def create_chat_run(
        self,
        chat_uuid: str,
        *,
        user_uuid: str,
        agent_id: str = "default",
        context: dict | None = None,
        **kwargs,
    ) -> IUserChatRun:
        """Create a chat run without starting it."""
