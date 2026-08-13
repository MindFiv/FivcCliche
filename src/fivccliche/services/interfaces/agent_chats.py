from abc import abstractmethod

from fivcglue import IComponent
from fivcplayground.agents import (
    AgentRunRepository as UserChatRepository,
)
from fivcplayground.tools import Tool


class IUserChatContext(IComponent):
    """IUserChatContext is an interface for defining user chat context."""

    @abstractmethod
    async def get_tools_async(self, **kwargs) -> list[Tool]:
        """Get the chat tools."""

    @abstractmethod
    async def get_is_skills_enabled_async(self, **kwargs) -> bool:
        """Get if skills are enabled."""


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
    ) -> IUserChatContext | None:
        """Get the chat context."""
