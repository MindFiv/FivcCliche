from abc import abstractmethod

from fivcglue import IComponent
from fivcplayground.agents import (
    AgentRunRepository as UserChatRepository,
)
from fivcplayground.tools import Tool
from sqlalchemy.ext.asyncio.session import AsyncSession


class IUserChatContext(IComponent):
    """IUserChatContext is an interface for defining user chat context."""

    @abstractmethod
    async def get_tools_async(self, **kwargs) -> list[Tool]:
        """Get the chat tools."""


class IUserChatProvider(IComponent):
    """IUserChatProvider is an interface for defining user chat providers."""

    @abstractmethod
    def get_chat_repository(
        self,
        user_uuid: str,
        session: AsyncSession,
        **kwargs,  # ignore additional arguments
    ) -> UserChatRepository:
        """Get the chat repository."""

    @abstractmethod
    def get_chat_context(
        self,
        user_uuid: str,
        session: AsyncSession,
        context: dict | None = None,
        **kwargs,
    ) -> IUserChatContext | None:
        """Get the chat context."""
