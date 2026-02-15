from abc import abstractmethod
from collections.abc import Callable

from fivcglue import IComponent

from fivcplayground.agents import (
    AgentRunRepository as UserChatRepository,
)
from sqlalchemy.ext.asyncio.session import AsyncSession


class IUserChatContext(IComponent):
    """IUserChatContext is an interface for defining user chat context."""

    @abstractmethod
    async def get_tool_funcs_async(self, **kwargs) -> list[Callable]:
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
        **kwargs,
    ) -> IUserChatContext | None:
        """Get the chat context."""
