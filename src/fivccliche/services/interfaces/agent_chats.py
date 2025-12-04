from abc import abstractmethod

from fivcglue import IComponent

from fivcplayground.agents.types.repositories import (
    AgentRunRepository as UserChatRepository,
)
from sqlalchemy.ext.asyncio.session import AsyncSession


class IUserChatProvider(IComponent):
    """IUserChatProvider is an interface for defining user chat providers."""

    @abstractmethod
    def get_chat_repository(
        self,
        user_uuid: str | None = None,
        session: AsyncSession | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserChatRepository:
        """Get the chat repository."""
