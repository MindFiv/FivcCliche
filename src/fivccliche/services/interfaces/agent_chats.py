from abc import abstractmethod

from fivcglue import IComponent
from fivcplayground.agents import (
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
