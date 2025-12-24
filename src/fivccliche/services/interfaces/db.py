from abc import abstractmethod

from fivcglue import IComponent
from sqlalchemy.ext.asyncio.session import AsyncSession


class IDatabase(IComponent):
    """
    IDatabase is an interface for defining database models in the Fivccliche framework.
    """

    @abstractmethod
    async def setup_async(self) -> None:
        """Create all database tables."""

    @abstractmethod
    async def get_session_async(self) -> AsyncSession:
        """Get an async database session for dependency injection."""
