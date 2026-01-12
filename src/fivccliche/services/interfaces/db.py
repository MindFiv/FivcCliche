from abc import abstractmethod

from fivcglue import IComponent
from sqlalchemy.ext.asyncio.engine import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.sql.schema import MetaData


class IDatabase(IComponent):
    """
    IDatabase is an interface for database management in the Fivccliche framework.

    This interface provides methods to access database metadata, configuration,
    and create async database sessions for use throughout the application.
    """

    @abstractmethod
    def get_metadata(self) -> MetaData:
        """
        Get the database metadata.

        Returns:
            MetaData: SQLAlchemy metadata object containing all registered models.
        """

    @abstractmethod
    def get_url(self) -> str:
        """
        Get the database URL.

        Returns:
            str: The database connection URL.
        """

    @abstractmethod
    def get_engine(self) -> AsyncEngine:
        """
        Get the async database engine.

        Returns:
            AsyncEngine: SQLAlchemy async engine for database operations.
        """

    @abstractmethod
    def create_session(self) -> AsyncSession:
        """
        Create an async database session for dependency injection.

        Returns:
            AsyncSession: A new async database session instance.
        """
