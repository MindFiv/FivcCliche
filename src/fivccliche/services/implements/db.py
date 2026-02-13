from functools import cached_property

from sqlalchemy.engine.url import make_url, URL
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.engine import create_async_engine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.pool.impl import NullPool
from sqlalchemy.sql.schema import MetaData
from sqlmodel import SQLModel

from fivcglue import IComponentSite, query_component
from fivcglue.interfaces.configs import IConfig

from fivccliche.services.interfaces.db import IDatabase


class DatabaseImpl(IDatabase):
    """
    DatabaseImpl is a default implementation of the IDatabase interface.
    """

    def __init__(self, component_site: IComponentSite, **kwargs):
        """Initialize the database."""
        config = query_component(component_site, IConfig)
        config = config.get_session("database")
        config_url = config.get_value("DB_URL") or "sqlite:///./fivccliche.db"
        self.parsed_url = make_url(config_url)
        print(self.parsed_url)

    @cached_property
    def engine(self) -> AsyncEngine:
        """
        Create and cache the async database engine.

        Handles different database types:
        - SQLite: Uses aiosqlite driver with NullPool
        - Other databases: Uses default async driver

        Returns:
            AsyncEngine: The cached async engine instance.
        """
        if self.parsed_url.drivername.startswith("sqlite"):
            url = URL.create(
                "sqlite+aiosqlite",
                self.parsed_url.username,
                self.parsed_url.password,
                self.parsed_url.host,
                self.parsed_url.port,
                self.parsed_url.database,
                self.parsed_url.query,
            )
            # For SQLite with async support
            return create_async_engine(
                url.render_as_string(hide_password=False),
                connect_args={"check_same_thread": False},
                poolclass=NullPool,
                echo=False,
            )
        else:
            # For other databases (PostgreSQL, MySQL, etc.)
            return create_async_engine(
                self.parsed_url.render_as_string(hide_password=False), echo=False
            )

    def get_url(self) -> str:
        """
        Get the database URL.

        Returns:
            str: The database connection URL.
        """
        return self.parsed_url.render_as_string(hide_password=False)

    def get_metadata(self) -> MetaData:
        """
        Get the database metadata.

        Returns:
            MetaData: SQLModel metadata containing all registered models.
        """
        return SQLModel.metadata

    def get_engine(self) -> AsyncEngine:
        """
        Get the async database engine.

        Returns:
            AsyncEngine: The cached async engine instance.
        """
        return self.engine

    def create_session(self) -> AsyncSession:
        """
        Create a new async database session.

        Returns:
            AsyncSession: A new async session with expire_on_commit=False.
        """
        return AsyncSession(self.engine, expire_on_commit=False)
