from functools import cached_property

from sqlalchemy.ext.asyncio.engine import create_async_engine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.pool.impl import NullPool
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
        self.url = config.get_value("URL") or "sqlite+aiosqlite:///./fivccliche.db"

    @cached_property
    def engine_async(self):
        if self.url.startswith("sqlite"):
            # For SQLite with async support
            return create_async_engine(
                self.url,
                connect_args={"check_same_thread": False},
                poolclass=NullPool,
                echo=False,
            )
        else:
            # For other databases (PostgreSQL, MySQL, etc.)
            return create_async_engine(
                self.url,
                echo=False,
            )

    async def setup_async(self) -> None:
        async with self.engine_async.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    async def get_session_async(self) -> AsyncSession:
        return AsyncSession(self.engine_async, expire_on_commit=False)
