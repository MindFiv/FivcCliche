from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.pool.impl import NullPool

from fivccliche.services.implements.db import DatabaseImpl
from fivccliche.utils.deps import get_db_session_context_async


def _make_db(config_values: dict[str, str | None]) -> DatabaseImpl:
    session = MagicMock()
    session.get_value.side_effect = lambda key: config_values.get(key)
    config = MagicMock()
    config.get_session.return_value = session
    component_site = MagicMock()
    with patch(
        "fivccliche.services.implements.db.query_component",
        return_value=config,
    ):
        return DatabaseImpl(component_site)


class TestDatabaseImplDefaultUrl:
    def test_missing_db_url_starts_pg0(self):
        mock_pg = MagicMock()
        mock_pg.uri = "postgresql://postgres:postgres@127.0.0.1:54321/postgres"
        with patch("pg0.Pg0", return_value=mock_pg) as mock_pg0_cls:
            db = _make_db({"DB_URL": None})

        mock_pg0_cls.assert_called_once_with(name="fivccliche")
        mock_pg.start.assert_called_once()
        assert db.get_url() == "postgresql+asyncpg://postgres:postgres@127.0.0.1:54321/postgres"

    def test_missing_db_url_reuses_running_pg0(self):
        from pg0 import Pg0AlreadyRunningError

        mock_pg = MagicMock()
        mock_pg.uri = "postgresql://postgres:postgres@127.0.0.1:54321/postgres"
        mock_pg.start.side_effect = Pg0AlreadyRunningError("already")
        with patch("pg0.Pg0", return_value=mock_pg):
            db = _make_db({"DB_URL": None})

        assert db.get_url() == "postgresql+asyncpg://postgres:postgres@127.0.0.1:54321/postgres"


class TestDatabaseImplPoolConfig:
    def test_non_sqlite_passes_pool_settings_to_engine(self):
        db = _make_db(
            {
                "DB_URL": "postgresql+asyncpg://user:pass@localhost/dbname",
                "DB_POOL_SIZE": "2",
                "DB_MAX_OVERFLOW": "3",
            }
        )
        mock_engine = MagicMock()
        with patch(
            "fivccliche.services.implements.db.create_async_engine",
            return_value=mock_engine,
        ) as mock_create:
            engine = db.get_engine()

        assert engine is mock_engine
        mock_create.assert_called_once()
        _, kwargs = mock_create.call_args
        assert kwargs["pool_size"] == 2
        assert kwargs["max_overflow"] == 3
        assert kwargs["echo"] is False

    def test_non_sqlite_omits_pool_settings_when_unset(self):
        db = _make_db({"DB_URL": "postgresql+asyncpg://user:pass@localhost/dbname"})
        mock_engine = MagicMock()
        with patch(
            "fivccliche.services.implements.db.create_async_engine",
            return_value=mock_engine,
        ) as mock_create:
            db.get_engine()

        _, kwargs = mock_create.call_args
        assert "pool_size" not in kwargs
        assert "max_overflow" not in kwargs

    def test_non_sqlite_ignores_invalid_pool_settings(self):
        db = _make_db(
            {
                "DB_URL": "postgresql+asyncpg://user:pass@localhost/dbname",
                "DB_POOL_SIZE": "not-a-number",
                "DB_MAX_OVERFLOW": "",
            }
        )
        mock_engine = MagicMock()
        with patch(
            "fivccliche.services.implements.db.create_async_engine",
            return_value=mock_engine,
        ) as mock_create:
            db.get_engine()

        _, kwargs = mock_create.call_args
        assert "pool_size" not in kwargs
        assert "max_overflow" not in kwargs

    def test_sqlite_uses_null_pool_and_ignores_pool_settings(self):
        pytest.importorskip("aiosqlite")
        db = _make_db(
            {
                "DB_URL": "sqlite:///:memory:",
                "DB_POOL_SIZE": "2",
                "DB_MAX_OVERFLOW": "3",
            }
        )
        engine = db.get_engine()
        assert isinstance(engine.pool, NullPool)


class TestGetDbSessionContextAsync:
    @pytest.mark.asyncio
    async def test_reuses_provided_session(self):
        provided = object()
        async with get_db_session_context_async(session=provided) as scoped:
            assert scoped is provided

    @pytest.mark.asyncio
    async def test_opens_short_lived_session_when_none(self):
        owned = AsyncMock()
        mock_db = MagicMock()
        mock_db.create_session.return_value = owned

        with patch("fivccliche.utils.deps.default_db", lambda: mock_db):
            async with get_db_session_context_async() as scoped:
                assert scoped is owned
        owned.close.assert_awaited_once()
