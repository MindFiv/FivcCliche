from unittest.mock import MagicMock, patch

from sqlalchemy.pool.impl import NullPool

from fivccliche.services.implements.db import DatabaseImpl, _parse_optional_int


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


class TestParseOptionalInt:
    def test_parses_int_and_numeric_string(self):
        assert _parse_optional_int(2) == 2
        assert _parse_optional_int("10") == 10

    def test_returns_none_for_missing_or_invalid(self):
        assert _parse_optional_int(None) is None
        assert _parse_optional_int("") is None
        assert _parse_optional_int("abc") is None


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
        db = _make_db(
            {
                "DB_URL": "sqlite:///:memory:",
                "DB_POOL_SIZE": "2",
                "DB_MAX_OVERFLOW": "3",
            }
        )
        engine = db.get_engine()
        assert isinstance(engine.pool, NullPool)
