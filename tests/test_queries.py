"""Tests for reusable query parsing and SQL helpers."""

from sqlalchemy import Column, JSON, String, select
from sqlalchemy.orm import declarative_base
import pytest

from fivccliche.utils import queries


Base = declarative_base()


class QueryTestModel(Base):
    """Minimal SQLAlchemy model for compiling JSON filter statements."""

    __tablename__ = "query_test_model"

    uuid = Column(String, primary_key=True)
    context = Column(JSON)
    options = Column(JSON)


def test_parse_dotted_json_filters_groups_values_by_root_field():
    """Dotted query params are grouped by their root JSON field."""
    filters = queries.parse_dotted_json_filters(
        [
            ("context.key1", "xxx"),
            ("context.key2", "yyy"),
        ],
        allowed_fields={"context"},
    )

    assert filters == {"context": {"key1": "xxx", "key2": "yyy"}}


def test_parse_dotted_json_filters_supports_multiple_allowed_roots():
    """The parser supports endpoints with more than one JSON root field."""
    filters = queries.parse_dotted_json_filters(
        [
            ("context.key1", "xxx"),
            ("options.key2", "yyy"),
        ],
        allowed_fields={"context", "options"},
    )

    assert filters == {
        "context": {"key1": "xxx"},
        "options": {"key2": "yyy"},
    }


def test_parse_dotted_json_filters_uses_last_repeated_value():
    """Repeated query params for the same path use the last value."""
    filters = queries.parse_dotted_json_filters(
        [
            ("context.key1", "first"),
            ("context.key1", "second"),
        ],
        allowed_fields={"context"},
    )

    assert filters == {"context": {"key1": "second"}}


def test_parse_dotted_json_filters_rejects_bare_root():
    """A bare JSON root is not a valid dotted filter."""
    with pytest.raises(
        queries.InvalidDottedJsonFilterError,
        match=r"context\.<key>",
    ):
        queries.parse_dotted_json_filters(
            [("context", "xxx")],
            allowed_fields={"context"},
        )


def test_parse_dotted_json_filters_rejects_empty_key():
    """A dotted filter must include a non-empty top-level key."""
    with pytest.raises(queries.InvalidDottedJsonFilterError, match="top-level"):
        queries.parse_dotted_json_filters(
            [("context.", "xxx")],
            allowed_fields={"context"},
        )


def test_parse_dotted_json_filters_rejects_nested_key():
    """Only one level below the JSON root is supported."""
    with pytest.raises(queries.InvalidDottedJsonFilterError, match="top-level"):
        queries.parse_dotted_json_filters(
            [("context.profile.uuid", "xxx")],
            allowed_fields={"context"},
        )


def test_parse_dotted_json_filters_rejects_unknown_root():
    """Endpoints must explicitly whitelist each JSON root field."""
    with pytest.raises(queries.InvalidDottedJsonFilterError, match="not supported"):
        queries.parse_dotted_json_filters(
            [("options.key2", "xxx")],
            allowed_fields={"context"},
        )


def test_apply_dotted_json_filters_adds_json_predicates():
    """Grouped filters can be applied to SQLAlchemy statements."""
    statement = queries.apply_dotted_json_filters(
        select(QueryTestModel),
        filters={
            "context": {"key1": "xxx"},
            "options": {"key2": "yyy"},
        },
        field_map={
            "context": QueryTestModel.context,
            "options": QueryTestModel.options,
        },
    )

    compiled = str(statement.compile(compile_kwargs={"literal_binds": True}))

    assert "context" in compiled
    assert "key1" in compiled
    assert "xxx" in compiled
    assert "options" in compiled
    assert "key2" in compiled
    assert "yyy" in compiled
