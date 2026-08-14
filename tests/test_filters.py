"""Tests for FilterSet HTTP query-to-SQL binding."""

import operator

import pytest
from sqlalchemy import JSON, Column, String, select
from sqlalchemy.orm import declarative_base

from fivccliche.utils.filters import (
    FilterEditableField,
    FilterError,
    FilterField,
    FilterJsonField,
    FilterReadableField,
    FilterSet,
    FilterSimpleField,
)

Base = declarative_base()


class QueryTestModel(Base):
    """Minimal SQLAlchemy model for compiling filter statements."""

    __tablename__ = "query_test_model"

    uuid = Column(String, primary_key=True)
    agent_id = Column(String)
    context = Column(JSON)
    options = Column(JSON)


def make_filterset() -> FilterSet:
    return FilterSet(
        [
            FilterSimpleField("agent_id", QueryTestModel.agent_id, operator.eq),
            FilterJsonField("context", QueryTestModel.context, operator.eq),
            FilterJsonField("options", QueryTestModel.options, operator.eq),
        ]
    )


def compile_statement(statement: object) -> str:
    return str(statement.compile(compile_kwargs={"literal_binds": True}))


def test_parse_groups_dotted_json_values_by_root_field():
    fs = make_filterset()
    fs.parse(**{"context.key1": "xxx", "context.key2": "yyy"})
    compiled = compile_statement(fs.filter(select(QueryTestModel)))

    assert "key1" in compiled
    assert "xxx" in compiled
    assert "key2" in compiled
    assert "yyy" in compiled


def test_parse_supports_multiple_json_roots():
    fs = make_filterset()
    fs.parse(**{"context.key1": "xxx", "options.key2": "yyy"})
    compiled = compile_statement(fs.filter(select(QueryTestModel)))

    assert "context" in compiled
    assert "key1" in compiled
    assert "xxx" in compiled
    assert "options" in compiled
    assert "key2" in compiled
    assert "yyy" in compiled


def test_parse_accepts_grouped_json_mapping():
    fs = make_filterset()
    fs.parse(context={"key1": "xxx", "key2": "yyy"})
    compiled = compile_statement(fs.filter(select(QueryTestModel)))

    assert "key1" in compiled
    assert "xxx" in compiled
    assert "key2" in compiled
    assert "yyy" in compiled


def test_filter_rejects_bare_json_root():
    fs = make_filterset()
    fs.parse(context="xxx")
    with pytest.raises(FilterError, match=r"context\.<key>"):
        fs.filter(select(QueryTestModel))


def test_parse_rejects_empty_json_key():
    with pytest.raises(FilterError, match="top-level"):
        make_filterset().parse(**{"context.": "xxx"})


def test_parse_rejects_nested_json_key():
    with pytest.raises(FilterError, match="top-level"):
        make_filterset().parse(**{"context.profile.uuid": "xxx"})


def test_parse_rejects_unknown_json_root():
    with pytest.raises(FilterError, match="not supported"):
        FilterSet([FilterJsonField("context", QueryTestModel.context, operator.eq)]).parse(
            **{"options.key2": "xxx"}
        )


def test_parse_ignores_undeclared_scalar_params():
    base = select(QueryTestModel)
    fs = make_filterset()
    fs.parse(skip="0", limit="10")
    filtered = fs.filter(base)

    assert compile_statement(filtered) == compile_statement(base)


def test_filter_applies_exact_column_predicate():
    fs = make_filterset()
    fs.parse(agent_id="agent_1")
    compiled = compile_statement(fs.filter(select(QueryTestModel)))

    assert "agent_1" in compiled


def test_filter_skips_empty_exact_value():
    base = select(QueryTestModel)
    fs = make_filterset()
    fs.parse(agent_id="")
    filtered = fs.filter(base)

    assert compile_statement(filtered) == compile_statement(base)


def test_filter_without_parse_returns_statement_unchanged():
    base = select(QueryTestModel)
    filtered = make_filterset().filter(base)

    assert compile_statement(filtered) == compile_statement(base)


def test_filter_adds_json_predicates():
    fs = make_filterset()
    fs.parse(**{"context.key1": "xxx", "options.key2": "yyy"})
    compiled = compile_statement(fs.filter(select(QueryTestModel)))

    assert "context" in compiled
    assert "key1" in compiled
    assert "xxx" in compiled
    assert "options" in compiled
    assert "key2" in compiled
    assert "yyy" in compiled


def test_fields_property_is_read_only_tuple():
    fs = make_filterset()
    assert isinstance(fs.fields, tuple)
    assert [field.name for field in fs.fields] == ["agent_id", "context", "options"]


class FixedOwnerField(FilterField):
    """Fixed predicate that does not depend on query params."""

    def __init__(self, owner_uuid: str) -> None:
        self._owner_uuid = owner_uuid

    @property
    def name(self) -> str:
        return "owner"

    def filter(self, statement: object) -> object:
        return statement.where(QueryTestModel.uuid == self._owner_uuid)


def test_fixed_field_applies_without_parse():
    fs = FilterSet([FixedOwnerField("user-1")])
    compiled = compile_statement(fs.filter(select(QueryTestModel)))

    assert "user-1" in compiled


def test_fixed_field_combines_with_simple_field():
    fs = FilterSet(
        [
            FixedOwnerField("user-1"),
            FilterSimpleField("agent_id", QueryTestModel.agent_id, operator.eq),
        ]
    )
    fs.parse(agent_id="agent_1")
    compiled = compile_statement(fs.filter(select(QueryTestModel)))

    assert "user-1" in compiled
    assert "agent_1" in compiled


class OwnershipTestModel(Base):
    """Model with a nullable owner column for readable/editable fields."""

    __tablename__ = "ownership_test_model"

    uuid = Column(String, primary_key=True)
    user_uuid = Column(String)


def test_readable_field_includes_own_and_global_for_regular_user():
    fs = FilterSet(
        [
            FilterReadableField(
                "readable",
                OwnershipTestModel.user_uuid,
                "user-1",
                is_superuser=False,
            )
        ]
    )
    compiled = compile_statement(fs.filter(select(OwnershipTestModel)))

    assert "user-1" in compiled
    assert "IS NULL" in compiled.upper() or " = NULL" in compiled.upper() or "IS NULL" in compiled


def test_readable_field_same_predicate_for_superuser():
    regular = compile_statement(
        FilterSet(
            [
                FilterReadableField(
                    "readable",
                    OwnershipTestModel.user_uuid,
                    "user-1",
                    is_superuser=False,
                )
            ]
        ).filter(select(OwnershipTestModel))
    )
    admin = compile_statement(
        FilterSet(
            [
                FilterReadableField(
                    "readable",
                    OwnershipTestModel.user_uuid,
                    "user-1",
                    is_superuser=True,
                )
            ]
        ).filter(select(OwnershipTestModel))
    )

    assert regular == admin


def test_editable_field_own_only_for_regular_user():
    fs = FilterSet(
        [
            FilterEditableField(
                "editable",
                OwnershipTestModel.user_uuid,
                "user-1",
                is_superuser=False,
            )
        ]
    )
    compiled = compile_statement(fs.filter(select(OwnershipTestModel)))

    assert "user-1" in compiled
    assert "IS NULL" not in compiled.upper()


def test_editable_field_own_or_global_for_superuser():
    fs = FilterSet(
        [
            FilterEditableField(
                "editable",
                OwnershipTestModel.user_uuid,
                "user-1",
                is_superuser=True,
            )
        ]
    )
    compiled = compile_statement(fs.filter(select(OwnershipTestModel)))

    assert "user-1" in compiled
    assert "IS NULL" in compiled.upper() or "NULL" in compiled.upper()
