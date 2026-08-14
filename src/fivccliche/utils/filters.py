from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any

FilterColumnOperator = Callable[[Any, Any], Any]


class FilterError(ValueError):
    """Raised when a query filter is malformed or unsupported."""


class FilterField(ABC):
    """Bind filter state to a SQLAlchemy statement.

    Subclasses must implement ``name`` and ``filter``. Query-bound fields
    override ``parse`` to store values; fixed fields may set state in
    ``__init__`` and leave ``parse`` as a no-op.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identity / query param root name for this field."""

    def parse(self, value: Any) -> None:
        """Store a value for later ``filter``. Default is a no-op."""
        return None

    @abstractmethod
    def filter(self, statement: Any) -> Any:
        """Apply this field's predicate to ``statement``, or return it unchanged."""


class FilterSimpleField(FilterField):
    """Exact match on a scalar column when a non-empty value was parsed."""

    def __init__(
        self,
        name: str,
        col: Any,
        op: FilterColumnOperator = operator.eq,
    ) -> None:
        self._name = name
        self.col = col
        self.op = op
        self._value: Any = None

    @property
    def name(self) -> str:
        return self._name

    def parse(self, value: Any) -> None:
        self._value = value

    def filter(self, statement: Any) -> Any:
        if self._value is None or self._value == "":
            return statement
        return statement.where(self.op(self.col, self._value))


class FilterJsonField(FilterField):
    """Match top-level keys on a JSON column from a parsed mapping."""

    def __init__(
        self,
        name: str,
        col: Any,
        op: FilterColumnOperator = operator.eq,
    ) -> None:
        self._name = name
        self.col = col
        self.op = op
        self._value: Any = None

    @property
    def name(self) -> str:
        return self._name

    def parse(self, value: Any) -> None:
        self._value = value

    def filter(self, statement: Any) -> Any:
        if self._value is None:
            return statement
        if not isinstance(self._value, Mapping):
            raise FilterError(f"JSON filters must use {self.name}.<key> query parameters")
        for json_key, json_value in self._value.items():
            statement = statement.where(
                self.op(self.col[str(json_key)].as_string(), str(json_value))
            )
        return statement


class FilterReadableField(FilterField):
    """Rows the user may read: owned or global (same for regular and superuser today)."""

    def __init__(
        self,
        name: str,
        col: Any,
        user_uuid: str,
        *,
        is_superuser: bool,
    ) -> None:
        self._name = name
        self.col = col
        self.user_uuid = user_uuid
        self.is_superuser = is_superuser

    @property
    def name(self) -> str:
        return self._name

    def filter(self, statement: Any) -> Any:
        # is_superuser is accepted for API symmetry with Editable; predicate matches
        # current list/get visibility for both roles (own or global, not others').
        return statement.where((self.col == self.user_uuid) | (self.col == None))  # noqa: E711


class FilterEditableField(FilterField):
    """Rows the user may edit: owned only; superusers may also edit globals."""

    def __init__(
        self,
        name: str,
        col: Any,
        user_uuid: str,
        *,
        is_superuser: bool,
    ) -> None:
        self._name = name
        self.col = col
        self.user_uuid = user_uuid
        self.is_superuser = is_superuser

    @property
    def name(self) -> str:
        return self._name

    def filter(self, statement: Any) -> Any:
        if self.is_superuser:
            return statement.where((self.col == self.user_uuid) | (self.col == None))  # noqa: E711
        return statement.where(self.col == self.user_uuid)


class FilterSet:
    """Parse HTTP query params and apply them as SQLAlchemy WHERE clauses."""

    def __init__(self, fields: Sequence[FilterField]) -> None:
        self._fields: tuple[FilterField, ...] = tuple(fields)

    @property
    def fields(self) -> tuple[FilterField, ...]:
        return self._fields

    def parse(self, **query_params: Any) -> None:
        json_roots = {field.name for field in self._fields if isinstance(field, FilterJsonField)}
        for query_key in query_params:
            root, separator, json_key = query_key.partition(".")
            if not separator:
                continue
            if root not in json_roots:
                raise FilterError(f"JSON filter root '{root}' is not supported")
            if not json_key or "." in json_key:
                raise FilterError(f"JSON filters only support top-level keys under '{root}'")

        for field in self._fields:
            if isinstance(field, FilterJsonField):
                grouped: dict[str, Any] = {}
                bare = query_params.get(field.name)
                for query_key, query_value in query_params.items():
                    root, separator, json_key = query_key.partition(".")
                    if separator and root == field.name and json_key:
                        grouped[json_key] = query_value
                if bare is not None:
                    field.parse(bare)
                elif grouped:
                    field.parse(grouped)
            elif field.name in query_params:
                field.parse(query_params[field.name])

    def filter(self, statement: Any) -> Any:
        for field in self._fields:
            statement = field.filter(statement)
        return statement
