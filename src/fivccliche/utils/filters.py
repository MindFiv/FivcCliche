from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any

FilterColumnOperator = Callable[[Any, Any], Any]


class FilterError(ValueError):
    """Raised when a query filter is malformed or unsupported."""


class FilterField(ABC):
    """Bind a query param key to a column predicate.

    Subclasses must implement ``name`` and ``filter``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Query param root name for this field."""

    @abstractmethod
    def filter(self, statement: Any, key: str, value: Any) -> Any:
        """Apply this field's predicate for ``key``/``value`` to ``statement``."""


class FilterSimpleField(FilterField):
    """Exact match on a scalar column when ``key`` equals ``name``."""

    def __init__(
        self,
        name: str,
        col: Any,
        op: FilterColumnOperator = operator.eq,
    ) -> None:
        self._name = name
        self.col = col
        self.op = op

    @property
    def name(self) -> str:
        return self._name

    def filter(self, statement: Any, key: str, value: Any) -> Any:
        if key != self.name:
            return statement
        if value is None or value == "":
            return statement
        return statement.where(self.op(self.col, value))


class FilterJsonField(FilterField):
    """Match ``name.key=value`` or a grouped ``name={...}`` mapping on a JSON column."""

    def __init__(
        self,
        name: str,
        col: Any,
        op: FilterColumnOperator = operator.eq,
    ) -> None:
        self._name = name
        self.col = col
        self.op = op

    @property
    def name(self) -> str:
        return self._name

    def filter(self, statement: Any, key: str, value: Any) -> Any:
        if key == self.name:
            if not isinstance(value, Mapping):
                raise FilterError(f"JSON filters must use {self.name}.<key> query parameters")
            for json_key, json_value in value.items():
                statement = statement.where(
                    self.op(self.col[str(json_key)].as_string(), str(json_value))
                )
            return statement

        root, _separator, json_key = key.partition(".")
        if root != self.name or not json_key:
            return statement
        if value is None or value == "":
            return statement
        return statement.where(self.op(self.col[json_key].as_string(), str(value)))


class FilterSet:
    """Parse HTTP query params and apply them as SQLAlchemy WHERE clauses."""

    def __init__(self, fields: Sequence[FilterField]) -> None:
        self._fields: tuple[FilterField, ...] = tuple(fields)
        self._params: dict[str, Any] | None = None

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

        self._params = dict(query_params)

    def filter(self, statement: Any) -> Any:
        if self._params is None:
            return statement
        for key, value in self._params.items():
            field = self._resolve_field(key)
            if field is None:
                continue
            statement = field.filter(statement, key, value)
        return statement

    def _resolve_field(self, key: str) -> FilterField | None:
        for field in self._fields:
            if isinstance(field, FilterJsonField):
                if key == field.name or key.startswith(f"{field.name}."):
                    return field
            elif key == field.name:
                return field
        return None
