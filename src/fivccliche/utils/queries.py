from collections.abc import Collection, Iterable, Mapping
from typing import Any


class InvalidDottedJsonFilterError(ValueError):
    """Raised when a dotted JSON query filter is malformed or unsupported."""


def parse_dotted_json_filters(
    query_items: Iterable[tuple[str, str]],
    *,
    allowed_fields: Collection[str],
) -> dict[str, dict[str, str]]:
    """Parse dotted query params into grouped JSON field filters.

    Only one level below each allowed JSON root is supported:
    ``context.profile_uuid=xxx`` becomes ``{"context": {"profile_uuid": "xxx"}}``.
    Repeated paths use the last value.
    """
    filters: dict[str, dict[str, str]] = {}

    for query_key, query_value in query_items:
        root, separator, json_key = query_key.partition(".")
        is_allowed_root = root in allowed_fields

        if query_key in allowed_fields:
            raise InvalidDottedJsonFilterError(
                f"JSON filters must use {query_key}.<key> query parameters"
            )

        if not separator:
            continue

        if not is_allowed_root:
            raise InvalidDottedJsonFilterError(f"JSON filter root '{root}' is not supported")

        if not json_key or "." in json_key:
            raise InvalidDottedJsonFilterError(
                f"JSON filters only support top-level keys under '{root}'"
            )

        filters.setdefault(root, {})[json_key] = query_value

    return filters


def apply_dotted_json_filters(
    statement: Any,
    *,
    filters: Mapping[str, Mapping[str, str]] | None,
    field_map: Mapping[str, Any],
) -> Any:
    """Apply exact string predicates for grouped JSON filters to a statement."""
    for field_name, field_filters in (filters or {}).items():
        column = field_map[field_name]
        for key, value in field_filters.items():
            statement = statement.where(column[key].as_string() == value)
    return statement
