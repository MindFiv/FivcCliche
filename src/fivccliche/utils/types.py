from typing import Any


class UnsetType:
    """Sentinel type for omitted optional update fields."""

    def __repr__(self) -> str:
        return "UNSET"


UNSET = UnsetType()

_TRUE_STRINGS = {"true", "1", "yes", "on"}
_FALSE_STRINGS = {"false", "0", "no", "off"}


def _is_missing(value: Any) -> bool:
    return value is None or value == ""


def to_optional_bool(value: Any, default: bool | None = None) -> bool | None:
    """Parse a bool; missing or unrecognized values return ``default``."""
    if isinstance(value, bool):
        return value
    if _is_missing(value):
        return default
    text = str(value).strip().lower()
    if text in _TRUE_STRINGS:
        return True
    if text in _FALSE_STRINGS:
        return False
    return default


def to_bool(value: Any, default: bool) -> bool:
    """Parse a bool; missing or unrecognized values return ``default``."""
    parsed = to_optional_bool(value, default)
    return default if parsed is None else parsed


def to_optional_int(value: Any, default: int | None = None) -> int | None:
    """Parse an int (accepts ``"15"`` / ``"15.0"``); missing or invalid values return ``default``."""
    if _is_missing(value):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int) -> int:
    """Parse an int; missing or invalid values return ``default``."""
    parsed = to_optional_int(value, default)
    return default if parsed is None else parsed


def to_optional_float(value: Any, default: float | None = None) -> float | None:
    """Parse a float; missing or invalid values return ``default``."""
    if _is_missing(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float) -> float:
    """Parse a float; missing or invalid values return ``default``."""
    parsed = to_optional_float(value, default)
    return default if parsed is None else parsed


def to_optional_string(value: Any, default: str | None = None) -> str | None:
    """Parse a stripped string; missing or blank values return ``default``."""
    if _is_missing(value):
        return default
    text = str(value).strip()
    return default if text == "" else text


def to_string(value: Any, default: str) -> str:
    """Parse a stripped string; missing or blank values return ``default``."""
    parsed = to_optional_string(value, default)
    return default if parsed is None else parsed
