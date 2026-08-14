from fivccliche.utils.types import (
    to_bool,
    to_float,
    to_int,
    to_optional_bool,
    to_optional_float,
    to_optional_int,
    to_optional_string,
    to_string,
)


class TestToBool:
    def test_bool_passthrough(self):
        assert to_bool(True, False) is True
        assert to_bool(False, True) is False

    def test_true_strings(self):
        for value in ("true", "TRUE", "1", "yes", "on"):
            assert to_bool(value, False) is True

    def test_false_strings(self):
        for value in ("false", "FALSE", "0", "no", "off"):
            assert to_bool(value, True) is False

    def test_missing_or_unrecognized_returns_default(self):
        assert to_bool(None, True) is True
        assert to_bool("", False) is False
        assert to_bool("maybe", True) is True


class TestToOptionalBool:
    def test_missing_returns_default(self):
        assert to_optional_bool(None) is None
        assert to_optional_bool("", False) is False
        assert to_optional_bool("nope") is None


class TestToInt:
    def test_int_and_numeric_strings(self):
        assert to_int(2, 0) == 2
        assert to_int("10", 0) == 10
        assert to_int("15.0", 0) == 15

    def test_missing_or_invalid_returns_default(self):
        assert to_int(None, 7) == 7
        assert to_int("", 7) == 7
        assert to_int("abc", 7) == 7


class TestToOptionalInt:
    def test_parses_int_and_numeric_string(self):
        assert to_optional_int(2) == 2
        assert to_optional_int("10") == 10
        assert to_optional_int("15.0") == 15

    def test_returns_none_for_missing_or_invalid(self):
        assert to_optional_int(None) is None
        assert to_optional_int("") is None
        assert to_optional_int("abc") is None

    def test_custom_default(self):
        assert to_optional_int("abc", 3) == 3


class TestToFloat:
    def test_float_and_numeric_strings(self):
        assert to_float(1.5, 0.0) == 1.5
        assert to_float("12", 0.0) == 12.0
        assert to_float("120.0", 0.0) == 120.0

    def test_missing_or_invalid_returns_default(self):
        assert to_float(None, 300.0) == 300.0
        assert to_float("", 300.0) == 300.0
        assert to_float("bad", 300.0) == 300.0


class TestToOptionalFloat:
    def test_missing_returns_default(self):
        assert to_optional_float(None) is None
        assert to_optional_float("x") is None
        assert to_optional_float("1.25") == 1.25


class TestToString:
    def test_strips_and_stringifies(self):
        assert to_string("  HS256  ", "x") == "HS256"
        assert to_string(256, "x") == "256"

    def test_missing_or_blank_returns_default(self):
        assert to_string(None, "HS256") == "HS256"
        assert to_string("", "HS256") == "HS256"
        assert to_string("   ", "HS256") == "HS256"


class TestToOptionalString:
    def test_missing_returns_default(self):
        assert to_optional_string(None) is None
        assert to_optional_string("  ") is None
        assert to_optional_string("ok") == "ok"
