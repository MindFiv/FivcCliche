from __future__ import annotations

import operator
from typing import Any

from fivccliche.utils.filters import (
    FilterEditableField,
    FilterReadableField,
    FilterSet,
    FilterSimpleField,
)

from . import models


class UserScopedReadableFilterSet(FilterSet):
    """Readable ownership only for user-scoped config get/list/count."""

    def __init__(self, col: Any, user_uuid: str, *, is_superuser: bool) -> None:
        super().__init__(
            [
                FilterReadableField("readable", col, user_uuid, is_superuser=is_superuser),
            ]
        )


class UserScopedEditableFilterSet(FilterSet):
    """Editable ownership only for user-scoped config update/delete."""

    def __init__(self, col: Any, user_uuid: str, *, is_superuser: bool) -> None:
        super().__init__(
            [
                FilterEditableField("editable", col, user_uuid, is_superuser=is_superuser),
            ]
        )


class QuestionFilterSet(FilterSet):
    """HTTP query filters for listing question configs (includes readable ownership)."""

    def __init__(self, user_uuid: str, *, is_superuser: bool) -> None:
        super().__init__(
            [
                FilterReadableField(
                    "readable",
                    models.UserQuestion.user_uuid,
                    user_uuid,
                    is_superuser=is_superuser,
                ),
                FilterSimpleField("is_active", models.UserQuestion.is_active, operator.eq),
            ]
        )
