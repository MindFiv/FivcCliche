from __future__ import annotations

import operator

from fivccliche.utils.filters import FilterSet, FilterSimpleField

from . import models


class QuestionFilterSet(FilterSet):
    """HTTP query filters for listing question configs."""

    def __init__(self) -> None:
        super().__init__(
            [
                FilterSimpleField("is_active", models.UserQuestion.is_active, operator.eq),
            ]
        )
