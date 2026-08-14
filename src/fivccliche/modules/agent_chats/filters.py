from __future__ import annotations

import operator

from fivccliche.utils.filters import FilterJsonField, FilterSet, FilterSimpleField

from . import models


class ChatFilterSet(FilterSet):
    """HTTP query filters for listing chat sessions."""

    def __init__(self) -> None:
        super().__init__(
            [
                FilterSimpleField("agent_id", models.UserChat.agent_id, operator.eq),
                FilterJsonField("context", models.UserChat.context, operator.eq),
            ]
        )
