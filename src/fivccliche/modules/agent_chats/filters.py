from __future__ import annotations

import operator

from fivccliche.utils.filters import (
    FilterEditableField,
    FilterJsonField,
    FilterReadableField,
    FilterSet,
    FilterSimpleField,
)

from . import models


class ChatFilterSet(FilterSet):
    """HTTP query filters for listing chat sessions (includes readable ownership)."""

    def __init__(self, user_uuid: str, *, is_superuser: bool) -> None:
        super().__init__(
            [
                FilterReadableField(
                    "readable",
                    models.UserChat.user_uuid,
                    user_uuid,
                    is_superuser=is_superuser,
                ),
                FilterSimpleField("agent_id", models.UserChat.agent_id, operator.eq),
                FilterSimpleField("created_at_from", models.UserChat.created_at, operator.ge),
                FilterSimpleField("created_at_to", models.UserChat.created_at, operator.le),
                FilterSimpleField("updated_at_from", models.UserChat.updated_at, operator.ge),
                FilterSimpleField("updated_at_to", models.UserChat.updated_at, operator.le),
                FilterJsonField("context", models.UserChat.context, operator.eq),
            ]
        )


class ChatEditableFilterSet(FilterSet):
    """Editable ownership only for chat update/delete."""

    def __init__(self, user_uuid: str, *, is_superuser: bool) -> None:
        super().__init__(
            [
                FilterEditableField(
                    "editable",
                    models.UserChat.user_uuid,
                    user_uuid,
                    is_superuser=is_superuser,
                ),
            ]
        )
