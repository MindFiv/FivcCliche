"""HTTP and ownership filters for storage files."""

from __future__ import annotations

import operator

from fivccliche.utils.filters import (
    FilterEditableField,
    FilterReadableField,
    FilterSimpleField,
    FilterSet,
)

from . import models


class StorageFileFilterSet(FilterSet):
    """List filters including readable ownership (owned or global)."""

    def __init__(self, user_uuid: str, *, is_superuser: bool) -> None:
        super().__init__(
            [
                FilterReadableField(
                    "readable",
                    models.StorageFile.user_uuid,
                    user_uuid,
                    is_superuser=is_superuser,
                ),
                FilterSimpleField("status", models.StorageFile.status, operator.eq),
                FilterSimpleField("is_public", models.StorageFile.is_public, operator.eq),
                FilterSimpleField("content_type", models.StorageFile.content_type, operator.eq),
                FilterSimpleField("created_at_from", models.StorageFile.created_at, operator.ge),
                FilterSimpleField("created_at_to", models.StorageFile.created_at, operator.le),
            ]
        )


class StorageFileReadableFilterSet(FilterSet):
    """Readable ownership for get / download-url."""

    def __init__(self, user_uuid: str, *, is_superuser: bool) -> None:
        super().__init__(
            [
                FilterReadableField(
                    "readable",
                    models.StorageFile.user_uuid,
                    user_uuid,
                    is_superuser=is_superuser,
                ),
            ]
        )


class StorageFileEditableFilterSet(FilterSet):
    """Editable ownership for complete / delete."""

    def __init__(self, user_uuid: str, *, is_superuser: bool) -> None:
        super().__init__(
            [
                FilterEditableField(
                    "editable",
                    models.StorageFile.user_uuid,
                    user_uuid,
                    is_superuser=is_superuser,
                ),
            ]
        )
