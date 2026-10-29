"""HTTP and ownership filters for storage files."""

from __future__ import annotations

import operator

from fivccliche.utils.filters import (
    FilterEditableField,
    FilterSimpleField,
    FilterSet,
)

from . import models


class StorageFileFilterSet(FilterSet):
    """Owned-file list filters; storage rows are never global."""

    def __init__(self, user_uuid: str) -> None:
        super().__init__(
            [
                FilterEditableField(
                    "owner",
                    models.StorageFile.user_uuid,
                    user_uuid,
                    is_superuser=False,
                ),
                FilterSimpleField("status", models.StorageFile.status, operator.eq),
                FilterSimpleField("is_public", models.StorageFile.is_public, operator.eq),
                FilterSimpleField("content_type", models.StorageFile.content_type, operator.eq),
                FilterSimpleField("created_at_from", models.StorageFile.created_at, operator.ge),
                FilterSimpleField("created_at_to", models.StorageFile.created_at, operator.le),
            ]
        )


class StorageFileOwnedFilterSet(FilterSet):
    """Ownership predicate for one known storage file."""

    def __init__(self, user_uuid: str) -> None:
        super().__init__(
            [
                FilterEditableField(
                    "editable",
                    models.StorageFile.user_uuid,
                    user_uuid,
                    is_superuser=False,
                ),
            ]
        )
