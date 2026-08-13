from fastapi import HTTPException, status

from fivccliche.services.interfaces.auth import IUser


def assert_user_owns_resource(
    user: IUser,
    owner_uuid: str | None,
    *,
    global_detail: str,
    other_detail: str,
) -> None:
    """Raise 403 unless the user owns ``owner_uuid`` (or is a superuser on globals)."""
    if owner_uuid is None and not user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=global_detail)
    if owner_uuid is not None and owner_uuid != user.uuid:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=other_detail)
