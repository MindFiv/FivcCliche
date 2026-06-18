from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic_strict_partial import create_partial_model
from sqlalchemy.ext.asyncio import AsyncSession

from fivccliche.services.interfaces.auth import IUser
from fivccliche.utils.deps import (
    get_db_session_async,
    get_authenticated_user_async,
    get_admin_user_async,
    default_auth,
)
from fivccliche.utils.schemas import PaginatedResponse

from . import methods, models, schemas

router = APIRouter(prefix="/users", tags=["users"])
UserSelfPatch = create_partial_model(schemas.UserSelfUpdate)


@router.post(
    "/",
    summary="Create a new user (admin only).",
    response_model=schemas.UserRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_user_async(
    user_create: schemas.UserCreate,
    admin_user: IUser = Depends(get_admin_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> models.User:
    """Create a new user."""
    if not admin_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not a admin",
        )
    # Check if username already exists
    existing_user = await methods.get_user_async(session, username=user_create.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    # Check if email already exists (only if email is provided)
    if user_create.email:
        existing_email = await methods.get_user_async(session, email=str(user_create.email))
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

    user = await methods.create_user_async(
        session,
        username=user_create.username,
        email=str(user_create.email) if user_create.email else None,
        full_name=user_create.full_name,
        password=user_create.password,
        preferences=user_create.preferences,
    )
    return user


@router.post(
    "/login/",
    summary="Authenticate a user and return JWT token.",
    response_model=schemas.UserLoginResponse,
)
async def login_user_async(
    user_login: schemas.UserLogin,
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.UserLoginResponse:
    """Authenticate a user and return user data with JWT token."""
    credential = await default_auth.create_credential_async(
        user_login.username,
        user_login.password,
        session=session,
    )
    if not credential:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    return schemas.UserLoginResponse(
        access_token=credential.access_token,
        expires_in=credential.expires_in,
    )


@router.get(
    "/self/",
    summary="Get the authenticated user's profile.",
    response_model=schemas.UserRead,
)
async def get_self_async(
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> models.User:
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    user = await methods.get_user_async(session, user_uuid=user.uuid)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return user


@router.patch(
    "/self/",
    summary="Update the authenticated user's profile.",
    response_model=schemas.UserRead,
)
async def update_self_async(
    data: UserSelfPatch,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> models.User:
    """Update the current user's profile."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    db_user = await methods.get_user_async(session, user_uuid=user.uuid)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    update_data = {field: getattr(data, field) for field in data.model_fields_set}
    return await methods.update_user_async(session, db_user, **update_data)


@router.patch(
    "/self/password/",
    summary="Change the authenticated user's password.",
    response_model=schemas.UserRead,
)
async def change_password_async(
    data: schemas.UserPasswordUpdate,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> models.User:
    """Change the current user's password."""
    try:
        return await methods.change_user_password_async(
            session, str(user.uuid), data.current_password, data.new_password
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.get(
    "/", summary="List all users (admin only).", response_model=PaginatedResponse[schemas.UserRead]
)
async def list_users_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    order_by: schemas.UserOrderBy = Query(schemas.UserOrderBy.created_at),
    order_dir: schemas.UserOrderDir = Query(schemas.UserOrderDir.asc),
    admin_user: IUser = Depends(get_admin_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse[schemas.UserRead]:
    """List all users with pagination."""
    if not admin_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not a admin",
        )
    users = await methods.list_users_async(
        session, skip=skip, limit=limit, order_by=order_by.value, order_dir=order_dir.value
    )
    total = await methods.count_users_async(session)
    return PaginatedResponse[schemas.UserRead](total=total, results=users)


@router.get(
    "/{user_uuid}/", summary="Get a user by ID (admin only).", response_model=schemas.UserRead
)
async def get_user_async(
    user_uuid: str,
    admin_user: IUser = Depends(get_admin_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> models.User:
    """Get a user by ID."""
    if not admin_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not a admin",
        )

    user = await methods.get_user_async(session, user_uuid=user_uuid)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return user


@router.delete(
    "/{user_uuid}/",
    summary="Delete a user by ID (admin only).",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_user_async(
    user_uuid: str,
    admin_user: IUser = Depends(get_admin_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    """Delete a user."""
    if not admin_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not a admin",
        )
    user = await methods.get_user_async(session, user_uuid=user_uuid)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    await methods.delete_user_async(session, user)


@router.patch(
    "/{user_uuid}/status/",
    summary="Update user active status (admin only).",
    response_model=schemas.UserRead,
)
async def update_user_status_async(
    user_uuid: str,
    status_update: schemas.UserStatusUpdate,
    admin_user: IUser = Depends(get_admin_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> models.User:
    """Update a user's active status."""
    user = await methods.get_user_async(session, user_uuid=user_uuid)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if user_uuid == admin_user.uuid and not status_update.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account",
        )

    if user.is_active == status_update.is_active:
        return user

    return await methods.update_user_async(session, user, is_active=status_update.is_active)


@router.get(
    "/{user_uuid}/impersonate/",
    summary="Impersonate a user (admin only).",
    response_model=schemas.UserLoginResponse,
)
async def impersonate_user_async(
    user_uuid: str,
    admin_user: IUser = Depends(get_admin_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.UserLoginResponse:
    if not admin_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not a admin",
        )
    user = await methods.get_user_async(session, user_uuid=user_uuid)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    credential = await default_auth.create_credential_async(
        user.username,
        "",
        session=session,
        ignore_password=True,
    )
    return schemas.UserLoginResponse(
        access_token=credential.access_token,
        expires_in=credential.expires_in,
    )
