from fastapi import APIRouter, Depends, HTTPException, Query, status
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

    # Check if email already exists
    existing_email = await methods.get_user_async(session, email=str(user_create.email))
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    user = await methods.create_user_async(
        session,
        username=user_create.username,
        email=str(user_create.email),
        password=user_create.password,
    )
    return user


@router.post(
    "/login",
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
    "/self",
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


@router.get(
    "/", summary="List all users (admin only).", response_model=PaginatedResponse[schemas.UserRead]
)
async def list_users_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    admin_user: IUser = Depends(get_admin_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse[schemas.UserRead]:
    """List all users with pagination."""
    if not admin_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not a admin",
        )
    users = await methods.list_users_async(session, skip=skip, limit=limit)
    total = await methods.count_users_async(session)
    return PaginatedResponse[schemas.UserRead](total=total, results=users)


@router.get(
    "/{user_uuid}", summary="Get a user by ID (admin only).", response_model=schemas.UserRead
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
    "/{user_uuid}",
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
