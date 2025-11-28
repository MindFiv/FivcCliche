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


@router.post("/", response_model=schemas.UserRead, status_code=status.HTTP_201_CREATED)
async def create_user_async(
    user_create: schemas.UserCreate,
    session: AsyncSession = Depends(get_db_session_async),
) -> models.User:
    """Create a new user."""
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

    user = await methods.create_user_async(session, user_create)
    return user


@router.get("/", response_model=PaginatedResponse[schemas.UserRead])
async def list_users_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse[schemas.UserRead]:
    """List all users with pagination."""
    users = await methods.list_users_async(session, skip=skip, limit=limit)
    total = await methods.count_users_async(session)
    return PaginatedResponse[schemas.UserRead](total=total, results=users)


@router.get("/self", response_model=schemas.UserRead)
async def get_self_async(
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> models.User:
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    user = await methods.get_user_async(session, user_id=user.id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return user


@router.get("/{user_id}", response_model=schemas.UserRead)
async def get_user_async(
    user_id: str,
    admin_user: IUser = Depends(get_admin_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> models.User:
    """Get a user by ID."""
    if not admin_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not a admin",
        )

    user = await methods.get_user_async(session, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_async(
    user_id: str,
    admin_user: IUser = Depends(get_admin_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    """Delete a user."""
    if not admin_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not a admin",
        )
    user = await methods.get_user_async(session, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    await methods.delete_user_async(session, user)


@router.post("/login", response_model=schemas.UserLoginResponse)
async def login_user_async(
    user_login: schemas.UserLogin,
    session: AsyncSession = Depends(get_db_session_async),
) -> schemas.UserLoginResponse:
    """Authenticate a user and return user data with JWT token."""
    access_token = await default_auth.create_access_token_async(
        user_login.username,
        user_login.password,
        session=session,
    )
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    return schemas.UserLoginResponse(
        access_token=access_token,
        token_type="bearer",
    )
