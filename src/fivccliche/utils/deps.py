from typing import cast
from collections.abc import AsyncGenerator

from fastapi import status, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fivcglue import query_component, IComponentSite, LazyValue
from fivccliche.services.interfaces.db import IDatabase
from fivccliche.services.interfaces.auth import IUser, IUserAuthenticator
from fivccliche.services.implements import service_site
from sqlalchemy.ext.asyncio.session import AsyncSession

default_security = HTTPBearer()

default_db: LazyValue[IDatabase] = LazyValue(
    lambda: query_component(cast(IComponentSite, service_site), IDatabase)
)

default_auth: LazyValue[IUserAuthenticator] = LazyValue(
    lambda: query_component(cast(IComponentSite, service_site), IUserAuthenticator)
)


async def get_db_session_async() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session for dependency injection."""
    db = default_db
    async_session = await db.get_session_async()
    try:
        yield async_session
    finally:
        await async_session.close()


async def get_authenticated_user_async(
    credentials: HTTPAuthorizationCredentials = Depends(default_security),
    session: AsyncSession = Depends(get_db_session_async),
) -> IUser | None:
    """Get the user authenticator for dependency injection."""
    auth = default_auth
    user = await auth.verify_access_token_async(
        credentials.credentials,
        session=session,
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_admin_user_async(
    user: IUser = Depends(get_authenticated_user_async),
):
    """Get the admin user for dependency injection."""
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not a super user",
        )
    return user
