from typing import cast
from collections.abc import AsyncGenerator

from cas import CASClient, CASClientBase
from fastapi import status, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fivcglue import query_component, IComponentSite, LazyValue
from fivcglue.interfaces import configs

from fivccliche.services.interfaces.agent_chats import IUserChatProvider
from fivccliche.services.interfaces.agent_configs import IUserConfigProvider
from fivccliche.services.interfaces.db import IDatabase
from fivccliche.services.interfaces.auth import IUser, IUserAuthenticator
from fivccliche.services.implements import service_site
from sqlalchemy.ext.asyncio.session import AsyncSession

default_security = HTTPBearer()
default_security_optional = HTTPBearer(auto_error=False)

default_config: LazyValue[configs.IConfig] = LazyValue(
    lambda: query_component(cast(IComponentSite, service_site), configs.IConfig)
)

default_db: LazyValue[IDatabase] = LazyValue(
    lambda: query_component(cast(IComponentSite, service_site), IDatabase)
)

default_auth: LazyValue[IUserAuthenticator] = LazyValue(
    lambda: query_component(cast(IComponentSite, service_site), IUserAuthenticator)
)

default_config_provider: LazyValue[IUserConfigProvider] = LazyValue(
    lambda: query_component(cast(IComponentSite, service_site), IUserConfigProvider)
)

default_chat_provider: LazyValue[IUserChatProvider] = LazyValue(
    lambda: query_component(cast(IComponentSite, service_site), IUserChatProvider)
)


async def get_cas_client_async() -> CASClientBase:
    """Get the CAS client for dependency injection."""
    sess = default_config.get_session("cas")
    return CASClient(
        version=sess.get_value("VERSION"),
        service_url=sess.get_value("SERVICE_URL"),
        server_url=sess.get_value("SERVER_URL"),
        verify_ssl_certificate=bool(sess.get_value("VERIFY_SSL_CERTIFICATE")),
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
) -> IUser:
    """Get the user authenticator for dependency injection."""
    auth = default_auth
    user = await auth.verify_credential_async(
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


async def get_authenticated_user_optional_async(
    credentials: HTTPAuthorizationCredentials = Depends(default_security_optional),
    session: AsyncSession = Depends(get_db_session_async),
) -> IUser | None:
    if not credentials:
        return None

    return await default_auth.verify_credential_async(
        credentials.credentials,
        session=session,
    )


async def get_admin_user_async(
    user: IUser = Depends(get_authenticated_user_async),
):
    """Get the admin user for dependency injection."""
    if not user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not a super user",
        )
    return user


async def get_config_provider_async() -> IUserConfigProvider:
    """Get the user config provider for dependency injection."""
    return default_config_provider()


async def get_chat_provider_async() -> IUserChatProvider:
    """Get the user chat provider for dependency injection."""
    return default_chat_provider()
