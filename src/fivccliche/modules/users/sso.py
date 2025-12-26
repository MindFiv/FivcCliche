from cas import CASClient
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    responses,
)
from sqlalchemy.ext.asyncio import AsyncSession

from fivccliche.services.interfaces.auth import IUser, IUserAuthenticator
from fivccliche.utils.deps import (
    get_authenticator_async,
    get_authenticated_user_optional_async,
    get_db_session_async,
    get_config_async,
    configs,
)

router = APIRouter(prefix="/sso", tags=["sso"])


@router.get(
    "/login",
    summary="Login with SSO.",
)
async def login(
    next: str | None = None,  # noqa
    ticket: str | None = None,
    user: IUser = Depends(get_authenticated_user_optional_async),
    auth: IUserAuthenticator = Depends(get_authenticator_async),
    config: configs.IConfig = Depends(get_config_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    # check if user is already logged in
    if user:
        return responses.RedirectResponse(next)

    config_sess = config.get_session("cas")
    cas_client = CASClient(
        version=config_sess.get_value("VERSION"),
        service_url=config_sess.get_value("SERVICE_URL"),
        server_url=config_sess.get_value("SERVER_URL"),
        verify_ssl_certificate=bool(config_sess.get_value("VERIFY_SSL_CERTIFICATE")),
    )
    if not ticket:
        return responses.RedirectResponse(cas_client.get_login_url())

    username, attributes, _ = cas_client.verify_ticket(ticket)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid ticket",
        )

    # Create or get user and generate credential
    credential = await auth.create_sso_credential_async(
        username=username,
        attributes=attributes or {},
        session=session,
    )

    if not credential:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user credential",
        )

    # TODO: Set the access token in a cookie or return it in the response
    # For now, just redirect to the next URL
    return responses.RedirectResponse(next)
