from cas import CASClientBase
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    responses,
)
from sqlalchemy.ext.asyncio import AsyncSession

from fivccliche.services.interfaces.auth import IUser
from fivccliche.utils.deps import (
    get_cas_client_async,
    get_authenticated_user_optional_async,
    get_db_session_async,
    default_auth,
)


router = APIRouter(prefix="/sso/ctrip", tags=["sso-ctrip"])


@router.get(
    "/login",
    summary="Login with Ctrip SSO.",
)
async def login(
    next: str | None = None,  # noqa
    ticket: str | None = None,
    user: IUser = Depends(get_authenticated_user_optional_async),
    cas_client: CASClientBase = Depends(get_cas_client_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    # check if user is already logged in
    if user:
        return responses.RedirectResponse(next)

    # next = request.args.get('next')
    # ticket = request.args.get('ticket')
    if not ticket:
        return responses.RedirectResponse(cas_client.get_login_url())

    # There is a ticket, the request come from CAS as callback.
    # need call `verify_ticket()` to validate ticket and get user profile.
    print("ticket: %s", ticket)
    print("next: %s", next)

    user, attributes, pgtiou = cas_client.verify_ticket(ticket)

    print(
        "CAS verify ticket response: user: %s, attributes: %s, pgtiou: %s", user, attributes, pgtiou
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid ticket",
        )

    print("user: %s", user)

    # Create or get user and generate credential
    credential = await default_auth.create_sso_credential_async(
        username=user,
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
