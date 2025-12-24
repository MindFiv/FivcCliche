from cas import CASClientBase
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    responses,
)

from fivccliche.utils.deps import (
    get_cas_client_async,
    get_authenticated_user_async,
    # default_auth,
)


router = APIRouter(prefix="/sso/ctrip", tags=["sso-ctrip"])


@router.get(
    "/login",
    summary="Login with Ctrip SSO.",
)
async def login(
    next: str | None = None,  # noqa
    ticket: str | None = None,
    cas_client: CASClientBase = Depends(get_cas_client_async),
):
    # check if user is already logged in
    try:
        user = get_authenticated_user_async()
        if user:
            return responses.RedirectResponse(next)

    except HTTPException:
        pass

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

    # TODO: create user if not exists
    # return RedirectResponse(next)
    # default_auth.create_access_token_async(user, response)
    return responses.RedirectResponse(next)
