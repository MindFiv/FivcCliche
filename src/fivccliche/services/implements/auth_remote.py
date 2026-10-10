import logging

import jwt
from fivcglue import IComponentSite, query_component
from fivcglue.interfaces.configs import IConfig

from fivccliche.services.interfaces.auth import IUser, IUserAuthenticator, UserCredential
from fivccliche.utils.types import to_string

logger = logging.getLogger(__name__)


class RemoteUserImpl(IUser):
    """Authenticated principal reconstructed from JWT claims."""

    def __init__(self, uuid: str, username: str, email: str | None, is_superuser: bool):
        self._uuid = uuid
        self._username = username
        self._email = email
        self._is_superuser = is_superuser

    @property
    def uuid(self) -> str:
        return self._uuid

    @property
    def username(self) -> str:
        return self._username

    @property
    def email(self) -> str:
        return self._email or ""

    @property
    def is_superuser(self) -> bool:
        return self._is_superuser


class RemoteUserAuthenticatorImpl(IUserAuthenticator):
    """Verify access tokens from JWT claims. Does not issue credentials."""

    def __init__(self, component_site: IComponentSite, **kwargs):
        logger.info("remote authenticator initialized")
        config = query_component(component_site, IConfig)
        config = config.get_session("auth")
        self.token_algorithm = to_string(config.get_value("ALGORITHM"), "HS256")
        self.token_secret_key = to_string(
            config.get_value("SECRET_KEY"), "your-secret-key-change-this-in-production"
        )

    async def create_user_async(
        self,
        username: str,
        email: str | None = None,
        full_name: str | None = None,
        password: str | None = None,
        is_superuser: bool = False,
        preferences: dict | None = None,
        **kwargs,
    ) -> IUser | None:
        """Create a new user."""
        raise NotImplementedError

    async def create_credential_async(
        self,
        username: str,
        password: str,
        **kwargs,
    ) -> UserCredential | None:
        """Login a user and return a credential."""
        raise NotImplementedError

    async def create_sso_credential_async(
        self,
        username: str,
        attributes: dict,
        **kwargs,
    ) -> UserCredential | None:
        """Create a credential for SSO user."""
        raise NotImplementedError

    async def verify_credential_async(self, access_token: str, **kwargs) -> IUser | None:
        """Authenticate a user from JWT claims without reading the user table."""
        try:
            payload = jwt.decode(
                access_token, self.token_secret_key, algorithms=[self.token_algorithm]
            )
        except jwt.InvalidTokenError:
            return None

        user_uuid = payload.get("sub")
        username = payload.get("username")
        is_superuser = payload.get("is_superuser")
        if "email" not in payload:
            return None
        email = payload.get("email")
        if not isinstance(user_uuid, str) or not isinstance(username, str):
            return None
        if not isinstance(is_superuser, bool):
            return None
        if email is not None and not isinstance(email, str):
            return None
        return RemoteUserImpl(
            uuid=user_uuid,
            username=username,
            email=email,
            is_superuser=is_superuser,
        )
