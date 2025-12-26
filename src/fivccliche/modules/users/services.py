import json
from datetime import datetime, timezone, timedelta

import jwt
from fastapi import FastAPI
from fivcglue import query_component, IComponentSite
from fivcglue.interfaces.caches import ICache
from fivcglue.interfaces.configs import IConfig
from sqlalchemy.ext.asyncio.session import AsyncSession

from fivccliche.services.interfaces.auth import IUser, IUserAuthenticator, UserCredential
from fivccliche.services.interfaces.modules import IModule
from fivccliche.utils.deps import get_db_session_async

from .models import User
from .methods import create_user_async, get_user_async, authenticate_user_async
from .routers import router


class UserImpl(IUser):
    """User implementation."""

    def __init__(self, user: User):
        self.user = user

    @property
    def uuid(self) -> str:
        return self.user.uuid

    @property
    def username(self) -> str:
        return self.user.username

    @property
    def email(self) -> str:
        return str(self.user.email)

    @property
    def is_superuser(self) -> bool:
        return self.user.is_superuser


class UserAuthenticatorImpl(IUserAuthenticator):
    """User authenticator implementation."""

    def __init__(self, component_site: IComponentSite, **kwargs):
        print("users authenticator initialized...")
        self.cache = query_component(component_site, ICache)
        config = query_component(component_site, IConfig)
        config = config.get_session("auth")
        self.token_expire_hours = float(config.get_value("EXPIRATION_HOURS") or 12)
        self.token_algorithm = config.get_value("ALGORITHM") or "HS256"
        self.token_secret_key = (
            config.get_value("SECRET_KEY") or "your-secret-key-change-this-in-production"
        )

    def _create_access_token(self, user_uuid: str) -> UserCredential:
        """Create a JWT access token for a user."""
        time_now = datetime.now(timezone.utc)
        time_expire = time_now + timedelta(hours=self.token_expire_hours)
        access_token = jwt.encode(
            {
                "sub": user_uuid,  # Subject (user ID)
                "iat": time_now,  # Issued at
                "exp": time_expire,  # Expiration time
            },
            self.token_secret_key,
            algorithm=self.token_algorithm,
        )
        expires_in = int(self.token_expire_hours * 3600)  # Convert hours to seconds
        return UserCredential(access_token=access_token, expires_in=expires_in)

    def _decode_access_token(self, access_token: str) -> str | None:
        """Decode and validate a JWT access token."""
        try:
            payload = jwt.decode(
                access_token, self.token_secret_key, algorithms=[self.token_algorithm]
            )
            return payload.get("sub")
        except jwt.ExpiredSignatureError as e:
            raise ValueError("Token has expired") from e
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {e!s}") from e

    async def create_user_async(
        self,
        username: str,
        email: str | None = None,
        password: str | None = None,
        is_superuser: bool = False,
        session: AsyncSession | None = None,
        **kwargs,
    ) -> IUser | None:
        """Create a new user."""
        if session:
            user = await create_user_async(
                session,
                username=username,
                email=email,
                password=password,
                is_superuser=is_superuser,
            )
            return UserImpl(user) if user else None

        async with get_db_session_async() as session:
            user = await create_user_async(
                session,
                username=username,
                email=email,
                password=password,
                is_superuser=is_superuser,
            )
            return UserImpl(user) if user else None

    async def create_credential_async(
        self,
        username: str,
        password: str,
        session: AsyncSession | None = None,
        **kwargs,
    ) -> UserCredential | None:
        """Login a user and return a credential."""
        if session:
            user = await authenticate_user_async(session, username, password)
            return self._create_access_token(user.uuid) if user else None

        async with get_db_session_async() as session:
            user = await authenticate_user_async(session, username, password)
            return self._create_access_token(user.uuid) if user else None

    async def create_sso_credential_async(
        self,
        username: str,
        attributes: dict,
        session: AsyncSession | None = None,
        **kwargs,
    ) -> UserCredential | None:
        """Create a credential for SSO user.

        This method will get or create a user based on SSO authentication.
        If the user doesn't exist, it will be created without a password.

        Args:
            username: Username from SSO provider
            attributes: Additional attributes from SSO provider (may contain email, etc.)
            session: Database session (optional)
            **kwargs: Additional arguments (ignored)

        Returns:
            UserCredential if successful, None otherwise
        """
        # Extract email from attributes if available
        email = attributes.get("email") or attributes.get("mail")

        if session:
            # Try to get existing user
            user = await get_user_async(session, username=username)

            # Create user if doesn't exist
            if not user:
                user = await create_user_async(
                    session,
                    username=username,
                    email=email,
                    password=None,  # SSO users don't have passwords
                    is_superuser=False,
                )

            return self._create_access_token(user.uuid) if user else None

        async with get_db_session_async() as session:
            # Try to get existing user
            user = await get_user_async(session, username=username)

            # Create user if doesn't exist
            if not user:
                user = await create_user_async(
                    session,
                    username=username,
                    email=email,
                    password=None,  # SSO users don't have passwords
                    is_superuser=False,
                )

            return self._create_access_token(user.uuid) if user else None

    async def verify_credential_async(
        self, access_token: str, session: AsyncSession | None = None, **kwargs
    ) -> IUser | None:
        """Authenticate a user by token."""
        try:
            user_uuid = self._decode_access_token(access_token)
        except ValueError:
            return None

        if user_uuid is None:
            return None

        user = None
        user_info = self.cache.get_value(f"user: {user_uuid}")
        if user_info:
            user_info = json.loads(user_info)
            user = User(**user_info)

        try:
            if session:
                user = await get_user_async(session, user_uuid=user_uuid)

            else:
                async with get_db_session_async() as session:
                    user = await get_user_async(session, user_uuid=user_uuid)

            return UserImpl(user) if user else None

        except Exception as e:
            # do nothing
            print(e)

        finally:
            if user:
                self.cache.set_value(
                    f"user: {user.uuid}",
                    user.model_dump_json().encode("utf-8"),
                    expire=timedelta(hours=self.token_expire_hours),
                )


class ModuleImpl(IModule):
    """User module implementation."""

    def __init__(self, _: IComponentSite, **kwargs):
        print("users module initialized...")

    @property
    def name(self):
        return "users"

    @property
    def description(self):
        return "User management module."

    def mount(self, app: FastAPI, **kwargs) -> None:
        print("users module mounted.")
        app.include_router(router, **kwargs)
