import logging
from datetime import datetime, timezone, timedelta
from typing import cast

import jwt
from fastapi import FastAPI
from fivcglue import query_component, IComponentSite
from fivcglue.interfaces.caches import ICache
from fivcglue.interfaces.configs import IConfig

from fivccliche.services.interfaces.auth import IUser, IUserAuthenticator, UserCredential
from fivccliche.services.interfaces.modules import IModule, IModuleJob
from fivccliche.utils.deps import get_db_session_context_async
from fivccliche.utils.types import to_float, to_string

from .models import User
from .utils import create_user_async, get_user_async
from .routers import router

logger = logging.getLogger(__name__)


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

    def check_password(self, password: str) -> bool:
        return self.user.check_password(password)

    def change_password(self, password: str) -> None:
        self.user.change_password(password)


class UserAuthenticatorImpl(IUserAuthenticator):
    """User authenticator implementation."""

    def __init__(self, component_site: IComponentSite, **kwargs):
        logger.info("users authenticator initialized")
        self.cache = query_component(component_site, ICache)
        config = query_component(component_site, IConfig)
        config = config.get_session("auth")
        self.token_expire_hours = to_float(config.get_value("EXPIRATION_HOURS"), 12)
        self.token_algorithm = to_string(config.get_value("ALGORITHM"), "HS256")
        self.token_secret_key = to_string(
            config.get_value("SECRET_KEY"), "your-secret-key-change-this-in-production"
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
            return cast("str | None", payload.get("sub"))
        except jwt.ExpiredSignatureError as e:
            raise ValueError("Token has expired") from e
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {e!s}") from e

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
        async with get_db_session_context_async() as db_session:
            user = await create_user_async(
                db_session,
                username=username,
                email=email,
                full_name=full_name,
                password=password,
                is_superuser=is_superuser,
                preferences=preferences,
            )
            await db_session.commit()
            await db_session.refresh(user)
            return UserImpl(user) if user else None

    async def create_credential_async(
        self,
        username: str,
        password: str,
        ignore_password: bool = False,
        **kwargs,
    ) -> UserCredential | None:
        """Login a user and return a credential."""
        async with get_db_session_context_async() as db_session:
            user = await get_user_async(db_session, username=username)
            if user and not ignore_password and not user.check_password(password):
                user = None
            if user and not user.is_active:
                user = None
            if user:
                user.signed_in_at = datetime.now(timezone.utc)
                db_session.add(user)
                await db_session.commit()
            return self._create_access_token(user.uuid) if user else None

    async def create_sso_credential_async(
        self,
        username: str,
        attributes: dict,
        **kwargs,
    ) -> UserCredential | None:
        """Create a credential for SSO user.

        This method will get or create a user based on SSO authentication.
        If the user doesn't exist, it will be created without a password.

        Args:
            username: Username from SSO provider
            attributes: Additional attributes from SSO provider (may contain email, etc.)
            **kwargs: Additional arguments (ignored)

        Returns:
            UserCredential if successful, None otherwise
        """
        email = attributes.get("email") or attributes.get("mail")

        async with get_db_session_context_async() as db_session:
            user = await get_user_async(db_session, username=username)
            if not user:
                user = await create_user_async(
                    db_session,
                    username=username,
                    email=email,
                    password=None,  # SSO users don't have passwords
                    is_superuser=False,
                )

            if user:
                user.signed_in_at = datetime.now(timezone.utc)
                db_session.add(user)
                await db_session.commit()
            return self._create_access_token(user.uuid) if user else None

    async def verify_credential_async(self, access_token: str, **kwargs) -> IUser | None:
        """Authenticate a user by token."""
        try:
            user_uuid = self._decode_access_token(access_token)
        except ValueError:
            return None

        if user_uuid is None:
            return None

        try:
            async with get_db_session_context_async() as db_session:
                user = await get_user_async(db_session, user_uuid=user_uuid)
            if user and not user.is_active:
                return None
            return UserImpl(user) if user else None
        except Exception:
            logger.exception("Failed to verify credential")
            return None


class ModuleImpl(IModule):
    """User module implementation."""

    def __init__(self, _: IComponentSite, **kwargs):
        logger.info("users module initialized")

    @property
    def name(self):
        return "users"

    @property
    def description(self):
        return "User management module."

    def list_jobs(self) -> list[IModuleJob]:
        return []

    def get_job(self, job_name: str) -> IModuleJob | None:
        return None

    def mount(self, app: FastAPI, **kwargs) -> None:
        logger.info("users module mounted")
        app.include_router(router, **kwargs)
