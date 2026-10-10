import logging
from datetime import datetime, timezone, timedelta

import jwt
from fastapi import FastAPI
from fivcglue import query_component, IComponentSite
from fivcglue.interfaces.caches import ICache
from fivcglue.interfaces.configs import IConfig

from fivccliche.services.implements.auth_remote import RemoteUserAuthenticatorImpl
from fivccliche.services.interfaces.auth import IUser, UserCredential
from fivccliche.services.interfaces.modules import IModule, IModuleJob
from fivccliche.utils.deps import get_db_session_context_async
from fivccliche.utils.types import to_float

from .jobs import ChangePasswordJob, CreateSuperuserJob
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


class UserAuthenticatorImpl(RemoteUserAuthenticatorImpl):
    """User authenticator implementation."""

    def __init__(self, component_site: IComponentSite, **kwargs):
        super().__init__(component_site, **kwargs)
        logger.info("users authenticator initialized")
        self.cache = query_component(component_site, ICache)
        config = query_component(component_site, IConfig)
        config = config.get_session("auth")
        self.token_expire_hours = to_float(config.get_value("EXPIRATION_HOURS"), 12)

    def _create_access_token(self, user: User) -> UserCredential:
        """Create a JWT access token for a user."""
        time_now = datetime.now(timezone.utc)
        time_expire = time_now + timedelta(hours=self.token_expire_hours)
        access_token = jwt.encode(
            {
                "sub": user.uuid,
                "username": user.username,
                "email": None if user.email is None else str(user.email),
                "is_superuser": user.is_superuser,
                "iat": time_now,
                "exp": time_expire,
            },
            self.token_secret_key,
            algorithm=self.token_algorithm,
        )
        expires_in = int(self.token_expire_hours * 3600)  # Convert hours to seconds
        return UserCredential(access_token=access_token, expires_in=expires_in)

    async def verify_credential_async(self, access_token: str, **kwargs) -> IUser | None:
        """Decode the token, and confirm superuser claims against the user table."""
        user = await super().verify_credential_async(access_token, **kwargs)
        if user is None or not user.is_superuser:
            return user
        async with get_db_session_context_async() as db_session:
            row = await get_user_async(db_session, user_uuid=user.uuid)
        if row is None or not row.is_active or not row.is_superuser:
            return None
        return user

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
            if not user:
                return None
            user.signed_in_at = datetime.now(timezone.utc)
            db_session.add(user)
            credential = self._create_access_token(user)
            await db_session.commit()
            return credential

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

            if not user or not user.is_active:
                return None
            user.signed_in_at = datetime.now(timezone.utc)
            db_session.add(user)
            credential = self._create_access_token(user)
            await db_session.commit()
            return credential


class ModuleImpl(IModule):
    """User module implementation."""

    def __init__(self, component_site: IComponentSite, **kwargs):
        self._jobs: list[IModuleJob] = [
            CreateSuperuserJob(component_site),
            ChangePasswordJob(component_site),
        ]
        logger.info("users module initialized")

    @property
    def name(self):
        return "users"

    @property
    def description(self):
        return "User management module."

    def list_jobs(self) -> list[IModuleJob]:
        return list(self._jobs)

    def get_job(self, job_name: str) -> IModuleJob | None:
        for job in self._jobs:
            if job.name == job_name:
                return job
        return None

    def mount(self, app: FastAPI, **kwargs) -> None:
        logger.info("users module mounted")
        app.include_router(router, **kwargs)
