import json
from datetime import datetime, timezone, timedelta

import jwt
from fastapi import FastAPI
from fivcglue import query_component, IComponentSite
from fivcglue.interfaces.caches import ICache
from fivcglue.interfaces.configs import IConfig
from sqlalchemy.ext.asyncio.session import AsyncSession

from fivccliche.services.interfaces.auth import IUser, IUserAuthenticator
from fivccliche.services.interfaces.modules import IModule
from fivccliche.utils.deps import get_db_session_async

from .models import User
from .methods import get_user_async, authenticate_user_async
from .routers import router


class UserImpl(IUser):
    """User implementation."""

    def __init__(self, user: User):
        self.user = user

    @property
    def id(self) -> str:
        return self.user.id

    @property
    def username(self) -> str:
        return self.user.username

    @property
    def email(self) -> str:
        return str(self.user.email)

    @property
    def is_admin(self) -> bool:
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

    def _create_access_token(self, user_id: str) -> str:
        """Create a JWT access token for a user."""
        time_now = datetime.now(timezone.utc)
        time_expire = time_now + timedelta(hours=self.token_expire_hours)
        return jwt.encode(
            {
                "sub": user_id,  # Subject (user ID)
                "iat": time_now,  # Issued at
                "exp": time_expire,  # Expiration time
            },
            self.token_secret_key,
            algorithm=self.token_algorithm,
        )

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

    async def verify_access_token_async(
        self, access_token: str, session: AsyncSession | None = None, **kwargs
    ) -> IUser | None:
        """Authenticate a user by token."""
        try:
            user_id = self._decode_access_token(access_token)
        except ValueError:
            return None

        if user_id is None:
            return None

        user = None
        user_info = self.cache.get_value(f"user: {user_id}")
        if user_info:
            user_info = json.loads(user_info)
            user = User(**user_info)

        try:
            if session:
                user = await get_user_async(session, user_id=user_id)

            else:
                async with get_db_session_async() as session:
                    user = await get_user_async(session, user_id=user_id)

        finally:
            if user:
                user_info = {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "is_active": user.is_active,
                    "is_superuser": user.is_superuser,
                }
                self.cache.set_value(
                    f"user: {user.id}",
                    json.dumps(user_info).encode("utf-8"),
                    expire=timedelta(hours=self.token_expire_hours),
                )

        return UserImpl(user) if user else None

    async def create_access_token_async(
        self,
        username: str,
        password: str,
        session: AsyncSession | None = None,
        **kwargs,
    ) -> str | None:
        """Login a user and return a access token."""
        if session:
            user = await authenticate_user_async(session, username, password)
            return self._create_access_token(user.id) if user else None

        async with get_db_session_async() as session:
            user = await authenticate_user_async(session, username, password)
            return self._create_access_token(user.id) if user else None


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
        app.include_router(router)
