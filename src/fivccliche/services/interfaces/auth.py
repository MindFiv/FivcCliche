from abc import abstractmethod

from fivcglue import IComponent
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio.session import AsyncSession


class UserCredential(BaseModel):
    """User credential model."""

    access_token: str = Field(..., description="Access token")
    expires_in: int = Field(..., description="Token expiration time in seconds")


class IUser(IComponent):
    """
    IUser is an interface for defining user models in the Fivccliche framework.
    """

    @property
    @abstractmethod
    def uuid(self) -> str:
        """UUID of the user."""

    @property
    @abstractmethod
    def username(self) -> str:
        """Username of the user."""

    @property
    @abstractmethod
    def email(self) -> str:
        """Email of the user."""

    @property
    @abstractmethod
    def is_superuser(self) -> bool:
        """Whether the user is a superuser."""


class IUserAuthenticator(IComponent):
    """
    IUserAuthenticator is an interface for authenticating users in the Fivccliche framework.
    """

    @abstractmethod
    async def create_user_async(
        self,
        username: str,
        email: str | None = None,
        password: str | None = None,
        is_superuser: bool = False,
        session: AsyncSession | None = None,
        **kwargs,  # ignore additional arguments
    ) -> IUser | None:
        """Create a new user."""

    @abstractmethod
    async def create_credential_async(
        self,
        username: str,
        password: str,
        session: AsyncSession | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserCredential | None:
        """Login a user and return a credential."""

    @abstractmethod
    async def create_sso_credential_async(
        self,
        username: str,
        attributes: dict,
        session: AsyncSession | None = None,
        **kwargs,  # ignore additional arguments
    ) -> UserCredential | None:
        """Create a credential for SSO user."""

    @abstractmethod
    async def verify_credential_async(
        self,
        access_token: str,
        session: AsyncSession | None = None,
        **kwargs,  # ignore additional arguments
    ) -> IUser | None:
        """Authenticate a user by token."""
