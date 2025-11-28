from abc import abstractmethod

from fivcglue import IComponent
from sqlalchemy.ext.asyncio.session import AsyncSession


class IUser(IComponent):
    """
    IUser is an interface for defining user models in the Fivccliche framework.
    """

    @property
    @abstractmethod
    def id(self) -> str:
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
    def is_admin(self) -> bool:
        """Whether the user is an admin."""


class IUserAuthenticator(IComponent):
    """
    IUserAuthenticator is an interface for authenticating users in the Fivccliche framework.
    """

    @abstractmethod
    async def create_access_token_async(
        self,
        username: str,
        password: str,
        session: AsyncSession | None = None,
        **kwargs,  # ignore additional arguments
    ) -> str | None:
        """Login a user and return a access token."""

    @abstractmethod
    async def verify_access_token_async(
        self,
        access_token: str,
        session: AsyncSession | None = None,
        **kwargs,  # ignore additional arguments
    ) -> IUser | None:
        """Authenticate a user by token."""
