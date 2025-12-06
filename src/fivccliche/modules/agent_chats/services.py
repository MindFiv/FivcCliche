from fastapi import FastAPI

from fivcglue import IComponentSite
from sqlalchemy.ext.asyncio.session import AsyncSession

from fivccliche.services.interfaces.modules import IModule
from fivccliche.services.interfaces.agent_chats import IUserChatProvider, UserChatRepository

from . import routers


class UserChatRepositoryImpl(UserChatRepository):
    """Chat repository implementation."""

    def __init__(self, user_uuid: str | None = None, session: AsyncSession | None = None):
        self.user_uuid = user_uuid
        self.session = session

    async def create_run_async(self, run_data: dict) -> dict:
        """Create a new chat run (session)."""
        if not self.session or not self.user_uuid:
            raise ValueError("Session and user_uuid are required for create_run operation")
        # Implementation for creating a run
        raise NotImplementedError("create_run_async not yet implemented")

    async def get_run_async(self, run_id: str) -> dict | None:
        """Get a chat run by ID."""
        if not self.session or not self.user_uuid:
            raise ValueError("Session and user_uuid are required for get_run operation")
        # Implementation for getting a run
        raise NotImplementedError("get_run_async not yet implemented")

    async def list_runs_async(self, **kwargs) -> list[dict]:
        """List all chat runs."""
        if not self.session or not self.user_uuid:
            raise ValueError("Session and user_uuid are required for list_runs operation")
        # Implementation for listing runs
        raise NotImplementedError("list_runs_async not yet implemented")

    async def delete_run_async(self, run_id: str) -> None:
        """Delete a chat run."""
        if not self.session or not self.user_uuid:
            raise ValueError("Session and user_uuid are required for delete_run operation")
        # Implementation for deleting a run
        raise NotImplementedError("delete_run_async not yet implemented")


class UserChatProviderImpl(IUserChatProvider):
    """Chat provider implementation."""

    def __init__(self, component_site: IComponentSite, **kwargs):
        print("agent chats provider initialized...")
        self.component_site = component_site

    def get_chat_repository(
        self,
        user_uuid: str | None = None,
        session: AsyncSession | None = None,
        **kwargs,
    ) -> UserChatRepository:
        """Get the chat repository."""
        return UserChatRepositoryImpl(user_uuid=user_uuid, session=session)


class ModuleImpl(IModule):
    """Agent chats module implementation."""

    def __init__(self, _: IComponentSite, **kwargs):
        print("agent chats module initialized.")

    @property
    def name(self):
        return "agent_chats"

    @property
    def description(self):
        return "Agent Chat management module."

    def mount(self, app: FastAPI, **kwargs) -> None:
        print("agent_chats module mounted.")
        app.include_router(routers.router_chats, prefix="/chats")
        app.include_router(routers.router_messages, prefix="/chats")
