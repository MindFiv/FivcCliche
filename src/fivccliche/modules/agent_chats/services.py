import asyncio

from fastapi import FastAPI

from fivcglue import IComponentSite
from fivcplayground.agents import AgentRunSession, AgentRun
from sqlalchemy.ext.asyncio.session import AsyncSession

from fivccliche.services.interfaces.modules import IModule
from fivccliche.services.interfaces.agent_chats import IUserChatProvider, UserChatRepository

from . import methods, models, routers


class UserChatRepositoryImpl(UserChatRepository):
    """Chat repository implementation."""

    def __init__(self, user_uuid: str | None = None, session: AsyncSession | None = None):
        self.user_uuid = user_uuid
        self.session = session

    # ========================================================================
    # Synchronous wrapper methods (for interface compatibility)
    # ========================================================================

    def update_agent_run_session(self, session: AgentRunSession) -> None:
        """Update an agent run session."""
        asyncio.run(self.update_agent_run_session_async(session))

    def get_agent_run_session(self, session_id: str) -> AgentRunSession | None:
        """Get an agent run session."""
        return asyncio.run(self.get_agent_run_session_async(session_id))

    def list_agent_run_sessions(self, **kwargs) -> list[AgentRunSession]:
        """List all agent run sessions."""
        return asyncio.run(self.list_agent_run_sessions_async(**kwargs))

    def delete_agent_run_session(self, session_id: str) -> None:
        """Delete an agent run session."""
        asyncio.run(self.delete_agent_run_session_async(session_id))

    def update_agent_run(self, session_id: str, agent_run: AgentRun) -> None:
        """Update an agent run."""
        asyncio.run(self.update_agent_run_async(session_id, agent_run))

    def get_agent_run(self, session_id: str, run_id: str) -> AgentRun | None:
        """Get an agent run."""
        return asyncio.run(self.get_agent_run_async(session_id, run_id))

    def list_agent_runs(self, session_id: str, **kwargs) -> list[AgentRun]:
        """List all agent runs."""
        return asyncio.run(self.list_agent_runs_async(session_id, **kwargs))

    def delete_agent_run(self, session_id: str, run_id: str) -> None:
        """Delete an agent run."""
        asyncio.run(self.delete_agent_run_async(session_id, run_id))

    # ========================================================================
    # Async methods for agent run sessions (chats)
    # ========================================================================

    async def update_agent_run_session_async(self, session: AgentRunSession) -> None:
        """Create or update an agent run session (chat).

        Args:
            session: AgentRunSession to create or update

        Raises:
            ValueError: If session or user_uuid is not set
        """
        if not self.session or not self.user_uuid:
            raise ValueError(
                "Session and user_uuid are required for update_agent_run_session operation"
            )

        # Check if chat exists
        existing = await methods.get_chat_async(self.session, session.id, self.user_uuid)
        if existing:
            # Update existing chat
            if session.description is not None:
                existing.description = session.description
            self.session.add(existing)
            await self.session.commit()
            await self.session.refresh(existing)
        else:
            # Create new chat using create_chat_async
            await methods.create_chat_async(
                self.session,
                user_uuid=self.user_uuid,
                agent_id=session.agent_id,
                chat_uuid=session.id,
                description=session.description,
            )

    async def get_agent_run_session_async(self, session_id: str) -> AgentRunSession | None:
        """Retrieve an agent run session (chat) by ID.

        Args:
            session_id: Chat UUID

        Returns:
            AgentRunSession or None if not found

        Raises:
            ValueError: If session or user_uuid is not set
        """
        if not self.session or not self.user_uuid:
            raise ValueError(
                "Session and user_uuid are required for get_agent_run_session operation"
            )

        chat = await methods.get_chat_async(self.session, session_id, self.user_uuid)
        return chat.to_schema() if chat else None

    async def list_agent_run_sessions_async(self, **kwargs) -> list[AgentRunSession]:
        """List all agent run sessions (chats) for the user.

        Args:
            **kwargs: Optional skip and limit parameters for pagination

        Returns:
            List of AgentRunSession objects

        Raises:
            ValueError: If session or user_uuid is not set
        """
        if not self.session or not self.user_uuid:
            raise ValueError(
                "Session and user_uuid are required for list_agent_run_sessions operation"
            )

        skip = kwargs.get("skip", 0)
        limit = kwargs.get("limit", 100)
        chats = await methods.list_chats_async(self.session, self.user_uuid, skip=skip, limit=limit)
        return [chat.to_schema() for chat in chats]

    async def delete_agent_run_session_async(self, session_id: str) -> None:
        """Delete an agent run session (chat).

        Args:
            session_id: Chat UUID

        Raises:
            ValueError: If session or user_uuid is not set
        """
        if not self.session or not self.user_uuid:
            raise ValueError(
                "Session and user_uuid are required for delete_agent_run_session operation"
            )

        chat = await methods.get_chat_async(self.session, session_id, self.user_uuid)
        if chat:
            await methods.delete_chat_async(self.session, chat)

    # ========================================================================
    # Async methods for agent runs (chat messages)
    # ========================================================================

    async def update_agent_run_async(self, session_id: str, agent_run: AgentRun) -> None:
        """Create or update an agent run (chat message).

        Args:
            session_id: Chat UUID
            agent_run: AgentRun to create or update

        Raises:
            ValueError: If session or user_uuid is not set
        """
        if not self.session or not self.user_uuid:
            raise ValueError("Session and user_uuid are required for update_agent_run operation")

        # Verify the chat exists and belongs to the user
        chat = await methods.get_chat_async(self.session, session_id, self.user_uuid)
        if not chat:
            raise ValueError(f"Chat session {session_id} not found for user {self.user_uuid}")

        # Check if message exists
        existing = await methods.get_chat_message_async(self.session, agent_run.id, session_id)
        if existing:
            # Update existing message
            if agent_run.status is not None:
                existing.status = agent_run.status
            if agent_run.query is not None:
                # Convert AgentRunContent to dict if needed
                existing.query = (
                    agent_run.query.model_dump()
                    if hasattr(agent_run.query, "model_dump")
                    else agent_run.query
                )
            if agent_run.reply is not None:
                # Convert AgentRunContent to dict if needed
                existing.reply = (
                    agent_run.reply.model_dump()
                    if hasattr(agent_run.reply, "model_dump")
                    else agent_run.reply
                )
            if agent_run.tool_calls is not None:
                existing.tool_calls = agent_run.tool_calls
            if agent_run.completed_at is not None:
                existing.completed_at = agent_run.completed_at
            self.session.add(existing)
            await self.session.commit()
            await self.session.refresh(existing)
        else:
            # Create new message
            # Convert AgentRunContent objects to dicts if needed
            query_data = (
                agent_run.query.model_dump()
                if hasattr(agent_run.query, "model_dump")
                else agent_run.query
            )
            reply_data = (
                agent_run.reply.model_dump()
                if hasattr(agent_run.reply, "model_dump")
                else agent_run.reply
            )
            tool_calls_data = agent_run.tool_calls or {}

            message = models.UserChatMessage(
                uuid=agent_run.id,
                chat_uuid=session_id,
                status=agent_run.status or "pending",
                query=query_data,
                reply=reply_data,
                tool_calls=tool_calls_data,
                completed_at=agent_run.completed_at,
            )
            self.session.add(message)
            await self.session.commit()
            await self.session.refresh(message)

    async def get_agent_run_async(self, session_id: str, run_id: str) -> AgentRun | None:
        """Retrieve an agent run (chat message) by ID.

        Args:
            session_id: Chat UUID
            run_id: Message UUID

        Returns:
            AgentRun or None if not found

        Raises:
            ValueError: If session or user_uuid is not set
        """
        if not self.session or not self.user_uuid:
            raise ValueError("Session and user_uuid are required for get_agent_run operation")

        # Verify the chat exists and belongs to the user
        chat = await methods.get_chat_async(self.session, session_id, self.user_uuid)
        if not chat:
            return None

        message = await methods.get_chat_message_async(self.session, run_id, session_id)
        return message.to_schema() if message else None

    async def list_agent_runs_async(self, session_id: str, **kwargs) -> list[AgentRun]:
        """List all agent runs (chat messages) for a session.

        Args:
            session_id: Chat UUID
            **kwargs: Optional skip and limit parameters for pagination

        Returns:
            List of AgentRun objects

        Raises:
            ValueError: If session or user_uuid is not set
        """
        if not self.session or not self.user_uuid:
            raise ValueError("Session and user_uuid are required for list_agent_runs operation")

        # Verify the chat exists and belongs to the user
        chat = await methods.get_chat_async(self.session, session_id, self.user_uuid)
        if not chat:
            return []

        skip = kwargs.get("skip", 0)
        limit = kwargs.get("limit", 100)
        messages = await methods.list_chat_messages_async(
            self.session, session_id, skip=skip, limit=limit
        )
        return [message.to_schema() for message in messages]

    async def delete_agent_run_async(self, session_id: str, run_id: str) -> None:
        """Delete an agent run (chat message).

        Args:
            session_id: Chat UUID
            run_id: Message UUID

        Raises:
            ValueError: If session or user_uuid is not set
        """
        if not self.session or not self.user_uuid:
            raise ValueError("Session and user_uuid are required for delete_agent_run operation")

        # Verify the chat exists and belongs to the user
        chat = await methods.get_chat_async(self.session, session_id, self.user_uuid)
        if not chat:
            raise ValueError(f"Chat session {session_id} not found for user {self.user_uuid}")

        message = await methods.get_chat_message_async(self.session, run_id, session_id)
        if message:
            await methods.delete_chat_message_async(self.session, message)


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
        app.include_router(routers.router_chats, **kwargs)
        app.include_router(routers.router_messages, **kwargs)
