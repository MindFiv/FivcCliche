import asyncio
import logging

from fastapi import FastAPI
from fivcglue import IComponentSite
from fivcplayground.agents import AgentRun, AgentRunSession

from fivccliche.services.interfaces.agent_chats import (
    IUserChatContext,
    IUserChatProvider,
    UserChatRepository,
)
from fivccliche.services.interfaces.modules import IModule, IModuleJob
from fivccliche.utils.deps import get_db_session_context_async

from . import methods, routers
from .jobs import ChatMemorizeJob

logger = logging.getLogger(__name__)


class UserChatRepositoryImpl(UserChatRepository):
    """Chat repository implementation."""

    def __init__(self, user_uuid: str):
        self.user_uuid = user_uuid

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
        """Create or update an agent run session (chat)."""
        async with get_db_session_context_async() as db_session:
            existing = await methods.get_chat_async(db_session, session.id, self.user_uuid)
            if existing:
                if session.description is not None:
                    existing.description = session.description
                if hasattr(session, "context") and session.context is not None:
                    existing.context = session.context
                db_session.add(existing)
                await db_session.commit()
                await db_session.refresh(existing)
            else:
                context = getattr(session, "context", None)
                await methods.create_chat_async(
                    db_session,
                    user_uuid=self.user_uuid,
                    agent_id=session.agent_id,
                    chat_uuid=session.id,
                    description=session.description,
                    context=context,
                )

    async def get_agent_run_session_async(self, session_id: str) -> AgentRunSession | None:
        """Retrieve an agent run session (chat) by ID."""
        async with get_db_session_context_async() as db_session:
            chat = await methods.get_chat_async(db_session, session_id, self.user_uuid)
            return chat.to_schema() if chat else None

    async def list_agent_run_sessions_async(self, **kwargs) -> list[AgentRunSession]:
        """List all agent run sessions (chats) for the user."""
        skip = kwargs.get("skip", 0)
        limit = kwargs.get("limit", 100)
        async with get_db_session_context_async() as db_session:
            chats = await methods.list_chats_async(
                db_session, self.user_uuid, skip=skip, limit=limit
            )
            return [chat.to_schema() for chat in chats]

    async def delete_agent_run_session_async(self, session_id: str) -> None:
        """Delete an agent run session (chat)."""
        async with get_db_session_context_async() as db_session:
            chat = await methods.get_chat_async(db_session, session_id, self.user_uuid)
            if chat:
                await methods.delete_chat_async(db_session, chat)

    async def update_agent_run_async(self, session_id: str, agent_run: AgentRun) -> None:
        """Create or update an agent run (chat message)."""
        async with get_db_session_context_async() as db_session:
            chat = await methods.get_chat_async(db_session, session_id, self.user_uuid)
            if not chat:
                raise ValueError(f"Chat session {session_id} not found for user {self.user_uuid}")

            existing = await methods.get_chat_message_async(db_session, agent_run.id, session_id)
            if not existing:
                await methods.create_chat_message_async(
                    db_session,
                    chat_uuid=session_id,
                    status=agent_run.status,
                    query=agent_run.query.model_dump(mode="json") if agent_run.query else None,
                    reply=agent_run.reply.model_dump(mode="json") if agent_run.reply else None,
                    tool_calls={
                        k: v.model_dump(mode="json") for k, v in agent_run.tool_calls.items()
                    },
                    message_uuid=agent_run.id,
                )
            else:
                await methods.update_chat_message_async(
                    db_session,
                    existing,
                    status=agent_run.status,
                    query=agent_run.query.model_dump(mode="json") if agent_run.query else None,
                    reply=agent_run.reply.model_dump(mode="json") if agent_run.reply else None,
                    tool_calls={
                        k: v.model_dump(mode="json") for k, v in agent_run.tool_calls.items()
                    },
                    completed_at=agent_run.completed_at,
                )

    async def get_agent_run_async(self, session_id: str, run_id: str) -> AgentRun | None:
        """Retrieve an agent run (chat message) by ID."""
        async with get_db_session_context_async() as db_session:
            chat = await methods.get_chat_async(db_session, session_id, self.user_uuid)
            if not chat:
                return None

            message = await methods.get_chat_message_async(db_session, run_id, session_id)
            return message.to_schema() if message else None

    async def list_agent_runs_async(self, session_id: str, **kwargs) -> list[AgentRun]:
        """List all agent runs (chat messages) for a session."""
        async with get_db_session_context_async() as db_session:
            chat = await methods.get_chat_async(db_session, session_id, self.user_uuid)
            if not chat:
                return []

            skip = kwargs.get("skip", 0)
            limit = kwargs.get("limit", 30)
            messages = await methods.list_chat_messages_async(
                db_session, session_id, skip=skip, limit=limit
            )
            return [message.to_schema() for message in messages]

    async def delete_agent_run_async(self, session_id: str, run_id: str) -> None:
        """Delete an agent run (chat message)."""
        async with get_db_session_context_async() as db_session:
            chat = await methods.get_chat_async(db_session, session_id, self.user_uuid)
            if not chat:
                raise ValueError(f"Chat session {session_id} not found for user {self.user_uuid}")

            message = await methods.get_chat_message_async(db_session, run_id, session_id)
            if message:
                await methods.delete_chat_message_async(db_session, message)


class UserChatProviderImpl(IUserChatProvider):
    """Chat provider implementation."""

    def __init__(self, component_site: IComponentSite, **kwargs):
        logger.info("agent chats provider initialized")
        self.component_site = component_site

    def get_chat_repository(
        self,
        user_uuid: str,
        **kwargs,
    ) -> UserChatRepository:
        """Get the chat repository without binding a DB session."""
        return UserChatRepositoryImpl(user_uuid=user_uuid)

    def get_chat_context(
        self,
        user_uuid: str,
        context: dict | None = None,
        **kwargs,
    ) -> IUserChatContext | None:
        """Get the chat context for providing chat-specific tools."""
        return None


class ModuleImpl(IModule):
    """Agent chats module implementation."""

    def __init__(self, component_site: IComponentSite, **kwargs):
        self._component_site = component_site
        self._jobs: list[IModuleJob] = [ChatMemorizeJob(component_site)]
        logger.info("agent chats module initialized")

    @property
    def name(self):
        return "agent_chats"

    @property
    def description(self):
        return "Agent Chat management module."

    def list_jobs(self) -> list[IModuleJob]:
        return list(self._jobs)

    def get_job(self, job_name: str) -> IModuleJob | None:
        for job in self._jobs:
            if job.name == job_name:
                return job
        return None

    def mount(self, app: FastAPI, **kwargs) -> None:
        logger.info("agent_chats module mounted")
        app.include_router(routers.router_chats, **kwargs)
        app.include_router(routers.router_messages, **kwargs)
