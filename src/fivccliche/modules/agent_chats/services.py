import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Any, cast
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fivcglue import IComponentSite, query_component
from fivcglue.interfaces.mutexes import IMutex
from fivcplayground.agents import (
    AgentRun,
    AgentRunEvent,
    AgentRunSession,
    create_agent_async,
)
from fivcplayground.skills import create_skill_retriever_async
from fivcplayground.tools import create_tool_retriever_async

from fivccliche.services.interfaces.agent_chats import (
    IUserChatProvider,
    IUserChatRun,
    IUserChatRunProvider,
    UserChatRepository,
)
from fivccliche.services.interfaces.agent_configs import IUserConfigProvider
from fivccliche.services.interfaces.modules import IModule, IModuleJob
from fivccliche.utils.chats.warmup import SessionWarmup
from fivccliche.utils.deps import get_db_session_context_async

from . import routers, utils
from .filters import ChatFilterSet
from .jobs.describe import ChatDescribeJob

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
            existing = await utils.get_chat_async(
                db_session, session.id, filters=ChatFilterSet(self.user_uuid, is_superuser=False)
            )
            if existing:
                if session.description is not None:
                    existing.description = session.description
                    existing.updated_at = datetime.now(timezone.utc)
                existing.context = session.context
                db_session.add(existing)
                await db_session.commit()
                await db_session.refresh(existing)
            else:
                await utils.create_chat_async(
                    db_session,
                    user_uuid=self.user_uuid,
                    agent_id=session.agent_id,
                    chat_uuid=session.id,
                    description=session.description,
                    context=session.context,
                )
                await db_session.commit()

    async def get_agent_run_session_async(self, session_id: str) -> AgentRunSession | None:
        """Retrieve an agent run session (chat) by ID."""
        async with get_db_session_context_async() as db_session:
            chat = await utils.get_chat_async(
                db_session, session_id, filters=ChatFilterSet(self.user_uuid, is_superuser=False)
            )
            return chat.to_schema() if chat else None

    async def list_agent_run_sessions_async(self, **kwargs) -> list[AgentRunSession]:
        """List all agent run sessions (chats) for the user."""
        skip = kwargs.get("skip", 0)
        limit = kwargs.get("limit", 100)
        async with get_db_session_context_async() as db_session:
            chats = await utils.list_chats_async(
                db_session,
                filters=ChatFilterSet(self.user_uuid, is_superuser=False),
                skip=skip,
                limit=limit,
            )
            return [chat.to_schema() for chat in chats]

    async def delete_agent_run_session_async(self, session_id: str) -> None:
        """Delete an agent run session (chat)."""
        async with get_db_session_context_async() as db_session:
            chat = await utils.get_chat_async(
                db_session, session_id, filters=ChatFilterSet(self.user_uuid, is_superuser=False)
            )
            if chat:
                await db_session.delete(chat)
                await db_session.commit()

    async def update_agent_run_async(self, session_id: str, agent_run: AgentRun) -> None:
        """Create or update an agent run (chat message)."""
        async with get_db_session_context_async() as db_session:
            chat = await utils.get_chat_async(
                db_session, session_id, filters=ChatFilterSet(self.user_uuid, is_superuser=False)
            )
            if not chat:
                raise ValueError(f"Chat session {session_id} not found for user {self.user_uuid}")

            existing = await utils.get_chat_message_async(db_session, agent_run.id, session_id)
            if not existing:
                await utils.create_chat_message_async(
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
                await utils.update_chat_message_async(
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
            await db_session.commit()

    async def get_agent_run_async(self, session_id: str, run_id: str) -> AgentRun | None:
        """Retrieve an agent run (chat message) by ID."""
        async with get_db_session_context_async() as db_session:
            chat = await utils.get_chat_async(
                db_session, session_id, filters=ChatFilterSet(self.user_uuid, is_superuser=False)
            )
            if not chat:
                return None

            message = await utils.get_chat_message_async(db_session, run_id, session_id)
            return message.to_schema() if message else None

    async def list_agent_runs_async(self, session_id: str, **kwargs) -> list[AgentRun]:
        """List all agent runs (chat messages) for a session."""
        async with get_db_session_context_async() as db_session:
            chat = await utils.get_chat_async(
                db_session, session_id, filters=ChatFilterSet(self.user_uuid, is_superuser=False)
            )
            if not chat:
                return []

            skip = kwargs.get("skip", 0)
            limit = kwargs.get("limit", 30)
            messages = await utils.list_chat_messages_async(
                db_session, session_id, skip=skip, limit=limit
            )
            return [message.to_schema() for message in messages]

    async def delete_agent_run_async(self, session_id: str, run_id: str) -> None:
        """Delete an agent run (chat message)."""
        async with get_db_session_context_async() as db_session:
            chat = await utils.get_chat_async(
                db_session, session_id, filters=ChatFilterSet(self.user_uuid, is_superuser=False)
            )
            if not chat:
                raise ValueError(f"Chat session {session_id} not found for user {self.user_uuid}")

            message = await utils.get_chat_message_async(db_session, run_id, session_id)
            if message:
                await db_session.delete(message)
                await db_session.commit()


class LazyChatTime:
    """ISO time computed from context timezone when stringified."""

    def __init__(self, context: dict):
        self._context = context

    def __str__(self) -> str:
        tz = ZoneInfo(str(self._context["timezone"]))
        return datetime.now(tz).isoformat()


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
    ) -> dict:
        """Return a copy of context with user_uuid and kwargs merged in."""
        user_context = {**context} if context else {}
        user_context.update(user_uuid=user_uuid)
        user_context.update(kwargs)
        user_context.setdefault("timezone", "Asia/Shanghai")
        user_context["time"] = LazyChatTime(user_context)
        return user_context


class UserChatRunImpl(IUserChatRun):
    """Run one user query through the configured chat agent."""

    class _Dependencies:
        """Provider-resolved collaborators reused for every turn."""

        def __init__(
            self,
            config_provider: IUserConfigProvider,
            chat_provider: IUserChatProvider,
            *,
            user_uuid: str,
        ):
            self.embedding_repository = config_provider.get_embedding_repository(
                user_uuid=user_uuid
            )
            self.embedding_backend = config_provider.get_embedding_backend()
            self.model_repository = config_provider.get_model_repository(user_uuid=user_uuid)
            self.model_backend = config_provider.get_model_backend()
            self.tool_repository = config_provider.get_tool_repository(user_uuid=user_uuid)
            self.tool_backend = config_provider.get_tool_backend()
            self.skill_repository = config_provider.get_skill_repository(user_uuid=user_uuid)
            self.agent_repository = config_provider.get_agent_repository(user_uuid=user_uuid)
            self.agent_backend = config_provider.get_agent_backend()
            self.chat_repository = chat_provider.get_chat_repository(user_uuid=user_uuid)

    def __init__(
        self,
        config_provider: IUserConfigProvider,
        chat_provider: IUserChatProvider,
        chat_uuid: str,
        *,
        user_uuid: str,
        agent_id: str = "default",
        context: dict | None = None,
    ) -> None:
        self._config_provider = config_provider
        self._chat_provider = chat_provider
        self._chat_uuid = chat_uuid
        self._user_uuid = user_uuid
        self._agent_id = agent_id
        self._context = context
        self._dependencies: UserChatRunImpl._Dependencies | None = None

    async def __aenter__(self) -> "UserChatRunImpl":
        self._dependencies = self._Dependencies(
            self._config_provider,
            self._chat_provider,
            user_uuid=self._user_uuid,
        )
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self._dependencies = None

    async def stream_async(
        self,
        query: str,
        *,
        mutex: IMutex | None = None,
        timeout: timedelta | None = None,
    ) -> AsyncIterator[tuple[AgentRunEvent, AgentRun]]:
        context_copy = self._chat_provider.get_chat_context(
            user_uuid=self._user_uuid,
            context=self._context,
            chat_uuid=self._chat_uuid,
        )
        dependencies = self._dependencies
        if dependencies is None:
            dependencies = self._Dependencies(
                self._config_provider,
                self._chat_provider,
                user_uuid=self._user_uuid,
            )
            self._dependencies = dependencies
        try:
            async with asyncio.timeout(timeout.total_seconds() if timeout else None):
                prepared = await SessionWarmup.take(self._agent_id)
                if prepared is None:
                    prepared = await build_chat_collaborators(self)
                agent, agent_tools, agent_skills = prepared

                async for event in agent.stream_async(
                    query=query,
                    agent_run_repository=dependencies.chat_repository,
                    agent_run_session_id=self._chat_uuid,
                    tool_retriever=agent_tools,
                    tool_ids=[],
                    skill_retriever=agent_skills,
                    skill_ids=[],
                    context=context_copy,
                ):
                    yield event
        except TimeoutError:
            logger.warning("Chat run timed out chat_uuid=%s", self._chat_uuid)
            raise
        finally:
            if mutex:
                await mutex.release_async()


async def build_chat_collaborators(run: IUserChatRun) -> tuple[Any, Any, Any]:
    """Build the agent, tool retriever, and skill retriever for one chat run."""
    chat_run = cast(UserChatRunImpl, run)
    dependencies = chat_run._dependencies
    if dependencies is None:
        dependencies = UserChatRunImpl._Dependencies(
            chat_run._config_provider,
            chat_run._chat_provider,
            user_uuid=chat_run._user_uuid,
        )
        chat_run._dependencies = dependencies
    agent = await create_agent_async(
        model_backend=dependencies.model_backend,
        model_config_repository=dependencies.model_repository,
        agent_backend=dependencies.agent_backend,
        agent_config_repository=dependencies.agent_repository,
        agent_config_id=chat_run._agent_id,
    )
    tools = await create_tool_retriever_async(
        tool_backend=dependencies.tool_backend,
        tools=None,
        tool_config_repository=dependencies.tool_repository,
        embedding_backend=dependencies.embedding_backend,
        embedding_config_repository=dependencies.embedding_repository,
        space_id=chat_run._user_uuid,
    )
    skills = await create_skill_retriever_async(
        tool_backend=dependencies.tool_backend,
        skill_config_repository=dependencies.skill_repository,
        embedding_backend=dependencies.embedding_backend,
        embedding_config_repository=dependencies.embedding_repository,
        space_id=chat_run._user_uuid,
    )
    return agent, tools, skills


class UserChatRunProviderImpl(IUserChatRunProvider):
    """Create user chat runs from registered config and chat providers."""

    def __init__(self, component_site: IComponentSite, **kwargs):
        self.component_site = component_site

    def create_chat_run(
        self,
        chat_uuid: str,
        *,
        user_uuid: str,
        agent_id: str = "default",
        context: dict | None = None,
        **kwargs,
    ) -> IUserChatRun:
        return UserChatRunImpl(
            query_component(self.component_site, IUserConfigProvider),
            query_component(self.component_site, IUserChatProvider),
            chat_uuid,
            user_uuid=user_uuid,
            agent_id=agent_id,
            context=context,
        )


class ModuleImpl(IModule):
    """Agent chats module implementation."""

    def __init__(self, component_site: IComponentSite, **kwargs):
        self._component_site = component_site
        self._jobs: list[IModuleJob] = [ChatDescribeJob(component_site)]
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
