"""Unit tests for agent_chats service layer."""

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel

from fivccliche.modules.agent_chats import methods
from fivccliche.modules.users.models import User  # noqa: F401
from fivccliche.modules.agent_chats.models import UserChat, UserChatMessage

if TYPE_CHECKING:
    from fivccliche.modules.agent_chats.services import UserChatRepositoryImpl


@pytest.fixture
async def session():
    """Create a temporary SQLite database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        database_url = f"sqlite+aiosqlite:///{db_path}"

        engine = create_async_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=NullPool,
            echo=False,
        )

        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        # Create session
        async_session = AsyncSession(engine, expire_on_commit=False)
        try:
            yield async_session
        finally:
            await async_session.close()
            await engine.dispose()


@pytest.fixture
async def test_user(session: AsyncSession):
    """Create a test user."""
    from fivccliche.modules.users import methods as user_methods

    user = await user_methods.create_user_async(
        session,
        username="testuser",
        email="test@example.com",
        password="password123",
    )
    return user


@pytest.fixture
async def test_chat(session: AsyncSession, test_user):
    """Create a test chat."""
    chat = UserChat(
        user_uuid=test_user.uuid,
        agent_id="test_agent",
        description="Test chat",
    )
    session.add(chat)
    await session.commit()
    await session.refresh(chat)
    return chat


class TestChatMethods:
    """Test cases for chat service methods."""

    async def test_get_chat_async(self, session: AsyncSession, test_chat: UserChat):
        """Test getting a chat by UUID."""
        chat = await methods.get_chat_async(session, test_chat.uuid, test_chat.user_uuid)
        assert chat is not None
        assert chat.uuid == test_chat.uuid
        assert chat.agent_id == "test_agent"

    async def test_get_chat_async_not_found(self, session: AsyncSession, test_user):
        """Test getting a non-existent chat."""
        chat = await methods.get_chat_async(session, "nonexistent", test_user.uuid)
        assert chat is None

    async def test_get_chat_async_wrong_user(self, session: AsyncSession, test_chat: UserChat):
        """Test getting a chat with wrong user UUID."""
        chat = await methods.get_chat_async(session, test_chat.uuid, "wrong_user_uuid")
        assert chat is None

    async def test_list_chats_async(self, session: AsyncSession, test_user, test_chat: UserChat):
        """Test listing chats for a user."""
        chats = await methods.list_chats_async(session, test_user.uuid)
        assert len(chats) == 1
        assert chats[0].uuid == test_chat.uuid

    async def test_list_chats_async_empty(self, session: AsyncSession, test_user):
        """Test listing chats when none exist."""
        chats = await methods.list_chats_async(session, test_user.uuid)
        assert len(chats) == 0

    async def test_list_chats_async_multiple_users(self, session: AsyncSession, test_user):
        """Test that chats are isolated per user."""
        from fivccliche.modules.users import methods as user_methods

        # Create another user
        user2 = await user_methods.create_user_async(
            session,
            username="testuser2",
            email="test2@example.com",
            password="password123",
        )

        # Create chats for both users
        chat1 = UserChat(user_uuid=test_user.uuid, agent_id="agent1")
        chat2 = UserChat(user_uuid=user2.uuid, agent_id="agent2")
        session.add(chat1)
        session.add(chat2)
        await session.commit()

        # List chats for user1
        chats1 = await methods.list_chats_async(session, test_user.uuid)
        assert len(chats1) == 1
        assert chats1[0].user_uuid == test_user.uuid

        # List chats for user2
        chats2 = await methods.list_chats_async(session, user2.uuid)
        assert len(chats2) == 1
        assert chats2[0].user_uuid == user2.uuid

    async def test_list_chats_async_pagination(self, session: AsyncSession, test_user):
        """Test listing chats with pagination."""
        # Create multiple chats
        for i in range(5):
            chat = UserChat(
                user_uuid=test_user.uuid,
                agent_id=f"agent_{i}",
                description=f"Chat {i}",
            )
            session.add(chat)
        await session.commit()

        # Test pagination
        chats = await methods.list_chats_async(session, test_user.uuid, skip=0, limit=2)
        assert len(chats) == 2

        chats = await methods.list_chats_async(session, test_user.uuid, skip=2, limit=2)
        assert len(chats) == 2

    async def test_count_chats_async(self, session: AsyncSession, test_user):
        """Test counting chats for a user."""
        # Create multiple chats
        for i in range(3):
            chat = UserChat(
                user_uuid=test_user.uuid,
                agent_id=f"agent_{i}",
            )
            session.add(chat)
        await session.commit()

        count = await methods.count_chats_async(session, test_user.uuid)
        assert count == 3

    async def test_delete_chat_async(self, session: AsyncSession, test_chat: UserChat):
        """Test deleting a chat."""
        await methods.delete_chat_async(session, test_chat)
        chat = await methods.get_chat_async(session, test_chat.uuid, test_chat.user_uuid)
        assert chat is None


class TestChatMessageMethods:
    """Test cases for chat message service methods."""

    async def test_list_chat_messages_async(self, session: AsyncSession, test_chat: UserChat):
        """Test listing messages for a chat."""
        # Create messages
        for i in range(3):
            message = UserChatMessage(
                chat_uuid=test_chat.uuid,
                status="completed",
                query={"text": f"Query {i}"},
            )
            session.add(message)
        await session.commit()

        messages = await methods.list_chat_messages_async(session, test_chat.uuid)
        assert len(messages) == 3

    async def test_list_chat_messages_async_pagination(
        self, session: AsyncSession, test_chat: UserChat
    ):
        """Test listing messages with pagination."""
        # Create messages
        for _ in range(5):
            message = UserChatMessage(
                chat_uuid=test_chat.uuid,
                status="completed",
            )
            session.add(message)
        await session.commit()

        messages = await methods.list_chat_messages_async(session, test_chat.uuid, skip=0, limit=2)
        assert len(messages) == 2

    async def test_count_chat_messages_async(self, session: AsyncSession, test_chat: UserChat):
        """Test counting messages for a chat."""
        # Create messages
        for _ in range(4):
            message = UserChatMessage(chat_uuid=test_chat.uuid)
            session.add(message)
        await session.commit()

        count = await methods.count_chat_messages_async(session, test_chat.uuid)
        assert count == 4

    async def test_get_chat_message_async(self, session: AsyncSession, test_chat: UserChat):
        """Test getting a message by UUID."""
        message = UserChatMessage(chat_uuid=test_chat.uuid)
        session.add(message)
        await session.commit()
        await session.refresh(message)

        retrieved = await methods.get_chat_message_async(session, message.uuid, test_chat.uuid)
        assert retrieved is not None
        assert retrieved.uuid == message.uuid

    async def test_delete_chat_message_async(self, session: AsyncSession, test_chat: UserChat):
        """Test deleting a message."""
        message = UserChatMessage(chat_uuid=test_chat.uuid)
        session.add(message)
        await session.commit()

        await methods.delete_chat_message_async(session, message)
        retrieved = await methods.get_chat_message_async(session, message.uuid, test_chat.uuid)
        assert retrieved is None

    async def test_get_chat_message_async_not_found(
        self, session: AsyncSession, test_chat: UserChat
    ):
        """Test getting a non-existent message."""
        retrieved = await methods.get_chat_message_async(session, "nonexistent", test_chat.uuid)
        assert retrieved is None

    async def test_list_chat_messages_empty(self, session: AsyncSession, test_chat: UserChat):
        """Test listing messages when none exist."""
        messages = await methods.list_chat_messages_async(session, test_chat.uuid)
        assert len(messages) == 0

    async def test_list_chat_messages_ordered_by_created_at(
        self, session: AsyncSession, test_chat: UserChat
    ):
        """Test that messages are ordered by created_at."""
        import asyncio

        # Create messages with slight delays to ensure different timestamps
        messages_data = []
        for _ in range(3):
            message = UserChatMessage(chat_uuid=test_chat.uuid)
            session.add(message)
            messages_data.append(message)
            await asyncio.sleep(0.01)  # Small delay to ensure different timestamps

        await session.commit()

        # Retrieve messages
        messages = await methods.list_chat_messages_async(session, test_chat.uuid)
        assert len(messages) == 3

        # Verify they're ordered by created_at
        for i in range(len(messages) - 1):
            assert messages[i].created_at <= messages[i + 1].created_at

    async def test_chat_message_with_data(self, session: AsyncSession, test_chat: UserChat):
        """Test creating and retrieving a message with data."""
        query_data = {"text": "What is the weather?"}
        reply_data = {"text": "It's sunny"}
        tool_calls_data = [{"name": "get_weather", "args": {}}]

        message = UserChatMessage(
            chat_uuid=test_chat.uuid,
            status="completed",
            query=query_data,
            reply=reply_data,
            tool_calls=tool_calls_data,
        )
        session.add(message)
        await session.commit()
        await session.refresh(message)

        retrieved = await methods.get_chat_message_async(session, message.uuid, test_chat.uuid)
        assert retrieved is not None
        assert retrieved.query == query_data
        assert retrieved.reply == reply_data
        assert retrieved.tool_calls == tool_calls_data
        assert retrieved.status == "completed"

    async def test_count_chat_messages_empty(self, session: AsyncSession, test_chat: UserChat):
        """Test counting messages when none exist."""
        count = await methods.count_chat_messages_async(session, test_chat.uuid)
        assert count == 0

    async def test_list_chat_messages_different_chats(self, session: AsyncSession, test_user):
        """Test that messages are isolated per chat."""
        # Create two chats
        chat1 = UserChat(user_uuid=test_user.uuid, agent_id="agent1")
        chat2 = UserChat(user_uuid=test_user.uuid, agent_id="agent2")
        session.add(chat1)
        session.add(chat2)
        await session.commit()

        # Create messages for both chats
        for _ in range(2):
            msg1 = UserChatMessage(chat_uuid=chat1.uuid)
            msg2 = UserChatMessage(chat_uuid=chat2.uuid)
            session.add(msg1)
            session.add(msg2)
        await session.commit()

        # List messages for chat1
        messages1 = await methods.list_chat_messages_async(session, chat1.uuid)
        assert len(messages1) == 2
        assert all(m.chat_uuid == chat1.uuid for m in messages1)

        # List messages for chat2
        messages2 = await methods.list_chat_messages_async(session, chat2.uuid)
        assert len(messages2) == 2
        assert all(m.chat_uuid == chat2.uuid for m in messages2)


# ============================================================================
# UserChatRepositoryImpl Tests
# ============================================================================


class TestUserChatRepositoryImpl:
    """Test cases for UserChatRepositoryImpl."""

    @pytest.fixture
    def repository(self, session: AsyncSession, test_user):
        """Create a repository instance for testing."""
        from fivccliche.modules.agent_chats.services import UserChatRepositoryImpl

        return UserChatRepositoryImpl(user_uuid=test_user.uuid, session=session)

    # ========================================================================
    # Session Validation Tests
    # ========================================================================

    async def test_update_agent_run_session_missing_session(self, test_user):
        """Test that update_agent_run_session raises error when session is missing."""
        from fivccliche.modules.agent_chats.services import UserChatRepositoryImpl
        from fivcplayground.agents.types import AgentRunSession

        repo = UserChatRepositoryImpl(user_uuid=test_user.uuid, session=None)
        session_data = AgentRunSession(id="test-id", agent_id="agent1")

        with pytest.raises(ValueError, match="Session and user_uuid are required"):
            await repo.update_agent_run_session_async(session_data)

    async def test_update_agent_run_session_missing_user_uuid(self, session: AsyncSession):
        """Test that update_agent_run_session raises error when user_uuid is missing."""
        from fivccliche.modules.agent_chats.services import UserChatRepositoryImpl
        from fivcplayground.agents.types import AgentRunSession

        repo = UserChatRepositoryImpl(user_uuid=None, session=session)
        session_data = AgentRunSession(id="test-id", agent_id="agent1")

        with pytest.raises(ValueError, match="Session and user_uuid are required"):
            await repo.update_agent_run_session_async(session_data)

    # ========================================================================
    # Agent Run Session (Chat) Tests
    # ========================================================================

    async def test_update_agent_run_session_create_new(
        self, repository: "UserChatRepositoryImpl", test_user
    ):
        """Test creating a new agent run session."""
        from fivcplayground.agents.types import AgentRunSession

        session_data = AgentRunSession(
            id="chat-1",
            agent_id="agent1",
            description="Test chat session",
        )

        await repository.update_agent_run_session_async(session_data)

        # Verify it was created
        chat = await methods.get_chat_async(repository.session, "chat-1", test_user.uuid)
        assert chat is not None
        assert chat.uuid == "chat-1"
        assert chat.agent_id == "agent1"
        assert chat.description == "Test chat session"

    async def test_update_agent_run_session_update_existing(
        self, repository: "UserChatRepositoryImpl", test_user, test_chat: UserChat
    ):
        """Test updating an existing agent run session."""
        from fivcplayground.agents.types import AgentRunSession

        session_data = AgentRunSession(
            id=test_chat.uuid,
            agent_id=test_chat.agent_id,
            description="Updated description",
        )

        await repository.update_agent_run_session_async(session_data)

        # Verify it was updated
        chat = await methods.get_chat_async(repository.session, test_chat.uuid, test_user.uuid)
        assert chat is not None
        assert chat.description == "Updated description"

    async def test_get_agent_run_session_found(
        self, repository: "UserChatRepositoryImpl", test_user, test_chat: UserChat
    ):
        """Test retrieving an existing agent run session."""
        session_data = await repository.get_agent_run_session_async(test_chat.uuid)

        assert session_data is not None
        assert session_data.id == test_chat.uuid
        assert session_data.agent_id == test_chat.agent_id

    async def test_get_agent_run_session_not_found(self, repository: "UserChatRepositoryImpl"):
        """Test retrieving a non-existent agent run session."""
        session_data = await repository.get_agent_run_session_async("nonexistent")
        assert session_data is None

    async def test_get_agent_run_session_wrong_user(
        self, session: AsyncSession, test_chat: UserChat
    ):
        """Test that users cannot access other users' chats."""
        from fivccliche.modules.agent_chats.services import UserChatRepositoryImpl

        repo = UserChatRepositoryImpl(user_uuid="different_user", session=session)
        session_data = await repo.get_agent_run_session_async(test_chat.uuid)
        assert session_data is None

    async def test_list_agent_run_sessions_empty(self, repository: "UserChatRepositoryImpl"):
        """Test listing agent run sessions when none exist."""
        sessions = await repository.list_agent_run_sessions_async()
        assert len(sessions) == 0

    async def test_list_agent_run_sessions_multiple(
        self, repository: "UserChatRepositoryImpl", test_user
    ):
        """Test listing multiple agent run sessions."""
        # Create multiple chats
        for i in range(3):
            chat = UserChat(
                user_uuid=test_user.uuid,
                agent_id=f"agent_{i}",
                description=f"Chat {i}",
            )
            repository.session.add(chat)
        await repository.session.commit()

        sessions = await repository.list_agent_run_sessions_async()
        assert len(sessions) == 3

    async def test_list_agent_run_sessions_pagination(
        self, repository: "UserChatRepositoryImpl", test_user
    ):
        """Test listing agent run sessions with pagination."""
        # Create multiple chats
        for i in range(5):
            chat = UserChat(
                user_uuid=test_user.uuid,
                agent_id=f"agent_{i}",
            )
            repository.session.add(chat)
        await repository.session.commit()

        # Test pagination
        sessions = await repository.list_agent_run_sessions_async(skip=0, limit=2)
        assert len(sessions) == 2

        sessions = await repository.list_agent_run_sessions_async(skip=2, limit=2)
        assert len(sessions) == 2

    async def test_delete_agent_run_session(
        self, repository: "UserChatRepositoryImpl", test_user, test_chat: UserChat
    ):
        """Test deleting an agent run session."""
        await repository.delete_agent_run_session_async(test_chat.uuid)

        # Verify it was deleted
        chat = await methods.get_chat_async(repository.session, test_chat.uuid, test_user.uuid)
        assert chat is None

    async def test_delete_agent_run_session_not_found(self, repository: "UserChatRepositoryImpl"):
        """Test deleting a non-existent agent run session (should not raise)."""
        # Should not raise an error
        await repository.delete_agent_run_session_async("nonexistent")

    # ========================================================================
    # Agent Run (Chat Message) Tests
    # ========================================================================

    async def test_update_agent_run_create_new(
        self, repository: "UserChatRepositoryImpl", test_user, test_chat: UserChat
    ):
        """Test creating a new agent run (chat message)."""
        from fivcplayground.agents.types import AgentRun

        agent_run = AgentRun(
            id="run-1",
            agent_id="agent1",
            status="completed",
            query={"text": "Hello"},  # Use plain dict
            reply={"text": "Hi there"},  # Use plain dict
        )

        await repository.update_agent_run_async(test_chat.uuid, agent_run)

        # Verify it was created
        message = await methods.get_chat_message_async(repository.session, "run-1", test_chat.uuid)
        assert message is not None
        assert message.uuid == "run-1"
        assert message.chat_uuid == test_chat.uuid
        assert message.status == "completed"
        # Query and reply are stored as dicts
        assert message.query is not None
        assert message.reply is not None

    async def test_update_agent_run_update_existing(
        self, repository: "UserChatRepositoryImpl", test_user, test_chat: UserChat
    ):
        """Test updating an existing agent run (chat message)."""
        from fivcplayground.agents.types import AgentRun

        # Create initial message
        message = UserChatMessage(
            uuid="run-1",
            chat_uuid=test_chat.uuid,
            status="pending",
            query={"text": "Hello"},
        )
        repository.session.add(message)
        await repository.session.commit()

        # Update it
        agent_run = AgentRun(
            id="run-1",
            agent_id="agent1",
            status="completed",
            reply={"text": "Hi there"},  # Use plain dict
        )

        await repository.update_agent_run_async(test_chat.uuid, agent_run)

        # Verify it was updated
        updated = await methods.get_chat_message_async(repository.session, "run-1", test_chat.uuid)
        assert updated is not None
        assert updated.status == "completed"
        assert updated.reply is not None

    async def test_update_agent_run_chat_not_found(self, repository: "UserChatRepositoryImpl"):
        """Test that updating a message for non-existent chat raises error."""
        from fivcplayground.agents.types import AgentRun

        agent_run = AgentRun(id="run-1", agent_id="agent1")

        with pytest.raises(ValueError, match=r"Chat session .* not found"):
            await repository.update_agent_run_async("nonexistent_chat", agent_run)

    async def test_update_agent_run_wrong_user_chat(
        self, session: AsyncSession, test_chat: UserChat
    ):
        """Test that users cannot update messages in other users' chats."""
        from fivccliche.modules.agent_chats.services import UserChatRepositoryImpl
        from fivcplayground.agents.types import AgentRun

        repo = UserChatRepositoryImpl(user_uuid="different_user", session=session)
        agent_run = AgentRun(id="run-1", agent_id="agent1")

        with pytest.raises(ValueError, match=r"Chat session .* not found"):
            await repo.update_agent_run_async(test_chat.uuid, agent_run)

    async def test_get_agent_run_found(
        self, repository: "UserChatRepositoryImpl", test_user, test_chat: UserChat
    ):
        """Test retrieving an existing agent run (chat message)."""
        # Create a message
        message = UserChatMessage(
            uuid="run-1",
            chat_uuid=test_chat.uuid,
            status="completed",
            query={"text": "Hello"},
        )
        repository.session.add(message)
        await repository.session.commit()

        agent_run = await repository.get_agent_run_async(test_chat.uuid, "run-1")

        assert agent_run is not None
        assert agent_run.id == "run-1"

    async def test_get_agent_run_not_found(
        self, repository: "UserChatRepositoryImpl", test_chat: UserChat
    ):
        """Test retrieving a non-existent agent run."""
        agent_run = await repository.get_agent_run_async(test_chat.uuid, "nonexistent")
        assert agent_run is None

    async def test_get_agent_run_chat_not_found(self, repository: "UserChatRepositoryImpl"):
        """Test retrieving a message from non-existent chat returns None."""
        agent_run = await repository.get_agent_run_async("nonexistent_chat", "run-1")
        assert agent_run is None

    async def test_list_agent_runs_empty(
        self, repository: "UserChatRepositoryImpl", test_chat: UserChat
    ):
        """Test listing agent runs when none exist."""
        runs = await repository.list_agent_runs_async(test_chat.uuid)
        assert len(runs) == 0

    async def test_list_agent_runs_multiple(
        self, repository: "UserChatRepositoryImpl", test_chat: UserChat
    ):
        """Test listing multiple agent runs."""
        # Create multiple messages
        for i in range(3):
            message = UserChatMessage(
                chat_uuid=test_chat.uuid,
                status="completed",
                query={"text": f"Query {i}"},
            )
            repository.session.add(message)
        await repository.session.commit()

        runs = await repository.list_agent_runs_async(test_chat.uuid)
        assert len(runs) == 3

    async def test_list_agent_runs_pagination(
        self, repository: "UserChatRepositoryImpl", test_chat: UserChat
    ):
        """Test listing agent runs with pagination."""
        # Create multiple messages
        for _i in range(5):
            message = UserChatMessage(
                chat_uuid=test_chat.uuid,
                status="completed",
            )
            repository.session.add(message)
        await repository.session.commit()

        # Test pagination
        runs = await repository.list_agent_runs_async(test_chat.uuid, skip=0, limit=2)
        assert len(runs) == 2

        runs = await repository.list_agent_runs_async(test_chat.uuid, skip=2, limit=2)
        assert len(runs) == 2

    async def test_list_agent_runs_chat_not_found(self, repository: "UserChatRepositoryImpl"):
        """Test listing runs for non-existent chat returns empty list."""
        runs = await repository.list_agent_runs_async("nonexistent_chat")
        assert len(runs) == 0

    async def test_delete_agent_run(
        self, repository: "UserChatRepositoryImpl", test_user, test_chat: UserChat
    ):
        """Test deleting an agent run (chat message)."""
        # Create a message
        message = UserChatMessage(
            uuid="run-1",
            chat_uuid=test_chat.uuid,
            status="completed",
        )
        repository.session.add(message)
        await repository.session.commit()

        await repository.delete_agent_run_async(test_chat.uuid, "run-1")

        # Verify it was deleted
        deleted = await methods.get_chat_message_async(repository.session, "run-1", test_chat.uuid)
        assert deleted is None

    async def test_delete_agent_run_chat_not_found(self, repository: "UserChatRepositoryImpl"):
        """Test deleting a message from non-existent chat raises error."""
        with pytest.raises(ValueError, match=r"Chat session .* not found"):
            await repository.delete_agent_run_async("nonexistent_chat", "run-1")

    async def test_delete_agent_run_not_found(
        self, repository: "UserChatRepositoryImpl", test_chat: UserChat
    ):
        """Test deleting a non-existent message (should not raise)."""
        # Should not raise an error
        await repository.delete_agent_run_async(test_chat.uuid, "nonexistent")

    # ========================================================================
    # User Isolation Tests
    # ========================================================================

    async def test_user_isolation_list_sessions(self, session: AsyncSession, test_user):
        """Test that users can only see their own chat sessions."""
        from fivccliche.modules.agent_chats.services import UserChatRepositoryImpl
        from fivccliche.modules.users import methods as user_methods

        # Create another user
        user2 = await user_methods.create_user_async(
            session,
            username="testuser2",
            email="test2@example.com",
            password="password123",
        )

        # Create chats for both users
        chat1 = UserChat(user_uuid=test_user.uuid, agent_id="agent1")
        chat2 = UserChat(user_uuid=user2.uuid, agent_id="agent2")
        session.add(chat1)
        session.add(chat2)
        await session.commit()

        # User 1 should only see their chat
        repo1 = UserChatRepositoryImpl(user_uuid=test_user.uuid, session=session)
        sessions1 = await repo1.list_agent_run_sessions_async()
        assert len(sessions1) == 1
        assert sessions1[0].id == chat1.uuid

        # User 2 should only see their chat
        repo2 = UserChatRepositoryImpl(user_uuid=user2.uuid, session=session)
        sessions2 = await repo2.list_agent_run_sessions_async()
        assert len(sessions2) == 1
        assert sessions2[0].id == chat2.uuid

    async def test_user_isolation_get_session(self, session: AsyncSession, test_user):
        """Test that users cannot get other users' chat sessions."""
        from fivccliche.modules.agent_chats.services import UserChatRepositoryImpl
        from fivccliche.modules.users import methods as user_methods

        # Create another user
        user2 = await user_methods.create_user_async(
            session,
            username="testuser2",
            email="test2@example.com",
            password="password123",
        )

        # Create a chat for user2
        chat = UserChat(user_uuid=user2.uuid, agent_id="agent1")
        session.add(chat)
        await session.commit()

        # User 1 should not be able to get user2's chat
        repo1 = UserChatRepositoryImpl(user_uuid=test_user.uuid, session=session)
        session_data = await repo1.get_agent_run_session_async(chat.uuid)
        assert session_data is None

    async def test_user_isolation_delete_session(self, session: AsyncSession, test_user):
        """Test that users cannot delete other users' chat sessions."""
        from fivccliche.modules.agent_chats.services import UserChatRepositoryImpl
        from fivccliche.modules.users import methods as user_methods

        # Create another user
        user2 = await user_methods.create_user_async(
            session,
            username="testuser2",
            email="test2@example.com",
            password="password123",
        )

        # Create a chat for user2
        chat = UserChat(user_uuid=user2.uuid, agent_id="agent1")
        session.add(chat)
        await session.commit()

        # User 1 tries to delete user2's chat (should not delete)
        repo1 = UserChatRepositoryImpl(user_uuid=test_user.uuid, session=session)
        await repo1.delete_agent_run_session_async(chat.uuid)

        # Chat should still exist
        chat_check = await methods.get_chat_async(session, chat.uuid, user2.uuid)
        assert chat_check is not None

    async def test_user_isolation_list_runs(self, session: AsyncSession, test_user):
        """Test that users can only see messages in their own chats."""
        from fivccliche.modules.agent_chats.services import UserChatRepositoryImpl
        from fivccliche.modules.users import methods as user_methods

        # Create another user
        user2 = await user_methods.create_user_async(
            session,
            username="testuser2",
            email="test2@example.com",
            password="password123",
        )

        # Create chats for both users
        chat1 = UserChat(user_uuid=test_user.uuid, agent_id="agent1")
        chat2 = UserChat(user_uuid=user2.uuid, agent_id="agent2")
        session.add(chat1)
        session.add(chat2)
        await session.commit()

        # Add messages to both chats
        msg1 = UserChatMessage(chat_uuid=chat1.uuid)
        msg2 = UserChatMessage(chat_uuid=chat2.uuid)
        session.add(msg1)
        session.add(msg2)
        await session.commit()

        # User 1 should not be able to list messages from user2's chat
        repo1 = UserChatRepositoryImpl(user_uuid=test_user.uuid, session=session)
        runs = await repo1.list_agent_runs_async(chat2.uuid)
        assert len(runs) == 0

    # ========================================================================
    # Synchronous Wrapper Methods Tests
    # ========================================================================

    def test_update_agent_run_session_sync(self, repository: "UserChatRepositoryImpl", test_user):
        """Test synchronous wrapper for update_agent_run_session."""
        from fivcplayground.agents.types import AgentRunSession

        session_data = AgentRunSession(
            id="chat-sync-1",
            agent_id="agent1",
            description="Test sync chat",
        )

        # Call synchronous method
        repository.update_agent_run_session(session_data)

        # Verify it was created (using async method)
        import asyncio

        async def verify():
            chat = await methods.get_chat_async(repository.session, "chat-sync-1", test_user.uuid)
            return chat

        chat = asyncio.run(verify())
        assert chat is not None
        assert chat.uuid == "chat-sync-1"

    def test_get_agent_run_session_sync(
        self, repository: "UserChatRepositoryImpl", test_chat: UserChat
    ):
        """Test synchronous wrapper for get_agent_run_session."""
        session_data = repository.get_agent_run_session(test_chat.uuid)
        assert session_data is not None
        assert session_data.id == test_chat.uuid

    def test_list_agent_run_sessions_sync(self, repository: "UserChatRepositoryImpl", test_user):
        """Test synchronous wrapper for list_agent_run_sessions."""
        import asyncio

        # Create a chat first
        async def create_chat():
            chat = UserChat(user_uuid=test_user.uuid, agent_id="agent1")
            repository.session.add(chat)
            await repository.session.commit()

        asyncio.run(create_chat())

        # Call synchronous method
        sessions = repository.list_agent_run_sessions()
        assert len(sessions) == 1

    def test_delete_agent_run_session_sync(
        self, repository: "UserChatRepositoryImpl", test_chat: UserChat
    ):
        """Test synchronous wrapper for delete_agent_run_session."""
        import asyncio

        # Delete using sync method
        repository.delete_agent_run_session(test_chat.uuid)

        # Verify it was deleted
        async def verify():
            chat = await methods.get_chat_async(
                repository.session, test_chat.uuid, test_chat.user_uuid
            )
            return chat

        chat = asyncio.run(verify())
        assert chat is None

    def test_update_agent_run_sync(self, repository: "UserChatRepositoryImpl", test_chat: UserChat):
        """Test synchronous wrapper for update_agent_run."""
        from fivcplayground.agents.types import AgentRun

        agent_run = AgentRun(
            id="run-sync-1",
            agent_id="agent1",
            status="completed",
        )

        # Call synchronous method
        repository.update_agent_run(test_chat.uuid, agent_run)

        # Verify it was created
        import asyncio

        async def verify():
            message = await methods.get_chat_message_async(
                repository.session, "run-sync-1", test_chat.uuid
            )
            return message

        message = asyncio.run(verify())
        assert message is not None
        assert message.uuid == "run-sync-1"

    def test_get_agent_run_sync(self, repository: "UserChatRepositoryImpl", test_chat: UserChat):
        """Test synchronous wrapper for get_agent_run."""
        import asyncio

        # Create a message first
        async def create_message():
            message = UserChatMessage(uuid="run-sync-1", chat_uuid=test_chat.uuid)
            repository.session.add(message)
            await repository.session.commit()

        asyncio.run(create_message())

        # Call synchronous method
        agent_run = repository.get_agent_run(test_chat.uuid, "run-sync-1")
        assert agent_run is not None
        assert agent_run.id == "run-sync-1"

    def test_list_agent_runs_sync(self, repository: "UserChatRepositoryImpl", test_chat: UserChat):
        """Test synchronous wrapper for list_agent_runs."""
        import asyncio

        # Create messages first
        async def create_messages():
            for _i in range(2):
                message = UserChatMessage(chat_uuid=test_chat.uuid)
                repository.session.add(message)
            await repository.session.commit()

        asyncio.run(create_messages())

        # Call synchronous method
        runs = repository.list_agent_runs(test_chat.uuid)
        assert len(runs) == 2

    def test_delete_agent_run_sync(self, repository: "UserChatRepositoryImpl", test_chat: UserChat):
        """Test synchronous wrapper for delete_agent_run."""
        import asyncio

        # Create a message first
        async def create_message():
            message = UserChatMessage(uuid="run-sync-1", chat_uuid=test_chat.uuid)
            repository.session.add(message)
            await repository.session.commit()

        asyncio.run(create_message())

        # Delete using sync method
        repository.delete_agent_run(test_chat.uuid, "run-sync-1")

        # Verify it was deleted
        async def verify():
            message = await methods.get_chat_message_async(
                repository.session, "run-sync-1", test_chat.uuid
            )
            return message

        message = asyncio.run(verify())
        assert message is None
