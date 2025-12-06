"""Unit tests for agent_chats service layer."""

import tempfile
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel

from fivccliche.modules.agent_chats import methods
from fivccliche.modules.users.models import User  # noqa: F401
from fivccliche.modules.agent_chats.models import Chat, ChatMessage


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
    chat = Chat(
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

    async def test_get_chat_async(self, session: AsyncSession, test_chat: Chat):
        """Test getting a chat by UUID."""
        chat = await methods.get_chat_async(session, test_chat.uuid, test_chat.user_uuid)
        assert chat is not None
        assert chat.uuid == test_chat.uuid
        assert chat.agent_id == "test_agent"

    async def test_get_chat_async_not_found(self, session: AsyncSession, test_user):
        """Test getting a non-existent chat."""
        chat = await methods.get_chat_async(session, "nonexistent", test_user.uuid)
        assert chat is None

    async def test_get_chat_async_wrong_user(self, session: AsyncSession, test_chat: Chat):
        """Test getting a chat with wrong user UUID."""
        chat = await methods.get_chat_async(session, test_chat.uuid, "wrong_user_uuid")
        assert chat is None

    async def test_list_chats_async(self, session: AsyncSession, test_user, test_chat: Chat):
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
        chat1 = Chat(user_uuid=test_user.uuid, agent_id="agent1")
        chat2 = Chat(user_uuid=user2.uuid, agent_id="agent2")
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
            chat = Chat(
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
            chat = Chat(
                user_uuid=test_user.uuid,
                agent_id=f"agent_{i}",
            )
            session.add(chat)
        await session.commit()

        count = await methods.count_chats_async(session, test_user.uuid)
        assert count == 3

    async def test_delete_chat_async(self, session: AsyncSession, test_chat: Chat):
        """Test deleting a chat."""
        await methods.delete_chat_async(session, test_chat)
        chat = await methods.get_chat_async(session, test_chat.uuid, test_chat.user_uuid)
        assert chat is None


class TestChatMessageMethods:
    """Test cases for chat message service methods."""

    async def test_list_chat_messages_async(self, session: AsyncSession, test_chat: Chat):
        """Test listing messages for a chat."""
        # Create messages
        for i in range(3):
            message = ChatMessage(
                chat_uuid=test_chat.uuid,
                status="completed",
                query={"text": f"Query {i}"},
            )
            session.add(message)
        await session.commit()

        messages = await methods.list_chat_messages_async(session, test_chat.uuid)
        assert len(messages) == 3

    async def test_list_chat_messages_async_pagination(
        self, session: AsyncSession, test_chat: Chat
    ):
        """Test listing messages with pagination."""
        # Create messages
        for _ in range(5):
            message = ChatMessage(
                chat_uuid=test_chat.uuid,
                status="completed",
            )
            session.add(message)
        await session.commit()

        messages = await methods.list_chat_messages_async(session, test_chat.uuid, skip=0, limit=2)
        assert len(messages) == 2

    async def test_count_chat_messages_async(self, session: AsyncSession, test_chat: Chat):
        """Test counting messages for a chat."""
        # Create messages
        for _ in range(4):
            message = ChatMessage(chat_uuid=test_chat.uuid)
            session.add(message)
        await session.commit()

        count = await methods.count_chat_messages_async(session, test_chat.uuid)
        assert count == 4

    async def test_get_chat_message_async(self, session: AsyncSession, test_chat: Chat):
        """Test getting a message by UUID."""
        message = ChatMessage(chat_uuid=test_chat.uuid)
        session.add(message)
        await session.commit()
        await session.refresh(message)

        retrieved = await methods.get_chat_message_async(session, message.uuid, test_chat.uuid)
        assert retrieved is not None
        assert retrieved.uuid == message.uuid

    async def test_delete_chat_message_async(self, session: AsyncSession, test_chat: Chat):
        """Test deleting a message."""
        message = ChatMessage(chat_uuid=test_chat.uuid)
        session.add(message)
        await session.commit()

        await methods.delete_chat_message_async(session, message)
        retrieved = await methods.get_chat_message_async(session, message.uuid, test_chat.uuid)
        assert retrieved is None

    async def test_get_chat_message_async_not_found(self, session: AsyncSession, test_chat: Chat):
        """Test getting a non-existent message."""
        retrieved = await methods.get_chat_message_async(session, "nonexistent", test_chat.uuid)
        assert retrieved is None

    async def test_list_chat_messages_empty(self, session: AsyncSession, test_chat: Chat):
        """Test listing messages when none exist."""
        messages = await methods.list_chat_messages_async(session, test_chat.uuid)
        assert len(messages) == 0

    async def test_list_chat_messages_ordered_by_created_at(
        self, session: AsyncSession, test_chat: Chat
    ):
        """Test that messages are ordered by created_at."""
        import asyncio

        # Create messages with slight delays to ensure different timestamps
        messages_data = []
        for _ in range(3):
            message = ChatMessage(chat_uuid=test_chat.uuid)
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

    async def test_chat_message_with_data(self, session: AsyncSession, test_chat: Chat):
        """Test creating and retrieving a message with data."""
        query_data = {"text": "What is the weather?"}
        reply_data = {"text": "It's sunny"}
        tool_calls_data = [{"name": "get_weather", "args": {}}]

        message = ChatMessage(
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

    async def test_count_chat_messages_empty(self, session: AsyncSession, test_chat: Chat):
        """Test counting messages when none exist."""
        count = await methods.count_chat_messages_async(session, test_chat.uuid)
        assert count == 0

    async def test_list_chat_messages_different_chats(self, session: AsyncSession, test_user):
        """Test that messages are isolated per chat."""
        # Create two chats
        chat1 = Chat(user_uuid=test_user.uuid, agent_id="agent1")
        chat2 = Chat(user_uuid=test_user.uuid, agent_id="agent2")
        session.add(chat1)
        session.add(chat2)
        await session.commit()

        # Create messages for both chats
        for _ in range(2):
            msg1 = ChatMessage(chat_uuid=chat1.uuid)
            msg2 = ChatMessage(chat_uuid=chat2.uuid)
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
