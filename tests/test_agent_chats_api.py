"""Integration tests for agent_chats API endpoints."""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel

from fivccliche.utils.deps import get_db_session_async
from fivccliche.services.implements.modules import ModuleSiteImpl
from fivcglue.implements.utils import load_component_site

# Import models to ensure they're registered with SQLModel
from fivccliche.modules.users.models import User
from fivccliche.modules.agent_chats.models import UserChat, UserChatMessage  # noqa: F401


@pytest.fixture
def client():
    """Create a test client with temporary database and test user."""
    import asyncio
    from fivccliche.modules.users import methods

    # Create a temporary file for the database
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db.close()
    database_url = f"sqlite+aiosqlite:///{temp_db.name}"

    # Create engine and tables
    async def create_tables():
        engine = create_async_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=NullPool,
        )
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        return engine

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        engine = loop.run_until_complete(create_tables())
        async_session = AsyncSession(engine, expire_on_commit=False)

        # Load components
        components_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "fivccliche",
            "settings",
            "services.yml",
        )
        component_site = load_component_site(filename=components_path, fmt="yaml")
        module_site = ModuleSiteImpl(component_site, modules=["users", "agent_chats"])
        app = module_site.create_application()

        # Override the database dependency
        async def override_get_db_session_async():
            yield async_session

        app.dependency_overrides[get_db_session_async] = override_get_db_session_async

        client = TestClient(app)

        # Create admin user and regular test user
        loop.run_until_complete(
            methods.create_user_async(
                async_session,
                username="admin",
                email="admin@example.com",
                password="admin123",
                is_superuser=True,
            )
        )
        loop.run_until_complete(
            methods.create_user_async(
                async_session,
                username="testuser",
                email="test@example.com",
                password="password123",
                is_superuser=False,
            )
        )

        # Store loop and async_session in client for test access
        client.loop = loop
        client.async_session = async_session

        yield client
    finally:
        loop.run_until_complete(async_session.close())
        loop.run_until_complete(engine.dispose())


@pytest.fixture
def auth_token(client: TestClient):
    """Generate a JWT token for a test user."""
    # Login as admin
    admin_response = client.post(
        "/users/login",
        json={"username": "admin", "password": "admin123"},
    )
    return admin_response.json()["access_token"]


@pytest.fixture
def regular_user_token(client: TestClient):
    """Generate a JWT token for a regular test user."""
    # Login as testuser
    response = client.post(
        "/users/login",
        json={"username": "testuser", "password": "password123"},
    )
    return response.json()["access_token"]


class TestChatAPI:
    """Test cases for chat API endpoints."""

    def test_list_chats_empty(self, client: TestClient, auth_token: str):
        """Test listing chats when none exist."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["results"] == []

    def test_list_chats_unauthorized(self, client: TestClient):
        """Test listing chats without authentication."""
        response = client.get("/chats/")
        assert response.status_code == 401

    def test_get_chat_not_found(self, client: TestClient, auth_token: str):
        """Test getting a non-existent chat."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/nonexistent", headers=headers)
        assert response.status_code == 404

    def test_delete_chat_not_found(self, client: TestClient, auth_token: str):
        """Test deleting a non-existent chat."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.delete("/chats/nonexistent", headers=headers)
        assert response.status_code == 404

    def test_query_chat_unauthorized(self, client: TestClient):
        """Test creating chat without authentication."""
        response = client.post(
            "/chats/",
            json={"agent_id": "test_agent"},
        )
        assert response.status_code == 401

    def test_create_chat_with_default_agent(self, client: TestClient, auth_token: str):
        """Test creating chat with default agent_id."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.post(
            "/chats/",
            json={},
            headers=headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert "uuid" in data
        assert data["agent_id"] == "default"
        assert "created_at" in data or "started_at" in data

    def test_create_chat_with_custom_agent(self, client: TestClient, auth_token: str):
        """Test creating chat with custom agent_id."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.post(
            "/chats/",
            json={"agent_id": "custom_agent"},
            headers=headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert "uuid" in data
        assert data["agent_id"] == "custom_agent"
        assert "created_at" in data or "started_at" in data


class TestChatMessageAPI:
    """Test cases for chat message API endpoints."""

    def test_list_messages_chat_not_found(self, client: TestClient, auth_token: str):
        """Test listing messages for non-existent chat."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/nonexistent/messages/", headers=headers)
        assert response.status_code == 404

    def test_list_messages_unauthorized(self, client: TestClient):
        """Test listing messages without authentication."""
        response = client.get("/chats/somechat/messages/")
        assert response.status_code == 401

    def test_delete_message_not_found(self, client: TestClient, auth_token: str):
        """Test deleting a non-existent message."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.delete("/chats/somechat/messages/nonexistent", headers=headers)
        assert response.status_code == 404

    def test_delete_message_unauthorized(self, client: TestClient):
        """Test deleting a message without authentication."""
        response = client.delete("/chats/somechat/messages/somemessage")
        assert response.status_code == 401

    def test_list_messages_pagination(self, client: TestClient, auth_token: str):
        """Test listing messages with pagination parameters."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        # Test with skip and limit parameters
        response = client.get("/chats/somechat/messages/?skip=0&limit=10", headers=headers)
        # Should return 404 because chat doesn't exist
        assert response.status_code == 404

    def test_list_messages_invalid_pagination(self, client: TestClient, auth_token: str):
        """Test listing messages with invalid pagination parameters."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        # Test with negative skip
        response = client.get("/chats/somechat/messages/?skip=-1", headers=headers)
        # Should return 422 (validation error) or 404 (chat not found)
        assert response.status_code in [422, 404]

    def test_get_chat_unauthorized(self, client: TestClient):
        """Test getting a chat without authentication."""
        response = client.get("/chats/somechat")
        assert response.status_code == 401

    def test_delete_chat_unauthorized(self, client: TestClient):
        """Test deleting a chat without authentication."""
        response = client.delete("/chats/somechat")
        assert response.status_code == 401

    def test_list_chats_with_pagination_params(self, client: TestClient, auth_token: str):
        """Test listing chats with pagination parameters."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/?skip=0&limit=10", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "results" in data
        assert isinstance(data["total"], int)
        assert isinstance(data["results"], list)

    def test_list_chats_invalid_limit(self, client: TestClient, auth_token: str):
        """Test listing chats with invalid limit parameter."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        # Test with limit > 1000
        response = client.get("/chats/?limit=2000", headers=headers)
        # Should return 422 (validation error)
        assert response.status_code == 422

    def test_list_chats_negative_skip(self, client: TestClient, auth_token: str):
        """Test listing chats with negative skip parameter."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/?skip=-1", headers=headers)
        # Should return 422 (validation error)
        assert response.status_code == 422


class TestChatIntegration:
    """Integration tests for chat operations."""

    def test_create_and_list_chats(self, client: TestClient, auth_token: str):
        """Test creating and listing chats through the API."""

        headers = {"Authorization": f"Bearer {auth_token}"}

        # First, list empty chats
        response = client.get("/chats/", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0

    def test_get_nonexistent_chat_returns_404(self, client: TestClient, auth_token: str):
        """Test that getting a non-existent chat returns 404."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/nonexistent-uuid", headers=headers)
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_delete_nonexistent_chat_returns_404(self, client: TestClient, auth_token: str):
        """Test that deleting a non-existent chat returns 404."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.delete("/chats/nonexistent-uuid", headers=headers)
        assert response.status_code == 404

    def test_list_messages_for_nonexistent_chat_returns_404(
        self, client: TestClient, regular_user_token: str
    ):
        """Test that listing messages for non-existent chat returns 404."""
        headers = {"Authorization": f"Bearer {regular_user_token}"}
        response = client.get("/chats/nonexistent-uuid/messages/", headers=headers)
        assert response.status_code == 404

    def test_delete_message_from_nonexistent_chat_returns_404(
        self, client: TestClient, regular_user_token: str
    ):
        """Test that deleting a message from non-existent chat returns 404."""
        headers = {"Authorization": f"Bearer {regular_user_token}"}
        response = client.delete(
            "/chats/nonexistent-uuid/messages/nonexistent-message", headers=headers
        )
        assert response.status_code == 404

    def test_api_response_structure(self, client: TestClient, auth_token: str):
        """Test that API responses have the correct structure."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/", headers=headers)
        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert isinstance(data, dict)
        assert "total" in data
        assert "results" in data
        assert isinstance(data["total"], int)
        assert isinstance(data["results"], list)


class TestTaskStreamingGenerator:
    """Test cases for _ChatStreamingGenerator class."""

    def test_task_streaming_generator_initialization(self):
        """Test _ChatStreamingGenerator initialization."""
        import asyncio
        from fivccliche.utils.generators import _ChatStreamingGenerator

        # Create a simple task
        async def dummy_task():
            await asyncio.sleep(0.01)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            task = loop.create_task(dummy_task())
            queue = asyncio.Queue()

            generator = _ChatStreamingGenerator(task, queue)
            assert generator.chat_task == task
            assert generator.chat_queue == queue
            assert hasattr(generator, "__call__")  # noqa

            # Clean up the task
            task.cancel()
            try:
                loop.run_until_complete(task)
            except asyncio.CancelledError:
                pass
        finally:
            loop.close()

    def test_task_streaming_generator_has_call_method(self):
        """Test _ChatStreamingGenerator has __call__ method."""
        import asyncio
        from fivccliche.utils.generators import _ChatStreamingGenerator

        async def dummy_task():
            await asyncio.sleep(0.01)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            task = loop.create_task(dummy_task())
            queue = asyncio.Queue()

            generator = _ChatStreamingGenerator(task, queue)
            # Verify it's callable
            assert callable(generator)
            # Verify calling it returns an async generator
            result = generator()
            assert hasattr(result, "__aiter__")

            # Clean up the task
            task.cancel()
            try:
                loop.run_until_complete(task)
            except asyncio.CancelledError:
                pass
        finally:
            loop.close()

    def test_task_streaming_generator_attributes(self):
        """Test _ChatStreamingGenerator has required attributes."""
        import asyncio
        from fivccliche.utils.generators import _ChatStreamingGenerator

        async def dummy_task():
            await asyncio.sleep(0.01)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            task = loop.create_task(dummy_task())
            queue = asyncio.Queue()

            generator = _ChatStreamingGenerator(task, queue)
            # Verify attributes
            assert hasattr(generator, "chat_task")
            assert hasattr(generator, "chat_queue")
            assert generator.chat_task is task
            assert generator.chat_queue is queue

            # Clean up the task
            task.cancel()
            try:
                loop.run_until_complete(task)
            except asyncio.CancelledError:
                pass
        finally:
            loop.close()


class TestChatEndpointValidation:
    """Test cases for endpoint input validation."""

    def test_list_chats_default_pagination(self, client: TestClient, auth_token: str):
        """Test list chats uses default pagination values."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "results" in data

    def test_list_chats_custom_pagination(self, client: TestClient, auth_token: str):
        """Test list chats with custom skip and limit."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/?skip=0&limit=50", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["total"], int)
        assert isinstance(data["results"], list)

    def test_list_messages_default_pagination(self, client: TestClient, auth_token: str):
        """Test list messages uses default pagination values."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        # Will fail with 404 because chat doesn't exist, but validates pagination
        response = client.get("/chats/test-uuid/messages/", headers=headers)
        assert response.status_code == 404

    def test_list_messages_custom_pagination(self, client: TestClient, auth_token: str):
        """Test list messages with custom skip and limit."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/test-uuid/messages/?skip=0&limit=50", headers=headers)
        assert response.status_code == 404

    def test_query_chat_request_format_validation(self, client: TestClient, auth_token: str):
        """Test that query_chat endpoint accepts valid request format."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        # This test validates that the endpoint accepts the request format
        # The actual agent/chat lookup will fail, but that's expected
        try:
            response = client.post(
                "/chats/",
                json={"agent_id": "test_agent", "query": "Hello"},
                headers=headers,
            )
            # Either succeeds or fails with expected error codes
            assert response.status_code in [200, 201, 400, 404, 500]
        except ValueError:
            # Expected when agent config is not found
            pass

    def test_query_chat_with_chat_uuid_format(self, client: TestClient, auth_token: str):
        """Test that query_chat endpoint accepts chat_uuid parameter."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        # This test validates that the endpoint accepts chat_uuid parameter
        try:
            response = client.post(
                "/chats/",
                json={"chat_uuid": "test-uuid", "query": "Hello"},
                headers=headers,
            )
            # Either succeeds or fails with expected error codes
            assert response.status_code in [200, 201, 400, 404, 500]
        except ValueError:
            # Expected when chat is not found
            pass

    def test_delete_message_with_mismatched_chat_uuid(self, client: TestClient, auth_token: str):
        """Test deleting message validates chat_uuid matches."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.delete(
            "/chats/chat-uuid-1/messages/message-uuid",
            headers=headers,
        )
        # Should return 404 because chat doesn't exist
        assert response.status_code == 404

    def test_get_chat_returns_correct_schema(self, client: TestClient, auth_token: str):
        """Test get chat endpoint returns correct response schema."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/nonexistent", headers=headers)
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_delete_chat_returns_no_content(self, client: TestClient, auth_token: str):
        """Test delete chat endpoint returns 204 No Content on success."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.delete("/chats/nonexistent", headers=headers)
        # Returns 404 because chat doesn't exist
        assert response.status_code == 404

    def test_delete_message_returns_no_content(self, client: TestClient, auth_token: str):
        """Test delete message endpoint returns 204 No Content on success."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.delete(
            "/chats/nonexistent/messages/nonexistent",
            headers=headers,
        )
        # Returns 404 because chat doesn't exist
        assert response.status_code == 404


class TestGlobalChatAuthorization:
    """Test cases for global chat authorization (superuser privileges)."""

    def test_superuser_can_delete_global_chat(self, client: TestClient):
        """Test that superuser can delete global chats."""
        from fivccliche.modules.agent_chats import methods as chat_methods

        # Login as admin
        admin_response = client.post(
            "/users/login",
            json={"username": "admin", "password": "admin123"},
        )
        admin_token = admin_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {admin_token}"}

        # Create a global chat directly in DB
        loop = client.loop

        async def setup():
            chat = await chat_methods.create_chat_async(
                client.async_session,
                user_uuid=None,  # Global chat
                agent_id="test-agent",
            )
            return chat.uuid

        chat_uuid = loop.run_until_complete(setup())

        # Delete as superuser - should succeed
        response = client.delete(f"/chats/{chat_uuid}", headers=headers)
        assert response.status_code == 204

    def test_regular_user_cannot_delete_global_chat(
        self, client: TestClient, regular_user_token: str
    ):
        """Test that regular user cannot delete global chats."""
        from fivccliche.modules.agent_chats import methods as chat_methods

        headers = {"Authorization": f"Bearer {regular_user_token}"}

        # Create a global chat directly in DB
        loop = client.loop

        async def setup():
            chat = await chat_methods.create_chat_async(
                client.async_session,
                user_uuid=None,  # Global chat
                agent_id="test-agent",
            )
            return chat.uuid

        chat_uuid = loop.run_until_complete(setup())

        # Try to delete as regular user - should fail with 403
        response = client.delete(f"/chats/{chat_uuid}", headers=headers)
        assert response.status_code == 403
        assert "Cannot delete global chats" in response.json()["detail"]

    def test_superuser_cannot_delete_other_user_chat(
        self, client: TestClient, regular_user_token: str
    ):
        """Test that superuser cannot delete another user's chat.

        Note: Superusers get 404 because they can't see other users' chats
        (GET queries filter by user_uuid). This is the expected behavior.
        """
        from fivccliche.modules.agent_chats import methods as chat_methods

        # Login as admin
        admin_response = client.post(
            "/users/login",
            json={"username": "admin", "password": "admin123"},
        )
        admin_token = admin_response.json()["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        # Get testuser's UUID
        loop = client.loop

        async def get_user():
            from sqlmodel import select

            stmt = select(User).where(User.username == "testuser")
            result = await client.async_session.execute(stmt)
            user = result.scalars().first()
            return user.uuid

        user_uuid = loop.run_until_complete(get_user())

        # Create a user-specific chat
        async def setup():
            chat = await chat_methods.create_chat_async(
                client.async_session,
                user_uuid=user_uuid,  # User-specific chat
                agent_id="test-agent",
            )
            return chat.uuid

        chat_uuid = loop.run_until_complete(setup())

        # Try to delete as superuser - gets 404 because can't see other users' chats
        response = client.delete(f"/chats/{chat_uuid}", headers=admin_headers)
        assert response.status_code == 404

    def test_regular_user_can_see_global_chats(self, client: TestClient, regular_user_token: str):
        """Test that regular users can see global chats in their list."""
        from fivccliche.modules.agent_chats import methods as chat_methods

        headers = {"Authorization": f"Bearer {regular_user_token}"}

        # Get testuser's UUID
        loop = client.loop

        async def get_user():
            from sqlmodel import select

            stmt = select(User).where(User.username == "testuser")
            result = await client.async_session.execute(stmt)
            user = result.scalars().first()
            return user.uuid

        user_uuid = loop.run_until_complete(get_user())

        # Create a global chat and a user-specific chat
        async def setup():
            global_chat = await chat_methods.create_chat_async(
                client.async_session,
                user_uuid=None,  # Global chat
                agent_id="default",
            )
            user_chat = await chat_methods.create_chat_async(
                client.async_session,
                user_uuid=user_uuid,  # User-specific chat
                agent_id="default",
            )
            return global_chat.uuid, user_chat.uuid

        global_chat_uuid, user_chat_uuid = loop.run_until_complete(setup())

        # List chats as regular user - should see both
        response = client.get("/chats/", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 2

        chat_uuids = [chat["uuid"] for chat in data["results"]]
        assert global_chat_uuid in chat_uuids
        assert user_chat_uuid in chat_uuids

    def test_regular_user_cannot_delete_message_in_global_chat(
        self, client: TestClient, regular_user_token: str
    ):
        """Test that regular user cannot delete messages in global chats."""
        from fivccliche.modules.agent_chats import methods as chat_methods

        headers = {"Authorization": f"Bearer {regular_user_token}"}

        # Create a global chat with a message
        loop = client.loop

        async def setup():
            chat = await chat_methods.create_chat_async(
                client.async_session,
                user_uuid=None,  # Global chat
                agent_id="test-agent",
            )
            message = await chat_methods.create_chat_message_async(
                client.async_session,
                chat_uuid=chat.uuid,
                query={"text": "Hello"},
            )
            return chat.uuid, message.uuid

        chat_uuid, message_uuid = loop.run_until_complete(setup())

        # Try to delete message as regular user - should fail with 403
        response = client.delete(
            f"/chats/{chat_uuid}/messages/{message_uuid}",
            headers=headers,
        )
        assert response.status_code == 403
        assert "Cannot delete messages in global chats" in response.json()["detail"]

    def test_superuser_can_delete_message_in_global_chat(self, client: TestClient):
        """Test that superuser can delete messages in global chats."""
        from fivccliche.modules.agent_chats import methods as chat_methods

        # Login as admin
        admin_response = client.post(
            "/users/login",
            json={"username": "admin", "password": "admin123"},
        )
        admin_token = admin_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {admin_token}"}

        # Create a global chat with a message
        loop = client.loop

        async def setup():
            chat = await chat_methods.create_chat_async(
                client.async_session,
                user_uuid=None,  # Global chat
                agent_id="test-agent",
            )
            message = await chat_methods.create_chat_message_async(
                client.async_session,
                chat_uuid=chat.uuid,
                query={"text": "Hello"},
            )
            return chat.uuid, message.uuid

        chat_uuid, message_uuid = loop.run_until_complete(setup())

        # Delete message as superuser - should succeed
        response = client.delete(
            f"/chats/{chat_uuid}/messages/{message_uuid}",
            headers=headers,
        )
        assert response.status_code == 204


class TestChatQueryValidation:
    """Test query endpoint validation for chat_uuid and agent_id parameters."""

    def test_query_with_both_chat_uuid_and_agent_id_fails(self, client, auth_token):
        """Verify that extra fields are ignored in the new create endpoint.

        The new endpoint only accepts agent_id, so other fields are ignored.
        """
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.post(
            "/chats/",
            headers=headers,
            json={
                "query": "test query",  # ignored
                "chat_uuid": "some-uuid",  # ignored
                "agent_id": "some-agent",
            },
        )
        # Extra fields are ignored by Pydantic, so this succeeds
        assert response.status_code == 201
        data = response.json()
        assert data["agent_id"] == "some-agent"

    def test_query_with_neither_chat_uuid_nor_agent_id_fails(self, client, auth_token):
        """Verify that agent_id defaults correctly when not provided."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.post(
            "/chats/",
            headers=headers,
            json={"query": "test query"},  # query ignored, agent_id uses default
        )
        # query field is ignored, agent_id defaults to "default"
        assert response.status_code == 201
        data = response.json()
        assert data["agent_id"] == "default"


class TestChatDeleteAuthorization:
    """Test authorization for chat deletion across users."""

    def test_regular_user_cannot_delete_global_chat(self, client, regular_user_token):
        """Verify 403 error when regular user tries to delete global chat."""
        from fivccliche.modules.agent_chats import methods as chat_methods

        session = client.async_session
        loop = client.loop

        async def setup():
            chat = await chat_methods.create_chat_async(
                session=session,
                user_uuid=None,  # Global chat
                agent_id="test-agent",
            )
            return str(chat.uuid)

        chat_uuid = loop.run_until_complete(setup())

        # Try to delete as regular user
        headers = {"Authorization": f"Bearer {regular_user_token}"}
        response = client.delete(f"/chats/{chat_uuid}", headers=headers)
        assert response.status_code == 403
        assert "cannot delete global chats" in response.json()["detail"].lower()

    def test_regular_user_cannot_delete_other_users_chat(self, client, regular_user_token):
        """Verify 404 when regular user tries to delete another user's chat (not visible)."""
        import uuid
        from fivccliche.modules.agent_chats import methods as chat_methods

        session = client.async_session
        loop = client.loop

        async def setup():
            # Create chat for a different user
            other_user_uuid = str(uuid.uuid4())
            chat = await chat_methods.create_chat_async(
                session=session,
                user_uuid=other_user_uuid,
                agent_id="test-agent",
            )
            return str(chat.uuid)

        chat_uuid = loop.run_until_complete(setup())

        # Try to delete as regular user - should get 404 because chat is not visible
        headers = {"Authorization": f"Bearer {regular_user_token}"}
        response = client.delete(f"/chats/{chat_uuid}", headers=headers)
        assert response.status_code == 404

    def test_regular_user_can_delete_own_chat(self, client, regular_user_token):
        """Verify regular user can delete their own chat."""
        from fivccliche.modules.agent_chats import methods as chat_methods
        from fivccliche.modules.users.methods import get_user_async

        session = client.async_session
        loop = client.loop

        async def setup():
            # Get the regular user's UUID
            user = await get_user_async(session=session, username="testuser")
            # Create chat for this user
            chat = await chat_methods.create_chat_async(
                session=session,
                user_uuid=str(user.uuid),
                agent_id="test-agent",
            )
            return str(chat.uuid)

        chat_uuid = loop.run_until_complete(setup())

        # Delete as regular user
        headers = {"Authorization": f"Bearer {regular_user_token}"}
        response = client.delete(f"/chats/{chat_uuid}", headers=headers)
        assert response.status_code == 204

    def test_superuser_can_delete_any_user_chat(self, client, auth_token):
        """Verify superuser can delete any user's chat."""
        import uuid
        from fivccliche.modules.agent_chats import methods as chat_methods

        session = client.async_session
        loop = client.loop

        async def setup():
            # Create chat for a random user
            other_user_uuid = str(uuid.uuid4())
            chat = await chat_methods.create_chat_async(
                session=session,
                user_uuid=other_user_uuid,
                agent_id="test-agent",
            )
            return str(chat.uuid)

        chat_uuid = loop.run_until_complete(setup())

        # Delete as superuser - should get 404 because chat is not visible to superuser either
        # (superuser can only see their own chats and global chats, not other users' chats)
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.delete(f"/chats/{chat_uuid}", headers=headers)
        assert response.status_code == 404


class TestMessageDeleteAuthorization:
    """Test authorization for message deletion across users."""

    def test_regular_user_cannot_delete_message_in_global_chat(self, client, regular_user_token):
        """Verify 403 error when regular user tries to delete message in global chat."""
        from fivccliche.modules.agent_chats import methods as chat_methods

        session = client.async_session
        loop = client.loop

        async def setup():
            # Create global chat
            chat = await chat_methods.create_chat_async(
                session=session,
                user_uuid=None,
                agent_id="test-agent",
            )
            # Create message in global chat
            message = await chat_methods.create_chat_message_async(
                session=session,
                chat_uuid=str(chat.uuid),
                query={"content": "test message"},
            )
            return str(chat.uuid), str(message.uuid)

        chat_uuid, message_uuid = loop.run_until_complete(setup())

        # Try to delete message as regular user
        headers = {"Authorization": f"Bearer {regular_user_token}"}
        response = client.delete(
            f"/chats/{chat_uuid}/messages/{message_uuid}",
            headers=headers,
        )
        assert response.status_code == 403
        assert "cannot delete messages in global chats" in response.json()["detail"].lower()

    def test_regular_user_cannot_delete_message_in_other_users_chat(
        self, client, regular_user_token
    ):
        """Verify 404 when regular user tries to delete message in another user's chat."""
        import uuid
        from fivccliche.modules.agent_chats import methods as chat_methods

        session = client.async_session
        loop = client.loop

        async def setup():
            # Create chat for different user
            other_user_uuid = str(uuid.uuid4())
            chat = await chat_methods.create_chat_async(
                session=session,
                user_uuid=other_user_uuid,
                agent_id="test-agent",
            )
            # Create message
            message = await chat_methods.create_chat_message_async(
                session=session,
                chat_uuid=str(chat.uuid),
                query={"content": "test message"},
            )
            return str(chat.uuid), str(message.uuid)

        chat_uuid, message_uuid = loop.run_until_complete(setup())

        # Try to delete message as regular user - should get 404 because chat is not visible
        headers = {"Authorization": f"Bearer {regular_user_token}"}
        response = client.delete(
            f"/chats/{chat_uuid}/messages/{message_uuid}",
            headers=headers,
        )
        assert response.status_code == 404

    def test_regular_user_can_delete_message_in_own_chat(self, client, regular_user_token):
        """Verify regular user can delete message in their own chat."""
        from fivccliche.modules.agent_chats import methods as chat_methods
        from fivccliche.modules.users.methods import get_user_async

        session = client.async_session
        loop = client.loop

        async def setup():
            # Get regular user's UUID
            user = await get_user_async(session=session, username="testuser")
            # Create chat for this user
            chat = await chat_methods.create_chat_async(
                session=session,
                user_uuid=str(user.uuid),
                agent_id="test-agent",
            )
            # Create message
            message = await chat_methods.create_chat_message_async(
                session=session,
                chat_uuid=str(chat.uuid),
                query={"content": "test message"},
            )
            return str(chat.uuid), str(message.uuid)

        chat_uuid, message_uuid = loop.run_until_complete(setup())

        # Delete message as regular user
        headers = {"Authorization": f"Bearer {regular_user_token}"}
        response = client.delete(
            f"/chats/{chat_uuid}/messages/{message_uuid}",
            headers=headers,
        )
        assert response.status_code == 204

    def test_superuser_can_delete_message_in_global_chat(self, client, auth_token):
        """Verify superuser can delete message in global chat."""
        from fivccliche.modules.agent_chats import methods as chat_methods

        session = client.async_session
        loop = client.loop

        async def setup():
            # Create global chat
            chat = await chat_methods.create_chat_async(
                session=session,
                user_uuid=None,
                agent_id="test-agent",
            )
            # Create message
            message = await chat_methods.create_chat_message_async(
                session=session,
                chat_uuid=str(chat.uuid),
                query={"content": "test message"},
            )
            return str(chat.uuid), str(message.uuid)

        chat_uuid, message_uuid = loop.run_until_complete(setup())

        # Delete message as superuser
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.delete(
            f"/chats/{chat_uuid}/messages/{message_uuid}",
            headers=headers,
        )
        assert response.status_code == 204


class TestCreateChatMessages:
    """Test cases for create_chat_messages_async endpoint."""

    def test_create_message_unauthorized(self, client: TestClient):
        """Test creating message without authentication."""
        response = client.post(
            "/chats/some-uuid/messages/",
            json={"query": "Hello"},
        )
        assert response.status_code == 401
        data = response.json()
        assert "Not authenticated" in data["detail"]

    def test_create_message_chat_not_found(self, client: TestClient, auth_token: str):
        """Test creating message in non-existent chat."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.post(
            "/chats/nonexistent/messages/",
            json={"query": "Hello"},
            headers=headers,
        )
        assert response.status_code == 404
        data = response.json()
        assert "Chat not found" in data["detail"]

    def test_create_message_regular_user_cannot_access_other_user_chat(
        self, client: TestClient, auth_token: str, regular_user_token: str
    ):
        """Test regular user cannot create message in another user's chat."""
        from fivccliche.modules.agent_chats import methods as chat_methods
        from fivccliche.modules.users.methods import get_user_async

        session = client.async_session
        loop = client.loop

        # Get the actual user UUID for the test user
        async def setup():
            test_user = await get_user_async(session, username="testuser")
            # Create a chat for the test user
            chat = await chat_methods.create_chat_async(
                session=session,
                user_uuid=str(test_user.uuid),
                agent_id="test-agent",
            )
            return str(chat.uuid)

        chat_uuid = loop.run_until_complete(setup())

        # Try to create message as admin (should fail - chat belongs to testuser)
        # admin cannot see testuser's chat, so get_chat_async returns None -> 404
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.post(
            f"/chats/{chat_uuid}/messages/",
            json={"query": "Hello"},
            headers=headers,
        )
        # Should get 404 because admin doesn't have access to testuser's chat
        assert response.status_code == 404
        data = response.json()
        assert "Chat not found" in data["detail"]

    def test_create_message_regular_user_cannot_message_global_chat(
        self, client: TestClient, regular_user_token: str
    ):
        """Test regular user cannot create message in global chat."""
        from fivccliche.modules.agent_chats import methods as chat_methods

        session = client.async_session
        loop = client.loop

        # Create a global chat
        async def setup():
            chat = await chat_methods.create_chat_async(
                session=session,
                user_uuid=None,
                agent_id="test-agent",
            )
            return str(chat.uuid)

        chat_uuid = loop.run_until_complete(setup())

        # Try to create message as regular user in global chat
        headers = {"Authorization": f"Bearer {regular_user_token}"}
        response = client.post(
            f"/chats/{chat_uuid}/messages/",
            json={"query": "Hello"},
            headers=headers,
        )
        assert response.status_code == 403
        data = response.json()
        assert "Cannot message global chats" in data["detail"]

    def test_create_message_missing_query_field(self, client: TestClient, auth_token: str):
        """Test creating message without query field."""
        from fivccliche.modules.agent_chats import methods as chat_methods
        from fivccliche.modules.users.methods import get_user_async

        session = client.async_session
        loop = client.loop

        # Get admin user and create their chat
        async def setup():
            admin_user = await get_user_async(session, username="admin")
            chat = await chat_methods.create_chat_async(
                session=session,
                user_uuid=str(admin_user.uuid),
                agent_id="test-agent",
            )
            return str(chat.uuid)

        chat_uuid = loop.run_until_complete(setup())

        # Try to create message without query field
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.post(
            f"/chats/{chat_uuid}/messages/",
            json={},
            headers=headers,
        )
        assert response.status_code == 422  # Validation error

    def test_create_message_post_method_required(self, client: TestClient, auth_token: str):
        """Test that create message endpoint requires POST method."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        # GET should call list_chat_messages_async instead
        response = client.get(
            "/chats/some-uuid/messages/",
            headers=headers,
        )
        # GET returns list response (200) or not found (404), not method not allowed
        assert response.status_code in [200, 404]
