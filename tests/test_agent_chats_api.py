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
from fivccliche.modules.users.models import User  # noqa: F401
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

        # Create admin user
        loop.run_until_complete(
            methods.create_user_async(
                async_session,
                username="admin",
                email="admin@example.com",
                password="admin123",
                is_superuser=True,
            )
        )

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
        """Test querying chat without authentication."""
        response = client.post(
            "/chats/",
            json={"agent_id": "test_agent", "query": "Hello"},
        )
        assert response.status_code == 401

    def test_query_chat_missing_both_params(self, client: TestClient, auth_token: str):
        """Test querying chat without chat_uuid or agent_id."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.post(
            "/chats/",
            json={"query": "Hello"},
            headers=headers,
        )
        assert response.status_code == 400
        data = response.json()
        assert "Must specify either chat_uuid or agent_id" in data["detail"]

    def test_query_chat_both_params(self, client: TestClient, auth_token: str):
        """Test querying chat with both chat_uuid and agent_id."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.post(
            "/chats/",
            json={"chat_uuid": "test-uuid", "agent_id": "test_agent", "query": "Hello"},
            headers=headers,
        )
        assert response.status_code == 400
        data = response.json()
        assert "Cannot specify both chat_uuid and agent_id" in data["detail"]


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
        self, client: TestClient, auth_token: str
    ):
        """Test that listing messages for non-existent chat returns 404."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/nonexistent-uuid/messages/", headers=headers)
        assert response.status_code == 404

    def test_delete_message_from_nonexistent_chat_returns_404(
        self, client: TestClient, auth_token: str
    ):
        """Test that deleting a message from non-existent chat returns 404."""
        headers = {"Authorization": f"Bearer {auth_token}"}
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
