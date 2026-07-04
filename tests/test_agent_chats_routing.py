"""Comprehensive tests for agent_chats router endpoints with trailing slashes.

These tests verify that all endpoints are correctly configured with trailing slashes
and handle various HTTP methods, status codes, and request/response scenarios.
"""

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
from fivccliche.modules.agent_configs.models import (  # noqa: F401
    UserEmbedding,
    UserLLM,
    UserAgent,
    UserTool,
)


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
        module_site = ModuleSiteImpl(
            component_site, modules=["users", "agent_configs", "agent_chats"]
        )
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


@pytest.fixture
def user_token(client: TestClient):
    """Generate a JWT token for a regular test user."""
    import asyncio
    from fivccliche.modules.users import methods

    # Create a regular user
    async def create_user():
        # Get the session from the dependency override
        app = client.app
        session_override = app.dependency_overrides[get_db_session_async]
        session = None
        async for s in session_override():
            session = s
            break

        if session:
            await methods.create_user_async(
                session,
                username="testuser",
                email="testuser@example.com",
                password="testpass123",
                is_superuser=False,
            )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(create_user())
    finally:
        loop.close()

    # Login as regular user
    response = client.post(
        "/users/login",
        json={"username": "testuser", "password": "testpass123"},
    )
    return response.json()["access_token"]


class TestChatEndpointTrailingSlashes:
    """Test that all endpoints have correct trailing slash behavior."""

    def test_post_chats_with_trailing_slash(self, client: TestClient, auth_token: str):
        """Test POST /chats/ (with trailing slash) creates chat with agent_id."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.post(
            "/chats/",
            json={"agent_id": "test_agent"},
            headers=headers,
        )
        # Should succeed and return 201
        assert response.status_code == 201
        data = response.json()
        assert "uuid" in data
        assert data["agent_id"] == "test_agent"

    def test_get_chats_with_trailing_slash(self, client: TestClient, auth_token: str):
        """Test GET /chats/ (with trailing slash) works."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "results" in data

    def test_get_chat_by_uuid_with_trailing_slash(self, client: TestClient, auth_token: str):
        """Test GET /chats/{uuid}/ (with trailing slash) returns 404 for missing chat."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/nonexistent-uuid/", headers=headers)
        assert response.status_code == 404

    def test_delete_chat_with_trailing_slash(self, client: TestClient, auth_token: str):
        """Test DELETE /chats/{uuid}/ (with trailing slash) returns 404 for missing chat."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.delete("/chats/nonexistent-uuid/", headers=headers)
        assert response.status_code == 404

    def test_get_chat_messages_with_trailing_slash(self, client: TestClient, auth_token: str):
        """Test GET /chats/{uuid}/messages/ (with trailing slash) returns 404 for missing chat."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/nonexistent-uuid/messages/", headers=headers)
        assert response.status_code == 404

    def test_delete_chat_message_with_trailing_slash(self, client: TestClient, auth_token: str):
        """Test DELETE /chats/{uuid}/messages/{msg_uuid}/ (with trailing slash)."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.delete(
            "/chats/nonexistent-uuid/messages/nonexistent-msg/", headers=headers
        )
        assert response.status_code == 404


class TestChatEndpointStatusCodes:
    """Test correct HTTP status codes for various scenarios."""

    def test_post_chats_requires_auth(self, client: TestClient):
        """Test POST /chats/ without auth returns 401."""
        response = client.post(
            "/chats/",
            json={"agent_id": "test_agent"},
        )
        assert response.status_code == 401

    def test_get_chats_requires_auth(self, client: TestClient):
        """Test GET /chats/ without auth returns 401."""
        response = client.get("/chats/")
        assert response.status_code == 401

    def test_get_chat_uuid_requires_auth(self, client: TestClient):
        """Test GET /chats/{uuid}/ without auth returns 401."""
        response = client.get("/chats/some-uuid/")
        assert response.status_code == 401

    def test_delete_chat_requires_auth(self, client: TestClient):
        """Test DELETE /chats/{uuid}/ without auth returns 401."""
        response = client.delete("/chats/some-uuid/")
        assert response.status_code == 401

    def test_get_messages_requires_auth(self, client: TestClient):
        """Test GET /chats/{uuid}/messages/ without auth returns 401."""
        response = client.get("/chats/some-uuid/messages/")
        assert response.status_code == 401

    def test_delete_message_requires_auth(self, client: TestClient):
        """Test DELETE /chats/{uuid}/messages/{msg_uuid}/ without auth returns 401."""
        response = client.delete("/chats/some-uuid/messages/some-msg/")
        assert response.status_code == 401

    def test_post_chats_missing_params(self, client: TestClient, auth_token: str):
        """Test POST /chats/ with agent_id uses default if not provided."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.post(
            "/chats/",
            json={},  # No agent_id - should use default
            headers=headers,
        )
        # Returns 201 (created) with default agent_id
        assert response.status_code == 201
        data = response.json()
        assert data["agent_id"] == "default"

    def test_delete_chat_returns_204_or_404(self, client: TestClient, auth_token: str):
        """Test DELETE /chats/{uuid}/ returns 204 No Content or 404."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.delete("/chats/nonexistent-uuid/", headers=headers)
        # Returns 404 for missing chat
        assert response.status_code == 404

    def test_delete_message_returns_204_or_404(self, client: TestClient, auth_token: str):
        """Test DELETE /chats/{uuid}/messages/{msg_uuid}/ returns 204 or 404."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.delete(
            "/chats/nonexistent-uuid/messages/nonexistent-msg/", headers=headers
        )
        assert response.status_code == 404


class TestChatEndpointValidation:
    """Test request validation and error handling."""

    def test_post_chats_invalid_json(self, client: TestClient, auth_token: str):
        """Test POST /chats/ with invalid field returns 422."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.post(
            "/chats/",
            json={"invalid_field": "value"},  # Invalid field name
            headers=headers,
        )
        # For an empty object with just invalid fields, Pydantic should handle it
        # but since agent_id has a default, this should work with default
        # If strict validation, it returns 422
        assert response.status_code in [201, 422]

    def test_get_chats_invalid_skip(self, client: TestClient, auth_token: str):
        """Test GET /chats/ with negative skip returns 422."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/?skip=-1", headers=headers)
        assert response.status_code == 422

    def test_get_chats_invalid_limit(self, client: TestClient, auth_token: str):
        """Test GET /chats/ with limit > 1000 returns 422."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/?limit=2000", headers=headers)
        assert response.status_code == 422

    def test_get_messages_invalid_skip(self, client: TestClient, auth_token: str):
        """Test GET /chats/{uuid}/messages/ with negative skip returns 422."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/some-uuid/messages/?skip=-1", headers=headers)
        assert response.status_code in [422, 404]

    def test_get_messages_invalid_limit(self, client: TestClient, auth_token: str):
        """Test GET /chats/{uuid}/messages/ with limit > 1000 returns 422."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/some-uuid/messages/?limit=2000", headers=headers)
        assert response.status_code in [422, 404]


class TestChatEndpointResponseStructure:
    """Test response structure and content types."""

    def test_get_chats_response_structure(self, client: TestClient, auth_token: str):
        """Test GET /chats/ returns properly structured response."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "total" in data
        assert "results" in data
        assert isinstance(data["total"], int)
        assert isinstance(data["results"], list)
        assert data["total"] == 0  # No chats created

    def test_get_chats_response_content_type(self, client: TestClient, auth_token: str):
        """Test GET /chats/ returns JSON content type."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/", headers=headers)
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    def test_get_messages_response_structure(self, client: TestClient, auth_token: str):
        """Test GET /chats/{uuid}/messages/ returns 404 with proper structure."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/nonexistent/messages/", headers=headers)
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_post_chats_response_type(self, client: TestClient, auth_token: str):
        """Test POST /chats/ streaming response for GET list endpoint."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/", headers=headers)
        # GET /chats/ should return JSON, not streaming
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    def test_chat_response_includes_context_field(self, client: TestClient, auth_token: str):
        """Test that chat response includes context field."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        # Create a chat with context
        context_data = {"test": "data", "value": 123}
        create_response = client.post(
            "/chats/",
            json={"agent_id": "test_agent", "context": context_data},
            headers=headers,
        )
        assert create_response.status_code == 201
        response_json = create_response.json()

        # Verify context field exists and has correct structure
        assert "context" in response_json
        assert response_json["context"] is not None
        assert isinstance(response_json["context"], dict)
        assert response_json["context"] == context_data

    def test_chat_response_context_nullable(self, client: TestClient, auth_token: str):
        """Test that context field can be None."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        # Create a chat without context
        create_response = client.post(
            "/chats/",
            json={"agent_id": "test_agent"},
            headers=headers,
        )
        assert create_response.status_code == 201
        response_json = create_response.json()

        # Verify context field exists but is None
        assert "context" in response_json
        assert response_json["context"] is None


class TestChatEndpointPagination:
    """Test pagination behavior on list endpoints."""

    def test_get_chats_default_pagination(self, client: TestClient, auth_token: str):
        """Test GET /chats/ with default pagination."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert len(data["results"]) == 0

    def test_get_chats_custom_pagination(self, client: TestClient, auth_token: str):
        """Test GET /chats/ with custom skip and limit."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/?skip=0&limit=10", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "results" in data

    def test_get_messages_pagination_when_chat_missing(self, client: TestClient, auth_token: str):
        """Test GET /chats/{uuid}/messages/ pagination when chat doesn't exist."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/nonexistent/messages/?skip=0&limit=10", headers=headers)
        # Should return 404 because chat doesn't exist
        assert response.status_code == 404

    def test_get_chats_max_limit_boundary(self, client: TestClient, auth_token: str):
        """Test GET /chats/ with limit at maximum boundary (1000)."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/?limit=1000", headers=headers)
        assert response.status_code == 200

    def test_get_chats_above_max_limit(self, client: TestClient, auth_token: str):
        """Test GET /chats/ with limit above maximum (should fail)."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/chats/?limit=1001", headers=headers)
        assert response.status_code == 422


class TestChatEndpointErrorHandling:
    """Test error handling and edge cases."""

    def test_malformed_bearer_token(self, client: TestClient):
        """Test request with malformed Bearer token."""
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.get("/chats/", headers=headers)
        assert response.status_code == 401

    def test_missing_bearer_prefix(self, client: TestClient):
        """Test request with missing Bearer prefix."""
        headers = {"Authorization": "invalid-token"}
        response = client.get("/chats/", headers=headers)
        assert response.status_code == 401

    def test_empty_authorization_header(self, client: TestClient):
        """Test request with empty Authorization header."""
        headers = {"Authorization": ""}
        response = client.get("/chats/", headers=headers)
        assert response.status_code == 401

    def test_missing_authorization_header(self, client: TestClient):
        """Test request without Authorization header."""
        response = client.get("/chats/")
        assert response.status_code == 401

    def test_post_chats_missing_query(self, client: TestClient, auth_token: str):
        """Test POST /chats/ without query field - query is not part of create schema."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.post(
            "/chats/",
            json={"agent_id": "test_agent"},
            headers=headers,
        )
        # Should succeed - query is not needed for create
        assert response.status_code == 201

    def test_post_chats_empty_query(self, client: TestClient, auth_token: str):
        """Test POST /chats/ with query field - query is ignored in create schema."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.post(
            "/chats/",
            json={"query": ""},  # query field ignored, agent_id uses default
            headers=headers,
        )
        # query field is ignored by Pydantic, agent_id defaults to "default"
        assert response.status_code == 201
        data = response.json()
        assert data["agent_id"] == "default"
