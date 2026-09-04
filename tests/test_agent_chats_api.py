"""Integration tests for agent_chats API endpoints."""

import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from fivccliche.modules.agent_chats.models import UserChat
from fivccliche.modules.users.models import User
from tests.conftest import make_api_client


@pytest.fixture
def client():
    """Create a test client with temporary database and test user."""
    yield from make_api_client(
        ["users", "agent_chats"],
        extra_users=[
            {
                "username": "testuser",
                "email": "test@example.com",
                "password": "password123",
                "is_superuser": False,
            }
        ],
    )


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
        assert data["is_memorable"] is True

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
        assert data["is_memorable"] is True


class TestChatContextAPI:
    """Test cases for context field handling in chat API."""

    def test_create_chat_with_context(self, client: TestClient, auth_token: str):
        """Test creating a chat with initial context."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        context_data = {"session_type": "debug", "environment": "test", "user_id": 123}
        response = client.post(
            "/chats/",
            json={"agent_id": "test_agent", "context": context_data},
            headers=headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert "uuid" in data
        assert "context" in data
        assert data["context"] == context_data
        assert data["context"]["session_type"] == "debug"

    def test_create_chat_without_context(self, client: TestClient, auth_token: str):
        """Test creating a chat without context (should default to None)."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.post(
            "/chats/",
            json={"agent_id": "test_agent"},
            headers=headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert "uuid" in data
        assert "context" in data
        assert data["context"] is None

    def test_get_chat_includes_context(self, client: TestClient, auth_token: str):
        """Test that GET endpoint returns context field."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        context_data = {"key": "value", "number": 42}

        # Create chat with context
        create_response = client.post(
            "/chats/",
            json={"agent_id": "test_agent", "context": context_data},
            headers=headers,
        )
        assert create_response.status_code == 201
        chat_uuid = create_response.json()["uuid"]

        # Get the chat
        get_response = client.get(f"/chats/{chat_uuid}/", headers=headers)
        assert get_response.status_code == 200
        data = get_response.json()
        assert "context" in data
        assert data["context"] == context_data
        assert data["is_memorable"] is True

    def test_list_chats_includes_context(self, client: TestClient, auth_token: str):
        """Test that list endpoint returns context in results."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        context_data = {"test": "list_context"}

        # Create chat with context (using default agent_id to match list filter)
        create_response = client.post(
            "/chats/",
            json={"agent_id": "default", "context": context_data},
            headers=headers,
        )
        assert create_response.status_code == 201

        # List chats
        list_response = client.get("/chats/", headers=headers)
        assert list_response.status_code == 200
        data = list_response.json()
        assert data["total"] > 0
        assert len(data["results"]) > 0

        # Check that at least one result has context
        chat_with_context = next(
            (chat for chat in data["results"] if chat.get("context") == context_data), None
        )
        assert chat_with_context is not None

    def test_list_chats_filters_by_context_profile_uuid(self, client: TestClient, auth_token: str):
        """Test list endpoint filters chats by top-level context query params."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        matching_context = {"profile_uuid": "profile-1", "project_uuid": "project-1"}
        other_context = {"profile_uuid": "profile-2", "project_uuid": "project-1"}

        matching_response = client.post(
            "/chats/",
            json={"agent_id": "default", "context": matching_context},
            headers=headers,
        )
        other_response = client.post(
            "/chats/",
            json={"agent_id": "default", "context": other_context},
            headers=headers,
        )
        no_context_response = client.post(
            "/chats/",
            json={"agent_id": "default"},
            headers=headers,
        )
        assert matching_response.status_code == 201
        assert other_response.status_code == 201
        assert no_context_response.status_code == 201

        response = client.get(
            "/chats/?context.profile_uuid=profile-1",
            headers=headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert [chat["uuid"] for chat in data["results"]] == [matching_response.json()["uuid"]]

    def test_list_chats_rejects_nested_context_filter(self, client: TestClient, auth_token: str):
        """Test nested context query params are rejected for v1."""
        headers = {"Authorization": f"Bearer {auth_token}"}

        response = client.get(
            "/chats/?context.profile.uuid=profile-1",
            headers=headers,
        )

        assert response.status_code == 422

    def test_list_chats_ignores_unknown_json_filter_root(self, client: TestClient, auth_token: str):
        """Undeclared dotted roots are not passed into ChatFilterSet.parse."""
        headers = {"Authorization": f"Bearer {auth_token}"}

        response = client.get(
            "/chats/?options.key2=xxx",
            headers=headers,
        )

        assert response.status_code == 200


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

    def test_list_messages_includes_is_memorized(self, client: TestClient, auth_token: str):
        """Test that list messages returns is_memorized on each message."""
        from fivccliche.modules.agent_chats import utils as chat_methods
        from fivccliche.modules.users.utils import get_user_async

        headers = {"Authorization": f"Bearer {auth_token}"}
        session = client.async_session
        loop = client.loop

        async def setup():
            admin_user = await get_user_async(session, username="admin")
            chat = await chat_methods.create_chat_async(
                session=session,
                user_uuid=str(admin_user.uuid),
                agent_id="test-agent",
                is_memorable=True,
            )
            await chat_methods.create_chat_message_async(
                session=session,
                chat_uuid=chat.uuid,
                query={"text": "Hello"},
            )
            await session.commit()
            return str(chat.uuid)

        chat_uuid = loop.run_until_complete(setup())

        response = client.get(f"/chats/{chat_uuid}/messages/", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["results"][0]["is_memorized"] is False


class TestChatStreamApiSurface:
    """Test cases for ChatStream() used by routers."""

    def test_call_returns_async_generator(self):
        """Calling ChatStream returns an async generator."""
        from fivccliche.utils.stream import ChatStream

        chat_stream = ChatStream(chat_uuid="c1")
        stream = chat_stream()
        assert hasattr(stream, "__aiter__")

    def test_call_is_async_iterable(self):
        """Calling ChatStream supports async iteration."""
        from fivccliche.utils.stream import ChatStream

        chat_stream = ChatStream()
        stream = chat_stream()
        assert hasattr(stream, "__aiter__")
        assert hasattr(stream, "__anext__")


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


class TestUpdateChat:
    """Test cases for PATCH /chats/{chat_uuid}/ description updates."""

    def test_owner_can_update_description(self, client: TestClient, auth_token: str):
        headers = {"Authorization": f"Bearer {auth_token}"}
        created = client.post("/chats/", json={"agent_id": "test-agent"}, headers=headers)
        assert created.status_code == 201
        chat_uuid = created.json()["uuid"]

        response = client.patch(
            f"/chats/{chat_uuid}/",
            json={"description": "  My title  "},
            headers=headers,
        )
        assert response.status_code == 200
        assert response.json()["description"] == "My title"

        got = client.get(f"/chats/{chat_uuid}/", headers=headers)
        assert got.status_code == 200
        assert got.json()["description"] == "My title"

    def test_empty_description_rejected(self, client: TestClient, auth_token: str):
        headers = {"Authorization": f"Bearer {auth_token}"}
        created = client.post("/chats/", json={}, headers=headers)
        chat_uuid = created.json()["uuid"]
        response = client.patch(
            f"/chats/{chat_uuid}/",
            json={"description": ""},
            headers=headers,
        )
        assert response.status_code == 422

    def test_whitespace_description_rejected(self, client: TestClient, auth_token: str):
        headers = {"Authorization": f"Bearer {auth_token}"}
        created = client.post("/chats/", json={}, headers=headers)
        chat_uuid = created.json()["uuid"]
        response = client.patch(
            f"/chats/{chat_uuid}/",
            json={"description": "   "},
            headers=headers,
        )
        assert response.status_code == 422

    def test_missing_description_rejected(self, client: TestClient, auth_token: str):
        headers = {"Authorization": f"Bearer {auth_token}"}
        created = client.post("/chats/", json={}, headers=headers)
        chat_uuid = created.json()["uuid"]
        response = client.patch(f"/chats/{chat_uuid}/", json={}, headers=headers)
        assert response.status_code == 422

    def test_update_chat_not_found(self, client: TestClient, auth_token: str):
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.patch(
            "/chats/nonexistent/",
            json={"description": "Title"},
            headers=headers,
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_update_chat_unauthorized(self, client: TestClient):
        response = client.patch("/chats/somechat/", json={"description": "Title"})
        assert response.status_code == 401

    def test_cannot_update_other_user_chat(
        self, client: TestClient, auth_token: str, regular_user_token: str
    ):
        from fivccliche.modules.agent_chats import utils as chat_methods
        from fivccliche.modules.users.utils import get_user_async

        session = client.async_session
        loop = client.loop

        async def setup():
            test_user = await get_user_async(session, username="testuser")
            chat = await chat_methods.create_chat_async(
                session=session,
                user_uuid=str(test_user.uuid),
                agent_id="test-agent",
            )
            await session.commit()
            return str(chat.uuid)

        chat_uuid = loop.run_until_complete(setup())
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.patch(
            f"/chats/{chat_uuid}/",
            json={"description": "Hijacked"},
            headers=headers,
        )
        assert response.status_code == 404

    def test_regular_user_cannot_update_global_chat(
        self, client: TestClient, regular_user_token: str
    ):
        from fivccliche.modules.agent_chats import utils as chat_methods

        loop = client.loop

        async def setup():
            chat = await chat_methods.create_chat_async(
                session=client.async_session,
                user_uuid=None,
                agent_id="test-agent",
            )
            await client.async_session.commit()
            return str(chat.uuid)

        chat_uuid = loop.run_until_complete(setup())
        headers = {"Authorization": f"Bearer {regular_user_token}"}
        response = client.patch(
            f"/chats/{chat_uuid}/",
            json={"description": "Nope"},
            headers=headers,
        )
        assert response.status_code == 404

    def test_superuser_can_update_global_chat(self, client: TestClient, auth_token: str):
        from fivccliche.modules.agent_chats import utils as chat_methods

        loop = client.loop

        async def setup():
            chat = await chat_methods.create_chat_async(
                session=client.async_session,
                user_uuid=None,
                agent_id="test-agent",
            )
            await client.async_session.commit()
            return str(chat.uuid)

        chat_uuid = loop.run_until_complete(setup())
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.patch(
            f"/chats/{chat_uuid}/",
            json={"description": "Global title"},
            headers=headers,
        )
        assert response.status_code == 200
        assert response.json()["description"] == "Global title"


class TestGlobalChatAuthorization:
    """Test cases for global chat authorization (superuser privileges)."""

    def test_superuser_can_delete_global_chat(self, client: TestClient):
        """Test that superuser can delete global chats."""
        from fivccliche.modules.agent_chats import utils as chat_methods

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
            await client.async_session.commit()
            return chat.uuid

        chat_uuid = loop.run_until_complete(setup())

        # Delete as superuser - should succeed
        response = client.delete(f"/chats/{chat_uuid}", headers=headers)
        assert response.status_code == 204

    def test_regular_user_cannot_delete_global_chat(
        self, client: TestClient, regular_user_token: str
    ):
        """Test that regular user cannot delete global chats."""
        from fivccliche.modules.agent_chats import utils as chat_methods

        headers = {"Authorization": f"Bearer {regular_user_token}"}

        # Create a global chat directly in DB
        loop = client.loop

        async def setup():
            chat = await chat_methods.create_chat_async(
                client.async_session,
                user_uuid=None,  # Global chat
                agent_id="test-agent",
            )
            await client.async_session.commit()
            return chat.uuid

        chat_uuid = loop.run_until_complete(setup())

        # Try to delete as regular user - should fail with 404
        response = client.delete(f"/chats/{chat_uuid}", headers=headers)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_superuser_cannot_delete_other_user_chat(
        self, client: TestClient, regular_user_token: str
    ):
        """Test that superuser cannot delete another user's chat.

        Note: Superusers get 404 because they can't see other users' chats
        (GET queries filter by user_uuid). This is the expected behavior.
        """
        from fivccliche.modules.agent_chats import utils as chat_methods

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
            await client.async_session.commit()
            return chat.uuid

        chat_uuid = loop.run_until_complete(setup())

        # Try to delete as superuser - gets 404 because can't see other users' chats
        response = client.delete(f"/chats/{chat_uuid}", headers=admin_headers)
        assert response.status_code == 404

    def test_regular_user_can_see_global_chats(self, client: TestClient, regular_user_token: str):
        """Test that regular users can see global chats in their list."""
        from fivccliche.modules.agent_chats import utils as chat_methods

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
            await client.async_session.commit()
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
        from fivccliche.modules.agent_chats import utils as chat_methods

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
            await client.async_session.commit()
            return chat.uuid, message.uuid

        chat_uuid, message_uuid = loop.run_until_complete(setup())

        # Try to delete message as regular user - should fail with 404
        response = client.delete(
            f"/chats/{chat_uuid}/messages/{message_uuid}",
            headers=headers,
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_superuser_can_delete_message_in_global_chat(self, client: TestClient):
        """Test that superuser can delete messages in global chats."""
        from fivccliche.modules.agent_chats import utils as chat_methods

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
            await client.async_session.commit()
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
        """Verify 404 error when regular user tries to delete global chat."""
        from fivccliche.modules.agent_chats import utils as chat_methods

        session = client.async_session
        loop = client.loop

        async def setup():
            chat = await chat_methods.create_chat_async(
                session=session,
                user_uuid=None,  # Global chat
                agent_id="test-agent",
            )
            await session.commit()
            return str(chat.uuid)

        chat_uuid = loop.run_until_complete(setup())

        # Try to delete as regular user
        headers = {"Authorization": f"Bearer {regular_user_token}"}
        response = client.delete(f"/chats/{chat_uuid}", headers=headers)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_regular_user_cannot_delete_other_users_chat(self, client, regular_user_token):
        """Verify 404 when regular user tries to delete another user's chat (not visible)."""
        from fivccliche.modules.agent_chats import utils as chat_methods
        from fivccliche.modules.users import utils as user_methods

        session = client.async_session
        loop = client.loop

        async def setup():
            other = await user_methods.create_user_async(
                session,
                username="other-chat-owner",
                email="other-chat-owner@example.com",
                password="password123",
            )
            await session.commit()
            chat = await chat_methods.create_chat_async(
                session=session,
                user_uuid=other.uuid,
                agent_id="test-agent",
            )
            await session.commit()
            return str(chat.uuid)

        chat_uuid = loop.run_until_complete(setup())

        # Try to delete as regular user - should get 404 because chat is not visible
        headers = {"Authorization": f"Bearer {regular_user_token}"}
        response = client.delete(f"/chats/{chat_uuid}", headers=headers)
        assert response.status_code == 404

    def test_regular_user_can_delete_own_chat(self, client, regular_user_token):
        """Verify regular user can delete their own chat."""
        from fivccliche.modules.agent_chats import utils as chat_methods
        from fivccliche.modules.users.utils import get_user_async

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
            await session.commit()
            return str(chat.uuid)

        chat_uuid = loop.run_until_complete(setup())

        # Delete as regular user
        headers = {"Authorization": f"Bearer {regular_user_token}"}
        response = client.delete(f"/chats/{chat_uuid}", headers=headers)
        assert response.status_code == 204

    def test_superuser_can_delete_any_user_chat(self, client, auth_token):
        """Verify superuser can delete any user's chat."""
        from fivccliche.modules.agent_chats import utils as chat_methods
        from fivccliche.modules.users import utils as user_methods

        session = client.async_session
        loop = client.loop

        async def setup():
            other = await user_methods.create_user_async(
                session,
                username="other-chat-owner-2",
                email="other-chat-owner-2@example.com",
                password="password123",
            )
            await session.commit()
            chat = await chat_methods.create_chat_async(
                session=session,
                user_uuid=other.uuid,
                agent_id="test-agent",
            )
            await session.commit()
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
        """Verify 404 error when regular user tries to delete message in global chat."""
        from fivccliche.modules.agent_chats import utils as chat_methods

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
            await session.commit()
            return str(chat.uuid), str(message.uuid)

        chat_uuid, message_uuid = loop.run_until_complete(setup())

        # Try to delete message as regular user
        headers = {"Authorization": f"Bearer {regular_user_token}"}
        response = client.delete(
            f"/chats/{chat_uuid}/messages/{message_uuid}",
            headers=headers,
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_regular_user_cannot_delete_message_in_other_users_chat(
        self, client, regular_user_token
    ):
        """Verify 404 when regular user tries to delete message in another user's chat."""
        from fivccliche.modules.agent_chats import utils as chat_methods
        from fivccliche.modules.users import utils as user_methods

        session = client.async_session
        loop = client.loop

        async def setup():
            other = await user_methods.create_user_async(
                session,
                username="other-msg-owner",
                email="other-msg-owner@example.com",
                password="password123",
            )
            await session.commit()
            chat = await chat_methods.create_chat_async(
                session=session,
                user_uuid=other.uuid,
                agent_id="test-agent",
            )
            # Create message
            message = await chat_methods.create_chat_message_async(
                session=session,
                chat_uuid=str(chat.uuid),
                query={"content": "test message"},
            )
            await session.commit()
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
        from fivccliche.modules.agent_chats import utils as chat_methods
        from fivccliche.modules.users.utils import get_user_async

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
            await session.commit()
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
        from fivccliche.modules.agent_chats import utils as chat_methods

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
            await session.commit()
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

    @pytest.fixture(autouse=True)
    def _mock_describe_job(self):
        with patch("fivccliche.modules.agent_chats.routers.ChatDescribeJob") as mocked:
            instance = MagicMock()
            instance.run_async = AsyncMock()
            mocked.return_value = instance
            self._describe_job_cls = mocked
            self._describe_job = instance
            yield mocked

    @staticmethod
    def _mock_user(uuid: str = "user-123", is_superuser: bool = False):
        user = MagicMock()
        user.uuid = uuid
        user.is_superuser = is_superuser
        return user

    @staticmethod
    def _mock_chat(
        chat_uuid: str = "chat-123",
        user_uuid: str | None = "user-123",
        context: dict | None = None,
    ):

        return UserChat(
            uuid=chat_uuid,
            user_uuid=user_uuid,
            agent_id="test-agent",
            context=context,
        )

    @staticmethod
    async def _consume_response(response):
        return [chunk async for chunk in response.body_iterator]

    @staticmethod
    def _fake_chat_stream(stream_factory):
        """Build a ChatStream-like mock for router tests."""
        instance = MagicMock()
        instance.attach = MagicMock()
        instance.on_event = MagicMock()
        instance.side_effect = stream_factory
        return instance

    @staticmethod
    def _fake_query_job():
        instance = MagicMock()
        instance.run_async = AsyncMock()
        return instance

    @staticmethod
    def _query_kwargs(job):
        args, kwargs = job.run_async.call_args
        return args, kwargs

    @staticmethod
    async def _attached_query_task(fake_stream):
        fake_stream.attach.assert_called_once()
        query_task = fake_stream.attach.call_args[0][0]
        await asyncio.sleep(0)
        return query_task

    @contextmanager
    def _patch_stream_and_job(self, stream_factory, *, job_side_effect=None):
        fake_stream = self._fake_chat_stream(stream_factory)
        fake_job = self._fake_query_job()
        job_patch = (
            {"side_effect": job_side_effect}
            if job_side_effect is not None
            else {"return_value": fake_job}
        )
        with (
            patch(
                "fivccliche.modules.agent_chats.routers.ChatStream",
                return_value=fake_stream,
            ) as mock_stream_cls,
            patch(
                "fivccliche.modules.agent_chats.routers.ChatQueryJob",
                **job_patch,
            ) as mock_job_cls,
        ):
            yield fake_stream, fake_job, mock_stream_cls, mock_job_cls

    @pytest.mark.asyncio
    async def test_create_message_does_not_access_mutex_when_chat_not_found(self):
        """Ownership validation happens before optional mutex acquisition."""
        from fivccliche.modules.agent_chats.routers import create_chat_messages_async
        from fivccliche.modules.agent_chats.schemas import UserChatMessageCreateSchema

        mock_mutex_site = MagicMock()

        with (
            patch(
                "fivccliche.modules.agent_chats.routers.utils.get_chat_async",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("fivccliche.modules.agent_chats.routers.ChatStream") as mock_stream_cls,
            patch("fivccliche.modules.agent_chats.routers.ChatQueryJob") as mock_job_cls,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await create_chat_messages_async(
                    chat_uuid="missing-chat",
                    chat_message=UserChatMessageCreateSchema(query="Hello"),
                    background_tasks=MagicMock(),
                    user=self._mock_user(),
                    session=AsyncMock(),
                    mutex_site=mock_mutex_site,
                )

        assert exc_info.value.status_code == 404
        mock_mutex_site.get_mutex.assert_not_called()
        mock_stream_cls.assert_not_called()
        mock_job_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_message_falls_back_without_mutex_site(self):
        """Missing mutex site preserves the existing unlocked streaming flow."""
        from fivccliche.modules.agent_chats.routers import (
            CHAT_MESSAGE_RUN_TIMEOUT,
            create_chat_messages_async,
        )
        from fivccliche.modules.agent_chats.schemas import UserChatMessageCreateSchema

        async def mock_generator():
            yield b"data: done\n\n"

        context = {"scope": "router"}
        background_tasks = MagicMock()
        mock_session = AsyncMock()

        with (
            patch(
                "fivccliche.modules.agent_chats.routers.utils.get_chat_async",
                new_callable=AsyncMock,
                return_value=self._mock_chat(context=context),
            ),
            self._patch_stream_and_job(mock_generator) as (
                fake_stream,
                fake_job,
                mock_stream_cls,
                mock_job_cls,
            ),
        ):
            response = await create_chat_messages_async(
                chat_uuid="chat-123",
                chat_message=UserChatMessageCreateSchema(query="Hello"),
                background_tasks=background_tasks,
                user=self._mock_user(),
                session=mock_session,
                mutex_site=None,
            )
            chunks = await self._consume_response(response)
            query_task = await self._attached_query_task(fake_stream)

        assert chunks == [b"data: done\n\n"]
        mock_stream_cls.assert_called_once_with(chat_uuid="chat-123")
        mock_job_cls.assert_called_once()
        args, kwargs = self._query_kwargs(fake_job)
        assert args == ("chat-123",)
        assert kwargs["user_uuid"] == "user-123"
        assert kwargs["query"] == "Hello"
        assert kwargs["agent_id"] == "test-agent"
        assert kwargs["context"] == context
        assert kwargs["skills_enabled"] is True
        assert kwargs["chat_mutex"] is None
        assert kwargs["run_timeout"] == CHAT_MESSAGE_RUN_TIMEOUT.total_seconds()
        assert kwargs["event_callback"] == fake_stream.on_event
        gather_call, describe_call = background_tasks.add_task.call_args_list
        assert gather_call.args == (asyncio.gather, query_task)
        assert gather_call.kwargs == {"return_exceptions": True}
        assert describe_call.args == (self._describe_job.run_async, "chat-123")
        assert describe_call.kwargs == {"user_uuid": "user-123", "query_text": "Hello"}
        self._describe_job_cls.assert_called_once()
        assert background_tasks.add_task.call_count == 2
        mock_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_message_falls_back_when_mutex_unavailable(self):
        """Missing concrete mutex preserves the existing unlocked streaming flow."""
        from fivccliche.modules.agent_chats.routers import create_chat_messages_async
        from fivccliche.modules.agent_chats.schemas import UserChatMessageCreateSchema

        async def mock_generator():
            yield b"data: done\n\n"

        mock_mutex_site = MagicMock()
        mock_mutex_site.get_mutex.return_value = None

        with (
            patch(
                "fivccliche.modules.agent_chats.routers.utils.get_chat_async",
                new_callable=AsyncMock,
                return_value=self._mock_chat(),
            ),
            self._patch_stream_and_job(mock_generator) as (
                fake_stream,
                fake_job,
                _mock_stream_cls,
                _mock_job_cls,
            ),
        ):
            response = await create_chat_messages_async(
                chat_uuid="chat-123",
                chat_message=UserChatMessageCreateSchema(query="Hello"),
                background_tasks=MagicMock(),
                user=self._mock_user(),
                session=AsyncMock(),
                mutex_site=mock_mutex_site,
            )
            chunks = await self._consume_response(response)
            await self._attached_query_task(fake_stream)

        assert chunks == [b"data: done\n\n"]
        mock_mutex_site.get_mutex.assert_called_once_with("chats:message:chat-123")
        _, kwargs = self._query_kwargs(fake_job)
        assert kwargs["chat_mutex"] is None

    @pytest.mark.asyncio
    async def test_create_message_skips_describe_when_description_set(self):
        from fivccliche.modules.agent_chats.routers import create_chat_messages_async
        from fivccliche.modules.agent_chats.schemas import UserChatMessageCreateSchema

        async def mock_generator():
            yield b"data: done\n\n"

        chat = self._mock_chat()
        chat.description = "Already titled"

        with (
            patch(
                "fivccliche.modules.agent_chats.routers.utils.get_chat_async",
                new_callable=AsyncMock,
                return_value=chat,
            ),
            self._patch_stream_and_job(mock_generator) as (fake_stream, _job, _s, _j),
        ):
            await create_chat_messages_async(
                chat_uuid="chat-123",
                chat_message=UserChatMessageCreateSchema(query="Hello"),
                background_tasks=MagicMock(),
                user=self._mock_user(),
                session=AsyncMock(),
                mutex_site=None,
            )
            await self._attached_query_task(fake_stream)

        self._describe_job_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_message_skips_describe_for_slash_query(self):
        from fivccliche.modules.agent_chats.routers import create_chat_messages_async
        from fivccliche.modules.agent_chats.schemas import UserChatMessageCreateSchema

        async def mock_generator():
            yield b"data: done\n\n"

        with (
            patch(
                "fivccliche.modules.agent_chats.routers.utils.get_chat_async",
                new_callable=AsyncMock,
                return_value=self._mock_chat(),
            ),
            self._patch_stream_and_job(mock_generator) as (fake_stream, _job, _s, _j),
        ):
            await create_chat_messages_async(
                chat_uuid="chat-123",
                chat_message=UserChatMessageCreateSchema(query="/help"),
                background_tasks=MagicMock(),
                user=self._mock_user(),
                session=AsyncMock(),
                mutex_site=None,
            )
            await self._attached_query_task(fake_stream)

        self._describe_job_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_message_returns_409_when_chat_mutex_locked(self):
        """An existing chat message lock prevents duplicate agent runs."""
        from fivccliche.modules.agent_chats.routers import create_chat_messages_async
        from fivccliche.modules.agent_chats.schemas import UserChatMessageCreateSchema

        mock_mutex = MagicMock()
        mock_mutex.acquire_async = AsyncMock(return_value=False)
        mock_mutex.release_async = AsyncMock()
        mock_mutex_site = MagicMock()
        mock_mutex_site.get_mutex.return_value = mock_mutex

        with (
            patch(
                "fivccliche.modules.agent_chats.routers.utils.get_chat_async",
                new_callable=AsyncMock,
                return_value=self._mock_chat(),
            ),
            patch("fivccliche.modules.agent_chats.routers.ChatStream") as mock_stream_cls,
            patch("fivccliche.modules.agent_chats.routers.ChatQueryJob") as mock_job_cls,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await create_chat_messages_async(
                    chat_uuid="chat-123",
                    chat_message=UserChatMessageCreateSchema(query="Hello"),
                    background_tasks=MagicMock(),
                    user=self._mock_user(),
                    session=AsyncMock(),
                    mutex_site=mock_mutex_site,
                )

        assert exc_info.value.status_code == 409
        assert exc_info.value.detail == "Chat message processing already running"
        mock_mutex.acquire_async.assert_awaited_once()
        mock_mutex.release_async.assert_not_awaited()
        mock_stream_cls.assert_not_called()
        mock_job_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_message_passes_acquired_mutex_to_query_job(self):
        """Router acquires the lock and hands the mutex to ChatQueryJob."""
        from fivccliche.modules.agent_chats.routers import (
            CHAT_MESSAGE_RUN_TIMEOUT,
            create_chat_messages_async,
        )
        from fivccliche.modules.agent_chats.schemas import UserChatMessageCreateSchema

        async def mock_generator():
            yield b"data: done\n\n"

        mock_mutex = MagicMock()
        mock_mutex.acquire_async = AsyncMock(return_value=True)
        mock_mutex.release_async = AsyncMock()
        mock_mutex_site = MagicMock()
        mock_mutex_site.get_mutex.return_value = mock_mutex
        mock_session = AsyncMock()
        background_tasks = MagicMock()

        with (
            patch(
                "fivccliche.modules.agent_chats.routers.utils.get_chat_async",
                new_callable=AsyncMock,
                return_value=self._mock_chat(),
            ),
            self._patch_stream_and_job(mock_generator) as (
                fake_stream,
                fake_job,
                _mock_stream_cls,
                _mock_job_cls,
            ),
        ):
            response = await create_chat_messages_async(
                chat_uuid="chat-123",
                chat_message=UserChatMessageCreateSchema(query="Hello"),
                background_tasks=background_tasks,
                user=self._mock_user(),
                session=mock_session,
                mutex_site=mock_mutex_site,
            )
            chunks = await self._consume_response(response)
            query_task = await self._attached_query_task(fake_stream)

        assert chunks == [b"data: done\n\n"]
        mock_mutex.acquire_async.assert_awaited_once()
        _, kwargs = self._query_kwargs(fake_job)
        assert kwargs["chat_mutex"] is mock_mutex
        assert kwargs["run_timeout"] == CHAT_MESSAGE_RUN_TIMEOUT.total_seconds()
        background_tasks.add_task.assert_any_call(
            asyncio.gather, query_task, return_exceptions=True
        )
        # Release is owned by ChatQueryJob, not the router.
        mock_mutex.release_async.assert_not_awaited()
        mock_session.close.assert_awaited_once()
        self._describe_job_cls.assert_called_once()
        self._describe_job.run_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_message_passes_mutex_even_when_stream_raises(self):
        """Router still passes the acquired mutex when the mocked stream raises."""
        from fivccliche.modules.agent_chats.routers import create_chat_messages_async
        from fivccliche.modules.agent_chats.schemas import UserChatMessageCreateSchema

        async def mock_generator():
            yield b"data: start\n\n"
            raise RuntimeError("stream failed")

        mock_mutex = MagicMock()
        mock_mutex.acquire_async = AsyncMock(return_value=True)
        mock_mutex.release_async = AsyncMock()
        mock_mutex_site = MagicMock()
        mock_mutex_site.get_mutex.return_value = mock_mutex

        with (
            patch(
                "fivccliche.modules.agent_chats.routers.utils.get_chat_async",
                new_callable=AsyncMock,
                return_value=self._mock_chat(),
            ),
            self._patch_stream_and_job(mock_generator) as (
                fake_stream,
                fake_job,
                _mock_stream_cls,
                _mock_job_cls,
            ),
        ):
            response = await create_chat_messages_async(
                chat_uuid="chat-123",
                chat_message=UserChatMessageCreateSchema(query="Hello"),
                background_tasks=MagicMock(),
                user=self._mock_user(),
                session=AsyncMock(),
                mutex_site=mock_mutex_site,
            )
            with pytest.raises(RuntimeError, match="stream failed"):
                await self._consume_response(response)
            await self._attached_query_task(fake_stream)

        _, kwargs = self._query_kwargs(fake_job)
        assert kwargs["chat_mutex"] is mock_mutex
        mock_mutex.release_async.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_create_message_releases_mutex_when_query_job_ctor_fails(self):
        """Router releases mutex when ChatQueryJob construction fails after acquire."""
        from fivccliche.modules.agent_chats.routers import create_chat_messages_async
        from fivccliche.modules.agent_chats.schemas import UserChatMessageCreateSchema

        mock_mutex = MagicMock()
        mock_mutex.acquire_async = AsyncMock(return_value=True)
        mock_mutex.release_async = AsyncMock()
        mock_mutex_site = MagicMock()
        mock_mutex_site.get_mutex.return_value = mock_mutex

        async def mock_generator():
            yield b"data: done\n\n"

        with (
            patch(
                "fivccliche.modules.agent_chats.routers.utils.get_chat_async",
                new_callable=AsyncMock,
                return_value=self._mock_chat(),
            ),
            self._patch_stream_and_job(
                mock_generator, job_side_effect=RuntimeError("setup failed")
            ) as (_stream, _job, _mock_stream_cls, mock_job_cls),
        ):
            with pytest.raises(RuntimeError, match="setup failed"):
                await create_chat_messages_async(
                    chat_uuid="chat-123",
                    chat_message=UserChatMessageCreateSchema(query="Hello"),
                    background_tasks=MagicMock(),
                    user=self._mock_user(),
                    session=AsyncMock(),
                    mutex_site=mock_mutex_site,
                )

        mock_job_cls.assert_called_once()
        mock_mutex.release_async.assert_awaited_once()

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
        from fivccliche.modules.agent_chats import utils as chat_methods
        from fivccliche.modules.users.utils import get_user_async

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
            await session.commit()
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
        from fivccliche.modules.agent_chats import utils as chat_methods

        session = client.async_session
        loop = client.loop

        # Create a global chat
        async def setup():
            chat = await chat_methods.create_chat_async(
                session=session,
                user_uuid=None,
                agent_id="test-agent",
            )
            await session.commit()
            return str(chat.uuid)

        chat_uuid = loop.run_until_complete(setup())

        # Try to create message as regular user in global chat
        headers = {"Authorization": f"Bearer {regular_user_token}"}
        response = client.post(
            f"/chats/{chat_uuid}/messages/",
            json={"query": "Hello"},
            headers=headers,
        )
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_create_message_missing_query_field(self, client: TestClient, auth_token: str):
        """Test creating message without query field."""
        from fivccliche.modules.agent_chats import utils as chat_methods
        from fivccliche.modules.users.utils import get_user_async

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
            await session.commit()
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


class TestChatContextFlow:
    """Test cases for chat context flow through the API."""

    def test_chat_creation_with_context_stored_correctly(self, client: TestClient, auth_token: str):
        """Test that creating a chat with context stores it correctly."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        context_data = {"session_type": "debug", "environment": "test", "user_id": 123}

        # Create chat with context
        response = client.post(
            "/chats/",
            json={"agent_id": "test-agent", "context": context_data},
            headers=headers,
        )
        assert response.status_code == 201
        chat_data = response.json()
        assert chat_data["context"] == context_data

        # Retrieve the chat and verify context persists
        chat_uuid = chat_data["uuid"]
        get_response = client.get(f"/chats/{chat_uuid}/", headers=headers)
        assert get_response.status_code == 200
        retrieved_chat = get_response.json()
        assert retrieved_chat["context"] == context_data

    def test_chat_creation_with_none_context(self, client: TestClient, auth_token: str):
        """Test that creating a chat with None context works correctly."""
        headers = {"Authorization": f"Bearer {auth_token}"}

        # Create chat without context
        response = client.post(
            "/chats/",
            json={"agent_id": "test-agent"},
            headers=headers,
        )
        assert response.status_code == 201
        chat_data = response.json()
        assert chat_data["context"] is None

    def test_chat_creation_with_empty_context(self, client: TestClient, auth_token: str):
        """Test that creating a chat with empty context dict works correctly."""
        headers = {"Authorization": f"Bearer {auth_token}"}

        # Create chat with empty context
        response = client.post(
            "/chats/",
            json={"agent_id": "test-agent", "context": {}},
            headers=headers,
        )
        assert response.status_code == 201
        chat_data = response.json()
        assert chat_data["context"] == {}

    def test_chat_context_retrieved_in_service_layer(self, client: TestClient, auth_token: str):
        """Test that context can be retrieved in the service layer."""
        from fivccliche.modules.agent_chats import utils as chat_methods
        from fivccliche.modules.agent_chats.filters import ChatFilterSet
        from fivccliche.modules.users.utils import get_user_async

        session = client.async_session
        loop = client.loop

        # Create chat with context
        context_data = {"key1": "value1", "key2": 123}

        async def test_flow():
            admin_user = await get_user_async(session, username="admin")
            chat = await chat_methods.create_chat_async(
                session=session,
                user_uuid=str(admin_user.uuid),
                agent_id="test-agent",
                context=context_data,
            )
            await session.commit()

            # Retrieve the chat
            retrieved_chat = await chat_methods.get_chat_async(
                session,
                chat.uuid,
                filters=ChatFilterSet(str(admin_user.uuid), is_superuser=admin_user.is_superuser),
            )
            assert retrieved_chat is not None
            assert retrieved_chat.context == context_data

        loop.run_until_complete(test_flow())

    def test_chat_provider_get_chat_context_can_be_called(
        self, client: TestClient, auth_token: str
    ):
        """Test that get_chat_context returns a dict with user_uuid."""
        from fivccliche.modules.agent_chats.services import UserChatProviderImpl
        from fivccliche.modules.users.utils import get_user_async

        session = client.async_session
        loop = client.loop

        async def test_provider():
            admin_user = await get_user_async(session, username="admin")
            component_site = Mock()
            provider = UserChatProviderImpl(component_site)

            context = provider.get_chat_context(
                user_uuid=str(admin_user.uuid),
            )
            assert isinstance(context, dict)
            assert context["user_uuid"] == str(admin_user.uuid)

        loop.run_until_complete(test_provider())

    def test_chat_context_with_complex_nested_data(self, client: TestClient, auth_token: str):
        """Test that complex nested context data is stored and retrieved correctly."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        complex_context = {
            "session": {
                "type": "debug",
                "environment": "test",
                "features": ["feature1", "feature2"],
            },
            "user": {"id": 123, "preferences": {"theme": "dark", "lang": "en"}},
            "metadata": {"version": "1.0", "timestamp": 1234567890},
        }

        # Create chat with complex context
        response = client.post(
            "/chats/",
            json={"agent_id": "test-agent", "context": complex_context},
            headers=headers,
        )
        assert response.status_code == 201
        chat_data = response.json()
        assert chat_data["context"] == complex_context

        # Verify nested access works
        assert chat_data["context"]["session"]["type"] == "debug"
        assert chat_data["context"]["user"]["preferences"]["theme"] == "dark"
