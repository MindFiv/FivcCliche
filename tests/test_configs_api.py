"""Integration tests for configs API endpoints."""

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel

from fivccliche.utils.deps import get_db_session_async
from fivccliche.services.implements.modules import ModuleSiteImpl
from fivcglue.implements.utils import load_component_site


@pytest.fixture
def client():
    """Create a test client with temporary database and admin user."""
    import asyncio
    from fivccliche.modules.users import methods

    # Import models to ensure they're registered with SQLModel
    from fivccliche.modules.users.models import User  # noqa: F401
    from fivccliche.modules.agent_configs.models import (  # noqa: F401
        UserEmbedding,
        UserLLM,
        UserAgent,
        UserTool,
    )

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
        module_site = ModuleSiteImpl(component_site, modules=["users", "agent_configs"])
        app = module_site.create_application()

        # Override the database dependency
        async def override_get_db_session_async():
            yield async_session

        app.dependency_overrides[get_db_session_async] = override_get_db_session_async

        # Create an admin user for testing
        async def create_admin_user():
            admin_user = await methods.create_user_async(
                async_session,
                username="admin",
                email="admin@example.com",
                password="admin123",
                is_superuser=True,
            )
            return admin_user

        admin_user = loop.run_until_complete(create_admin_user())

        with TestClient(app) as test_client:
            # Store admin user and session in test_client for test access
            test_client.admin_user = admin_user
            test_client.async_session = async_session
            test_client.loop = loop
            yield test_client

        # Cleanup
        async def cleanup():
            await async_session.close()
            await engine.dispose()

        loop.run_until_complete(cleanup())
    finally:
        loop.close()
        # Clean up the temporary database file
        try:
            Path(temp_db.name).unlink()
        except Exception:
            pass


@pytest.fixture
def auth_token(client: TestClient):
    """Generate a JWT token for a test user."""
    # Create a user via the API with admin headers
    admin_token = client.post(
        "/users/login",
        json={
            "username": "admin",
            "password": "admin123",
        },
    ).json()["access_token"]

    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # Create a test user
    client.post(
        "/users/",
        json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "password123",
        },
        headers=admin_headers,
    )
    # Login to get token
    response = client.post(
        "/users/login",
        json={
            "username": "testuser",
            "password": "password123",
        },
    )
    return response.json()["access_token"]


@pytest.fixture
def admin_token(client: TestClient):
    """Generate a JWT token for the admin user."""
    response = client.post(
        "/users/login",
        json={
            "username": "admin",
            "password": "admin123",
        },
    )
    return response.json()["access_token"]


@pytest.fixture
def other_user_token(client: TestClient):
    """Generate a JWT token for another test user."""
    admin_token = client.post(
        "/users/login",
        json={
            "username": "admin",
            "password": "admin123",
        },
    ).json()["access_token"]

    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # Create another test user
    client.post(
        "/users/",
        json={
            "username": "otheruser",
            "email": "other@example.com",
            "password": "password123",
        },
        headers=admin_headers,
    )
    # Login to get token
    response = client.post(
        "/users/login",
        json={
            "username": "otheruser",
            "password": "password123",
        },
    )
    return response.json()["access_token"]


class TestEmbeddingConfigAPI:
    """Test cases for Embedding Config API endpoints."""

    def test_create_embedding_config_unauthorized(self, client: TestClient):
        """Test creating embedding config without authentication."""
        response = client.post(
            "/configs/embeddings/",
            json={
                "id": "embedding-1",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
            },
        )
        assert response.status_code == 401

    def test_create_embedding_config(self, client: TestClient, auth_token: str):
        """Test creating a new embedding config."""
        response = client.post(
            "/configs/embeddings/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "embedding-1",
                "description": "Test embedding",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
                "base_url": "https://api.openai.com",
                "dimension": 1536,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "embedding-1"
        assert data["model"] == "text-embedding-3-small"
        assert data["dimension"] == 1536
        assert "user_uuid" in data

    def test_list_embedding_configs(self, client: TestClient, auth_token: str):
        """Test listing embedding configs."""
        # Create multiple configs
        for i in range(3):
            client.post(
                "/configs/embeddings/",
                headers={"Authorization": f"Bearer {auth_token}"},
                json={
                    "id": f"embedding-{i}",
                    "provider": "openai",
                    "model": f"model-{i}",
                    "api_key": f"key-{i}",
                },
            )

        response = client.get(
            "/configs/embeddings/",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["results"]) == 3

    def test_list_embedding_configs_pagination(self, client: TestClient, auth_token: str):
        """Test listing embedding configs with pagination."""
        for i in range(5):
            client.post(
                "/configs/embeddings/",
                headers={"Authorization": f"Bearer {auth_token}"},
                json={
                    "id": f"embedding-page-{i}",
                    "provider": "openai",
                    "model": f"model-{i}",
                    "api_key": f"key-{i}",
                },
            )

        response = client.get(
            "/configs/embeddings/?skip=0&limit=2",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["results"]) == 2

    def test_get_embedding_config(self, client: TestClient, auth_token: str):
        """Test getting a specific embedding config."""
        # Create a config
        create_response = client.post(
            "/configs/embeddings/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "embedding-get",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
            },
        )
        config_uuid = create_response.json()["uuid"]

        response = client.get(
            f"/configs/embeddings/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "embedding-get"
        assert "user_uuid" in data

    def test_get_embedding_config_not_found(self, client: TestClient, auth_token: str):
        """Test getting non-existent embedding config."""
        response = client.get(
            "/configs/embeddings/nonexistent",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 404

    def test_update_embedding_config(self, client: TestClient, auth_token: str):
        """Test updating an embedding config."""
        # Create a config
        create_response = client.post(
            "/configs/embeddings/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "embedding-update",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
                "dimension": 1536,
            },
        )
        config_uuid = create_response.json()["uuid"]

        response = client.patch(
            f"/configs/embeddings/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "embedding-update",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
                "dimension": 3072,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["dimension"] == 3072

    def test_delete_embedding_config(self, client: TestClient, auth_token: str):
        """Test deleting an embedding config."""
        # Create a config
        create_response = client.post(
            "/configs/embeddings/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "embedding-delete",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
            },
        )
        config_uuid = create_response.json()["uuid"]

        response = client.delete(
            f"/configs/embeddings/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 204

        # Verify it's deleted
        response = client.get(
            f"/configs/embeddings/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 404


class TestLLMConfigAPI:
    """Test cases for LLM Config API endpoints."""

    def test_create_llm_config_unauthorized(self, client: TestClient):
        """Test creating LLM config without authentication."""
        response = client.post(
            "/configs/models/",
            json={
                "id": "llm-1",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
            },
        )
        assert response.status_code == 401

    def test_create_llm_config(self, client: TestClient, auth_token: str):
        """Test creating a new LLM config."""
        response = client.post(
            "/configs/models/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "llm-1",
                "description": "Test LLM",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
                "temperature": 0.7,
                "max_tokens": 2048,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "llm-1"
        assert data["model"] == "gpt-4"
        assert data["temperature"] == 0.7
        assert "user_uuid" in data

    def test_list_llm_configs(self, client: TestClient, auth_token: str):
        """Test listing LLM configs."""
        for i in range(3):
            client.post(
                "/configs/models/",
                headers={"Authorization": f"Bearer {auth_token}"},
                json={
                    "id": f"llm-{i}",
                    "provider": "openai",
                    "model": f"model-{i}",
                    "api_key": f"key-{i}",
                },
            )

        response = client.get(
            "/configs/models/",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["results"]) == 3

    def test_get_llm_config(self, client: TestClient, auth_token: str):
        """Test getting a specific LLM config."""
        create_response = client.post(
            "/configs/models/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "llm-get",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
            },
        )
        config_uuid = create_response.json()["uuid"]

        response = client.get(
            f"/configs/models/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "llm-get"

    def test_get_llm_config_not_found(self, client: TestClient, auth_token: str):
        """Test getting non-existent LLM config."""
        response = client.get(
            "/configs/models/nonexistent",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 404

    def test_update_llm_config(self, client: TestClient, auth_token: str):
        """Test updating an LLM config."""
        create_response = client.post(
            "/configs/models/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "llm-update",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
                "temperature": 0.5,
            },
        )
        config_uuid = create_response.json()["uuid"]

        response = client.patch(
            f"/configs/models/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "llm-update",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
                "temperature": 0.9,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["temperature"] == 0.9

    def test_delete_llm_config(self, client: TestClient, auth_token: str):
        """Test deleting an LLM config."""
        create_response = client.post(
            "/configs/models/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "llm-delete",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
            },
        )
        config_uuid = create_response.json()["uuid"]

        response = client.delete(
            f"/configs/models/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 204

        response = client.get(
            f"/configs/models/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 404


class TestAgentConfigAPI:
    """Test cases for Agent Config API endpoints."""

    def test_create_agent_config_unauthorized(self, client: TestClient):
        """Test creating agent config without authentication."""
        response = client.post(
            "/configs/agents/",
            json={
                "id": "agent-1",
                "model_id": "model123",
            },
        )
        assert response.status_code == 401

    def test_create_agent_config(self, client: TestClient, auth_token: str):
        """Test creating a new agent config."""
        response = client.post(
            "/configs/agents/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "agent-1",
                "description": "Test agent",
                "model_id": "model123",
                "system_prompt": "You are a helpful assistant",
                "response_format": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                },
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "agent-1"
        assert data["model_id"] == "model123"
        assert data["response_format"] == {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        }
        assert "user_uuid" in data

    def test_list_agent_configs(self, client: TestClient, auth_token: str):
        """Test listing agent configs."""
        for i in range(3):
            client.post(
                "/configs/agents/",
                headers={"Authorization": f"Bearer {auth_token}"},
                json={
                    "id": f"agent-{i}",
                    "model_id": f"model-{i}",
                },
            )

        response = client.get(
            "/configs/agents/",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["results"]) == 3

    def test_get_agent_config(self, client: TestClient, auth_token: str):
        """Test getting a specific agent config."""
        create_response = client.post(
            "/configs/agents/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "agent-get",
                "model_id": "model123",
            },
        )
        config_uuid = create_response.json()["uuid"]

        response = client.get(
            f"/configs/agents/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "agent-get"
        assert "user_uuid" in data

    def test_get_agent_config_not_found(self, client: TestClient, auth_token: str):
        """Test getting non-existent agent config."""
        response = client.get(
            "/configs/agents/nonexistent",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 404

    def test_update_agent_config(self, client: TestClient, auth_token: str):
        """Test updating an agent config."""
        create_response = client.post(
            "/configs/agents/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "agent-update",
                "model_id": "model123",
                "system_prompt": "Old prompt",
            },
        )
        config_uuid = create_response.json()["uuid"]

        response = client.patch(
            f"/configs/agents/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "agent-update",
                "model_id": "model123",
                "system_prompt": "New prompt",
                "response_format": {
                    "type": "object",
                    "properties": {"result": {"type": "integer"}},
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["system_prompt"] == "New prompt"
        assert data["response_format"] == {
            "type": "object",
            "properties": {"result": {"type": "integer"}},
        }

    def test_delete_agent_config(self, client: TestClient, auth_token: str):
        """Test deleting an agent config."""
        create_response = client.post(
            "/configs/agents/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "agent-delete",
                "model_id": "model123",
            },
        )
        config_uuid = create_response.json()["uuid"]

        response = client.delete(
            f"/configs/agents/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 204

        response = client.get(
            f"/configs/agents/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 404


class TestToolConfigAPI:
    """Test cases for Tool Config API endpoints."""

    def test_indexing_tool_unauthorized(self, client: TestClient):
        """Test indexing tool without authentication."""
        response = client.post("/configs/tools/index")
        assert response.status_code == 401

    def test_indexing_tool_success(self, client: TestClient, auth_token: str):
        """Test successful tool indexing."""
        from unittest.mock import patch

        # Mock the create_tool_retriever function and its return value
        from unittest.mock import AsyncMock

        mock_tool_retriever = AsyncMock()
        mock_tool_retriever.index_tools_async = AsyncMock()

        with patch(
            "fivccliche.modules.agent_configs.routers.create_tool_retriever_async"
        ) as mock_create:
            mock_create.return_value = mock_tool_retriever

            response = client.post(
                "/configs/tools/index",
                headers={"Authorization": f"Bearer {auth_token}"},
            )

            assert response.status_code == 200
            # Verify create_tool_retriever_async was called
            assert mock_create.called
            # Verify index_tools_async was called on the retriever
            mock_tool_retriever.index_tools_async.assert_called_once()

    def test_indexing_tool_with_existing_tools(self, client: TestClient, auth_token: str):
        """Test indexing tool when user has existing tool configs."""
        from unittest.mock import patch

        # Create some tool configs first
        for i in range(3):
            client.post(
                "/configs/tools/",
                headers={"Authorization": f"Bearer {auth_token}"},
                json={
                    "id": f"tool-index-{i}",
                    "description": f"Tool for indexing {i}",
                    "transport": "stdio",
                    "command": "python",
                },
            )

        # Mock the create_tool_retriever function
        from unittest.mock import AsyncMock

        mock_tool_retriever = AsyncMock()
        mock_tool_retriever.index_tools_async = AsyncMock()

        with patch(
            "fivccliche.modules.agent_configs.routers.create_tool_retriever_async"
        ) as mock_create:
            mock_create.return_value = mock_tool_retriever

            response = client.post(
                "/configs/tools/index",
                headers={"Authorization": f"Bearer {auth_token}"},
            )

            assert response.status_code == 200
            # Verify the tool retriever was created with correct parameters
            assert mock_create.called
            call_kwargs = mock_create.call_args.kwargs
            assert "tool_backend" in call_kwargs
            assert "tool_config_repository" in call_kwargs
            assert "embedding_backend" in call_kwargs
            assert "embedding_config_repository" in call_kwargs
            assert "space_id" in call_kwargs
            # Verify index_tools_async was called
            mock_tool_retriever.index_tools_async.assert_called_once()

    def test_indexing_tool_calls_config_provider_methods(self, client: TestClient, auth_token: str):
        """Test that indexing tool calls all required config provider methods."""
        from unittest.mock import AsyncMock, patch

        mock_tool_retriever = AsyncMock()
        mock_tool_retriever.index_tools_async = AsyncMock()

        with patch(
            "fivccliche.modules.agent_configs.routers.create_tool_retriever_async"
        ) as mock_create:
            mock_create.return_value = mock_tool_retriever

            response = client.post(
                "/configs/tools/index",
                headers={"Authorization": f"Bearer {auth_token}"},
            )

            assert response.status_code == 200
            # Verify create_tool_retriever_async was called with all required parameters
            call_kwargs = mock_create.call_args.kwargs
            assert "tool_backend" in call_kwargs
            assert "tool_config_repository" in call_kwargs
            assert "embedding_backend" in call_kwargs
            assert "embedding_config_repository" in call_kwargs
            assert "space_id" in call_kwargs

    def test_indexing_tool_exception_handling(self, client: TestClient, auth_token: str):
        """Test indexing tool handles exceptions from index_tools."""
        from unittest.mock import AsyncMock, patch

        mock_tool_retriever = AsyncMock()
        mock_tool_retriever.index_tools_async = AsyncMock(side_effect=Exception("Indexing failed"))

        with patch(
            "fivccliche.modules.agent_configs.routers.create_tool_retriever_async"
        ) as mock_create:
            mock_create.return_value = mock_tool_retriever

            # The endpoint doesn't explicitly handle exceptions, so it should raise
            with pytest.raises(Exception) as exc_info:  # noqa
                client.post(
                    "/configs/tools/index",
                    headers={"Authorization": f"Bearer {auth_token}"},
                )
            assert "Indexing failed" in str(exc_info.value)

    def test_indexing_tool_with_different_users(self, client: TestClient):
        """Test that indexing tool uses correct user context."""
        from unittest.mock import patch

        # Create two different users
        admin_token = client.post(
            "/users/login",
            json={"username": "admin", "password": "admin123"},
        ).json()["access_token"]

        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        # Create first user
        client.post(
            "/users/",
            json={
                "username": "user1",
                "email": "user1@example.com",
                "password": "password123",
            },
            headers=admin_headers,
        )
        user1_token = client.post(
            "/users/login",
            json={"username": "user1", "password": "password123"},
        ).json()["access_token"]

        # Create second user
        client.post(
            "/users/",
            json={
                "username": "user2",
                "email": "user2@example.com",
                "password": "password123",
            },
            headers=admin_headers,
        )
        user2_token = client.post(
            "/users/login",
            json={"username": "user2", "password": "password123"},
        ).json()["access_token"]

        from unittest.mock import AsyncMock

        mock_tool_retriever = AsyncMock()
        mock_tool_retriever.index_tools_async = AsyncMock()

        with patch(
            "fivccliche.modules.agent_configs.routers.create_tool_retriever_async"
        ) as mock_create:
            mock_create.return_value = mock_tool_retriever

            # Call indexing for user1
            response1 = client.post(
                "/configs/tools/index",
                headers={"Authorization": f"Bearer {user1_token}"},
            )
            assert response1.status_code == 200

            # Get the space_id used for user1
            call_kwargs_1 = mock_create.call_args.kwargs
            space_id_1 = call_kwargs_1["space_id"]

            # Call indexing for user2
            response2 = client.post(
                "/configs/tools/index",
                headers={"Authorization": f"Bearer {user2_token}"},
            )
            assert response2.status_code == 200

            # Get the space_id used for user2
            call_kwargs_2 = mock_create.call_args.kwargs
            space_id_2 = call_kwargs_2["space_id"]

            # Verify different users have different space_ids
            assert space_id_1 != space_id_2

    def test_create_tool_config_unauthorized(self, client: TestClient):
        """Test creating tool config without authentication."""
        response = client.post(
            "/configs/tools/",
            json={
                "id": "test-tool",
                "transport": "stdio",
            },
        )
        assert response.status_code == 401

    def test_create_tool_config(self, client: TestClient, auth_token: str):
        """Test creating a tool config."""
        response = client.post(
            "/configs/tools/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "test-tool",
                "description": "Test tool",
                "transport": "stdio",
                "command": "python",
                "args": ["script.py"],
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "test-tool"
        assert data["description"] == "Test tool"
        assert data["transport"] == "stdio"
        assert data["command"] == "python"
        assert data["args"] == ["script.py"]
        assert data["is_active"] is True  # Default value should be True
        assert "uuid" in data
        assert "user_uuid" in data

    def test_list_tool_configs(self, client: TestClient, auth_token: str):
        """Test listing tool configs."""
        # Create multiple tool configs
        for i in range(3):
            client.post(
                "/configs/tools/",
                headers={"Authorization": f"Bearer {auth_token}"},
                json={
                    "id": f"tool-{i}",
                    "description": f"Tool {i}",
                    "transport": "stdio",
                },
            )

        response = client.get(
            "/configs/tools/",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["results"]) == 3

    def test_list_tool_configs_with_pagination(self, client: TestClient, auth_token: str):
        """Test listing tool configs with pagination."""
        # Create multiple tool configs
        for i in range(5):
            client.post(
                "/configs/tools/",
                headers={"Authorization": f"Bearer {auth_token}"},
                json={
                    "id": f"tool-page-{i}",
                    "description": f"Tool page {i}",
                    "transport": "stdio",
                },
            )

        response = client.get(
            "/configs/tools/?skip=0&limit=2",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["results"]) == 2

    def test_get_tool_config(self, client: TestClient, auth_token: str):
        """Test getting a tool config."""
        create_response = client.post(
            "/configs/tools/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "tool-get",
                "description": "Tool get",
                "transport": "sse",
            },
        )
        config_uuid = create_response.json()["uuid"]

        response = client.get(
            f"/configs/tools/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "tool-get"
        assert data["transport"] == "sse"
        assert "user_uuid" in data

    def test_get_tool_config_not_found(self, client: TestClient, auth_token: str):
        """Test getting a non-existent tool config."""
        response = client.get(
            "/configs/tools/nonexistent-uuid",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 404

    def test_update_tool_config(self, client: TestClient, auth_token: str):
        """Test updating a tool config."""
        create_response = client.post(
            "/configs/tools/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "tool-update",
                "description": "Tool update",
                "transport": "stdio",
                "command": "python",
            },
        )
        config_uuid = create_response.json()["uuid"]

        response = client.patch(
            f"/configs/tools/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "tool-update",
                "description": "Tool update",
                "transport": "sse",
                "command": "node",
                "url": "http://example.com",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["transport"] == "sse"
        assert data["command"] == "node"
        assert data["url"] == "http://example.com"

    def test_update_tool_config_is_active(self, client: TestClient, auth_token: str):
        """Test updating the is_active field of a tool config."""
        create_response = client.post(
            "/configs/tools/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "tool-is-active",
                "description": "Tool is active",
                "transport": "stdio",
            },
        )
        config_uuid = create_response.json()["uuid"]
        assert create_response.json()["is_active"] is True

        # Update to deactivate
        response = client.patch(
            f"/configs/tools/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "tool-is-active",
                "description": "Tool is active",
                "transport": "stdio",
                "is_active": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_active"] is False

        # Update to reactivate
        response = client.patch(
            f"/configs/tools/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "tool-is-active",
                "description": "Tool is active",
                "transport": "stdio",
                "is_active": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_active"] is True

    def test_delete_tool_config(self, client: TestClient, auth_token: str):
        """Test deleting a tool config."""
        create_response = client.post(
            "/configs/tools/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "tool-delete",
                "description": "Tool delete",
                "transport": "stdio",
            },
        )
        config_uuid = create_response.json()["uuid"]

        response = client.delete(
            f"/configs/tools/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 204

        response = client.get(
            f"/configs/tools/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 404

    def test_tool_config_with_all_fields(self, client: TestClient, auth_token: str):
        """Test creating a tool config with all fields."""
        response = client.post(
            "/configs/tools/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "tool-full",
                "transport": "streamable_http",
                "description": "A complete tool config",
                "command": "bash",
                "args": ["script.sh", "--verbose"],
                "env": {"PATH": "/usr/bin", "DEBUG": "true"},
                "url": "http://localhost:8000/tool",
                "is_active": True,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "tool-full"
        assert data["transport"] == "streamable_http"
        assert data["description"] == "A complete tool config"
        assert data["command"] == "bash"
        assert data["args"] == ["script.sh", "--verbose"]
        assert data["env"] == {"PATH": "/usr/bin", "DEBUG": "true"}
        assert data["url"] == "http://localhost:8000/tool"
        assert data["is_active"] is True

    def test_update_tool_config_via_repository(self, client: TestClient, auth_token: str):
        """Test updating a tool config via repository (create new)."""
        # This test verifies that the update_tool_config_async method works
        # by creating a new tool config through the API
        response = client.post(
            "/configs/tools/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "tool-repo-update",
                "description": "Tool repo update",
                "transport": "stdio",
                "command": "python",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "tool-repo-update"
        assert data["transport"] == "stdio"
        assert data["command"] == "python"

    def test_update_tool_config_via_repository_update_existing(
        self, client: TestClient, auth_token: str
    ):
        """Test updating an existing tool config via repository."""
        # Create initial config
        create_response = client.post(
            "/configs/tools/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "tool-repo-update-existing",
                "description": "Initial description",
                "transport": "stdio",
                "command": "python",
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]

        # Update via PATCH endpoint
        update_response = client.patch(
            f"/configs/tools/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "tool-repo-update-existing",
                "description": "Updated description",
                "transport": "sse",
                "command": "node",
                "url": "http://example.com",
            },
        )
        assert update_response.status_code == 200
        data = update_response.json()
        assert data["description"] == "Updated description"
        assert data["transport"] == "sse"
        assert data["command"] == "node"
        assert data["url"] == "http://example.com"

    def test_update_tool_config_partial_update(self, client: TestClient, auth_token: str):
        """Test partial update of a tool config."""
        # Create initial config
        create_response = client.post(
            "/configs/tools/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "tool-partial-update",
                "description": "Initial description",
                "transport": "stdio",
                "command": "python",
                "args": ["script.py"],
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]

        # Partial update - only change description and transport
        update_response = client.patch(
            f"/configs/tools/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "tool-partial-update",
                "description": "Updated description",
                "transport": "sse",
            },
        )
        assert update_response.status_code == 200
        data = update_response.json()
        assert data["description"] == "Updated description"
        assert data["transport"] == "sse"
        # Command and args should remain unchanged
        assert data["command"] == "python"
        assert data["args"] == ["script.py"]


class TestConfigsAuthorizationEmbedding:
    """Test authorization for embedding config endpoints."""

    def test_user_cannot_update_other_user_embedding_config(
        self, client: TestClient, auth_token: str, other_user_token: str
    ):
        """Test that user cannot update another user's embedding config."""
        # Create config as auth_token user
        create_response = client.post(
            "/configs/embeddings/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "emb-auth-user",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]

        # Try to update as other_user_token - should get 404 because other user can't see it
        response = client.patch(
            f"/configs/embeddings/{config_uuid}",
            headers={"Authorization": f"Bearer {other_user_token}"},
            json={
                "id": "emb-auth-user",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
                "description": "Hacked",
            },
        )
        # Other user shouldn't be able to find the config, so 404 is correct
        assert response.status_code == 404

    def test_user_cannot_delete_other_user_embedding_config(
        self, client: TestClient, auth_token: str, other_user_token: str
    ):
        """Test that user cannot delete another user's embedding config."""
        # Create config as auth_token user
        create_response = client.post(
            "/configs/embeddings/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "emb-delete-auth",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]

        # Try to delete as other_user_token - should get 404 because other user can't see it
        response = client.delete(
            f"/configs/embeddings/{config_uuid}",
            headers={"Authorization": f"Bearer {other_user_token}"},
        )
        # Other user shouldn't be able to find the config, so 404 is correct
        assert response.status_code == 404


class TestConfigsAuthorizationLLM:
    """Test authorization for LLM config endpoints."""

    def test_user_cannot_update_other_user_llm_config(
        self, client: TestClient, auth_token: str, other_user_token: str
    ):
        """Test that user cannot update another user's LLM config."""
        # Create config as auth_token user
        create_response = client.post(
            "/configs/models/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "llm-auth-user",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]

        # Try to update as other_user_token - should get 404 because other user can't see it
        response = client.patch(
            f"/configs/models/{config_uuid}",
            headers={"Authorization": f"Bearer {other_user_token}"},
            json={
                "id": "llm-auth-user",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
                "description": "Hacked",
            },
        )
        # Other user shouldn't be able to find the config, so 404 is correct
        assert response.status_code == 404

    def test_user_cannot_delete_other_user_llm_config(
        self, client: TestClient, auth_token: str, other_user_token: str
    ):
        """Test that user cannot delete another user's LLM config."""
        # Create config as auth_token user
        create_response = client.post(
            "/configs/models/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "llm-delete-auth",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]

        # Try to delete as other_user_token - should get 404 because other user can't see it
        response = client.delete(
            f"/configs/models/{config_uuid}",
            headers={"Authorization": f"Bearer {other_user_token}"},
        )
        # Other user shouldn't be able to find the config, so 404 is correct
        assert response.status_code == 404


class TestConfigsAuthorizationAgent:
    """Test authorization for agent config endpoints."""

    def test_user_cannot_update_other_user_agent_config(
        self, client: TestClient, auth_token: str, other_user_token: str
    ):
        """Test that user cannot update another user's agent config."""
        # First create an LLM config for the agent to reference
        llm_response = client.post(
            "/configs/models/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "llm-for-agent",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
            },
        )
        assert llm_response.status_code == 201

        # Create agent config as auth_token user
        create_response = client.post(
            "/configs/agents/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "agent-auth-user",
                "model_id": "llm-for-agent",
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]

        # Try to update as other_user_token - should get 404 because other user can't see it
        response = client.patch(
            f"/configs/agents/{config_uuid}",
            headers={"Authorization": f"Bearer {other_user_token}"},
            json={
                "id": "agent-auth-user",
                "model_id": "llm-for-agent",
                "description": "Hacked",
            },
        )
        # Other user shouldn't be able to find the config, so 404 is correct
        assert response.status_code == 404

    def test_user_cannot_delete_other_user_agent_config(
        self, client: TestClient, auth_token: str, other_user_token: str
    ):
        """Test that user cannot delete another user's agent config."""
        # First create an LLM config for the agent to reference
        llm_response = client.post(
            "/configs/models/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "llm-for-agent-delete",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
            },
        )
        assert llm_response.status_code == 201

        # Create agent config as auth_token user
        create_response = client.post(
            "/configs/agents/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "agent-delete-auth",
                "model_id": "llm-for-agent-delete",
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]

        # Try to delete as other_user_token - should get 404 because other user can't see it
        response = client.delete(
            f"/configs/agents/{config_uuid}",
            headers={"Authorization": f"Bearer {other_user_token}"},
        )
        # Other user shouldn't be able to find the config, so 404 is correct
        assert response.status_code == 404


class TestConfigsAuthorizationTool:
    """Test authorization for tool config endpoints."""

    def test_user_cannot_update_other_user_tool_config(
        self, client: TestClient, auth_token: str, other_user_token: str
    ):
        """Test that user cannot update another user's tool config."""
        # Create config as auth_token user with all required fields
        create_response = client.post(
            "/configs/tools/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "tool-auth-user",
                "transport": "stdio",
                "description": "Test tool",
                "command": "python",
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]

        # Try to update as other_user_token - should get 404 because other user can't see it
        response = client.patch(
            f"/configs/tools/{config_uuid}",
            headers={"Authorization": f"Bearer {other_user_token}"},
            json={
                "id": "tool-auth-user",
                "transport": "sse",
                "description": "Hacked",
                "command": "node",
            },
        )
        # Other user shouldn't be able to find the config, so 404 is correct
        assert response.status_code == 404

    def test_user_cannot_delete_other_user_tool_config(
        self, client: TestClient, auth_token: str, other_user_token: str
    ):
        """Test that user cannot delete another user's tool config."""
        # Create config as auth_token user with all required fields
        create_response = client.post(
            "/configs/tools/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "tool-delete-auth",
                "transport": "stdio",
                "description": "Test tool",
                "command": "python",
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]

        # Try to delete as other_user_token - should get 404 because other user can't see it
        response = client.delete(
            f"/configs/tools/{config_uuid}",
            headers={"Authorization": f"Bearer {other_user_token}"},
        )
        # Other user shouldn't be able to find the config, so 404 is correct
        assert response.status_code == 404


class TestGlobalConfigAuthorization:
    """Test cases for global config authorization (superuser privileges)."""

    def test_superuser_can_update_global_embedding_config(
        self, client: TestClient, admin_token: str
    ):
        """Test that superuser can update global embedding configs."""
        # Create global config as superuser
        create_response = client.post(
            "/configs/embeddings/",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "id": "global-embedding-1",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]
        assert create_response.json()["user_uuid"] is None  # Global config

        # Update as superuser - should succeed
        response = client.patch(
            f"/configs/embeddings/{config_uuid}",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "id": "global-embedding-1",
                "model": "text-embedding-3-large",
            },
        )
        assert response.status_code == 200
        assert response.json()["model"] == "text-embedding-3-large"

    def test_superuser_can_delete_global_embedding_config(
        self, client: TestClient, admin_token: str
    ):
        """Test that superuser can delete global embedding configs."""
        # Create global config as superuser
        create_response = client.post(
            "/configs/embeddings/",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "id": "global-embedding-2",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]

        # Delete as superuser - should succeed
        response = client.delete(
            f"/configs/embeddings/{config_uuid}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == 204

    def test_regular_user_cannot_update_global_embedding_config(
        self, client: TestClient, admin_token: str, auth_token: str
    ):
        """Test that regular user cannot update global embedding configs."""
        # Create global config as superuser
        create_response = client.post(
            "/configs/embeddings/",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "id": "global-embedding-3",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]

        # Try to update as regular user - should fail with 403
        response = client.patch(
            f"/configs/embeddings/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "global-embedding-3",
                "model": "text-embedding-3-large",
            },
        )
        assert response.status_code == 403
        assert "Cannot update global configs" in response.json()["detail"]

    def test_regular_user_cannot_delete_global_embedding_config(
        self, client: TestClient, admin_token: str, auth_token: str
    ):
        """Test that regular user cannot delete global embedding configs."""
        # Create global config as superuser
        create_response = client.post(
            "/configs/embeddings/",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "id": "global-embedding-4",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]

        # Try to delete as regular user - should fail with 403
        response = client.delete(
            f"/configs/embeddings/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 403
        assert "Cannot delete global configs" in response.json()["detail"]

    def test_superuser_cannot_update_other_user_config(
        self, client: TestClient, admin_token: str, auth_token: str
    ):
        """Test that superuser cannot update another user's config.

        Note: Superusers get 404 because they can't see other users' configs
        (GET queries filter by user_uuid). This is the expected behavior.
        """
        # Create user-specific config as regular user
        create_response = client.post(
            "/configs/embeddings/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "user-embedding-1",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]
        assert create_response.json()["user_uuid"] is not None  # User-specific config

        # Try to update as superuser - gets 404 because can't see other users' configs
        response = client.patch(
            f"/configs/embeddings/{config_uuid}",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "id": "user-embedding-1",
                "model": "text-embedding-3-large",
            },
        )
        assert response.status_code == 404

    def test_superuser_cannot_delete_other_user_config(
        self, client: TestClient, admin_token: str, auth_token: str
    ):
        """Test that superuser cannot delete another user's config.

        Note: Superusers get 404 because they can't see other users' configs
        (GET queries filter by user_uuid). This is the expected behavior.
        """
        # Create user-specific config as regular user
        create_response = client.post(
            "/configs/embeddings/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "user-embedding-2",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]

        # Try to delete as superuser - gets 404 because can't see other users' configs
        response = client.delete(
            f"/configs/embeddings/{config_uuid}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == 404

    def test_superuser_can_update_global_llm_config(self, client: TestClient, admin_token: str):
        """Test that superuser can update global LLM configs."""
        # Create global config as superuser
        create_response = client.post(
            "/configs/models/",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "id": "global-llm-1",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]

        # Update as superuser - should succeed
        response = client.patch(
            f"/configs/models/{config_uuid}",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "id": "global-llm-1",
                "model": "gpt-4-turbo",
            },
        )
        assert response.status_code == 200
        assert response.json()["model"] == "gpt-4-turbo"

    def test_superuser_can_delete_global_llm_config(self, client: TestClient, admin_token: str):
        """Test that superuser can delete global LLM configs."""
        # Create global config as superuser
        create_response = client.post(
            "/configs/models/",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "id": "global-llm-2",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]

        # Delete as superuser - should succeed
        response = client.delete(
            f"/configs/models/{config_uuid}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == 204

    def test_regular_user_cannot_update_global_llm_config(
        self, client: TestClient, admin_token: str, auth_token: str
    ):
        """Test that regular user cannot update global LLM configs."""
        # Create global config as superuser
        create_response = client.post(
            "/configs/models/",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "id": "global-llm-3",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]

        # Try to update as regular user - should fail with 403
        response = client.patch(
            f"/configs/models/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "global-llm-3",
                "model": "gpt-4-turbo",
            },
        )
        assert response.status_code == 403
        assert "Cannot update global configs" in response.json()["detail"]

    def test_regular_user_cannot_delete_global_llm_config(
        self, client: TestClient, admin_token: str, auth_token: str
    ):
        """Test that regular user cannot delete global LLM configs."""
        # Create global config as superuser
        create_response = client.post(
            "/configs/models/",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "id": "global-llm-4",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]

        # Try to delete as regular user - should fail with 403
        response = client.delete(
            f"/configs/models/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 403
        assert "Cannot delete global configs" in response.json()["detail"]


# ============================================================================
# Regression Tests for Models Backward Compatibility
# ============================================================================


class TestConfigsBackwardCompatibilityEmbedding:
    """Backward compatibility tests for embedding config API responses."""

    def test_embedding_config_response_has_all_fields(self, client: TestClient, auth_token: str):
        """Test that embedding config response includes all expected fields."""
        create_response = client.post(
            "/configs/embeddings/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "embedding-bc-1",
                "description": "Test embedding",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
                "base_url": "https://api.openai.com",
                "dimension": 1536,
            },
        )
        assert create_response.status_code == 201
        data = create_response.json()

        # Verify all fields are present
        assert "uuid" in data
        assert "id" in data
        assert "description" in data
        assert "provider" in data
        assert "model" in data
        assert "api_key" in data
        assert "base_url" in data
        assert "dimension" in data
        assert "user_uuid" in data

    def test_embedding_config_response_field_types(self, client: TestClient, auth_token: str):
        """Test that embedding config response has correct field types."""
        create_response = client.post(
            "/configs/embeddings/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "embedding-bc-2",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
                "dimension": 1536,
            },
        )
        data = create_response.json()

        assert isinstance(data["uuid"], str)
        assert isinstance(data["id"], str)
        assert isinstance(data["provider"], str)
        assert isinstance(data["model"], str)
        assert isinstance(data["api_key"], str)
        assert isinstance(data["dimension"], int)
        assert data["user_uuid"] is None or isinstance(data["user_uuid"], str)

    def test_embedding_config_response_preserves_null_values(
        self, client: TestClient, auth_token: str
    ):
        """Test that embedding config response preserves null/None values."""
        create_response = client.post(
            "/configs/embeddings/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "embedding-bc-3",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
            },
        )
        data = create_response.json()

        # These should be null in the response
        assert data["description"] is None
        assert data["base_url"] is None


class TestConfigsBackwardCompatibilityLLM:
    """Backward compatibility tests for LLM config API responses."""

    def test_llm_config_response_has_all_fields(self, client: TestClient, auth_token: str):
        """Test that LLM config response includes all expected fields."""
        create_response = client.post(
            "/configs/models/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "llm-bc-1",
                "description": "Test LLM",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
                "base_url": "https://api.openai.com",
                "temperature": 0.7,
                "max_tokens": 2048,
            },
        )
        assert create_response.status_code == 201
        data = create_response.json()

        # Verify all fields are present
        assert "uuid" in data
        assert "id" in data
        assert "description" in data
        assert "provider" in data
        assert "model" in data
        assert "api_key" in data
        assert "base_url" in data
        assert "temperature" in data
        assert "max_tokens" in data
        assert "user_uuid" in data

    def test_llm_config_response_field_types(self, client: TestClient, auth_token: str):
        """Test that LLM config response has correct field types."""
        create_response = client.post(
            "/configs/models/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "llm-bc-2",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
                "temperature": 0.5,
                "max_tokens": 4096,
            },
        )
        data = create_response.json()

        assert isinstance(data["uuid"], str)
        assert isinstance(data["id"], str)
        assert isinstance(data["provider"], str)
        assert isinstance(data["model"], str)
        assert isinstance(data["api_key"], str)
        assert isinstance(data["temperature"], float)
        assert isinstance(data["max_tokens"], int)


class TestConfigsBackwardCompatibilityAgent:
    """Backward compatibility tests for agent config API responses."""

    def test_agent_config_response_has_all_fields(self, client: TestClient, auth_token: str):
        """Test that agent config response includes all expected fields."""
        create_response = client.post(
            "/configs/agents/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "agent-bc-1",
                "description": "Test agent",
                "model_id": "llm-1",
                "tool_ids": ["tool1", "tool2"],
                "system_prompt": "You are helpful",
                "response_format": {"type": "object", "properties": {"answer": {"type": "string"}}},
            },
        )
        assert create_response.status_code == 201
        data = create_response.json()

        # Verify all fields are present
        assert "uuid" in data
        assert "id" in data
        assert "description" in data
        assert "model_id" in data
        assert "tool_ids" in data
        assert "system_prompt" in data
        assert "response_format" in data
        assert "user_uuid" in data

    def test_agent_config_response_json_field_preservation(
        self, client: TestClient, auth_token: str
    ):
        """Test that agent config response preserves JSON field data."""
        response_format = {"type": "object", "properties": {"answer": {"type": "string"}}}
        create_response = client.post(
            "/configs/agents/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "agent-bc-2",
                "model_id": "llm-1",
                "tool_ids": ["tool1", "tool2"],
                "response_format": response_format,
            },
        )
        data = create_response.json()

        assert data["tool_ids"] == ["tool1", "tool2"]
        assert data["response_format"] == response_format


class TestConfigsBackwardCompatibilityTool:
    """Backward compatibility tests for tool config API responses."""

    def test_tool_config_response_has_all_fields(self, client: TestClient, auth_token: str):
        """Test that tool config response includes all expected fields."""
        create_response = client.post(
            "/configs/tools/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "tool-bc-1",
                "description": "Test tool",
                "transport": "stdio",
                "command": "python",
                "args": ["script.py"],
                "env": {"VAR": "value"},
                "url": "http://localhost:8000",
                "is_active": True,
            },
        )
        assert create_response.status_code == 201
        data = create_response.json()

        # Verify all fields are present
        assert "uuid" in data
        assert "id" in data
        assert "description" in data
        assert "transport" in data
        assert "command" in data
        assert "args" in data
        assert "env" in data
        assert "url" in data
        assert "is_active" in data
        assert "user_uuid" in data

    def test_tool_config_response_json_field_preservation(
        self, client: TestClient, auth_token: str
    ):
        """Test that tool config response preserves JSON field data."""
        args = ["arg1", "arg2", "--verbose"]
        env = {"PATH": "/usr/bin", "DEBUG": "true"}

        create_response = client.post(
            "/configs/tools/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "tool-bc-2",
                "description": "Test tool with JSON fields",
                "transport": "stdio",
                "command": "bash",
                "args": args,
                "env": env,
            },
        )
        assert create_response.status_code == 201
        data = create_response.json()

        # Tool response includes args and env
        if "args" in data:
            assert data["args"] == args
        if "env" in data:
            assert data["env"] == env

    def test_tool_config_response_preserves_null_json_fields(
        self, client: TestClient, auth_token: str
    ):
        """Test that tool config response preserves null JSON fields."""
        create_response = client.post(
            "/configs/tools/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "tool-bc-3",
                "description": "Minimal tool",
                "transport": "stdio",
            },
        )
        assert create_response.status_code == 201
        data = create_response.json()

        # These should be null in the response if present
        if "args" in data:
            assert data["args"] is None
        if "env" in data:
            assert data["env"] is None


class TestConfigsRegressionIntegrationEnd2End:
    """End-to-end integration regression tests for config operations."""

    def test_full_embedding_workflow(self, client: TestClient, auth_token: str):
        """Test complete embedding config workflow: create, retrieve, update, delete."""
        # Create
        create_response = client.post(
            "/configs/embeddings/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "embedding-e2e-1",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
                "dimension": 1536,
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]
        assert create_response.json()["dimension"] == 1536

        # Retrieve
        get_response = client.get(
            f"/configs/embeddings/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert get_response.status_code == 200
        assert get_response.json()["id"] == "embedding-e2e-1"

        # Update
        update_response = client.patch(
            f"/configs/embeddings/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "embedding-e2e-1",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
                "dimension": 3072,
            },
        )
        assert update_response.status_code == 200
        assert update_response.json()["dimension"] == 3072

        # Delete
        delete_response = client.delete(
            f"/configs/embeddings/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert delete_response.status_code == 204

        # Verify deleted
        get_response = client.get(
            f"/configs/embeddings/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert get_response.status_code == 404

    def test_full_llm_workflow(self, client: TestClient, auth_token: str):
        """Test complete LLM config workflow: create, retrieve, update, delete."""
        # Create
        create_response = client.post(
            "/configs/models/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "llm-e2e-1",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
                "temperature": 0.5,
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]

        # Retrieve
        get_response = client.get(
            f"/configs/models/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert get_response.status_code == 200
        assert get_response.json()["temperature"] == 0.5

        # Update
        update_response = client.patch(
            f"/configs/models/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "llm-e2e-1",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
                "temperature": 0.8,
            },
        )
        assert update_response.status_code == 200
        assert update_response.json()["temperature"] == 0.8

        # Delete
        delete_response = client.delete(
            f"/configs/models/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert delete_response.status_code == 204

    def test_full_agent_workflow(self, client: TestClient, auth_token: str):
        """Test complete agent config workflow: create, retrieve, update, delete."""
        # Create
        create_response = client.post(
            "/configs/agents/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "agent-e2e-1",
                "model_id": "llm-1",
                "system_prompt": "Old prompt",
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]

        # Retrieve
        get_response = client.get(
            f"/configs/agents/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert get_response.status_code == 200
        assert get_response.json()["system_prompt"] == "Old prompt"

        # Update
        update_response = client.patch(
            f"/configs/agents/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "agent-e2e-1",
                "model_id": "llm-1",
                "system_prompt": "New prompt",
            },
        )
        assert update_response.status_code == 200
        assert update_response.json()["system_prompt"] == "New prompt"

        # Delete
        delete_response = client.delete(
            f"/configs/agents/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert delete_response.status_code == 204

    def test_full_tool_workflow(self, client: TestClient, auth_token: str):
        """Test complete tool config workflow: create, retrieve, update, delete."""
        # Create
        create_response = client.post(
            "/configs/tools/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "tool-e2e-1",
                "description": "Test tool",
                "transport": "stdio",
                "command": "python",
                "is_active": True,
            },
        )
        assert create_response.status_code == 201
        config_uuid = create_response.json()["uuid"]

        # Retrieve
        get_response = client.get(
            f"/configs/tools/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert get_response.status_code == 200
        assert get_response.json()["is_active"] is True

        # Update
        update_response = client.patch(
            f"/configs/tools/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "tool-e2e-1",
                "description": "Updated tool",
                "transport": "sse",
                "command": "node",
                "is_active": False,
            },
        )
        assert update_response.status_code == 200
        assert update_response.json()["is_active"] is False

        # Delete
        delete_response = client.delete(
            f"/configs/tools/{config_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert delete_response.status_code == 204

    def test_user_isolation_maintained_across_configs(
        self, client: TestClient, auth_token: str, other_user_token: str
    ):
        """Test that user isolation is maintained across all config types."""
        # Create multiple configs as auth_token user
        emb_response = client.post(
            "/configs/embeddings/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "isolation-emb",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
            },
        )
        llm_response = client.post(
            "/configs/models/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "isolation-llm",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
            },
        )

        # Verify other_user cannot see them
        emb_uuid = emb_response.json()["uuid"]
        llm_uuid = llm_response.json()["uuid"]

        emb_get = client.get(
            f"/configs/embeddings/{emb_uuid}",
            headers={"Authorization": f"Bearer {other_user_token}"},
        )
        assert emb_get.status_code == 404

        llm_get = client.get(
            f"/configs/models/{llm_uuid}",
            headers={"Authorization": f"Bearer {other_user_token}"},
        )
        assert llm_get.status_code == 404
