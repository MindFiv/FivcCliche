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
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "agent-1"
        assert data["model_id"] == "model123"

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
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["system_prompt"] == "New prompt"

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
        response = client.post("/configs/tools/indexing")
        assert response.status_code == 401

    def test_indexing_tool_success(self, client: TestClient, auth_token: str):
        """Test successful tool indexing."""
        from unittest.mock import Mock, patch

        # Mock the create_tool_retriever function and its return value
        mock_tool_retriever = Mock()
        mock_tool_retriever.index_tools = Mock()

        with patch("fivccliche.modules.agent_configs.routers.create_tool_retriever") as mock_create:
            mock_create.return_value = mock_tool_retriever

            response = client.post(
                "/configs/tools/indexing",
                headers={"Authorization": f"Bearer {auth_token}"},
            )

            assert response.status_code == 200
            # Verify create_tool_retriever was called
            assert mock_create.called
            # Verify index_tools was called on the retriever
            mock_tool_retriever.index_tools.assert_called_once()

    def test_indexing_tool_with_existing_tools(self, client: TestClient, auth_token: str):
        """Test indexing tool when user has existing tool configs."""
        from unittest.mock import Mock, patch

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
        mock_tool_retriever = Mock()
        mock_tool_retriever.index_tools = Mock()

        with patch("fivccliche.modules.agent_configs.routers.create_tool_retriever") as mock_create:
            mock_create.return_value = mock_tool_retriever

            response = client.post(
                "/configs/tools/indexing",
                headers={"Authorization": f"Bearer {auth_token}"},
            )

            assert response.status_code == 200
            # Verify the tool retriever was created with correct parameters
            assert mock_create.called
            call_kwargs = mock_create.call_args.kwargs
            assert "tool_backend" in call_kwargs
            assert "tool_repository" in call_kwargs
            assert "embedding_backend" in call_kwargs
            assert "embedding_repository" in call_kwargs
            assert "space_id" in call_kwargs
            # Verify index_tools was called
            mock_tool_retriever.index_tools.assert_called_once()

    def test_indexing_tool_calls_config_provider_methods(self, client: TestClient, auth_token: str):
        """Test that indexing tool calls all required config provider methods."""
        from unittest.mock import Mock, patch

        mock_tool_retriever = Mock()
        mock_tool_retriever.index_tools = Mock()

        with patch("fivccliche.modules.agent_configs.routers.create_tool_retriever") as mock_create:
            mock_create.return_value = mock_tool_retriever

            response = client.post(
                "/configs/tools/indexing",
                headers={"Authorization": f"Bearer {auth_token}"},
            )

            assert response.status_code == 200
            # Verify create_tool_retriever was called with all required parameters
            call_kwargs = mock_create.call_args.kwargs
            assert "tool_backend" in call_kwargs
            assert "tool_repository" in call_kwargs
            assert "embedding_backend" in call_kwargs
            assert "embedding_repository" in call_kwargs
            assert "space_id" in call_kwargs

    def test_indexing_tool_exception_handling(self, client: TestClient, auth_token: str):
        """Test indexing tool handles exceptions from index_tools."""
        from unittest.mock import Mock, patch

        mock_tool_retriever = Mock()
        mock_tool_retriever.index_tools = Mock(side_effect=Exception("Indexing failed"))

        with patch("fivccliche.modules.agent_configs.routers.create_tool_retriever") as mock_create:
            mock_create.return_value = mock_tool_retriever

            # The endpoint doesn't explicitly handle exceptions, so it should raise
            with pytest.raises(Exception) as exc_info:  # noqa
                client.post(
                    "/configs/tools/indexing",
                    headers={"Authorization": f"Bearer {auth_token}"},
                )
            assert "Indexing failed" in str(exc_info.value)

    def test_indexing_tool_with_different_users(self, client: TestClient):
        """Test that indexing tool uses correct user context."""
        from unittest.mock import Mock, patch

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

        mock_tool_retriever = Mock()
        mock_tool_retriever.index_tools = Mock()

        with patch("fivccliche.modules.agent_configs.routers.create_tool_retriever") as mock_create:
            mock_create.return_value = mock_tool_retriever

            # Call indexing for user1
            response1 = client.post(
                "/configs/tools/indexing",
                headers={"Authorization": f"Bearer {user1_token}"},
            )
            assert response1.status_code == 200

            # Get the space_id used for user1
            call_kwargs_1 = mock_create.call_args.kwargs
            space_id_1 = call_kwargs_1["space_id"]

            # Call indexing for user2
            response2 = client.post(
                "/configs/tools/indexing",
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
        assert "uuid" in data

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
