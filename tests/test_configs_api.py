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
async def test_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        database_url = f"sqlite+aiosqlite:///{db_path}"

        engine = create_async_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=NullPool,
        )

        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        async_session = AsyncSession(engine, expire_on_commit=False)
        try:
            yield async_session
        finally:
            await async_session.close()
            await engine.dispose()


@pytest.fixture
async def auth_token(client: TestClient):
    """Generate a JWT token for a test user."""
    # Create a user via the API
    client.post(
        "/users/",
        json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "password123",
        },
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
async def client(test_db):
    """Create a test client with temporary database."""
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
    module_site = ModuleSiteImpl(component_site, modules=["users", "configs"])
    app = module_site.create_application()

    # Override the database dependency with async generator
    async def override_get_db_session_async():
        yield test_db

    app.dependency_overrides[get_db_session_async] = override_get_db_session_async

    with TestClient(app) as client:
        yield client


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
        client.post(
            "/configs/embeddings/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "embedding-get",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
            },
        )

        response = client.get(
            "/configs/embeddings/embedding-get",
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
        client.post(
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

        response = client.patch(
            "/configs/embeddings/embedding-update",
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
        client.post(
            "/configs/embeddings/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "embedding-delete",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
            },
        )

        response = client.delete(
            "/configs/embeddings/embedding-delete",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 204

        # Verify it's deleted
        response = client.get(
            "/configs/embeddings/embedding-delete",
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
        client.post(
            "/configs/models/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "llm-get",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
            },
        )

        response = client.get(
            "/configs/models/llm-get",
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
        client.post(
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

        response = client.patch(
            "/configs/models/llm-update",
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
        client.post(
            "/configs/models/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "llm-delete",
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
            },
        )

        response = client.delete(
            "/configs/models/llm-delete",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 204

        response = client.get(
            "/configs/models/llm-delete",
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
        client.post(
            "/configs/agents/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "agent-get",
                "model_id": "model123",
            },
        )

        response = client.get(
            "/configs/agents/agent-get",
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
        client.post(
            "/configs/agents/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "agent-update",
                "model_id": "model123",
                "system_prompt": "Old prompt",
            },
        )

        response = client.patch(
            "/configs/agents/agent-update",
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
        client.post(
            "/configs/agents/",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "id": "agent-delete",
                "model_id": "model123",
            },
        )

        response = client.delete(
            "/configs/agents/agent-delete",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 204

        response = client.get(
            "/configs/agents/agent-delete",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 404
