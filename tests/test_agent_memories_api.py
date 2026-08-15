"""Integration tests for agent_memories HTTP API."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from fivccliche.services.interfaces.agent_memories import (
    MemoryContent,
    MemoryListResult,
    MemoryRecallResult,
    MemoryRetainResult,
)
from fivccliche.utils.deps import get_memory_provider_async
from tests.conftest import make_api_client


@pytest.fixture
def client():
    """Create a test client with users + agent_memories modules."""
    yield from make_api_client(["users", "agent_memories"])


@pytest.fixture
def auth_token(client: TestClient) -> str:
    response = client.post(
        "/users/login",
        json={"username": "admin", "password": "admin123"},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def _override_memory_provider(app, provider):
    async def override():
        return provider

    app.dependency_overrides[get_memory_provider_async] = override


class TestMemoriesApiAuthAndMounting:
    def test_list_requires_auth(self, client: TestClient):
        response = client.get("/memories/")
        assert response.status_code == 401

    def test_recall_requires_auth(self, client: TestClient):
        response = client.get("/memories/recall/", params={"query": "hi"})
        assert response.status_code == 401

    def test_retain_requires_auth(self, client: TestClient):
        response = client.post("/memories/retain/", json={"content": "hello"})
        assert response.status_code == 401

    def test_list_returns_503_when_provider_unmounted(self, client: TestClient, auth_token: str):
        _override_memory_provider(client.app, None)
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/memories/", headers=headers)
        assert response.status_code == 503
        assert response.json()["detail"] == "Memory provider is not mounted"

    def test_recall_returns_503_when_provider_unmounted(self, client: TestClient, auth_token: str):
        _override_memory_provider(client.app, None)
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get(
            "/memories/recall/",
            params={"query": "hi"},
            headers=headers,
        )
        assert response.status_code == 503
        assert response.json()["detail"] == "Memory provider is not mounted"

    def test_retain_returns_503_when_provider_unmounted(self, client: TestClient, auth_token: str):
        _override_memory_provider(client.app, None)
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.post("/memories/retain/", json={"content": "hello"}, headers=headers)
        assert response.status_code == 503
        assert response.json()["detail"] == "Memory provider is not mounted"

    def test_retain_returns_403_for_regular_user(self, client: TestClient, auth_token: str):
        memory = MagicMock()
        memory.retain_async = AsyncMock()
        provider = MagicMock()
        provider.get_memory.return_value = memory
        _override_memory_provider(client.app, provider)

        admin_headers = {"Authorization": f"Bearer {auth_token}"}
        created = client.post(
            "/users/",
            json={"username": "bob", "email": "bob@example.com", "password": "bob12345"},
            headers=admin_headers,
        )
        assert created.status_code == 201
        login = client.post("/users/login", json={"username": "bob", "password": "bob12345"})
        assert login.status_code == 200
        user_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

        response = client.post("/memories/retain/", json={"content": "hello"}, headers=user_headers)
        assert response.status_code == 403
        assert response.json()["detail"] == "Not a super user"
        memory.retain_async.assert_not_awaited()


class TestMemoriesApiSuccess:
    def test_list_memories_returns_paginated_results(self, client: TestClient, auth_token: str):
        memory = MagicMock()
        memory.list_async = AsyncMock(
            return_value=MemoryListResult(
                total=2,
                items=[
                    MemoryContent(
                        id="m1",
                        content="Alice loves AI",
                        score=0.9,
                        categories=["world"],
                        created_at=datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc),
                    )
                ],
            )
        )
        provider = MagicMock()
        provider.get_memory.return_value = memory
        _override_memory_provider(client.app, provider)

        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/memories/", params={"skip": 0, "limit": 10}, headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["results"]) == 1
        assert data["results"][0]["id"] == "m1"
        assert data["results"][0]["content"] == "Alice loves AI"
        assert data["results"][0]["categories"] == ["world"]
        memory.list_async.assert_awaited_once_with(skip=0, limit=10)
        provider.get_memory.assert_called_once()
        assert provider.get_memory.call_args.kwargs["space_id"]

    def test_recall_memories_returns_results(self, client: TestClient, auth_token: str):
        memory = MagicMock()
        memory.recall_async = AsyncMock(
            return_value=MemoryRecallResult(
                items=[
                    MemoryContent(
                        id="r1",
                        content="Alice loves AI",
                        score=0.95,
                    )
                ],
            )
        )
        provider = MagicMock()
        provider.get_memory.return_value = memory
        _override_memory_provider(client.app, provider)

        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get(
            "/memories/recall/",
            params={"query": "what does Alice like?"},
            headers=headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["id"] == "r1"
        assert data["results"][0]["content"] == "Alice loves AI"
        memory.recall_async.assert_awaited_once_with("what does Alice like?")

    def test_retain_memories_returns_result(self, client: TestClient, auth_token: str):
        memory = MagicMock()
        memory.retain_async = AsyncMock(
            return_value=MemoryRetainResult(
                success=True, count=1, ids=["m1"], raw={"ignored": True}
            )
        )
        provider = MagicMock()
        provider.get_memory.return_value = memory
        _override_memory_provider(client.app, provider)

        me = client.get("/users/self/", headers={"Authorization": f"Bearer {auth_token}"})
        assert me.status_code == 200
        admin_uuid = me.json()["uuid"]

        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.post("/memories/retain/", json={"content": "hello"}, headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 1
        assert data["ids"] == ["m1"]
        assert "raw" not in data
        memory.retain_async.assert_awaited_once_with("hello")
        provider.get_memory.assert_called_once_with(space_id=admin_uuid)
