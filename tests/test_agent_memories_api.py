"""Integration tests for agent_memories read-only HTTP API."""

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel

from fivccliche.modules.users.models import User  # noqa: F401
from fivccliche.services.implements.modules import ModuleSiteImpl
from fivccliche.services.interfaces.agent_memories import (
    MemoryContent,
    MemoryListResult,
    MemoryRecallResult,
)
from fivccliche.utils.deps import get_db_session_async, get_memory_provider_async
from fivccliche.utils import deps
from fivcglue.implements.utils import load_component_site


@pytest.fixture
def client():
    """Create a test client with users + agent_memories modules."""
    import asyncio

    from fivccliche.modules.users import methods

    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db.close()
    database_url = f"sqlite+aiosqlite:///{temp_db.name}"

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

        components_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "fivccliche",
            "settings",
            "services.yml",
        )
        component_site = load_component_site(filename=components_path, fmt="yaml")
        module_site = ModuleSiteImpl(component_site, modules=["users", "agent_memories"])
        app = module_site.create_application()

        async def override_get_db_session_async():
            yield async_session

        app.dependency_overrides[get_db_session_async] = override_get_db_session_async

        loop.run_until_complete(
            methods.create_user_async(
                async_session,
                username="admin",
                email="admin@example.com",
                password="admin123",
                is_superuser=True,
            )
        )

        @asynccontextmanager
        async def override_get_db_session_context_async(session=None):
            yield async_session

        with patch.object(
            deps, "get_db_session_context_async", override_get_db_session_context_async
        ):
            with TestClient(app) as test_client:
                yield test_client

        async def cleanup():
            await async_session.close()
            await engine.dispose()

        loop.run_until_complete(cleanup())
    finally:
        loop.close()
        try:
            Path(temp_db.name).unlink()
        except Exception:
            pass


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
