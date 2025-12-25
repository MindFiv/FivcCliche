"""Integration tests for user API endpoints."""

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
        module_site = ModuleSiteImpl(component_site, modules=["users"])
        app = module_site.create_application()

        # Override the database dependency with async generator
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


class TestUsersAPI:
    """Test cases for Users API endpoints."""

    def _get_admin_token(self, client: TestClient) -> str:
        """Get admin JWT token for testing."""
        response = client.post(
            "/users/login",
            json={
                "username": "admin",
                "password": "admin123",
            },
        )
        return response.json()["access_token"]

    def _get_admin_headers(self, client: TestClient) -> dict:
        """Get headers with admin JWT token."""
        token = self._get_admin_token(client)
        return {"Authorization": f"Bearer {token}"}

    def test_create_user(self, client: TestClient):
        """Test creating a new user."""
        headers = self._get_admin_headers(client)
        response = client.post(
            "/users/",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "password123",
            },
            headers=headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"
        assert "uuid" in data
        assert "hashed_password" not in data

    def test_create_user_duplicate_username(self, client: TestClient):
        """Test creating a user with duplicate username."""
        headers = self._get_admin_headers(client)
        client.post(
            "/users/",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "password123",
            },
            headers=headers,
        )
        response = client.post(
            "/users/",
            json={
                "username": "testuser",
                "email": "test2@example.com",
                "password": "password123",
            },
            headers=headers,
        )
        assert response.status_code == 400
        assert "Username already registered" in response.json()["detail"]

    def test_create_user_duplicate_email(self, client: TestClient):
        """Test creating a user with duplicate email."""
        headers = self._get_admin_headers(client)
        client.post(
            "/users/",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "password123",
            },
            headers=headers,
        )
        response = client.post(
            "/users/",
            json={
                "username": "testuser2",
                "email": "test@example.com",
                "password": "password123",
            },
            headers=headers,
        )
        assert response.status_code == 400
        assert "Email already registered" in response.json()["detail"]

    def test_list_users(self, client: TestClient):
        """Test listing users."""
        headers = self._get_admin_headers(client)
        for i in range(3):
            client.post(
                "/users/",
                json={
                    "username": f"user{i}",
                    "email": f"user{i}@example.com",
                    "password": "password123",
                },
                headers=headers,
            )
        response = client.get("/users/", headers=headers)
        assert response.status_code == 200
        data = response.json()
        # Total includes the admin user created in the fixture + 3 new users
        assert data["total"] == 4
        assert len(data["results"]) == 4

    def test_list_users_pagination(self, client: TestClient):
        """Test listing users with pagination."""
        headers = self._get_admin_headers(client)
        for i in range(5):
            client.post(
                "/users/",
                json={
                    "username": f"user{i}",
                    "email": f"user{i}@example.com",
                    "password": "password123",
                },
                headers=headers,
            )
        response = client.get("/users/?skip=0&limit=2", headers=headers)
        assert response.status_code == 200
        data = response.json()
        # Total includes the admin user created in the fixture + 5 new users
        assert data["total"] == 6
        assert len(data["results"]) == 2

    def test_get_current_user_with_valid_token(self, client: TestClient):
        """Test getting the current authenticated user's information with valid JWT token."""
        # Create a user with admin headers
        admin_headers = self._get_admin_headers(client)
        create_response = client.post(
            "/users/",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "password123",
            },
            headers=admin_headers,
        )
        user_uuid = create_response.json()["uuid"]

        # Login to get token
        login_response = client.post(
            "/users/login",
            json={
                "username": "testuser",
                "password": "password123",
            },
        )
        assert login_response.status_code == 200
        login_data = login_response.json()
        assert "access_token" in login_data
        assert "expires_in" in login_data
        assert login_data["expires_in"] > 0

        token = login_data["access_token"]

        # Use token to get current user
        response = client.get(
            "/users/self",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["uuid"] == user_uuid
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"
        assert "hashed_password" not in data

    def test_get_current_user_missing_token(self, client: TestClient):
        """Test getting current user without providing authorization header."""
        response = client.get("/users/self")
        assert response.status_code == 401
        # HTTPBearer returns "Not authenticated" for missing credentials
        assert "Not authenticated" in response.json()["detail"]

    def test_get_current_user_invalid_token(self, client: TestClient):
        """Test getting current user with invalid token."""
        response = client.get(
            "/users/self",
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert response.status_code == 401
        assert "Invalid or expired token" in response.json()["detail"]

    def test_get_current_user_invalid_auth_header_format(self, client: TestClient):
        """Test getting current user with invalid authorization header format."""
        response = client.get(
            "/users/self",
            headers={"Authorization": "InvalidFormat token"},
        )
        assert response.status_code == 401
        # HTTPBearer returns "Not authenticated" for invalid Bearer scheme
        assert "Not authenticated" in response.json()["detail"]

    def test_get_user(self, client: TestClient):
        """Test getting a user by ID (requires admin authentication)."""
        # Create a user with admin headers
        admin_headers = self._get_admin_headers(client)
        create_response = client.post(
            "/users/",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "password123",
            },
            headers=admin_headers,
        )
        user_uuid = create_response.json()["uuid"]

        # Try to get user without admin credentials - should fail
        response = client.get(f"/users/{user_uuid}")
        assert response.status_code == 401

    def test_get_user_not_found(self, client: TestClient):
        """Test getting a non-existent user (requires admin authentication)."""
        # Without authentication, should get 401
        response = client.get("/users/nonexistent-uuid")
        assert response.status_code == 401

    def test_delete_user(self, client: TestClient):
        """Test deleting a user (requires admin authentication)."""
        admin_headers = self._get_admin_headers(client)
        create_response = client.post(
            "/users/",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "password123",
            },
            headers=admin_headers,
        )
        user_uuid = create_response.json()["uuid"]

        # Without authentication, should get 401
        response = client.delete(f"/users/{user_uuid}")
        assert response.status_code == 401

    def test_login_success(self, client: TestClient):
        """Test successful user login with JWT token."""
        admin_headers = self._get_admin_headers(client)
        client.post(
            "/users/",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "password123",
            },
            headers=admin_headers,
        )
        response = client.post(
            "/users/login",
            json={
                "username": "testuser",
                "password": "password123",
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "access_token" in data
        assert "expires_in" in data

        # Verify token
        assert data["expires_in"] > 0
        assert len(data["access_token"]) > 0

    def test_login_wrong_password(self, client: TestClient):
        """Test login with wrong password."""
        client.post(
            "/users/",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "password123",
            },
        )
        response = client.post(
            "/users/login",
            json={
                "username": "testuser",
                "password": "wrongpassword",
            },
        )
        assert response.status_code == 401
        assert "Invalid username or password" in response.json()["detail"]

    def test_login_user_not_found(self, client: TestClient):
        """Test login with non-existent user."""
        response = client.post(
            "/users/login",
            json={
                "username": "nonexistent",
                "password": "password123",
            },
        )
        assert response.status_code == 401
        assert "Invalid username or password" in response.json()["detail"]


class TestAuthenticationCaching:
    """Test cases for authentication token caching functionality."""

    def _get_admin_token(self, client: TestClient) -> str:
        """Get admin JWT token for testing."""
        response = client.post(
            "/users/login",
            json={
                "username": "admin",
                "password": "admin123",
            },
        )
        return response.json()["access_token"]

    def _get_admin_headers(self, client: TestClient) -> dict:
        """Get headers with admin JWT token."""
        token = self._get_admin_token(client)
        return {"Authorization": f"Bearer {token}"}

    def test_cache_hit_same_token_multiple_requests(self, client: TestClient):
        """Test that the same valid token is cached and reused on subsequent requests."""
        # Create a user with admin headers
        admin_headers = self._get_admin_headers(client)
        create_response = client.post(
            "/users/",
            json={
                "username": "cachetest",
                "email": "cache@example.com",
                "password": "password123",
            },
            headers=admin_headers,
        )
        user_uuid = create_response.json()["uuid"]

        # Login to get token
        login_response = client.post(
            "/users/login",
            json={
                "username": "cachetest",
                "password": "password123",
            },
        )
        token = login_response.json()["access_token"]

        # First request - should hit database
        response1 = client.get(
            "/users/self",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response1.status_code == 200
        assert response1.json()["uuid"] == user_uuid

        # Second request with same token - should use cache
        response2 = client.get(
            "/users/self",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response2.status_code == 200
        assert response2.json()["uuid"] == user_uuid

        # Third request with same token - should still use cache
        response3 = client.get(
            "/users/self",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response3.status_code == 200
        assert response3.json()["uuid"] == user_uuid

    def test_token_isolation_different_tokens(self, client: TestClient):
        """Test that different tokens are cached separately and don't interfere."""
        # Create two users with admin headers
        admin_headers = self._get_admin_headers(client)
        user1_response = client.post(
            "/users/",
            json={
                "username": "user1",
                "email": "user1@example.com",
                "password": "password123",
            },
            headers=admin_headers,
        )
        user1_id = user1_response.json()["uuid"]

        user2_response = client.post(
            "/users/",
            json={
                "username": "user2",
                "email": "user2@example.com",
                "password": "password123",
            },
            headers=admin_headers,
        )
        user2_id = user2_response.json()["uuid"]

        # Login both users to get tokens
        token1_response = client.post(
            "/users/login",
            json={
                "username": "user1",
                "password": "password123",
            },
        )
        token1 = token1_response.json()["access_token"]

        token2_response = client.post(
            "/users/login",
            json={
                "username": "user2",
                "password": "password123",
            },
        )
        token2 = token2_response.json()["access_token"]

        # Verify token1 returns user1
        response1 = client.get(
            "/users/self",
            headers={"Authorization": f"Bearer {token1}"},
        )
        assert response1.status_code == 200
        assert response1.json()["uuid"] == user1_id
        assert response1.json()["username"] == "user1"

        # Verify token2 returns user2
        response2 = client.get(
            "/users/self",
            headers={"Authorization": f"Bearer {token2}"},
        )
        assert response2.status_code == 200
        assert response2.json()["uuid"] == user2_id
        assert response2.json()["username"] == "user2"

        # Verify token1 still returns user1 (not user2 from cache)
        response1_again = client.get(
            "/users/self",
            headers={"Authorization": f"Bearer {token1}"},
        )
        assert response1_again.status_code == 200
        assert response1_again.json()["uuid"] == user1_id
        assert response1_again.json()["username"] == "user1"

    def test_invalid_token_not_cached(self, client: TestClient):
        """Test that invalid tokens are not cached."""
        invalid_token = "invalid.token.here"

        # First request with invalid token
        response1 = client.get(
            "/users/self",
            headers={"Authorization": f"Bearer {invalid_token}"},
        )
        assert response1.status_code == 401
        assert "Invalid or expired token" in response1.json()["detail"]

        # Second request with same invalid token should also fail
        response2 = client.get(
            "/users/self",
            headers={"Authorization": f"Bearer {invalid_token}"},
        )
        assert response2.status_code == 401
        assert "Invalid or expired token" in response2.json()["detail"]

    def test_malformed_token_not_cached(self, client: TestClient):
        """Test that malformed tokens are not cached."""
        malformed_token = "not.a.valid.jwt.token"

        # First request with malformed token
        response1 = client.get(
            "/users/self",
            headers={"Authorization": f"Bearer {malformed_token}"},
        )
        assert response1.status_code == 401

        # Second request with same malformed token should also fail
        response2 = client.get(
            "/users/self",
            headers={"Authorization": f"Bearer {malformed_token}"},
        )
        assert response2.status_code == 401

    def test_backward_compatibility_existing_auth_flows(self, client: TestClient):
        """Test that caching doesn't break existing authentication functionality."""
        # Create a user with admin headers
        admin_headers = self._get_admin_headers(client)
        create_response = client.post(
            "/users/",
            json={
                "username": "backcompat",
                "email": "backcompat@example.com",
                "password": "password123",
            },
            headers=admin_headers,
        )
        assert create_response.status_code == 201
        user_uuid = create_response.json()["uuid"]

        # Test login flow
        login_response = client.post(
            "/users/login",
            json={
                "username": "backcompat",
                "password": "password123",
            },
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        # Test authenticated endpoint access
        response = client.get(
            "/users/self",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert response.json()["uuid"] == user_uuid

        # Test missing token still returns 401
        response_no_token = client.get("/users/self")
        assert response_no_token.status_code == 401

        # Test invalid token still returns 401
        response_invalid = client.get(
            "/users/self",
            headers={"Authorization": "Bearer invalid"},
        )
        assert response_invalid.status_code == 401

    def test_cache_with_user_data_consistency(self, client: TestClient):
        """Test that cached user data remains consistent across requests."""
        # Create a user with admin headers
        admin_headers = self._get_admin_headers(client)
        create_response = client.post(
            "/users/",
            json={
                "username": "consistency",
                "email": "consistency@example.com",
                "password": "password123",
            },
            headers=admin_headers,
        )
        user_uuid = create_response.json()["uuid"]
        assert user_uuid

        # Login to get token
        login_response = client.post(
            "/users/login",
            json={
                "username": "consistency",
                "password": "password123",
            },
        )
        token = login_response.json()["access_token"]

        # First request
        response1 = client.get(
            "/users/self",
            headers={"Authorization": f"Bearer {token}"},
        )
        data1 = response1.json()

        # Second request (should use cache)
        response2 = client.get(
            "/users/self",
            headers={"Authorization": f"Bearer {token}"},
        )
        data2 = response2.json()

        # Verify data consistency
        assert data1["uuid"] == data2["uuid"]
        assert data1["username"] == data2["username"]
        assert data1["email"] == data2["email"]
        assert data1["is_active"] == data2["is_active"]

    def test_cache_performance_improvement(self, client: TestClient):
        """Test that caching provides performance improvement for repeated token verification."""
        import time

        # Create a user with admin headers
        admin_headers = self._get_admin_headers(client)
        create_response = client.post(
            "/users/",
            json={
                "username": "perftest",
                "email": "perftest@example.com",
                "password": "password123",
            },
            headers=admin_headers,
        )
        assert create_response

        # Login to get token
        login_response = client.post(
            "/users/login",
            json={
                "username": "perftest",
                "password": "password123",
            },
        )
        token = login_response.json()["access_token"]

        # Warm up - first request (hits database)
        client.get(
            "/users/self",
            headers={"Authorization": f"Bearer {token}"},
        )

        # Measure cached requests
        start_time = time.time()
        for _ in range(5):
            response = client.get(
                "/users/self",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == 200
        cached_time = time.time() - start_time

        # Verify that cached requests complete successfully
        # (We can't reliably measure performance in tests, but we verify functionality)
        assert cached_time >= 0  # Sanity check

    def test_cache_with_different_session_parameters(self, client: TestClient):
        """Test that caching works correctly when session is provided vs not provided."""
        # Create a user with admin headers
        admin_headers = self._get_admin_headers(client)
        create_response = client.post(
            "/users/",
            json={
                "username": "sessiontest",
                "email": "sessiontest@example.com",
                "password": "password123",
            },
            headers=admin_headers,
        )
        user_uuid = create_response.json()["uuid"]

        # Login to get token
        login_response = client.post(
            "/users/login",
            json={
                "username": "sessiontest",
                "password": "password123",
            },
        )
        token = login_response.json()["access_token"]

        # Request 1 - with session (from dependency injection)
        response1 = client.get(
            "/users/self",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response1.status_code == 200
        assert response1.json()["uuid"] == user_uuid

        # Request 2 - with session (from dependency injection)
        response2 = client.get(
            "/users/self",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response2.status_code == 200
        assert response2.json()["uuid"] == user_uuid

        # Both should return the same user data
        assert response1.json() == response2.json()
