"""Unit tests for user service layer."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel

from fivccliche.modules.users import methods
from fivccliche.modules.users.services import UserAuthenticatorImpl, UserImpl


@pytest.fixture
async def session():
    """Create a temporary SQLite database for testing."""
    # Create a temporary database file
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        database_url = f"sqlite+aiosqlite:///{db_path}"

        engine = create_async_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=NullPool,
            echo=False,
        )

        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        # Create session
        async_session = AsyncSession(engine, expire_on_commit=False)
        try:
            yield async_session
        finally:
            await async_session.close()
            await engine.dispose()


class TestUserService:
    """Test cases for user service functions."""

    def test_hash_password(self):
        """Test password hashing."""
        password = "test_password_123"
        hashed = methods.hash_user_password(password)
        assert hashed != password
        assert len(hashed) > 0

    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        password = "test_password_123"
        hashed = methods.hash_user_password(password)
        assert methods.verify_user_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password."""
        password = "test_password_123"
        wrong_password = "wrong_password"
        hashed = methods.hash_user_password(password)
        assert methods.verify_user_password(wrong_password, hashed) is False

    async def test_create_user(self, session: AsyncSession):
        """Test creating a new user."""
        user = await methods.create_user_async(
            session,
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        assert user.uuid is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.hashed_password != "password123"
        assert user.is_active is True
        assert user.is_superuser is False

    async def test_get_user_by_uuid(self, session: AsyncSession):
        """Test getting a user by ID."""
        created_user = await methods.create_user_async(
            session,
            username="testuser",
            email="test@example.com",
            password="password123",
        )
        retrieved_user = await methods.get_user_async(session, user_uuid=created_user.uuid)

        assert retrieved_user is not None
        assert retrieved_user.uuid == created_user.uuid
        assert retrieved_user.username == "testuser"

    async def test_get_user_by_username(self, session: AsyncSession):
        """Test getting a user by username."""
        await methods.create_user_async(
            session,
            username="testuser",
            email="test@example.com",
            password="password123",
        )
        user = await methods.get_user_async(session, username="testuser")

        assert user is not None
        assert user.username == "testuser"

    async def test_get_user_by_email(self, session: AsyncSession):
        """Test getting a user by email."""
        await methods.create_user_async(
            session,
            username="testuser",
            email="test@example.com",
            password="password123",
        )
        user = await methods.get_user_async(session, email="test@example.com")

        assert user is not None
        assert user.email == "test@example.com"

    async def test_get_all_users(self, session: AsyncSession):
        """Test getting all users."""
        for i in range(3):
            await methods.create_user_async(
                session,
                username=f"user{i}",
                email=f"user{i}@example.com",
                password="password123",
            )

        users = await methods.list_users_async(session)
        assert len(users) == 3

    async def test_get_all_users_pagination(self, session: AsyncSession):
        """Test getting users with pagination."""
        for i in range(5):
            await methods.create_user_async(
                session,
                username=f"user{i}",
                email=f"user{i}@example.com",
                password="password123",
            )

        users = await methods.list_users_async(session, skip=0, limit=2)
        assert len(users) == 2

    async def test_update_user(self, session: AsyncSession):
        """Test updating a user."""
        user = await methods.create_user_async(
            session,
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        updated_user = await methods.update_user_async(session, user, username="newusername")

        assert updated_user.username == "newusername"

    async def test_delete_user(self, session: AsyncSession):
        """Test deleting a user."""
        user = await methods.create_user_async(
            session,
            username="testuser",
            email="test@example.com",
            password="password123",
        )
        await methods.delete_user_async(session, user)

        retrieved_user = await methods.get_user_async(session, user_uuid=user.uuid)
        assert retrieved_user is None

    async def test_authenticate_user_success(self, session: AsyncSession):
        """Test successful user authentication."""
        await methods.create_user_async(
            session,
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        user = await methods.authenticate_user_async(session, "testuser", "password123")
        assert user is not None
        assert user.username == "testuser"

    async def test_authenticate_user_wrong_password(self, session: AsyncSession):
        """Test authentication with wrong password."""
        await methods.create_user_async(
            session,
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        user = await methods.authenticate_user_async(session, "testuser", "wrongpassword")
        assert user is None

    async def test_authenticate_user_not_found(self, session: AsyncSession):
        """Test authentication with non-existent user."""
        user = await methods.authenticate_user_async(session, "nonexistent", "password123")
        assert user is None


class TestSSOAuthentication:
    """Test cases for SSO authentication functionality."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock cache for testing."""
        return Mock()

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config_session = Mock()
        config_session.get_value.side_effect = lambda key: {
            "EXPIRATION_HOURS": "12",
            "ALGORITHM": "HS256",
            "SECRET_KEY": "test-secret-key",
        }.get(key)

        config = Mock()
        config.get_session.return_value = config_session
        return config

    @pytest.fixture
    def authenticator(self, mock_cache, mock_config):
        """Create a UserAuthenticatorImpl instance with mocked dependencies."""
        with patch("fivccliche.modules.users.services.query_component") as mock_query:

            def query_side_effect(site, interface):
                from fivcglue.interfaces.caches import ICache
                from fivcglue.interfaces.configs import IConfig

                if interface == ICache:
                    return mock_cache
                elif interface == IConfig:
                    return mock_config
                return None

            mock_query.side_effect = query_side_effect
            authenticator = UserAuthenticatorImpl(Mock())
            return authenticator

    async def test_create_sso_credential_new_user(
        self, authenticator, mock_cache, session: AsyncSession
    ):
        """Test creating SSO credential for a new user."""
        # Create SSO credential for a user that doesn't exist yet
        credential = await authenticator.create_sso_credential_async(
            username="ssouser",
            attributes={"email": "sso@example.com"},
            session=session,
        )

        # Verify credential was created
        assert credential is not None
        assert credential.access_token is not None
        assert credential.expires_in > 0

        # Verify user was created in database
        user = await methods.get_user_async(session, username="ssouser")
        assert user is not None
        assert user.username == "ssouser"
        assert user.email == "sso@example.com"
        assert user.hashed_password is None  # SSO users don't have passwords

    async def test_create_sso_credential_existing_user(
        self, authenticator, mock_cache, session: AsyncSession
    ):
        """Test creating SSO credential for an existing user."""
        # Create a user first
        existing_user = await methods.create_user_async(
            session,
            username="existinguser",
            email="existing@example.com",
            password=None,
        )

        # Create SSO credential for existing user
        credential = await authenticator.create_sso_credential_async(
            username="existinguser",
            attributes={"email": "different@example.com"},  # Different email in attributes
            session=session,
        )

        # Verify credential was created
        assert credential is not None
        assert credential.access_token is not None
        assert credential.expires_in > 0

        # Verify user still exists with original email (not updated)
        user = await methods.get_user_async(session, username="existinguser")
        assert user is not None
        assert user.uuid == existing_user.uuid
        assert user.email == "existing@example.com"  # Email should not change

    async def test_create_sso_credential_without_email(
        self, authenticator, mock_cache, session: AsyncSession
    ):
        """Test creating SSO credential without email in attributes."""
        # Create SSO credential without email
        credential = await authenticator.create_sso_credential_async(
            username="noemailuser",
            attributes={},
            session=session,
        )

        # Verify credential was created
        assert credential is not None
        assert credential.access_token is not None

        # Verify user was created without email
        user = await methods.get_user_async(session, username="noemailuser")
        assert user is not None
        assert user.username == "noemailuser"
        assert user.email is None
        assert user.hashed_password is None  # SSO users don't have passwords

    async def test_create_sso_credential_without_session(
        self, authenticator, mock_cache, session: AsyncSession
    ):
        """Test creating SSO credential without providing session."""
        # Mock get_db_session_async to return our test session
        with patch("fivccliche.modules.users.services.get_db_session_async") as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = session
            mock_get_session.return_value.__aexit__.return_value = None

            # Create SSO credential without session parameter
            credential = await authenticator.create_sso_credential_async(
                username="sessionlessuser",
                attributes={"email": "sessionless@example.com"},
                session=None,
            )

            # Verify credential was created
            assert credential is not None
            assert credential.access_token is not None

            # Verify user was created
            user = await methods.get_user_async(session, username="sessionlessuser")
            assert user is not None


class TestUserAuthenticatorCaching:
    """Test cases for UserAuthenticatorImpl caching functionality."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock cache for testing."""
        return Mock()

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config_session = Mock()
        config_session.get_value.side_effect = lambda key: {
            "EXPIRATION_HOURS": "12",
            "ALGORITHM": "HS256",
            "SECRET_KEY": "test-secret-key",
        }.get(key)

        config = Mock()
        config.get_session.return_value = config_session
        return config

    @pytest.fixture
    def mock_component_site(self, mock_cache, mock_config):
        """Create a mock component site for testing."""
        component_site = Mock()

        def query_component_side_effect(site, interface):
            from fivcglue.interfaces.caches import ICache
            from fivcglue.interfaces.configs import IConfig

            if interface == ICache:
                return mock_cache
            elif interface == IConfig:
                return mock_config
            return None

        with patch(
            "fivccliche.modules.users.services.query_component",
            side_effect=query_component_side_effect,
        ):
            yield component_site

    @pytest.fixture
    def authenticator(self, mock_cache, mock_config):
        """Create a UserAuthenticatorImpl instance with mocked dependencies."""
        with patch("fivccliche.modules.users.services.query_component") as mock_query:

            def query_side_effect(site, interface):
                from fivcglue.interfaces.caches import ICache
                from fivcglue.interfaces.configs import IConfig

                if interface == ICache:
                    return mock_cache
                elif interface == IConfig:
                    return mock_config
                return None

            mock_query.side_effect = query_side_effect
            authenticator = UserAuthenticatorImpl(Mock())
            return authenticator

    async def test_cache_hit_verification(self, authenticator, mock_cache, session: AsyncSession):
        """Test that cache hits are verified and used correctly."""
        # Create a test user
        user = await methods.create_user_async(
            session,
            username="cacheuser",
            email="cache@example.com",
            password="password123",
        )

        # Create a token for the user
        credential = authenticator._create_access_token(user.uuid)
        token = credential.access_token

        # Setup cache to return user data on first call
        user_info = {
            "uuid": user.uuid,
            "username": user.username,
            "email": user.email,
            "is_active": user.is_active,
            "is_superuser": user.is_superuser,
        }
        mock_cache.get_value.return_value = json.dumps(user_info).encode("utf-8")

        # First verification - cache hit
        result = await authenticator.verify_credential_async(token, session=session)

        # Verify cache was checked
        mock_cache.get_value.assert_called()
        assert result is not None
        assert result.uuid == user.uuid

    async def test_cache_miss_database_query(
        self, authenticator, mock_cache, session: AsyncSession
    ):
        """Test that cache miss triggers database query."""
        # Create a test user
        user = await methods.create_user_async(
            session,
            username="dbuser",
            email="db@example.com",
            password="password123",
        )

        # Create a token for the user
        credential = authenticator._create_access_token(user.uuid)
        token = credential.access_token

        # Setup cache to return None (cache miss)
        mock_cache.get_value.return_value = None

        # Verify token - should query database
        result = await authenticator.verify_credential_async(token, session=session)

        # Verify result is correct
        assert result is not None
        assert result.uuid == user.uuid

        # Verify cache was set after database query
        mock_cache.set_value.assert_called()

    async def test_cache_set_after_database_query(
        self, authenticator, mock_cache, session: AsyncSession
    ):
        """Test that cache is set after successful database query."""
        # Create a test user
        user = await methods.create_user_async(
            session,
            username="setcacheuser",
            email="setcache@example.com",
            password="password123",
        )

        # Create a token for the user
        credential = authenticator._create_access_token(user.uuid)
        token = credential.access_token

        # Setup cache to return None (cache miss)
        mock_cache.get_value.return_value = None

        # Verify token
        result = await authenticator.verify_credential_async(token, session=session)
        assert result

        # Verify cache.set_value was called with correct parameters
        assert mock_cache.set_value.called
        call_args = mock_cache.set_value.call_args

        # Check cache key
        assert call_args[0][0] == f"user: {user.uuid}"

        # Check cache value contains user info
        cached_value = json.loads(call_args[0][1].decode("utf-8"))
        assert cached_value["uuid"] == user.uuid
        assert cached_value["username"] == user.username
        assert cached_value["email"] == user.email

    async def test_cache_expiration_set_correctly(
        self, authenticator, mock_cache, session: AsyncSession
    ):
        """Test that cache expiration is set to token expiration hours."""
        # Create a test user
        user = await methods.create_user_async(
            session,
            username="expireuser",
            email="expire@example.com",
            password="password123",
        )

        # Create a token for the user
        credential = authenticator._create_access_token(user.uuid)
        token = credential.access_token

        # Setup cache to return None
        mock_cache.get_value.return_value = None

        # Verify token
        await authenticator.verify_credential_async(token, session=session)

        # Verify cache.set_value was called with correct expiration
        call_args = mock_cache.set_value.call_args
        expire_param = call_args[1]["expire"]

        # Should be a timedelta with hours equal to token_expire_hours
        assert isinstance(expire_param, timedelta)
        assert expire_param.total_seconds() == 12 * 3600  # 12 hours

    async def test_invalid_token_not_cached(self, authenticator, mock_cache, session: AsyncSession):
        """Test that invalid tokens don't trigger cache operations."""
        invalid_token = "invalid.token.format"

        # Setup cache
        mock_cache.get_value.return_value = None

        # Verify invalid token
        result = await authenticator.verify_credential_async(invalid_token, session=session)

        # Should return None
        assert result is None

        # Cache should not be set for invalid tokens
        mock_cache.set_value.assert_not_called()

    async def test_expired_token_not_cached(self, authenticator, mock_cache, session: AsyncSession):
        """Test that expired tokens don't trigger cache operations."""
        from datetime import datetime, timezone, timedelta
        import jwt

        # Create an expired token
        time_now = datetime.now(timezone.utc)
        time_expire = time_now - timedelta(hours=1)  # Expired 1 hour ago

        expired_token = jwt.encode(
            {
                "sub": "test-user-id",
                "iat": time_now,
                "exp": time_expire,
            },
            authenticator.token_secret_key,
            algorithm=authenticator.token_algorithm,
        )

        # Setup cache
        mock_cache.get_value.return_value = None

        # Verify expired token
        result = await authenticator.verify_credential_async(expired_token, session=session)

        # Should return None
        assert result is None

        # Cache should not be set for expired tokens
        mock_cache.set_value.assert_not_called()

    async def test_token_isolation_different_users(
        self, authenticator, mock_cache, session: AsyncSession
    ):
        """Test that different users' tokens are cached separately."""
        # Create two test users
        user1 = await methods.create_user_async(
            session,
            username="user1cache",
            email="user1cache@example.com",
            password="password123",
        )

        user2 = await methods.create_user_async(
            session,
            username="user2cache",
            email="user2cache@example.com",
            password="password123",
        )

        # Create tokens for both users
        credential1 = authenticator._create_access_token(user1.uuid)
        token1 = credential1.access_token
        credential2 = authenticator._create_access_token(user2.uuid)
        token2 = credential2.access_token

        # Setup cache to return None initially
        mock_cache.get_value.return_value = None

        # Verify token1
        result1 = await authenticator.verify_credential_async(token1, session=session)
        assert result1.uuid == user1.uuid

        # Verify token2
        result2 = await authenticator.verify_credential_async(token2, session=session)
        assert result2.uuid == user2.uuid

        # Verify cache was called with different keys
        cache_calls = mock_cache.get_value.call_args_list
        assert len(cache_calls) >= 2
        assert cache_calls[0][0][0] == f"user: {user1.uuid}"
        assert cache_calls[1][0][0] == f"user: {user2.uuid}"

    async def test_user_impl_wrapping(self, authenticator, mock_cache, session: AsyncSession):
        """Test that verified users are wrapped in UserImpl."""
        # Create a test user
        user = await methods.create_user_async(
            session,
            username="impluser",
            email="impl@example.com",
            password="password123",
        )

        # Create a token for the user
        credential = authenticator._create_access_token(user.uuid)
        token = credential.access_token

        # Setup cache to return None
        mock_cache.get_value.return_value = None

        # Verify token
        result = await authenticator.verify_credential_async(token, session=session)

        # Verify result is UserImpl instance
        assert isinstance(result, UserImpl)
        assert result.uuid == user.uuid
        assert result.username == user.username
        assert result.email == user.email
        assert result.is_superuser == user.is_superuser

    async def test_cache_with_none_session(self, authenticator, mock_cache, session: AsyncSession):
        """Test that caching works when session is None (creates its own session)."""
        # Create a test user
        user = await methods.create_user_async(
            session,
            username="nosessionuser",
            email="nosession@example.com",
            password="password123",
        )

        # Create a token for the user
        credential = authenticator._create_access_token(user.uuid)
        token = credential.access_token

        # Setup cache to return None
        mock_cache.get_value.return_value = None

        # Verify token without providing session
        # This will use get_db_session_async() internally
        with patch("fivccliche.modules.users.services.get_db_session_async") as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = session
            mock_get_session.return_value.__aexit__.return_value = None

            result = await authenticator.verify_credential_async(token, session=None)

            # Verify result is correct
            assert result is not None
            assert result.uuid == user.uuid
