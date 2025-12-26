"""Unit tests for config service layer."""

import tempfile
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from fivcplayground.embeddings.types import EmbeddingConfig
from fivcplayground.models.types import ModelConfig
from fivcplayground.agents.types import AgentConfig
from fivcplayground.tools.types import ToolConfig

from fivccliche.modules.agent_configs import methods
from fivccliche.modules.agent_configs.services import (
    UserEmbeddingRepositoryImpl,
    UserLLMRepositoryImpl,
    UserAgentRepositoryImpl,
    UserToolRepositoryImpl,
)

# Import models to ensure they're registered with SQLModel
from fivccliche.modules.users.models import User  # noqa: F401
from fivccliche.modules.agent_configs.models import UserEmbedding, UserLLM, UserAgent


@pytest.fixture
async def session():
    """Create a temporary SQLite database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        database_url = f"sqlite+aiosqlite:///{db_path}"

        engine = create_async_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=NullPool,
            echo=False,
        )

        # Create only the tables we need for testing using raw SQL
        async with engine.begin() as conn:
            # Disable foreign key constraints for SQLite during table creation
            await conn.execute(text("PRAGMA foreign_keys=OFF"))

            # Create user table
            await conn.execute(
                text(
                    """
                CREATE TABLE "user" (
                    uuid VARCHAR NOT NULL,
                    username VARCHAR NOT NULL,
                    email VARCHAR NOT NULL,
                    hashed_password VARCHAR NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    PRIMARY KEY (uuid),
                    UNIQUE (username),
                    UNIQUE (email)
                )
            """
                )
            )

            # Create user_embedding table
            await conn.execute(
                text(
                    """
                CREATE TABLE user_embedding (
                    uuid VARCHAR NOT NULL,
                    id VARCHAR NOT NULL,
                    description VARCHAR,
                    provider VARCHAR NOT NULL DEFAULT 'openai',
                    model VARCHAR NOT NULL,
                    api_key VARCHAR NOT NULL,
                    base_url VARCHAR,
                    dimension INTEGER NOT NULL DEFAULT 1024,
                    user_uuid VARCHAR,
                    PRIMARY KEY (uuid),
                    UNIQUE (id, user_uuid),
                    FOREIGN KEY(user_uuid) REFERENCES "user" (uuid)
                )
            """
                )
            )

            # Create user_llm table
            await conn.execute(
                text(
                    """
                CREATE TABLE user_llm (
                    uuid VARCHAR NOT NULL,
                    id VARCHAR NOT NULL,
                    description VARCHAR,
                    provider VARCHAR NOT NULL DEFAULT 'openai',
                    model VARCHAR NOT NULL,
                    api_key VARCHAR NOT NULL,
                    base_url VARCHAR,
                    temperature FLOAT NOT NULL DEFAULT 0.5,
                    max_tokens INTEGER NOT NULL DEFAULT 4096,
                    user_uuid VARCHAR,
                    PRIMARY KEY (uuid),
                    UNIQUE (id, user_uuid),
                    FOREIGN KEY(user_uuid) REFERENCES "user" (uuid)
                )
            """
                )
            )

            # Create agent_models table (required for user_agent foreign key)
            await conn.execute(
                text(
                    """
                CREATE TABLE agent_models (
                    id VARCHAR NOT NULL,
                    PRIMARY KEY (id)
                )
            """
                )
            )

            # Create user_agent table
            await conn.execute(
                text(
                    """
                CREATE TABLE user_agent (
                    uuid VARCHAR NOT NULL,
                    id VARCHAR NOT NULL,
                    description VARCHAR,
                    model_id VARCHAR NOT NULL,
                    tools_ids JSON,
                    system_prompt VARCHAR,
                    user_uuid VARCHAR,
                    PRIMARY KEY (uuid),
                    UNIQUE (id, user_uuid),
                    FOREIGN KEY(user_uuid) REFERENCES "user" (uuid),
                    FOREIGN KEY(model_id) REFERENCES agent_models (id)
                )
            """
                )
            )

            # Create user_tool table
            await conn.execute(
                text(
                    """
                CREATE TABLE user_tool (
                    uuid VARCHAR NOT NULL,
                    id VARCHAR NOT NULL,
                    description VARCHAR,
                    transport VARCHAR NOT NULL,
                    command VARCHAR,
                    args JSON,
                    env JSON,
                    url VARCHAR,
                    user_uuid VARCHAR,
                    PRIMARY KEY (uuid),
                    UNIQUE (id, user_uuid),
                    FOREIGN KEY(user_uuid) REFERENCES "user" (uuid)
                )
            """
                )
            )

            await conn.execute(text("PRAGMA foreign_keys=ON"))

        # Create session
        async_session = AsyncSession(engine, expire_on_commit=False)
        try:
            yield async_session
        finally:
            await async_session.close()
            await engine.dispose()


class TestEmbeddingConfigService:
    """Test cases for embedding config service functions."""

    async def test_create_embedding_config(self, session: AsyncSession):
        """Test creating a new embedding config."""
        config_create = EmbeddingConfig(
            id="embedding-1",
            description="Test embedding",
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            base_url="https://api.openai.com",
            dimension=1536,
        )
        config = await methods.create_embedding_config_async(session, "user123", config_create)

        assert config.id is not None
        assert config.user_uuid == "user123"
        assert config.model == "text-embedding-3-small"
        assert config.dimension == 1536

    async def test_get_embedding_config(self, session: AsyncSession):
        """Test getting an embedding config by ID."""
        config_create = EmbeddingConfig(
            id="embedding-2",
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
        )
        created = await methods.create_embedding_config_async(session, "user123", config_create)
        retrieved = await methods.get_embedding_config_async(
            session, "user123", config_uuid=created.uuid
        )

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.user_uuid == "user123"

    async def test_get_embedding_config_wrong_user(self, session: AsyncSession):
        """Test getting embedding config with wrong user ID returns None for user-specific configs."""
        config_create = EmbeddingConfig(
            id="embedding-wrong-user",
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
        )
        created = await methods.create_embedding_config_async(session, "user123", config_create)
        # User456 should not be able to access user123's config
        retrieved = await methods.get_embedding_config_async(session, created.uuid, "user456")

        assert retrieved is None

    async def test_get_embedding_config_global_accessible(self, session: AsyncSession):
        """Test that global embedding configs (user_uuid=None) are accessible to all users."""
        # Create a global config by directly inserting into the database
        import uuid as uuid_lib

        config_uuid = str(uuid_lib.uuid4())
        global_config = UserEmbedding(
            uuid=config_uuid,
            id="global-embedding",
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            user_uuid=None,  # Global config
        )
        session.add(global_config)
        await session.commit()

        # Any user should be able to access the global config
        retrieved_user1 = await methods.get_embedding_config_async(
            session, "user123", config_uuid=config_uuid
        )
        retrieved_user2 = await methods.get_embedding_config_async(
            session, "user456", config_uuid=config_uuid
        )

        assert retrieved_user1 is not None, "User1 should be able to access global config"
        assert retrieved_user2 is not None, "User2 should be able to access global config"
        assert retrieved_user1.user_uuid is None
        assert retrieved_user2.user_uuid is None

    async def test_list_embedding_configs(self, session: AsyncSession):
        """Test listing embedding configs for a user."""
        for i in range(3):
            config_create = EmbeddingConfig(
                id=f"embedding-list-{i}",
                provider="openai",
                model=f"model-{i}",
                api_key=f"key-{i}",
            )
            await methods.create_embedding_config_async(session, "user123", config_create)

        configs = await methods.list_embedding_configs_async(session, "user123")
        assert len(configs) == 3

    async def test_list_embedding_configs_pagination(self, session: AsyncSession):
        """Test listing embedding configs with pagination."""
        for i in range(5):
            config_create = EmbeddingConfig(
                id=f"embedding-page-{i}",
                provider="openai",
                model=f"model-{i}",
                api_key=f"key-{i}",
            )
            await methods.create_embedding_config_async(session, "user123", config_create)

        configs = await methods.list_embedding_configs_async(session, "user123", skip=0, limit=2)
        assert len(configs) == 2

    async def test_update_embedding_config(self, session: AsyncSession):
        """Test updating an embedding config."""
        config_create = EmbeddingConfig(
            id="embedding-3",
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            dimension=1536,
        )
        config = await methods.create_embedding_config_async(session, "user123", config_create)

        config_update = EmbeddingConfig(
            id="embedding-3",
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            dimension=3072,
        )
        updated = await methods.update_embedding_config_async(session, config, config_update)

        assert updated.dimension == 3072
        assert updated.model == "text-embedding-3-small"

    async def test_delete_embedding_config(self, session: AsyncSession):
        """Test deleting an embedding config."""
        config_create = EmbeddingConfig(
            id="embedding-4",
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
        )
        config = await methods.create_embedding_config_async(session, "user123", config_create)
        await methods.delete_embedding_config_async(session, config)

        retrieved = await methods.get_embedding_config_async(session, config.uuid, "user123")
        assert retrieved is None

    async def test_count_embedding_configs(self, session: AsyncSession):
        """Test counting embedding configs for a user."""
        for i in range(3):
            config_create = EmbeddingConfig(
                id=f"embedding-count-{i}",
                provider="openai",
                model=f"model-{i}",
                api_key=f"key-{i}",
            )
            await methods.create_embedding_config_async(session, "user123", config_create)

        count = await methods.count_embedding_configs_async(session, "user123")
        assert count == 3

    async def test_list_embedding_configs_includes_global(self, session: AsyncSession):
        """Test that listing embedding configs includes both user-specific and global configs."""
        import uuid as uuid_lib

        # Create user-specific configs
        for i in range(2):
            config_create = EmbeddingConfig(
                id=f"embedding-list-global-{i}",
                provider="openai",
                model=f"model-{i}",
                api_key=f"key-{i}",
            )
            await methods.create_embedding_config_async(session, "user123", config_create)

        # Create a global config
        global_config = UserEmbedding(
            uuid=str(uuid_lib.uuid4()),
            id="global-embedding-list",
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            user_uuid=None,  # Global config
        )
        session.add(global_config)
        await session.commit()

        # List should include both user-specific and global configs
        configs = await methods.list_embedding_configs_async(session, "user123")
        assert len(configs) == 3  # 2 user-specific + 1 global

        # Verify we have both types
        user_specific = [c for c in configs if c.user_uuid == "user123"]
        global_configs = [c for c in configs if c.user_uuid is None]
        assert len(user_specific) == 2
        assert len(global_configs) == 1

    async def test_count_embedding_configs_includes_global(self, session: AsyncSession):
        """Test that counting embedding configs includes both user-specific and global configs."""
        import uuid as uuid_lib

        # Create user-specific configs
        for i in range(2):
            config_create = EmbeddingConfig(
                id=f"embedding-count-global-{i}",
                provider="openai",
                model=f"model-{i}",
                api_key=f"key-{i}",
            )
            await methods.create_embedding_config_async(session, "user456", config_create)

        # Create a global config
        global_config = UserEmbedding(
            uuid=str(uuid_lib.uuid4()),
            id="global-embedding-count",
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            user_uuid=None,  # Global config
        )
        session.add(global_config)
        await session.commit()

        # Count should include both user-specific and global configs
        count = await methods.count_embedding_configs_async(session, "user456")
        assert count == 3  # 2 user-specific + 1 global


class TestLLMConfigService:
    """Test cases for LLM config service functions."""

    async def test_create_llm_config(self, session: AsyncSession):
        """Test creating a new LLM config."""
        config_create = ModelConfig(
            id="llm-1",
            description="Test LLM",
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            temperature=0.7,
            max_tokens=2048,
        )
        config = await methods.create_llm_config_async(session, "user123", config_create)

        assert config.id is not None
        assert config.user_uuid == "user123"
        assert config.model == "gpt-4"
        assert config.temperature == 0.7

    async def test_get_llm_config(self, session: AsyncSession):
        """Test getting an LLM config by ID."""
        config_create = ModelConfig(
            id="llm-2",
            provider="openai",
            model="gpt-4",
            api_key="test-key",
        )
        created = await methods.create_llm_config_async(session, "user123", config_create)
        retrieved = await methods.get_llm_config_async(session, "user123", config_uuid=created.uuid)

        assert retrieved is not None
        assert retrieved.id == created.id

    async def test_list_llm_configs(self, session: AsyncSession):
        """Test listing LLM configs for a user."""
        for i in range(3):
            config_create = ModelConfig(
                id=f"llm-list-{i}",
                provider="openai",
                model=f"model-{i}",
                api_key=f"key-{i}",
            )
            await methods.create_llm_config_async(session, "user123", config_create)

        configs = await methods.list_llm_configs_async(session, "user123")
        assert len(configs) == 3

    async def test_update_llm_config(self, session: AsyncSession):
        """Test updating an LLM config."""
        config_create = ModelConfig(
            id="llm-3",
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            temperature=0.5,
        )
        config = await methods.create_llm_config_async(session, "user123", config_create)

        config_update = ModelConfig(
            id="llm-3",
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            temperature=0.9,
        )
        updated = await methods.update_llm_config_async(session, config, config_update)

        assert updated.temperature == 0.9
        assert updated.model == "gpt-4"

    async def test_delete_llm_config(self, session: AsyncSession):
        """Test deleting an LLM config."""
        config_create = ModelConfig(
            id="llm-4",
            provider="openai",
            model="gpt-4",
            api_key="test-key",
        )
        config = await methods.create_llm_config_async(session, "user123", config_create)
        await methods.delete_llm_config_async(session, config)

        retrieved = await methods.get_llm_config_async(session, config.uuid, "user123")
        assert retrieved is None

    async def test_count_llm_configs(self, session: AsyncSession):
        """Test counting LLM configs for a user."""
        for i in range(3):
            config_create = ModelConfig(
                id=f"llm-count-{i}",
                provider="openai",
                model=f"model-{i}",
                api_key=f"key-{i}",
            )
            await methods.create_llm_config_async(session, "user123", config_create)

        count = await methods.count_llm_configs_async(session, "user123")
        assert count == 3

    async def test_get_llm_config_global_accessible(self, session: AsyncSession):
        """Test that global LLM configs are accessible to all users."""
        import uuid as uuid_lib

        # Create a global LLM config
        config_uuid = str(uuid_lib.uuid4())
        global_config = UserLLM(
            uuid=config_uuid,
            id="global-llm",
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            user_uuid=None,  # Global config
        )
        session.add(global_config)
        await session.commit()

        # Any user should be able to access the global config
        retrieved_user1 = await methods.get_llm_config_async(
            session, "user123", config_uuid=config_uuid
        )
        retrieved_user2 = await methods.get_llm_config_async(
            session, "user456", config_uuid=config_uuid
        )

        assert retrieved_user1 is not None
        assert retrieved_user1.user_uuid is None
        assert retrieved_user1.id == "global-llm"

        assert retrieved_user2 is not None
        assert retrieved_user2.user_uuid is None
        assert retrieved_user2.id == "global-llm"

    async def test_list_llm_configs_includes_global(self, session: AsyncSession):
        """Test that listing LLM configs includes both user-specific and global configs."""
        import uuid as uuid_lib

        # Create user-specific configs
        for i in range(2):
            config_create = ModelConfig(
                id=f"llm-list-global-{i}",
                provider="openai",
                model=f"model-{i}",
                api_key=f"key-{i}",
            )
            await methods.create_llm_config_async(session, "user123", config_create)

        # Create a global config
        global_config = UserLLM(
            uuid=str(uuid_lib.uuid4()),
            id="global-llm-list",
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            user_uuid=None,  # Global config
        )
        session.add(global_config)
        await session.commit()

        # List should include both user-specific and global configs
        configs = await methods.list_llm_configs_async(session, "user123")
        assert len(configs) == 3  # 2 user-specific + 1 global

        # Verify we have both types
        user_specific = [c for c in configs if c.user_uuid == "user123"]
        global_configs = [c for c in configs if c.user_uuid is None]
        assert len(user_specific) == 2
        assert len(global_configs) == 1

    async def test_count_llm_configs_includes_global(self, session: AsyncSession):
        """Test that counting LLM configs includes both user-specific and global configs."""
        import uuid as uuid_lib

        # Create user-specific configs
        for i in range(2):
            config_create = ModelConfig(
                id=f"llm-count-global-{i}",
                provider="openai",
                model=f"model-{i}",
                api_key=f"key-{i}",
            )
            await methods.create_llm_config_async(session, "user456", config_create)

        # Create a global config
        global_config = UserLLM(
            uuid=str(uuid_lib.uuid4()),
            id="global-llm-count",
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            user_uuid=None,  # Global config
        )
        session.add(global_config)
        await session.commit()

        # Count should include both user-specific and global configs
        count = await methods.count_llm_configs_async(session, "user456")
        assert count == 3  # 2 user-specific + 1 global


class TestAgentConfigService:
    """Test cases for agent config service functions."""

    async def test_create_agent_config(self, session: AsyncSession):
        """Test creating a new agent config."""
        config_create = AgentConfig(
            id="agent-1",
            description="Test agent",
            model_id="model123",
            system_prompt="You are a helpful assistant",
        )
        config = await methods.create_agent_config_async(session, "user123", config_create)

        assert config.id is not None
        assert config.user_uuid == "user123"
        assert config.model_id == "model123"

    async def test_get_agent_config(self, session: AsyncSession):
        """Test getting an agent config by ID."""
        config_create = AgentConfig(
            id="agent-2",
            model_id="model123",
        )
        created = await methods.create_agent_config_async(session, "user123", config_create)
        retrieved = await methods.get_agent_config_async(
            session, "user123", config_uuid=created.uuid
        )

        assert retrieved is not None
        assert retrieved.id == created.id

    async def test_list_agent_configs(self, session: AsyncSession):
        """Test listing agent configs for a user."""
        for i in range(3):
            config_create = AgentConfig(
                id=f"agent-list-{i}",
                model_id=f"model-{i}",
            )
            await methods.create_agent_config_async(session, "user123", config_create)

        configs = await methods.list_agent_configs_async(session, "user123")
        assert len(configs) == 3

    async def test_update_agent_config(self, session: AsyncSession):
        """Test updating an agent config."""
        config_create = AgentConfig(
            id="agent-3",
            model_id="model123",
            system_prompt="Old prompt",
        )
        config = await methods.create_agent_config_async(session, "user123", config_create)

        config_update = AgentConfig(
            id="agent-3",
            model_id="model123",
            system_prompt="New prompt",
        )
        updated = await methods.update_agent_config_async(session, config, config_update)

        assert updated.system_prompt == "New prompt"
        assert updated.model_id == "model123"

    async def test_delete_agent_config(self, session: AsyncSession):
        """Test deleting an agent config."""
        config_create = AgentConfig(
            id="agent-4",
            model_id="model123",
        )
        config = await methods.create_agent_config_async(session, "user123", config_create)
        await methods.delete_agent_config_async(session, config)

        retrieved = await methods.get_agent_config_async(session, config.uuid, "user123")
        assert retrieved is None

    async def test_count_agent_configs(self, session: AsyncSession):
        """Test counting agent configs for a user."""
        for i in range(3):
            config_create = AgentConfig(
                id=f"agent-count-{i}",
                model_id=f"model-{i}",
            )
            await methods.create_agent_config_async(session, "user123", config_create)

        count = await methods.count_agent_configs_async(session, "user123")
        assert count == 3

    async def test_get_agent_config_global_accessible(self, session: AsyncSession):
        """Test that global agent configs are accessible to all users."""
        import uuid as uuid_lib

        # Create a global agent config
        config_uuid = str(uuid_lib.uuid4())
        global_config = UserAgent(
            uuid=config_uuid,
            id="global-agent",
            model_id="model123",
            system_prompt="You are a helpful assistant",
            user_uuid=None,  # Global config
        )
        session.add(global_config)
        await session.commit()

        # Any user should be able to access the global config
        retrieved_user1 = await methods.get_agent_config_async(
            session, "user123", config_uuid=config_uuid
        )
        retrieved_user2 = await methods.get_agent_config_async(
            session, "user456", config_uuid=config_uuid
        )

        assert retrieved_user1 is not None
        assert retrieved_user1.user_uuid is None
        assert retrieved_user1.id == "global-agent"

        assert retrieved_user2 is not None
        assert retrieved_user2.user_uuid is None
        assert retrieved_user2.id == "global-agent"

    async def test_list_agent_configs_includes_global(self, session: AsyncSession):
        """Test that listing agent configs includes both user-specific and global configs."""
        import uuid as uuid_lib

        # Create user-specific configs
        for i in range(2):
            config_create = AgentConfig(
                id=f"agent-list-global-{i}",
                model_id=f"model-{i}",
            )
            await methods.create_agent_config_async(session, "user123", config_create)

        # Create a global config
        global_config = UserAgent(
            uuid=str(uuid_lib.uuid4()),
            id="global-agent-list",
            model_id="model123",
            system_prompt="You are a helpful assistant",
            user_uuid=None,  # Global config
        )
        session.add(global_config)
        await session.commit()

        # List should include both user-specific and global configs
        configs = await methods.list_agent_configs_async(session, "user123")
        assert len(configs) == 3  # 2 user-specific + 1 global

        # Verify we have both types
        user_specific = [c for c in configs if c.user_uuid == "user123"]
        global_configs = [c for c in configs if c.user_uuid is None]
        assert len(user_specific) == 2
        assert len(global_configs) == 1

    async def test_count_agent_configs_includes_global(self, session: AsyncSession):
        """Test that counting agent configs includes both user-specific and global configs."""
        import uuid as uuid_lib

        # Create user-specific configs
        for i in range(2):
            config_create = AgentConfig(
                id=f"agent-count-global-{i}",
                model_id=f"model-{i}",
            )
            await methods.create_agent_config_async(session, "user456", config_create)

        # Create a global config
        global_config = UserAgent(
            uuid=str(uuid_lib.uuid4()),
            id="global-agent-count",
            model_id="model123",
            system_prompt="You are a helpful assistant",
            user_uuid=None,  # Global config
        )
        session.add(global_config)
        await session.commit()

        # Count should include both user-specific and global configs
        count = await methods.count_agent_configs_async(session, "user456")
        assert count == 3  # 2 user-specific + 1 global


class TestEmbeddingRepositoryImpl:
    """Test cases for UserEmbeddingRepositoryImpl abstract methods."""

    async def test_update_embedding_config_create_new(self, session: AsyncSession):
        """Test creating a new embedding config via update_embedding_config."""
        repo = UserEmbeddingRepositoryImpl(user_uuid="user123", session=session)

        config = EmbeddingConfig(
            id="repo-embedding-1",
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            dimension=1536,
        )

        await repo.update_embedding_config_async(config)

        # Verify it was created
        retrieved = await methods.get_embedding_config_async(
            session, "user123", config_id="repo-embedding-1"
        )
        assert retrieved is not None
        assert retrieved.dimension == 1536

    async def test_update_embedding_config_update_existing(self, session: AsyncSession):
        """Test updating an existing embedding config via update_embedding_config."""
        repo = UserEmbeddingRepositoryImpl(user_uuid="user123", session=session)

        # Create initial config
        config_create = EmbeddingConfig(
            id="repo-embedding-2",
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            dimension=1536,
        )
        await methods.create_embedding_config_async(session, "user123", config_create)

        # Update via repository
        config_update = EmbeddingConfig(
            id="repo-embedding-2",
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            dimension=3072,
        )
        await repo.update_embedding_config_async(config_update)

        # Verify it was updated
        retrieved = await methods.get_embedding_config_async(
            session, "user123", config_id="repo-embedding-2"
        )
        assert retrieved.dimension == 3072

    async def test_get_embedding_config_returns_config_type(self, session: AsyncSession):
        """Test that get_embedding_config returns EmbeddingConfig type."""
        repo = UserEmbeddingRepositoryImpl(user_uuid="user123", session=session)

        config_create = EmbeddingConfig(
            id="repo-embedding-3",
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
        )
        await methods.create_embedding_config_async(session, "user123", config_create)

        result = await repo.get_embedding_config_async("repo-embedding-3")

        assert result is not None
        assert isinstance(result, EmbeddingConfig)
        assert result.id == "repo-embedding-3"
        assert result.provider == "openai"

    async def test_get_embedding_config_returns_none_when_not_found(self, session: AsyncSession):
        """Test that get_embedding_config returns None when config not found."""
        repo = UserEmbeddingRepositoryImpl(user_uuid="user123", session=session)

        result = await repo.get_embedding_config_async("nonexistent")

        assert result is None

    async def test_list_embedding_configs_returns_config_types(self, session: AsyncSession):
        """Test that list_embedding_configs returns list of EmbeddingConfig types."""
        repo = UserEmbeddingRepositoryImpl(user_uuid="user123", session=session)

        for i in range(3):
            config = EmbeddingConfig(
                id=f"repo-embedding-list-{i}",
                provider="openai",
                model=f"model-{i}",
                api_key=f"key-{i}",
            )
            await methods.create_embedding_config_async(session, "user123", config)

        results = await repo.list_embedding_configs_async()

        assert len(results) == 3
        assert all(isinstance(c, EmbeddingConfig) for c in results)

    async def test_delete_embedding_config(self, session: AsyncSession):
        """Test deleting an embedding config."""
        repo = UserEmbeddingRepositoryImpl(user_uuid="user123", session=session)

        config = EmbeddingConfig(
            id="repo-embedding-delete",
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
        )
        await methods.create_embedding_config_async(session, "user123", config)

        await repo.delete_embedding_config_async("repo-embedding-delete")

        # Verify it was deleted
        retrieved = await methods.get_embedding_config_async(
            session, "repo-embedding-delete", "user123"
        )
        assert retrieved is None

    async def test_embedding_repo_raises_error_without_session(self, session: AsyncSession):
        """Test that repository raises error when session is None."""
        repo = UserEmbeddingRepositoryImpl(user_uuid="user123", session=None)

        config = EmbeddingConfig(
            id="test",
            provider="openai",
            model="test",
            api_key="test",
        )

        with pytest.raises(ValueError, match="Session and user_uuid are required"):
            await repo.update_embedding_config_async(config)

    async def test_embedding_repo_raises_error_without_user_id(self, session: AsyncSession):
        """Test that repository raises error when user_uuid is None."""
        repo = UserEmbeddingRepositoryImpl(user_uuid=None, session=session)

        config = EmbeddingConfig(
            id="test",
            provider="openai",
            model="test",
            api_key="test",
        )

        with pytest.raises(ValueError, match="Session and user_uuid are required"):
            await repo.update_embedding_config_async(config)


class TestLLMRepositoryImpl:
    """Test cases for UserLLMRepositoryImpl abstract methods."""

    async def test_update_model_config_create_new(self, session: AsyncSession):
        """Test creating a new model config via update_model_config."""
        repo = UserLLMRepositoryImpl(user_uuid="user123", session=session)

        config = ModelConfig(
            id="repo-llm-1",
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            temperature=0.7,
            max_tokens=2048,
        )

        await repo.update_model_config_async(config)

        # Verify it was created
        retrieved = await methods.get_llm_config_async(session, "user123", config_id="repo-llm-1")
        assert retrieved is not None
        assert retrieved.temperature == 0.7

    async def test_update_model_config_update_existing(self, session: AsyncSession):
        """Test updating an existing model config via update_model_config."""
        repo = UserLLMRepositoryImpl(user_uuid="user123", session=session)

        # Create initial config
        config_create = ModelConfig(
            id="repo-llm-2",
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            temperature=0.5,
        )
        await methods.create_llm_config_async(session, "user123", config_create)

        # Update via repository
        config_update = ModelConfig(
            id="repo-llm-2",
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            temperature=0.9,
        )
        await repo.update_model_config_async(config_update)

        # Verify it was updated
        retrieved = await methods.get_llm_config_async(session, "user123", config_id="repo-llm-2")
        assert retrieved.temperature == 0.9

    async def test_get_model_config_returns_config_type(self, session: AsyncSession):
        """Test that get_model_config returns ModelConfig type."""
        repo = UserLLMRepositoryImpl(user_uuid="user123", session=session)

        config_create = ModelConfig(
            id="repo-llm-3",
            provider="openai",
            model="gpt-4",
            api_key="test-key",
        )
        await methods.create_llm_config_async(session, "user123", config_create)

        result = await repo.get_model_config_async("repo-llm-3")

        assert result is not None
        assert isinstance(result, ModelConfig)
        assert result.id == "repo-llm-3"
        assert result.provider == "openai"

    async def test_get_model_config_returns_none_when_not_found(self, session: AsyncSession):
        """Test that get_model_config returns None when config not found."""
        repo = UserLLMRepositoryImpl(user_uuid="user123", session=session)

        result = await repo.get_model_config_async("nonexistent")

        assert result is None

    async def test_list_model_configs_returns_config_types(self, session: AsyncSession):
        """Test that list_model_configs returns list of ModelConfig types."""
        repo = UserLLMRepositoryImpl(user_uuid="user123", session=session)

        for i in range(3):
            config = ModelConfig(
                id=f"repo-llm-list-{i}",
                provider="openai",
                model=f"model-{i}",
                api_key=f"key-{i}",
            )
            await methods.create_llm_config_async(session, "user123", config)

        results = await repo.list_model_configs_async()

        assert len(results) == 3
        assert all(isinstance(c, ModelConfig) for c in results)

    async def test_delete_model_config(self, session: AsyncSession):
        """Test deleting a model config."""
        repo = UserLLMRepositoryImpl(user_uuid="user123", session=session)

        config = ModelConfig(
            id="repo-llm-delete",
            provider="openai",
            model="gpt-4",
            api_key="test-key",
        )
        await methods.create_llm_config_async(session, "user123", config)

        await repo.delete_model_config_async("repo-llm-delete")

        # Verify it was deleted
        retrieved = await methods.get_llm_config_async(session, "repo-llm-delete", "user123")
        assert retrieved is None

    async def test_llm_repo_raises_error_without_session(self, session: AsyncSession):
        """Test that repository raises error when session is None."""
        repo = UserLLMRepositoryImpl(user_uuid="user123", session=None)

        config = ModelConfig(
            id="test",
            provider="openai",
            model="test",
            api_key="test",
        )

        with pytest.raises(ValueError, match="Session and user_uuid are required"):
            await repo.update_model_config_async(config)

    async def test_llm_repo_raises_error_without_user_id(self, session: AsyncSession):
        """Test that repository raises error when user_uuid is None."""
        repo = UserLLMRepositoryImpl(user_uuid=None, session=session)

        config = ModelConfig(
            id="test",
            provider="openai",
            model="test",
            api_key="test",
        )

        with pytest.raises(ValueError, match="Session and user_uuid are required"):
            await repo.update_model_config_async(config)


class TestAgentRepositoryImpl:
    """Test cases for UserAgentRepositoryImpl abstract methods."""

    async def test_update_agent_config_create_new(self, session: AsyncSession):
        """Test creating a new agent config via update_agent_config."""
        repo = UserAgentRepositoryImpl(user_uuid="user123", session=session)

        config = AgentConfig(
            id="repo-agent-1",
            model_id="model123",
            description="Test agent",
            system_prompt="You are helpful",
        )

        await repo.update_agent_config_async(config)

        # Verify it was created
        retrieved = await methods.get_agent_config_async(
            session, "user123", config_id="repo-agent-1"
        )
        assert retrieved is not None
        assert retrieved.system_prompt == "You are helpful"

    async def test_update_agent_config_update_existing(self, session: AsyncSession):
        """Test updating an existing agent config via update_agent_config."""
        repo = UserAgentRepositoryImpl(user_uuid="user123", session=session)

        # Create initial config
        config_create = AgentConfig(
            id="repo-agent-2",
            model_id="model123",
            system_prompt="Old prompt",
        )
        await methods.create_agent_config_async(session, "user123", config_create)

        # Update via repository
        config_update = AgentConfig(
            id="repo-agent-2",
            model_id="model123",
            system_prompt="New prompt",
        )
        await repo.update_agent_config_async(config_update)

        # Verify it was updated
        retrieved = await methods.get_agent_config_async(
            session, "user123", config_id="repo-agent-2"
        )
        assert retrieved.system_prompt == "New prompt"

    async def test_get_agent_config_returns_config_type(self, session: AsyncSession):
        """Test that get_agent_config returns AgentConfig type."""
        repo = UserAgentRepositoryImpl(user_uuid="user123", session=session)

        config_create = AgentConfig(
            id="repo-agent-3",
            model_id="model123",
        )
        await methods.create_agent_config_async(session, "user123", config_create)

        result = await repo.get_agent_config_async("repo-agent-3")

        assert result is not None
        assert isinstance(result, AgentConfig)
        assert result.id == "repo-agent-3"
        assert result.model_id == "model123"

    async def test_get_agent_config_returns_none_when_not_found(self, session: AsyncSession):
        """Test that get_agent_config returns None when config not found."""
        repo = UserAgentRepositoryImpl(user_uuid="user123", session=session)

        result = await repo.get_agent_config_async("nonexistent")

        assert result is None

    async def test_list_agent_configs_returns_config_types(self, session: AsyncSession):
        """Test that list_agent_configs returns list of AgentConfig types."""
        repo = UserAgentRepositoryImpl(user_uuid="user123", session=session)

        for i in range(3):
            config = AgentConfig(
                id=f"repo-agent-list-{i}",
                model_id=f"model-{i}",
            )
            await methods.create_agent_config_async(session, "user123", config)

        results = await repo.list_agent_configs_async()

        assert len(results) == 3
        assert all(isinstance(c, AgentConfig) for c in results)

    async def test_delete_agent_config(self, session: AsyncSession):
        """Test deleting an agent config."""
        repo = UserAgentRepositoryImpl(user_uuid="user123", session=session)

        config = AgentConfig(
            id="repo-agent-delete",
            model_id="model123",
        )
        await methods.create_agent_config_async(session, "user123", config)

        await repo.delete_agent_config_async("repo-agent-delete")

        # Verify it was deleted
        retrieved = await methods.get_agent_config_async(session, "repo-agent-delete", "user123")
        assert retrieved is None

    async def test_agent_repo_raises_error_without_session(self, session: AsyncSession):
        """Test that repository raises error when session is None."""
        repo = UserAgentRepositoryImpl(user_uuid="user123", session=None)

        config = AgentConfig(
            id="test",
            model_id="test",
        )

        with pytest.raises(ValueError, match="Session and user_uuid are required"):
            await repo.update_agent_config_async(config)

    async def test_agent_repo_raises_error_without_user_id(self, session: AsyncSession):
        """Test that repository raises error when user_uuid is None."""
        repo = UserAgentRepositoryImpl(user_uuid=None, session=session)

        config = AgentConfig(
            id="test",
            model_id="test",
        )

        with pytest.raises(ValueError, match="Session and user_uuid are required"):
            await repo.update_agent_config_async(config)


# ============================================================================
# Tool Config Tests
# ============================================================================


class TestToolConfigMethods:
    """Test cases for tool config methods."""

    async def test_create_tool_config(self, session: AsyncSession):
        """Test creating a tool config."""
        config_create = ToolConfig(
            id="test-tool",
            description="Test tool",
            transport="stdio",
            command="python",
            args=["script.py"],
        )

        config = await methods.create_tool_config_async(session, "user123", config_create)

        assert config.id == "test-tool"
        assert config.user_uuid == "user123"
        assert config.transport == "stdio"
        assert config.command == "python"
        assert config.args == ["script.py"]
        assert config.uuid is not None

    async def test_get_tool_config_by_uuid(self, session: AsyncSession):
        """Test getting a tool config by UUID."""
        config_create = ToolConfig(
            id="test-tool-uuid",
            description="Test tool UUID",
            transport="sse",
        )
        created = await methods.create_tool_config_async(session, "user123", config_create)

        retrieved = await methods.get_tool_config_async(
            session, "user123", config_uuid=created.uuid
        )

        assert retrieved is not None
        assert retrieved.id == "test-tool-uuid"
        assert retrieved.uuid == created.uuid

    async def test_get_tool_config_by_id(self, session: AsyncSession):
        """Test getting a tool config by ID."""
        config_create = ToolConfig(
            id="test-tool-id",
            description="Test tool ID",
            transport="streamable_http",
            url="http://localhost:8000",
        )
        await methods.create_tool_config_async(session, "user123", config_create)

        retrieved = await methods.get_tool_config_async(
            session, "user123", config_id="test-tool-id"
        )

        assert retrieved is not None
        assert retrieved.id == "test-tool-id"
        assert retrieved.transport == "streamable_http"
        assert retrieved.url == "http://localhost:8000"

    async def test_get_tool_config_not_found(self, session: AsyncSession):
        """Test getting a non-existent tool config."""
        result = await methods.get_tool_config_async(session, "user123", config_id="nonexistent")

        assert result is None

    async def test_list_tool_configs(self, session: AsyncSession):
        """Test listing tool configs."""
        for i in range(3):
            config = ToolConfig(
                id=f"tool-{i}",
                description=f"Tool {i}",
                transport="stdio",
            )
            await methods.create_tool_config_async(session, "user123", config)

        configs = await methods.list_tool_configs_async(session, "user123")

        assert len(configs) == 3
        assert all(c.user_uuid == "user123" for c in configs)

    async def test_list_tool_configs_with_pagination(self, session: AsyncSession):
        """Test listing tool configs with pagination."""
        for i in range(5):
            config = ToolConfig(
                id=f"tool-page-{i}",
                description=f"Tool page {i}",
                transport="stdio",
            )
            await methods.create_tool_config_async(session, "user123", config)

        configs = await methods.list_tool_configs_async(session, "user123", skip=0, limit=2)

        assert len(configs) == 2

        configs = await methods.list_tool_configs_async(session, "user123", skip=2, limit=2)

        assert len(configs) == 2

    async def test_count_tool_configs(self, session: AsyncSession):
        """Test counting tool configs."""
        for i in range(3):
            config = ToolConfig(
                id=f"tool-count-{i}",
                description=f"Tool count {i}",
                transport="stdio",
            )
            await methods.create_tool_config_async(session, "user123", config)

        count = await methods.count_tool_configs_async(session, "user123")

        assert count == 3

    async def test_update_tool_config(self, session: AsyncSession):
        """Test updating a tool config."""
        config_create = ToolConfig(
            id="tool-update",
            description="Tool update",
            transport="stdio",
            command="python",
        )
        created = await methods.create_tool_config_async(session, "user123", config_create)

        config_update = ToolConfig(
            id="tool-update",
            description="Tool update",
            transport="sse",
            command="node",
            url="http://example.com",
        )
        updated = await methods.update_tool_config_async(session, created, config_update)

        assert updated.id == "tool-update"
        assert updated.transport == "sse"
        assert updated.command == "node"
        assert updated.url == "http://example.com"

    async def test_delete_tool_config(self, session: AsyncSession):
        """Test deleting a tool config."""
        config_create = ToolConfig(
            id="tool-delete",
            description="Tool delete",
            transport="stdio",
        )
        created = await methods.create_tool_config_async(session, "user123", config_create)

        await methods.delete_tool_config_async(session, created)

        retrieved = await methods.get_tool_config_async(session, "user123", config_id="tool-delete")
        assert retrieved is None

    async def test_tool_config_user_scoped_uniqueness(self, session: AsyncSession):
        """Test that tool config IDs are unique within user scope."""
        config1 = ToolConfig(
            id="shared-id",
            description="Shared ID 1",
            transport="stdio",
        )
        await methods.create_tool_config_async(session, "user1", config1)

        # Same ID for different user should work
        config2 = ToolConfig(
            id="shared-id",
            description="Shared ID 2",
            transport="sse",
        )
        created = await methods.create_tool_config_async(session, "user2", config2)
        assert created.user_uuid == "user2"

        # Same ID for same user should fail
        config3 = ToolConfig(
            id="shared-id",
            description="Shared ID 3",
            transport="streamable_http",
        )
        with pytest.raises(Exception):  # SQLAlchemy integrity error. # noqa
            await methods.create_tool_config_async(session, "user1", config3)


class TestToolConfigRepository:
    """Test cases for tool config repository."""

    async def test_create_tool_config_via_repo(self, session: AsyncSession):
        """Test creating a tool config via repository."""
        repo = UserToolRepositoryImpl(user_uuid="user123", session=session)

        config = ToolConfig(
            id="repo-tool",
            description="Repo tool",
            transport="stdio",
        )
        await repo.update_tool_config_async(config)

        # Verify it was created
        retrieved = await methods.get_tool_config_async(session, "user123", config_id="repo-tool")
        assert retrieved is not None
        assert retrieved.id == "repo-tool"

    async def test_get_tool_config_via_repo(self, session: AsyncSession):
        """Test getting a tool config via repository."""
        repo = UserToolRepositoryImpl(user_uuid="user123", session=session)

        config = ToolConfig(
            id="repo-tool-get",
            description="Repo tool get",
            transport="sse",
        )
        await methods.create_tool_config_async(session, "user123", config)

        result = await repo.get_tool_config_async("repo-tool-get")

        assert result is not None
        assert isinstance(result, ToolConfig)
        assert result.id == "repo-tool-get"

    async def test_get_tool_config_via_repo_returns_none_when_not_found(
        self, session: AsyncSession
    ):
        """Test that get_tool_config returns None when config not found."""
        repo = UserToolRepositoryImpl(user_uuid="user123", session=session)

        result = await repo.get_tool_config_async("nonexistent")

        assert result is None

    async def test_list_tool_configs_via_repo(self, session: AsyncSession):
        """Test listing tool configs via repository."""
        repo = UserToolRepositoryImpl(user_uuid="user123", session=session)

        for i in range(3):
            config = ToolConfig(
                id=f"repo-tool-list-{i}",
                description=f"Repo tool list {i}",
                transport="stdio",
            )
            await methods.create_tool_config_async(session, "user123", config)

        results = await repo.list_tool_configs_async()

        assert len(results) == 3
        assert all(isinstance(c, ToolConfig) for c in results)

    async def test_delete_tool_config_via_repo(self, session: AsyncSession):
        """Test deleting a tool config via repository."""
        repo = UserToolRepositoryImpl(user_uuid="user123", session=session)

        config = ToolConfig(
            id="repo-tool-delete",
            description="Repo tool delete",
            transport="stdio",
        )
        await methods.create_tool_config_async(session, "user123", config)

        await repo.delete_tool_config_async("repo-tool-delete")

        # Verify it was deleted
        retrieved = await methods.get_tool_config_async(
            session, "user123", config_id="repo-tool-delete"
        )
        assert retrieved is None

    async def test_tool_repo_raises_error_without_session(self, session: AsyncSession):
        """Test that repository raises error when session is None."""
        repo = UserToolRepositoryImpl(user_uuid="user123", session=None)

        config = ToolConfig(
            id="test",
            description="Test",
            transport="stdio",
        )

        with pytest.raises(ValueError, match="Session and user_uuid are required"):
            await repo.update_tool_config_async(config)

    async def test_tool_repo_raises_error_without_user_id(self, session: AsyncSession):
        """Test that repository raises error when user_uuid is None."""
        repo = UserToolRepositoryImpl(user_uuid=None, session=session)

        config = ToolConfig(
            id="test",
            description="Test",
            transport="stdio",
        )

        with pytest.raises(ValueError, match="Session and user_uuid are required"):
            await repo.update_tool_config_async(config)

    async def test_update_tool_config_create_new(self, session: AsyncSession):
        """Test creating a new tool config via update_tool_config_async."""
        repo = UserToolRepositoryImpl(user_uuid="user123", session=session)

        config = ToolConfig(
            id="repo-tool-update-1",
            description="Repo tool update",
            transport="stdio",
            command="python",
            args=["script.py"],
        )

        await repo.update_tool_config_async(config)

        # Verify it was created
        retrieved = await methods.get_tool_config_async(
            session, "user123", config_id="repo-tool-update-1"
        )
        assert retrieved is not None
        assert retrieved.id == "repo-tool-update-1"
        assert retrieved.transport == "stdio"
        assert retrieved.command == "python"
        assert retrieved.args == ["script.py"]

    async def test_update_tool_config_update_existing(self, session: AsyncSession):
        """Test updating an existing tool config via update_tool_config_async."""
        repo = UserToolRepositoryImpl(user_uuid="user123", session=session)

        # Create initial config
        config_create = ToolConfig(
            id="repo-tool-update-2",
            description="Initial description",
            transport="stdio",
            command="python",
        )
        await methods.create_tool_config_async(session, "user123", config_create)

        # Update via repository
        config_update = ToolConfig(
            id="repo-tool-update-2",
            description="Updated description",
            transport="sse",
            command="node",
            url="http://example.com",
        )
        await repo.update_tool_config_async(config_update)

        # Verify it was updated
        retrieved = await methods.get_tool_config_async(
            session, "user123", config_id="repo-tool-update-2"
        )
        assert retrieved.description == "Updated description"
        assert retrieved.transport == "sse"
        assert retrieved.command == "node"
        assert retrieved.url == "http://example.com"

    async def test_update_tool_config_returns_none(self, session: AsyncSession):
        """Test that update_tool_config_async returns None."""
        repo = UserToolRepositoryImpl(user_uuid="user123", session=session)

        config = ToolConfig(
            id="repo-tool-update-3",
            description="Test",
            transport="stdio",
        )

        result = await repo.update_tool_config_async(config)

        assert result is None

    async def test_update_tool_config_raises_error_without_session(self, session: AsyncSession):
        """Test that update_tool_config_async raises error when session is None."""
        repo = UserToolRepositoryImpl(user_uuid="user123", session=None)

        config = ToolConfig(
            id="test",
            description="Test",
            transport="stdio",
        )

        with pytest.raises(ValueError, match="Session and user_uuid are required"):
            await repo.update_tool_config_async(config)

    async def test_update_tool_config_raises_error_without_user_id(self, session: AsyncSession):
        """Test that update_tool_config_async raises error when user_uuid is None."""
        repo = UserToolRepositoryImpl(user_uuid=None, session=session)

        config = ToolConfig(
            id="test",
            description="Test",
            transport="stdio",
        )

        with pytest.raises(ValueError, match="Session and user_uuid are required"):
            await repo.update_tool_config_async(config)
