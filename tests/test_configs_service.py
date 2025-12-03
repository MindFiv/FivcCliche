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

from fivccliche.modules.configs import methods

# Import models to ensure they're registered with SQLModel
from fivccliche.modules.users.models import User  # noqa: F401
from fivccliche.modules.configs.models import UserEmbedding, UserLLM, UserAgent  # noqa: F401


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
                    id VARCHAR NOT NULL,
                    username VARCHAR NOT NULL,
                    email VARCHAR NOT NULL,
                    hashed_password VARCHAR NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    PRIMARY KEY (id),
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
                    id VARCHAR NOT NULL,
                    description VARCHAR,
                    provider VARCHAR NOT NULL DEFAULT 'openai',
                    model VARCHAR NOT NULL,
                    api_key VARCHAR NOT NULL,
                    base_url VARCHAR,
                    dimension INTEGER NOT NULL DEFAULT 1024,
                    user_id VARCHAR NOT NULL,
                    PRIMARY KEY (id),
                    FOREIGN KEY(user_id) REFERENCES "user" (id)
                )
            """
                )
            )

            # Create user_llm table
            await conn.execute(
                text(
                    """
                CREATE TABLE user_llm (
                    id VARCHAR NOT NULL,
                    description VARCHAR,
                    provider VARCHAR NOT NULL DEFAULT 'openai',
                    model VARCHAR NOT NULL,
                    api_key VARCHAR NOT NULL,
                    base_url VARCHAR,
                    temperature FLOAT NOT NULL DEFAULT 0.5,
                    max_tokens INTEGER NOT NULL DEFAULT 4096,
                    user_id VARCHAR NOT NULL,
                    PRIMARY KEY (id),
                    FOREIGN KEY(user_id) REFERENCES "user" (id)
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
                    id VARCHAR NOT NULL,
                    description VARCHAR,
                    model_id VARCHAR NOT NULL,
                    system_prompt VARCHAR,
                    user_id VARCHAR NOT NULL,
                    PRIMARY KEY (id),
                    FOREIGN KEY(user_id) REFERENCES "user" (id),
                    FOREIGN KEY(model_id) REFERENCES agent_models (id)
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
        assert config.user_id == "user123"
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
        retrieved = await methods.get_embedding_config_async(session, created.id, "user123")

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.user_id == "user123"

    async def test_get_embedding_config_wrong_user(self, session: AsyncSession):
        """Test getting embedding config with wrong user ID."""
        config_create = EmbeddingConfig(
            id="embedding-wrong-user",
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
        )
        created = await methods.create_embedding_config_async(session, "user123", config_create)
        retrieved = await methods.get_embedding_config_async(session, created.id, "user456")

        assert retrieved is None

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

        retrieved = await methods.get_embedding_config_async(session, config.id, "user123")
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
        assert config.user_id == "user123"
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
        retrieved = await methods.get_llm_config_async(session, created.id, "user123")

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

        retrieved = await methods.get_llm_config_async(session, config.id, "user123")
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
        assert config.user_id == "user123"
        assert config.model_id == "model123"

    async def test_get_agent_config(self, session: AsyncSession):
        """Test getting an agent config by ID."""
        config_create = AgentConfig(
            id="agent-2",
            model_id="model123",
        )
        created = await methods.create_agent_config_async(session, "user123", config_create)
        retrieved = await methods.get_agent_config_async(session, created.id, "user123")

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

        retrieved = await methods.get_agent_config_async(session, config.id, "user123")
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
