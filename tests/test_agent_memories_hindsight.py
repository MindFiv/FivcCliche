"""Tests for the Hindsight-backed agent memory implementation.

The Hindsight SDK client is mocked throughout so these tests run without a
live Hindsight server and without exercising the network. They assert both
correct delegation to the SDK and correct mapping onto the implementation
agnostic neutral models (MemoryRetainResult / MemoryRecallResult / MemoryContent).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fivcglue.interfaces import configs

from fivccliche.services.implements.agent_memories_hindsight import (
    UserMemoryHindsightImpl,
    UserMemoryProviderImpl,
)
from fivccliche.services.interfaces.agent_memories import (
    IUserMemory,
    IUserMemoryProvider,
    MemoryContent,
    MemoryRecallResult,
    MemoryRetainResult,
)


def _make_recall_result_item(text: str, item_type: str | None = None, score: float = 0.9):
    """Build a Hindsight-native recall-result-like object for assertions."""
    item = MagicMock()
    item.text = text
    item.type = item_type
    item.score = score
    item.id = f"id-{text}"
    item.metadata = {"k": "v"}
    item.timestamp = "2026-08-01T10:00:00Z"

    resp = MagicMock()
    resp.results = [item]
    resp.success = True
    return resp, item


class TestUserMemoryHindsightImpl:
    """UserMemoryHindsightImpl delegation + mapping."""

    @pytest.mark.asyncio
    async def test_retain_delegates_and_returns_neutral_result(self):
        hindsight = MagicMock()
        native = MagicMock()
        native.success = True
        hindsight.aretain = AsyncMock(return_value=native)

        memory = UserMemoryHindsightImpl(hindsight, bank_id="alice")

        result = await memory.retain_async("hello world")

        hindsight.aretain.assert_awaited_once_with(bank_id="alice", content="hello world")
        assert isinstance(result, MemoryRetainResult)
        assert result.success is True
        assert result.count == 1
        assert result.raw is native

    @pytest.mark.asyncio
    async def test_retain_propagates_failure_flag(self):
        hindsight = MagicMock()
        native = MagicMock()
        native.success = False
        hindsight.aretain = AsyncMock(return_value=native)

        memory = UserMemoryHindsightImpl(hindsight, bank_id="alice")
        result = await memory.retain_async("x")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_recall_maps_native_results_to_memory_content(self):
        hindsight = MagicMock()
        native, _ = _make_recall_result_item("Alice loves AI", item_type="world")
        hindsight.arecall = AsyncMock(return_value=native)

        memory = UserMemoryHindsightImpl(hindsight, bank_id="alice")
        result = await memory.recall_async("what does Alice like?")

        hindsight.arecall.assert_awaited_once_with(bank_id="alice", query="what does Alice like?")
        assert isinstance(result, MemoryRecallResult)
        assert result.raw is native
        assert len(result.items) == 1
        item = result.items[0]
        assert isinstance(item, MemoryContent)
        assert item.content == "Alice loves AI"
        assert item.categories == ["world"]
        assert item.score == 0.9
        assert item.id == "id-Alice loves AI"
        assert item.metadata == {"k": "v"}
        assert item.created_at is not None
        assert item.created_at.isoformat() == "2026-08-01T10:00:00+00:00"

    @pytest.mark.asyncio
    async def test_recall_without_type_yields_none_categories(self):
        hindsight = MagicMock()
        native, _ = _make_recall_result_item("plain text", item_type=None)
        hindsight.arecall = AsyncMock(return_value=native)

        memory = UserMemoryHindsightImpl(hindsight, bank_id="alice")
        result = await memory.recall_async("q")

        assert result.items[0].categories is None

    @pytest.mark.asyncio
    async def test_recall_empty_results_yields_empty_items(self):
        hindsight = MagicMock()
        native = MagicMock()
        native.results = []
        hindsight.arecall = AsyncMock(return_value=native)

        memory = UserMemoryHindsightImpl(hindsight, bank_id="alice")
        result = await memory.recall_async("q")

        assert result.items == []

    @pytest.mark.asyncio
    async def test_recall_missing_results_attr_is_tolerated(self):
        hindsight = MagicMock()
        native = MagicMock(spec=[])  # no `results` attribute
        hindsight.arecall = AsyncMock(return_value=native)

        memory = UserMemoryHindsightImpl(hindsight, bank_id="alice")
        result = await memory.recall_async("q")

        assert result.items == []
        assert result.raw is native


class TestUserMemoryProviderImpl:
    """UserMemoryProviderImpl config, lazy init, and factory behavior."""

    def test_is_an_iuser_memory_provider(self):
        provider = UserMemoryProviderImpl(MagicMock())
        assert isinstance(provider, IUserMemoryProvider)

    def test_constructor_does_not_create_hindsight_client(self):
        with patch(
            "fivccliche.services.implements.agent_memories_hindsight.Hindsight"
        ) as mock_hindsight:
            UserMemoryProviderImpl(MagicMock())
            mock_hindsight.assert_not_called()

    def test_get_memory_returns_iuser_memory_bound_to_space_id(self):
        provider = UserMemoryProviderImpl(MagicMock())
        with patch.object(provider, "_get_hindsight", return_value=MagicMock()) as mock_get:
            memory = provider.get_memory(space_id="alice")

        mock_get.assert_called_once()
        assert isinstance(memory, IUserMemory)
        assert memory._bank_id == "alice"

    def test_get_memory_defaults_to_default_bank_when_space_id_none(self):
        provider = UserMemoryProviderImpl(MagicMock())
        with patch.object(provider, "_get_hindsight", return_value=MagicMock()):
            memory = provider.get_memory(space_id=None)

        assert memory._bank_id == "default"

    def test_build_hindsight_reads_config_session(self):
        session = MagicMock()
        session.get_value.side_effect = lambda key: {
            "BASE_URL": "http://hindsight.example:9999",
            "API_KEY": "secret-token",
            "TIMEOUT": "120.0",
        }[key]

        config = MagicMock()
        config.get_session.return_value = session
        component_site = MagicMock()
        with (
            patch(
                "fivccliche.services.implements.agent_memories_hindsight.query_component",
                return_value=config,
            ) as mock_q,
            patch(
                "fivccliche.services.implements.agent_memories_hindsight.Hindsight"
            ) as mock_hindsight,
        ):
            provider = UserMemoryProviderImpl(component_site)
            provider.get_memory(space_id="alice")

        mock_q.assert_called_once_with(component_site, configs.IConfig)
        config.get_session.assert_called_once_with("hindsight")
        mock_hindsight.assert_called_once_with(
            base_url="http://hindsight.example:9999",
            api_key="secret-token",
            timeout=120.0,
        )

    def test_build_hindsight_falls_back_to_defaults_when_no_session(self):
        config = MagicMock()
        config.get_session.return_value = None
        component_site = MagicMock()
        with (
            patch(
                "fivccliche.services.implements.agent_memories_hindsight.query_component",
                return_value=config,
            ),
            patch(
                "fivccliche.services.implements.agent_memories_hindsight.Hindsight"
            ) as mock_hindsight,
        ):
            provider = UserMemoryProviderImpl(component_site)
            provider.get_memory(space_id="alice")

        mock_hindsight.assert_called_once_with(
            base_url="http://localhost:8888",
            api_key=None,
            timeout=300.0,
        )

    def test_build_hindsight_falls_back_when_no_config_component(self):
        component_site = MagicMock()
        with (
            patch(
                "fivccliche.services.implements.agent_memories_hindsight.query_component",
                return_value=None,
            ),
            patch(
                "fivccliche.services.implements.agent_memories_hindsight.Hindsight"
            ) as mock_hindsight,
        ):
            provider = UserMemoryProviderImpl(component_site)
            provider.get_memory(space_id="alice")

        mock_hindsight.assert_called_once_with(
            base_url="http://localhost:8888",
            api_key=None,
            timeout=300.0,
        )

    def test_hindsight_client_is_created_once_and_reused(self):
        config = MagicMock()
        config.get_session.return_value = None
        provider = UserMemoryProviderImpl(MagicMock())
        with (
            patch(
                "fivccliche.services.implements.agent_memories_hindsight.query_component",
                return_value=config,
            ),
            patch(
                "fivccliche.services.implements.agent_memories_hindsight.Hindsight"
            ) as mock_hindsight,
        ):
            provider.get_memory(space_id="a")
            provider.get_memory(space_id="b")

        mock_hindsight.assert_called_once()


class TestGetMemoryProviderAsync:
    """deps.get_memory_provider_async optional mounting semantics."""

    @pytest.mark.asyncio
    async def test_returns_none_when_provider_not_registered(self):
        from fivccliche.utils import deps

        with patch(
            "fivccliche.utils.deps.default_memory_provider",
            MagicMock(return_value=None),
        ):
            result = await deps.get_memory_provider_async()

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_provider_when_registered(self):
        from fivccliche.utils import deps

        provider = MagicMock(spec=IUserMemoryProvider)
        with patch(
            "fivccliche.utils.deps.default_memory_provider",
            MagicMock(return_value=provider),
        ):
            result = await deps.get_memory_provider_async()

        assert result is provider
