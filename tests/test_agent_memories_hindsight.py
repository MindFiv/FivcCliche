"""Tests for the Hindsight-backed agent memory implementation.

The Hindsight SDK client is mocked throughout so these tests run without a
live Hindsight server and without exercising the network. They assert both
correct delegation to the SDK and correct mapping onto the implementation
agnostic neutral models (MemoryRetainResult / MemoryRecallResult / MemoryContent).
"""

from types import SimpleNamespace
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
    MemoryListResult,
    MemoryRecallResult,
    MemoryRetainResult,
)


def _make_recall_result_item(text: str, item_type: str | None = None, score: float = 0.9):
    """Build a Hindsight-native recall-result-like object for assertions."""
    item = SimpleNamespace(
        id=f"id-{text}",
        text=text,
        type=item_type,
        metadata={"k": "v"},
        mentioned_at="2026-08-01T10:00:00Z",
        scores=SimpleNamespace(final=score),
    )

    return SimpleNamespace(results=[item]), item


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

    @pytest.mark.asyncio
    async def test_list_maps_dict_items_and_forwards_pagination(self):
        hindsight = MagicMock()
        native = MagicMock()
        native.items = [
            {
                "id": "m1",
                "text": "Alice loves AI",
                "fact_type": "world",
                "metadata": {"k": "v"},
                "date": "2026-07-01T00:00:00Z",
                "mentioned_at": "2026-08-01T10:00:00Z",
            }
        ]
        native.total = 5
        hindsight.memory.list_memories = AsyncMock(return_value=native)

        memory = UserMemoryHindsightImpl(hindsight, bank_id="alice")
        result = await memory.list_async(skip=10, limit=25)

        hindsight.memory.list_memories.assert_awaited_once_with(
            bank_id="alice",
            type=None,
            q=None,
            limit=25,
            offset=10,
        )
        assert isinstance(result, MemoryListResult)
        assert result.total == 5
        assert result.raw is native
        assert len(result.items) == 1
        item = result.items[0]
        assert isinstance(item, MemoryContent)
        assert item.id == "m1"
        assert item.content == "Alice loves AI"
        assert item.categories == ["world"]
        assert item.score is None
        assert item.metadata == {"k": "v"}
        assert item.created_at is not None
        assert item.created_at.isoformat() == "2026-08-01T10:00:00+00:00"

    @pytest.mark.asyncio
    async def test_list_maps_object_items_and_content_fallback(self):
        hindsight = MagicMock()
        item = SimpleNamespace(
            id="m2",
            text=None,
            content="from content field",
            type=None,
            fact_type=None,
            score=None,
            scores=None,
            metadata=None,
            mentioned_at=None,
            occurred_start=None,
            date=None,
            timestamp=None,
            created_at="2026-07-01T00:00:00Z",
        )
        native = MagicMock()
        native.items = [item]
        native.total = 1
        hindsight.memory.list_memories = AsyncMock(return_value=native)

        memory = UserMemoryHindsightImpl(hindsight, bank_id="alice")
        result = await memory.list_async()

        assert result.items[0].content == "from content field"
        assert result.items[0].categories is None
        assert result.items[0].created_at is not None

    @pytest.mark.asyncio
    async def test_list_empty_items_yields_empty_result(self):
        hindsight = MagicMock()
        native = MagicMock()
        native.items = []
        native.total = 0
        hindsight.memory.list_memories = AsyncMock(return_value=native)

        memory = UserMemoryHindsightImpl(hindsight, bank_id="alice")
        result = await memory.list_async()

        assert result.items == []
        assert result.total == 0


_IMPORT_HINDSIGHT = "fivccliche.services.implements.agent_memories_hindsight._import_hindsight"


class TestUserMemoryProviderImpl:
    """UserMemoryProviderImpl config, lazy init, and factory behavior."""

    def test_is_an_iuser_memory_provider(self):
        with patch(_IMPORT_HINDSIGHT, return_value=MagicMock()):
            provider = UserMemoryProviderImpl(MagicMock())
        assert isinstance(provider, IUserMemoryProvider)

    def test_constructor_imports_sdk_but_does_not_create_client(self):
        mock_hindsight_cls = MagicMock()
        with patch(_IMPORT_HINDSIGHT, return_value=mock_hindsight_cls) as mock_import:
            provider = UserMemoryProviderImpl(MagicMock())

        mock_import.assert_called_once()
        mock_hindsight_cls.assert_not_called()
        assert provider._hindsight is None

    def test_constructor_raises_when_hindsight_client_missing(self):
        with (
            patch(
                _IMPORT_HINDSIGHT,
                side_effect=ImportError(
                    "hindsight-client is required to use UserMemoryProviderImpl. "
                    "Install it with: pip install hindsight-client"
                ),
            ),
            pytest.raises(ImportError, match="hindsight-client is required") as exc_info,
        ):
            UserMemoryProviderImpl(MagicMock())

        assert "pip install hindsight-client" in str(exc_info.value)

    def test_get_memory_returns_iuser_memory_bound_to_space_id(self):
        with patch(_IMPORT_HINDSIGHT, return_value=MagicMock()):
            provider = UserMemoryProviderImpl(MagicMock())
        with patch.object(provider, "_get_hindsight", return_value=MagicMock()) as mock_get:
            memory = provider.get_memory(space_id="alice")

        mock_get.assert_called_once()
        assert isinstance(memory, IUserMemory)
        assert memory._bank_id == "alice"

    def test_get_memory_defaults_to_default_bank_when_space_id_none(self):
        with patch(_IMPORT_HINDSIGHT, return_value=MagicMock()):
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
        mock_hindsight_cls = MagicMock()
        with (
            patch(
                "fivccliche.services.implements.agent_memories_hindsight.query_component",
                return_value=config,
            ) as mock_q,
            patch(_IMPORT_HINDSIGHT, return_value=mock_hindsight_cls),
        ):
            provider = UserMemoryProviderImpl(component_site)
            provider.get_memory(space_id="alice")

        mock_q.assert_called_once_with(component_site, configs.IConfig)
        config.get_session.assert_called_once_with("hindsight")
        mock_hindsight_cls.assert_called_once_with(
            base_url="http://hindsight.example:9999",
            api_key="secret-token",
            timeout=120.0,
        )

    def test_build_hindsight_falls_back_to_defaults_when_no_session(self):
        config = MagicMock()
        config.get_session.return_value = None
        component_site = MagicMock()
        mock_hindsight_cls = MagicMock()
        with (
            patch(
                "fivccliche.services.implements.agent_memories_hindsight.query_component",
                return_value=config,
            ),
            patch(_IMPORT_HINDSIGHT, return_value=mock_hindsight_cls),
        ):
            provider = UserMemoryProviderImpl(component_site)
            provider.get_memory(space_id="alice")

        mock_hindsight_cls.assert_called_once_with(
            base_url="http://localhost:8888",
            api_key=None,
            timeout=300.0,
        )

    def test_build_hindsight_falls_back_when_no_config_component(self):
        component_site = MagicMock()
        mock_hindsight_cls = MagicMock()
        with (
            patch(
                "fivccliche.services.implements.agent_memories_hindsight.query_component",
                return_value=None,
            ),
            patch(_IMPORT_HINDSIGHT, return_value=mock_hindsight_cls),
        ):
            provider = UserMemoryProviderImpl(component_site)
            provider.get_memory(space_id="alice")

        mock_hindsight_cls.assert_called_once_with(
            base_url="http://localhost:8888",
            api_key=None,
            timeout=300.0,
        )

    def test_hindsight_client_is_created_once_and_reused(self):
        config = MagicMock()
        config.get_session.return_value = None
        mock_hindsight_cls = MagicMock()
        with (
            patch(
                "fivccliche.services.implements.agent_memories_hindsight.query_component",
                return_value=config,
            ),
            patch(_IMPORT_HINDSIGHT, return_value=mock_hindsight_cls),
        ):
            provider = UserMemoryProviderImpl(MagicMock())
            provider.get_memory(space_id="a")
            provider.get_memory(space_id="b")

        mock_hindsight_cls.assert_called_once()


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
