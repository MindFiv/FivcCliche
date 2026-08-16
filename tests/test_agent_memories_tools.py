"""Unit tests for agent memory function tools."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fivccliche.modules.agent_memories.tools import MemoryList, MemoryRecall, MemoryRetain
from fivccliche.services.interfaces.agent_memories import (
    MemoryContent,
    MemoryListResult,
    MemoryRecallResult,
    MemoryRetainResult,
)

USER_UUID = "user-alice"
CONTENT = MemoryContent(
    id="m1",
    content="Alice likes tea",
    score=0.9,
    categories=["world"],
    created_at=datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc),
)


def _provider_with_memory(memory: MagicMock) -> MagicMock:
    provider = MagicMock()
    provider.get_memory.return_value = memory
    return provider


class TestMemoryRetain:
    @pytest.mark.asyncio
    async def test_raises_without_user_uuid(self):
        tool = MemoryRetain()
        with pytest.raises(ValueError, match="No user_uuid specified"):
            await tool("remember this")

    @pytest.mark.asyncio
    async def test_raises_when_provider_missing(self):
        tool = MemoryRetain(user_uuid=USER_UUID)
        with patch(
            "fivccliche.modules.agent_memories.tools.get_memory_provider_async",
            new=AsyncMock(return_value=None),
        ):
            with pytest.raises(ValueError, match="No memory provider specified"):
                await tool("remember this")

    @pytest.mark.asyncio
    async def test_retains_for_context_user_and_returns_json(self):
        memory = MagicMock()
        memory.retain_async = AsyncMock(
            return_value=MemoryRetainResult(success=True, count=1, ids=["id-1"], raw={"x": 1})
        )
        tool = MemoryRetain(user_uuid=USER_UUID)
        with patch(
            "fivccliche.modules.agent_memories.tools.get_memory_provider_async",
            new=AsyncMock(return_value=_provider_with_memory(memory)),
        ) as get_provider:
            result = await tool("remember this")

        get_provider.return_value.get_memory.assert_called_once_with(space_id=USER_UUID)
        memory.retain_async.assert_awaited_once_with("remember this")
        assert result == MemoryRetainResult(success=True, count=1, ids=["id-1"]).model_dump_json(
            exclude={"raw"}
        )

    @pytest.mark.asyncio
    async def test_returns_json_when_retain_fails(self):
        memory = MagicMock()
        memory.retain_async = AsyncMock(
            return_value=MemoryRetainResult(success=False, count=0, ids=None)
        )
        tool = MemoryRetain(user_uuid=USER_UUID)
        with patch(
            "fivccliche.modules.agent_memories.tools.get_memory_provider_async",
            new=AsyncMock(return_value=_provider_with_memory(memory)),
        ):
            result = await tool("remember this")

        assert result == MemoryRetainResult(success=False, count=0, ids=None).model_dump_json(
            exclude={"raw"}
        )


class TestMemoryRecall:
    @pytest.mark.asyncio
    async def test_raises_without_user_uuid(self):
        tool = MemoryRecall()
        with pytest.raises(ValueError, match="No user_uuid specified"):
            await tool("tea")

    @pytest.mark.asyncio
    async def test_raises_when_provider_missing(self):
        tool = MemoryRecall(user_uuid=USER_UUID)
        with patch(
            "fivccliche.modules.agent_memories.tools.get_memory_provider_async",
            new=AsyncMock(return_value=None),
        ):
            with pytest.raises(ValueError, match="No memory provider specified"):
                await tool("tea")

    @pytest.mark.asyncio
    async def test_recalls_for_context_user_and_returns_items_json(self):
        memory = MagicMock()
        memory.recall_async = AsyncMock(
            return_value=MemoryRecallResult(items=[CONTENT], raw={"ignored": True})
        )
        tool = MemoryRecall(user_uuid=USER_UUID)
        with patch(
            "fivccliche.modules.agent_memories.tools.get_memory_provider_async",
            new=AsyncMock(return_value=_provider_with_memory(memory)),
        ) as get_provider:
            result = await tool("tea")

        get_provider.return_value.get_memory.assert_called_once_with(space_id=USER_UUID)
        memory.recall_async.assert_awaited_once_with("tea")
        assert result == MemoryRecallResult(items=[CONTENT]).model_dump_json(exclude={"raw"})


class TestMemoryList:
    @pytest.mark.asyncio
    async def test_raises_without_user_uuid(self):
        tool = MemoryList()
        with pytest.raises(ValueError, match="No user_uuid specified"):
            await tool()

    @pytest.mark.asyncio
    async def test_raises_when_provider_missing(self):
        tool = MemoryList(user_uuid=USER_UUID)
        with patch(
            "fivccliche.modules.agent_memories.tools.get_memory_provider_async",
            new=AsyncMock(return_value=None),
        ):
            with pytest.raises(ValueError, match="No memory provider specified"):
                await tool()

    @pytest.mark.asyncio
    async def test_lists_for_context_user_and_returns_json(self):
        memory = MagicMock()
        memory.list_async = AsyncMock(
            return_value=MemoryListResult(items=[CONTENT], total=1, raw={"ignored": True})
        )
        tool = MemoryList(user_uuid=USER_UUID)
        with patch(
            "fivccliche.modules.agent_memories.tools.get_memory_provider_async",
            new=AsyncMock(return_value=_provider_with_memory(memory)),
        ) as get_provider:
            result = await tool(skip=2, limit=5)

        get_provider.return_value.get_memory.assert_called_once_with(space_id=USER_UUID)
        memory.list_async.assert_awaited_once_with(skip=2, limit=5)
        assert result == MemoryListResult(items=[CONTENT], total=1).model_dump_json(exclude={"raw"})
