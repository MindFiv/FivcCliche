"""Unit tests for streaming generator utilities."""

import asyncio
import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from fivcplayground.agents import AgentRunEvent

from fivccliche.utils.generators import _ChatStreamingGenerator


class TestChatStreamingGenerator:
    """Test the _ChatStreamingGenerator class."""

    @pytest.fixture
    def mock_task(self):
        """Create a mock asyncio.Task."""
        task = MagicMock(spec=asyncio.Task)
        task.done.return_value = False
        task.result.return_value = None
        return task

    @pytest.fixture
    def mock_queue(self):
        """Create a mock asyncio.Queue."""
        queue = MagicMock()
        queue.empty = Mock(return_value=False)
        queue.task_done = Mock()
        return queue

    @pytest.fixture
    def generator(self, mock_task, mock_queue):
        """Create a generator instance."""
        return _ChatStreamingGenerator(
            chat_task=mock_task,
            chat_queue=mock_queue,
            chat_uuid="test-chat-uuid",
        )

    def test_initialization(self, mock_task, mock_queue):
        """Test generator initializes correctly."""
        generator = _ChatStreamingGenerator(
            chat_task=mock_task,
            chat_queue=mock_queue,
            chat_uuid="test-uuid",
        )
        assert generator.chat_task == mock_task
        assert generator.chat_queue == mock_queue
        assert generator.chat_uuid == "test-uuid"

    def test_initialization_without_chat_uuid(self, mock_task, mock_queue):
        """Test generator initializes correctly without chat_uuid."""
        generator = _ChatStreamingGenerator(
            chat_task=mock_task,
            chat_queue=mock_queue,
        )
        assert generator.chat_uuid is None

    @pytest.mark.asyncio
    async def test_start_event_formatting(self, generator, mock_task, mock_queue):
        """Test START event is formatted correctly."""
        # Create mock agent run
        mock_run = Mock()
        mock_run.model_dump.return_value = {
            "id": "run-1",
            "agent_id": "agent-1",
            "started_at": "2024-01-01T00:00:00",
            "completed_at": None,
            "query": "test query",
            "reply": None,
            "tool_calls": [],
        }

        # Setup queue to return START event then finish
        async def mock_wait_for(coro, timeout):
            if mock_task.done.return_value:
                raise asyncio.TimeoutError()
            mock_task.done.return_value = True
            return (AgentRunEvent.START, mock_run)

        mock_queue.empty.return_value = True

        # Patch asyncio.wait_for
        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in generator():
                results.append(chunk)

            assert len(results) == 1
            data = json.loads(results[0].replace("data: ", "").strip())
            assert data["event"] == "start"
            assert data["info"]["chat_uuid"] == "test-chat-uuid"
            assert data["info"]["id"] == "run-1"

    @pytest.mark.asyncio
    async def test_finish_event_formatting(self, generator, mock_task, mock_queue):
        """Test FINISH event is formatted correctly."""
        mock_run = Mock()
        mock_run.model_dump.return_value = {
            "id": "run-1",
            "agent_id": "agent-1",
            "started_at": "2024-01-01T00:00:00",
            "completed_at": "2024-01-01T00:01:00",
            "query": "test query",
            "reply": "test reply",
            "tool_calls": [],
        }

        async def mock_wait_for(coro, timeout):
            if mock_task.done.return_value:
                raise asyncio.TimeoutError()
            mock_task.done.return_value = True
            return (AgentRunEvent.FINISH, mock_run)

        mock_queue.empty.return_value = True

        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in generator():
                results.append(chunk)

            assert len(results) == 1
            data = json.loads(results[0].replace("data: ", "").strip())
            assert data["event"] == "finish"
            assert data["info"]["chat_uuid"] == "test-chat-uuid"
            assert data["info"]["reply"] == "test reply"

    @pytest.mark.asyncio
    async def test_stream_event_with_delta(self, generator, mock_task, mock_queue):
        """Test STREAM event with delta is formatted correctly."""
        mock_delta = Mock()
        mock_delta.model_dump.return_value = {"content": "partial text"}

        mock_run = Mock()
        mock_run.model_dump.return_value = {
            "id": "run-1",
            "agent_id": "agent-1",
            "started_at": "2024-01-01T00:00:00",
            "completed_at": None,
            "query": "test query",
            "reply": None,
            "tool_calls": [],
        }
        mock_run.delta = mock_delta

        async def mock_wait_for(coro, timeout):
            if mock_task.done.return_value:
                raise asyncio.TimeoutError()
            mock_task.done.return_value = True
            return (AgentRunEvent.STREAM, mock_run)

        mock_queue.empty.return_value = True

        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in generator():
                results.append(chunk)

            assert len(results) == 1
            data = json.loads(results[0].replace("data: ", "").strip())
            assert data["event"] == "stream"
            assert data["info"]["chat_uuid"] == "test-chat-uuid"
            assert data["info"]["delta"] == {"content": "partial text"}

    @pytest.mark.asyncio
    async def test_stream_event_without_delta(self, generator, mock_task, mock_queue):
        """Test STREAM event without delta is formatted correctly."""
        mock_run = Mock()
        mock_run.model_dump.return_value = {
            "id": "run-1",
            "agent_id": "agent-1",
            "started_at": "2024-01-01T00:00:00",
            "completed_at": None,
            "query": "test query",
            "reply": None,
            "tool_calls": [],
        }
        mock_run.delta = None

        async def mock_wait_for(coro, timeout):
            if mock_task.done.return_value:
                raise asyncio.TimeoutError()
            mock_task.done.return_value = True
            return (AgentRunEvent.STREAM, mock_run)

        mock_queue.empty.return_value = True

        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in generator():
                results.append(chunk)

            assert len(results) == 1
            data = json.loads(results[0].replace("data: ", "").strip())
            assert data["event"] == "stream"
            assert data["info"]["delta"] is None

    @pytest.mark.asyncio
    async def test_tool_event_formatting(self, generator, mock_task, mock_queue):
        """Test TOOL event is formatted correctly."""
        mock_run = Mock()
        mock_run.model_dump.return_value = {
            "id": "run-1",
            "agent_id": "agent-1",
            "started_at": "2024-01-01T00:00:00",
            "completed_at": None,
            "query": "test query",
            "reply": None,
            "tool_calls": [{"name": "search", "args": {}}],
        }

        async def mock_wait_for(coro, timeout):
            if mock_task.done.return_value:
                raise asyncio.TimeoutError()
            mock_task.done.return_value = True
            return (AgentRunEvent.TOOL, mock_run)

        mock_queue.empty.return_value = True

        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in generator():
                results.append(chunk)

            assert len(results) == 1
            data = json.loads(results[0].replace("data: ", "").strip())
            assert data["event"] == "tool"
            assert data["info"]["chat_uuid"] == "test-chat-uuid"
            assert len(data["info"]["tool_calls"]) == 1

    @pytest.mark.asyncio
    async def test_error_handling(self, generator, mock_task, mock_queue):
        """Test error event is generated on exception."""

        # Make wait_for raise an exception
        async def mock_wait_for(coro, timeout):
            raise ValueError("Test error")

        mock_queue.empty.return_value = False

        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in generator():
                results.append(chunk)

            assert len(results) == 1
            data = json.loads(results[0].replace("data: ", "").strip())
            assert data["event"] == "error"
            assert "Test error" in data["info"]["message"]

    @pytest.mark.asyncio
    async def test_timeout_handling(self, generator, mock_task, mock_queue):
        """Test timeout behavior when queue is empty."""
        # First call times out, second call task is done
        call_count = 0

        async def mock_wait_for(coro, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError()
            # Second call, task is done
            mock_task.done.return_value = True
            raise asyncio.TimeoutError()

        mock_queue.empty.return_value = True

        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in generator():
                results.append(chunk)

            # Should exit cleanly without yielding anything
            assert len(results) == 0
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_chat_uuid_added_to_all_events(self, generator, mock_task, mock_queue):
        """Test chat_uuid is added to all event types."""
        events = [
            (AgentRunEvent.START, Mock()),
            (AgentRunEvent.STREAM, Mock()),
            (AgentRunEvent.TOOL, Mock()),
            (AgentRunEvent.FINISH, Mock()),
        ]

        for event_mock in events:
            event_mock[1].model_dump.return_value = {
                "id": "run-1",
                "agent_id": "agent-1",
                "started_at": "2024-01-01T00:00:00",
                "completed_at": None,
                "query": "test",
                "reply": None,
                "tool_calls": [],
            }
            event_mock[1].delta = None

        event_index = 0

        async def mock_wait_for(coro, timeout):
            nonlocal event_index
            if event_index >= len(events):
                mock_task.done.return_value = True
                raise asyncio.TimeoutError()
            event = events[event_index]
            event_index += 1
            return event

        # Make empty() return True after all events are processed
        def mock_empty():
            return event_index >= len(events)

        mock_queue.empty.side_effect = mock_empty

        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in generator():
                results.append(chunk)

            assert len(results) == 4
            for result in results:
                data = json.loads(result.replace("data: ", "").strip())
                assert data["info"]["chat_uuid"] == "test-chat-uuid"

    @pytest.mark.asyncio
    async def test_task_done_and_queue_empty_exits(self, generator, mock_task, mock_queue):
        """Test generator exits when task is done and queue is empty."""
        mock_task.done.return_value = True
        mock_queue.empty.return_value = True

        results = []
        async for chunk in generator():
            results.append(chunk)

        assert len(results) == 0
        mock_task.result.assert_called_once()  # Should check for exceptions
