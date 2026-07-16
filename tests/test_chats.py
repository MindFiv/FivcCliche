"""Unit tests for streaming generator utilities."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fivcplayground.agents import AgentRunEvent

from fivccliche.utils.chats import ChatTask


class TestChatTaskGetStream:
    """Test ChatTask.get_stream_async() SSE formatting."""

    def _make_chat_task(self, chat_uuid: str | None = "test-chat-uuid"):
        user = Mock()
        user.uuid = "user-1"
        with patch("fivccliche.utils.chats.default_db", new_callable=Mock) as mock_db:
            mock_db.return_value.create_session.return_value = AsyncMock()
            chat_task = ChatTask(user, Mock(), Mock(), chat_uuid=chat_uuid)
        chat_task._asyncio_task = MagicMock(spec=asyncio.Task)
        chat_task._asyncio_task.done.return_value = False
        chat_task._asyncio_task.result.return_value = None
        chat_task._chat_queue = MagicMock()
        chat_task._chat_queue.empty = Mock(return_value=False)
        chat_task._chat_queue.task_done = Mock()
        return chat_task

    def test_get_stream_returns_async_generator(self):
        """get_stream_async() returns an async generator."""
        chat_task = self._make_chat_task()
        stream = chat_task.get_stream_async()
        assert hasattr(stream, "__aiter__")

    def test_get_stream_without_chat_uuid(self):
        """get_stream works when chat_uuid is None."""
        chat_task = self._make_chat_task(chat_uuid=None)
        stream = chat_task.get_stream_async()
        assert hasattr(stream, "__aiter__")

    @pytest.mark.asyncio
    async def test_start_event_formatting(self):
        """Test START event is formatted correctly."""
        chat_task = self._make_chat_task()
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

        async def mock_wait_for(coro, timeout):
            if chat_task._asyncio_task.done.return_value:
                raise asyncio.TimeoutError()
            chat_task._asyncio_task.done.return_value = True
            return (AgentRunEvent.START, mock_run)

        chat_task._chat_queue.empty.return_value = True
        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in chat_task.get_stream_async():
                results.append(chunk)

        assert len(results) == 1
        data = json.loads(results[0].replace("data: ", "").strip())
        assert data["event"] == "start"
        assert data["info"]["chat_uuid"] == "test-chat-uuid"
        assert data["info"]["id"] == "run-1"

    @pytest.mark.asyncio
    async def test_finish_event_formatting(self):
        """Test FINISH event is formatted correctly."""
        chat_task = self._make_chat_task()
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
            if chat_task._asyncio_task.done.return_value:
                raise asyncio.TimeoutError()
            chat_task._asyncio_task.done.return_value = True
            return (AgentRunEvent.FINISH, mock_run)

        chat_task._chat_queue.empty.return_value = True
        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in chat_task.get_stream_async():
                results.append(chunk)

        assert len(results) == 1
        data = json.loads(results[0].replace("data: ", "").strip())
        assert data["event"] == "finish"
        assert data["info"]["chat_uuid"] == "test-chat-uuid"
        assert data["info"]["reply"] == "test reply"

    @pytest.mark.asyncio
    async def test_stream_event_with_delta(self):
        """Test STREAM event with delta is formatted correctly."""
        chat_task = self._make_chat_task()
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
            if chat_task._asyncio_task.done.return_value:
                raise asyncio.TimeoutError()
            chat_task._asyncio_task.done.return_value = True
            return (AgentRunEvent.STREAM, mock_run)

        chat_task._chat_queue.empty.return_value = True
        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in chat_task.get_stream_async():
                results.append(chunk)

        assert len(results) == 1
        data = json.loads(results[0].replace("data: ", "").strip())
        assert data["event"] == "stream"
        assert data["info"]["chat_uuid"] == "test-chat-uuid"
        assert data["info"]["delta"] == {"content": "partial text"}

    @pytest.mark.asyncio
    async def test_stream_event_without_delta(self):
        """Test STREAM event without delta is formatted correctly."""
        chat_task = self._make_chat_task()
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
            if chat_task._asyncio_task.done.return_value:
                raise asyncio.TimeoutError()
            chat_task._asyncio_task.done.return_value = True
            return (AgentRunEvent.STREAM, mock_run)

        chat_task._chat_queue.empty.return_value = True
        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in chat_task.get_stream_async():
                results.append(chunk)

        assert len(results) == 1
        data = json.loads(results[0].replace("data: ", "").strip())
        assert data["event"] == "stream"
        assert data["info"]["delta"] is None

    @pytest.mark.asyncio
    async def test_tool_event_formatting(self):
        """Test TOOL event is formatted correctly."""
        chat_task = self._make_chat_task()
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
            if chat_task._asyncio_task.done.return_value:
                raise asyncio.TimeoutError()
            chat_task._asyncio_task.done.return_value = True
            return (AgentRunEvent.TOOL, mock_run)

        chat_task._chat_queue.empty.return_value = True
        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in chat_task.get_stream_async():
                results.append(chunk)

        assert len(results) == 1
        data = json.loads(results[0].replace("data: ", "").strip())
        assert data["event"] == "tool"
        assert data["info"]["chat_uuid"] == "test-chat-uuid"
        assert len(data["info"]["tool_calls"]) == 1

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error event is generated on exception."""
        chat_task = self._make_chat_task()

        async def mock_wait_for(coro, timeout):
            raise ValueError("Test error")

        chat_task._chat_queue.empty.return_value = False
        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in chat_task.get_stream_async():
                results.append(chunk)

        assert len(results) == 1
        data = json.loads(results[0].replace("data: ", "").strip())
        assert data["event"] == "error"
        assert "Test error" in data["info"]["message"]

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout behavior when queue is empty."""
        chat_task = self._make_chat_task()
        call_count = 0

        async def mock_wait_for(coro, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError()
            chat_task._asyncio_task.done.return_value = True
            raise asyncio.TimeoutError()

        chat_task._chat_queue.empty.return_value = True
        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in chat_task.get_stream_async():
                results.append(chunk)

        assert len(results) == 0
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_chat_uuid_added_to_all_events(self):
        """Test chat_uuid is added to all event types."""
        chat_task = self._make_chat_task()
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
                chat_task._asyncio_task.done.return_value = True
                raise asyncio.TimeoutError()
            event = events[event_index]
            event_index += 1
            return event

        def mock_empty():
            return event_index >= len(events)

        chat_task._chat_queue.empty.side_effect = mock_empty
        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in chat_task.get_stream_async():
                results.append(chunk)

        assert len(results) == 4
        for result in results:
            data = json.loads(result.replace("data: ", "").strip())
            assert data["info"]["chat_uuid"] == "test-chat-uuid"

    @pytest.mark.asyncio
    async def test_task_done_and_queue_empty_exits(self):
        """Test generator exits when task is done and queue is empty."""
        chat_task = self._make_chat_task()
        chat_task._asyncio_task.done.return_value = True
        chat_task._chat_queue.empty.return_value = True

        results = []
        async for chunk in chat_task.get_stream_async():
            results.append(chunk)

        assert len(results) == 0
        chat_task._asyncio_task.result.assert_called_once()


class TestChatTask:
    """Test ChatTask agent execution and get_stream integration."""

    def _make_mock_user(self):
        user = Mock()
        user.uuid = "user-uuid-123"
        return user

    def _make_mock_config_provider(self):
        provider = Mock()
        provider.get_model_backend.return_value = Mock()
        provider.get_model_repository.return_value = Mock()
        provider.get_agent_backend.return_value = Mock()
        provider.get_agent_repository.return_value = Mock()
        provider.get_tool_backend.return_value = Mock()
        provider.get_tool_repository.return_value = Mock()
        provider.get_embedding_backend.return_value = Mock()
        provider.get_embedding_repository.return_value = Mock()
        provider.get_skill_repository.return_value = Mock()
        return provider

    def _make_mock_chat_provider(self):
        provider = Mock()
        provider.get_chat_repository.return_value = Mock()
        provider.get_chat_context.return_value = None
        return provider

    @staticmethod
    def _make_mock_tool(name: str):
        tool = Mock()
        tool.name = name
        return tool

    def _start_chat_task(self, user, user_config_provider, user_chat_provider, **kwargs):
        """Construct ChatTask, start it, return (chat_task, stream)."""
        chat_task = ChatTask(user, user_config_provider, user_chat_provider, **kwargs)
        chat_task.start()
        return chat_task, chat_task.get_stream_async()

    @pytest.mark.asyncio
    @patch("fivccliche.utils.chats.create_skill_retriever_async")
    @patch("fivccliche.utils.chats.create_tool_retriever_async")
    @patch("fivccliche.utils.chats.create_agent_async")
    async def test_create_generator_skills_disabled_by_default(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        """create_skill_retriever_async not called when chat_skills_enabled=False."""
        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock()
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        user = self._make_mock_user()
        config_provider = self._make_mock_config_provider()
        chat_provider = self._make_mock_chat_provider()

        with patch("fivccliche.utils.chats.default_db", new_callable=Mock) as mock_default_db:
            owned_session = AsyncMock()
            owned_session.close = AsyncMock()
            mock_default_db.return_value.create_session.return_value = owned_session
            chat_task, result = self._start_chat_task(
                user,
                config_provider,
                chat_provider,
                chat_uuid="chat-uuid-1",
                chat_query="hello",
                chat_skills_enabled=False,
            )
            await chat_task.join_async()

        mock_create_skill_retriever.assert_not_called()
        chat_provider.get_chat_context.assert_not_called()
        _, kwargs = mock_agent.run_async.call_args
        assert kwargs["skill_retriever"] is None
        assert hasattr(result, "__aiter__")

    @pytest.mark.asyncio
    @patch("fivccliche.utils.chats.create_skill_retriever_async")
    @patch("fivccliche.utils.chats.create_tool_retriever_async")
    @patch("fivccliche.utils.chats.create_agent_async")
    async def test_create_generator_skills_enabled(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        """create_skill_retriever_async is called when chat_skills_enabled=True."""
        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock()
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_skill_retriever = AsyncMock()
        mock_create_skill_retriever.return_value = mock_skill_retriever

        user = self._make_mock_user()
        config_provider = self._make_mock_config_provider()
        chat_provider = self._make_mock_chat_provider()

        with patch("fivccliche.utils.chats.default_db", new_callable=Mock) as mock_default_db:
            owned_session = AsyncMock()
            owned_session.close = AsyncMock()
            mock_default_db.return_value.create_session.return_value = owned_session
            chat_task, result = self._start_chat_task(
                user,
                config_provider,
                chat_provider,
                chat_uuid="chat-uuid-2",
                chat_query="hello",
                chat_skills_enabled=True,
            )
            await chat_task.join_async()

        mock_create_skill_retriever.assert_called_once()
        chat_provider.get_chat_context.assert_not_called()
        _, kwargs = mock_agent.run_async.call_args
        assert kwargs["skill_retriever"] is mock_skill_retriever
        assert hasattr(result, "__aiter__")

    @pytest.mark.asyncio
    @patch("fivccliche.utils.chats.create_skill_retriever_async")
    @patch("fivccliche.utils.chats.create_tool_retriever_async")
    @patch("fivccliche.utils.chats.create_agent_async")
    async def test_create_generator_passes_chat_context_without_provider_lookup(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        """Chat context is passed to the agent without provider context resolution."""
        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock()
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_skill_retriever = AsyncMock()
        mock_create_skill_retriever.return_value = mock_skill_retriever

        user = self._make_mock_user()
        config_provider = self._make_mock_config_provider()
        chat_provider = self._make_mock_chat_provider()
        context = {"project": "alpha"}
        owned_session = AsyncMock()
        owned_session.close = AsyncMock()

        with patch("fivccliche.utils.chats.default_db", new_callable=Mock) as mock_default_db:
            mock_default_db.return_value.create_session.return_value = owned_session
            chat_task, result = self._start_chat_task(
                user,
                config_provider,
                chat_provider,
                chat_uuid="chat-uuid-context",
                chat_query="hello",
                chat_context=context,
            )
            await chat_task.join_async()

        chat_provider.get_chat_context.assert_not_called()
        _, tool_kwargs = mock_create_tool_retriever.call_args
        assert tool_kwargs["tools"] is None
        _, run_kwargs = mock_agent.run_async.call_args
        assert run_kwargs["context"] == {
            "project": "alpha",
            "user_uuid": user.uuid,
            "chat_uuid": "chat-uuid-context",
            "session": owned_session,
        }
        assert run_kwargs["tool_ids"] == []
        assert run_kwargs["skill_retriever"] is mock_skill_retriever
        assert hasattr(result, "__aiter__")
        config_provider.get_model_repository.assert_called_with(
            user_uuid=user.uuid, session=owned_session
        )
        config_provider.get_agent_repository.assert_called_with(
            user_uuid=user.uuid, session=owned_session
        )

    @pytest.mark.asyncio
    @patch("fivccliche.utils.chats.create_skill_retriever_async")
    @patch("fivccliche.utils.chats.create_tool_retriever_async")
    @patch("fivccliche.utils.chats.create_agent_async")
    async def test_create_generator_uses_only_explicit_chat_tools(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        """Only explicitly supplied chat tools are passed to the tool retriever."""
        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock()
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        user = self._make_mock_user()
        config_provider = self._make_mock_config_provider()
        chat_provider = self._make_mock_chat_provider()
        external_primary = self._make_mock_tool("shared-tool")
        external_secondary = self._make_mock_tool("external-only")
        chat_context = Mock()
        chat_context.get_tools_async = AsyncMock(
            return_value=[self._make_mock_tool("context-only")]
        )
        chat_context.get_is_skills_enabled_async = AsyncMock(return_value=False)
        chat_provider.get_chat_context.return_value = chat_context

        with patch("fivccliche.utils.chats.default_db", new_callable=Mock) as mock_default_db:
            owned_session = AsyncMock()
            owned_session.close = AsyncMock()
            mock_default_db.return_value.create_session.return_value = owned_session
            chat_task, _ = self._start_chat_task(
                user,
                config_provider,
                chat_provider,
                chat_uuid="chat-uuid-tools",
                chat_query="hello",
                chat_tools=[external_primary, external_secondary],
                chat_context={"scope": "tools"},
            )
            await chat_task.join_async()

        chat_provider.get_chat_context.assert_not_called()
        chat_context.get_tools_async.assert_not_awaited()
        chat_context.get_is_skills_enabled_async.assert_not_awaited()
        _, tool_kwargs = mock_create_tool_retriever.call_args
        resolved_tools_by_name = {tool.name: tool for tool in tool_kwargs["tools"]}
        assert resolved_tools_by_name == {
            "shared-tool": external_primary,
            "external-only": external_secondary,
        }
        _, run_kwargs = mock_agent.run_async.call_args
        assert set(run_kwargs["tool_ids"]) == {"shared-tool", "external-only"}
        mock_create_skill_retriever.assert_called_once()

    @pytest.mark.asyncio
    @patch("fivccliche.utils.chats.create_skill_retriever_async")
    @patch("fivccliche.utils.chats.create_tool_retriever_async")
    @patch("fivccliche.utils.chats.create_agent_async")
    async def test_create_generator_uses_explicit_skill_setting(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        """Provider context skill settings do not override explicit generator settings."""
        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock()
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_skill_retriever = AsyncMock()
        mock_create_skill_retriever.return_value = mock_skill_retriever

        user = self._make_mock_user()
        config_provider = self._make_mock_config_provider()
        chat_provider = self._make_mock_chat_provider()
        chat_context = Mock()
        chat_context.get_tools_async = AsyncMock(return_value=[])
        chat_context.get_is_skills_enabled_async = AsyncMock(return_value=False)
        chat_provider.get_chat_context.return_value = chat_context

        with patch("fivccliche.utils.chats.default_db", new_callable=Mock) as mock_default_db:
            owned_session = AsyncMock()
            owned_session.close = AsyncMock()
            mock_default_db.return_value.create_session.return_value = owned_session
            chat_task, _ = self._start_chat_task(
                user,
                config_provider,
                chat_provider,
                chat_uuid="chat-uuid-no-skills",
                chat_query="hello",
                chat_context={"skills": "disabled"},
                chat_skills_enabled=True,
            )
            await chat_task.join_async()

        chat_provider.get_chat_context.assert_not_called()
        chat_context.get_tools_async.assert_not_awaited()
        chat_context.get_is_skills_enabled_async.assert_not_awaited()
        mock_create_skill_retriever.assert_called_once()
        _, run_kwargs = mock_agent.run_async.call_args
        assert run_kwargs["skill_retriever"] is mock_skill_retriever

    @pytest.mark.asyncio
    @patch("fivccliche.utils.chats.create_skill_retriever_async")
    @patch("fivccliche.utils.chats.create_tool_retriever_async")
    @patch("fivccliche.utils.chats.create_agent_async")
    async def test_create_generator_returns_streaming_generator(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        """get_stream returns an async generator."""
        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock()
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        user = self._make_mock_user()
        config_provider = self._make_mock_config_provider()
        chat_provider = self._make_mock_chat_provider()

        with patch("fivccliche.utils.chats.default_db", new_callable=Mock) as mock_default_db:
            owned_session = AsyncMock()
            owned_session.close = AsyncMock()
            mock_default_db.return_value.create_session.return_value = owned_session
            chat_task, result = self._start_chat_task(
                user,
                config_provider,
                chat_provider,
                chat_uuid="my-chat-uuid",
                chat_query="test query",
            )
            await chat_task.join_async()

        assert hasattr(result, "__aiter__")
        assert hasattr(result, "__aiter__")

    def _patch_owned_session(self):
        owned_session = AsyncMock()
        owned_session.close = AsyncMock()
        mock_db = Mock()
        mock_db.create_session.return_value = owned_session
        return owned_session, mock_db

    @staticmethod
    def _finish_run_mock():
        mock_run = Mock()
        mock_run.model_dump.return_value = {
            "id": "run-finish",
            "agent_id": "agent-1",
            "started_at": "2024-01-01T00:00:00",
            "completed_at": "2024-01-01T00:00:01",
            "query": "hello",
            "reply": "world",
            "tool_calls": [],
        }
        return mock_run

    @pytest.mark.asyncio
    @patch("fivccliche.utils.chats.default_db", new_callable=Mock)
    @patch("fivccliche.utils.chats.create_skill_retriever_async")
    @patch("fivccliche.utils.chats.create_tool_retriever_async")
    @patch("fivccliche.utils.chats.create_agent_async")
    async def test_finish_callback_called_once_on_normal_completion(
        self,
        mock_create_agent,
        mock_create_tool_retriever,
        mock_create_skill_retriever,
        mock_default_db,
    ):
        """Finish callback runs exactly once when the agent emits FINISH."""
        owned_session, mock_db = self._patch_owned_session()
        mock_default_db.return_value = mock_db
        finish_run = self._finish_run_mock()
        callback = Mock()

        async def run_async(**kwargs):
            kwargs["event_callback"](AgentRunEvent.FINISH, finish_run)

        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock(side_effect=run_async)
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        chat_task, result = self._start_chat_task(
            self._make_mock_user(),
            self._make_mock_config_provider(),
            self._make_mock_chat_provider(),
            chat_uuid="chat-callback-ok",
            chat_query="hello",
            chat_finish_callback=callback,
            chat_skills_enabled=False,
        )

        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        await chat_task.join_async()
        callback.assert_called_once_with(finish_run)
        owned_session.close.assert_awaited()
        assert any('"event": "finish"' in chunk for chunk in chunks)

    @pytest.mark.asyncio
    @patch("fivccliche.utils.chats.default_db", new_callable=Mock)
    @patch("fivccliche.utils.chats.create_skill_retriever_async")
    @patch("fivccliche.utils.chats.create_tool_retriever_async")
    @patch("fivccliche.utils.chats.create_agent_async")
    async def test_finish_callback_called_after_generator_aclose(
        self,
        mock_create_agent,
        mock_create_tool_retriever,
        mock_create_skill_retriever,
        mock_default_db,
    ):
        """Client disconnect (aclose) still invokes finish callback after FINISH."""
        owned_session, mock_db = self._patch_owned_session()
        mock_default_db.return_value = mock_db
        finish_run = self._finish_run_mock()
        callback = Mock()
        started = asyncio.Event()

        async def run_async(**kwargs):
            started.set()
            await asyncio.sleep(0.05)
            kwargs["event_callback"](AgentRunEvent.FINISH, finish_run)

        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock(side_effect=run_async)
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        chat_task, result = self._start_chat_task(
            self._make_mock_user(),
            self._make_mock_config_provider(),
            self._make_mock_chat_provider(),
            chat_uuid="chat-callback-disconnect",
            chat_query="hello",
            chat_finish_callback=callback,
            chat_skills_enabled=False,
        )

        agen = result
        await started.wait()
        await agen.aclose()

        await chat_task.join_async()
        callback.assert_called_once_with(finish_run)
        owned_session.close.assert_awaited()

    @pytest.mark.asyncio
    @patch("fivccliche.utils.chats.default_db", new_callable=Mock)
    @patch("fivccliche.utils.chats.create_skill_retriever_async")
    @patch("fivccliche.utils.chats.create_tool_retriever_async")
    @patch("fivccliche.utils.chats.create_agent_async")
    async def test_async_finish_callback_awaited(
        self,
        mock_create_agent,
        mock_create_tool_retriever,
        mock_create_skill_retriever,
        mock_default_db,
    ):
        """Async finish callbacks are awaited."""
        _, mock_db = self._patch_owned_session()
        mock_default_db.return_value = mock_db
        finish_run = self._finish_run_mock()
        callback = AsyncMock()

        async def run_async(**kwargs):
            kwargs["event_callback"](AgentRunEvent.FINISH, finish_run)

        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock(side_effect=run_async)
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        chat_task, result = self._start_chat_task(
            self._make_mock_user(),
            self._make_mock_config_provider(),
            self._make_mock_chat_provider(),
            chat_uuid="chat-callback-async",
            chat_query="hello",
            chat_finish_callback=callback,
            chat_skills_enabled=False,
        )

        async for _ in result:
            pass

        await chat_task.join_async()
        callback.assert_awaited_once_with(finish_run)

    @pytest.mark.asyncio
    @patch("fivccliche.utils.chats.default_db", new_callable=Mock)
    @patch("fivccliche.utils.chats.create_skill_retriever_async")
    @patch("fivccliche.utils.chats.create_tool_retriever_async")
    @patch("fivccliche.utils.chats.create_agent_async")
    async def test_finish_callback_not_called_without_finish_event(
        self,
        mock_create_agent,
        mock_create_tool_retriever,
        mock_create_skill_retriever,
        mock_default_db,
    ):
        """Finish callback is skipped when the agent run fails before FINISH."""
        _, mock_db = self._patch_owned_session()
        mock_default_db.return_value = mock_db
        callback = Mock()

        async def run_async(**kwargs):
            raise RuntimeError("agent failed")

        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock(side_effect=run_async)
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        chat_task, result = self._start_chat_task(
            self._make_mock_user(),
            self._make_mock_config_provider(),
            self._make_mock_chat_provider(),
            chat_uuid="chat-callback-error",
            chat_query="hello",
            chat_finish_callback=callback,
            chat_skills_enabled=False,
        )

        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        await chat_task.join_async()
        callback.assert_not_called()
        assert any('"event": "error"' in chunk for chunk in chunks)

    @pytest.mark.asyncio
    @patch("fivccliche.utils.chats.default_db", new_callable=Mock)
    @patch("fivccliche.utils.chats.create_skill_retriever_async")
    @patch("fivccliche.utils.chats.create_tool_retriever_async")
    @patch("fivccliche.utils.chats.create_agent_async")
    async def test_run_uses_owned_session_not_request_session(
        self,
        mock_create_agent,
        mock_create_tool_retriever,
        mock_create_skill_retriever,
        mock_default_db,
    ):
        """Agent run binds repositories/context to an owned session."""
        owned_session, mock_db = self._patch_owned_session()
        mock_default_db.return_value = mock_db

        async def run_async(**kwargs):
            kwargs["event_callback"](AgentRunEvent.FINISH, self._finish_run_mock())

        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock(side_effect=run_async)
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        user = self._make_mock_user()
        config_provider = self._make_mock_config_provider()
        chat_provider = self._make_mock_chat_provider()

        chat_task, _ = self._start_chat_task(
            user,
            config_provider,
            chat_provider,
            chat_uuid="chat-owned-session",
            chat_query="hello",
            chat_context={"project": "alpha"},
            chat_skills_enabled=False,
        )

        await chat_task.join_async()
        _, run_kwargs = mock_agent.run_async.call_args
        assert run_kwargs["context"]["session"] is owned_session
        assert run_kwargs["context"]["project"] == "alpha"
        chat_provider.get_chat_repository.assert_called_with(
            user_uuid=user.uuid, session=owned_session
        )
        config_provider.get_tool_repository.assert_any_call(
            user_uuid=user.uuid, session=owned_session
        )
        config_provider.get_model_repository.assert_called_with(
            user_uuid=user.uuid, session=owned_session
        )
        owned_session.close.assert_awaited()

    @pytest.mark.asyncio
    @patch("fivccliche.utils.chats.default_db", new_callable=Mock)
    @patch("fivccliche.utils.chats.create_skill_retriever_async")
    @patch("fivccliche.utils.chats.create_tool_retriever_async")
    @patch("fivccliche.utils.chats.create_agent_async")
    async def test_run_uses_injected_chat_db_when_provided(
        self,
        mock_create_agent,
        mock_create_tool_retriever,
        mock_create_skill_retriever,
        mock_default_db,
    ):
        """Explicit chat_db is used for owned session; default_db is not called."""
        owned_session, chat_db = self._patch_owned_session()

        async def run_async(**kwargs):
            kwargs["event_callback"](AgentRunEvent.FINISH, self._finish_run_mock())

        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock(side_effect=run_async)
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        user = self._make_mock_user()
        config_provider = self._make_mock_config_provider()
        chat_provider = self._make_mock_chat_provider()

        chat_task, _ = self._start_chat_task(
            user,
            config_provider,
            chat_provider,
            chat_uuid="chat-injected-db",
            chat_query="hello",
            chat_skills_enabled=False,
            chat_db=chat_db,
        )

        await chat_task.join_async()
        mock_default_db.assert_not_called()
        chat_db.create_session.assert_called_once_with()
        _, run_kwargs = mock_agent.run_async.call_args
        assert run_kwargs["context"]["session"] is owned_session
        owned_session.close.assert_awaited()

    @pytest.mark.asyncio
    @patch("fivccliche.utils.chats.default_db", new_callable=Mock)
    @patch("fivccliche.utils.chats.create_skill_retriever_async")
    @patch("fivccliche.utils.chats.create_tool_retriever_async")
    @patch("fivccliche.utils.chats.create_agent_async")
    async def test_mutex_released_after_normal_completion(
        self,
        mock_create_agent,
        mock_create_tool_retriever,
        mock_create_skill_retriever,
        mock_default_db,
    ):
        """Acquired mutex is released once when the chat task finishes."""
        _, mock_db = self._patch_owned_session()
        mock_default_db.return_value = mock_db
        mutex = Mock()
        mutex.release_async = AsyncMock()

        async def run_async(**kwargs):
            kwargs["event_callback"](AgentRunEvent.FINISH, self._finish_run_mock())

        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock(side_effect=run_async)
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        chat_task, result = self._start_chat_task(
            self._make_mock_user(),
            self._make_mock_config_provider(),
            self._make_mock_chat_provider(),
            chat_uuid="chat-mutex-ok",
            chat_query="hello",
            chat_skills_enabled=False,
            chat_mutex=mutex,
        )

        async for _ in result:
            pass

        await chat_task.join_async()
        mutex.release_async.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("fivccliche.utils.chats.default_db", new_callable=Mock)
    @patch("fivccliche.utils.chats.create_skill_retriever_async")
    @patch("fivccliche.utils.chats.create_tool_retriever_async")
    @patch("fivccliche.utils.chats.create_agent_async")
    async def test_mutex_released_after_generator_aclose(
        self,
        mock_create_agent,
        mock_create_tool_retriever,
        mock_create_skill_retriever,
        mock_default_db,
    ):
        """Mutex is still released when the SSE consumer disconnects early."""
        _, mock_db = self._patch_owned_session()
        mock_default_db.return_value = mock_db
        mutex = Mock()
        mutex.release_async = AsyncMock()
        started = asyncio.Event()

        async def run_async(**kwargs):
            started.set()
            await asyncio.sleep(0.05)
            kwargs["event_callback"](AgentRunEvent.FINISH, self._finish_run_mock())

        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock(side_effect=run_async)
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        chat_task, result = self._start_chat_task(
            self._make_mock_user(),
            self._make_mock_config_provider(),
            self._make_mock_chat_provider(),
            chat_uuid="chat-mutex-disconnect",
            chat_query="hello",
            chat_skills_enabled=False,
            chat_mutex=mutex,
        )

        agen = result
        await started.wait()
        await agen.aclose()
        await chat_task.join_async()
        mutex.release_async.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("fivccliche.utils.chats.default_db", new_callable=Mock)
    @patch("fivccliche.utils.chats.create_agent_async")
    async def test_mutex_released_when_agent_setup_fails(self, mock_create_agent, mock_default_db):
        """Mutex is released when agent setup fails inside the chat task."""
        owned_session, mock_db = self._patch_owned_session()
        mock_default_db.return_value = mock_db
        mock_create_agent.side_effect = RuntimeError("setup failed")
        mutex = Mock()
        mutex.release_async = AsyncMock()

        chat_task, _ = self._start_chat_task(
            self._make_mock_user(),
            self._make_mock_config_provider(),
            self._make_mock_chat_provider(),
            chat_uuid="chat-mutex-create-fail",
            chat_query="hello",
            chat_skills_enabled=False,
            chat_mutex=mutex,
        )

        await chat_task.join_async()
        mutex.release_async.assert_awaited_once()
        owned_session.close.assert_awaited()
