"""Unit tests for streaming generator utilities."""

import asyncio
import json
from contextlib import contextmanager
from json import JSONDecodeError
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import WebSocketDisconnect
from fivcplayground.agents import AgentRunEvent

from fivccliche.modules.agent_chats.jobs import ChatQueryJob
from fivccliche.utils.stream import ChatEventHandler, ChatStream, ChatWSHandler

_QUERY = "fivccliche.modules.agent_chats.jobs.query"


@contextmanager
def _patch_query_providers(config_provider, chat_provider):
    with (
        patch(
            f"{_QUERY}.get_config_provider_async",
            new_callable=AsyncMock,
            return_value=config_provider,
        ),
        patch(
            f"{_QUERY}.get_chat_provider_async",
            new_callable=AsyncMock,
            return_value=chat_provider,
        ),
    ):
        yield


class TestChatEventHandlerGetStream:
    """Test ChatEventHandler() SSE formatting."""

    def _make_chat_stream(self, chat_uuid: str | None = "test-chat-uuid"):
        chat_stream = ChatEventHandler(chat_uuid=chat_uuid)
        chat_stream._asyncio_task = MagicMock(spec=asyncio.Task)
        chat_stream._asyncio_task.done.return_value = False
        chat_stream._asyncio_task.result.return_value = None
        chat_stream._chat_queue = MagicMock()
        chat_stream._chat_queue.empty = Mock(return_value=False)
        chat_stream._chat_queue.task_done = Mock()
        return chat_stream

    def test_get_stream_returns_async_generator(self):
        """Calling ChatEventHandler returns an async generator."""
        chat_stream = self._make_chat_stream()
        stream = chat_stream()
        assert hasattr(stream, "__aiter__")

    def test_get_stream_without_chat_uuid(self):
        """Calling ChatEventHandler works when chat_uuid is None."""
        chat_stream = self._make_chat_stream(chat_uuid=None)
        stream = chat_stream()
        assert hasattr(stream, "__aiter__")

    def test_chat_stream_alias_is_chat_event_handler(self):
        """ChatStream remains a compatible alias of ChatEventHandler."""
        assert ChatStream is ChatEventHandler

    @pytest.mark.asyncio
    async def test_events_and_sse_chunks_use_the_same_payload(self):
        """The WebSocket event payload and SSE chunk contain the same object."""
        chat_stream = self._make_chat_stream()
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
            if chat_stream._asyncio_task.done.return_value:
                raise TimeoutError()
            chat_stream._asyncio_task.done.return_value = True
            return (AgentRunEvent.START, mock_run)

        chat_stream._chat_queue.empty.return_value = True
        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            events = [event async for event in chat_stream.events()]

        async def repeated_events():
            yield events[0]

        chat_stream.events = repeated_events
        chunks = [chunk async for chunk in chat_stream()]

        assert len(events) == 1
        assert len(chunks) == 1
        assert json.loads(chunks[0].removeprefix("data: ").strip()) == events[0]

    @pytest.mark.asyncio
    async def test_start_event_formatting(self):
        """Test START event is formatted correctly."""
        chat_stream = self._make_chat_stream()
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
            if chat_stream._asyncio_task.done.return_value:
                raise TimeoutError()
            chat_stream._asyncio_task.done.return_value = True
            return (AgentRunEvent.START, mock_run)

        chat_stream._chat_queue.empty.return_value = True
        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in chat_stream():
                results.append(chunk)

        assert len(results) == 1
        data = json.loads(results[0].replace("data: ", "").strip())
        assert data["event"] == "start"
        assert data["info"]["chat_uuid"] == "test-chat-uuid"
        assert data["info"]["id"] == "run-1"

    @pytest.mark.asyncio
    async def test_finish_event_formatting(self):
        """Test FINISH event is formatted correctly."""
        chat_stream = self._make_chat_stream()
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
            if chat_stream._asyncio_task.done.return_value:
                raise TimeoutError()
            chat_stream._asyncio_task.done.return_value = True
            return (AgentRunEvent.FINISH, mock_run)

        chat_stream._chat_queue.empty.return_value = True
        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in chat_stream():
                results.append(chunk)

        assert len(results) == 1
        data = json.loads(results[0].replace("data: ", "").strip())
        assert data["event"] == "finish"
        assert data["info"]["chat_uuid"] == "test-chat-uuid"
        assert data["info"]["reply"] == "test reply"

    @pytest.mark.asyncio
    async def test_stream_event_with_delta(self):
        """Test STREAM event with delta is formatted correctly."""
        chat_stream = self._make_chat_stream()
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
            if chat_stream._asyncio_task.done.return_value:
                raise TimeoutError()
            chat_stream._asyncio_task.done.return_value = True
            return (AgentRunEvent.STREAM, mock_run)

        chat_stream._chat_queue.empty.return_value = True
        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in chat_stream():
                results.append(chunk)

        assert len(results) == 1
        data = json.loads(results[0].replace("data: ", "").strip())
        assert data["event"] == "stream"
        assert data["info"]["chat_uuid"] == "test-chat-uuid"
        assert data["info"]["delta"] == {"content": "partial text"}

    @pytest.mark.asyncio
    async def test_stream_event_without_delta(self):
        """Test STREAM event without delta is formatted correctly."""
        chat_stream = self._make_chat_stream()
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
            if chat_stream._asyncio_task.done.return_value:
                raise TimeoutError()
            chat_stream._asyncio_task.done.return_value = True
            return (AgentRunEvent.STREAM, mock_run)

        chat_stream._chat_queue.empty.return_value = True
        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in chat_stream():
                results.append(chunk)

        assert len(results) == 1
        data = json.loads(results[0].replace("data: ", "").strip())
        assert data["event"] == "stream"
        assert data["info"]["delta"] is None

    @pytest.mark.asyncio
    async def test_tool_event_formatting(self):
        """Test TOOL event is formatted correctly."""
        chat_stream = self._make_chat_stream()
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
            if chat_stream._asyncio_task.done.return_value:
                raise TimeoutError()
            chat_stream._asyncio_task.done.return_value = True
            return (AgentRunEvent.TOOL, mock_run)

        chat_stream._chat_queue.empty.return_value = True
        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in chat_stream():
                results.append(chunk)

        assert len(results) == 1
        data = json.loads(results[0].replace("data: ", "").strip())
        assert data["event"] == "tool"
        assert data["info"]["chat_uuid"] == "test-chat-uuid"
        assert len(data["info"]["tool_calls"]) == 1

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error event is generated on exception."""
        chat_stream = self._make_chat_stream()

        async def mock_wait_for(coro, timeout):
            raise ValueError("Test error")

        chat_stream._chat_queue.empty.return_value = False
        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in chat_stream():
                results.append(chunk)

        assert len(results) == 1
        data = json.loads(results[0].replace("data: ", "").strip())
        assert data["event"] == "error"
        assert "Test error" in data["info"]["message"]

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout behavior when queue is empty."""
        chat_stream = self._make_chat_stream()
        call_count = 0

        async def mock_wait_for(coro, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError()
            chat_stream._asyncio_task.done.return_value = True
            raise TimeoutError()

        chat_stream._chat_queue.empty.return_value = True
        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in chat_stream():
                results.append(chunk)

        assert len(results) == 0
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_chat_uuid_added_to_all_events(self):
        """Test chat_uuid is added to all event types."""
        chat_stream = self._make_chat_stream()
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
                chat_stream._asyncio_task.done.return_value = True
                raise TimeoutError()
            event = events[event_index]
            event_index += 1
            return event

        def mock_empty():
            return event_index >= len(events)

        chat_stream._chat_queue.empty.side_effect = mock_empty
        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = []
            async for chunk in chat_stream():
                results.append(chunk)

        assert len(results) == 4
        for result in results:
            data = json.loads(result.replace("data: ", "").strip())
            assert data["info"]["chat_uuid"] == "test-chat-uuid"

    @pytest.mark.asyncio
    async def test_task_done_and_queue_empty_exits(self):
        """Test generator exits when task is done and queue is empty."""
        chat_stream = self._make_chat_stream()
        chat_stream._asyncio_task.done.return_value = True
        chat_stream._chat_queue.empty.return_value = True

        results = []
        async for chunk in chat_stream():
            results.append(chunk)

        assert len(results) == 0
        chat_stream._asyncio_task.result.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_uses_attach_and_on_event(self):
        """attach + on_event + __call__ complete a stream without private fields."""
        chat_stream = ChatEventHandler(chat_uuid="public-api")
        mock_run = Mock()
        mock_run.model_dump.return_value = {
            "id": "run-1",
            "agent_id": "agent-1",
            "started_at": "2024-01-01T00:00:00",
            "completed_at": None,
            "query": "hello",
            "reply": None,
            "tool_calls": [],
        }

        async def produce():
            chat_stream.on_event(AgentRunEvent.START, mock_run)
            chat_stream.on_event(AgentRunEvent.FINISH, mock_run)

        query_task = asyncio.create_task(produce())
        chat_stream.attach(query_task)
        chunks = [chunk async for chunk in chat_stream()]
        await query_task

        events = [json.loads(chunk.replace("data: ", "").strip())["event"] for chunk in chunks]
        assert events == ["start", "finish"]


class TestChatQueryJob:
    """Test ChatQueryJob agent execution and ChatEventHandler integration."""

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

        def _get_chat_context(user_uuid, context=None, **kwargs):
            return {**(context or {}), "user_uuid": user_uuid, **kwargs}

        provider.get_chat_context.side_effect = _get_chat_context
        return provider

    @staticmethod
    def _make_mock_tool(name: str):
        tool = Mock()
        tool.name = name
        return tool

    def _start_query(self, user, **kwargs):
        """Create ChatQueryJob task, attach ChatEventHandler, return (stream, task, agen)."""
        chat_uuid = kwargs.pop("chat_uuid")
        query = kwargs.pop("query")
        chat_stream = ChatEventHandler(chat_uuid=chat_uuid)
        query_task = asyncio.create_task(
            ChatQueryJob(MagicMock()).run_async(
                chat_uuid,
                user_uuid=user.uuid,
                query=query,
                event_callback=chat_stream.on_event,
                **kwargs,
            )
        )
        chat_stream.attach(query_task)
        return chat_stream, query_task, chat_stream()

    @staticmethod
    async def _join_query(query_task):
        await asyncio.gather(query_task, return_exceptions=True)

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

    def test_name_and_config(self):
        job = ChatQueryJob(MagicMock())
        assert job.name == "agent-chats-query"
        assert job.config is None

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_skill_retriever_async")
    @patch(f"{_QUERY}.create_tool_retriever_async")
    @patch(f"{_QUERY}.create_agent_async")
    async def test_create_generator_skills_disabled_by_default(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        """create_skill_retriever_async not called when skills_enabled=False."""
        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock()
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        user = self._make_mock_user()
        config_provider = self._make_mock_config_provider()
        chat_provider = self._make_mock_chat_provider()

        with _patch_query_providers(config_provider, chat_provider):
            _, query_task, result = self._start_query(
                user,
                chat_uuid="chat-uuid-1",
                query="hello",
                skills_enabled=False,
            )
            await self._join_query(query_task)

        mock_create_skill_retriever.assert_not_called()
        chat_provider.get_chat_context.assert_called_once_with(
            user_uuid=user.uuid,
            context=None,
            chat_uuid="chat-uuid-1",
        )
        _, kwargs = mock_agent.run_async.call_args
        assert kwargs["skill_retriever"] is None
        assert hasattr(result, "__aiter__")

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_skill_retriever_async")
    @patch(f"{_QUERY}.create_tool_retriever_async")
    @patch(f"{_QUERY}.create_agent_async")
    async def test_create_generator_skills_enabled(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        """create_skill_retriever_async is called when skills_enabled=True."""
        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock()
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_skill_retriever = AsyncMock()
        mock_create_skill_retriever.return_value = mock_skill_retriever

        user = self._make_mock_user()
        config_provider = self._make_mock_config_provider()
        chat_provider = self._make_mock_chat_provider()

        with _patch_query_providers(config_provider, chat_provider):
            _, query_task, result = self._start_query(
                user,
                chat_uuid="chat-uuid-2",
                query="hello",
                skills_enabled=True,
            )
            await self._join_query(query_task)

        mock_create_skill_retriever.assert_called_once()
        chat_provider.get_chat_context.assert_called_once_with(
            user_uuid=user.uuid,
            context=None,
            chat_uuid="chat-uuid-2",
        )
        _, kwargs = mock_agent.run_async.call_args
        assert kwargs["skill_retriever"] is mock_skill_retriever
        assert hasattr(result, "__aiter__")

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_skill_retriever_async")
    @patch(f"{_QUERY}.create_tool_retriever_async")
    @patch(f"{_QUERY}.create_agent_async")
    async def test_create_generator_resolves_chat_context_via_provider(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        """ChatQueryJob passes the provider context dict through to the agent run."""
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

        with _patch_query_providers(config_provider, chat_provider):
            _, query_task, result = self._start_query(
                user,
                chat_uuid="chat-uuid-context",
                query="hello",
                context=context,
            )
            await self._join_query(query_task)

        chat_provider.get_chat_context.assert_called_once_with(
            user_uuid=user.uuid,
            context=context,
            chat_uuid="chat-uuid-context",
        )
        _, tool_kwargs = mock_create_tool_retriever.call_args
        assert tool_kwargs["tools"] is None
        _, run_kwargs = mock_agent.run_async.call_args
        assert run_kwargs["context"] == {
            "project": "alpha",
            "user_uuid": user.uuid,
            "chat_uuid": "chat-uuid-context",
        }
        assert run_kwargs["tool_ids"] == []
        assert run_kwargs["skill_retriever"] is mock_skill_retriever
        assert hasattr(result, "__aiter__")
        config_provider.get_model_repository.assert_called_with(user_uuid=user.uuid)
        config_provider.get_agent_repository.assert_called_with(user_uuid=user.uuid)

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_skill_retriever_async")
    @patch(f"{_QUERY}.create_tool_retriever_async")
    @patch(f"{_QUERY}.create_agent_async")
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

        with _patch_query_providers(config_provider, chat_provider):
            _, query_task, _ = self._start_query(
                user,
                chat_uuid="chat-uuid-tools",
                query="hello",
                tools=[external_primary, external_secondary],
                context={"scope": "tools"},
            )
            await self._join_query(query_task)

        chat_provider.get_chat_context.assert_called_once_with(
            user_uuid=user.uuid,
            context={"scope": "tools"},
            chat_uuid="chat-uuid-tools",
        )
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
    @patch(f"{_QUERY}.create_skill_retriever_async")
    @patch(f"{_QUERY}.create_tool_retriever_async")
    @patch(f"{_QUERY}.create_agent_async")
    async def test_duplicate_tool_names_keep_last(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        """Tools with the same name collapse to the last instance."""
        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock()
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        first = self._make_mock_tool("shared-tool")
        last = self._make_mock_tool("shared-tool")

        with _patch_query_providers(
            self._make_mock_config_provider(), self._make_mock_chat_provider()
        ):
            _, query_task, _ = self._start_query(
                self._make_mock_user(),
                chat_uuid="chat-uuid-dedup",
                query="hello",
                tools=[first, last],
                skills_enabled=False,
            )
            await self._join_query(query_task)

        _, tool_kwargs = mock_create_tool_retriever.call_args
        assert tool_kwargs["tools"] == [last]
        _, run_kwargs = mock_agent.run_async.call_args
        assert run_kwargs["tool_ids"] == ["shared-tool"]

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_skill_retriever_async")
    @patch(f"{_QUERY}.create_tool_retriever_async")
    @patch(f"{_QUERY}.create_agent_async")
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

        with _patch_query_providers(config_provider, chat_provider):
            _, query_task, _ = self._start_query(
                user,
                chat_uuid="chat-uuid-no-skills",
                query="hello",
                context={"skills": "disabled"},
                skills_enabled=True,
            )
            await self._join_query(query_task)

        chat_provider.get_chat_context.assert_called_once_with(
            user_uuid=user.uuid,
            context={"skills": "disabled"},
            chat_uuid="chat-uuid-no-skills",
        )
        mock_create_skill_retriever.assert_called_once()
        _, run_kwargs = mock_agent.run_async.call_args
        assert run_kwargs["skill_retriever"] is mock_skill_retriever

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_skill_retriever_async")
    @patch(f"{_QUERY}.create_tool_retriever_async")
    @patch(f"{_QUERY}.create_agent_async")
    async def test_create_generator_returns_streaming_generator(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        """Calling ChatEventHandler returns an async generator."""
        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock()
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        user = self._make_mock_user()
        config_provider = self._make_mock_config_provider()
        chat_provider = self._make_mock_chat_provider()

        with _patch_query_providers(config_provider, chat_provider):
            _, query_task, result = self._start_query(
                user,
                chat_uuid="my-chat-uuid",
                query="test query",
            )
            await self._join_query(query_task)

        assert hasattr(result, "__aiter__")

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_skill_retriever_async")
    @patch(f"{_QUERY}.create_tool_retriever_async")
    @patch(f"{_QUERY}.create_agent_async")
    async def test_finish_callback_called_once_on_normal_completion(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        """Finish callback runs exactly once when the agent emits FINISH."""
        finish_run = self._finish_run_mock()
        callback = Mock()

        async def run_async(**kwargs):
            kwargs["event_callback"](AgentRunEvent.FINISH, finish_run)

        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock(side_effect=run_async)
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        with _patch_query_providers(
            self._make_mock_config_provider(), self._make_mock_chat_provider()
        ):
            _, query_task, result = self._start_query(
                self._make_mock_user(),
                chat_uuid="chat-callback-ok",
                query="hello",
                finish_callback=callback,
                skills_enabled=False,
            )

            chunks = []
            async for chunk in result:
                chunks.append(chunk)

            await self._join_query(query_task)

        callback.assert_called_once_with(finish_run)
        assert any('"event": "finish"' in chunk for chunk in chunks)

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_skill_retriever_async")
    @patch(f"{_QUERY}.create_tool_retriever_async")
    @patch(f"{_QUERY}.create_agent_async")
    async def test_finish_callback_called_after_generator_aclose(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        """Client disconnect (aclose) still invokes finish callback after FINISH."""
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

        with _patch_query_providers(
            self._make_mock_config_provider(), self._make_mock_chat_provider()
        ):
            _, query_task, result = self._start_query(
                self._make_mock_user(),
                chat_uuid="chat-callback-disconnect",
                query="hello",
                finish_callback=callback,
                skills_enabled=False,
            )

            agen = result
            await started.wait()
            await agen.aclose()

            await self._join_query(query_task)

        callback.assert_called_once_with(finish_run)

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_skill_retriever_async")
    @patch(f"{_QUERY}.create_tool_retriever_async")
    @patch(f"{_QUERY}.create_agent_async")
    async def test_async_finish_callback_awaited(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        """Async finish callbacks are awaited."""
        finish_run = self._finish_run_mock()
        callback = AsyncMock()

        async def run_async(**kwargs):
            kwargs["event_callback"](AgentRunEvent.FINISH, finish_run)

        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock(side_effect=run_async)
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        with _patch_query_providers(
            self._make_mock_config_provider(), self._make_mock_chat_provider()
        ):
            _, query_task, result = self._start_query(
                self._make_mock_user(),
                chat_uuid="chat-callback-async",
                query="hello",
                finish_callback=callback,
                skills_enabled=False,
            )

            async for _ in result:
                pass

            await self._join_query(query_task)

        callback.assert_awaited_once_with(finish_run)

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_skill_retriever_async")
    @patch(f"{_QUERY}.create_tool_retriever_async")
    @patch(f"{_QUERY}.create_agent_async")
    async def test_finish_callback_not_called_without_finish_event(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        """Finish callback is skipped when the agent run fails before FINISH."""
        callback = Mock()

        async def run_async(**kwargs):
            raise RuntimeError("agent failed")

        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock(side_effect=run_async)
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        with _patch_query_providers(
            self._make_mock_config_provider(), self._make_mock_chat_provider()
        ):
            _, query_task, result = self._start_query(
                self._make_mock_user(),
                chat_uuid="chat-callback-error",
                query="hello",
                finish_callback=callback,
                skills_enabled=False,
            )

            chunks = []
            async for chunk in result:
                chunks.append(chunk)

            await self._join_query(query_task)

        callback.assert_not_called()
        assert any('"event": "error"' in chunk for chunk in chunks)

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_skill_retriever_async")
    @patch(f"{_QUERY}.create_tool_retriever_async")
    @patch(f"{_QUERY}.create_agent_async")
    async def test_run_does_not_create_owned_session(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        """Agent run does not bind a long-lived DB session into repositories or context."""

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

        with _patch_query_providers(config_provider, chat_provider):
            _, query_task, _ = self._start_query(
                user,
                chat_uuid="chat-no-owned-session",
                query="hello",
                context={"project": "alpha"},
                skills_enabled=False,
            )

            await self._join_query(query_task)

        _, run_kwargs = mock_agent.run_async.call_args
        assert "session" not in run_kwargs["context"]
        assert run_kwargs["context"]["project"] == "alpha"
        chat_provider.get_chat_repository.assert_called_with(user_uuid=user.uuid)
        config_provider.get_tool_repository.assert_any_call(user_uuid=user.uuid)
        config_provider.get_model_repository.assert_called_with(user_uuid=user.uuid)

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_skill_retriever_async")
    @patch(f"{_QUERY}.create_tool_retriever_async")
    @patch(f"{_QUERY}.create_agent_async")
    async def test_mutex_released_after_normal_completion(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        """Acquired mutex is released once when the chat task finishes."""
        mutex = Mock()
        mutex.release_async = AsyncMock()

        async def run_async(**kwargs):
            kwargs["event_callback"](AgentRunEvent.FINISH, self._finish_run_mock())

        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock(side_effect=run_async)
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        with _patch_query_providers(
            self._make_mock_config_provider(), self._make_mock_chat_provider()
        ):
            _, query_task, result = self._start_query(
                self._make_mock_user(),
                chat_uuid="chat-mutex-ok",
                query="hello",
                skills_enabled=False,
                chat_mutex=mutex,
            )

            async for _ in result:
                pass

            await self._join_query(query_task)

        mutex.release_async.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_skill_retriever_async")
    @patch(f"{_QUERY}.create_tool_retriever_async")
    @patch(f"{_QUERY}.create_agent_async")
    async def test_mutex_released_after_generator_aclose(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        """Mutex is still released when the SSE consumer disconnects early."""
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

        with _patch_query_providers(
            self._make_mock_config_provider(), self._make_mock_chat_provider()
        ):
            _, query_task, result = self._start_query(
                self._make_mock_user(),
                chat_uuid="chat-mutex-disconnect",
                query="hello",
                skills_enabled=False,
                chat_mutex=mutex,
            )

            agen = result
            await started.wait()
            await agen.aclose()
            await self._join_query(query_task)

        mutex.release_async.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_agent_async")
    async def test_mutex_released_when_agent_setup_fails(self, mock_create_agent):
        """Mutex is released when agent setup fails inside the query job."""
        mock_create_agent.side_effect = RuntimeError("setup failed")
        mutex = Mock()
        mutex.release_async = AsyncMock()

        with _patch_query_providers(
            self._make_mock_config_provider(), self._make_mock_chat_provider()
        ):
            _, query_task, _ = self._start_query(
                self._make_mock_user(),
                chat_uuid="chat-mutex-create-fail",
                query="hello",
                skills_enabled=False,
                chat_mutex=mutex,
            )

            await self._join_query(query_task)

        mutex.release_async.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_skill_retriever_async")
    @patch(f"{_QUERY}.create_tool_retriever_async")
    @patch(f"{_QUERY}.create_agent_async")
    async def test_mutex_released_when_run_times_out(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        """Run timeout cancels the agent and still releases the mutex."""
        mutex = Mock()
        mutex.release_async = AsyncMock()
        finished = False

        async def run_async(**kwargs):
            nonlocal finished
            await asyncio.sleep(1)
            finished = True

        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock(side_effect=run_async)
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        started = asyncio.get_running_loop().time()
        with _patch_query_providers(
            self._make_mock_config_provider(), self._make_mock_chat_provider()
        ):
            _, query_task, _ = self._start_query(
                self._make_mock_user(),
                chat_uuid="chat-mutex-timeout",
                query="hello",
                skills_enabled=False,
                chat_mutex=mutex,
                run_timeout=0.05,
            )

            await self._join_query(query_task)
        elapsed = asyncio.get_running_loop().time() - started

        mutex.release_async.assert_awaited_once()
        assert finished is False
        assert elapsed < 0.5

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_skill_retriever_async")
    @patch(f"{_QUERY}.create_tool_retriever_async")
    @patch(f"{_QUERY}.create_agent_async")
    async def test_stream_emits_error_on_run_timeout(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        """Connected SSE clients receive an error event when the run times out."""

        async def run_async(**kwargs):
            await asyncio.sleep(1)

        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock(side_effect=run_async)
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        with _patch_query_providers(
            self._make_mock_config_provider(), self._make_mock_chat_provider()
        ):
            _, query_task, stream = self._start_query(
                self._make_mock_user(),
                chat_uuid="chat-stream-timeout",
                query="hello",
                skills_enabled=False,
                run_timeout=0.05,
            )

            chunks = [chunk async for chunk in stream]
            await self._join_query(query_task)

        payload = json.loads(chunks[-1].removeprefix("data: ").strip())
        assert payload["event"] == "error"
        assert "timed out" in payload["info"]["message"].lower()

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_skill_retriever_async")
    @patch(f"{_QUERY}.create_tool_retriever_async")
    @patch(f"{_QUERY}.create_agent_async")
    async def test_mutex_released_after_normal_completion_within_timeout(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        """A generous run timeout does not fire when the agent finishes promptly."""
        mutex = Mock()
        mutex.release_async = AsyncMock()

        async def run_async(**kwargs):
            kwargs["event_callback"](AgentRunEvent.FINISH, self._finish_run_mock())

        mock_agent = AsyncMock()
        mock_agent.run_async = AsyncMock(side_effect=run_async)
        mock_create_agent.return_value = mock_agent
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        with _patch_query_providers(
            self._make_mock_config_provider(), self._make_mock_chat_provider()
        ):
            _, query_task, result = self._start_query(
                self._make_mock_user(),
                chat_uuid="chat-mutex-timeout-ok",
                query="hello",
                skills_enabled=False,
                chat_mutex=mutex,
                run_timeout=1.0,
            )

            chunks = [chunk async for chunk in result]
            await self._join_query(query_task)

        mutex.release_async.assert_awaited_once()
        assert any('"event": "finish"' in chunk for chunk in chunks)


class TestChatWSHandler:
    """Test first-frame auth, typed receive, error close, and event send."""

    @staticmethod
    def _fake_websocket(frames=None, *, fail_on_send=0, hang_on_empty=False):
        class FakeWebSocket:
            def __init__(self):
                self.frames = list(frames or [])
                self.sent = []
                self.closed = []
                self.accepted = False

            async def accept(self):
                self.accepted = True

            async def receive_json(self):
                if not self.frames:
                    if hang_on_empty:
                        await asyncio.sleep(10)
                    raise WebSocketDisconnect
                frame = self.frames.pop(0)
                if isinstance(frame, Exception):
                    raise frame
                return frame

            async def send_json(self, message):
                self.sent.append(message)
                if fail_on_send and len(self.sent) >= fail_on_send:
                    raise WebSocketDisconnect

            async def close(self, code=1000):
                self.closed.append(code)

        return FakeWebSocket()

    @pytest.mark.asyncio
    async def test_authenticate_times_out_without_auth_frame(self):
        websocket = self._fake_websocket(hang_on_empty=True)
        auth = AsyncMock()
        chat_ws = ChatWSHandler(websocket, auth_timeout=0.01)

        user = await chat_ws.authenticate(auth)

        assert user is None
        auth.verify_credential_async.assert_not_called()
        assert websocket.accepted is True
        assert websocket.sent == [
            {
                "event": "error",
                "info": {"code": "unauthorized", "message": "Authentication timed out"},
            }
        ]
        assert websocket.closed == [1008]

    @pytest.mark.asyncio
    async def test_authenticate_rejects_invalid_token(self):
        websocket = self._fake_websocket([{"type": "auth", "access_token": "invalid-token"}])
        auth = MagicMock()
        auth.verify_credential_async = AsyncMock(return_value=None)

        user = await ChatWSHandler(websocket).authenticate(auth)

        assert user is None
        auth.verify_credential_async.assert_awaited_once_with("invalid-token")
        assert websocket.sent == [
            {"event": "error", "info": {"code": "unauthorized", "message": "Invalid token"}}
        ]
        assert websocket.closed == [1008]

    @pytest.mark.asyncio
    async def test_authenticate_rejects_non_json_frame(self):
        websocket = self._fake_websocket([JSONDecodeError("Invalid JSON", "", 0)])
        auth = AsyncMock()

        user = await ChatWSHandler(websocket).authenticate(auth)

        assert user is None
        auth.verify_credential_async.assert_not_called()
        assert websocket.sent == [
            {"event": "error", "info": {"code": "invalid_frame", "message": "Invalid JSON frame"}}
        ]
        assert websocket.closed == [1003]

    @pytest.mark.asyncio
    async def test_authenticate_rejects_value_error_as_invalid_frame(self):
        websocket = self._fake_websocket([ValueError("invalid json")])
        auth = AsyncMock()

        user = await ChatWSHandler(websocket).authenticate(auth)

        assert user is None
        assert websocket.sent[-1]["info"]["code"] == "invalid_frame"
        assert websocket.closed == [1003]

    @pytest.mark.asyncio
    async def test_authenticate_returns_verified_user(self):
        websocket = self._fake_websocket([{"type": "auth", "access_token": "valid-token"}])
        verified = MagicMock()
        auth = MagicMock()
        auth.verify_credential_async = AsyncMock(return_value=verified)

        user = await ChatWSHandler(websocket).authenticate(auth)

        assert user is verified
        assert websocket.accepted is True
        assert websocket.sent == []
        assert websocket.closed == []

    @pytest.mark.asyncio
    async def test_receive_returns_typed_frame(self):
        websocket = self._fake_websocket([{"type": "message", "query": "hello"}])
        chat_ws = ChatWSHandler(websocket)
        await websocket.accept()

        frame = await chat_ws.receive(expected_type="message", invalid_code="invalid_message")

        assert frame == {"type": "message", "query": "hello"}
        assert websocket.closed == []

    @pytest.mark.asyncio
    async def test_receive_rejects_wrong_type(self):
        websocket = self._fake_websocket([{"type": "report"}])
        chat_ws = ChatWSHandler(websocket)
        await websocket.accept()

        frame = await chat_ws.receive(expected_type="message", invalid_code="invalid_message")

        assert frame is None
        assert websocket.sent == [
            {
                "event": "error",
                "info": {
                    "code": "invalid_message",
                    "message": "A message frame is required",
                },
            }
        ]
        assert websocket.closed == [1003]

    @pytest.mark.asyncio
    async def test_receive_closes_normally_on_disconnect(self):
        websocket = self._fake_websocket()
        chat_ws = ChatWSHandler(websocket)
        await websocket.accept()

        frame = await chat_ws.receive(expected_type="message", invalid_code="invalid_message")

        assert frame is None
        assert websocket.sent == []
        assert websocket.closed == [1000]

    @pytest.mark.asyncio
    async def test_fail_sends_error_envelope_and_closes(self):
        websocket = self._fake_websocket()
        chat_ws = ChatWSHandler(websocket)

        await chat_ws.fail(code="chat_not_found", message="Chat not found", close_code=4004)

        assert websocket.sent == [
            {"event": "error", "info": {"code": "chat_not_found", "message": "Chat not found"}}
        ]
        assert websocket.closed == [4004]

    @pytest.mark.asyncio
    async def test_send_pushes_events_and_returns_false(self):
        websocket = self._fake_websocket()
        events = [
            {"event": "start", "info": {"chat_uuid": "chat-1"}},
            {"event": "finish", "info": {"chat_uuid": "chat-1"}},
        ]

        async def event_generator():
            for event in events:
                yield event

        disconnected = await ChatWSHandler(websocket).send(event_generator())

        assert disconnected is False
        assert websocket.sent == events
        assert websocket.closed == []

    @pytest.mark.asyncio
    async def test_send_returns_true_when_client_disconnects(self):
        websocket = self._fake_websocket(fail_on_send=1)
        events = [
            {"event": "start", "info": {}},
            {"event": "finish", "info": {}},
        ]

        async def event_generator():
            for event in events:
                yield event

        disconnected = await ChatWSHandler(websocket).send(event_generator())

        assert disconnected is True
        assert websocket.sent == [events[0]]
        assert websocket.closed == []
