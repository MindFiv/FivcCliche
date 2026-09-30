"""Unit tests for streaming generator utilities."""

import asyncio
import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fivcplayground.agents import AgentRunEvent
from fivccliche.services.interfaces.agent_speeches import SpeechEvent, SpeechRequestError

from datetime import timedelta

from fivccliche.modules.agent_chats import UserChatRunProviderImpl
from fivccliche.utils.chats import (
    ChatRecognizeParser,
    ChatRunParser,
)
from fivccliche.utils.chats.parsers import _chat_voice_clean

_QUERY = "fivccliche.modules.agent_chats.services"


@contextmanager
def _patch_query_providers(config_provider, chat_provider):
    with (
        patch(
            f"{_QUERY}.query_component",
            side_effect=[config_provider, chat_provider],
        ),
    ):
        yield


class TestChatRunParser:
    """Test direct ChatRunParser stream consumption and output formatting."""

    def _make_chat_stream(
        self,
        chat_uuid: str | None = "test-chat-uuid",
        *,
        output_type: str = "raw",
    ):
        return ChatRunParser(chat_uuid=chat_uuid, output_type=output_type)

    @staticmethod
    def _make_run(
        *,
        reply=None,
        completed_at=None,
        tool_calls=None,
        delta=None,
    ):
        run = Mock()
        run_data = {
            "id": "run-1",
            "agent_id": "agent-1",
            "started_at": "2024-01-01T00:00:00",
            "completed_at": completed_at,
            "query": "test query",
            "reply": reply,
            "tool_calls": tool_calls if tool_calls is not None else [],
        }
        run.model_dump.side_effect = lambda mode, include: {
            key: value for key, value in run_data.items() if key in include
        }
        run.delta = delta
        return run

    @staticmethod
    def _source(events, *, error=None, hang_event=None):
        async def produce():
            for event in events:
                yield event
            if hang_event is not None:
                hang_event.set()
                await asyncio.sleep(10)
            if error is not None:
                raise error

        return produce()

    @staticmethod
    async def _collect(handler, source):
        return [payload async for payload in handler.parse_async(source)]

    @pytest.mark.parametrize("output_type", ["raw", "sse"])
    async def test_output_type_formats_one_source_stream(self, output_type):
        """All transports use one normalized copy of the source event stream."""
        start = self._make_run()
        stream_one_delta = Mock()
        stream_one_delta.model_dump.return_value = {"text": "Hello "}
        stream_two_delta = Mock()
        stream_two_delta.model_dump.return_value = {"text": "world"}
        stream_one = self._make_run(delta=stream_one_delta)
        stream_two = self._make_run(delta=stream_two_delta)
        finish = self._make_run(
            completed_at="2024-01-01T00:01:00",
            reply="done",
            tool_calls=[{"name": "search"}],
        )
        source = self._source(
            [
                (AgentRunEvent.START, start),
                (AgentRunEvent.STREAM, stream_one),
                (Mock(), start),
                (AgentRunEvent.TOOL, start),
                (AgentRunEvent.STREAM, stream_two),
                (AgentRunEvent.FINISH, finish),
            ]
        )

        results = await self._collect(self._make_chat_stream(output_type=output_type), source)

        basic_info = {
            "id": "run-1",
            "agent_id": "agent-1",
            "started_at": "2024-01-01T00:00:00",
            "completed_at": None,
        }
        full_info = {
            **basic_info,
            "query": "test query",
            "reply": None,
            "tool_calls": [],
        }
        raw_results = [
            {
                "event": "start",
                "info": {**full_info, "chat_uuid": "test-chat-uuid"},
            },
            {
                "event": "stream",
                "info": {
                    "id": "run-1",
                    "agent_id": "agent-1",
                    "started_at": "2024-01-01T00:00:00",
                    "completed_at": None,
                    "chat_uuid": "test-chat-uuid",
                    "delta": {"text": "Hello "},
                },
            },
            {
                "event": "tool",
                "info": {**full_info, "chat_uuid": "test-chat-uuid"},
            },
            {
                "event": "stream",
                "info": {
                    "id": "run-1",
                    "agent_id": "agent-1",
                    "started_at": "2024-01-01T00:00:00",
                    "completed_at": None,
                    "chat_uuid": "test-chat-uuid",
                    "delta": {"text": "world"},
                },
            },
            {
                "event": "finish",
                "info": {
                    **basic_info,
                    "completed_at": "2024-01-01T00:01:00",
                    "query": "test query",
                    "reply": "done",
                    "tool_calls": [{"name": "search"}],
                    "chat_uuid": "test-chat-uuid",
                },
            },
        ]

        if output_type == "raw":
            assert results == raw_results
        else:
            assert all(chunk.endswith("\n\n") for chunk in results)
            assert [json.loads(chunk.removeprefix("data: ")) for chunk in results] == raw_results

    async def test_voice_output_keeps_events_and_buffers_speech(self):
        delta = Mock()
        delta.text = "看[这里](https://a.dev) **重要**"
        delta.model_dump.return_value = {"text": delta.text}
        parser = self._make_chat_stream(output_type="voice")
        source = self._source([(AgentRunEvent.STREAM, self._make_run(delta=delta))])

        results = await self._collect(parser, source)

        assert results == ["看这里 重要"]

    async def test_voice_output_splits_speech_on_sentence_boundaries(self):
        first = Mock()
        first.text = "第一句。第二"
        first.model_dump.return_value = {"text": first.text}
        second = Mock()
        second.text = "句。尾巴"
        second.model_dump.return_value = {"text": second.text}
        parser = self._make_chat_stream(output_type="voice")
        source = self._source(
            [
                (AgentRunEvent.STREAM, self._make_run(delta=first)),
                (AgentRunEvent.STREAM, self._make_run(delta=second)),
            ]
        )

        results = await self._collect(parser, source)

        assert results == ["第一句。", "第二句。", "尾巴"]

    async def test_voice_output_sends_finish_after_speech(self):
        delta = Mock()
        delta.text = "第一句。尾巴"
        delta.model_dump.return_value = {"text": delta.text}
        parser = self._make_chat_stream(output_type="voice")
        source = self._source(
            [
                (AgentRunEvent.START, self._make_run()),
                (AgentRunEvent.STREAM, self._make_run(delta=delta)),
                (AgentRunEvent.TOOL, self._make_run()),
                (AgentRunEvent.FINISH, self._make_run(reply="done")),
            ]
        )

        results = await self._collect(parser, source)

        assert results == ["第一句。", "尾巴"]
        assert parser.current_run_id == "run-1"

    async def test_invalid_output_type_is_rejected(self):
        with pytest.raises(ValueError, match="output_type"):
            self._make_chat_stream(output_type="queue")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("output_type", ["raw", "sse", "voice"])
    async def test_error_handling(self, output_type):
        """Test error event is generated on exception."""
        chat_stream = self._make_chat_stream(output_type=output_type)
        source = self._source([], error=ValueError("Test error"))
        if output_type == "voice":
            with pytest.raises(ValueError, match="Test error"):
                await self._collect(chat_stream, source)
            return

        results = await self._collect(chat_stream, source)

        assert len(results) == 1
        if output_type == "raw":
            assert results[0] == {
                "event": "error",
                "info": {"message": "Test error"},
            }
        else:
            data = json.loads(results[0].removeprefix("data: "))
            assert data["event"] == "error"
            assert data["info"]["message"] == "Test error"


class TestChatRecognizeParser:
    """Test recognition stream parsing used by voice orchestration."""

    @staticmethod
    async def _collect(handler, source):
        return [payload async for payload in handler.parse_async(source)]

    @staticmethod
    def _events(*items: tuple[str, str]) -> list[SpeechEvent]:
        return [SpeechEvent(type=event_type, text=text) for event_type, text in items]

    @pytest.mark.asyncio
    async def test_recognition_flushes_after_five_empty_partials(self):
        source = iter(
            self._events(
                ("final", "one"),
                ("partial", ""),
                ("partial", ""),
                ("partial", ""),
                ("partial", ""),
                ("partial", ""),
                ("final", "two"),
            )
        )

        async def events():
            for event in source:
                yield event

        results = await self._collect(ChatRecognizeParser(), events())

        assert results == ["one", "two"]

    @pytest.mark.asyncio
    async def test_non_empty_partial_resets_empty_streak(self):
        source = iter(
            self._events(
                ("final", "one"),
                ("partial", ""),
                ("partial", ""),
                ("partial", "spoken"),
                ("partial", ""),
                ("partial", ""),
            )
        )

        async def events():
            for event in source:
                yield event

        results = await self._collect(ChatRecognizeParser(), events())

        assert results == ["one"]

    @pytest.mark.asyncio
    async def test_recognition_joins_finals_with_newlines(self):
        source = iter(self._events(("final", "one"), ("final", "two")))

        async def events():
            for event in source:
                yield event

        results = await self._collect(ChatRecognizeParser(), events())

        assert results == ["one\ntwo"]

    @pytest.mark.asyncio
    async def test_recognition_stops_after_first_limited_sentence(self):
        parsed: list[str] = []

        async def events():
            for event_type, text in (("final", "one"), ("final", "unused")):
                parsed.append(event_type)
                yield SpeechEvent(type=event_type, text=text)

        results = await self._collect(ChatRecognizeParser(max_sentences=1), events())

        assert results == ["one"]
        assert parsed == ["final"]

    @pytest.mark.asyncio
    async def test_recognition_without_events_yields_nothing(self):
        source = iter(self._events(("partial", ""), ("partial", "")))

        async def events():
            for event in source:
                yield event

        results = await self._collect(ChatRecognizeParser(), events())

        assert results == []

    @pytest.mark.asyncio
    async def test_recognition_error_propagates_without_flush(self):
        async def events():
            yield SpeechEvent(type="final", text="pending")
            yield SpeechEvent(type="partial", text="")
            raise SpeechRequestError("vendor failed")

        with pytest.raises(SpeechRequestError, match="vendor failed"):
            await self._collect(ChatRecognizeParser(), events())

    def test_recognition_rejects_non_positive_sentence_limit(self):
        with pytest.raises(ValueError, match="max_sentences"):
            ChatRecognizeParser(max_sentences=0)

    def test_voice_text_cleaning_removes_non_speech_markup(self):
        assert _chat_voice_clean("看[这里](https://example.com) **重要** #news") == (
            "看这里 重要 news"
        )


class TestUserChatRun:
    """Test true streaming through the user chat run."""

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
    def _make_run_mock(name="run-1"):
        run = Mock()
        run.model_dump.return_value = {
            "id": name,
            "agent_id": "agent-1",
            "started_at": "2024-01-01T00:00:00",
            "completed_at": None,
            "query": "hello",
            "reply": None,
            "tool_calls": {},
        }
        return run

    @staticmethod
    def _make_agent(events, run_kwargs):
        async def stream_async(**kwargs):
            run_kwargs.update(kwargs)
            for event in events:
                yield event

        agent = MagicMock()
        agent.stream_async = stream_async
        return agent

    def _start_query(self, user, **kwargs):
        """Create a direct raw chat run stream."""
        chat_uuid = kwargs.pop("chat_uuid")
        query = kwargs.pop("query")
        context = kwargs.pop("context", None)
        chat_run = UserChatRunProviderImpl(MagicMock()).create_chat_run(
            chat_uuid,
            user_uuid=user.uuid,
            context=context,
        )
        return chat_run.stream_async(query, **kwargs)

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_skill_retriever_async")
    @patch(f"{_QUERY}.create_tool_retriever_async")
    @patch(f"{_QUERY}.create_agent_async")
    async def test_stream_yields_events_and_binds_context(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        run_kwargs = {}
        runs = [self._make_run_mock("start"), self._make_run_mock("finish")]
        mock_create_agent.return_value = self._make_agent(
            [(AgentRunEvent.START, runs[0]), (AgentRunEvent.FINISH, runs[1])], run_kwargs
        )
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()
        user = self._make_mock_user()

        with _patch_query_providers(
            self._make_mock_config_provider(), self._make_mock_chat_provider()
        ):
            stream = self._start_query(
                user,
                chat_uuid="chat-streaming",
                query="hello",
                context={"project": "alpha"},
                timeout=timedelta(seconds=1),
            )
            events = [event async for event in stream]

        assert [event[0] for event in events] == [AgentRunEvent.START, AgentRunEvent.FINISH]
        assert run_kwargs["context"]["project"] == "alpha"
        assert run_kwargs["context"]["chat_uuid"] == "chat-streaming"
        assert run_kwargs["tool_ids"] == []
        assert run_kwargs["skill_ids"] == []
        assert mock_create_tool_retriever.return_value is run_kwargs["tool_retriever"]
        assert mock_create_skill_retriever.return_value is run_kwargs["skill_retriever"]
        await stream.aclose()

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_skill_retriever_async")
    @patch(f"{_QUERY}.create_tool_retriever_async")
    @patch(f"{_QUERY}.create_agent_async")
    async def test_retrievers_are_always_created(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        mock_create_agent.return_value = self._make_agent([], {})
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()
        user = self._make_mock_user()

        with _patch_query_providers(
            self._make_mock_config_provider(), self._make_mock_chat_provider()
        ):
            stream = self._start_query(
                user,
                chat_uuid="chat-retrievers",
                query="hello",
            )
            async for _event in stream:
                pass

        tool_kwargs = mock_create_tool_retriever.call_args.kwargs
        skill_kwargs = mock_create_skill_retriever.call_args.kwargs
        assert tool_kwargs["tools"] is None
        assert tool_kwargs["space_id"] == user.uuid
        assert skill_kwargs["space_id"] == user.uuid
        mock_create_tool_retriever.assert_awaited_once()
        mock_create_skill_retriever.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_skill_retriever_async")
    @patch(f"{_QUERY}.create_tool_retriever_async")
    @patch(f"{_QUERY}.create_agent_async")
    async def test_context_reuses_resolved_dependencies(
        self, mock_create_agent, mock_create_tool_retriever, mock_create_skill_retriever
    ):
        """Entering a run resolves provider dependencies once for many turns."""
        first_kwargs = {}
        second_kwargs = {}
        mock_create_agent.side_effect = [
            self._make_agent(
                [(AgentRunEvent.START, self._make_run_mock("first-run"))], first_kwargs
            ),
            self._make_agent(
                [(AgentRunEvent.FINISH, self._make_run_mock("second-run"))], second_kwargs
            ),
        ]
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()
        config_provider = self._make_mock_config_provider()
        chat_provider = self._make_mock_chat_provider()
        user = self._make_mock_user()

        with _patch_query_providers(config_provider, chat_provider):
            chat_run = UserChatRunProviderImpl(MagicMock()).create_chat_run(
                "chat-dependencies",
                user_uuid=user.uuid,
            )
            async with chat_run as run:
                first = [event async for event in run.stream_async("first")]
                second = [event async for event in run.stream_async("second")]

        assert [event[0] for event in first] == [AgentRunEvent.START]
        assert [event[0] for event in second] == [AgentRunEvent.FINISH]
        assert first_kwargs["query"] == "first"
        assert second_kwargs["query"] == "second"
        for provider_method in (
            config_provider.get_embedding_backend,
            config_provider.get_embedding_repository,
            config_provider.get_model_backend,
            config_provider.get_model_repository,
            config_provider.get_tool_backend,
            config_provider.get_tool_repository,
            config_provider.get_skill_repository,
            config_provider.get_agent_backend,
            config_provider.get_agent_repository,
        ):
            assert provider_method.call_count == 1
        assert chat_provider.get_chat_repository.call_count == 1

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_agent_async")
    async def test_setup_failure_releases_mutex(self, mock_create_agent):
        mock_create_agent.side_effect = RuntimeError("setup failed")
        mutex = Mock()
        mutex.release_async = AsyncMock()

        with _patch_query_providers(
            self._make_mock_config_provider(), self._make_mock_chat_provider()
        ):
            stream = self._start_query(
                self._make_mock_user(),
                chat_uuid="chat-setup-failed",
                query="hello",
                mutex=mutex,
            )
        with pytest.raises(RuntimeError, match="setup failed"):
            async for _event in stream:
                pass

        mutex.release_async.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_agent_async")
    @patch(f"{_QUERY}.create_skill_retriever_async")
    @patch(f"{_QUERY}.create_tool_retriever_async")
    async def test_cancellation_releases_mutex(
        self, mock_create_tool_retriever, mock_create_skill_retriever, mock_create_agent
    ):
        started = asyncio.Event()
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        class HangingAgent:
            async def stream_async(self, **kwargs):
                started.set()
                yield AgentRunEvent.START, self._make_run_mock()
                await asyncio.sleep(10)

            @staticmethod
            def _make_run_mock(name="run-1"):
                run = Mock()
                run.model_dump.return_value = {"id": name}
                return run

        mock_create_agent.return_value = HangingAgent()
        mutex = Mock()
        mutex.release_async = AsyncMock()

        with _patch_query_providers(
            self._make_mock_config_provider(), self._make_mock_chat_provider()
        ):
            stream = self._start_query(
                self._make_mock_user(),
                chat_uuid="chat-cancelled",
                query="hello",
                mutex=mutex,
            )
            first = await anext(stream)
            assert first[0] is AgentRunEvent.START
            await started.wait()
            await stream.aclose()

        mutex.release_async.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{_QUERY}.create_agent_async")
    @patch(f"{_QUERY}.create_skill_retriever_async")
    @patch(f"{_QUERY}.create_tool_retriever_async")
    async def test_timeout_emits_error_and_releases_mutex(
        self, mock_create_tool_retriever, mock_create_skill_retriever, mock_create_agent
    ):
        mock_create_tool_retriever.return_value = AsyncMock()
        mock_create_skill_retriever.return_value = AsyncMock()

        class SlowAgent:
            async def stream_async(self, **kwargs):
                await asyncio.sleep(0.1)
                yield AgentRunEvent.FINISH, Mock()

        mock_create_agent.return_value = SlowAgent()
        mutex = Mock()
        mutex.release_async = AsyncMock()

        with _patch_query_providers(
            self._make_mock_config_provider(), self._make_mock_chat_provider()
        ):
            stream = self._start_query(
                self._make_mock_user(),
                chat_uuid="chat-timeout",
                query="hello",
                mutex=mutex,
                timeout=timedelta(seconds=0.01),
            )
            with pytest.raises(TimeoutError):
                async for _event in stream:
                    pass

        mutex.release_async.assert_awaited_once()
