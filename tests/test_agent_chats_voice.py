"""Tests for chat voice setup and Pipecat session orchestration."""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fivcplayground.agents import AgentRunSession

from fivccliche.utils.chats.speeches import (
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    process_chat_speech_async,
)


def _voice_authenticator():
    auth = MagicMock()
    auth.verify_credential_async = AsyncMock(return_value=MagicMock(uuid="user-1"))
    return auth


def _chat_session(
    session_id: str = "chat-1",
    agent_id: str = "agent-1",
) -> AgentRunSession:
    return AgentRunSession(id=session_id, agent_id=agent_id, context={}, description=None)


def test_voice_sample_rates():
    assert INPUT_SAMPLE_RATE == 16000
    assert OUTPUT_SAMPLE_RATE == 24000


@pytest.mark.asyncio
async def test_router_resolves_realtime_voice_values():
    asr = MagicMock(id="asr", model_type="dashscope_realtime", model="asr-model", api_key="k")
    tts = MagicMock(id="tts", model_type="dashscope_realtime", model="tts-model", api_key="k")
    recognizer = MagicMock()
    synthesizer = MagicMock()
    provider = MagicMock()
    provider.get_recognizer = AsyncMock(return_value=recognizer)
    provider.get_synthesizer = AsyncMock(return_value=synthesizer)
    session = AsyncMock()
    user = _voice_authenticator().verify_credential_async.return_value

    with (
        patch(
            "fivccliche.modules.agent_configs.utils.get_user_scoped_async",
            new=AsyncMock(side_effect=[asr, tts]),
        ),
        patch(
            "fivccliche.utils.deps.get_speech_provider_async",
            new=AsyncMock(return_value=provider),
        ),
    ):
        from fivccliche.modules.agent_chats.routers import _load_realtime_voice_setup

        opened = await _load_realtime_voice_setup(
            session,
            MagicMock(),
            user,
            _chat_session(),
            {"asr_id": "chat-asr", "tts_id": "chat-tts"},
        )

    assert opened == (recognizer, synthesizer)
    assert provider.get_recognizer.await_args.kwargs["id"] == "asr"
    assert provider.get_synthesizer.await_args.kwargs["id"] == "tts"
    session.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_router_defaults_to_realtime_config_ids():
    asr = MagicMock(id="realtime", model_type="dashscope_realtime", model="m", api_key="k")
    tts = MagicMock(id="realtime", model_type="dashscope_realtime", model="m", api_key="k")
    provider = MagicMock()
    provider.get_recognizer = AsyncMock(return_value=MagicMock())
    provider.get_synthesizer = AsyncMock(return_value=MagicMock())
    session = AsyncMock()
    user = _voice_authenticator().verify_credential_async.return_value

    with (
        patch(
            "fivccliche.modules.agent_configs.utils.get_user_scoped_async",
            new=AsyncMock(side_effect=[asr, tts]),
        ) as get_config,
        patch(
            "fivccliche.utils.deps.get_speech_provider_async",
            new=AsyncMock(return_value=provider),
        ),
    ):
        from fivccliche.modules.agent_chats.routers import _load_realtime_voice_setup

        await _load_realtime_voice_setup(
            session,
            MagicMock(),
            user,
            _chat_session(),
            {},
        )

    assert [call.kwargs["config_id"] for call in get_config.await_args_list] == [
        "realtime",
        "realtime",
    ]


@pytest.mark.asyncio
async def test_process_chat_voice_sends_ready_and_runs_agent():
    channel = MagicMock()
    channel.socket = MagicMock(name="channel.socket")
    channel.send_json_async = AsyncMock()
    channel.shutdown = MagicMock()
    channel.fail_async = AsyncMock(side_effect=AssertionError("fail_async should not be called"))

    recognizer = MagicMock(id="asr")
    recognizer.get_option.return_value = MagicMock(format="pcm", sample_rate=16000)
    synthesizer = MagicMock(id="tts")
    synthesizer.get_option.return_value = MagicMock(format="pcm", sample_rate=24000)

    agent = MagicMock()
    agent.run_async = AsyncMock(return_value=MagicMock())
    agent_config = MagicMock(model_id="llm-1")
    model_config = MagicMock(provider="openai")

    config_provider = MagicMock()
    model_repo = MagicMock()
    model_repo.get_model_config_async = AsyncMock(return_value=model_config)
    agent_repo = MagicMock()
    agent_repo.get_agent_config_async = AsyncMock(return_value=agent_config)
    config_provider.get_model_repository.return_value = model_repo
    config_provider.get_agent_repository.return_value = agent_repo
    config_provider.get_embedding_repository.return_value = MagicMock()
    config_provider.get_embedding_backend.return_value = MagicMock()
    config_provider.get_tool_repository.return_value = MagicMock()
    config_provider.get_skill_repository.return_value = MagicMock()

    chat_provider = MagicMock()
    chat_provider.get_chat_context.return_value = {"user_uuid": "user-1"}
    chat_provider.get_chat_repository.return_value = MagicMock()

    mutex = MagicMock()
    mutex.acquire_async = AsyncMock(return_value=True)
    mutex.release_async = AsyncMock()

    with (
        patch(
            "fivccliche.utils.chats.speeches.create_agent_async",
            new=AsyncMock(return_value=agent),
        ),
        patch(
            "fivccliche.utils.chats.speeches.create_tool_retriever_async",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "fivccliche.utils.chats.speeches.create_skill_retriever_async",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "fivccliche.utils.chats.speeches.FastAPIWebsocketTransport",
        ) as transport_cls,
        patch(
            "fivccliche.utils.chats.speeches.SpeechSTTService",
        ) as stt_cls,
        patch(
            "fivccliche.utils.chats.speeches.SpeechTTSService",
        ) as tts_cls,
    ):
        transport_cls.return_value = MagicMock()
        stt_cls.return_value = MagicMock()
        tts_cls.return_value = MagicMock()

        await process_chat_speech_async(
            channel=channel,
            user=MagicMock(uuid="user-1"),
            chat=_chat_session(),
            recognizer=recognizer,
            synthesizer=synthesizer,
            mutex=mutex,
            timeout=timedelta(minutes=5),
            config_provider=config_provider,
            chat_provider=chat_provider,
        )

    ready = channel.send_json_async.await_args.args[0]
    assert ready["event"] == "ready"
    assert ready["info"]["asr_id"] == "asr"
    assert ready["info"]["tts_id"] == "tts"
    channel.shutdown.assert_called_once()
    transport_cls.assert_called_once()
    assert transport_cls.call_args.kwargs["websocket"] is channel.socket
    agent.run_async.assert_awaited_once()
    run_kwargs = agent.run_async.await_args.kwargs
    assert run_kwargs["agent_run_session_id"] == "chat-1"
    assert run_kwargs["stt"] is stt_cls.return_value
    assert run_kwargs["tts"] is tts_cls.return_value
    assert run_kwargs["transport"] is transport_cls.return_value
    mutex.acquire_async.assert_awaited_once()
    mutex.release_async.assert_awaited_once()
