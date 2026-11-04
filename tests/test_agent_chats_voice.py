"""Tests for chat voice setup and Pipecat Protobuf WebSocket orchestration."""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fivcplayground.agents import AgentRunSession
from pipecat.serializers.protobuf import ProtobufFrameSerializer

from fivccliche.modules.agent_chats.routers import VoiceSetupError
from fivccliche.utils.chats.speeches import (
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    process_chat_speech_async,
)


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
    user = MagicMock(uuid="user-1", is_superuser=False)

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
            user,
            {"stt_id": "chat-asr", "tts_id": "chat-tts"},
        )

    assert opened == (recognizer, synthesizer)
    assert provider.get_recognizer.await_args.kwargs["id"] == "asr"
    assert provider.get_synthesizer.await_args.kwargs["id"] == "tts"


@pytest.mark.asyncio
async def test_router_uses_agent_extra_stt_tts_defaults():
    asr = MagicMock(id="from-extra", model_type="dashscope_realtime", model="m", api_key="k")
    tts = MagicMock(id="from-extra-tts", model_type="dashscope_realtime", model="m", api_key="k")
    provider = MagicMock()
    provider.get_recognizer = AsyncMock(return_value=MagicMock())
    provider.get_synthesizer = AsyncMock(return_value=MagicMock())
    session = AsyncMock()
    user = MagicMock(uuid="user-1", is_superuser=False)

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
            user,
            {},
            agent_extra={"stt_id": "from-extra", "tts_id": "from-extra-tts"},
        )

    assert [call.kwargs["config_id"] for call in get_config.await_args_list] == [
        "from-extra",
        "from-extra-tts",
    ]


@pytest.mark.asyncio
async def test_router_ignores_chat_context_stt_tts():
    asr = MagicMock(id="realtime", model_type="dashscope_realtime", model="m", api_key="k")
    tts = MagicMock(id="realtime", model_type="dashscope_realtime", model="m", api_key="k")
    provider = MagicMock()
    provider.get_recognizer = AsyncMock(return_value=MagicMock())
    provider.get_synthesizer = AsyncMock(return_value=MagicMock())
    session = AsyncMock()
    user = MagicMock(uuid="user-1", is_superuser=False)

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

        # chat.context is no longer consulted; even if callers still pass a chat
        # with context asr/tts, setup only sees start_frame + agent_extra.
        await _load_realtime_voice_setup(session, user, {})

    assert [call.kwargs["config_id"] for call in get_config.await_args_list] == [
        "realtime",
        "realtime",
    ]


@pytest.mark.asyncio
async def test_router_rejects_non_realtime_asr():
    asr = MagicMock(id="asr", model_type="dashscope", model="m", api_key="k")
    tts = MagicMock(id="tts", model_type="dashscope_realtime", model="m", api_key="k")
    session = AsyncMock()
    user = MagicMock(uuid="user-1", is_superuser=False)

    with patch(
        "fivccliche.modules.agent_configs.utils.get_user_scoped_async",
        new=AsyncMock(side_effect=[asr, tts]),
    ):
        from fivccliche.modules.agent_chats.routers import _load_realtime_voice_setup

        with pytest.raises(VoiceSetupError) as exc_info:
            await _load_realtime_voice_setup(session, user, {})

    assert exc_info.value.close_code == 1003
    assert exc_info.value.code == "invalid_asr"


@pytest.mark.asyncio
async def test_router_defaults_to_realtime_config_ids():
    asr = MagicMock(id="realtime", model_type="dashscope_realtime", model="m", api_key="k")
    tts = MagicMock(id="realtime", model_type="dashscope_realtime", model="m", api_key="k")
    provider = MagicMock()
    provider.get_recognizer = AsyncMock(return_value=MagicMock())
    provider.get_synthesizer = AsyncMock(return_value=MagicMock())
    session = AsyncMock()
    user = MagicMock(uuid="user-1", is_superuser=False)

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

        await _load_realtime_voice_setup(session, user, {})

    assert [call.kwargs["config_id"] for call in get_config.await_args_list] == [
        "realtime",
        "realtime",
    ]


@pytest.mark.asyncio
async def test_process_chat_voice_sends_ready_and_runs_agent():
    channel = MagicMock()
    channel.socket = MagicMock(name="channel.socket")
    channel.send_json_async = AsyncMock()
    channel.mark_prepared = MagicMock()
    channel.begin_streaming = MagicMock()
    channel.fail_async = AsyncMock(side_effect=AssertionError("fail_async should not be called"))

    recognizer = MagicMock(id="asr")
    recognizer.get_option.return_value = MagicMock(format="pcm", sample_rate=16000)
    synthesizer = MagicMock(id="tts")
    synthesizer.get_option.return_value = MagicMock(format="pcm", sample_rate=24000)

    agent = MagicMock()
    agent.run_async = AsyncMock(return_value=MagicMock())
    agent_config = MagicMock(model_id="llm-1", extra=None)
    model_config = MagicMock(provider="openai", extra=None)

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
        ) as create_agent,
        patch(
            "fivccliche.utils.chats.speeches.create_tool_retriever_async",
            new=AsyncMock(return_value=MagicMock(tools={})),
        ),
        patch(
            "fivccliche.utils.chats.speeches.create_skill_retriever_async",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "fivccliche.utils.chats.speeches.FastAPIWebsocketTransport",
        ) as transport_cls,
        patch(
            "fivccliche.utils.chats.speeches.SileroVADAnalyzer",
        ) as vad_cls,
    ):
        transport_cls.return_value = MagicMock()
        vad_cls.return_value = MagicMock()

        await process_chat_speech_async(
            channel=channel,
            user=MagicMock(uuid="user-1"),
            chat=_chat_session(),
            stt=recognizer,
            tts=synthesizer,
            mutex=mutex,
            timeout=timedelta(minutes=5),
            config_provider=config_provider,
            chat_provider=chat_provider,
        )

    channel.send_json_async.assert_awaited_once()
    ready = channel.send_json_async.await_args.args[0]
    assert ready["event"] == "ready"
    assert ready["info"]["stt_id"] == "asr"
    assert ready["info"]["tts_id"] == "tts"
    assert ready["info"]["input"]["sample_rate"] == 16000
    assert ready["info"]["output"]["sample_rate"] == 24000

    transport_cls.assert_called_once()
    assert transport_cls.call_args.kwargs["websocket"] is channel.socket
    params = transport_cls.call_args.kwargs["params"]
    assert isinstance(params.serializer, ProtobufFrameSerializer)
    assert params.audio_in_sample_rate == INPUT_SAMPLE_RATE
    assert params.audio_out_sample_rate == OUTPUT_SAMPLE_RATE

    channel.mark_prepared.assert_called_once_with()
    channel.begin_streaming.assert_called_once_with()
    create_agent.assert_awaited_once()
    assert create_agent.await_args.kwargs["agent_config_id"] == "agent-1"
    agent.run_async.assert_awaited_once()
    run_kwargs = agent.run_async.await_args.kwargs
    assert run_kwargs["agent_run_session_id"] == "chat-1"
    assert run_kwargs["stt"] is recognizer
    assert run_kwargs["tts"] is synthesizer
    assert run_kwargs["vad"] is vad_cls.return_value
    assert run_kwargs["transport"] is transport_cls.return_value
    assert run_kwargs["tool_ids"] == []
    mutex.acquire_async.assert_awaited_once()
    mutex.release_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_chat_voice_e2e_mode_skips_stt_tts_vad():
    channel = MagicMock()
    channel.socket = MagicMock(name="channel.socket")
    channel.send_json_async = AsyncMock()
    channel.mark_prepared = MagicMock()
    channel.begin_streaming = MagicMock()
    channel.fail_async = AsyncMock(side_effect=AssertionError("fail_async should not be called"))

    underlying = MagicMock(name="omni-service")
    e2e_model = MagicMock()
    e2e_model.get_underlying.return_value = underlying
    agent_config = MagicMock(
        id="text-agent",
        name="text-agent",
        model_id="text-llm",
        extra={"e2e_id": "omni-realtime-llm"},
        tool_ids=None,
        skill_ids=None,
        response_model=None,
    )
    model_config = MagicMock(
        provider="qwen_realtime",
        extra={"input_sample_rate": 16000, "output_sample_rate": 24000},
    )

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

    runnable = MagicMock()
    runnable.run_async = AsyncMock(return_value=MagicMock())
    model_backend = MagicMock()
    model_backend.create_model.return_value = e2e_model

    with (
        patch(
            "fivccliche.utils.chats.speeches.create_agent_async",
            new=AsyncMock(),
        ) as create_agent,
        patch(
            "fivccliche.utils.chats.speeches.create_tool_retriever_async",
            new=AsyncMock(return_value=MagicMock(tools={})),
        ),
        patch(
            "fivccliche.utils.chats.speeches.create_skill_retriever_async",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "fivccliche.utils.chats.speeches.FastAPIWebsocketTransport",
        ) as transport_cls,
        patch(
            "fivccliche.utils.chats.speeches.SileroVADAnalyzer",
        ) as vad_cls,
        patch(
            "fivcplayground.backends.pipecat.agents.PipecatAgentRunnable",
            return_value=runnable,
        ) as runnable_cls,
        patch(
            "fivcplayground.backends.pipecat.models.PipecatModelBackend",
            return_value=model_backend,
        ),
    ):
        transport_cls.return_value = MagicMock()

        await process_chat_speech_async(
            channel=channel,
            user=MagicMock(uuid="user-1"),
            chat=_chat_session(agent_id="text-agent"),
            config_provider=config_provider,
            chat_provider=chat_provider,
        )

    ready = channel.send_json_async.await_args.args[0]
    assert ready["event"] == "ready"
    assert ready["info"]["stt_id"] is None
    assert ready["info"]["tts_id"] is None
    assert ready["info"]["input"]["sample_rate"] == 16000
    assert ready["info"]["output"]["sample_rate"] == 24000

    create_agent.assert_not_awaited()
    agent_repo.get_agent_config_async.assert_awaited_with("text-agent")
    model_repo.get_model_config_async.assert_awaited_with("omni-realtime-llm")
    model_backend.create_model.assert_called_once_with(model_config)
    runnable_cls.assert_called_once_with(agent_config, underlying)
    vad_cls.assert_not_called()
    channel.mark_prepared.assert_called_once_with()
    channel.begin_streaming.assert_called_once_with()
    run_kwargs = runnable.run_async.await_args.kwargs
    assert run_kwargs["stt"] is None
    assert run_kwargs["tts"] is None
    assert run_kwargs["vad"] is None


@pytest.mark.asyncio
async def test_process_chat_voice_missing_e2e_model_config_fails():
    from fivccliche.utils.chats.channels import ChatChannelError

    channel = MagicMock()
    channel.fail_async = AsyncMock(side_effect=ChatChannelError(code=4404, msg="missing"))

    agent_config = MagicMock(
        model_id="text-llm",
        extra={"e2e_id": "missing-omni"},
    )

    config_provider = MagicMock()
    agent_repo = MagicMock()
    agent_repo.get_agent_config_async = AsyncMock(return_value=agent_config)
    model_repo = MagicMock()
    model_repo.get_model_config_async = AsyncMock(return_value=None)
    config_provider.get_agent_repository.return_value = agent_repo
    config_provider.get_model_repository.return_value = model_repo
    config_provider.get_embedding_repository.return_value = MagicMock()
    config_provider.get_embedding_backend.return_value = MagicMock()
    config_provider.get_tool_repository.return_value = MagicMock()
    config_provider.get_skill_repository.return_value = MagicMock()

    with pytest.raises(ChatChannelError):
        await process_chat_speech_async(
            channel=channel,
            user=MagicMock(uuid="user-1"),
            chat=_chat_session(),
            config_provider=config_provider,
            chat_provider=MagicMock(),
        )

    channel.fail_async.assert_awaited_once()
    assert channel.fail_async.await_args.kwargs["code"] == "model_not_found"
    assert channel.fail_async.await_args.kwargs["close_code"] == 4404
