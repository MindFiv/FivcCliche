"""Tests for chat voice setup and Pipecat WebRTC session orchestration."""

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fivcplayground.agents import AgentRunSession

from fivccliche.modules.agent_chats.routers import VoiceSetupError
from fivccliche.utils.chats import ChatChannelError
from fivccliche.utils.chats.speeches import (
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    ChatWebRTCRegistry,
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
            _chat_session(),
            {"asr_id": "chat-asr", "tts_id": "chat-tts"},
        )

    assert opened == (recognizer, synthesizer)
    assert provider.get_recognizer.await_args.kwargs["id"] == "asr"
    assert provider.get_synthesizer.await_args.kwargs["id"] == "tts"


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
            await _load_realtime_voice_setup(session, user, _chat_session(), {})

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

        await _load_realtime_voice_setup(session, user, _chat_session(), {})

    assert [call.kwargs["config_id"] for call in get_config.await_args_list] == [
        "realtime",
        "realtime",
    ]


@pytest.mark.asyncio
async def test_process_chat_voice_runs_webrtc_agent():
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
    connection = MagicMock()

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
            "fivccliche.utils.chats.speeches.SmallWebRTCTransport",
        ) as transport_cls,
        patch(
            "fivccliche.utils.chats.speeches.SileroVADAnalyzer",
        ) as vad_cls,
    ):
        transport_cls.return_value = MagicMock()
        vad_cls.return_value = MagicMock()

        await process_chat_speech_async(
            connection=connection,
            user=MagicMock(uuid="user-1"),
            chat=_chat_session(),
            recognizer=recognizer,
            synthesizer=synthesizer,
            mutex=mutex,
            timeout=timedelta(minutes=5),
            config_provider=config_provider,
            chat_provider=chat_provider,
        )

    transport_cls.assert_called_once()
    assert transport_cls.call_args.kwargs["webrtc_connection"] is connection
    agent.run_async.assert_awaited_once()
    run_kwargs = agent.run_async.await_args.kwargs
    assert run_kwargs["agent_run_session_id"] == "chat-1"
    assert run_kwargs["stt"] is recognizer
    assert run_kwargs["tts"] is synthesizer
    assert run_kwargs["vad"] is vad_cls.return_value
    assert run_kwargs["transport"] is transport_cls.return_value
    mutex.acquire_async.assert_awaited_once()
    mutex.release_async.assert_awaited_once()


class _FakeChannel:
    def __init__(self, user, frames: list[dict | None]) -> None:
        self.user = user
        self.frames = list(frames)
        self.sent: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def authenticate_async(self):
        return self.user

    async def receive_json_async(self):
        if not self.frames:
            return None
        return self.frames.pop(0)

    async def send_json_async(self, payload):
        self.sent.append(payload)

    async def fail_async(self, *, code: str, message: str, close_code: int):
        raise ChatChannelError(code=close_code, msg=message)


class _Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return None


@pytest.mark.asyncio
async def test_webrtc_offer_sends_answer_without_waiting_for_session():
    from fivccliche.modules.agent_chats.routers import start_chat_webrtc_async

    user = MagicMock(uuid="user-1", is_superuser=False)
    chat_row = MagicMock()
    chat_row.to_schema.return_value = _chat_session()
    recognizer = MagicMock(id="asr")
    recognizer.get_option.return_value = MagicMock(format="pcm", sample_rate=16000)
    synthesizer = MagicMock(id="tts")
    synthesizer.get_option.return_value = MagicMock(format="pcm", sample_rate=24000)

    registry = ChatWebRTCRegistry()
    started = asyncio.Event()
    release = asyncio.Event()

    async def _session(**_kwargs):
        started.set()
        await release.wait()

    async def _handle(_webrtc_request, callback):
        connection = MagicMock()
        connection.pc_id = "pc-1"
        connection.event_handler = lambda _name: (lambda fn: fn)
        await callback(connection)
        return {"sdp": "answer-sdp", "type": "answer", "pc_id": "pc-1"}

    registry.handler.handle_web_request = _handle  # type: ignore[method-assign]

    channel = _FakeChannel(
        user,
        [
            {
                "type": "offer",
                "sdp": "offer-sdp",
                "sdp_type": "offer",
                "asr_id": "realtime",
                "tts_id": "realtime",
            },
            None,
        ],
    )
    websocket = MagicMock()
    websocket.app.state.chat_webrtc = registry

    with (
        patch(
            "fivccliche.modules.agent_chats.routers.get_authenticator_async",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "fivccliche.utils.chats.ChatChannel",
            side_effect=lambda *_args, **_kwargs: channel,
        ),
        patch(
            "fivccliche.modules.agent_chats.routers.utils.get_chat_async",
            new=AsyncMock(return_value=chat_row),
        ),
        patch(
            "fivccliche.modules.agent_chats.routers._load_realtime_voice_setup",
            new=AsyncMock(return_value=(recognizer, synthesizer)),
        ),
        patch(
            "fivccliche.modules.agent_chats.routers.deps.get_db_session_context_async",
            return_value=_Session(),
        ),
        patch(
            "fivccliche.utils.chats.speeches.process_chat_speech_async",
            new=_session,
        ),
    ):
        await start_chat_webrtc_async("chat-1", websocket, None)

    assert channel.sent[0]["event"] == "answer"
    info = channel.sent[0]["info"]
    assert info["sdp"] == "answer-sdp"
    assert info["asr_id"] == "asr"
    assert info["tts_id"] == "tts"
    assert info["input"]["sample_rate"] == 16000
    assert registry.owned_by("pc-1", "user-1", "chat-1")
    await asyncio.sleep(0)
    assert started.is_set()
    release.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_webrtc_ice_requires_the_same_user_chat():
    from fivccliche.modules.agent_chats.routers import start_chat_webrtc_async

    user = MagicMock(uuid="user-1", is_superuser=False)
    chat_row = MagicMock()
    chat_row.to_schema.return_value = _chat_session()
    recognizer = MagicMock(id="asr")
    recognizer.get_option.return_value = MagicMock(format="pcm", sample_rate=16000)
    synthesizer = MagicMock(id="tts")
    synthesizer.get_option.return_value = MagicMock(format="pcm", sample_rate=24000)

    registry = ChatWebRTCRegistry()
    registry.handler.handle_patch_request = AsyncMock()  # type: ignore[method-assign]

    async def _handle(_webrtc_request, callback):
        connection = MagicMock()
        connection.pc_id = "pc-1"
        connection.event_handler = lambda _name: (lambda fn: fn)
        await callback(connection)
        return {"sdp": "answer-sdp", "type": "answer", "pc_id": "pc-1"}

    registry.handler.handle_web_request = _handle  # type: ignore[method-assign]

    channel = _FakeChannel(
        user,
        [
            {"type": "offer", "sdp": "offer-sdp", "sdp_type": "offer"},
            {
                "type": "ice",
                "pc_id": "other-pc",
                "candidates": [
                    {"candidate": "cand", "sdp_mid": "0", "sdp_mline_index": 0},
                ],
            },
        ],
    )
    websocket = MagicMock()
    websocket.app.state.chat_webrtc = registry

    with (
        patch(
            "fivccliche.modules.agent_chats.routers.get_authenticator_async",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "fivccliche.utils.chats.ChatChannel",
            side_effect=lambda *_args, **_kwargs: channel,
        ),
        patch(
            "fivccliche.modules.agent_chats.routers.utils.get_chat_async",
            new=AsyncMock(return_value=chat_row),
        ),
        patch(
            "fivccliche.modules.agent_chats.routers._load_realtime_voice_setup",
            new=AsyncMock(return_value=(recognizer, synthesizer)),
        ),
        patch(
            "fivccliche.modules.agent_chats.routers.deps.get_db_session_context_async",
            return_value=_Session(),
        ),
        patch(
            "fivccliche.utils.chats.speeches.process_chat_speech_async",
            new=AsyncMock(),
        ),
        pytest.raises(ChatChannelError) as exc_info,
    ):
        await start_chat_webrtc_async("chat-1", websocket, None)

    assert exc_info.value.code == 4404
    registry.handler.handle_patch_request.assert_not_awaited()

    channel_ok = _FakeChannel(
        user,
        [
            {"type": "offer", "sdp": "offer-sdp", "sdp_type": "offer"},
            {
                "type": "ice",
                "pc_id": "pc-1",
                "candidates": [
                    {"candidate": "cand", "sdp_mid": "0", "sdp_mline_index": 0},
                ],
            },
            None,
        ],
    )
    websocket.app.state.chat_webrtc = registry
    with (
        patch(
            "fivccliche.modules.agent_chats.routers.get_authenticator_async",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "fivccliche.utils.chats.ChatChannel",
            side_effect=lambda *_args, **_kwargs: channel_ok,
        ),
        patch(
            "fivccliche.modules.agent_chats.routers.utils.get_chat_async",
            new=AsyncMock(return_value=chat_row),
        ),
        patch(
            "fivccliche.modules.agent_chats.routers._load_realtime_voice_setup",
            new=AsyncMock(return_value=(recognizer, synthesizer)),
        ),
        patch(
            "fivccliche.modules.agent_chats.routers.deps.get_db_session_context_async",
            return_value=_Session(),
        ),
        patch(
            "fivccliche.utils.chats.speeches.process_chat_speech_async",
            new=AsyncMock(),
        ),
    ):
        await start_chat_webrtc_async("chat-1", websocket, None)

    registry.handler.handle_patch_request.assert_awaited_once()
