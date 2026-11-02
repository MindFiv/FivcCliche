"""Pipecat Protobuf WebSocket sessions for full-duplex chat voice."""

from __future__ import annotations

import logging
from datetime import timedelta

from fivcglue.interfaces.mutexes import IMutex
from fivcplayground.agents import AgentRunSession, create_agent_async
from fivcplayground.skills import create_skill_retriever_async
from fivcplayground.speeches import SpeechRecognizer, SpeechSynthesizer
from fivcplayground.tools import create_tool_retriever_async
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from fivccliche.services.interfaces.agent_chats import IUserChatProvider
from fivccliche.services.interfaces.agent_configs import IUserConfigProvider
from fivccliche.services.interfaces.auth import IUser
from fivccliche.utils.chats.channels import ChatChannel
from fivccliche.utils.deps import get_chat_provider_async, get_config_provider_async

logger = logging.getLogger(__name__)

INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000


async def process_chat_speech_async(
    *,
    channel: ChatChannel,
    user: IUser,
    chat: AgentRunSession,
    stt: SpeechRecognizer | None = None,
    tts: SpeechSynthesizer | None = None,
    mutex: IMutex | None = None,
    timeout: timedelta = timedelta(minutes=5),
    config_provider: IUserConfigProvider | None = None,
    chat_provider: IUserChatProvider | None = None,
) -> None:
    """Send ``ready``, then run a Pipecat Protobuf WebSocket pipeline until disconnect.

    When the chat agent's ``extra.e2e_id`` resolves to a ModelConfig, the session
    runs that end-to-end model with ``chat.agent_id`` (no local STT/TTS/VAD) and
    ignores ``stt`` / ``tts``. Otherwise cascade STT+LLM+TTS is used.
    """
    from fivcplayground.backends.pipecat.agents import (
        PipecatAgentBackend,
        PipecatAgentRunnable,
    )
    from fivcplayground.backends.pipecat.models import PipecatModelBackend
    from fivcplayground.backends.pipecat.tools import PipecatToolBackend

    if config_provider is None:
        config_provider = await get_config_provider_async()
    if chat_provider is None:
        chat_provider = await get_chat_provider_async()

    model_backend = PipecatModelBackend()
    tool_backend = PipecatToolBackend()
    agent_backend = PipecatAgentBackend()

    model_repository = config_provider.get_model_repository(user_uuid=user.uuid)
    agent_repository = config_provider.get_agent_repository(user_uuid=user.uuid)
    embedding_repository = config_provider.get_embedding_repository(user_uuid=user.uuid)
    embedding_backend = config_provider.get_embedding_backend()
    tool_repository = config_provider.get_tool_repository(user_uuid=user.uuid)
    skill_repository = config_provider.get_skill_repository(user_uuid=user.uuid)

    agent_config = await agent_repository.get_agent_config_async(chat.agent_id)
    if agent_config is None:
        await channel.fail_async(
            code="agent_not_found",
            message="Agent config not found",
            close_code=4404,
        )

    agent_extra = agent_config.extra or {}
    e2e_id = agent_extra.get("e2e_id")
    if isinstance(e2e_id, str) and e2e_id:
        model_config = await model_repository.get_model_config_async(e2e_id)
        if model_config is None:
            await channel.fail_async(
                code="model_not_found",
                message="LLM config not found",
                close_code=4404,
            )
        e2e_model = model_backend.create_model(model_config)
        model_extra = model_config.extra or {}
        input_sample_rate = int(model_extra.get("input_sample_rate") or INPUT_SAMPLE_RATE)
        output_sample_rate = int(model_extra.get("output_sample_rate") or OUTPUT_SAMPLE_RATE)
        agent = PipecatAgentRunnable(agent_config, e2e_model.get_underlying())
        await channel.send_json_async(
            {
                "event": "ready",
                "info": {
                    "stt_id": None,
                    "tts_id": None,
                    "input": {
                        "format": "pcm",
                        "sample_rate": input_sample_rate,
                    },
                    "output": {
                        "format": "pcm",
                        "sample_rate": output_sample_rate,
                    },
                },
            }
        )
        run_stt = None
        run_tts = None
        run_vad = None
    else:
        if stt is None or tts is None:
            await channel.fail_async(
                code="invalid_voice_setup",
                message="Cascade voice requires both STT and TTS",
                close_code=1003,
            )
        model_config = await model_repository.get_model_config_async(agent_config.model_id)
        if model_config is None:
            await channel.fail_async(
                code="model_not_found",
                message="LLM config not found",
                close_code=4404,
            )
        agent = await create_agent_async(
            model_backend=model_backend,
            model_config_repository=model_repository,
            agent_backend=agent_backend,
            agent_config_repository=agent_repository,
            agent_config_id=chat.agent_id,
        )
        recognize_opts = stt.get_option()
        synthesis_opts = tts.get_option()
        input_sample_rate = recognize_opts.sample_rate
        output_sample_rate = synthesis_opts.sample_rate
        await channel.send_json_async(
            {
                "event": "ready",
                "info": {
                    "stt_id": stt.id,
                    "tts_id": tts.id,
                    "input": {
                        "format": recognize_opts.format,
                        "sample_rate": input_sample_rate,
                    },
                    "output": {
                        "format": synthesis_opts.format,
                        "sample_rate": output_sample_rate,
                    },
                },
            }
        )
        run_stt = stt
        run_tts = tts
        run_vad = SileroVADAnalyzer(sample_rate=input_sample_rate)

    tool_retriever = await create_tool_retriever_async(
        tool_backend=tool_backend,
        tools=None,
        tool_config_repository=tool_repository,
        embedding_backend=embedding_backend,
        embedding_config_repository=embedding_repository,
        space_id=user.uuid,
    )
    skill_retriever = await create_skill_retriever_async(
        tool_backend=tool_backend,
        skill_config_repository=skill_repository,
        embedding_backend=embedding_backend,
        embedding_config_repository=embedding_repository,
        space_id=user.uuid,
    )
    context = chat_provider.get_chat_context(
        user_uuid=user.uuid,
        context=chat.context,
        chat_uuid=chat.id,
    )
    chat_repository = chat_provider.get_chat_repository(user_uuid=user.uuid)

    mutex_acquired = False
    if mutex is not None:
        # Mutex TTL is slightly longer than the session timeout so the lock does not
        # expire while we are still in cleanup (release / finally) at the timeout edge.
        mutex_acquired = await mutex.acquire_async(
            expire=timeout + timedelta(seconds=10),
            timeout=None,
        )
        if not mutex_acquired:
            await channel.fail_async(
                code="chat_busy",
                message="Chat is already processing",
                close_code=1008,
            )

    transport = FastAPIWebsocketTransport(
        websocket=channel.socket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=ProtobufFrameSerializer(),
            audio_in_sample_rate=input_sample_rate,
            audio_out_sample_rate=output_sample_rate,
        ),
    )

    # ChatChannel must not keep reading; Pipecat owns the socket after ready.
    channel.shutdown()
    try:
        await agent.run_async(
            transport=transport,
            stt=run_stt,
            tts=run_tts,
            vad=run_vad,
            agent_run_repository=chat_repository,
            agent_run_session_id=chat.id,
            tool_retriever=tool_retriever,
            tool_ids=[],
            skill_retriever=skill_retriever,
            skill_ids=[],
            context=context,
        )
    finally:
        if mutex_acquired and mutex is not None:
            await mutex.release_async()
