"""Pipecat WebRTC sessions for full-duplex chat voice."""

from __future__ import annotations

import logging
from datetime import timedelta

from fivcglue.interfaces.mutexes import IMutex
from fivcplayground.agents import AgentRunSession, create_agent_async
from fivcplayground.skills import create_skill_retriever_async
from fivcplayground.speeches import SpeechRecognizer, SpeechSynthesizer
from fivcplayground.tools import create_tool_retriever_async
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import SmallWebRTCRequestHandler
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from fivccliche.services.interfaces.agent_chats import IUserChatProvider
from fivccliche.services.interfaces.agent_configs import IUserConfigProvider
from fivccliche.services.interfaces.auth import IUser
from fivccliche.utils.deps import get_chat_provider_async, get_config_provider_async

logger = logging.getLogger(__name__)

INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000


class ChatWebRTCRegistry:
    """Process-local WebRTC peers so ICE patches can find the connection."""

    def __init__(self) -> None:
        self.handler = SmallWebRTCRequestHandler()
        self.owners: dict[str, tuple[str, str]] = {}

    def remember(self, pc_id: str, user_uuid: str, chat_uuid: str) -> None:
        self.owners[pc_id] = (user_uuid, chat_uuid)

    def owned_by(self, pc_id: str, user_uuid: str, chat_uuid: str) -> bool:
        return self.owners.get(pc_id) == (user_uuid, chat_uuid)

    def forget(self, pc_id: str) -> None:
        self.owners.pop(pc_id, None)


async def process_chat_speech_async(
    *,
    connection: SmallWebRTCConnection,
    user: IUser,
    chat: AgentRunSession,
    recognizer: SpeechRecognizer,
    synthesizer: SpeechSynthesizer,
    mutex: IMutex | None = None,
    timeout: timedelta = timedelta(minutes=5),
    config_provider: IUserConfigProvider | None = None,
    chat_provider: IUserChatProvider | None = None,
) -> None:
    """Run a Pipecat WebRTC pipeline until the peer disconnects."""
    from fivcplayground.backends.pipecat.agents import PipecatAgentBackend
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
    agent_config = await agent_repository.get_agent_config_async(chat.agent_id)
    if agent_config is None:
        logger.error("Agent config %s not found", chat.agent_id)
        await connection.disconnect()
        return
    model_config = await model_repository.get_model_config_async(agent_config.model_id)
    if model_config is None:
        logger.error("LLM config %s not found", agent_config.model_id)
        await connection.disconnect()
        return

    embedding_repository = config_provider.get_embedding_repository(user_uuid=user.uuid)
    embedding_backend = config_provider.get_embedding_backend()
    tool_repository = config_provider.get_tool_repository(user_uuid=user.uuid)
    skill_repository = config_provider.get_skill_repository(user_uuid=user.uuid)

    agent = await create_agent_async(
        model_backend=model_backend,
        model_config_repository=model_repository,
        agent_backend=agent_backend,
        agent_config_repository=agent_repository,
        agent_config_id=chat.agent_id,
    )
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
            logger.warning("Chat %s is already processing", chat.id)
            await connection.disconnect()
            return

    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=INPUT_SAMPLE_RATE,
            audio_out_sample_rate=OUTPUT_SAMPLE_RATE,
        ),
    )
    try:
        await agent.run_async(
            transport=transport,
            stt=recognizer,
            tts=synthesizer,
            vad=SileroVADAnalyzer(sample_rate=INPUT_SAMPLE_RATE),
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
