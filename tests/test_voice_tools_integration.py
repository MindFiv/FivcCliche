"""End-to-end voice session wiring for client tools."""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from fivcplayground.agents import AgentRunSession

from fivccliche.utils.chats.speeches import process_chat_speech_async
from fivccliche.utils.chats.tools import ChatToolSpec


async def test_process_voice_injects_client_tools():
    channel = MagicMock()
    channel.socket = MagicMock(name="socket")
    channel.tool_names = ["pick_file"]
    channel.tool_specs = [
        ChatToolSpec(
            name="pick_file",
            description="Pick a file",
            parameters={
                "type": "object",
                "properties": {"accept": {"type": "string"}},
                "required": ["accept"],
            },
        )
    ]
    channel.send_json_async = AsyncMock()
    channel.mark_prepared = MagicMock()
    channel.begin_streaming = MagicMock()
    channel.fail_async = AsyncMock(side_effect=AssertionError("failure not expected"))

    runnable = MagicMock()
    runnable.run_async = AsyncMock(return_value=MagicMock())
    agent_config = MagicMock(
        model_id="text-llm",
        extra={"e2e_id": "omni-realtime-llm"},
        tool_ids=None,
        skill_ids=None,
        response_model=None,
    )
    model_config = MagicMock(extra={})
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
    chat_provider.get_chat_context.return_value = {}
    chat_provider.get_chat_repository.return_value = MagicMock()
    tool_retriever = MagicMock(tools={})

    with (
        patch(
            "fivcplayground.backends.pipecat.models.PipecatModelBackend",
        ) as model_backend_cls,
        patch(
            "fivcplayground.backends.pipecat.agents.PipecatAgentRunnable",
            return_value=runnable,
        ),
        patch(
            "fivccliche.utils.chats.speeches.create_tool_retriever_async",
            new=AsyncMock(return_value=tool_retriever),
        ),
        patch(
            "fivccliche.utils.chats.speeches.create_skill_retriever_async",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "fivccliche.utils.chats.speeches.VoiceToolTransport",
        ) as transport_cls,
    ):
        model_backend_cls.return_value.create_model.return_value = MagicMock()
        transport_cls.return_value = MagicMock()

        await process_chat_speech_async(
            channel=channel,
            user=MagicMock(uuid="user-1"),
            chat=AgentRunSession(id="chat-1", agent_id="agent-1", context={}),
            timeout=timedelta(minutes=1),
            config_provider=config_provider,
            chat_provider=chat_provider,
        )

    assert "pick_file" in tool_retriever.tools
    assert tool_retriever.tools["pick_file"].get_underlying().properties == {
        "accept": {"type": "string"}
    }
    transport_cls.assert_called_once()
    assert transport_cls.call_args.kwargs["tool_session"].channel is channel
    assert runnable.run_async.await_args.kwargs["tool_ids"] == ["pick_file"]
    ready = channel.send_json_async.await_args.args[0]
    assert ready["event"] == "ready"
