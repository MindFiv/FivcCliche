"""Session warmup is task-local and reused by one text WebSocket turn."""

import asyncio
import json
from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from fivccliche.modules.agent_chats import services as chat_services
from fivccliche.utils.chats.channels import ChatChannelError
from fivccliche.utils.chats.processors import ChatSnapshot, ChatTextProcessor
from fivccliche.utils.chats.warmup import SessionWarmup


@pytest.mark.asyncio
async def test_take_misses_when_the_key_differs() -> None:
    async def factory() -> str:
        return "stack"

    async with SessionWarmup("agent-1", factory).open():
        assert await SessionWarmup.take("agent-1") == "stack"
        assert await SessionWarmup.take("agent-2") is None


@pytest.mark.asyncio
async def test_failed_warmup_is_not_reused() -> None:
    async def factory() -> str:
        raise RuntimeError("warmup failed")

    async with SessionWarmup("agent-1", factory).open():
        assert await SessionWarmup.take("agent-1") is None
        assert await SessionWarmup.take("agent-1") is None


@pytest.mark.asyncio
async def test_exit_cancels_an_unfinished_warmup() -> None:
    started = asyncio.Event()

    async def factory() -> str:
        started.set()
        await asyncio.Event().wait()
        return "stack"

    warm: SessionWarmup[str, str] = SessionWarmup("agent-1", factory)
    async with warm.open():
        await started.wait()
        task = warm._task
        assert task is not None
    assert task.cancelled()
    assert await SessionWarmup.take("agent-1") is None


@pytest.mark.asyncio
async def test_concurrent_warmups_stay_on_their_tasks() -> None:
    release = asyncio.Event()

    async def session(key: str, value: str, started: asyncio.Event) -> str | None:
        async def factory() -> str:
            return value

        async with SessionWarmup(key, factory).open():
            started.set()
            await release.wait()
            return await SessionWarmup.take(key)

    started_a = asyncio.Event()
    started_b = asyncio.Event()
    task_a = asyncio.create_task(session("a", "A", started_a))
    task_b = asyncio.create_task(session("b", "B", started_b))
    await started_a.wait()
    await started_b.wait()
    assert await SessionWarmup.take("a") is None
    assert await SessionWarmup.take("b") is None
    release.set()
    assert await task_a == "A"
    assert await task_b == "B"


def _text_processor(
    channel: Any,
    *,
    chat: ChatSnapshot | None,
    message_timeout: timedelta = timedelta(seconds=5),
) -> ChatTextProcessor:
    async def load_chat(_user: object) -> ChatSnapshot | None:
        return chat

    module_site = MagicMock()
    module_site.get_module.return_value = None
    return ChatTextProcessor(
        channel,
        chat_uuid="chat-1",
        chat_loader=load_chat,
        mutex=None,
        module_site=module_site,
        timeout=timedelta(seconds=5),
        message_timeout=message_timeout,
    )


@pytest.mark.asyncio
async def test_text_websocket_warms_before_the_message_and_reuses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = ("agent", "tools", "skills")
    started = asyncio.Event()
    order: list[str] = []

    async def build(_run: object) -> tuple[str, str, str]:
        order.append("build")
        started.set()
        return stack

    monkeypatch.setattr(chat_services, "build_chat_collaborators", build)

    async def receive_async(raise_exception: bool = False) -> str:
        await started.wait()
        order.append("receive")
        return json.dumps({"type": "message", "query": "Hello"})

    taken: list[object] = []

    async def stream_async(*_args: object, **_kwargs: object):
        order.append("stream")
        taken.append(await SessionWarmup.take("agent-1"))
        if False:
            yield None

    run = SimpleNamespace(stream_async=stream_async)

    async def get_provider() -> SimpleNamespace:
        return SimpleNamespace(create_chat_run=lambda *_args, **_kwargs: run)

    monkeypatch.setattr(
        "fivccliche.utils.chats.processors.get_chat_run_provider_async",
        get_provider,
    )
    channel = SimpleNamespace(
        authenticate_async=_async_value(SimpleNamespace(uuid="user-1")),
        receive_async=receive_async,
        fail_async=_async_value(None),
        send_json_async=_async_value(None),
    )
    processor = _text_processor(
        channel,
        chat=ChatSnapshot("chat-1", "agent-1", {}, "titled"),
    )
    await processor.process_async()

    assert order == ["build", "receive", "stream"]
    assert taken == [stack]


@pytest.mark.asyncio
async def test_text_message_timeout_cancels_warmup_without_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = asyncio.Event()

    async def build(_run: object) -> tuple[str, str, str]:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return ("agent", "tools", "skills")

    monkeypatch.setattr(chat_services, "build_chat_collaborators", build)

    streamed = False

    async def stream_async(*_args: object, **_kwargs: object):
        nonlocal streamed
        streamed = True
        if False:
            yield None

    async def receive_async(raise_exception: bool = False) -> str:
        await asyncio.Event().wait()
        return ""

    async def fail_async(*, code: str, message: str, close_code: int) -> None:
        assert code == "message_timeout"
        raise ChatChannelError(close_code, message)

    run = SimpleNamespace(stream_async=stream_async)

    async def get_provider() -> SimpleNamespace:
        return SimpleNamespace(create_chat_run=lambda *_args, **_kwargs: run)

    monkeypatch.setattr(
        "fivccliche.utils.chats.processors.get_chat_run_provider_async",
        get_provider,
    )
    channel = SimpleNamespace(
        authenticate_async=_async_value(SimpleNamespace(uuid="user-1")),
        receive_async=receive_async,
        fail_async=fail_async,
        send_json_async=_async_value(None),
    )
    processor = _text_processor(
        channel,
        chat=ChatSnapshot("chat-1", "agent-1", {}, "titled"),
        message_timeout=timedelta(milliseconds=20),
    )
    with pytest.raises(ChatChannelError) as exc_info:
        await processor.process_async()

    assert exc_info.value.code == 1008
    assert cancelled.is_set()
    assert streamed is False


def _async_value(value: object):
    async def inner(*_args: object, **_kwargs: object) -> object:
        return value

    return inner
