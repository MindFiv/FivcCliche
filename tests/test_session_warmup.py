"""Session warmup is task-local."""

import asyncio

import pytest

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
