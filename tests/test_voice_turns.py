"""DashScope realtime ASR/TTS sockets reopen after finish or cancel."""

import asyncio
import json
from typing import Any

import pytest
from fivccliche.services.implements.agent_speeches.dashscope_realtime import (
    _BACKGROUND_CLOSES,
    _DashScopeQwenTTSSocket,
    _DashScopeRealtimeRecognizer,
    _DashScopeRecognitionSocket,
    _DashScopeTTSSynthesizer,
)
from fivccliche.services.interfaces.agent_speeches import (
    SpeechSynthesisOptions,
)


class _FakeWebSocket:
    def __init__(self, frame: str) -> None:
        self.frame = frame
        self.sent: list[str] = []
        self._finished = asyncio.Event()

    async def send(self, payload: str) -> None:
        self.sent.append(payload)
        if "finish-task" in payload or "session.finish" in payload:
            asyncio.get_running_loop().call_soon(self._finished.set)

    async def recv(self) -> str:
        await self._finished.wait()
        return self.frame

    async def close(self) -> None:
        return None


def _finished_frame(event: str) -> str:
    return json.dumps({"header": {"event": event}, "payload": {}})


@pytest.mark.asyncio
async def test_recognizer_reopens_after_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[_DashScopeRecognitionSocket] = []

    async def start_async(self: _DashScopeRecognitionSocket) -> None:
        self._ws = _FakeWebSocket(_finished_frame("task-finished"))
        self._started = True
        started.append(self)

    monkeypatch.setattr(_DashScopeRecognitionSocket, "start_async", start_async)
    recognizer = _DashScopeRealtimeRecognizer(api_key="test-key", connect=None)
    current = _DashScopeRecognitionSocket(
        url=recognizer._ws_url,
        model=recognizer._model,
        headers={},
        options=recognizer._options,
        connect=recognizer._connect,
    )
    current._ws = _FakeWebSocket(_finished_frame("task-finished"))
    current._started = True
    recognizer._socket = current

    async def audio():
        for _chunk in ():
            yield _chunk

    async for _event in recognizer.stream_async(audio()):
        pass
    assert started == []
    assert current._finish_sent is True
    assert current._closed is False

    async for _event in recognizer.stream_async(audio()):
        pass

    assert len(started) == 1
    assert recognizer._socket is started[0]
    assert recognizer._socket is not current


@pytest.mark.asyncio
async def test_recognizer_reopens_after_cancelled_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[_DashScopeRecognitionSocket] = []
    commit_started = asyncio.Event()
    original_commit = _DashScopeRecognitionSocket.commit_async

    async def start_async(self: _DashScopeRecognitionSocket) -> None:
        self._ws = _FakeWebSocket(_finished_frame("task-finished"))
        self._started = True
        started.append(self)

    async def commit_async(self: _DashScopeRecognitionSocket) -> None:
        commit_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(_DashScopeRecognitionSocket, "start_async", start_async)
    monkeypatch.setattr(_DashScopeRecognitionSocket, "commit_async", commit_async)
    recognizer = _DashScopeRealtimeRecognizer(api_key="test-key", connect=None)
    current = _DashScopeRecognitionSocket(
        url=recognizer._ws_url,
        model=recognizer._model,
        headers={},
        options=recognizer._options,
        connect=recognizer._connect,
    )
    current._ws = _FakeWebSocket(_finished_frame("task-finished"))
    current._started = True
    recognizer._socket = current

    async def audio():
        if False:
            yield b""

    async def consume() -> None:
        async for _event in recognizer.stream_async(audio()):
            pass

    task = asyncio.create_task(consume())
    await commit_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert current._closed is True
    assert current._failed is True
    assert started == []

    monkeypatch.setattr(_DashScopeRecognitionSocket, "commit_async", original_commit)
    async for _event in recognizer.stream_async(audio()):
        pass

    assert len(started) == 1
    assert recognizer._socket is started[0]
    assert recognizer._socket is not current


@pytest.mark.asyncio
async def test_synthesizer_reopens_after_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[_DashScopeQwenTTSSocket] = []

    async def start_async(self: _DashScopeQwenTTSSocket) -> None:
        self._ws = _FakeWebSocket(json.dumps({"type": "session.finished"}))
        self._session_updated = True
        started.append(self)

    monkeypatch.setattr(_DashScopeQwenTTSSocket, "start_async", start_async)
    synthesizer = _DashScopeTTSSynthesizer(
        api_key="test-key",
        model="qwen-tts-realtime",
        options=SpeechSynthesisOptions(voice="Cherry", format="pcm", sample_rate=24000),
    )
    current = _DashScopeQwenTTSSocket(
        url=synthesizer._ws_url,
        headers={},
        options=synthesizer._options,
        connect=synthesizer._connect,
    )
    current._ws = _FakeWebSocket(json.dumps({"type": "session.finished"}))
    current._session_updated = True
    synthesizer._socket = current

    async for _chunk in synthesizer.stream_async("你好"):
        pass
    assert started == []
    assert current._finish_sent is True
    assert current._closed is False

    async for _chunk in synthesizer.stream_async("下一轮"):
        pass

    assert len(started) == 1
    assert synthesizer._socket is started[0]
    assert synthesizer._socket is not current


@pytest.mark.asyncio
async def test_synthesizer_reopens_after_cancelled_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[_DashScopeQwenTTSSocket] = []
    finish_started = asyncio.Event()
    original_finish = _DashScopeQwenTTSSocket.finish_async

    async def start_async(self: _DashScopeQwenTTSSocket) -> None:
        self._ws = _FakeWebSocket(json.dumps({"type": "session.finished"}))
        self._session_updated = True
        started.append(self)

    async def finish_async(self: _DashScopeQwenTTSSocket) -> None:
        finish_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(_DashScopeQwenTTSSocket, "start_async", start_async)
    monkeypatch.setattr(_DashScopeQwenTTSSocket, "finish_async", finish_async)
    synthesizer = _DashScopeTTSSynthesizer(
        api_key="test-key",
        model="qwen-tts-realtime",
        options=SpeechSynthesisOptions(voice="Cherry", format="pcm", sample_rate=24000),
    )
    current = _DashScopeQwenTTSSocket(
        url=synthesizer._ws_url,
        headers={},
        options=synthesizer._options,
        connect=synthesizer._connect,
    )
    current._ws = _FakeWebSocket(json.dumps({"type": "session.finished"}))
    current._session_updated = True
    synthesizer._socket = current

    async def consume() -> None:
        async for _chunk in synthesizer.stream_async("你好"):
            pass

    task = asyncio.create_task(consume())
    await finish_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert current._closed is True
    assert current._failed is True
    assert started == []

    monkeypatch.setattr(_DashScopeQwenTTSSocket, "finish_async", original_finish)
    async for _chunk in synthesizer.stream_async("下一轮"):
        pass

    assert len(started) == 1
    assert synthesizer._socket is started[0]
    assert synthesizer._socket is not current


@pytest.mark.asyncio
async def test_cancelled_stream_does_not_wait_for_socket_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finish_started = asyncio.Event()
    close_started = asyncio.Event()

    async def finish_async(self: _DashScopeQwenTTSSocket) -> None:
        finish_started.set()
        await asyncio.Event().wait()

    async def close_async(self: _DashScopeQwenTTSSocket, *_args: Any, **_kwargs: Any) -> None:
        close_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(_DashScopeQwenTTSSocket, "finish_async", finish_async)
    monkeypatch.setattr(_DashScopeQwenTTSSocket, "close_async", close_async)
    synthesizer = _DashScopeTTSSynthesizer(
        api_key="test-key",
        model="qwen-tts-realtime",
        options=SpeechSynthesisOptions(voice="Cherry", format="pcm", sample_rate=24000),
    )
    current = _DashScopeQwenTTSSocket(
        url=synthesizer._ws_url,
        headers={},
        options=synthesizer._options,
        connect=synthesizer._connect,
    )
    current._ws = _FakeWebSocket(json.dumps({"type": "session.finished"}))
    current._session_updated = True
    synthesizer._socket = current

    async def consume() -> None:
        async for _chunk in synthesizer.stream_async("你好"):
            pass

    task = asyncio.create_task(consume())
    await finish_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert current._closed is True
    assert close_started.is_set()
    for pending in list(_BACKGROUND_CLOSES):
        pending.cancel()
    await asyncio.gather(*_BACKGROUND_CLOSES, return_exceptions=True)
