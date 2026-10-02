"""Voice turns reopen finished sockets, wait for speech_end, and warm the agent."""

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fivccliche.modules.agent_chats import services as chat_services
from fivccliche.utils.chats.warmup import SessionWarmup
from fivccliche.services.implements.agent_speeches.dashscope_realtime import (
    _BACKGROUND_CLOSES,
    _DashScopeQwenTTSSocket,
    _DashScopeRealtimeRecognizer,
    _DashScopeRecognitionSocket,
    _DashScopeTTSSynthesizer,
)
from fivccliche.services.interfaces.agent_speeches import (
    SpeechEvent,
    SpeechSynthesisOptions,
)
from fivccliche.utils.chats.processors import ChatVoiceProcessor


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


@pytest.mark.asyncio
async def test_listening_waits_for_audio_end_before_joining_transcript() -> None:
    class _Recognizer:
        async def stream_async(self, audio):
            async for chunk in audio:
                yield SpeechEvent(type="final", text=chunk.decode())

    proc = SimpleNamespace(
        _chat=SimpleNamespace(uuid="chat-1"),
        get_recognizer=lambda: _Recognizer(),
        get_channel=lambda: SimpleNamespace(send_json_async=None),
        get_synthesizer=lambda: SimpleNamespace(),
    )
    phase = ChatVoiceProcessor.ListeningPhase(cast(Any, proc))
    queued: list[object] = []
    enqueue = phase._asr_queue.put_nowait

    def record(item: object) -> None:
        queued.append(item)
        enqueue(item)

    phase._asr_queue.put_nowait = record  # type: ignore[method-assign]
    task = asyncio.create_task(phase._recognize_async())
    record(b"hello")
    await asyncio.sleep(0)
    record(b"world")
    await asyncio.sleep(0)

    assert task.done() is False
    assert None not in queued

    record(None)
    assert await task == "hello\nworld"
    assert queued.count(None) == 1


def _warm_proc(agent_id: str = "agent-1") -> SimpleNamespace:
    dependencies = SimpleNamespace(
        model_backend="model-backend",
        model_repository="model-repo",
        agent_backend="agent-backend",
        agent_repository="agent-repo",
        tool_backend="tool-backend",
        tool_repository="tool-repo",
        embedding_backend="embedding-backend",
        embedding_repository="embedding-repo",
        skill_repository="skill-repo",
    )
    run = SimpleNamespace(
        _dependencies=dependencies,
        _agent_id=agent_id,
        _user_uuid="user-1",
    )

    def create_chat_run(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return run

    return SimpleNamespace(
        _chat=SimpleNamespace(uuid="chat-1", agent_id=agent_id, context={}),
        get_user=lambda: SimpleNamespace(uuid="user-1"),
        get_run_provider=lambda: SimpleNamespace(create_chat_run=create_chat_run),
    )


def _patch_warm_constructors(
    monkeypatch: pytest.MonkeyPatch,
    *,
    agent: Any,
    tools: Any,
    skills: Any,
) -> dict[str, int]:
    calls = {"agent": 0, "tools": 0, "skills": 0}

    async def create_agent(*_args: Any, **_kwargs: Any) -> Any:
        calls["agent"] += 1
        return await agent()

    async def create_tools(*_args: Any, **_kwargs: Any) -> Any:
        calls["tools"] += 1
        return await tools()

    async def create_skills(*_args: Any, **_kwargs: Any) -> Any:
        calls["skills"] += 1
        return await skills()

    monkeypatch.setattr(chat_services, "create_agent_async", create_agent)
    monkeypatch.setattr(chat_services, "create_tool_retriever_async", create_tools)
    monkeypatch.setattr(chat_services, "create_skill_retriever_async", create_skills)
    return calls


async def _load(agent_id: str = "agent-1") -> tuple[Any, Any, Any]:
    run = _warm_proc(agent_id).get_run_provider().create_chat_run()
    prepared = await SessionWarmup.take(run._agent_id)
    if prepared is None:
        prepared = await chat_services.build_chat_collaborators(run)
    return prepared


def _voice_warmup(agent_id: str = "agent-1"):
    run = _warm_proc(agent_id).get_run_provider().create_chat_run()
    return SessionWarmup(run._agent_id, lambda: chat_services.build_chat_collaborators(run)).open()


@pytest.mark.asyncio
async def test_voice_session_warms_agent_stack_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def agent() -> str:
        return "agent"

    async def tools() -> str:
        return "tools"

    async def skills() -> str:
        return "skills"

    calls = _patch_warm_constructors(monkeypatch, agent=agent, tools=tools, skills=skills)

    await _load()
    await _load()
    assert calls["agent"] == 2

    calls["agent"] = 0
    calls["tools"] = 0
    calls["skills"] = 0
    async with _voice_warmup():
        first_agent, first_tools, first_skills = await _load()
        second_agent, second_tools, second_skills = await _load()

    assert calls == {"agent": 1, "tools": 1, "skills": 1}
    assert first_agent == second_agent == "agent"
    assert first_tools == second_tools == "tools"
    assert first_skills == second_skills == "skills"


@pytest.mark.asyncio
async def test_voice_warmup_failure_rebuilds_on_the_next_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def agent() -> str:
        if calls["agent"] == 1:
            raise RuntimeError("warmup failed")
        return f"agent-{calls['agent']}"

    async def tools() -> str:
        return "tools"

    async def skills() -> str:
        return "skills"

    calls = _patch_warm_constructors(monkeypatch, agent=agent, tools=tools, skills=skills)

    async with _voice_warmup():
        first, _tools, _skills = await _load()
        second, _tools, _skills = await _load()

    assert first == "agent-2"
    assert second == "agent-3"
    assert calls["agent"] == 3


@pytest.mark.asyncio
async def test_listening_preopens_tts_before_the_next_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opens: list[str] = []
    release = asyncio.Event()

    async def open_socket(self: _DashScopeTTSSynthesizer) -> None:
        opens.append("start")
        await release.wait()
        self._socket = cast(
            Any,
            SimpleNamespace(
                _finish_sent=False,
                _closed=False,
                _failed=False,
            ),
        )
        opens.append("done")

    async def fake_stream(
        _self: _DashScopeTTSSynthesizer,
        _socket: Any,
        _text: str,
    ) -> AsyncIterator[bytes]:
        yield b"pcm"

    monkeypatch.setattr(_DashScopeTTSSynthesizer, "_open_socket_async", open_socket)
    monkeypatch.setattr(_DashScopeTTSSynthesizer, "_iter_audio_async", fake_stream)

    synthesizer = _DashScopeTTSSynthesizer(
        api_key="test-key",
        model="qwen-tts-realtime",
        options=SpeechSynthesisOptions(voice="Cherry", format="pcm", sample_rate=24000),
    )
    synthesizer._socket = cast(
        Any,
        SimpleNamespace(
            _finish_sent=True,
            _closed=False,
            _failed=False,
        ),
    )
    sent: list[dict[str, Any]] = []

    async def send_json_async(payload: dict[str, Any]) -> None:
        sent.append(payload)

    proc = SimpleNamespace(
        get_channel=lambda: SimpleNamespace(send_json_async=send_json_async),
        get_synthesizer=lambda: synthesizer,
    )
    phase = ChatVoiceProcessor.ListeningPhase(cast(Any, proc))
    await phase.__aenter__()
    assert sent == [{"event": "state", "info": {"state": "listening"}}]
    assert opens == []

    async def release_open() -> None:
        await asyncio.sleep(0)
        release.set()

    opener = asyncio.create_task(release_open())
    try:
        chunks = [chunk async for chunk in synthesizer.stream_async("下一轮")]
        await opener
    finally:
        await synthesizer.cancel_preopen_async()

    assert chunks == [b"pcm"]
    assert opens == ["start", "done"]
