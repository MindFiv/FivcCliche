# Speech recognition and synthesis

FivcCliche exposes an implementation-agnostic speech contract so chat and tools
can transcribe audio without depending on a vendor SDK. `ISpeechProvider` is a
factory; recognition runs on `ISpeechRecognizer` and one-shot synthesis runs on
`ISpeechSynthesizer`. Default YAML registers DashScope Flash, Flash-Realtime,
and DashScope TTS under distinct names. Continuous duplex voice and automatic
TTS for chat replies are out of scope for this phase.

## Status

- Interface + optional `ISpeechProvider` (DI + `get_speech_provider_async`)
- Fake provider for tests (`FakeSpeechProvider`)
- DashScope Flash and Realtime providers (each file is one implementation)
- Message WebSocket can accept one audio clip or one streamed utterance, then
  runs the existing text `ChatQueryJob`
- Function tool `SpeechTranscribe` for later `transport=function` wiring
- DashScope TTS and `UserTTS` probe configuration

The synchronous `dashscope` provider supports:

- `qwen3-asr-flash` using DashScope multimodal `asr_options`
- `qwen-audio-3.0-asr-flash` using its native `input_audio` request and
  `parameters.format`; context, hotwords, language, and ITN options are not
  sent for this protocol

The `dashscope_realtime` provider currently supports the DashScope Recognition
WebSocket protocol with `qwen-audio-3.1-asr-flash-streaming` and the DashScope
TTS WebSocket protocol. Fun-ASR realtime,
Paraformer realtime, and asynchronous file transcription are not integrated
yet. Recognition currently sends only audio format, sample rate, and
`SpeechRecognizeOptions.extra`; language, ITN, hotwords, and context are not
sent for this protocol.

## Interfaces

Defined in `src/fivccliche/services/interfaces/speech.py`:

- `ISpeechProvider.get_recognizer(options=None, *, api_key=None, model=None, base_url=None)`
  — abstract factory. Optional `api_key` / `model` / `base_url` override the
  SPEECH config when not `None`.
- `ISpeechRecognizer` is an async context manager (`async with`)
- `ISpeechRecognizer.stream_async(audio)` — async iterator of `SpeechEvent`
  (`partial` / `final` / `error`)
- `ISpeechProvider.get_synthesizer(options=None, *, api_key=None, model=None, base_url=None)`
  — abstract factory; providers that do not support synthesis reject it with
  `SpeechRequestError`
- `ISpeechSynthesizer` is an async context manager and
  `stream_async(text)` is an async iterator of encoded audio bytes; `text` is
  either a string or an `AsyncIterator[str]`
- `SpeechSynthesisOptions` are passed at factory time. A provider may apply
  defaults when they are omitted; when supplied, `voice` is required, and
  format, sample rate, volume, speech rate, pitch, and vendor extras are
  request parameters and are not persisted on `UserTTS`.
- `audio` is either `SpeechAudioInput` (URL or Base64 clip) or
  `AsyncIterator[bytes]` (one utterance; exhaustion is the commit)
- `SpeechRecognizeOptions` are passed at factory time (language, ITN, hotwords,
  context, PCM format hints)
- Vendor HTTP/WS failures raise `SpeechRequestError`

Callers do not choose clip vs stream at the recognizer API. Lookup a named
provider via `get_speech_provider_async`. Flash buffers a byte stream and POSTs
it; URL clips are passed through to DashScope. Realtime constructs in
`get_recognizer` and opens the vendor session in `__aenter__`. Realtime converts
`data_b64` to bytes and fetches `url` before sending audio as binary frames in
12800-byte chunks. Business code should not import DashScope response types.

```python
async with await speech_provider.get_recognizer(options) as recognizer:
    async for event in recognizer.stream_async(source):
        ...
```

```python
async with await speech_provider.get_synthesizer(options) as synthesizer:
    async for audio_chunk in synthesizer.stream_async("你好"):
        ...
```

## Configuration

Default [`services.yml`](../src/fivccliche/settings/services.yml) registers the
DashScope providers under distinct names:

```yaml
- entries:
    - interface: fivccliche.services.interfaces.speech.ISpeechProvider
      name: dashscope
  class: fivccliche.services.implements.speech.dashscope.DashScopeSpeechProvider

- entries:
    - interface: fivccliche.services.interfaces.speech.ISpeechProvider
      name: dashscope_realtime
  class: fivccliche.services.implements.speech.dashscope_realtime.DashScopeRealtimeSpeechProvider

- entries:
    - interface: fivccliche.services.interfaces.speech.ISpeechProvider
      name: dashscope_tts
  class: fivccliche.services.implements.speech.dashscope_realtime.DashScopeRealtimeSpeechProvider
```

`get_speech_provider_async(name)` looks up a named DashScope implementation.
`dashscope_realtime` and `dashscope_tts` are compatibility-oriented registration
names backed by the same realtime provider class; their defaults remain
separated into `ASR_*` and `TTS_*` settings.
Chat WebSocket ASR, `SpeechTranscribe`, and ASR probe do **not** use the default
name directly: they load a [`UserASR`](../src/fivccliche/modules/agent_configs/models.py)
row (`POST /api/configs/asrs/`) and pass `model_type` as `name`. If that row or
provider is missing, WebSocket audio fails with `speech_unavailable`,
`SpeechTranscribe` raises `ValueError`, and probe returns 503.

`POST /api/configs/asrs/{config_uuid}/probe/` transcribes one clip against that
config (`url` or `data_b64`, optional `format` / `language` / `hotwords` /
`context`) as SSE (`text/event-stream`), one `SpeechEvent` per `data:` line:

```
data: {"event": "partial", "info": {"text": "...", "language": "zh"}}

data: {"event": "final", "info": {"text": "...", "language": "zh"}}

data: {"event": "error", "info": {"message": "..."}}
```

Missing config is 404; missing provider is 503. Recognition failures after the
stream starts are SSE `error` events (HTTP 200), not 400.

DashScope WebSocket requests send one instance-specific Bearer API key. For a
MaaS host shaped `<workspace>.<region>.maas.aliyuncs.com`, they also send
`X-DashScope-WorkSpace: <workspace>`. The public `dashscope.aliyuncs.com`
endpoint does not receive a workspace header. Handshake failures include the
DashScope response body when the gateway supplies one; API keys are never
included. A 403 normally means the key was recognized but the workspace, model,
or region is not authorized—verify those account settings before changing the
URL.

`dashscope_tts` uses the DashScope TTS WebSocket protocol and defaults to
`qwen-audio-3.1-tts-flash`. It converts an HTTP base URL to
`wss://.../api-ws/v1/inference` using the same rule as realtime ASR. A complete
string is submitted in one synthesis turn; string chunks are streamed and
finished with `finish-task`.
`qwen-audio-3.1-tts-flash` defaults to `longanhuan_v3.1`; `Cherry` is not
compatible with this model and is rejected before opening a synthesis task.

TTS is exposed through [`UserTTS`](../src/fivccliche/modules/agent_configs/models.py)
(`POST /api/configs/tts/`). Persisted fields are `id`, `description`, `model`,
`base_url`, `api_key`, and `model_type`; `model_type` is always
`dashscope_tts` in this phase. Voice and audio-quality parameters stay in the
probe request. `POST /api/configs/tts/{config_uuid}/probe/` accepts `text` and
optional `voice` / audio options; voice defaults to `longanhuan_v3.1`. Probe
audio is emitted as Base64 SSE:

```
data: {"event": "audio", "info": {"data_b64": "..."}}

data: {"event": "complete", "info": {}}
```

Missing config is 404, missing provider is 503, and synthesis failures after
the stream starts are SSE `error` events.

User ASR configs (`model`, `base_url`, `api_key`, `model_type`) override the
shared `.env.json` session `SPEECH` when passed to `get_recognizer`:

```json
{
  "SPEECH": {
    "ASR_API_KEY": "sk-...",
    "ASR_MODEL": "qwen3-asr-flash",
    "ASR_BASE_URL": "https://dashscope.aliyuncs.com"
  }
}
```

| Key | Default | Meaning |
|-----|---------|---------|
| `ASR_API_KEY` | empty | DashScope API key |
| `ASR_MODEL` | Flash: `qwen3-asr-flash`; Realtime: `qwen-audio-3.1-asr-flash-streaming` | Model id for that provider |
| `ASR_BASE_URL` | `https://dashscope.aliyuncs.com` | HTTP origin or complete `ws://` / `wss://` endpoint; Flash derives its generation URL, Realtime derives `wss://.../api-ws/v1/inference` from an HTTP origin |

TTS uses the shared `.env.json` session `SPEECH` when a `UserTTS` row does not
supply an override:

```json
{
  "SPEECH": {
    "TTS_API_KEY": "sk-...",
    "TTS_MODEL": "qwen-audio-3.1-tts-flash",
    "TTS_BASE_URL": "https://dashscope.aliyuncs.com"
  }
}
```

| Key | Default | Meaning |
|-----|---------|---------|
| `TTS_API_KEY` | empty | DashScope API key |
| `TTS_MODEL` | `qwen-audio-3.1-tts-flash` | DashScope TTS model id |
| `TTS_BASE_URL` | `https://dashscope.aliyuncs.com` | HTTP origin or complete `ws://` / `wss://` inference endpoint; an HTTP origin derives `wss://.../api-ws/v1/inference` |

For tests, instantiate `FakeSpeechProvider(transcript="...")` directly.

## Message WebSocket

Existing one-turn protocol on `WS /api/chats/{chat_uuid}/messages/ws/` is
unchanged for text. After auth, the `message` frame is **one of**:

- `{"type": "message", "query": "..."}` — existing text path
- `{"type": "message", "audio": "<url-or-base64>", "format": "wav"}` — clip
- `{"type": "message", "audio_stream": true, "format": "pcm"}` then binary PCM
  frames then `{"type": "audio_commit"}` — one streamed utterance

`query` / `audio` / `audio_stream` are mutually exclusive. Chat `context.asr_id`
selects the UserASR `id` (default `"default"`). Recognition emits
`{"event": "transcript", "info": {"text": "...", "is_final": true|false}}`, then
the usual `start` / `stream` / `tool` / `finish` text events. The connection
still closes after one agent turn.

Error codes: `speech_unavailable` (1011), `asr_failed` (1011),
`empty_transcript` (1003). SSE `POST /messages/` does not accept audio.

Streaming WS uses manual commit: the PCM generator ends on `audio_commit`.

## Agent tool

Callable class in `src/fivccliche/modules/agent_chats/tools.py`. Attach it with
a tool config (`transport=function`) whose `functions` point at the dotted path:

- `fivccliche.modules.agent_chats.tools.SpeechTranscribe`

`SpeechTranscribe` takes `**context` (expects `user_uuid`; optional `asr_id`,
default `"default"`). It loads that UserASR row, calls
`get_speech_provider_async(model_type)`, then `get_recognizer` with
`api_key` / `model` / `base_url`, then `stream_async` with a `SpeechAudioInput`.
Arguments: `url` or `data_b64`, optional `audio_format`, `language`, `hotwords`,
`context`. Returns JSON `{ "text", "language" }` from the last `final` event.

There is no separate listen tool. Streaming ASR is used by the chat WebSocket.

## Later (not implemented)

Reuse `ISpeechRecognizer.stream_async` with a long-lived PCM generator (no
exhaustion until the voice session ends) for duplex turns. Do not turn the
current one-shot WS into a duplex connection in this phase.
