# Speech recognition and synthesis

FivcCliche exposes a multi-user speech factory so chat and tools can transcribe
and synthesize audio. Recognition and synthesis types come from
`fivcplayground.speeches` (`SpeechRecognizer`, `SpeechSynthesizer`,
`SpeechRecognizeOption`, `SpeechSynthesisOption`, `SpeechRecognizeEvent`,
`SpeechError`). `ISpeechProvider` creates those objects from a caller-supplied
`api_key`, `model`, and `base_url`. `stream_async` accepts
`bytes | AsyncIterator[bytes]`. HTTP probes and `SpeechTranscribe` decode
Base64 to `bytes` before calling; audio fragment URLs are not supported.
Vendor failures raise `SpeechError`.

Default YAML registers two DashScope providers: `dashscope` for native HTTP and
`dashscope_realtime` for the DashScope SDK streaming ASR and TTS. Both expose
recognizers and synthesizers.

Full-duplex chat voice is WebSocket signaling plus Pipecat SmallWebRTC media
(`WEBSOCKET /api/chats/{chat_uuid}/`).
[`process_chat_speech_async`](../src/fivccliche/utils/chats/speeches.py) passes
the playground recognizer and synthesizer as `stt` / `tts` and a Silero
`SileroVADAnalyzer` as `vad`. Probe routes and `SpeechTranscribe` still call
`stream_async` directly. Text chat SSE does not use speech.

A Pipecat OpenAI model with `base_url` set defaults `supports_developer_role` to
false, because compatible gateways reject the `developer` role. Set
`ModelConfig.extra["supports_developer_role"]` to true to turn it back on.

## Status

- `ISpeechProvider` (DI + `get_speech_provider_async`) plus playground speech types
- Fake provider for tests (`FakeSpeechProvider`)
- DashScope HTTP provider and DashScope SDK realtime provider
- Function tool `SpeechTranscribe` for `transport=function` wiring
- DashScope TTS and `UserTTS` / `UserASR` probe configuration
- Chat voice: Pipecat SmallWebRTC, with VAD passed through `vad=`

The native HTTP `dashscope` provider supports:

- `qwen3-asr-flash` using DashScope multimodal `asr_options`
- `qwen-audio-3.0-asr-flash` using its native `input_audio` request and
  `parameters.format` / `parameters.sample_rate`; context, hotwords, language, and ITN options are not
  sent for this protocol
- Qwen-Audio HTTP TTS with `qwen-audio-3.0-tts-plus`,
  `qwen-audio-3.1-tts-flash`, and `qwen-audio-3.0-tts-flash`. Other TTS model
  families are rejected by this provider.

The `dashscope_realtime` provider uses `dashscope.audio.asr.Recognition`
(default model `qwen-audio-3.1-asr-flash-streaming`, PCM 16 kHz) and
`dashscope.audio.tts_v2.SpeechSynthesizer` (default model
`qwen-audio-3.1-tts-flash`, voice `longanhuan_v3.1`, PCM 24 kHz). The
recognizer implements `begin_utterance` / `push_audio` / `end_utterance` /
`cancel_utterance` so playground's Pipecat STT service can push audio while the
user is speaking. `stream_async` uses the same utterance API with
`bytes | AsyncIterator[bytes]` and sends PCM frames. Fun-ASR realtime,
Paraformer realtime, and asynchronous file transcription are not integrated.
Recognition sends audio format, sample rate, and the resolved WebSocket URL;
language, ITN, hotwords, and context stay on `SpeechRecognizeOption` and are not
forwarded to the SDK.

## Interfaces

Playground types live in `fivcplayground.speeches`. This repository adds the
factory in `src/fivccliche/services/interfaces/agent_speeches.py`:

- `ISpeechProvider.get_recognizer(options=None, *, api_key=None, model=None, base_url=None, id=...)`
  returns a `SpeechRecognizer`. Optional `api_key` / `model` / `base_url` come
  from a database config row. `id` is that row's config id.
- `SpeechRecognizer` is an async context manager. `id` and `get_option()`
  expose the config id and the captured `SpeechRecognizeOption` (`hot_words`,
  not `hotwords`). `stream_async` yields `SpeechRecognizeEvent` (`partial` /
  `final` / `error`).
- `ISpeechProvider.get_synthesizer(...)` returns a `SpeechSynthesizer`.
  `stream_async(text)` yields encoded audio bytes; `text` is a string or an
  `AsyncIterator[str]`.
- `SpeechSynthesisOption` is captured at factory time. An empty `voice` selects
  the provider default. Format, sample rate, volume, speech rate, and pitch are
  request parameters and are not persisted on `UserTTS`.
- HTTP probe bodies keep the field name `hotwords`. Callers map it to
  `SpeechRecognizeOption.hot_words`. ASR probe bodies require `data_b64`
  (decoded to `bytes` before recognition).
- Vendor failures raise `SpeechError`.

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

Chat voice does not enter the recognizer in the route. Playground enters and
exits it when the Pipecat pipeline starts and cleans up.

## Configuration

Default [`services.yml`](../src/fivccliche/settings/services.yml) registers the
two DashScope providers:

```yaml
- entries:
    - interface: fivccliche.services.interfaces.agent_speeches.ISpeechProvider
      name: dashscope
  class: fivccliche.services.implements.agent_speeches.dashscope.DashScopeSpeechProvider

- entries:
    - interface: fivccliche.services.interfaces.agent_speeches.ISpeechProvider
      name: dashscope_realtime
  class: fivccliche.services.implements.agent_speeches.dashscope_realtime.DashScopeRealtimeSpeechProvider

```

`get_speech_provider_async(name)` looks up a named implementation.
`SpeechTranscribe` and the ASR probe do **not** use a hard-coded name: they
load a [`UserASR`](../src/fivccliche/modules/agent_configs/models.py) row
(`POST /api/configs/asrs/`) and pass `model_type` as `name`. If that row or
provider is missing, WebRTC voice returns 404 or 503, `SpeechTranscribe` raises
`ValueError`, and the HTTP probe returns 503.

`POST /api/configs/asrs/{config_uuid}/probe/` transcribes one clip against that
config (required `data_b64`, optional `format` / `language` / `hotwords` /
`context`) as SSE (`text/event-stream`), one `SpeechRecognizeEvent` per `data:`
line:

```
data: {"event": "partial", "info": {"text": "...", "language": "zh"}}

data: {"event": "final", "info": {"text": "...", "language": "zh"}}

data: {"event": "error", "info": {"message": "..."}}
```

Missing config is 404; missing provider is 503. Recognition failures after the
stream starts are SSE `error` events (HTTP 200), not 400.

`dashscope_realtime` resolves `base_url` to the DashScope inference WebSocket
(`wss://.../api-ws/v1/inference`). An HTTP origin becomes that socket; a
`ws` / `wss` URL that already has a path is kept. `Recognition` receives
`api_key` and `base_address` per session. `tts_v2` reads `dashscope.api_key`
when the client is constructed and accepts `url=`, so the provider sets the key
only inside a lock, passes the resolved URL, then restores the previous key.
It does not call playground's process-global WebSocket URL helper. An empty
realtime API key raises `SpeechError`. The HTTP provider still allows an empty
key.

Realtime TTS passes `volume`, `speech_rate`, and `pitch_rate` from
`SpeechSynthesisOption`. PCM sample rates supported by the SDK format enum are
8000, 16000, 22050, 24000, 44100, and 48000. An empty voice selects
`longanhuan_v3.1`.

`dashscope` defaults to `qwen-audio-3.0-tts-flash` and POSTs to
`/api/v1/services/audio/tts/SpeechSynthesizer`. It uses a non-streaming task
response, downloads `output.audio.url`, and yields the downloaded audio in
12800-byte chunks. An empty voice selects `longanhuan_v3.1` for
`qwen-audio-3.1-*` and `longanhuan_v3.6` for `qwen-audio-3.0-*`.

TTS is exposed through [`UserTTS`](../src/fivccliche/modules/agent_configs/models.py)
(`POST /api/configs/tts/`). Persisted fields are `id`, `description`, `model`,
`base_url`, `api_key`, and `model_type`; `model_type` is always
`dashscope` or `dashscope_realtime`. Deployments migrating from the former
standalone `dashscope_tts` registration must run `fivccliche migrate`; it
changes that legacy value to `dashscope_realtime`. Voice and audio-quality
parameters stay in the probe request. `POST
/api/configs/tts/{config_uuid}/probe/` accepts `text` and optional `voice` /
audio options; an empty voice lets the provider select a model-compatible
default. Other probe defaults are PCM and 24000 Hz.
Probe audio is emitted as Base64 SSE:

```
data: {"event": "audio", "info": {"data_b64": "..."}}

data: {"event": "complete", "info": {}}
```

Missing config is 404, missing provider is 503, and synthesis failures after
the stream starts are SSE `error` events.

`dashscope_realtime` configs can also probe through
`WEBSOCKET /api/configs/tts/{config_uuid}/probe/` and
`WEBSOCKET /api/configs/asrs/{config_uuid}/probe/`. The ASR socket still feeds a
PCM async iterator into `stream_async`. The TTS socket still consumes
`stream_async(text)`. After first-frame JWT authentication, the TTS probe
expects one `start` JSON frame:

```json
{"type":"start","text":"你好","voice":"","format":"pcm","sample_rate":24000,"volume":50,"speech_rate":1.0,"pitch_rate":1.0}
```

The server emits JSON `{event, info}` events: `ready`, Base64 `audio`, then
`complete`; it closes with `1000` on success. It closes `4404` for a missing
config, `1003` for a non-realtime config or invalid start frame, `1013` when
the provider is not mounted, and `1011` for synthesis failure. Errors use
`{"event":"error","info":{"code":"...","message":"..."}}`.

ASR and TTS credentials are caller-supplied. Production callers load
`UserASR` / `UserTTS` rows from the database and pass their values to the
provider factory:

```python
speech_provider.get_recognizer(
    options,
    api_key=asr.api_key,
    model=asr.model,
    base_url=asr.base_url,
    id=asr.id,
)
```

| Argument | Default when `None` | Meaning |
|-----|---------|---------|
| `api_key` | empty | DashScope API key. Realtime rejects an empty key |
| `model` | HTTP: `qwen3-asr-flash`; Realtime: `qwen-audio-3.1-asr-flash-streaming` | Model id for that provider |
| `base_url` | `https://dashscope.aliyuncs.com` | HTTP origin or complete `ws://` / `wss://` endpoint. HTTP ASR derives its generation URL. Realtime derives `wss://.../api-ws/v1/inference` |

TTS uses the same arguments from a `UserTTS` row:

| Argument | Default when `None` | Meaning |
|-----|---------|---------|
| `api_key` | empty | DashScope API key. Realtime rejects an empty key |
| `model` | HTTP: `qwen-audio-3.0-tts-flash`; Realtime: `qwen-audio-3.1-tts-flash` | Model id for that provider |
| `base_url` | `https://dashscope.aliyuncs.com` | HTTP origin or complete `ws://` / `wss://` endpoint. HTTP TTS derives the SpeechSynthesizer URL. Realtime derives `wss://.../api-ws/v1/inference` and passes it as `url=` |

For tests, instantiate `FakeSpeechProvider(transcript="...")` directly.

## Chat voice

`WEBSOCKET /api/chats/{chat_uuid}/` uses [`ChatChannel`](../src/fivccliche/utils/chats/channels.py)
first-frame JWT `auth`, then an `offer` JSON frame (`sdp`, `sdp_type`, optional
`pc_id` / `restart_pc`, plus `asr_id` / `tts_id` / `language` / `voice`).
Missing ids use `chat.context`, then `"realtime"`. Both rows must use
`model_type=dashscope_realtime`. The server replies with
`{"event":"answer","info":{...}}` (SDP answer plus `asr_id`, `tts_id`, and
input/output format and sample rate). Later `ice` frames carry trickle ICE
candidates for the `pc_id` created on that socket.

The session task acquires `chats:message:{uuid}`, then runs
`agent.run_async` with `SmallWebRTCTransport`, the SDK recognizer, the SDK
synthesizer, and `vad=SileroVADAnalyzer(sample_rate=16000)`.

## Agent tool

Callable class in `src/fivccliche/modules/agent_chats/tools.py`. Attach it with
a tool config (`transport=function`) whose `functions` point at the dotted path:

- `fivccliche.modules.agent_chats.tools.SpeechTranscribe`

`SpeechTranscribe` takes `**context` (expects `user_uuid`; optional `asr_id`,
default `"default"`). It loads that UserASR row, calls
`get_speech_provider_async(model_type)`, then `get_recognizer` with
`api_key` / `model` / `base_url`, decodes `data_b64` to `bytes`, and calls
`stream_async`. Arguments: required `data_b64`, optional `audio_format`,
`language`, `hotwords`, `context`. `hotwords` is mapped to
`SpeechRecognizeOption.hot_words`. Returns JSON `{ "text", "language" }` from
the last `final` event.

Streaming duplex audio is the WebRTC session above, not this tool.
