# Chat frontend tools (client protocol)

Frontend tools execute in the browser / app UI and are advertised per WebSocket
session. They are not stored as `UserTool` rows. SSE
`POST /api/chats/{uuid}/messages/` does not support them. Voice
`WEBSOCKET /api/chats/{chat_uuid}/voice/` supports them over the same socket;
after `ready`, RPC envelopes are carried inside Pipecat Protobuf message frames.

Shared JSON envelope (text WS frames and voice Protobuf message payloads):

```json
{
  "v": 1,
  "type": "register_tools | tools_ready | tool_call | tool_result | user_message",
  "id": "rpc id when needed",
  "payload": {}
}
```

## Text: `WEBSOCKET /api/chats/{chat_uuid}/text/`

1. Client `{"type":"auth","access_token":"..."}`
2. Client `{"type":"register_tools","payload":{"tools":[{name,description,parameters}]}}` (list may be empty)
3. Server `{"type":"tools_ready","payload":{"tools":[...],"warnings":[...]}}`
4. Client `{"type":"user_message","payload":{"query":"..."}}`
5. Server streams `{"event":"start|stream|tool|finish|error","info":...}` (same shape as SSE JSON payloads).
   During streaming the server keeps reading the socket for client `tool_result` frames
   (other inbound types are ignored while a tool call is pending, otherwise the
   connection fails with `invalid_message`).
6. If the model calls a client tool: server `tool_call` → client `tool_result`
7. Connection closes after one turn

## Tool item

- `name`: `^[A-Za-z_][A-Za-z0-9_]{0,63}$`
- `description`: string
- `parameters`: JSON Schema object (flat `object` with identifier property names;
  `$ref` / `anyOf` / non-identifier keys are not expanded into the model tool
  signature)

Registration limits:

- Up to 64 tools per chat WebSocket session. Exceeding the limit closes the
  socket with `invalid_message`.
- Up to 64 declared properties per tool. Invalid or oversized parameter schemas
  are skipped and reported in `tools_ready.warnings`.
- Property names must be Python identifiers and cannot be Python keywords.
- `required` entries must refer to declared properties.

Default client-tool wait is 30 seconds.
The server accepts at most 16 concurrently pending client tool calls.

For transport details, see [`voice-frontend-tools.md`](voice-frontend-tools.md).

Multiple text/voice WebSockets may connect to the same `chat_uuid`. Concurrent
agent runs are still rejected by the `chats:message:{uuid}` mutex
(`chat_busy` / close `1008` on WS, HTTP 409 on SSE).
