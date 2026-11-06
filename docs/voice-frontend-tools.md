# Voice frontend tools

Voice client tools use the same registration envelope as text tools, but the
RPC is carried inside Pipecat Protobuf `message` frames after `ready`.

## Lifecycle

1. Client sends JSON JWT `auth`.
2. Optionally send JSON `register_tools`. Omitting this frame preserves the
   legacy voice handshake and registers no client tools; an empty list is also
   valid.
3. Server sends JSON `tools_ready`, then the client sends JSON `start`.
4. Server sends JSON `ready`, then both directions use Pipecat Protobuf frames.
   Raw JSON must not be mixed into the socket after this point.
5. `tool_call` is sent as an urgent Pipecat output message. The client sends
   `tool_result` in a Protobuf message frame. The inner message uses the shared
   `v:1` envelope.

Registration is immutable for the voice socket after `start`. Both cascade and
end-to-end realtime agents receive the registered tools. Pipecat treats calls as
async so audio conversation is not blocked while the frontend executes a tool;
the 30 second default timeout and 16 pending-call limit still apply.
