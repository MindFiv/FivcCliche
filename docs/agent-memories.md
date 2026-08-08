# Agent Memories (Hindsight)

FivcCliche exposes an implementation-agnostic memory contract so business code
can store and recall user memories without depending on a specific backend.
The default backend is [Hindsight](https://github.com/vectorize-io/hindsight)
via `hindsight-client`.

## Status

- Interface + Hindsight provider (DI + `get_memory_provider_async`)
- Scheduled **chat-level** memorize job in `agent_chats` (see below)
- Read-only HTTP API in `agent_memories` (`GET /memories/`, `GET /memories/recall/`)
- `UserChatMessage.is_memorized` is written by the job after a successful retain
  (or when there is nothing to retain); it is not exposed on the public API yet

## Interfaces

Defined in `src/fivccliche/services/interfaces/agent_memories.py`:

| Type | Role |
|------|------|
| `IUserMemoryProvider` | Factory: `get_memory(space_id=...)` → `IUserMemory` |
| `IUserMemory` | Per-space API: `retain_async` / `recall_async` / `list_async` |
| `MemoryContent` | Normalized recalled/listed item |
| `MemoryRetainResult` / `MemoryRecallResult` / `MemoryListResult` | Normalized operation outcomes |

Business code should depend only on these types, never on Hindsight-native
response objects (`raw` is an escape hatch).

`list_async(*, skip=0, limit=100, **kwargs)` returns `MemoryListResult` with
`items` and `total`. Extra `kwargs` are backend-specific (e.g. Hindsight
`type` / `search_query`) and are not exposed by the HTTP module.

## Hindsight implementation

`UserMemoryProviderImpl` (`agent_memories_hindsight.py`):

- Creates the `Hindsight` SDK client lazily on first `get_memory` call
- Maps `space_id` directly to Hindsight `bank_id` (`None` → `"default"`)
- Banks are created automatically by Hindsight on first retain/recall
- `list_async` calls `client.memory.list_memories` (`offset=skip`)

Chat memorize job uses `space_id=chat.user_uuid`.

## HTTP API (`agent_memories` module)

Mounted under `/api` (same as other modules). Requires a Bearer JWT.

| Method | Path | Behavior |
|--------|------|----------|
| `GET` | `/memories/` | Paginated list (`skip` / `limit`) → `{ total, results }` |
| `GET` | `/memories/recall/?query=` | Semantic recall → `{ results }` |

`space_id` is always the authenticated user's `uuid`.

If `IUserMemoryProvider` is not registered, both endpoints return
**503** with `detail="Memory provider is not mounted"`.

The HTTP module depends only on `IUserMemory` / `IUserMemoryProvider`; it does
not import Hindsight.

## Configuration

### Hindsight client (`.env.json` session `hindsight`)

```json
{
  "hindsight": {
    "BASE_URL": "http://localhost:8888",
    "API_KEY": null,
    "TIMEOUT": "300"
  }
}
```

| Key | Default | Meaning |
|-----|---------|---------|
| `BASE_URL` | `http://localhost:8888` | Hindsight HTTP API |
| `API_KEY` | `null` | Optional API key |
| `TIMEOUT` | `300` | Client timeout (seconds) |

### Memorize job (`.env.json` session `agent_chats`)

```json
{
  "agent_chats": {
    "MEMORIZE_INTERVAL_MINUTES": "5",
    "MEMORIZE_BATCH_SIZE": "50",
    "MEMORIZE_MAX_BATCHES_PER_RUN": "20",
    "MEMORIZE_MIN_AGE_HOURS": "24"
  }
}
```

| Key | Default | Meaning |
|-----|---------|---------|
| `MEMORIZE_INTERVAL_MINUTES` | `5` | Scheduler interval |
| `MEMORIZE_BATCH_SIZE` | `50` | Max chats per drain batch |
| `MEMORIZE_MAX_BATCHES_PER_RUN` | `20` | Max batches while holding the job tick |
| `MEMORIZE_MIN_AGE_HOURS` | `24` | Only messages older than this are memorized |

Missing / invalid values fall back to the defaults above.

## Dependency injection

Registered in `src/fivccliche/settings/services.yml`:

```yaml
- entries:
    - interface: fivccliche.services.interfaces.agent_memories.IUserMemoryProvider
  class: fivccliche.services.implements.agent_memories_hindsight.UserMemoryProviderImpl

- entries:
    - interface: fivccliche.services.interfaces.modules.IModule
      name: agent_memories
  class: fivccliche.modules.agent_memories.ModuleImpl
```

`agent_memories` is also listed under `IModuleSite.modules`.

Remove the `IUserMemoryProvider` entry to leave memory unmounted.
`get_memory_provider_async` then returns `None`, the memorize job no-ops, and
the memories HTTP API returns 503.

A Redis-backed `IMutexSite` must also be available (`get_mutex_site_async`);
otherwise the job skips entirely.

## Chat memorize job

Implemented by `agent_chats.jobs.ChatMemorizeJob` and registered from
`ModuleImpl.mount` as APScheduler job `agent-chats-memorize`
(`max_instances=1`, `coalesce=True`).

Per tick:

1. Skip if memory provider or mutex site is missing.
2. Load chats that have completed, unmemorized messages with
   `created_at <= now - MEMORIZE_MIN_AGE_HOURS` (and non-null `user_uuid`).
3. Process up to `BATCH_SIZE` chats **sequentially** in this process, then
   drain further batches until empty or `MAX_BATCHES_PER_RUN`. Cross-node
   parallelism comes from multiple service replicas + per-chat Redis mutex.
4. For each chat, acquire mutex `agent-chats:memorize:{chat_uuid}`
   (non-blocking). On failure, skip that chat only.
5. Build a JSON conversation array and `retain_async` once with
   `space_id=chat.user_uuid`. On success (or empty content), mark those
   messages `is_memorized=True`.

Retain payload example:

```json
[
  {"role": "user", "content": "帮我看看这段代码为什么报错"},
  {"role": "assistant", "content": "你的第 12 行变量未定义，应该是..."}
]
```

Rules when building turns:

- Skip the **user** turn when `query.text` (stripped) starts with `/`
  (slash commands).
- Still include the **assistant** turn when `reply.text` is non-empty.
- If the resulting array is empty, do not call retain; still mark messages
  memorized so they are not scanned again.

## Manual injection example

```python
from fastapi import Depends

from fivccliche.services.interfaces.agent_memories import IUserMemoryProvider
from fivccliche.utils.deps import get_memory_provider_async

async def example(
    memory_provider: IUserMemoryProvider | None = Depends(get_memory_provider_async),
):
    if memory_provider is None:
        return []
    memory = memory_provider.get_memory(space_id=user.uuid)
    await memory.retain_async('[{"role":"user","content":"hi"}]')
    recalled = await memory.recall_async("hi")
    listed = await memory.list_async(skip=0, limit=20)
    return recalled.items, listed.items
```

## Testing

- `tests/test_agent_memories_hindsight.py` — Hindsight provider (mocked SDK), including `list_async`
- `tests/test_agent_memories_api.py` — HTTP auth, 503 when unmounted, list/recall success
- `tests/test_agent_chats_memorize.py` — conversation JSON, age filter, mutex
  skip, job marking, mount registration
