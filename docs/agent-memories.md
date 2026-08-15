# Agent Memories (Hindsight)

FivcCliche exposes an implementation-agnostic memory contract so business code
can store and recall user memories without depending on a specific backend.
A [Hindsight](https://github.com/vectorize-io/hindsight) backend is available
via optional `hindsight-client` (not a core package dependency).

## Status

- Interface + optional Hindsight provider (DI + `get_memory_provider_async`)
- Scheduled **chat-level** memorize job in `agent_chats` (see below)
- HTTP API in `agent_memories` (`GET /memories/`, `GET /memories/recall/`,
  superuser-only `POST /memories/retain/`)
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

`UserMemoryProviderImpl` (`agent_memories_hindsight.py`) is **optional**:

- Requires `hindsight-client` installed separately (`pip install hindsight-client`)
- Dynamically imports the SDK when the component is constructed; missing install
  raises `ImportError` with an install hint
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
| `POST` | `/memories/retain/` | Superuser-only retain → `{ success, count, ids }` (`raw` is not exposed) |

`space_id` is always the authenticated user's `uuid`. Retain uses
`get_admin_user_async`: unauthenticated requests return **401**; a non-superuser
JWT returns **403** with `detail="Not a super user"`. Backend `success=False`
is still HTTP **200** with that flag in the body.

If `IUserMemoryProvider` is not registered, these endpoints return
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

### Memorize job (`.env.json` session `CHAT_MEMORIZE`)

```json
{
  "CHAT_MEMORIZE": {
    "INTERVAL_MINUTES": "5",
    "BATCH_SIZE": "50",
    "MAX_BATCHES_PER_RUN": "20",
    "MIN_AGE_HOURS": "24"
  }
}
```

| Key | Default | Meaning |
|-----|---------|---------|
| `INTERVAL_MINUTES` | `5` | Scheduler interval |
| `BATCH_SIZE` | `50` | Max chats per drain batch |
| `MAX_BATCHES_PER_RUN` | `20` | Max batches while holding the job tick |
| `MIN_AGE_HOURS` | `24` | Only messages older than this are memorized |

Missing / invalid values fall back to the defaults above.

## Dependency injection

`IUserMemoryProvider` is **not** registered in the default
`src/fivccliche/settings/services.yml`. Without a mount,
`get_memory_provider_async` returns `None`, the memorize job no-ops, and
the memories HTTP API returns 503.

To enable Hindsight memory, install `hindsight-client` and register the
provider in your services config:

```yaml
- entries:
    - interface: fivccliche.services.interfaces.agent_memories.IUserMemoryProvider
  class: fivccliche.services.implements.agent_memories_hindsight.UserMemoryProviderImpl
```

The `agent_memories` HTTP module remains listed under `IModuleSite.modules`
in the default `services.yml`:

```yaml
- entries:
    - interface: fivccliche.services.interfaces.modules.IModule
      name: agent_memories
  class: fivccliche.modules.agent_memories.ModuleImpl
```

A Redis-backed `IMutexSite` must also be available (`get_mutex_site_async`);
otherwise the memorize job skips entirely.

## Chat memorize job

Implemented by `agent_chats.jobs.ChatMemorizeJob` (`IModuleJob`).
`ModuleImpl` constructs `ChatMemorizeJob(component_site)` in `__init__` and
exposes it via `list_jobs` / `get_job`. `ModuleSiteImpl.create_application`
registers APScheduler job `agent-chats-memorize` from `job.config`
(`max_instances=1`, `coalesce=True`). Use `fivccliche jobs run agent_chats
agent-chats-memorize` to run it immediately.

Per tick:

1. Skip if memory provider or mutex site is missing.
2. Load chats that have completed, unmemorized messages with
   `created_at <= now - MIN_AGE_HOURS` (and non-null `user_uuid`).
3. Process up to `BATCH_SIZE` chats **sequentially** in this process, then
   drain further batches until empty or `MAX_BATCHES_PER_RUN`. Cross-node
   parallelism comes from multiple service replicas + per-chat Redis mutex.
4. For each chat, acquire mutex `agent-chats:memorize:{chat_uuid}`
   (non-blocking). On failure, skip that chat only.
5. Build a JSON conversation array and `retain_async` once with
   `space_id=chat.user_uuid`. On success (or when there is no user turn to
   retain), mark that chat's completed, unmemorized messages with
   `created_at <= created_at_to` as `is_memorized=True`.

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
- If the resulting array has no `role=user` turn (empty, or assistant-only
  after slash filtering), do not call retain; still mark messages memorized
  so they are not scanned again.

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
- `tests/test_agent_memories_api.py` — HTTP auth, 503 when unmounted, list/recall/retain success, retain 403 for non-superuser
- `tests/test_agent_chats_memorize.py` — conversation JSON, age filter, mutex
  skip, job marking, mount registration
