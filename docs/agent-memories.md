# Agent Memories (Hindsight)

FivcCliche exposes an implementation-agnostic memory contract so business code
can store and recall user memories without depending on a specific backend.
A [Hindsight](https://github.com/vectorize-io/hindsight) backend is available
via optional `hindsight-client` (not a core package dependency).

## Status

- Interface + optional Hindsight provider (DI + `get_memory_provider_async`)
- Chat-level memorize job is implemented in `agent_chats.jobs.memorize` but **not
  registered** on the scheduler (`agent_chats` `list_jobs()` is empty). A
  job listed with `config is None` is also skipped at mount.
- Function tools in `agent_memories.tools` (`MemoryRetain` / `MemoryRecall` /
  `MemoryList`) for later `transport=function` wiring
- HTTP API in `agent_memories` (`GET /memories/`, `GET /memories/recall/`,
  superuser-only `POST /memories/retain/`)
- `UserChatMessage.is_memorized` is written by the job after a successful retain
  (or when there is nothing to retain, including when the memorize LLM extracts
  nothing); it is not exposed on the public API yet

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
    "MIN_AGE_MINUTES": "5"
  }
}
```

| Key | Default | Meaning |
|-----|---------|---------|
| `INTERVAL_MINUTES` | `5` | Scheduler interval |
| `BATCH_SIZE` | `50` | Max chats per drain batch |
| `MAX_BATCHES_PER_RUN` | `20` | Max batches while holding the job tick |
| `MIN_AGE_MINUTES` | `5` | Only messages older than this are memorized |

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

## Agent memory tools

Callable classes in `src/fivccliche/modules/agent_memories/tools.py`. Attach
them with a tool config (`transport=function`) whose `functions` point at
the dotted paths. They are not exposed through chat context.

- `fivccliche.modules.agent_memories.tools.MemoryRetain`
- `fivccliche.modules.agent_memories.tools.MemoryRecall`
- `fivccliche.modules.agent_memories.tools.MemoryList`

Each class takes `**context` (expects `user_uuid`) and returns normalized JSON
(`raw` is omitted). `space_id` is the context `user_uuid`. Missing `user_uuid`
or an unmounted provider raises `ValueError`.

| Class | `__call__` | Result JSON |
|-------|------------|-------------|
| `MemoryRetain` | `content` | `{ success, count, ids }` |
| `MemoryRecall` | `query` | `{ items }` |
| `MemoryList` | `skip=0`, `limit=20` | `{ total, items }` |

## Chat memorize job

Implemented by `agent_chats.jobs.ChatMemorizeJob` (`IModuleJob`; defined in
`agent_chats.jobs.memorize`). The class
and `CHAT_MEMORIZE` settings remain, but `agent_chats.ModuleImpl` currently
returns an empty `list_jobs()`, so `ModuleSiteImpl` does not register
`agent-chats-memorize` and `fivccliche jobs run agent_chats
agent-chats-memorize` cannot find it. Listing the job with `config is None`
would also skip the scheduler. Re-enable scheduling by constructing
`ChatMemorizeJob(component_site)` in `ModuleImpl.__init__` again
(its `config` is a schedule dict).

Per tick:

1. Skip if memory provider or mutex site is missing.
2. Load chats that have completed, unmemorized messages with
   `created_at <= now - MIN_AGE_MINUTES` (and non-null `user_uuid`).
3. Process up to `BATCH_SIZE` chats **sequentially** in this process, then
   drain further batches until empty or `MAX_BATCHES_PER_RUN`. Cross-node
   parallelism comes from multiple service replicas + per-chat Redis mutex.
4. For each chat, acquire mutex `agent-chats:memorize:{chat_uuid}`
   (non-blocking). On failure, skip that chat only.
5. Build a JSON conversation array of `{role, content}` turns. If it has no
   user turn (empty, or assistant-only after slash filtering), do not call
   retain; still mark messages memorized so they are not scanned again.
6. Otherwise look up the user's visible LLM config with `id=memorize` (owned
   or global). If it exists, use it to decide whether the transcript is worth
   storing and to extract short standalone memories from **user** turns only;
   `retain_async` then stores that extracted text (`space_id=chat.user_uuid`).
   If the config is missing, `retain_async` stores the raw conversation JSON
   (legacy behavior).
7. If extraction runs and the LLM says nothing is worth retaining, or returns
   an empty memory list, skip retain and still mark messages memorized.
8. If the `memorize` LLM exists but the call fails, or structured output
   cannot be parsed, leave messages unmemorized so the next tick can retry.
9. On successful retain, mark that chat's completed, unmemorized messages with
   `created_at <= created_at_to` as `is_memorized=True`.

A user-visible LLM config with `id=memorize` is optional (typically a global
row created by a superuser). Without it, chats are retained as raw transcripts.

Conversation JSON sent to the judge when `id=memorize` exists (extracted
text is stored, not this JSON). Without that LLM, this JSON is retained
as-is:

```json
[
  {"role": "user", "content": "我叫 Charlie，Python 打包用 uv"},
  {"role": "assistant", "content": "好的，已记下"}
]
```

Example retain payload after extraction (user-stated facts only, not the
assistant reply):

```text
The user is named Charlie
The user prefers uv for Python packaging
```

Rules when building turns for the judge:

- Skip the **user** turn when `query.text` (stripped) starts with `/`
  (slash commands).
- Still include the **assistant** turn when `reply.text` is non-empty.
- If the resulting array has no `role=user` turn, skip the LLM call.

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
- `tests/test_agent_memories_tools.py` — retain/recall/list tools, missing
  user/provider, JSON without `raw`
- `tests/test_agent_chats_memorize.py` — conversation JSON, LLM extract/skip/
  retry, age filter, mutex skip, job marking, job not registered on the
  scheduler
