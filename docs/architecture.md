# Architecture

Module layering, user-or-global ownership, and HTTP CRUD for user-scoped configs. Agent rules that override this file: [AGENTS.md](../AGENTS.md). Map and commands: [CLAUDE.md](../CLAUDE.md).

## Module layering

Each module under `src/fivccliche/modules/` owns its files. Shared HTTP helpers live in `src/fivccliche/utils/` (deps, FilterSet). Per-module shared SQL lives in that module's `utils.py`.

| File | Role |
|------|------|
| `models.py` | SQLModel tables (UUID primary keys) |
| `schemas.py` | Request/response bodies only |
| `filters.py` | Optional. HTTP list FilterSet (or extra Query params), only when list has extra filters |
| `utils.py` | Optional. SQL used by routers and services/jobs/CLI |
| `routers.py` | FastAPI handlers; HTTP-only SQL is written here |
| `services.py` | `ModuleImpl` (registers routers) plus provider/authenticator implementations |

Do not import SQL from `routers.py` (services would cycle). `utils.py` does not import routers.

Modules: `users`, `agent_configs`, `agent_chats`, `agent_memories`. All mounted under `/api`. All HTTP routes are hand-written FastAPI handlers.

`agent_configs` owns embeddings, models, agents, tools, skills, and questions. Extra config routes:

- `POST /configs/tools/index/` — index tools for the authenticated user
- `POST /configs/tools/{config_uuid}/probe/` — probe a tool config

Frozen agents: `_reject_frozen_agent_update` / `_reject_frozen_agent_delete` in [`agent_configs/routers.py`](../src/fivccliche/modules/agent_configs/routers.py), called from the agent PATCH and DELETE handlers. A frozen agent cannot be deleted. Updates may only set `is_frozen`; any other field in the PATCH body is 403.

HTTP list/get returns every matching row, including inactive tools/skills. Playground repositories filter `is_active` themselves so agents never pick up disabled configs. Question configs have no playground repository. Question list accepts an `is_active` query parameter.

Config responses call `config.to_schema()` with no include/exclude arguments. PATCH bodies use `create_partial_model(schema)`. `api_key` is write-only on the schema (`Field(exclude=True)` on `UserEmbeddingSchema` / `UserLLMSchema`).

Do not add get/list/count/delete wrappers around `get/list/count_user_scoped_async` or around `session.delete` + `commit`.

Module `utils.py` stages writes (`session.add` / field assignment) and does not `commit`. The caller that opened the session (router, service, job, CLI) commits; refresh after commit when returning DB defaults.

## Ownership

User-scoped configs are either owned (`user_uuid` = the user) or global (`user_uuid is None`).

### SQL ([`agent_configs/utils.py`](../src/fivccliche/modules/agent_configs/utils.py))

[`get_user_scoped_async` / `list_user_scoped_async` / `count_user_scoped_async`](../src/fivccliche/modules/agent_configs/utils.py):

- Lookup by exactly one of `config_uuid` or `config_id`
- Visibility via a required `filters: FilterSet` that includes [`FilterReadableField`](../src/fivccliche/utils/filters.py) (owned or global; same predicate for regular users and superusers today)
- List order: `id` ascending, with `skip` / `limit`

Callers construct [`UserScopedReadableFilterSet`](../src/fivccliche/modules/agent_configs/filters.py) (or a module FilterSet that embeds Readable). Create/update field mapping for playground types stays in `utils.py` and does not `commit`; the HTTP handler or repository commits. Questions create/update/delete SQL is in the HTTP handlers.

### HTTP

- **404** if get/update/delete returns nothing (update/delete look up via `FilterEditableField`)
- **Create:** `None if user.is_superuser else user.uuid` so superusers create globals

Chats use the same Readable/Editable FilterSets for chat ownership; they are not user-or-global configs in the HTTP create sense, but list/get still include globals.

HTTP list filters that bind query params to SQL belong in that module's `filters.py` as a `FilterSet` subclass. Do not put them in `schemas.py` or in [`utils/filters.py`](../src/fivccliche/utils/filters.py) (reusable `FilterField` / `FilterSimpleField` / `FilterJsonField` / `FilterReadableField` / `FilterEditableField` / `FilterSet`). Do not use FilterSet as a FastAPI `Depends`; declare scalar query params with `Query()` on the handler, instantiate the FilterSet, call `parse(...)` (plus any dotted JSON keys from the request), then pass it into SQL helpers that call `filter(statement)`.

Chat list uses [`ChatFilterSet`](../src/fivccliche/modules/agent_chats/filters.py): `ChatFilterSet(user_uuid, is_superuser=...)` → `parse(agent_id=..., context.*=...)` → `filter(statement)` (includes Readable).

Question list uses [`QuestionFilterSet`](../src/fivccliche/modules/agent_configs/filters.py): `QuestionFilterSet(user_uuid, is_superuser=...)` → `parse(is_active=...)` → passed into `list_user_scoped_async` / `count_user_scoped_async` as `filters`.

- `?agent_id=` exact match on the chat agent
- `?context.<key>=<value>` exact match on a top-level JSON key of `context` (one level only). `UserChat.context` stays a persisted dict. `UserChatProviderImpl.get_chat_context` returns a copy of that JSON plus `user_uuid`, merged `**kwargs` (for example `chat_uuid`), default `timezone` (`Asia/Shanghai`), and a lazy `time` whose `__str__` computes a timezone-aware ISO string and is not persisted. `ChatQueryJob` calls `get_chat_context` and passes the result to the agent run.
- Repeated paths use the last value
- Nested keys such as `context.profile.uuid` return 422; bare `context=` is invalid if passed into `parse`
- Question list: `?is_active=` exact match when provided
- Pagination `skip` / `limit` stay on the handler, not on the FilterSet
- Field classes implement `filter(statement)` and optionally override `parse(value)` to store query-bound state; `FilterSet.filter` always calls every field, and each field decides whether to add a WHERE or return the statement unchanged
- `FilterReadableField` / `FilterEditableField` take `user_uuid` and `is_superuser`: Readable is always owned-or-global; Editable is owned-only for regular users and owned-or-global for superusers. GET/LIST use Readable; PATCH/DELETE (and chat message create) look up via Editable and return 404 when the row is not editable.

### POST `/{chat_uuid}/messages/`

The handler looks up the chat via Editable (404 if missing), acquires mutex `chats:message:{uuid}` (409 if already held), then starts the agent **before** returning SSE:

1. `asyncio.create_task(ChatQueryJob.run_async(...))` — `get_chat_context`, agent/tools/skills, `event_callback=chat_stream.on_event`. Mutex is released in the job `finally`.
2. [`ChatStream`](../src/fivccliche/utils/stream.py)`()` yields SSE chunks from that task.
3. `BackgroundTasks` first `asyncio.gather(query_task, return_exceptions=True)` (keeps the run alive after client disconnect), then `ChatDescribeJob` when the chat still has an empty description and the query is not a slash command.

`ChatQueryJob` and `ChatDescribeJob` have `config is None` and are not on `list_jobs()`.

`PATCH /{chat_uuid}/` updates only a non-empty `description` (Editable lookup, 404 if not editable).
