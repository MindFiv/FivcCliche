# Architecture

Module layering, user-or-global ownership, and HTTP CRUD for user-scoped configs. Agent rules that override this file: [AGENTS.md](../AGENTS.md). Map and commands: [CLAUDE.md](../CLAUDE.md).

## Module layering

Each module under `src/fivccliche/modules/` owns its files. Shared HTTP helpers live in `src/fivccliche/utils/` (deps, permissions, chat dotted-JSON). Per-module shared SQL lives in that module's `utils.py`.

| File | Role |
|------|------|
| `models.py` | SQLModel tables (UUID primary keys) |
| `schemas.py` | Request/response bodies only |
| `queries.py` | Optional. HTTP list/filter query models, only when list has extra filters |
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
- Visibility: `(user_uuid == me) | (user_uuid == None)`
- List order: `id` ascending, with `skip` / `limit`

Do not extract that SQL condition into a named helper. Routers and playground repositories call these helpers with the model class. Create/update field mapping for playground types stays in `utils.py` and does not `commit`; the HTTP handler or repository commits. Questions create/update/delete SQL is in the HTTP handlers.

### HTTP

- **404** if get returns nothing
- **403** on update/delete via [`has_ownership`](../src/fivccliche/utils/permissions.py): non-superusers cannot change globals; nobody can change another user's rows; superusers may change globals
- **Create:** `None if user.is_superuser else user.uuid` so superusers create globals

Chats use the same assert helper for chat ownership; they are not user-or-global configs.

HTTP list/filter Pydantic models belong in that module's `queries.py`. Do not put them in `schemas.py` or in [`utils/queries.py`](../src/fivccliche/utils/queries.py) (that file parses dotted JSON for chats, e.g. `context.profile_uuid=`). List endpoints that only need extra scalar filters (for example question `is_active`) use FastAPI `Query()` parameters on the handler.
