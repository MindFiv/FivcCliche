# AGENTS.md

Project guidance for coding agents. Architecture and commands live in [CLAUDE.md](CLAUDE.md). Module layering, ownership, and HTTP CRUD live in [docs/architecture.md](docs/architecture.md). Follow those files unless a later instruction here is more specific.

## No thin wrappers

Do not extract one-liners or 2–4 line pass-throughs into named helpers. They lengthen the call chain without reducing complexity. Repeat a few lines at the call site instead.

Do not wrap, for example:

- a single SQL condition (`(user_uuid == me) | (user_uuid == None)`)
- `session.add` + `commit` + `refresh`
- `session.delete` + `commit`
- `None if user.is_superuser else user.uuid`
- `config.to_schema(...)` / `Schema.model_validate(...)`
- `datetime.now(timezone.utc)`
- an assert that only forwards error-message strings to another assert

This applies to every refactor, cleanup, and shared-layer extraction. Extract only logic with real rules (uuid/id dual lookup plus user-or-global visibility). Do not add get/list/count/delete wrappers around those helpers. Module `utils.py` must not `commit`; callers own the transaction.

## Query filter models

HTTP list/filter Pydantic models belong in that module's `queries.py`. Do not put them in `schemas.py` (`schemas.py` is request/response bodies only). Do not put module-specific filter models in `utils/queries.py` (that file is dotted JSON query parsing for chats).

## Config HTTP CRUD

User-scoped config HTTP CRUD is hand-written FastAPI handlers in [`agent_configs/routers.py`](src/fivccliche/modules/agent_configs/routers.py), same as `users`, `agent_chats`, and `agent_memories`. Extra routes (for example tools `index` / `probe`) live on the same routers. Shared SQL used by routers and playground repositories lives in that module's `utils.py`. Do not import SQL from `routers.py`. Do not `commit` inside module `utils.py`.

`api_key` is write-only via `Field(exclude=True)` on the schema (see [`src/fivccliche/modules/agent_configs/schemas.py`](src/fivccliche/modules/agent_configs/schemas.py)).
