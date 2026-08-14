# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Layering, ownership, and HTTP CRUD: [docs/architecture.md](docs/architecture.md). Scheduled jobs: [docs/scheduler.md](docs/scheduler.md). Memories: [docs/agent-memories.md](docs/agent-memories.md). Agent rules that override this file: [AGENTS.md](AGENTS.md). Getting started: [docs/getting-started.md](docs/getting-started.md).

## Project Overview

FivcCliche is a multi-user FastAPI backend framework for AI agents. It uses SQLModel (SQLAlchemy + Pydantic) for async database operations, `fivcglue` for dependency injection/component composition, and `fivcplayground` for AI agent capabilities. Python ≥3.10 required.

## Common Commands

```bash
# Package manager is uv
make install-dev          # Install with dev dependencies
make serve                # Start uvicorn dev server with auto-reload (0.0.0.0:8000)

# Code quality
make format               # Black formatter (line-length: 100)
make lint                 # Ruff linter
make typecheck            # MyPy type checker
make check                # All three above

# Testing
make test                 # pytest -v
make test-cov             # pytest with HTML coverage report
pytest tests/test_users_api.py -v              # Run a single test file
pytest tests/test_users_api.py::test_name -v   # Run a single test

# CLI
python -m fivccliche.cli run                   # Start server
python -m fivccliche.cli migrate               # Initialize database tables
python -m fivccliche.cli createsuperuser       # Create admin user (interactive)
python -m fivccliche.cli changepassword        # Change a user's password (interactive)
python -m fivccliche.cli jobs list             # List scheduled jobs
python -m fivccliche.cli jobs show <module> <job>
python -m fivccliche.cli jobs run <module> <job>
```

API tests share [`tests/conftest.py`](tests/conftest.py) (`make_api_client`: isolated pg0 Postgres DB, admin user, session override).

## Architecture

### Dependency Injection via fivcglue

Components are registered in `src/fivccliche/settings/services.yml` mapping interfaces to implementations. The `IComponentSite` registry resolves dependencies. Services are lazy-loaded via `LazyValue` in `src/fivccliche/utils/deps.py`.

Key interfaces (in `src/fivccliche/services/interfaces/`):
- `IDatabase` → `DatabaseImpl` — async engine/session management
- `IUserAuthenticator` → `UserAuthenticatorImpl` — JWT auth, password hashing (Argon2)
- `IUserConfigProvider` → `UserConfigProviderImpl` — LLM/embedding/tool/agent config repos
- `IUserChatProvider` → `UserChatProviderImpl` — conversation repos
- `IUserMemoryProvider` → optional `UserMemoryProviderImpl` (Hindsight; requires separate `hindsight-client`)
- `IModule` / `IModuleSite` → module registration and FastAPI app mounting

### Module Pattern

Each module in `src/fivccliche/modules/` uses this layout:
- `models.py` — SQLModel tables (UUID primary keys)
- `schemas.py` — Pydantic request/response bodies only
- `queries.py` — optional; HTTP list/filter query models, only when list has extra filters
- `utils.py` — optional; SQL shared by routers and services/jobs/CLI (stages writes; callers commit)
- `routers.py` — FastAPI handlers (HTTP-only SQL is written here)
- `services.py` — `ModuleImpl` (registers routers) plus provider/authenticator implementations; not a `UserService` business layer

Modules: `users`, `agent_configs`, `agent_chats`, `agent_memories`. All mounted under `/api`.

HTTP CRUD is hand-written FastAPI handlers in each module's `routers.py`. Shared helpers: [`src/fivccliche/utils/permissions.py`](src/fivccliche/utils/permissions.py), [`src/fivccliche/utils/queries.py`](src/fivccliche/utils/queries.py) (dotted JSON for chats, not filter models), [`src/fivccliche/utils/deps.py`](src/fivccliche/utils/deps.py).

### Ownership

List/get returns the caller's rows plus globals (`user_uuid is None`). Create: superuser gets `user_uuid=None`, everyone else their own uuid. Update/delete use `has_ownership` (superuser may change globals; nobody may change another user's rows). Details: [docs/architecture.md](docs/architecture.md).

HTTP list/get returns inactive tools/skills. Playground repositories filter `is_active` themselves so agents never pick up disabled configs.

### Scheduled Tasks

Modules expose jobs via `IModule.list_jobs` / `get_job` (`IModuleJob`: `name`, `config`, `run_async`). `ModuleSiteImpl.create_application` creates a single `AsyncIOScheduler`, attaches it to `app.state.scheduler`, mounts routers via `module.mount(app, ...)`, then registers each job from `job.config` onto the scheduler. Lifespan starts/stops the scheduler. CLI: `fivccliche jobs list|show|run`. See [docs/scheduler.md](docs/scheduler.md).

### Authentication Flow

JWT-based (HS256). Login returns token → Bearer token in Authorization header → `get_authenticated_user_async` dependency extracts user. SSO via CAS supported in `modules/users/sso.py`. Configurable via env vars: `SECRET_KEY`, `ALGORITHM`, `EXPIRATION_HOURS`.

### Database

Default: embedded PostgreSQL via [pg0](https://github.com/vectorize-io/pg0) (`Pg0(name="fivccliche")`) when `DB_URL` is unset; SQLAlchemy uses `postgresql+asyncpg://...`. Set `DB_URL` to point at an external Postgres (or explicit SQLite). Tests use a session-scoped pg0 instance (`fivccliche-test`) with an isolated database per fixture / API client.

## Code Style

- Black: line-length 100, target Python 3.10-3.12
- Ruff: rules `E, F, W, I, N, UP, B, A, C4, PIE, PT, RUF`; ignores `E501, I001, B008`
- MyPy: Python 3.10, `warn_return_any = true`
- Full async/await throughout; use `AsyncSession` for all DB operations
- Type annotations expected on function signatures

### No thin wrappers

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

### Query filter models

HTTP list/filter Pydantic models belong in that module's `queries.py`. Do not put them in `schemas.py` (`schemas.py` is request/response bodies only). Do not put module-specific filter models in `utils/queries.py` (that file is dotted JSON query parsing for chats).
