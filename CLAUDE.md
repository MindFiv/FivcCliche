# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
```

## Architecture

### Dependency Injection via fivcglue

Components are registered in `src/fivccliche/settings/services.yml` mapping interfaces to implementations. The `IComponentSite` registry resolves dependencies. Services are lazy-loaded via `LazyValue` in `src/fivccliche/utils/deps.py`.

Key interfaces (in `src/fivccliche/services/interfaces/`):
- `IDatabase` → `DatabaseImpl` — async engine/session management
- `IUserAuthenticator` → `UserAuthenticatorImpl` — JWT auth, password hashing (Argon2)
- `IUserConfigProvider` → `UserConfigProviderImpl` — LLM/embedding/tool/agent config repos
- `IUserChatProvider` → `UserChatProviderImpl` — conversation repos
- `IModule` / `IModuleSite` → module registration and FastAPI app mounting

### Module Pattern

Each module in `src/fivccliche/modules/` follows the same structure:
- `models.py` — SQLModel database models (UUID primary keys)
- `schemas.py` — Pydantic request/response schemas
- `methods.py` — core async CRUD operations (takes `AsyncSession`)
- `services.py` — `ModuleImpl` (registers routers) + provider/authenticator implementations
- `routers.py` — FastAPI route handlers

Modules: `users`, `agent_configs`, `agent_chats`. All mounted under `/api` prefix.

### Authentication Flow

JWT-based (HS256). Login returns token → Bearer token in Authorization header → `get_authenticated_user_async` dependency extracts user. SSO via CAS supported in `modules/users/sso.py`. Configurable via env vars: `SECRET_KEY`, `ALGORITHM`, `EXPIRATION_HOURS`.

### Database

Default: SQLite (`sqlite:///./fivccliche.db`) with aiosqlite for async. Uses `NullPool` for SQLite. Supports PostgreSQL/MySQL via `DATABASE_URL` env var. Tests use isolated temporary SQLite databases.

## Code Style

- Black: line-length 100, target Python 3.10-3.12
- Ruff: rules `E, F, W, I, N, UP, B, A, C4, PIE, PT, RUF`; ignores `E501, I001, B008`
- MyPy: Python 3.10, `warn_return_any = true`
- Full async/await throughout; use `AsyncSession` for all DB operations
- Type annotations expected on function signatures
