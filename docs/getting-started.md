# Getting Started

Run the real FivcCliche app. Do not add a `main.py`, `models.py`, or `database.py` — the CLI already creates the FastAPI application.

## Prerequisites

- Python 3.10 or higher
- [`uv`](https://docs.astral.sh/uv/)

## Install

```bash
cd FivcCliche
uv pip install -e ".[dev]"
```

## Database and admin user

Unset `DB_URL` to use embedded PostgreSQL via [pg0](https://github.com/vectorize-io/pg0) (instance name `fivccliche`). Or set `DB_URL` to any SQLAlchemy URL (for example `postgresql+asyncpg://...`).

```bash
python -m fivccliche.cli migrate
python -m fivccliche.cli createsuperuser
```

`migrate` creates missing tables; it does not alter existing ones.

## Run the server

```bash
make serve
# or
python -m fivccliche.cli run
```

- API: http://localhost:8000
- OpenAPI: http://localhost:8000/docs
- Routes are mounted under `/api`

## Authenticate

```bash
curl -X POST http://localhost:8000/api/users/login/ \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your-password"}'
```

Send the token as `Authorization: Bearer <access_token>` on later requests.

Create and list user-scoped configs under `/api/configs/` (embeddings, models, agents, tools, skills, questions). Superusers create globals (`user_uuid` is null); regular users can read those but cannot update or delete them.

## Next

- [Architecture](architecture.md) — module layering, ownership, HTTP CRUD
- [Scheduled Tasks](scheduler.md) — per-module APScheduler jobs
- [Agent Memories](agent-memories.md) — optional Hindsight memory API
