# Architecture

Module layering, user-or-global ownership, and the HTTP CRUD factory for user-scoped configs. Agent rules that override this file: [AGENTS.md](../AGENTS.md). Map and commands: [CLAUDE.md](../CLAUDE.md).

## Module layering

Each module under `src/fivccliche/modules/` owns its files. Shared HTTP/SQL helpers live in `src/fivccliche/utils/`, not in a second repository layer.

| File | Role |
|------|------|
| `models.py` | SQLModel tables (UUID primary keys) |
| `schemas.py` | Request/response bodies only |
| `queries.py` | Optional. HTTP list/filter query models, only when list has extra filters |
| `methods.py` | Async SQL (`AsyncSession`) |
| `routers.py` | FastAPI handlers |
| `services.py` | `ModuleImpl` (registers routers) plus provider/authenticator implementations |

`methods.py` and [`src/fivccliche/utils/crud.py`](../src/fivccliche/utils/crud.py) must not import each other. Routers wire both.

Modules: `users`, `agent_configs`, `agent_chats`, `agent_memories`. All mounted under `/api`.

### Who uses the CRUD factory

Only `agent_configs` uses `RouteConfig` / `register_routes` for embeddings, models, agents, tools, skills, and questions.

`users`, `agent_chats`, and `agent_memories` keep hand-written routes.

Extra config routes stay outside the factory:

- `POST /configs/tools/index/` — index tools for the authenticated user
- `POST /configs/tools/{config_uuid}/probe/` — probe a tool config

Frozen agents use `before_update` / `before_delete` on the agent `RouteConfig` (`_reject_frozen_agent_update` / `_reject_frozen_agent_delete` in [`agent_configs/routers.py`](../src/fivccliche/modules/agent_configs/routers.py)). A frozen agent cannot be deleted. Updates may only set `is_frozen`; any other field in the PATCH body is 403.

HTTP list/get returns every matching row, including inactive tools/skills. Playground repositories filter `is_active` themselves so agents never pick up disabled configs. Question configs have no playground repository.

## Ownership

User-scoped configs are either owned (`user_uuid` = the user) or global (`user_uuid is None`).

### SQL (`methods.py`)

[`_get_user_scoped_async` / `_list_user_scoped_async` / `_count_user_scoped_async`](../src/fivccliche/modules/agent_configs/methods.py):

- Lookup by exactly one of `config_uuid` or `config_id`
- Visibility: `(user_uuid == me) | (user_uuid == None)`
- List order: `id` ascending, with `skip` / `limit`

Do not extract that SQL condition into a named helper.

### HTTP

- **404** if `get_fn` returns nothing
- **403** on update/delete via [`assert_user_owns_resource`](../src/fivccliche/utils/asserts.py): non-superusers cannot change globals; nobody can change another user's rows; superusers may change globals
- **Create:** `None if user.is_superuser else user.uuid` so superusers create globals

Chats use the same assert helper for chat ownership; they are not user-or-global configs.

## CRUD factory

[`src/fivccliche/utils/crud.py`](../src/fivccliche/utils/crud.py) registers create/list/get/update/delete on a router from a `RouteConfig`. A `None` function skips that verb. PATCH and DELETE also require `get_fn`.

```python
@dataclass
class RouteConfig:
    slug: str
    noun: str
    schema: type[BaseModel]
    create_fn: Callable[..., Awaitable[Any]] | None = None
    get_fn: Callable[..., Awaitable[Any]] | None = None
    list_fn: Callable[..., Awaitable[list]] | None = None
    count_fn: Callable[..., Awaitable[int]] | None = None
    update_fn: Callable[..., Awaitable[Any]] | None = None
    delete_fn: Callable[..., Awaitable[None]] | None = None
    list_query: type[BaseModel] | None = None
    before_update: BeforeUpdate | None = None
    before_delete: BeforeDelete | None = None
```

- Responses call `config.to_schema()` with no include/exclude arguments
- PATCH body is `create_partial_model(schema)`
- `api_key` is write-only on the schema (`Field(exclude=True)` on `UserEmbeddingSchema` / `UserLLMSchema`), not a factory flag
- `list_fn` and `count_fn` must both be set to hang `GET /`

### `list_query`

Filter models live in the module's `queries.py`. Example: [`UserQuestionListQuery`](../src/fivccliche/modules/agent_configs/queries.py) (`is_active: bool | None = None`).

Pass `list_query=queries.UserQuestionListQuery` on that `RouteConfig`. The factory merges `skip` / `limit` onto the same Pydantic model (`create_model(f"{list_query.__name__}WithPaging", ...)`) because FastAPI will not flatten a Query model that sits beside separate `skip` / `limit` Query parameters. It then `model_dump()`s, pops paging, and forwards the rest as `**filters` to `list_fn` and `count_fn`.

If `list_query` is `None`, list only takes `skip` / `limit`.

Do not put filter models in `schemas.py` or in [`utils/queries.py`](../src/fivccliche/utils/queries.py) (that file parses dotted JSON for chats, e.g. `context.profile_uuid=`).
