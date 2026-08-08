# Scheduled Tasks (APScheduler)

FivcCliche ships with built-in support for per-module scheduled jobs via
[APScheduler](https://apscheduler.readthedocs.io/) (`AsyncIOScheduler`).

## How it works

`ModuleSiteImpl.create_application` is the single place that owns the scheduler
lifecycle:

1. It creates one `AsyncIOScheduler` instance.
2. Attaches it to `app.state.scheduler` (useful for runtime inspection and tests).
3. Passes it to every registered module's `mount(app, scheduler=..., prefix=...)`.
4. Wires it into the FastAPI lifespan:
   - `scheduler.start()` on application startup.
   - `scheduler.shutdown(wait=False)` on application shutdown.

Modules never create their own scheduler. They receive the shared instance in
`mount` and register jobs against it.

## Registering a job in a module

Inside a module's `ModuleImpl.mount`, call `scheduler.add_job(...)` when a
scheduler is provided:

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

class ModuleImpl(IModule):
    ...
    def mount(
        self,
        app: FastAPI,
        scheduler: AsyncIOScheduler | None = None,
        **kwargs,
    ) -> None:
        print("my module mounted.")
        if scheduler is not None:
            scheduler.add_job(
                self._cleanup_job,
                trigger="interval",
                seconds=60,
                id="my-module-cleanup",
                replace_existing=True,
            )
        app.include_router(router, **kwargs)

    def _cleanup_job(self) -> None:
        ...
```

The `scheduler` argument is optional (`None` by default) so modules stay
backwards-compatible with callers that don't supply a scheduler.

## Triggers

APScheduler supports several triggers; the most common ones are:

- `"interval"` — fixed cadence, e.g. `seconds=60`, `minutes=5`, `hours=1`.
- `"date"` — one-shot at a specific `run_date`.
- `"cron"` — cron-style schedules, e.g. `hour=2, minute=30`.

See the [APScheduler triggers docs](https://apscheduler.readthedocs.io/en/stable/modules/triggers.html)
for full details.

## Jobstore

The default scheduler uses an in-memory jobstore. Jobs do not survive process
restarts and are not shared across workers. If you need persistence or
multi-worker coordination, configure a persistent jobstore (e.g. SQLAlchemy)
on the scheduler instance before mounting modules. That configuration is out of
scope for the framework defaults.

## Testing

`tests/test_modules_scheduler.py` demonstrates the three guarantees:

1. `create_application` attaches an `AsyncIOScheduler` to `app.state.scheduler`
   and it is not running before the lifespan starts.
2. Entering `with TestClient(app)` starts the scheduler; exiting stops it.
3. A module's `mount` receives the scheduler and can register a job that is
   retrievable via `scheduler.get_job(id)`.

When writing tests that register jobs you don't want to actually fire, use a
far-future `"date"` trigger or a long `"interval"` so the job never executes
within the test window.

## Real module example

`agent_chats` constructs `ChatMemorizeJob(component_site, scheduler)` during
`mount`, which registers `agent-chats-memorize` in `__init__` (interval from
`agent_chats.MEMORIZE_INTERVAL_MINUTES`, default 5). See
[agent-memories.md](agent-memories.md) for chat-level retain semantics and
per-chat Redis mutex details.
