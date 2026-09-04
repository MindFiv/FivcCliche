# Scheduled Tasks (IModuleJob + APScheduler)

FivcCliche ships with built-in support for per-module scheduled jobs via
[APScheduler](https://apscheduler.readthedocs.io/) (`AsyncIOScheduler`),
abstracted behind `IModuleJob`.

## How it works

`ModuleSiteImpl.create_application` owns the scheduler lifecycle and job
registration:

1. It creates one `AsyncIOScheduler` instance.
2. Attaches it to `app.state.scheduler` (useful for runtime inspection and tests).
3. Calls each module's `mount(app, prefix=...)` for HTTP routers only
   (no scheduler argument).
4. For each module, iterates `module.list_jobs()` and registers each
   `IModuleJob` whose `config` is not `None` via `scheduler.add_job` using
   `job.name` as the job id and `job.config` as the schedule kwargs. Jobs
   with `config is None` stay visible to `list_jobs` / CLI but are not
   added to the scheduler.
5. Wires the scheduler into the FastAPI lifespan:
   - `scheduler.start()` on application startup.
   - `scheduler.shutdown(wait=False)` on application shutdown.

Modules never create their own scheduler. They expose jobs through
`list_jobs` / `get_job`; the site wires them onto the shared scheduler.

## IModuleJob

```python
class IModuleJob(IComponent):
    @property
    def name(self) -> str: ...

    @property
    def config(self) -> dict | None: ...

    async def run_async(self): ...
```

### `job.config` convention

`config` is the keyword arguments for APScheduler `add_job`, excluding `func`
and `id`. Return `None` to expose the job (CLI `list` / `show` / `run`)
without registering it on the scheduler:

```python
{
    "trigger": "interval",       # required
    "minutes": 5,                # trigger-specific args
    "max_instances": 1,          # optional job options
    "coalesce": True,
    "replace_existing": True,    # default True if omitted
}
```

Job names must be unique across all registered modules (the scheduler id is
`job.name`). Prefer a module-prefixed name such as `agent-chats-memorize`.

## Defining a job in a module

```python
from fastapi import FastAPI

from fivccliche.services.interfaces.modules import IModule, IModuleJob


class CleanupJob(IModuleJob):
    @property
    def name(self) -> str:
        return "my-module-cleanup"

    @property
    def config(self) -> dict | None:
        return {
            "trigger": "interval",
            "seconds": 60,
            "replace_existing": True,
        }

    async def run_async(self) -> None:
        ...


class ModuleImpl(IModule):
    def __init__(self, component_site, **kwargs):
        self._jobs: list[IModuleJob] = [CleanupJob()]

    @property
    def name(self):
        return "my_module"

    @property
    def description(self):
        return "Example module."

    def list_jobs(self) -> list[IModuleJob]:
        return list(self._jobs)

    def get_job(self, job_name: str) -> IModuleJob | None:
        for job in self._jobs:
            if job.name == job_name:
                return job
        return None

    def mount(self, app: FastAPI, **kwargs) -> None:
        app.include_router(router, **kwargs)
```

Modules without jobs return an empty list from `list_jobs` and `None` from
`get_job`.

## CLI

Ops can inspect and run jobs without waiting for the scheduler tick:

```bash
fivccliche jobs list
fivccliche jobs show MODULE JOB
fivccliche jobs run MODULE JOB
```

`jobs run` calls `job.run_async()` immediately via asyncio; it does not
require the FastAPI lifespan or a running scheduler.

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

`tests/test_modules_scheduler.py` demonstrates the guarantees:

1. `create_application` attaches an `AsyncIOScheduler` to `app.state.scheduler`
   and it is not running before the lifespan starts.
2. Entering `with TestClient(app)` starts the scheduler; exiting stops it.
3. Jobs from `module.list_jobs()` with a non-`None` `config` are registered
   on the scheduler and retrievable via `scheduler.get_job(id)`. Jobs whose
   `config` is `None` are not registered.
4. `mount` does not receive a `scheduler` argument.

When writing tests that register jobs you don't want to actually fire, use a
far-future `"date"` trigger or a long `"interval"` so the job never executes
within the test window.

## Real module example

`agent_chats` currently returns an empty `list_jobs()`, so the query, describe,
and memorize jobs are not registered on the scheduler. `ChatQueryJob` lives in
`agent_chats.jobs.query` (`config is None`; the message handler starts it with
`asyncio.create_task`). `ChatDescribeJob` lives in `agent_chats.jobs.describe`
(`config is None`; invoked from the same handler via `BackgroundTasks` after
the query task). `ChatMemorizeJob` lives in `agent_chats.jobs.memorize`;
re-attach it by constructing it in `ModuleImpl.__init__`. See
[agent-memories.md](agent-memories.md) for chat-level retain semantics and
per-chat Redis mutex details.
