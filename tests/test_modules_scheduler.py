"""Tests for APScheduler integration via IModuleJob / ModuleSiteImpl."""

import os

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fivccliche.services.implements.modules import ModuleSiteImpl
from fivccliche.services.interfaces.modules import IModule, IModuleJob
from fivcglue.implements.utils import load_component_site


def _load_component_site():
    components_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "fivccliche",
        "settings",
        "services.yml",
    )
    return load_component_site(filename=components_path, fmt="yaml")


class _DummyJob(IModuleJob):
    @property
    def name(self) -> str:
        return "dummy-job"

    @property
    def config(self) -> dict:
        return {
            "trigger": "interval",
            "seconds": 3600,
            "replace_existing": True,
        }

    async def run_async(self) -> None:
        return None


class _DummyModule(IModule):
    def __init__(self) -> None:
        self._jobs: list[IModuleJob] = [_DummyJob()]
        self.mount_kwargs: dict | None = None

    @property
    def name(self):
        return "dummy"

    @property
    def description(self):
        return "Dummy module for scheduler test."

    def list_jobs(self) -> list[IModuleJob]:
        return list(self._jobs)

    def get_job(self, job_name: str) -> IModuleJob | None:
        for job in self._jobs:
            if job.name == job_name:
                return job
        return None

    def mount(self, app: FastAPI, **kwargs) -> None:
        self.mount_kwargs = dict(kwargs)
        assert "scheduler" not in kwargs


def test_scheduler_attached_to_app_state():
    """create_application should attach an AsyncIOScheduler to app.state."""
    component_site = _load_component_site()
    module_site = ModuleSiteImpl(component_site, modules=[])
    app = module_site.create_application()

    assert isinstance(app.state.scheduler, AsyncIOScheduler)
    # Before lifespan runs, the scheduler should not be running.
    assert app.state.scheduler.running is False


def test_scheduler_lifecycle_with_testclient():
    """Scheduler should start on lifespan startup and stop on shutdown."""
    component_site = _load_component_site()
    module_site = ModuleSiteImpl(component_site, modules=[])
    app = module_site.create_application()
    scheduler: AsyncIOScheduler = app.state.scheduler

    with TestClient(app) as client:
        assert scheduler.running is True
        # client is unused but kept for clarity
        assert client is not None

    assert scheduler.running is False


def test_module_jobs_registered_via_list_jobs():
    """create_application should register jobs from module.list_jobs()."""
    component_site = _load_component_site()
    module_site = ModuleSiteImpl(component_site, modules=[])
    dummy = _DummyModule()
    module_site.register_module(dummy)
    app = module_site.create_application()
    scheduler: AsyncIOScheduler = app.state.scheduler

    assert dummy.mount_kwargs is not None
    assert "scheduler" not in dummy.mount_kwargs

    with TestClient(app):
        job = scheduler.get_job("dummy-job")
        assert job is not None
        assert job.id == "dummy-job"
