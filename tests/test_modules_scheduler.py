"""Tests for APScheduler integration in ModuleSiteImpl / IModule.mount."""

import os

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fivccliche.services.implements.modules import ModuleSiteImpl
from fivccliche.services.interfaces.modules import IModule
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


def test_module_can_register_job_via_mount():
    """A module should be able to register a scheduled job during mount."""
    registered: dict = {}

    class _DummyModule(IModule):
        @property
        def name(self):
            return "dummy"

        @property
        def description(self):
            return "Dummy module for scheduler test."

        def mount(
            self,
            app: FastAPI,
            scheduler: AsyncIOScheduler | None = None,
            **kwargs,
        ) -> None:
            assert scheduler is not None
            scheduler.add_job(
                lambda: None,
                "interval",
                seconds=3600,
                id="dummy-job",
            )
            registered["mounted"] = True

    component_site = _load_component_site()
    module_site = ModuleSiteImpl(component_site, modules=[])
    module_site.register_module(_DummyModule())
    app = module_site.create_application()
    scheduler: AsyncIOScheduler = app.state.scheduler

    assert registered.get("mounted") is True

    with TestClient(app):
        job = scheduler.get_job("dummy-job")
        assert job is not None
        assert job.id == "dummy-job"
