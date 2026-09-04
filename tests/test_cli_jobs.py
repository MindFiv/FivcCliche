"""Tests for CLI jobs list/show/run and module job discovery."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from fivccliche.cli import cli
from fivccliche.services.implements.modules import ModuleSiteImpl
from fivccliche.services.interfaces.modules import IModule, IModuleJob


class _FakeJob(IModuleJob):
    def __init__(self) -> None:
        self.ran = False

    @property
    def name(self) -> str:
        return "fake-job"

    @property
    def config(self) -> dict | None:
        return {"trigger": "interval", "seconds": 3600}

    async def run_async(self) -> None:
        self.ran = True


class _FakeModule(IModule):
    def __init__(self) -> None:
        self._jobs: list[IModuleJob] = [_FakeJob()]

    @property
    def name(self):
        return "fake_module"

    @property
    def description(self):
        return "Fake module for CLI job tests."

    def list_jobs(self) -> list[IModuleJob]:
        return list(self._jobs)

    def get_job(self, job_name: str) -> IModuleJob | None:
        for job in self._jobs:
            if job.name == job_name:
                return job
        return None

    def mount(self, app, **kwargs) -> None:
        return None


def test_module_list_and_get_job():
    module = _FakeModule()
    jobs = module.list_jobs()
    assert len(jobs) == 1
    assert module.get_job("fake-job") is jobs[0]
    assert module.get_job("nope") is None


def test_module_site_registers_fake_job():
    from fivcglue.implements.utils import load_component_site
    import os

    components_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "fivccliche",
        "settings",
        "services.yml",
    )
    component_site = load_component_site(filename=components_path, fmt="yaml")
    site = ModuleSiteImpl(component_site, modules=[])
    module = _FakeModule()
    site.register_module(module)
    app = site.create_application()
    job = app.state.scheduler.get_job("fake-job")
    assert job is not None
    assert job.id == "fake-job"


@pytest.fixture
def fake_module_site():
    component_site = MagicMock()
    site = ModuleSiteImpl(component_site, modules=[])
    module = _FakeModule()
    site.register_module(module)
    return site, module


def test_cli_jobs_list(fake_module_site):
    site, _module = fake_module_site
    runner = CliRunner()
    with patch("fivccliche.cli.modules", site):
        result = runner.invoke(cli, ["jobs", "list"])
    assert result.exit_code == 0
    assert "fake_module" in result.stdout
    assert "fake-job" in result.stdout


def test_cli_jobs_show(fake_module_site):
    site, _module = fake_module_site
    runner = CliRunner()
    with patch("fivccliche.cli.modules", site):
        result = runner.invoke(cli, ["jobs", "show", "fake_module", "fake-job"])
    assert result.exit_code == 0
    assert "fake-job" in result.stdout
    assert "interval" in result.stdout


def test_cli_jobs_show_missing(fake_module_site):
    site, _module = fake_module_site
    runner = CliRunner()
    with patch("fivccliche.cli.modules", site):
        result = runner.invoke(cli, ["jobs", "show", "fake_module", "missing"])
    assert result.exit_code == 1


def test_cli_jobs_run(fake_module_site):
    site, module = fake_module_site
    job = module.get_job("fake-job")
    assert job is not None
    runner = CliRunner()
    with patch("fivccliche.cli.modules", site):
        result = runner.invoke(cli, ["jobs", "run", "fake_module", "fake-job"])
    assert result.exit_code == 0
    assert job.ran is True
    assert "completed successfully" in result.stdout
