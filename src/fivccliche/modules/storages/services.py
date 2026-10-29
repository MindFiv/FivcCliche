"""Storages module registration."""

import logging

from fastapi import FastAPI
from fivcglue import IComponentSite

from fivccliche.services.interfaces.modules import IModule, IModuleJob

from . import jobs, routers

logger = logging.getLogger(__name__)


class ModuleImpl(IModule):
    """User-owned storage file module."""

    def __init__(self, component_site: IComponentSite, **kwargs):
        self._component_site = component_site
        self._jobs: list[IModuleJob] = [jobs.StorageCleanupJob(component_site)]
        logger.info("storages module initialized")

    @property
    def name(self) -> str:
        return "storages"

    @property
    def description(self) -> str:
        return "User file storage backed by Aliyun OSS."

    def list_jobs(self) -> list[IModuleJob]:
        return list(self._jobs)

    def get_job(self, job_name: str) -> IModuleJob | None:
        for job in self._jobs:
            if job.name == job_name:
                return job
        return None

    def mount(self, app: FastAPI, **kwargs) -> None:
        logger.info("storages module mounted")
        app.include_router(routers.router_folders, **kwargs)
        app.include_router(routers.router_files, **kwargs)
