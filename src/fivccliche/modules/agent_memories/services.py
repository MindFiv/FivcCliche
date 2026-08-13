"""Agent memories module registration."""

import logging

from fastapi import FastAPI
from fivcglue import IComponentSite

from fivccliche.services.interfaces.modules import IModule, IModuleJob

from . import routers

logger = logging.getLogger(__name__)


class ModuleImpl(IModule):
    """Agent memories module: read-only HTTP API over IUserMemoryProvider."""

    def __init__(self, _: IComponentSite, **kwargs):
        logger.info("agent_memories module initialized")

    @property
    def name(self):
        return "agent_memories"

    @property
    def description(self):
        return "Agent Memories viewing module."

    def list_jobs(self) -> list[IModuleJob]:
        return []

    def get_job(self, job_name: str) -> IModuleJob | None:
        return None

    def mount(self, app: FastAPI, **kwargs) -> None:
        logger.info("agent_memories module mounted")
        app.include_router(routers.router_memories, **kwargs)
