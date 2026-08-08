"""Agent memories module registration."""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fivcglue import IComponentSite

from fivccliche.services.interfaces.modules import IModule

from . import routers


class ModuleImpl(IModule):
    """Agent memories module: read-only HTTP API over IUserMemoryProvider."""

    def __init__(self, _: IComponentSite, **kwargs):
        print("agent_memories module initialized.")

    @property
    def name(self):
        return "agent_memories"

    @property
    def description(self):
        return "Agent Memories viewing module."

    def mount(
        self,
        app: FastAPI,
        scheduler: AsyncIOScheduler | None = None,
        **kwargs,
    ) -> None:
        print("agent_memories module mounted.")
        app.include_router(routers.router_memories, **kwargs)
