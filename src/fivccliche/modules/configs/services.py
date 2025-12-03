from fastapi import FastAPI

from fivcglue import IComponentSite
from fivccliche.services.interfaces.modules import IModule
from .routers import router_embeddings, router_models, router_agents


class ModuleImpl(IModule):
    """User module implementation."""

    def __init__(self, _: IComponentSite, **kwargs):
        print("configs module initialized...")

    @property
    def name(self):
        return "configs"

    @property
    def description(self):
        return "Configs management module."

    def mount(self, app: FastAPI, **kwargs) -> None:
        print("configs module mounted.")
        app.include_router(router_embeddings, prefix="/configs")
        app.include_router(router_models, prefix="/configs")
        app.include_router(router_agents, prefix="/configs")
