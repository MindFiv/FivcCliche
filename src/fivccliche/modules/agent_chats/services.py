from fastapi import FastAPI

from fivcglue import IComponentSite

from fivccliche.services.interfaces.modules import IModule
from .routers import router


class ModuleImpl(IModule):

    def __init__(self, _: IComponentSite, **kwargs):
        print("agent chats module initialized.")

    @property
    def name(self):
        return "agent_charts"

    @property
    def description(self):
        return "Agent Chat management module."

    def mount(self, app: FastAPI, **kwargs) -> None:
        print("agent_chats module mounted.")
        app.include_router(router)
