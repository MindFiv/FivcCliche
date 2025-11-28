from fastapi import FastAPI

from fivcglue import IComponentSite

from fivccliche.services.interfaces.modules import IModule
from .routers import router


class ModuleImpl(IModule):

    def __init__(self, _: IComponentSite, **kwargs):
        print("chats module initialized.")

    @property
    def name(self):
        return "charts"

    @property
    def description(self):
        return "Chat management module."

    def mount(self, app: FastAPI, **kwargs) -> None:
        print("chats module mounted.")
        app.include_router(router)
