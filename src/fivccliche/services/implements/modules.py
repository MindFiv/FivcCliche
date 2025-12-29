from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fivcglue import IComponentSite
from fivcglue.interfaces.utils import query_component
from fivccliche import __version__

from fivccliche.services.interfaces.modules import (
    IModule,
    IModuleSite,
)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    # Startup
    print("Application starting up...")
    yield
    # Shutdown
    print("Application shutting down...")


class ModuleSiteImpl(IModuleSite):

    def __init__(
        self,
        component_site: IComponentSite,
        name: str = "FivcCliche",
        description: str = "A production-ready, multi-user backend framework for AI agents.",
        prefix: str = "",
        modules: list[str] | None = None,
        **kwargs,  # ignore additional arguments
    ):
        self._name = name
        self._description = description
        self._prefix = prefix
        self._modules = {}

        for mod in modules or []:
            mod_com = query_component(component_site, IModule, name=mod)
            if not mod_com:
                raise ValueError(f"Module {mod} not found.")
            self.register_module(mod_com, **kwargs)

    def register_module(
        self,
        module: IModule,
        **kwargs,  # ignore additional arguments
    ) -> None:
        if module.name in self._modules:
            raise ValueError(f"Module {module.name} already registered.")
        self._modules[module.name] = module

    def list_modules(self, **kwargs) -> list[IModule]:
        return list(self._modules.values())

    def create_application(self, **kwargs) -> FastAPI:
        app = FastAPI(
            title=self._name,
            description=self._description,
            version=__version__,
            lifespan=_lifespan,
        )
        # add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        for module in self._modules.values():
            module.mount(app, prefix=self._prefix)

        return app

    def run_application(
        self,
        app: FastAPI,
        host: str = "0.0.0.0",
        port: int = 8000,
        **kwargs,  # ignore additional arguments
    ) -> None:
        from fastapi_cdn_host import patch_docs
        from uvicorn import run as uvicorn_run

        # patch docs
        patch_docs(app)
        uvicorn_run(app, host=host, port=port)
