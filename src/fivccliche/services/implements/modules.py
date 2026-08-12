from contextlib import asynccontextmanager
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fivcglue import IComponentSite
from fivcglue.interfaces.utils import query_component
from fivccliche import __version__

from fivccliche.services.interfaces.modules import (
    IModule,
    IModuleJob,
    IModuleSite,
)


def _make_lifespan(scheduler: AsyncIOScheduler):
    @asynccontextmanager
    async def _lifespan(_: FastAPI):
        # Startup
        print("Application starting up...")
        scheduler.start()
        yield
        # Shutdown
        print("Application shutting down...")
        scheduler.shutdown(wait=False)

    return _lifespan


def _register_module_job(scheduler: AsyncIOScheduler, job: IModuleJob) -> None:
    """Register an IModuleJob on the shared scheduler from job.config."""
    cfg = dict(job.config)
    trigger = cfg.pop("trigger")
    replace_existing = cfg.pop("replace_existing", True)
    scheduler.add_job(
        job.run_async,
        trigger=trigger,
        id=job.name,
        replace_existing=replace_existing,
        **cfg,
    )


class ModuleSiteImpl(IModuleSite):

    def __init__(
        self,
        component_site: IComponentSite,
        name: str = "FivcCliche",
        description: str = "A production-ready, multi-user backend framework for AI agents.",
        prefix: str = "",
        docs_prefix: str = "",
        modules: list[str] | None = None,
        **kwargs,  # ignore additional arguments
    ):
        self._name = name
        self._description = description
        self._prefix = prefix
        self._docs_prefix = docs_prefix
        self._modules: dict[str, IModule] = {}

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
        scheduler = AsyncIOScheduler()

        app_kwargs: dict[str, Any] = {
            "docs_url": f"{self._docs_prefix}/docs",
            "redoc_url": f"{self._docs_prefix}/redoc",
            "openapi_url": f"{self._docs_prefix}/openapi.json",
            "title": self._name,
            "description": self._description,
            "version": __version__,
            "lifespan": _make_lifespan(scheduler),
        }
        app_kwargs = {k: v for k, v in app_kwargs.items() if v is not None}
        app = FastAPI(**app_kwargs)
        # add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        app.state.scheduler = scheduler

        for module in self._modules.values():
            module.mount(app, prefix=self._prefix)
            for job in module.list_jobs():
                _register_module_job(scheduler, job)

        return app

    def run_application(
        self,
        app: FastAPI,
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = True,
        **kwargs,  # ignore additional arguments
    ) -> None:
        from fastapi_cdn_host import patch_docs
        from uvicorn import run as uvicorn_run

        # patch docs
        patch_docs(app)
        uvicorn_run(app, host=host, port=port, reload=reload)
