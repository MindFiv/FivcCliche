from abc import abstractmethod

from fastapi import FastAPI

from fivcglue import IComponent


class IModule(IComponent):
    """
    IModule is an interface for defining modules in the Fivccliche framework.
    """

    @property
    @abstractmethod
    def name(self):
        """Name of the module."""

    @property
    @abstractmethod
    def description(self):
        """Description of the module."""

    @abstractmethod
    def mount(self, app: FastAPI, **kwargs) -> None:
        """Mount the module to the FastAPI app."""


class IModuleSite(IComponent):
    """
    IModuleSite is an interface for defining site modules in the Fivccliche framework.
    """

    @abstractmethod
    def register_module(self, module: IModule, **kwargs) -> None:
        """Register a module."""

    @abstractmethod
    def list_modules(self, **kwargs) -> list[IModule]:
        """Unregister a module."""

    @abstractmethod
    def create_application(self, **kwargs) -> FastAPI:
        """Create a FastAPI app."""

    @abstractmethod
    def run_application(
        self,
        app: FastAPI,
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = True,
        **kwargs,  # ignore additional arguments
    ) -> None:
        """Run the FastAPI app. for development purposes."""
