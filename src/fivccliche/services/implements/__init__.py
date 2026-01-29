__all__ = [
    "service_site",
]

import os

from fivcglue import LazyValue, IComponentSite
from fivcglue.implements.utils import load_component_site

service_yml = os.environ.get("SERVICE_YAML") or os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    "settings",
    "services.yml",
)
service_yml = os.path.abspath(service_yml)

service_site: LazyValue[IComponentSite] = LazyValue(
    lambda: load_component_site(
        filename=service_yml,
        fmt="yaml",
    )
)
