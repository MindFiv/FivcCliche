__all__ = [
    "service_site",
]

import os

from fivcglue import LazyValue, IComponentSite
from fivcglue.implements.utils import load_component_site

service_site: LazyValue[IComponentSite] = LazyValue(
    lambda: load_component_site(
        filename=os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            "settings",
            "services.yml",
        ),
        fmt="yaml",
    )
)
