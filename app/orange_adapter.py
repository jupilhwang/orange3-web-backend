"""
Backward-compatibility shim.

All components have been moved to ``app.core`` sub-modules.
Import directly from those modules for new code:

- ``app.core.orange_models``   — data classes, constants, Orange3 imports
- ``app.core.orange_utils``    — icon/color/path/addon helpers
- ``app.core.widget_discovery`` — FilesystemWidgetDiscovery
- ``app.core.registry_adapter`` — OrangeRegistryAdapter
- ``app.core.scheme_adapter``   — OrangeSchemeAdapter
"""

from app.core.orange_models import *  # noqa: F401,F403
from app.core.orange_utils import *  # noqa: F401,F403
from app.core.widget_discovery import *  # noqa: F401,F403
from app.core.registry_adapter import *  # noqa: F401,F403
from app.core.scheme_adapter import *  # noqa: F401,F403

__all__ = [
    # Orange3 availability
    "ORANGE3_AVAILABLE",
    "get_availability",
    # Adapters
    "OrangeRegistryAdapter",
    "OrangeSchemeAdapter",
    # Data classes
    "WebSchemeNode",
    "WebSchemeLink",
    "WebAnnotation",
    # Widget discovery
    "FilesystemWidgetDiscovery",
    "discover_widgets",
    "get_widget_discovery",
    "CATEGORY_COLORS",
    "CATEGORY_PRIORITIES",
]
