"""
Orange3 Adapters for Orange3 Web Backend.
- orange_adapter: Wrappers for Orange3 classes
- widget_discovery: AST-based widget discovery
"""

from .orange_adapter import (
    OrangeRegistryAdapter,
    OrangeSchemeAdapter,
    WebSchemeNode,
    WebSchemeLink,
    WebAnnotation,
    get_availability,
    ORANGE3_AVAILABLE,
)
from .widget_discovery import (
    WidgetDiscovery,
    get_widget_discovery,
    discover_widgets,
    CATEGORY_COLORS,
    CATEGORY_PRIORITIES,
)

__all__ = [
    # orange_adapter
    "OrangeRegistryAdapter",
    "OrangeSchemeAdapter",
    "WebSchemeNode",
    "WebSchemeLink",
    "WebAnnotation",
    "get_availability",
    "ORANGE3_AVAILABLE",
    # widget_discovery
    "WidgetDiscovery",
    "get_widget_discovery",
    "discover_widgets",
    "CATEGORY_COLORS",
    "CATEGORY_PRIORITIES",
]


