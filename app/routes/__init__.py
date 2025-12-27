"""
API Routes for Orange3 Web.
"""

from .workflow import router as workflow_router, websocket_endpoint
from .workflow import (
    get_workflow_adapter,
    create_workflow_adapter,
    delete_workflow_adapter,
    get_workflow_adapters,
)
from .widget_registry import (
    router as widget_registry_router,
    legacy_widgets_handler,
    set_registry_getter as set_widget_registry_getter,
    get_discovered_widgets,
)

__all__ = [
    "workflow_router",
    "websocket_endpoint",
    "get_workflow_adapter",
    "create_workflow_adapter",
    "delete_workflow_adapter",
    "get_workflow_adapters",
    "widget_registry_router",
    "legacy_widgets_handler",
    "set_widget_registry_getter",
    "get_discovered_widgets",
]

