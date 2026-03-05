"""
Shared application globals and singletons.

Module-level state that is initialized during startup and accessed
across multiple modules (health checks, data endpoints, lifespan, etc.).
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Server startup timestamp - used to detect server restarts
SERVER_START_TIME = int(time.time())


def get_server_version() -> str:
    """Read server version from VERSION file."""
    version_paths = [
        Path(__file__).parent.parent.parent / "VERSION",  # backend/../VERSION
        Path(__file__).parent.parent / "VERSION",  # backend/VERSION
        Path("VERSION"),  # current directory
    ]
    for path in version_paths:
        if path.exists():
            try:
                return path.read_text().strip()
            except Exception as e:
                logger.debug(f"Suppressed error: {e}")
    return "unknown"


SERVER_VERSION = get_server_version()

# Orange3 availability
try:
    from ..orange_adapter import (
        OrangeRegistryAdapter,
        get_availability,
        ORANGE3_AVAILABLE,
    )

    ORANGE_AVAILABLE = ORANGE3_AVAILABLE
except ImportError as e:
    logger.warning(f"Could not import Orange3 adapters: {e}")
    logger.info("Using fallback models")
    ORANGE_AVAILABLE = False
    ORANGE3_AVAILABLE = False
    OrangeRegistryAdapter = None  # type: ignore[assignment, misc]

    def get_availability():  # type: ignore[misc]
        return {"orange3": False}


# OpenTelemetry
try:
    from .telemetry import init_telemetry, get_telemetry, TelemetryConfig

    OTEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OpenTelemetry not available: {e}")
    OTEL_AVAILABLE = False
    init_telemetry = None  # type: ignore[assignment]
    get_telemetry = None  # type: ignore[assignment]
    TelemetryConfig = None  # type: ignore[assignment, misc]


# Widget registry (singleton, read-only after initialization)
_registry: Optional[Any] = None
_registry_initialized = False
_registry_lock = threading.Lock()


def get_registry() -> Optional[Any]:
    """Get or create the widget registry (thread-safe singleton)."""
    global _registry, _registry_initialized
    if _registry_initialized:
        return _registry
    with _registry_lock:
        if not _registry_initialized:  # double-check after acquiring lock
            if ORANGE_AVAILABLE and OrangeRegistryAdapter is not None:
                _registry = OrangeRegistryAdapter()
                _registry.discover_widgets()
            else:
                _registry = None
            _registry_initialized = True
    return _registry


# WorkflowManager singleton (initialized in lifespan after registry is ready)
_global_workflow_manager: Optional[Any] = None


def get_workflow_manager() -> Optional[Any]:
    """Get the global WorkflowManager instance."""
    return _global_workflow_manager


def set_global_workflow_manager(wm: Any) -> None:
    """Set the global WorkflowManager instance (called during startup)."""
    global _global_workflow_manager
    _global_workflow_manager = wm
