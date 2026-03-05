"""
Application lifespan management.

Contains the ``lifespan`` async context manager and all startup/shutdown
helpers (database init, registry setup, mDNS, task worker, telemetry).
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI

from .config import get_config
from .database import init_db, close_db
from .globals import (
    OTEL_AVAILABLE,
    SERVER_VERSION,
    TelemetryConfig,
    get_availability,
    get_registry,
    init_telemetry,
    set_global_workflow_manager,
)
from ..mdns import (
    MDNSConfig as MDNSServiceConfig,
    MDNSService,
    is_mdns_available,
    set_mdns_service,
)
from ..routes import (
    set_registry_getter,
    set_websocket_manager,
    set_workflow_manager,
)
from ..websocket_manager import task_ws_manager
from ..workflow_manager import WorkflowManager

logger = logging.getLogger(__name__)


# ============================================================================
# Startup helpers
# ============================================================================


def _init_telemetry(fastapi_app: FastAPI) -> None:
    """Initialize OpenTelemetry if available and enabled."""
    if not (OTEL_AVAILABLE and init_telemetry):
        return

    app_config = get_config()
    otel_enabled = app_config.otel.enabled
    otel_endpoint = app_config.otel.endpoint or os.getenv("OTEL_ENDPOINT")

    logger.info(
        f"[OTel] Config loaded — enabled={otel_enabled}, "
        f"endpoint={repr(otel_endpoint)}, "
        f"source={'file' if app_config.otel.endpoint else 'env/default'}"
    )

    if otel_enabled:
        config = TelemetryConfig(
            service_name=app_config.otel.service_name,
            service_version=app_config.otel.service_version,
            environment=app_config.otel.environment,
            otel_endpoint=otel_endpoint,
            enable_console=app_config.otel.enable_console,
            log_level=app_config.log.level,
            metric_interval_ms=app_config.otel.metric_interval_ms,
        )
        init_telemetry(fastapi_app, config)
        logger.info(
            f"OpenTelemetry initialized (endpoint: {config.get_otlp_endpoint() or 'none'})"
        )


async def _init_database() -> None:
    """Initialize database tables."""
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database ready (SQLite with WAL mode)")


def _init_registry() -> None:
    """Pre-load widget registry and wire up router dependencies."""
    availability = get_availability()
    logger.info(
        f"Orange3: {'Available' if availability.get('orange3') else 'Not available'}"
    )

    if not availability.get("orange3"):
        logger.info("Install with: pip install Orange3")

    registry = get_registry()
    if registry:
        categories = registry.list_categories()
        widgets = registry.list_widgets()
        logger.info(
            f"Discovered {len(widgets)} widgets in {len(categories)} categories"
        )

    set_registry_getter(get_registry)

    wm = WorkflowManager(registry=registry)
    set_global_workflow_manager(wm)
    logger.info(
        "WorkflowManager initialized"
        + (
            " with registry"
            if registry
            else " (no registry — permissive link validation)"
        )
    )
    return wm


async def _init_task_worker() -> bool:
    """Start background task worker if enabled."""
    task_worker_enabled = os.getenv("TASK_WORKER_ENABLED", "true").lower() == "true"
    if not task_worker_enabled:
        return False

    from .task_queue import (
        start_worker,
        cleanup_stale_tasks,
        set_progress_callback,
        set_completion_callback,
    )

    import app.tasks  # noqa: F401

    async def on_progress(task_id: str, progress: float, message: str = None) -> None:
        from .task_queue import get_task_status

        task_info = await get_task_status(task_id)
        if task_info:
            await task_ws_manager.send_progress(
                task_id, task_info["tenant_id"], progress, message
            )

    async def on_completion(
        task_id: str, status: str, result: Any = None, error: str = None
    ) -> None:
        from .task_queue import get_task_status

        task_info = await get_task_status(task_id)
        if task_info:
            await task_ws_manager.send_completion(
                task_id, task_info["tenant_id"], status, result, error
            )

    set_progress_callback(on_progress)
    set_completion_callback(on_completion)

    await start_worker(poll_interval=1.0)
    await cleanup_stale_tasks(timeout_minutes=30)
    logger.info("Task Queue worker started (WebSocket notifications enabled)")
    return True


async def _init_mdns(app_config) -> Optional[Any]:
    """Register mDNS service for discovery if configured."""
    mdns_config = app_config.mdns

    if not mdns_config.enabled:
        logger.info("mDNS disabled in configuration")
        return None

    if not is_mdns_available():
        logger.info("mDNS disabled (zeroconf not installed)")
        logger.info("Install with: pip install zeroconf")
        return None

    logger.info("Starting mDNS Service Discovery...")

    service_type = mdns_config.service_type
    if not service_type.endswith(".local."):
        service_type = service_type.rstrip(".") + ".local."

    mdns_svc_config = MDNSServiceConfig(
        enabled=mdns_config.enabled,
        service_type=service_type,
        service_name=mdns_config.service_name,
        port=app_config.server.port,
        multicast_address=mdns_config.multicast_address,
        udp_port=mdns_config.udp_port,
        interface=mdns_config.interface,
    )

    mdns_service = MDNSService(mdns_svc_config)
    set_mdns_service(mdns_service)

    txt_properties = {
        "version": SERVER_VERSION,
        "weight": "1",
        "environment": os.getenv("ENVIRONMENT", "development"),
    }

    await mdns_service.register(txt_properties)

    logger.info(
        f"mDNS registered - Multicast: {mdns_config.multicast_address}:{mdns_config.udp_port}"
    )
    if mdns_config.interface:
        logger.info(f"mDNS interface: {mdns_config.interface}")

    return mdns_service


# ============================================================================
# Lifespan context manager
# ============================================================================


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager — startup and shutdown logic."""

    logger.info("=" * 60)
    logger.info("Starting Orange3 Web Backend...")
    logger.info("=" * 60)

    _init_telemetry(fastapi_app)

    app_config = get_config()

    await _init_database()
    wm = _init_registry()

    set_websocket_manager(task_ws_manager)
    set_workflow_manager(wm)
    logger.info("Async locks enabled for concurrent access protection")

    task_worker_enabled = await _init_task_worker()

    mdns_service = await _init_mdns(app_config)
    fastapi_app.state.mdns_service = mdns_service

    # Background cleanup for DataSessionManager
    async def _session_cleanup_loop():
        while True:
            await asyncio.sleep(300)  # 5 minutes
            try:
                from .data_utils import DataSessionManager

                DataSessionManager.cleanup_expired()
            except Exception as e:
                logger.warning(f"Session cleanup error: {e}")

    cleanup_task = asyncio.create_task(_session_cleanup_loop())

    logger.info("=" * 60)

    yield

    # Cleanup
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    if fastapi_app.state.mdns_service:
        logger.info("Unregistering mDNS service...")
        await fastapi_app.state.mdns_service.unregister()

    logger.info("Shutting down Orange3 Web Backend...")

    if task_worker_enabled:
        from .task_queue import stop_worker

        await stop_worker()
        logger.info("Task Queue worker stopped.")

    from .data_utils import shutdown_executors

    shutdown_executors()
    logger.info("Executor pools shutdown.")

    await close_db()
    logger.info("Database connections closed.")
