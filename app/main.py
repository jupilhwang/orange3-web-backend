"""
Orange3 Web Backend - Multi-tenant FastAPI Application

This backend REUSES the existing orange-canvas-core and orange-widget-base code.
It only adds a thin web API layer on top.

Features:
- SQLite database for persistence
- Async locks for concurrent access protection
- Multi-tenant support via X-Tenant-ID header
"""

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    HTTPException,
    APIRouter,
    UploadFile,
    File,
    Header,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import ipaddress
import os
import logging
import threading
from urllib.parse import urlparse
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, List, Optional, Any
import json
import uuid
import time
from pydantic import BaseModel

# Server startup timestamp - used to detect server restarts
SERVER_START_TIME = int(time.time())


# Server version - read from VERSION file
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
            except Exception:
                pass
    return "unknown"


SERVER_VERSION = get_server_version()


def validate_url_for_ssrf(url: str) -> tuple[bool, str]:
    """
    Validate URL to prevent SSRF attacks.

    Returns:
        (is_valid, error_message)
    """
    try:
        parsed = urlparse(url)

        # Must have scheme
        if not parsed.scheme:
            return False, "URL must have a scheme (http/https)"

        # Only allow http/https
        if parsed.scheme not in ("http", "https"):
            return False, f"Scheme '{parsed.scheme}' not allowed. Only http/https."

        # Must have hostname
        if not parsed.hostname:
            return False, "URL must have a hostname"

        # Block common cloud metadata endpoints
        blocked_hostnames = [
            "169.254.169.254",  # AWS/Azure/GCP metadata
            "metadata.google.internal",
            "metadata",
        ]
        if parsed.hostname.lower() in blocked_hostnames:
            return False, f"Blocked hostname: {parsed.hostname}"

        # Block private/loopback IPs
        try:
            ip = ipaddress.ip_address(parsed.hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
                return (
                    False,
                    f"Private/internal IP addresses not allowed: {parsed.hostname}",
                )
        except ValueError:
            # Not an IP, it's a hostname - resolve and check
            import socket

            try:
                resolved_ip = socket.gethostbyname(parsed.hostname)
                ip = ipaddress.ip_address(resolved_ip)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    return False, f"Hostname resolves to private IP: {resolved_ip}"
            except socket.gaierror:
                return False, f"Cannot resolve hostname: {parsed.hostname}"

        return True, ""

    except Exception as e:
        return False, f"Invalid URL: {str(e)}"


# Setup logging
logger = logging.getLogger(__name__)

# Database and locks (from core/)
from .core import init_db, close_db, get_db, async_session_maker
from sqlalchemy.ext.asyncio import AsyncSession

# Import adapters that wrap existing Orange3 code
try:
    from .orange_adapter import (
        OrangeSchemeAdapter,
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

    def get_availability():
        return {"orange3": False}


from .core.models import (
    Workflow,
    WorkflowSummary,
    WorkflowCreate,
    WorkflowUpdate,
    WorkflowNode,
    NodeCreate,
    NodeUpdate,
    NodeState,
    WorkflowLink,
    LinkCreate,
    LinkUpdate,
    TextAnnotation,
    ArrowAnnotation,
    AnnotationCreate,
    WidgetDescription,
    WidgetCategory,
    Position,
    Rect,
    Tenant,
)

# Managers
from .core import TenantManager, get_current_tenant
from .websocket_manager import TaskWebSocketManager, task_ws_manager

# OpenTelemetry
try:
    from .core.telemetry import init_telemetry, get_telemetry, TelemetryConfig

    OTEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OpenTelemetry not available: {e}")
    OTEL_AVAILABLE = False
    init_telemetry = None
    get_telemetry = None

# Widget API routers
from .widgets import (
    scatter_plot_router,
    distributions_router,
    bar_plot_router,
    heat_map_router,
    select_columns_router,
    select_rows_router,
    file_upload_router,
    data_sampler_router,
    datasets_router,
    knn_router,
    tree_router,
    naive_bayes_router,
    logistic_regression_router,
    random_forest_router,
    linear_regression_router,
    predictions_router,
    test_and_score_router,
    confusion_matrix_router,
    kmeans_router,
    corpus_router,
    preprocess_text_router,
    bag_of_words_router,
    word_cloud_router,
    data_info_router,
    feature_statistics_router,
)
from .core.task_api import router as task_api_router
from .core.config import get_upload_dir, get_config
from .core.mock_data import get_mock_data_info
from .routes import (
    workflow_router,
    widget_registry_router,
    websocket_endpoint as workflow_ws_endpoint,
    get_workflow_adapters,
    legacy_widgets_handler,
    set_registry_getter,
    set_registry_getter as set_widget_registry_getter,
    set_websocket_manager,
)

# mDNS Service Discovery
from .mdns import (
    MDNSService,
    MDNSConfig as MDNSServiceConfig,
    is_mdns_available,
    set_mdns_service,
)


# ============================================================================
# Managers (Thread-safe singletons)
# ============================================================================

tenant_manager = TenantManager()
# Task WebSocket 매니저는 websocket_manager.py에서 전역 인스턴스 사용

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
            if ORANGE_AVAILABLE:
                _registry = OrangeRegistryAdapter()
                _registry.discover_widgets()
            else:
                _registry = None
            _registry_initialized = True
    return _registry


# ============================================================================
# Lifespan helpers
# ============================================================================


def _init_telemetry(fastapi_app: FastAPI) -> None:
    """Initialize OpenTelemetry if available and enabled."""
    if not (OTEL_AVAILABLE and init_telemetry):
        return

    otel_endpoint = os.getenv("OTEL_ENDPOINT")
    otel_enabled = os.getenv("OTEL_ENABLED", "true").lower() == "true"

    if otel_enabled:
        config = TelemetryConfig(
            service_name="orange3-web-backend",
            service_version=SERVER_VERSION,
            environment=os.getenv("ENVIRONMENT", "development"),
            otel_endpoint=otel_endpoint,
            enable_console=os.getenv("OTEL_CONSOLE", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
        init_telemetry(fastapi_app, config)
        logger.info(f"OpenTelemetry initialized (endpoint: {otel_endpoint or 'none'})")


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
    set_widget_registry_getter(get_registry)


async def _init_task_worker() -> bool:
    """Start background task worker if enabled. Returns whether worker was started."""
    task_worker_enabled = os.getenv("TASK_WORKER_ENABLED", "true").lower() == "true"
    if not task_worker_enabled:
        return False

    from .core.task_queue import (
        start_worker,
        cleanup_stale_tasks,
        set_progress_callback,
        set_completion_callback,
    )

    # Import tasks to register them
    import app.tasks  # noqa: F401

    # Wire WebSocket callbacks for task progress/completion
    async def on_progress(task_id: str, progress: float, message: str = None) -> None:
        from .core.task_queue import get_task_status

        task_info = await get_task_status(task_id)
        if task_info:
            await task_ws_manager.send_progress(
                task_id, task_info["tenant_id"], progress, message
            )

    async def on_completion(task_id: str, status: str, result: Any = None, error: str = None) -> None:
        from .core.task_queue import get_task_status

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
    """Register mDNS service for discovery if configured. Returns mdns_service or None."""
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
        service_type = service_type + (
            "local." if service_type.endswith(".") else ".local."
        )

    mdns_svc_config = MDNSServiceConfig(
        enabled=mdns_config.enabled,
        service_type=service_type,
        service_name=mdns_config.service_name,
        port=app_config.server.port,  # use server.port, not mdns.port
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


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""

    logger.info("=" * 60)
    logger.info("Starting Orange3 Web Backend...")
    logger.info("=" * 60)

    _init_telemetry(fastapi_app)

    app_config = get_config()

    await _init_database()
    _init_registry()

    set_websocket_manager(task_ws_manager)
    logger.info("Async locks enabled for concurrent access protection")

    task_worker_enabled = await _init_task_worker()

    mdns_service = await _init_mdns(app_config)
    fastapi_app.state.mdns_service = mdns_service

    logger.info("=" * 60)

    yield

    # Cleanup
    if fastapi_app.state.mdns_service:
        logger.info("Unregistering mDNS service...")
        await fastapi_app.state.mdns_service.unregister()

    logger.info("Shutting down Orange3 Web Backend...")

    if task_worker_enabled:
        from .core.task_queue import stop_worker

        await stop_worker()
        logger.info("Task Queue worker stopped.")

    from .core.data_utils import shutdown_executors

    shutdown_executors()
    logger.info("Executor pools shutdown.")

    await close_db()
    logger.info("Database connections closed.")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Orange3 Web API",
    description="""
    Multi-tenant web-based Orange3 workflow canvas backend.
    
    **This backend REUSES existing orange-canvas-core and orange-widget-base code.**
    
    ## Features
    - SQLite database for persistence
    - Async locks for concurrent access protection
    - Multi-tenant support via X-Tenant-ID header
    
    ## Architecture
    - Wraps existing `Scheme`, `SchemeNode`, `SchemeLink` classes
    - Uses existing `WidgetRegistry` and `WidgetDescription`
    - Uses existing `compatible_channels` for type checking
    - Uses existing `scheme_to_ows_stream` for .ows export
    
    ## Authentication
    Use `X-Tenant-ID` header for multi-tenant access.
    """,
    version=SERVER_VERSION,
    lifespan=lifespan,
)

# CORS
# Allow runtime configuration via CORS_ALLOW_ORIGINS (comma-separated list).
# Defaults to wildcard "*" for development; set explicit origins in production.
_cors_env = os.environ.get("CORS_ALLOW_ORIGINS", "*")
cors_origins = [o.strip() for o in _cors_env.split(",")] if _cors_env != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Reverse proxy header support
class ProxyHeadersMiddleware:
    """Process X-Forwarded-* headers from trusted reverse proxies."""

    TRUSTED_CIDRS = [
        ipaddress.ip_network(cidr.strip())
        for cidr in os.environ.get(
            "TRUSTED_PROXIES",
            "127.0.0.0/8,::1/128,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16",
        ).split(",")
        if cidr.strip()
    ]

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] in ("http", "websocket"):
            client = scope.get("client")
            if client:
                client_ip = ipaddress.ip_address(client[0])
                is_trusted = any(client_ip in network for network in self.TRUSTED_CIDRS)

                headers = dict(scope.get("headers", []))
                if is_trusted:
                    # Trust X-Forwarded-For
                    xff = headers.get(b"x-forwarded-for", b"").decode()
                    if xff:
                        # Use the leftmost IP as the real client
                        real_ip = xff.split(",")[0].strip()
                        scope["client"] = (real_ip, client[1])
                else:
                    # Strip forwarded headers from untrusted sources
                    filtered_headers = [
                        (k, v)
                        for k, v in scope.get("headers", [])
                        if k.lower()
                        not in (
                            b"x-forwarded-for",
                            b"x-forwarded-proto",
                            b"x-forwarded-host",
                            b"x-real-ip",
                        )
                    ]
                    scope["headers"] = filtered_headers

        await self.app(scope, receive, send)


app.add_middleware(ProxyHeadersMiddleware)


# ============================================================================
# API Router
# ============================================================================

api_v1 = APIRouter(prefix="/api/v1")


# ============================================================================
# Health
# ============================================================================


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    import time

    availability = get_availability()
    config = get_config()
    storage_type = (
        config.storage.type
    )  # 'sqlite', 'mysql', 'postgresql', 'oracle', 'filesystem', 'local'

    # Database storage types
    db_storage_types = {"sqlite", "mysql", "postgresql", "oracle", "database"}

    # Storage path depends on storage type
    if storage_type in db_storage_types:
        storage_path = config.database.url or "sqlite:///./orange3.db"
    else:
        storage_path = str(get_upload_dir())

    # Calculate uptime
    uptime_seconds = time.time() - SERVER_START_TIME

    # Determine database type from URL
    db_url = config.database.url or ""
    if "postgresql" in db_url or "postgres" in db_url:
        database_type = "postgresql"
    elif "mysql" in db_url:
        database_type = "mysql"
    elif "oracle" in db_url:
        database_type = "oracle"
    else:
        database_type = "sqlite"

    return {
        "status": "healthy",
        "service": "orange3-web-backend",
        "version": SERVER_VERSION,
        "uptime_seconds": round(uptime_seconds, 2),
        "orange3_available": availability.get("orange3", False),
        "database_type": database_type,
        "storage_type": storage_type,
        "storage_path": storage_path,
        "max_file_size_mb": config.storage.max_db_file_size // (1024 * 1024),
        "lock_type": "asyncio",
    }


@app.get("/internal/metrics")
async def get_metrics() -> dict:
    """Get OpenTelemetry metrics summary."""
    if OTEL_AVAILABLE and get_telemetry:
        telemetry = get_telemetry()
        if telemetry:
            return telemetry.get_metrics_summary()
    return {
        "service": "orange3-web-backend",
        "version": SERVER_VERSION,
        "otel_available": OTEL_AVAILABLE,
        "message": "OpenTelemetry not initialized",
    }


@app.get("/internal/logs")
async def get_logs(limit: int = 100) -> dict:
    """Get recent log entries."""
    if OTEL_AVAILABLE and get_telemetry:
        telemetry = get_telemetry()
        if telemetry:
            return telemetry.get_logs_response(limit)
    return {"service": "orange3-web-backend", "total": 0, "logs": []}


@app.get("/internal/resources")
async def get_resources() -> dict:
    """Get current resource usage (CPU, Memory, Disk, Throughput)."""
    if OTEL_AVAILABLE and get_telemetry:
        telemetry = get_telemetry()
        if telemetry:
            return telemetry.get_resource_usage()

    # Fallback without OpenTelemetry
    try:
        import psutil

        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return {
            "psutil_available": True,
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": memory.percent,
            "memory_used_mb": int(memory.used / (1024 * 1024)),
            "disk_percent": disk.percent,
            "disk_used_gb": round(disk.used / (1024 * 1024 * 1024), 2),
        }
    except ImportError:
        return {
            "psutil_available": False,
            "message": "psutil not installed. Run: pip install psutil",
        }


@app.get("/health/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)) -> dict:
    """Readiness probe - checks database connection."""
    from sqlalchemy import text

    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ready", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database not ready: {e}")


@app.get("/health/live")
async def liveness_check() -> dict:
    """Liveness probe - checks if service is alive."""
    return {"status": "alive"}


# ============================================================================
# Task Progress WebSocket
# ============================================================================


@app.websocket("/ws/tasks/{tenant_id}")
async def task_websocket_endpoint(websocket: WebSocket, tenant_id: str) -> None:
    """
    Task 진행률 및 완료 알림 WebSocket 엔드포인트.

    클라이언트는 연결 후 특정 task_id를 구독할 수 있습니다:
    - {"action": "subscribe", "task_id": "..."} - 특정 Task 구독
    - {"action": "subscribe", "task_id": "*"} - 모든 Task 구독
    - {"action": "unsubscribe", "task_id": "..."} - 구독 해제

    서버는 다음 메시지를 전송합니다:
    - {"type": "task_progress", "task_id": "...", "progress": 50.0, "message": "..."}
    - {"type": "task_completed", "task_id": "...", "status": "completed", "result": {...}}
    - {"type": "task_completed", "task_id": "...", "status": "failed", "error": "..."}
    """
    await task_ws_manager.connect(websocket, tenant_id)

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            task_id = data.get("task_id")

            if action == "subscribe" and task_id:
                task_ws_manager.subscribe(websocket, task_id)
                await websocket.send_json({"type": "subscribed", "task_id": task_id})
            elif action == "unsubscribe" and task_id:
                task_ws_manager.unsubscribe(websocket, task_id)
                await websocket.send_json({"type": "unsubscribed", "task_id": task_id})
            elif action == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        task_ws_manager.disconnect(websocket, tenant_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        task_ws_manager.disconnect(websocket, tenant_id)


# WebSocket 연결 수 조회 엔드포인트
@app.get("/internal/ws-stats")
async def get_websocket_stats() -> dict:
    """WebSocket 연결 통계."""
    return {"total_connections": task_ws_manager.get_connection_count(), "status": "ok"}


# ============================================================================
# Data Loading Endpoints
# ============================================================================


class UrlLoadRequest(BaseModel):
    url: str


def _extract_domain_columns(domain) -> list[dict]:
    """Extract column metadata from an Orange3 domain."""
    columns = []
    for var in domain.attributes:
        columns.append(
            {
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical",
                "role": "feature",
                "values": ", ".join(var.values)
                if hasattr(var, "values") and var.values
                else "",
            }
        )
    if domain.class_var:
        var = domain.class_var
        columns.append(
            {
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical",
                "role": "target",
                "values": ", ".join(var.values)
                if hasattr(var, "values") and var.values
                else "",
            }
        )
    for var in domain.metas:
        columns.append(
            {
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical",
                "role": "meta",
                "values": "",
            }
        )
    return columns


async def _resolve_file_path(
    path: str, tenant_id: Optional[str]
) -> tuple[str, Any, Any]:
    """Resolve path to actual filesystem path, returning (actual_path, temp_file, metadata).

    Handles file: prefix (storage lookup), uploads/ prefix, and datasets/ prefix.
    Callers are responsible for cleaning up temp_file if not None.
    """
    import tempfile
    from .core.file_storage import get_file, get_file_metadata

    actual_path = path
    temp_file = None
    metadata = None

    if path.startswith("file:"):
        file_id = path.replace("file:", "")
        logger.info(f"Loading file by ID: {file_id}, tenant: {tenant_id}")

        metadata = await get_file_metadata(file_id, tenant_id)
        if not metadata:
            return path, None, None  # Signal: file not found

        content = await get_file(file_id, tenant_id)
        if not content:
            return path, None, metadata  # Signal: content not found

        suffix = Path(metadata.filename).suffix or ".tab"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(content)
        temp_file.close()
        actual_path = temp_file.name
        logger.info(f"Created temp file: {actual_path} for {metadata.filename}")

    elif path.startswith("uploads/"):
        from .core.config import get_upload_dir as _get_upload_dir

        upload_dir = _get_upload_dir()
        full_path = upload_dir / path.replace("uploads/", "")
        if full_path.exists():
            actual_path = str(full_path)

    elif path.startswith("datasets/"):
        # Built-in Orange3 datasets — strip path prefix and extension
        actual_path = path.replace("datasets/", "").split(".")[0]

    return actual_path, temp_file, metadata


def _apply_metadata_overrides(
    columns: list, path: str, tenant_id: Optional[str]
) -> list:
    """Read .metadata.json sidecar for the given path and apply column type/role overrides."""
    from .core.config import get_tenant_upload_dir

    file_id_for_meta = path
    if path.startswith("file:"):
        file_id_for_meta = path.replace("file:", "")
    elif path.startswith("uploads/"):
        file_id_for_meta = path.replace("uploads/", "")

    upload_dir = get_tenant_upload_dir(tenant_id or "default")
    metadata_path = upload_dir / f"{file_id_for_meta}.metadata.json"

    if not metadata_path.exists():
        return columns

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_overrides = json.load(f)

        meta_cols = {col["name"]: col for col in metadata_overrides.get("columns", [])}

        for col in columns:
            if col["name"] in meta_cols:
                override = meta_cols[col["name"]]
                col["type"] = override.get("type", col["type"])
                col["role"] = override.get("role", col["role"])

        logger.info(f"Applied column metadata overrides from {metadata_path}")
    except Exception as e:
        logger.warning(f"Failed to load metadata overrides: {e}")

    return columns


def _paginate_orange_data(
    data: Any, offset: int, limit: int
) -> tuple[list | None, dict | None]:
    """Slice Orange Table rows and build a pagination dict.

    Returns (paginated_rows_list, pagination_dict), or (None, None) when limit <= 0.
    """
    total_rows = len(data)

    if limit is None or limit <= 0:
        return None, None

    end_idx = min(offset + limit, total_rows)
    paginated_rows = data[offset:end_idx]

    paginated_data = []
    for row in paginated_rows:
        row_data = []
        for val in row:
            if hasattr(val, "is_nan") and val.is_nan():
                row_data.append(None)
            else:
                row_data.append(float(val) if not isinstance(val, str) else val)
        if data.domain.class_var:
            class_val = row.get_class()
            if hasattr(class_val, "is_nan") and class_val.is_nan():
                row_data.append(None)
            else:
                row_data.append(str(class_val))
        paginated_data.append(row_data)

    pagination = {
        "offset": offset,
        "limit": limit,
        "total": total_rows,
        "hasMore": end_idx < total_rows,
    }

    return paginated_data, pagination


@api_v1.get("/data/load", tags=["Data"])
async def load_data_from_path(
    path: str,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    offset: int = 0,
    limit: Optional[int] = None,
) -> dict:
    """Load data from a local file path or file ID.

    [BE-PERF-001] Pagination support added for large datasets.

    Args:
        path: File path or file ID (file:{uuid})
        x_tenant_id: Tenant ID header
        offset: Starting row index (default: 0)
        limit: Maximum number of rows to return (default: None = all rows)
    """
    # If Orange3 not available, return mock data
    if not ORANGE_AVAILABLE:
        return get_mock_data_info(path)

    actual_path, temp_file, metadata = await _resolve_file_path(path, x_tenant_id)

    # Handle file-not-found / content-not-found cases from _resolve_file_path
    if path.startswith("file:") and metadata is None:
        return {
            "name": "File Not Found",
            "description": "파일을 찾을 수 없습니다. 파일을 다시 업로드해주세요.",
            "path": path,
            "instances": 0,
            "features": 0,
            "missingValues": False,
            "classType": "None",
            "classValues": None,
            "metaAttributes": 0,
            "columns": [],
            "error": f"File not found: {path.replace('file:', '')}",
        }

    if path.startswith("file:") and temp_file is None and metadata is not None:
        file_id = path.replace("file:", "")
        return {
            "name": metadata.original_filename or metadata.filename,
            "description": "파일 내용을 읽을 수 없습니다.",
            "path": path,
            "instances": 0,
            "features": 0,
            "missingValues": False,
            "classType": "None",
            "classValues": None,
            "metaAttributes": 0,
            "columns": [],
            "error": f"File content not found: {file_id}",
        }

    try:
        from Orange.data import Table

        data = Table(actual_path)

        columns = _extract_domain_columns(data.domain)
        columns = _apply_metadata_overrides(columns, path, x_tenant_id)

        # Build display name
        if path.startswith("file:") and metadata:
            original_name = metadata.original_filename or metadata.filename
            display_name = Path(original_name).stem
        else:
            display_name = data.name or path.split("/")[-1].split(":")[-1]

        # [BE-PERF-001] Apply pagination if specified
        paginated_data, pagination = _paginate_orange_data(data, offset, limit or 0)

        response = {
            "name": display_name,
            "description": "",
            "path": path,
            "instances": len(data),
            "features": len(data.domain.attributes),
            "missingValues": data.has_missing(),
            "classType": "Classification"
            if data.domain.class_var and not data.domain.class_var.is_continuous
            else "Regression"
            if data.domain.class_var
            else "None",
            "classValues": len(data.domain.class_var.values)
            if data.domain.class_var and hasattr(data.domain.class_var, "values")
            else None,
            "metaAttributes": len(data.domain.metas),
            "columns": columns,
        }

        if pagination:
            response["data"] = paginated_data
            response["pagination"] = pagination

        return response
    except Exception as e:
        logger.error(f"Failed to load data from {actual_path}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to load data from '{path}': {str(e)}"
        )
    finally:
        if temp_file and Path(temp_file.name).exists():
            try:
                Path(temp_file.name).unlink()
            except Exception:
                pass


@api_v1.post("/data/load-url", tags=["Data"])
async def load_data_from_url(request: UrlLoadRequest) -> dict:
    """
    Load data from a URL.

    Security: Validates URL to prevent SSRF attacks.
    """
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Orange3 not available")

    # Validate URL for SSRF
    url = request.url
    is_valid, error_msg = validate_url_for_ssrf(url)
    if not is_valid:
        logger.warning(f"SSRF attempt blocked: {url} - {error_msg}")
        raise HTTPException(status_code=403, detail=f"URL not allowed: {error_msg}")

    try:
        from Orange.data import Table
        import tempfile
        import urllib.request
        import os

        # Download to temp file
        logger.info(f"Downloading from validated URL: {url}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            tmp_path = tmp.name

        try:
            # Try to load the data
            data = Table(tmp_path)

            # Get column info
            columns = _extract_domain_columns(data.domain)

            return {
                "name": url.split("/")[-1],
                "description": f"Loaded from {url}",
                "instances": len(data),
                "features": len(data.domain.attributes),
                "missingValues": data.has_missing(),
                "classType": "Classification"
                if data.domain.class_var and not data.domain.class_var.is_continuous
                else "Regression"
                if data.domain.class_var
                else "None",
                "classValues": len(data.domain.class_var.values)
                if data.domain.class_var and hasattr(data.domain.class_var, "values")
                else None,
                "metaAttributes": len(data.domain.metas),
                "columns": columns,
            }
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Legacy Endpoints
# ============================================================================


@app.get("/api/widgets")
async def legacy_widgets() -> dict:
    """Legacy endpoint for frontend compatibility."""
    return await legacy_widgets_handler()


# ============================================================================
# Include Widget Routers
# ============================================================================

api_v1.include_router(scatter_plot_router)
api_v1.include_router(distributions_router)
api_v1.include_router(bar_plot_router)
api_v1.include_router(heat_map_router)
api_v1.include_router(select_columns_router)
api_v1.include_router(select_rows_router)
api_v1.include_router(file_upload_router)
api_v1.include_router(data_sampler_router)
api_v1.include_router(datasets_router)
api_v1.include_router(knn_router)
api_v1.include_router(tree_router)
api_v1.include_router(naive_bayes_router)
api_v1.include_router(logistic_regression_router)
api_v1.include_router(random_forest_router)
api_v1.include_router(linear_regression_router)
api_v1.include_router(predictions_router)
api_v1.include_router(test_and_score_router)
api_v1.include_router(confusion_matrix_router)
api_v1.include_router(kmeans_router)
api_v1.include_router(corpus_router)
api_v1.include_router(preprocess_text_router)
api_v1.include_router(bag_of_words_router)
api_v1.include_router(word_cloud_router)
api_v1.include_router(data_info_router)
api_v1.include_router(feature_statistics_router)

# Include Task Queue Router
api_v1.include_router(task_api_router)

# Include Workflow Router
api_v1.include_router(workflow_router)

# Include Widget Registry Router
api_v1.include_router(widget_registry_router)

# ============================================================================
# Include Main Router
# ============================================================================

app.include_router(api_v1)


# WebSocket endpoint (registered directly on app, not api_v1)
@app.websocket("/api/v1/workflows/{workflow_id}/ws")
async def websocket_endpoint(websocket: WebSocket, workflow_id: str) -> None:
    """WebSocket for real-time updates."""
    await workflow_ws_endpoint(websocket, workflow_id)


if __name__ == "__main__":
    import uvicorn

    trusted_proxies = os.environ.get(
        "TRUSTED_PROXIES", "127.0.0.0/8,::1/128,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
    )
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        proxy_headers=True,
        forwarded_allow_ips=trusted_proxies,
    )
