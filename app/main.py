"""
Orange3 Web Backend - Multi-tenant FastAPI Application

This backend REUSES the existing orange-canvas-core and orange-widget-base code.
It only adds a thin web API layer on top.

Features:
- SQLite database for persistence
- Async locks for concurrent access protection
- Multi-tenant support via X-Tenant-ID header
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, APIRouter, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
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

# Setup logging
logger = logging.getLogger(__name__)

# Database and locks (from core/)
from .core import init_db, close_db, get_db, async_session_maker
from sqlalchemy.ext.asyncio import AsyncSession

# Import adapters that wrap existing Orange3 code
try:
    from .orange_adapter import (
        OrangeSchemeAdapter, OrangeRegistryAdapter, 
        get_availability, ORANGE3_AVAILABLE
    )
    ORANGE_AVAILABLE = ORANGE3_AVAILABLE
except ImportError as e:
    print(f"Warning: Could not import Orange3 adapters: {e}")
    print("Using fallback models")
    ORANGE_AVAILABLE = False
    ORANGE3_AVAILABLE = False
    
    def get_availability():
        return {"orange3": False}

from .core.models import (
    Workflow, WorkflowSummary, WorkflowCreate, WorkflowUpdate,
    WorkflowNode, NodeCreate, NodeUpdate, NodeState,
    WorkflowLink, LinkCreate, LinkUpdate,
    TextAnnotation, ArrowAnnotation, AnnotationCreate,
    WidgetDescription, WidgetCategory,
    Position, Rect, Tenant
)
# Managers
from .core import TenantManager, get_current_tenant
from .websocket_manager import WebSocketManager

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
from .core.config import get_upload_dir, get_config
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


# ============================================================================
# Managers (Thread-safe singletons)
# ============================================================================

tenant_manager = TenantManager()
websocket_manager = WebSocketManager()

# Widget registry (singleton, read-only after initialization)
_registry: Optional[Any] = None
_registry_initialized = False


def get_registry():
    """Get or create the widget registry (thread-safe singleton)."""
    global _registry, _registry_initialized
    if not _registry_initialized:
        if ORANGE_AVAILABLE:
            _registry = OrangeRegistryAdapter()
            _registry.discover_widgets()
        else:
            _registry = None
        _registry_initialized = True
    return _registry


# ============================================================================
# Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    import httpx
    
    print("=" * 60)
    print("Starting Orange3 Web Backend...")
    print("=" * 60)
    
    # Initialize OpenTelemetry
    if OTEL_AVAILABLE and init_telemetry:
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
            init_telemetry(app, config)
            print(f"✅ OpenTelemetry initialized (endpoint: {otel_endpoint or 'none'})")
    
    # Load balancer configuration
    # FRONTEND_URL can be comma-separated for multiple frontends
    frontend_url_env = os.getenv("FRONTEND_URL", "http://localhost:3000")
    frontend_urls = [url.strip() for url in frontend_url_env.split(",") if url.strip()]
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    lb_enabled = os.getenv("LB_ENABLED", "true").lower() == "true"
    lb_weight = int(os.getenv("LB_WEIGHT", "1"))
    
    # Track successfully registered frontends for cleanup
    registered_frontends = []
    
    # Initialize database
    print("\n📦 Initializing database...")
    await init_db()
    print("   Database ready (SQLite with WAL mode)")
    
    # Check Orange3 availability
    availability = get_availability()
    print(f"\n🍊 Orange3: {'✓ Available' if availability.get('orange3') else '✗ Not available'}")
    
    if not availability.get('orange3'):
        print("   Install with: pip install Orange3")
    
    # Pre-load registry
    registry = get_registry()
    if registry:
        categories = registry.list_categories()
        widgets = registry.list_widgets()
        print(f"\n📊 Discovered {len(widgets)} widgets in {len(categories)} categories")
    
    # Setup workflow router dependencies
    set_registry_getter(get_registry)
    set_websocket_manager(websocket_manager)
    
    # Setup widget registry router dependencies
    set_widget_registry_getter(get_registry)
    
    print("\n🔒 Async locks enabled for concurrent access protection")
    
    # Register with Frontend Load Balancer(s)
    if lb_enabled:
        print(f"\n⚖️  Load Balancer registration ({len(frontend_urls)} frontend(s))...")
        async with httpx.AsyncClient(timeout=5.0) as client:
            for frontend_url in frontend_urls:
                try:
                    resp = await client.post(
                        f"{frontend_url}/internal/register",
                        json={"url": backend_url, "weight": lb_weight}
                    )
                    if resp.status_code == 200:
                        print(f"   ✓ Registered with: {frontend_url} (backend: {backend_url})")
                        registered_frontends.append(frontend_url)
                    else:
                        print(f"   ✗ Registration failed ({frontend_url}): {resp.status_code}")
                except Exception as e:
                    print(f"   ✗ Could not register with {frontend_url}: {e}")
    
    # Store registered frontends in app state for cleanup
    app.state.registered_frontends = registered_frontends
    app.state.backend_url = backend_url
    app.state.lb_enabled = lb_enabled
    
    print("=" * 60)
    
    yield
    
    # Cleanup - Deregister from Load Balancer(s)
    if app.state.lb_enabled and app.state.registered_frontends:
        print(f"\n⚖️  Deregistering from Load Balancer ({len(app.state.registered_frontends)} frontend(s))...")
        async with httpx.AsyncClient(timeout=3.0) as client:
            for frontend_url in app.state.registered_frontends:
                try:
                    await client.post(
                        f"{frontend_url}/internal/deregister",
                        json={"url": app.state.backend_url}
                    )
                    print(f"   ✓ Deregistered from: {frontend_url}")
                except Exception as e:
                    print(f"   ✗ Could not deregister from {frontend_url}: {e}")
    
    print("\nShutting down Orange3 Web Backend...")
    await close_db()
    print("Database connections closed.")


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
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Router
# ============================================================================

api_v1 = APIRouter(prefix="/api/v1")


# ============================================================================
# Health
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    availability = get_availability()
    config = get_config()
    storage_type = config.storage.type  # 'sqlite', 'mysql', 'postgresql', 'oracle', 'filesystem', 'local'
    
    # Database storage types
    db_storage_types = {'sqlite', 'mysql', 'postgresql', 'oracle', 'database'}
    
    # Storage path depends on storage type
    if storage_type in db_storage_types:
        storage_path = config.database.url or "sqlite:///./orange3.db"
    else:
        storage_path = str(get_upload_dir())
    
    return {
        "status": "healthy",
        "service": "orange3-web",
        "version": SERVER_VERSION,
        "orange3": availability.get("orange3", False),
        "database": "sqlite",
        "storage_type": storage_type,
        "storage_path": storage_path,
        "max_file_size_mb": config.storage.max_db_file_size // (1024 * 1024),
        "locks": "asyncio",
        "server_start_time": SERVER_START_TIME
    }


@app.get("/internal/metrics")
async def get_metrics():
    """Get OpenTelemetry metrics summary."""
    if OTEL_AVAILABLE and get_telemetry:
        telemetry = get_telemetry()
        if telemetry:
            return telemetry.get_metrics_summary()
    return {
        "service": "orange3-web-backend",
        "version": SERVER_VERSION,
        "otel_available": OTEL_AVAILABLE,
        "message": "OpenTelemetry not initialized"
    }


@app.get("/internal/logs")
async def get_logs(limit: int = 100):
    """Get recent log entries."""
    if OTEL_AVAILABLE and get_telemetry:
        telemetry = get_telemetry()
        if telemetry:
            return telemetry.get_logs_response(limit)
    return {
        "service": "orange3-web-backend",
        "total": 0,
        "logs": []
    }


@app.get("/health/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)):
    """Readiness probe - checks database connection."""
    from sqlalchemy import text
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ready", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database not ready: {e}")


@app.get("/health/live")
async def liveness_check():
    """Liveness probe - checks if service is alive."""
    return {"status": "alive"}


# ============================================================================
# Data Loading Endpoints
# ============================================================================

class UrlLoadRequest(BaseModel):
    url: str


def get_mock_data_info(path: str):
    """Return mock data info for known datasets when Orange3 is not available."""
    path_lower = path.lower()
    
    if "iris" in path_lower:
        return {
            "name": "Iris",
            "description": "Fisher's Iris dataset with measurements of iris flowers.",
            "instances": 150,
            "features": 4,
            "missingValues": False,
            "classType": "Classification",
            "classValues": 3,
            "metaAttributes": 0,
            "columns": [
                {"name": "sepal length", "type": "numeric", "role": "feature", "values": ""},
                {"name": "sepal width", "type": "numeric", "role": "feature", "values": ""},
                {"name": "petal length", "type": "numeric", "role": "feature", "values": ""},
                {"name": "petal width", "type": "numeric", "role": "feature", "values": ""},
                {"name": "iris", "type": "categorical", "role": "target", "values": "Iris-setosa, Iris-versicolor, Iris-virginica"}
            ]
        }
    elif "titanic" in path_lower:
        return {
            "name": "Titanic dataset",
            "description": "Passenger survival data from the Titanic disaster.",
            "instances": 1309,
            "features": 10,
            "missingValues": True,
            "classType": "Classification",
            "classValues": 2,
            "metaAttributes": 0,
            "columns": [
                {"name": "pclass", "type": "categorical", "role": "feature", "values": "first, second, third"},
                {"name": "sex", "type": "categorical", "role": "feature", "values": "female, male"},
                {"name": "age", "type": "numeric", "role": "feature", "values": ""},
                {"name": "sibsp", "type": "numeric", "role": "feature", "values": ""},
                {"name": "parch", "type": "numeric", "role": "feature", "values": ""},
                {"name": "fare", "type": "numeric", "role": "feature", "values": ""},
                {"name": "embarked", "type": "categorical", "role": "feature", "values": "C, Q, S"},
                {"name": "survived", "type": "categorical", "role": "target", "values": "no, yes"}
            ]
        }
    elif "housing" in path_lower:
        return {
            "name": "Housing",
            "description": "Boston housing dataset with median home values.",
            "instances": 506,
            "features": 13,
            "missingValues": False,
            "classType": "Regression",
            "classValues": None,
            "metaAttributes": 0,
            "columns": [
                {"name": "CRIM", "type": "numeric", "role": "feature", "values": ""},
                {"name": "ZN", "type": "numeric", "role": "feature", "values": ""},
                {"name": "INDUS", "type": "numeric", "role": "feature", "values": ""},
                {"name": "CHAS", "type": "categorical", "role": "feature", "values": "0, 1"},
                {"name": "NOX", "type": "numeric", "role": "feature", "values": ""},
                {"name": "RM", "type": "numeric", "role": "feature", "values": ""},
                {"name": "AGE", "type": "numeric", "role": "feature", "values": ""},
                {"name": "DIS", "type": "numeric", "role": "feature", "values": ""},
                {"name": "RAD", "type": "numeric", "role": "feature", "values": ""},
                {"name": "TAX", "type": "numeric", "role": "feature", "values": ""},
                {"name": "PTRATIO", "type": "numeric", "role": "feature", "values": ""},
                {"name": "B", "type": "numeric", "role": "feature", "values": ""},
                {"name": "LSTAT", "type": "numeric", "role": "feature", "values": ""},
                {"name": "MEDV", "type": "numeric", "role": "target", "values": ""}
            ]
        }
    elif "zoo" in path_lower:
        return {
            "name": "Zoo",
            "description": "Zoo animal classification dataset.",
            "instances": 101,
            "features": 16,
            "missingValues": False,
            "classType": "Classification",
            "classValues": 7,
            "metaAttributes": 1,
            "columns": [
                {"name": "hair", "type": "categorical", "role": "feature", "values": "0, 1"},
                {"name": "feathers", "type": "categorical", "role": "feature", "values": "0, 1"},
                {"name": "eggs", "type": "categorical", "role": "feature", "values": "0, 1"},
                {"name": "milk", "type": "categorical", "role": "feature", "values": "0, 1"},
                {"name": "airborne", "type": "categorical", "role": "feature", "values": "0, 1"},
                {"name": "aquatic", "type": "categorical", "role": "feature", "values": "0, 1"},
                {"name": "predator", "type": "categorical", "role": "feature", "values": "0, 1"},
                {"name": "toothed", "type": "categorical", "role": "feature", "values": "0, 1"},
                {"name": "backbone", "type": "categorical", "role": "feature", "values": "0, 1"},
                {"name": "breathes", "type": "categorical", "role": "feature", "values": "0, 1"},
                {"name": "venomous", "type": "categorical", "role": "feature", "values": "0, 1"},
                {"name": "fins", "type": "categorical", "role": "feature", "values": "0, 1"},
                {"name": "legs", "type": "numeric", "role": "feature", "values": ""},
                {"name": "tail", "type": "categorical", "role": "feature", "values": "0, 1"},
                {"name": "domestic", "type": "categorical", "role": "feature", "values": "0, 1"},
                {"name": "catsize", "type": "categorical", "role": "feature", "values": "0, 1"},
                {"name": "type", "type": "categorical", "role": "target", "values": "mammal, bird, reptile, fish, amphibian, insect, invertebrate"},
                {"name": "name", "type": "categorical", "role": "meta", "values": ""}
            ]
        }
    
    # Generic fallback for unknown datasets
    filename = path.split("/")[-1]
    return {
        "name": filename,
        "description": f"Dataset loaded from {filename}",
        "instances": 100,
        "features": 5,
        "missingValues": False,
        "classType": "Classification",
        "classValues": 2,
        "metaAttributes": 0,
        "columns": [
            {"name": "feature1", "type": "numeric", "role": "feature", "values": ""},
            {"name": "feature2", "type": "numeric", "role": "feature", "values": ""},
            {"name": "feature3", "type": "numeric", "role": "feature", "values": ""},
            {"name": "feature4", "type": "numeric", "role": "feature", "values": ""},
            {"name": "feature5", "type": "numeric", "role": "feature", "values": ""},
            {"name": "class", "type": "categorical", "role": "target", "values": "Class A, Class B"}
        ]
    }


@api_v1.get("/data/load", tags=["Data"])
async def load_data_from_path(path: str):
    """Load data from a local file path."""
    # If Orange3 not available, return mock data
    if not ORANGE_AVAILABLE:
        return get_mock_data_info(path)
    
    # Check if it's an uploaded file
    if path.startswith("uploads/"):
        upload_dir = get_upload_dir()
        full_path = upload_dir / path.replace("uploads/", "")
        if full_path.exists():
            path = str(full_path)
    elif path.startswith("datasets/"):
        # Built-in Orange3 datasets - extract just the name without extension
        # e.g., "datasets/iris.tab" -> "iris"
        dataset_name = path.replace("datasets/", "").split(".")[0]
        path = dataset_name
    
    try:
        from Orange.data import Table
        
        # Try to load the data
        data = Table(path)
        
        # Get column info
        columns = []
        
        # Features
        for var in data.domain.attributes:
            columns.append({
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical",
                "role": "feature",
                "values": ", ".join(var.values) if hasattr(var, 'values') and var.values else ""
            })
        
        # Target
        if data.domain.class_var:
            var = data.domain.class_var
            columns.append({
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical",
                "role": "target",
                "values": ", ".join(var.values) if hasattr(var, 'values') and var.values else ""
            })
        
        # Meta
        for var in data.domain.metas:
            columns.append({
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical",
                "role": "meta",
                "values": ""
            })
        
        return {
            "name": data.name or path.split("/")[-1],
            "description": "",
            "instances": len(data),
            "features": len(data.domain.attributes),
            "missingValues": data.has_missing(),
            "classType": "Classification" if data.domain.class_var and not data.domain.class_var.is_continuous else "Regression" if data.domain.class_var else "None",
            "classValues": len(data.domain.class_var.values) if data.domain.class_var and hasattr(data.domain.class_var, 'values') else None,
            "metaAttributes": len(data.domain.metas),
            "columns": columns
        }
    except Exception as e:
        # Fallback to mock data on error
        return get_mock_data_info(path)


@api_v1.post("/data/load-url", tags=["Data"])
async def load_data_from_url(request: UrlLoadRequest):
    """Load data from a URL."""
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Orange3 not available")
    
    try:
        from Orange.data import Table
        import tempfile
        import urllib.request
        import os
        
        url = request.url
        
        # Download to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            tmp_path = tmp.name
        
        try:
            # Try to load the data
            data = Table(tmp_path)
            
            # Get column info
            columns = []
            
            # Features
            for var in data.domain.attributes:
                columns.append({
                    "name": var.name,
                    "type": "numeric" if var.is_continuous else "categorical",
                    "role": "feature",
                    "values": ", ".join(var.values) if hasattr(var, 'values') and var.values else ""
                })
            
            # Target
            if data.domain.class_var:
                var = data.domain.class_var
                columns.append({
                    "name": var.name,
                    "type": "numeric" if var.is_continuous else "categorical",
                    "role": "target",
                    "values": ", ".join(var.values) if hasattr(var, 'values') and var.values else ""
                })
            
            # Meta
            for var in data.domain.metas:
                columns.append({
                    "name": var.name,
                    "type": "numeric" if var.is_continuous else "categorical",
                    "role": "meta",
                    "values": ""
                })
            
            return {
                "name": url.split("/")[-1],
                "description": f"Loaded from {url}",
                "instances": len(data),
                "features": len(data.domain.attributes),
                "missingValues": data.has_missing(),
                "classType": "Classification" if data.domain.class_var and not data.domain.class_var.is_continuous else "Regression" if data.domain.class_var else "None",
                "classValues": len(data.domain.class_var.values) if data.domain.class_var and hasattr(data.domain.class_var, 'values') else None,
                "metaAttributes": len(data.domain.metas),
                "columns": columns
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
async def legacy_widgets():
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
async def websocket_endpoint(websocket: WebSocket, workflow_id: str):
    """WebSocket for real-time updates."""
    await workflow_ws_endpoint(websocket, workflow_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)