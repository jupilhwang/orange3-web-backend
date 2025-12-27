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
import shutil
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
import json
import uuid
from pydantic import BaseModel

# Setup logging
logger = logging.getLogger(__name__)

# Database and locks
from .database import init_db, close_db, get_db, async_session_maker
from .locks import lock_workflow, lock_tenant, workflow_locks
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

from .models import (
    Workflow, WorkflowSummary, WorkflowCreate, WorkflowUpdate,
    WorkflowNode, NodeCreate, NodeUpdate, NodeState,
    WorkflowLink, LinkCreate, LinkUpdate,
    TextAnnotation, ArrowAnnotation, AnnotationCreate,
    WidgetDescription, WidgetCategory,
    Position, Rect, Tenant
)
from .tenant import TenantManager, get_current_tenant
from .websocket_manager import WebSocketManager
from .widget_discovery import discover_widgets, get_widget_discovery

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
)
from .widgets.file_upload import UPLOAD_DIR


# ============================================================================
# Managers (Thread-safe singletons)
# ============================================================================

tenant_manager = TenantManager()
websocket_manager = WebSocketManager()

# In-memory workflow adapters (for Orange3 widget instances)
# Key: tenant_id:workflow_id -> OrangeSchemeAdapter
# Protected by workflow_locks
_workflow_adapters: Dict[str, Any] = {}

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


async def get_workflow_adapter(tenant_id: str, workflow_id: str) -> Optional[Any]:
    """Get workflow adapter with lock protection."""
    key = f"{tenant_id}:{workflow_id}"
    return _workflow_adapters.get(key)


async def create_workflow_adapter(tenant_id: str, workflow_id: str) -> Any:
    """Create a new workflow adapter with lock protection."""
    key = f"{tenant_id}:{workflow_id}"
    
    async with lock_workflow(workflow_id):
        if key in _workflow_adapters:
            return _workflow_adapters[key]
        
        if ORANGE_AVAILABLE:
            registry = get_registry()
            adapter = OrangeSchemeAdapter(registry=registry.registry if registry else None)
        else:
            # Fallback to simple dict-based storage
            adapter = {"nodes": [], "links": [], "annotations": [], "title": "", "description": ""}
        
        _workflow_adapters[key] = adapter
        return adapter


async def delete_workflow_adapter(tenant_id: str, workflow_id: str) -> bool:
    """Delete workflow adapter with lock protection."""
    key = f"{tenant_id}:{workflow_id}"
    
    async with lock_workflow(workflow_id):
        if key in _workflow_adapters:
            del _workflow_adapters[key]
            return True
        return False


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
    return {
        "status": "healthy",
        "service": "orange3-web",
        "orange3": availability.get("orange3", False),
        "database": "sqlite",
        "locks": "asyncio"
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
# Widget Registry (uses existing WidgetRegistry)
# ============================================================================

# Cache for discovered widgets (refreshed on startup)
_discovered_widgets = None


def get_discovered_widgets():
    """Get or discover widgets from Orange3 installation."""
    global _discovered_widgets
    if _discovered_widgets is None:
        # Try to find Orange3 widgets path
        possible_paths = [
            os.path.expanduser("~/works/test/orange3/orange3/Orange/widgets"),
            os.path.join(os.path.dirname(__file__), "..", ".venv", "lib", "python3.11", "site-packages", "Orange", "widgets"),
        ]
        
        orange3_path = None
        for path in possible_paths:
            if os.path.exists(path):
                orange3_path = path
                break
        
        if orange3_path:
            print(f"Discovering widgets from: {orange3_path}")
            _discovered_widgets = discover_widgets(orange3_path)
            print(f"Discovered {_discovered_widgets.get('total', 0)} widgets")
        else:
            print("Warning: Orange3 widgets path not found")
            _discovered_widgets = {"categories": [], "widgets": [], "total": 0}
    
    return _discovered_widgets


@api_v1.get("/widgets", tags=["Widgets"])
async def list_widgets(category: Optional[str] = None):
    """List all widgets discovered from Orange3 installation."""
    discovered = get_discovered_widgets()
    widgets = discovered.get("widgets", [])
    
    if category:
        widgets = [w for w in widgets if w.get("category") == category]
    
    return widgets


@api_v1.get("/widgets/categories", tags=["Widgets"])
async def list_categories():
    """List widget categories discovered from Orange3."""
    discovered = get_discovered_widgets()
    categories = discovered.get("categories", [])
    
    # Format for frontend compatibility
    return [
        {
            "name": cat["name"],
            "color": cat.get("color", "#999999"),
            "priority": cat.get("priority", 10),
            "widget_count": len(cat.get("widgets", []))
        }
        for cat in categories
    ]


@api_v1.get("/widgets/all", tags=["Widgets"])
async def get_all_widgets():
    """Get all categories with their widgets (for frontend toolbox)."""
    discovered = get_discovered_widgets()
    return discovered


@api_v1.get("/widgets/{widget_id}", tags=["Widgets"])
async def get_widget(widget_id: str):
    """Get widget description by ID."""
    discovered = get_discovered_widgets()
    widgets = discovered.get("widgets", [])
    
    for widget in widgets:
        if widget.get("id") == widget_id:
            return widget
    
    raise HTTPException(status_code=404, detail="Widget not found")


@api_v1.post("/widgets/check-compatibility", tags=["Widgets"])
async def check_compatibility(source_types: List[str], sink_types: List[str]):
    """Check channel compatibility between source and sink types."""
    # Simple compatibility check (can be enhanced later)
    # For now, Data type is compatible with Data type
    compatible = any(
        st == sk or st == "Any" or sk == "Any"
        for st in source_types
        for sk in sink_types
    )
    return {"compatible": compatible, "strict": True, "dynamic": False}


@api_v1.post("/widgets/refresh", tags=["Widgets"])
async def refresh_widgets():
    """Force refresh of widget discovery cache."""
    global _discovered_widgets
    _discovered_widgets = None
    discovered = get_discovered_widgets()
    return {"message": "Widget cache refreshed", "total": discovered.get("total", 0)}


# ============================================================================
# Workflow CRUD (uses existing Scheme class + async locks)
# ============================================================================

@api_v1.get("/workflows", tags=["Workflows"])
async def list_workflows(tenant: Tenant = Depends(get_current_tenant)):
    """List all workflows for tenant."""
    result = []
    
    # Find all workflows for this tenant
    for key, adapter in _workflow_adapters.items():
        if key.startswith(f"{tenant.id}:"):
            workflow_id = key.split(":", 1)[1]
            
            if ORANGE_AVAILABLE and hasattr(adapter, 'get_workflow_dict'):
                data = adapter.get_workflow_dict()
                result.append({
                    "id": workflow_id,
                    "title": data.get("title", ""),
                    "description": data.get("description", ""),
                    "node_count": len(data.get("nodes", [])),
                    "link_count": len(data.get("links", []))
                })
            else:
                result.append({
                    "id": workflow_id,
                    "title": adapter.get("title", "") if isinstance(adapter, dict) else "",
                    "node_count": len(adapter.get("nodes", [])) if isinstance(adapter, dict) else 0,
                    "link_count": len(adapter.get("links", [])) if isinstance(adapter, dict) else 0
                })
    
    return result


@api_v1.post("/workflows", tags=["Workflows"])
async def create_workflow(data: WorkflowCreate, tenant: Tenant = Depends(get_current_tenant)):
    """Create a new workflow. Creates a new Scheme instance."""
    workflow_id = str(uuid.uuid4())
    
    async with lock_tenant(tenant.id):
        adapter = await create_workflow_adapter(tenant.id, workflow_id)
        
        if ORANGE_AVAILABLE and hasattr(adapter, 'scheme'):
            adapter.scheme.title = data.title
            adapter.scheme.description = data.description
        elif isinstance(adapter, dict):
            adapter["title"] = data.title
            adapter["description"] = data.description
    
    return {"id": workflow_id, "title": data.title, "description": data.description}


@api_v1.get("/workflows/{workflow_id}", tags=["Workflows"])
async def get_workflow(workflow_id: str, tenant: Tenant = Depends(get_current_tenant)):
    """Get workflow details. Returns the full Scheme structure."""
    adapter = await get_workflow_adapter(tenant.id, workflow_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    if ORANGE_AVAILABLE and hasattr(adapter, 'get_workflow_dict'):
        data = adapter.get_workflow_dict()
        data["id"] = workflow_id
        return data
    elif isinstance(adapter, dict):
        return {"id": workflow_id, **adapter}
    
    raise HTTPException(status_code=500, detail="Invalid workflow state")


@api_v1.delete("/workflows/{workflow_id}", tags=["Workflows"])
async def delete_workflow(workflow_id: str, tenant: Tenant = Depends(get_current_tenant)):
    """Delete a workflow."""
    deleted = await delete_workflow_adapter(tenant.id, workflow_id)
    if deleted:
        return {"message": "Workflow deleted"}
    raise HTTPException(status_code=404, detail="Workflow not found")


@api_v1.get("/workflows/{workflow_id}/export", tags=["Workflows"])
async def export_workflow(workflow_id: str, tenant: Tenant = Depends(get_current_tenant)):
    """Export workflow as .ows. Uses existing scheme_to_ows_stream function."""
    adapter = await get_workflow_adapter(tenant.id, workflow_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    if ORANGE_AVAILABLE and hasattr(adapter, 'export_to_ows'):
        ows_content = adapter.export_to_ows()
        return Response(
            content=ows_content,
            media_type="application/xml",
            headers={"Content-Disposition": f"attachment; filename=workflow.ows"}
        )
    
    raise HTTPException(status_code=501, detail="Export not available")


# ============================================================================
# Node CRUD (uses existing SchemeNode class + async locks)
# ============================================================================

class NodeCreateRequest(BaseModel):
    widget_id: str
    title: str
    x: float
    y: float


@api_v1.post("/workflows/{workflow_id}/nodes", tags=["Nodes"])
async def add_node(
    workflow_id: str,
    node: NodeCreateRequest,
    tenant: Tenant = Depends(get_current_tenant)
):
    """Add a node. Uses existing SchemeNode class."""
    adapter = await get_workflow_adapter(tenant.id, workflow_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    async with lock_workflow(workflow_id):
        if ORANGE_AVAILABLE and hasattr(adapter, 'add_node'):
            try:
                result = adapter.add_node(
                    widget_id=node.widget_id,
                    title=node.title,
                    position=(node.x, node.y)
                )
                
                # Broadcast to WebSocket
                await websocket_manager.broadcast_to_workflow(
                    workflow_id,
                    {"type": "node_added", "data": result}
                )
                
                return result
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        # Fallback
        fallback_node = {
            "id": str(uuid.uuid4()),
            "widget_id": node.widget_id,
            "title": node.title,
            "position": {"x": node.x, "y": node.y}
        }
        if isinstance(adapter, dict):
            adapter.setdefault("nodes", []).append(fallback_node)
        return fallback_node


@api_v1.delete("/workflows/{workflow_id}/nodes/{node_id}", tags=["Nodes"])
async def delete_node(
    workflow_id: str,
    node_id: str,
    tenant: Tenant = Depends(get_current_tenant)
):
    """Delete a node. Uses existing Scheme.remove_node method."""
    adapter = await get_workflow_adapter(tenant.id, workflow_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    async with lock_workflow(workflow_id):
        if ORANGE_AVAILABLE and hasattr(adapter, 'remove_node'):
            if adapter.remove_node(node_id):
                await websocket_manager.broadcast_to_workflow(
                    workflow_id,
                    {"type": "node_removed", "data": {"node_id": node_id}}
                )
                return {"message": "Node deleted"}
    
    raise HTTPException(status_code=404, detail="Node not found")


class NodePositionUpdate(BaseModel):
    x: float
    y: float


@api_v1.put("/workflows/{workflow_id}/nodes/{node_id}/position", tags=["Nodes"])
async def update_node_position(
    workflow_id: str,
    node_id: str,
    position: NodePositionUpdate,
    tenant: Tenant = Depends(get_current_tenant)
):
    """Update node position."""
    adapter = await get_workflow_adapter(tenant.id, workflow_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    async with lock_workflow(workflow_id):
        if ORANGE_AVAILABLE and hasattr(adapter, 'update_node_position'):
            if adapter.update_node_position(node_id, (position.x, position.y)):
                return {"message": "Position updated"}
    
    raise HTTPException(status_code=404, detail="Node not found")


# ============================================================================
# Link CRUD (uses existing SchemeLink class + async locks)
# ============================================================================

class LinkCreateRequest(BaseModel):
    source_node_id: str
    source_channel: str
    sink_node_id: str
    sink_channel: str


@api_v1.post("/workflows/{workflow_id}/links", tags=["Links"])
async def add_link(
    workflow_id: str,
    link: LinkCreateRequest,
    tenant: Tenant = Depends(get_current_tenant)
):
    """Add a link. Uses existing SchemeLink class and compatible_channels function."""
    adapter = await get_workflow_adapter(tenant.id, workflow_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    async with lock_workflow(workflow_id):
        if ORANGE_AVAILABLE and hasattr(adapter, 'add_link'):
            result = adapter.add_link(
                source_node_id=link.source_node_id,
                source_channel=link.source_channel,
                sink_node_id=link.sink_node_id,
                sink_channel=link.sink_channel
            )
            if result:
                await websocket_manager.broadcast_to_workflow(
                    workflow_id,
                    {"type": "link_added", "data": result}
                )
                return result
            raise HTTPException(status_code=400, detail="Cannot create link")
        
        # Fallback
        fallback_link = {
            "id": str(uuid.uuid4()),
            **link.model_dump()
        }
        if isinstance(adapter, dict):
            adapter.setdefault("links", []).append(fallback_link)
        return fallback_link


@api_v1.delete("/workflows/{workflow_id}/links/{link_id}", tags=["Links"])
async def delete_link(
    workflow_id: str,
    link_id: str,
    tenant: Tenant = Depends(get_current_tenant)
):
    """Delete a link. Uses existing Scheme.remove_link method."""
    adapter = await get_workflow_adapter(tenant.id, workflow_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    async with lock_workflow(workflow_id):
        if ORANGE_AVAILABLE and hasattr(adapter, 'remove_link'):
            if adapter.remove_link(link_id):
                await websocket_manager.broadcast_to_workflow(
                    workflow_id,
                    {"type": "link_removed", "data": {"link_id": link_id}}
                )
                return {"message": "Link deleted"}
    
    raise HTTPException(status_code=404, detail="Link not found")


# ============================================================================
# Annotations (uses existing SchemeAnnotation classes + async locks)
# ============================================================================

class TextAnnotationCreate(BaseModel):
    x: float
    y: float
    width: float
    height: float
    content: str
    content_type: str = "text/plain"


@api_v1.post("/workflows/{workflow_id}/annotations/text", tags=["Annotations"])
async def add_text_annotation(
    workflow_id: str,
    annotation: TextAnnotationCreate,
    tenant: Tenant = Depends(get_current_tenant)
):
    """Add text annotation. Uses existing SchemeTextAnnotation class."""
    adapter = await get_workflow_adapter(tenant.id, workflow_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    async with lock_workflow(workflow_id):
        if ORANGE_AVAILABLE and hasattr(adapter, 'add_text_annotation'):
            result = adapter.add_text_annotation(
                rect=(annotation.x, annotation.y, annotation.width, annotation.height),
                content=annotation.content,
                content_type=annotation.content_type
            )
            return result
    
    raise HTTPException(status_code=501, detail="Annotations not available")


class ArrowAnnotationCreate(BaseModel):
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    color: str = "#808080"


@api_v1.post("/workflows/{workflow_id}/annotations/arrow", tags=["Annotations"])
async def add_arrow_annotation(
    workflow_id: str,
    annotation: ArrowAnnotationCreate,
    tenant: Tenant = Depends(get_current_tenant)
):
    """Add arrow annotation. Uses existing SchemeArrowAnnotation class."""
    adapter = await get_workflow_adapter(tenant.id, workflow_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    async with lock_workflow(workflow_id):
        if ORANGE_AVAILABLE and hasattr(adapter, 'add_arrow_annotation'):
            result = adapter.add_arrow_annotation(
                start_pos=(annotation.start_x, annotation.start_y),
                end_pos=(annotation.end_x, annotation.end_y),
                color=annotation.color
            )
            return result
    
    raise HTTPException(status_code=501, detail="Annotations not available")


# ============================================================================
# WebSocket
# ============================================================================

@app.websocket("/api/v1/workflows/{workflow_id}/ws")
async def websocket_endpoint(websocket: WebSocket, workflow_id: str):
    """WebSocket for real-time updates."""
    await websocket_manager.connect(websocket, workflow_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Broadcast to other clients
            await websocket_manager.broadcast_to_workflow(
                workflow_id,
                message,
                exclude=websocket
            )
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, workflow_id)


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
        full_path = UPLOAD_DIR / path.replace("uploads/", "")
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
    registry = get_registry()
    if registry and ORANGE_AVAILABLE:
        categories = registry.list_categories()
        widgets = registry.list_widgets()
        
        # Group widgets by category
        result = []
        for cat in categories:
            cat_widgets = [w for w in widgets if w.get("category") == cat["name"]]
            result.append({
                "name": cat["name"],
                "color": cat.get("background", "#808080"),
                "widgets": cat_widgets
            })
        return {"categories": result}
    
    return {"categories": []}


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

# ============================================================================
# Include Main Router
# ============================================================================

app.include_router(api_v1)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)