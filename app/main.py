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


# Upload directory configuration
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@api_v1.post("/data/upload", tags=["Data"])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a data file from local PC.
    Supports: CSV, TSV, TAB, XLSX, PKL files
    """
    # Validate file extension
    allowed_extensions = {'.csv', '.tsv', '.tab', '.xlsx', '.xls', '.pkl', '.pickle', '.txt'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not supported. Allowed types: {', '.join(sorted(allowed_extensions))}"
        )
    
    # Generate unique filename to avoid conflicts
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = UPLOAD_DIR / unique_filename
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Try to load and parse the data with Orange3
        if ORANGE_AVAILABLE:
            try:
                from Orange.data import Table
                data = Table(str(file_path))
                
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
                    "success": True,
                    "filename": file.filename,
                    "savedPath": str(file_path),
                    "relativePath": f"uploads/{unique_filename}",
                    "name": data.name or file.filename,
                    "description": f"Uploaded file: {file.filename}",
                    "instances": len(data),
                    "features": len(data.domain.attributes),
                    "missingValues": data.has_missing(),
                    "classType": "Classification" if data.domain.class_var and not data.domain.class_var.is_continuous else "Regression" if data.domain.class_var else "None",
                    "classValues": len(data.domain.class_var.values) if data.domain.class_var and hasattr(data.domain.class_var, 'values') else None,
                    "metaAttributes": len(data.domain.metas),
                    "columns": columns
                }
            except Exception as e:
                print(f"Orange3 parsing failed: {e}")
                # Continue with basic file info
        
        # Fallback: Return basic file info without parsing
        return {
            "success": True,
            "filename": file.filename,
            "savedPath": str(file_path),
            "relativePath": f"uploads/{unique_filename}",
            "name": file.filename,
            "description": f"Uploaded file: {file.filename}",
            "instances": 0,
            "features": 0,
            "missingValues": False,
            "classType": "Unknown",
            "classValues": None,
            "metaAttributes": 0,
            "columns": [],
            "parseError": "Orange3 not available or parsing failed"
        }
        
    except Exception as e:
        # Clean up on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@api_v1.get("/data/uploaded", tags=["Data"])
async def list_uploaded_files():
    """List all uploaded files."""
    files = []
    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.iterdir():
            if f.is_file():
                files.append({
                    "filename": f.name,
                    "path": f"uploads/{f.name}",
                    "size": f.stat().st_size
                })
    return {"files": files}


@api_v1.delete("/data/uploaded/{filename}", tags=["Data"])
async def delete_uploaded_file(filename: str):
    """Delete an uploaded file."""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        file_path.unlink()
        return {"message": f"File {filename} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


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


# ============================================================================
# Visualization Endpoints
# ============================================================================

class ScatterPlotRequest(BaseModel):
    data_path: str
    axis_x: Optional[str] = None
    axis_y: Optional[str] = None
    color_attr: Optional[str] = None
    size_attr: Optional[str] = None
    shape_attr: Optional[str] = None
    jittering: float = 0
    subset_indices: Optional[List[int]] = None


@api_v1.post("/visualize/scatter-plot", tags=["Visualize"])
async def get_scatter_plot_data(request: ScatterPlotRequest):
    """Generate scatter plot data for visualization."""
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Orange3 not available")
    
    try:
        from Orange.data import Table
        import numpy as np
        
        # Load data
        data = Table(request.data_path)
        
        # Get variables
        variables = []
        numeric_vars = []
        categorical_vars = []
        
        for var in data.domain.attributes:
            var_info = {
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical"
            }
            if hasattr(var, 'values') and var.values:
                var_info["values"] = list(var.values)
            variables.append(var_info)
            
            if var.is_continuous:
                numeric_vars.append(var)
            else:
                categorical_vars.append(var)
        
        # Add class variable
        if data.domain.class_var:
            var = data.domain.class_var
            var_info = {
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical"
            }
            if hasattr(var, 'values') and var.values:
                var_info["values"] = list(var.values)
            variables.append(var_info)
            
            if var.is_continuous:
                numeric_vars.append(var)
            else:
                categorical_vars.append(var)
        
        # Determine axis variables
        axis_x_name = request.axis_x or (numeric_vars[0].name if len(numeric_vars) > 0 else None)
        axis_y_name = request.axis_y or (numeric_vars[1].name if len(numeric_vars) > 1 else numeric_vars[0].name if len(numeric_vars) > 0 else None)
        
        if not axis_x_name or not axis_y_name:
            raise HTTPException(status_code=400, detail="Not enough numeric variables for scatter plot")
        
        # Find variable objects
        axis_x_var = None
        axis_y_var = None
        color_var = None
        
        for var in list(data.domain.attributes) + ([data.domain.class_var] if data.domain.class_var else []):
            if var.name == axis_x_name:
                axis_x_var = var
            if var.name == axis_y_name:
                axis_y_var = var
            if request.color_attr and var.name == request.color_attr:
                color_var = var
        
        if not axis_x_var or not axis_y_var:
            raise HTTPException(status_code=400, detail="Specified axis variables not found")
        
        # Get data values
        x_col = data.get_column(axis_x_var)
        y_col = data.get_column(axis_y_var)
        
        # Handle missing values
        valid_mask = ~(np.isnan(x_col) | np.isnan(y_col))
        
        # Build points
        points = []
        
        # Color mapping
        colors = ['#5dade2', '#e74c3c', '#82e0aa', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#3498db']
        class_names = []
        class_colors = {}
        
        if color_var and hasattr(color_var, 'values') and color_var.values:
            class_names = list(color_var.values)
            for i, cls in enumerate(class_names):
                class_colors[cls] = colors[i % len(colors)]
        
        color_col = data.get_column(color_var) if color_var else None
        
        for i in range(len(data)):
            if not valid_mask[i]:
                continue
            
            x_val = float(x_col[i])
            y_val = float(y_col[i])
            
            # Apply jittering
            if request.jittering > 0:
                jitter_amount = request.jittering / 100
                x_range = float(np.nanmax(x_col) - np.nanmin(x_col))
                y_range = float(np.nanmax(y_col) - np.nanmin(y_col))
                x_val += (np.random.random() - 0.5) * x_range * jitter_amount
                y_val += (np.random.random() - 0.5) * y_range * jitter_amount
            
            point = {
                "x": x_val,
                "y": y_val,
                "index": i
            }
            
            # Add class info
            if color_var and color_col is not None:
                class_idx = int(color_col[i]) if not np.isnan(color_col[i]) else 0
                if class_names and class_idx < len(class_names):
                    cls_name = class_names[class_idx]
                    point["class"] = cls_name
                    point["color"] = class_colors.get(cls_name, colors[0])
            
            points.append(point)
        
        # Calculate ranges
        valid_x = x_col[valid_mask]
        valid_y = y_col[valid_mask]
        
        x_min = float(np.nanmin(valid_x))
        x_max = float(np.nanmax(valid_x))
        y_min = float(np.nanmin(valid_y))
        y_max = float(np.nanmax(valid_y))
        
        # Add padding
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05
        
        return {
            "points": points,
            "xLabel": axis_x_name,
            "yLabel": axis_y_name,
            "xMin": x_min - x_padding,
            "xMax": x_max + x_padding,
            "yMin": y_min - y_padding,
            "yMax": y_max + y_padding,
            "classes": class_names if class_names else None,
            "classColors": list(class_colors.values()) if class_colors else None,
            "variables": variables,
            "totalPoints": len(points),
            "instanceCount": len(data)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


class ScatterPlotSelectionRequest(BaseModel):
    data_path: str
    selected_indices: List[int]


@api_v1.post("/visualize/scatter-plot/select", tags=["Visualize"])
async def select_scatter_plot_data(request: ScatterPlotSelectionRequest):
    """Return selected data subset from scatter plot."""
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Orange3 not available")
    
    try:
        from Orange.data import Table
        
        data = Table(request.data_path)
        
        if not request.selected_indices:
            return {"selected_count": 0, "data": None}
        
        # Filter to selected indices
        selected_data = data[request.selected_indices]
        
        return {
            "selected_count": len(selected_data),
            "instances": len(selected_data),
            "features": len(selected_data.domain.attributes)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Data Sampling Endpoint (uses Orange3's sampling logic)
# ============================================================================

class SampleDataRequest(BaseModel):
    data_path: str
    sampling_type: int = 0  # 0=FixedProportion, 1=FixedSize, 2=CrossValidation, 3=Bootstrap
    sample_percentage: int = 70
    sample_size: int = 1
    replacement: bool = False
    number_of_folds: int = 10
    selected_fold: int = 1
    use_seed: bool = True
    stratify: bool = False


@api_v1.post("/data/sample", tags=["Data"])
async def sample_data(request: SampleDataRequest):
    """
    Sample data using Orange3's sampling algorithms.
    
    Sampling Types:
    - 0: Fixed proportion (percentage)
    - 1: Fixed sample size (number of instances)
    - 2: Cross validation (k-fold)
    - 3: Bootstrap
    """
    if not ORANGE_AVAILABLE:
        # Fallback calculation without Orange3
        return calculate_sample_fallback(request)
    
    try:
        from Orange.data import Table
        import numpy as np
        import math
        
        # Constants
        RANDOM_SEED = 42
        FixedProportion, FixedSize, CrossValidation, Bootstrap = range(4)
        
        # Resolve data path
        data_path = request.data_path
        if data_path.startswith("uploads/"):
            data_path = str(UPLOAD_DIR / data_path.replace("uploads/", ""))
        elif data_path.startswith("datasets/"):
            # Built-in Orange3 datasets - extract just the name without extension
            # e.g., "datasets/iris.tab" -> "iris"
            dataset_name = data_path.replace("datasets/", "").split(".")[0]
            data_path = dataset_name
        
        # Load data
        data = Table(data_path)
        data_length = len(data)
        
        if data_length == 0:
            raise HTTPException(status_code=400, detail="Dataset is empty")
        
        # Determine random state
        rnd = RANDOM_SEED if request.use_seed else None
        
        # Perform sampling based on type
        sample_indices = None
        remaining_indices = None
        
        if request.sampling_type == FixedProportion:
            # Fixed proportion sampling
            size = int(math.ceil(request.sample_percentage / 100 * data_length))
            sample_indices, remaining_indices = sample_random_n(
                data, size, stratified=request.stratify, replace=False, random_state=rnd
            )
            
        elif request.sampling_type == FixedSize:
            # Fixed size sampling
            size = request.sample_size
            if not request.replacement and size > data_length:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Sample size ({size}) cannot be larger than data size ({data_length}) without replacement"
                )
            sample_indices, remaining_indices = sample_random_n(
                data, size, stratified=request.stratify, replace=request.replacement, random_state=rnd
            )
            
        elif request.sampling_type == CrossValidation:
            # Cross validation
            if data_length < request.number_of_folds:
                raise HTTPException(
                    status_code=400,
                    detail=f"Number of folds ({request.number_of_folds}) exceeds data size ({data_length})"
                )
            folds = sample_fold_indices(
                data, request.number_of_folds, stratified=request.stratify, random_state=rnd
            )
            # For cross validation: sample = training set, remaining = test set (selected fold)
            sample_indices, remaining_indices = folds[request.selected_fold - 1]
            
        elif request.sampling_type == Bootstrap:
            # Bootstrap sampling
            sample_indices, remaining_indices = sample_bootstrap(data_length, random_state=rnd)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown sampling type: {request.sampling_type}")
        
        # Calculate counts
        sample_count = len(sample_indices) if sample_indices is not None else 0
        remaining_count = len(remaining_indices) if remaining_indices is not None else 0
        
        return {
            "success": True,
            "sample_count": sample_count,
            "remaining_count": remaining_count,
            "total_count": data_length,
            "sampling_type": request.sampling_type,
            "sample_indices": sample_indices.tolist() if sample_indices is not None else [],
            "remaining_indices": remaining_indices.tolist() if remaining_indices is not None else []
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


def sample_random_n(data, n, stratified=False, replace=False, random_state=None):
    """Sample n instances from data."""
    import numpy as np
    import sklearn.model_selection as skl
    
    data_length = len(data)
    
    if replace:
        # Sampling with replacement
        rgen = np.random.RandomState(random_state)
        sample = rgen.randint(0, data_length, n)
        others = np.ones(data_length)
        others[sample] = 0
        remaining = np.nonzero(others)[0]
        return sample, remaining
    
    if n == 0:
        rgen = np.random.RandomState(random_state)
        shuffled = np.arange(data_length)
        rgen.shuffle(shuffled)
        return np.array([], dtype=int), shuffled
    
    if n >= data_length:
        rgen = np.random.RandomState(random_state)
        shuffled = np.arange(data_length)
        rgen.shuffle(shuffled)
        return shuffled, np.array([], dtype=int)
    
    if stratified and data.domain.has_discrete_class:
        try:
            test_size = max(len(data.domain.class_var.values), n)
            splitter = skl.StratifiedShuffleSplit(
                n_splits=1, test_size=test_size,
                train_size=data_length - test_size,
                random_state=random_state
            )
            splitter.get_n_splits(data.X, data.Y)
            ind = splitter.split(data.X, data.Y)
            remaining, sample = next(iter(ind))
            return sample, remaining
        except:
            pass  # Fall through to non-stratified
    
    # Non-stratified sampling
    splitter = skl.ShuffleSplit(n_splits=1, test_size=n, random_state=random_state)
    splitter.get_n_splits(data)
    ind = splitter.split(data)
    remaining, sample = next(iter(ind))
    return sample, remaining


def sample_fold_indices(data, folds, stratified=False, random_state=None):
    """Generate k-fold cross validation indices."""
    import sklearn.model_selection as skl
    
    if stratified and data.domain.has_discrete_class:
        splitter = skl.StratifiedKFold(folds, shuffle=True, random_state=random_state)
        splitter.get_n_splits(data.X, data.Y)
        ind = splitter.split(data.X, data.Y)
    else:
        splitter = skl.KFold(folds, shuffle=True, random_state=random_state)
        splitter.get_n_splits(data)
        ind = splitter.split(data)
    
    return tuple(ind)


def sample_bootstrap(size, random_state=None):
    """Bootstrap sampling indices."""
    import numpy as np
    
    rgen = np.random.RandomState(random_state)
    sample = rgen.randint(0, size, size)
    sample.sort()
    
    insample = np.ones((size,), dtype=bool)
    insample[sample] = False
    remaining = np.flatnonzero(insample)
    
    return sample, remaining


def calculate_sample_fallback(request: SampleDataRequest):
    """Fallback sampling calculation when Orange3 is not available."""
    # Estimate total instances (mock)
    total = 100
    
    if request.sampling_type == 0:  # FixedProportion
        sample_count = int(total * request.sample_percentage / 100)
    elif request.sampling_type == 1:  # FixedSize
        sample_count = min(request.sample_size, total) if not request.replacement else request.sample_size
    elif request.sampling_type == 2:  # CrossValidation
        fold_size = total // request.number_of_folds
        sample_count = total - fold_size
    else:  # Bootstrap
        sample_count = total
    
    remaining_count = total - sample_count if request.sampling_type != 3 else int(total * 0.368)
    
    return {
        "success": True,
        "sample_count": max(0, sample_count),
        "remaining_count": max(0, remaining_count),
        "total_count": total,
        "sampling_type": request.sampling_type,
        "sample_indices": [],
        "remaining_indices": [],
        "note": "Fallback calculation (Orange3 not available)"
    }


# =============================================================================
# Select Columns API
# =============================================================================

class SelectColumnsRequest(BaseModel):
    """Request model for select columns."""
    data_path: Optional[str] = None
    features: List[str] = []
    target: List[str] = []
    metas: List[str] = []
    ignored: List[str] = []


@api_v1.post("/data/select-columns", tags=["Data"])
async def select_columns(request: SelectColumnsRequest):
    """
    Select and reorder columns in a dataset.
    
    This endpoint allows you to:
    - Assign columns as features, target, metas, or ignored
    - Reorder columns within each category
    - Create a new domain with the specified column assignments
    
    Args:
        request: SelectColumnsRequest with column assignments
        
    Returns:
        Modified data information with new column assignments
    """
    if not ORANGE_AVAILABLE:
        # Fallback - just return the column assignments
        return {
            "success": True,
            "features": request.features,
            "target": request.target,
            "metas": request.metas,
            "ignored": request.ignored,
            "instances": 0,
            "variables": len(request.features) + len(request.target) + len(request.metas),
            "note": "Fallback response (Orange3 not available)"
        }
    
    try:
        from Orange.data import Table, Domain
        
        # Resolve data path
        data_path = request.data_path
        if not data_path:
            # No data path - return column assignments only
            return {
                "success": True,
                "features": request.features,
                "target": request.target,
                "metas": request.metas,
                "ignored": request.ignored,
                "instances": 0,
                "variables": len(request.features) + len(request.target) + len(request.metas)
            }
        
        if data_path.startswith("uploads/"):
            data_path = str(UPLOAD_DIR / data_path.replace("uploads/", ""))
        elif data_path.startswith("datasets/"):
            # Built-in Orange3 datasets
            dataset_name = data_path.replace("datasets/", "").split(".")[0]
            data_path = dataset_name
        
        # Load original data
        original_data = Table(data_path)
        
        # Build new domain
        all_vars = {}
        for var in original_data.domain.attributes:
            all_vars[var.name] = var
        if original_data.domain.class_var:
            all_vars[original_data.domain.class_var.name] = original_data.domain.class_var
        for var in original_data.domain.metas:
            all_vars[var.name] = var
        
        # Create new domain components
        new_features = [all_vars[name] for name in request.features if name in all_vars]
        new_target = all_vars.get(request.target[0]) if request.target else None
        new_metas = [all_vars[name] for name in request.metas if name in all_vars]
        
        # Create new domain
        new_domain = Domain(new_features, new_target, new_metas)
        
        # Transform data to new domain
        new_data = original_data.transform(new_domain)
        
        # Helper function to convert cell value
        import math
        def get_cell_value(var, val):
            if var.is_continuous:
                return float(val) if not math.isnan(val) else None
            else:
                # Discrete: use str_val to get actual category name
                return var.str_val(val)
        
        # Convert data to list of lists
        data_rows = []
        for row in new_data:
            row_values = []
            # Attributes
            for var in new_domain.attributes:
                val = row[var]
                row_values.append(get_cell_value(var, val))
            # Class variable
            if new_domain.class_var:
                val = row[new_domain.class_var]
                row_values.append(get_cell_value(new_domain.class_var, val))
            # Metas
            for var in new_domain.metas:
                val = row[var]
                if var.is_string:
                    row_values.append(str(val) if val else None)
                else:
                    row_values.append(get_cell_value(var, val))
            data_rows.append(row_values)
        
        # Build column info for response
        columns = []
        for var in new_domain.attributes:
            columns.append({
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical",
                "role": "feature"
            })
        if new_domain.class_var:
            columns.append({
                "name": new_domain.class_var.name,
                "type": "numeric" if new_domain.class_var.is_continuous else "categorical",
                "role": "target"
            })
        for var in new_domain.metas:
            columns.append({
                "name": var.name,
                "type": "string" if var.is_string else ("numeric" if var.is_continuous else "categorical"),
                "role": "meta"
            })
        
        return {
            "success": True,
            "features": [v.name for v in new_domain.attributes],
            "target": [new_domain.class_var.name] if new_domain.class_var else [],
            "metas": [v.name for v in new_domain.metas],
            "ignored": request.ignored,
            "instances": len(new_data),
            "variables": len(new_domain.attributes) + (1 if new_domain.class_var else 0) + len(new_domain.metas),
            "columns": columns,
            "data": data_rows
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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
# Datasets Endpoints (Online Repository)
# ============================================================================

# Datasets server URL (Orange3's official dataset repository)
DATASETS_INDEX_URL = "https://datasets.biolab.si/"
DATASETS_CACHE_DIR = Path(__file__).parent.parent / "datasets_cache"
DATASETS_CACHE_DIR.mkdir(exist_ok=True)

# In-memory cache for datasets list
_datasets_cache = None
_datasets_cache_time = None
DATASETS_CACHE_TTL = 3600  # 1 hour cache


async def fetch_datasets_list():
    """Fetch datasets list from Orange3 server with caching."""
    global _datasets_cache, _datasets_cache_time
    import time
    
    # Return cached if available and fresh
    if _datasets_cache and _datasets_cache_time:
        if time.time() - _datasets_cache_time < DATASETS_CACHE_TTL:
            return _datasets_cache
    
    try:
        from serverfiles import ServerFiles, LocalFiles
        
        # Try to fetch remote list
        client = ServerFiles(server=DATASETS_INDEX_URL)
        allinfo = client.allinfo()
        
        # Also get local cached files
        local = LocalFiles(str(DATASETS_CACHE_DIR))
        local_info = local.allinfo()
        
        datasets = []
        for file_path, info in allinfo.items():
            # file_path is a tuple like ('core', 'iris.tab')
            prefix = '/'.join(file_path[:-1]) if len(file_path) > 1 else ''
            filename = file_path[-1]
            
            # Skip non-data files
            if not filename.endswith(('.tab', '.csv', '.xlsx', '.pkl')):
                continue
            
            islocal = file_path in local_info
            
            datasets.append({
                "id": '/'.join(file_path),
                "file_path": list(file_path),
                "prefix": prefix,
                "filename": filename,
                "title": info.get('title', filename),
                "description": info.get('description', ''),
                "size": info.get('size', 0),
                "instances": info.get('instances'),
                "variables": info.get('variables'),
                "target": info.get('target'),  # 'categorical', 'numeric', or None
                "tags": info.get('tags', []),
                "source": info.get('source', ''),
                "year": info.get('year'),
                "references": info.get('references', []),
                "seealso": info.get('seealso', []),
                "language": info.get('language', 'English'),
                "domain": info.get('domain'),
                "islocal": islocal,
                "version": info.get('version', '')
            })
        
        # Sort by title
        datasets.sort(key=lambda x: x['title'].lower())
        
        _datasets_cache = datasets
        _datasets_cache_time = time.time()
        
        return datasets
        
    except Exception as e:
        print(f"Error fetching datasets: {e}")
        import traceback
        traceback.print_exc()
        
        # Return fallback minimal dataset list
        return get_fallback_datasets()


def get_fallback_datasets():
    """Fallback dataset list when server is unavailable."""
    return [
        {
            "id": "core/iris.tab",
            "file_path": ["core", "iris.tab"],
            "prefix": "core",
            "filename": "iris.tab",
            "title": "Iris",
            "description": "Fisher's Iris data with 150 instances and 4 features.",
            "size": 4625,
            "instances": 150,
            "variables": 5,
            "target": "categorical",
            "tags": [],
            "source": "",
            "year": None,
            "references": [],
            "seealso": [],
            "language": "English",
            "domain": None,
            "islocal": True,
            "version": ""
        },
        {
            "id": "core/titanic.tab",
            "file_path": ["core", "titanic.tab"],
            "prefix": "core",
            "filename": "titanic.tab",
            "title": "Titanic",
            "description": "Titanic survival data.",
            "size": 77400,
            "instances": 2201,
            "variables": 4,
            "target": "categorical",
            "tags": [],
            "source": "",
            "year": None,
            "references": [],
            "seealso": [],
            "language": "English",
            "domain": None,
            "islocal": True,
            "version": ""
        },
        {
            "id": "core/housing.tab",
            "file_path": ["core", "housing.tab"],
            "prefix": "core",
            "filename": "housing.tab",
            "title": "Housing",
            "description": "Boston housing dataset.",
            "size": 52500,
            "instances": 506,
            "variables": 14,
            "target": "numeric",
            "tags": ["economy"],
            "source": "",
            "year": None,
            "references": [],
            "seealso": [],
            "language": "English",
            "domain": None,
            "islocal": True,
            "version": ""
        }
    ]


@api_v1.get("/datasets", tags=["Datasets"])
async def list_datasets(
    language: Optional[str] = None,
    domain: Optional[str] = None,
    search: Optional[str] = None
):
    """
    List available datasets from the Orange3 online repository.
    
    Supports filtering by:
    - language: e.g., 'English', 'Slovenian'
    - domain: e.g., 'biology', 'economy'
    - search: text search in title
    """
    datasets = await fetch_datasets_list()
    
    # Apply filters
    filtered = datasets
    
    if language:
        filtered = [d for d in filtered if d.get('language') == language]
    
    if domain:
        if domain == "(General)":
            filtered = [d for d in filtered if d.get('domain') is None]
        elif domain != "(Show all)":
            filtered = [d for d in filtered if d.get('domain') == domain]
    
    if search and len(search) >= 2:
        search_lower = search.lower()
        filtered = [d for d in filtered if search_lower in d['title'].lower()]
    
    # Get available languages and domains for filter dropdowns
    all_languages = sorted(set(d.get('language', 'English') for d in datasets))
    all_domains = sorted(set(d.get('domain') for d in datasets if d.get('domain')))
    
    return {
        "datasets": filtered,
        "total": len(filtered),
        "languages": all_languages,
        "domains": all_domains
    }


@api_v1.get("/datasets/{dataset_id:path}/info", tags=["Datasets"])
async def get_dataset_info(dataset_id: str):
    """Get detailed information about a specific dataset."""
    datasets = await fetch_datasets_list()
    
    for d in datasets:
        if d['id'] == dataset_id:
            return d
    
    raise HTTPException(status_code=404, detail="Dataset not found")


@api_v1.post("/datasets/{dataset_id:path}/download", tags=["Datasets"])
async def download_dataset(dataset_id: str):
    """
    Download a dataset from the online repository to local cache.
    Returns the local file path after download.
    """
    datasets = await fetch_datasets_list()
    
    dataset = None
    for d in datasets:
        if d['id'] == dataset_id:
            dataset = d
            break
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        from serverfiles import LocalFiles, ServerFiles
        
        file_path = tuple(dataset['file_path'])
        
        # Download file
        localfiles = LocalFiles(
            str(DATASETS_CACHE_DIR),
            serverfiles=ServerFiles(server=DATASETS_INDEX_URL)
        )
        local_path = localfiles.localpath_download(*file_path)
        
        return {
            "success": True,
            "local_path": local_path,
            "dataset_id": dataset_id,
            "title": dataset['title']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@api_v1.post("/datasets/{dataset_id:path}/load", tags=["Datasets"])
async def load_dataset(dataset_id: str):
    """
    Load a dataset and return its data information.
    Downloads the file first if not cached locally.
    """
    datasets = await fetch_datasets_list()
    
    dataset = None
    for d in datasets:
        if d['id'] == dataset_id:
            dataset = d
            break
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        from serverfiles import LocalFiles, ServerFiles
        from Orange.data import Table
        
        file_path = tuple(dataset['file_path'])
        
        # Download file if not local
        localfiles = LocalFiles(
            str(DATASETS_CACHE_DIR),
            serverfiles=ServerFiles(server=DATASETS_INDEX_URL)
        )
        local_path = localfiles.localpath_download(*file_path)
        
        # Load with Orange3
        data = Table(local_path)
        
        # Get column info
        columns = []
        
        for var in data.domain.attributes:
            columns.append({
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical",
                "role": "feature",
                "values": ", ".join(var.values) if hasattr(var, 'values') and var.values else ""
            })
        
        if data.domain.class_var:
            var = data.domain.class_var
            columns.append({
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical",
                "role": "target",
                "values": ", ".join(var.values) if hasattr(var, 'values') and var.values else ""
            })
        
        for var in data.domain.metas:
            columns.append({
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical",
                "role": "meta",
                "values": ""
            })
        
        # Helper function to convert Orange3 values to JSON-serializable format
        def get_cell_value(var, val):
            """Convert Orange3 cell value to proper format."""
            import math
            
            # Check for missing/unknown values
            if val is None:
                return None
            if isinstance(val, float) and math.isnan(val):
                return None
            
            if var.is_continuous:
                # Numeric variable: return as float
                return float(val) if not math.isnan(val) else None
            else:
                # Discrete/categorical variable: use str_val to get actual category name
                # str_val handles the index -> value conversion properly
                str_repr = var.str_val(val)
                if str_repr == '?' or str_repr == '':
                    return None
                return str_repr
        
        # Convert data to list of lists for JSON serialization
        data_rows = []
        for row in data:
            row_values = []
            # Attributes (features)
            for var in data.domain.attributes:
                val = row[var]
                row_values.append(get_cell_value(var, val))
            # Class variable (target)
            if data.domain.class_var:
                var = data.domain.class_var
                val = row[var]
                row_values.append(get_cell_value(var, val))
            # Meta attributes
            for var in data.domain.metas:
                val = row[var]
                row_values.append(get_cell_value(var, val))
            data_rows.append(row_values)
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "local_path": local_path,
            "name": data.name or dataset['title'],
            "description": dataset.get('description', ''),
            "instances": len(data),
            "features": len(data.domain.attributes),
            "missingValues": data.has_missing(),
            "classType": "Classification" if data.domain.class_var and not data.domain.class_var.is_continuous else "Regression" if data.domain.class_var else "None",
            "classValues": len(data.domain.class_var.values) if data.domain.class_var and hasattr(data.domain.class_var, 'values') else None,
            "metaAttributes": len(data.domain.metas),
            "columns": columns,
            "data": data_rows  # Include actual data rows
        }
        
    except ImportError:
        raise HTTPException(status_code=501, detail="Orange3 not available")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")


def sizeformat(size):
    """Format file size in human-readable format."""
    for unit in ['bytes', 'KB', 'MB', 'GB']:
        if abs(size) < 1024:
            return f"{size:.1f} {unit}" if unit != 'bytes' else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ============================================================================
# Select Rows API
# ============================================================================

class SelectRowsCondition(BaseModel):
    """A single filter condition."""
    variable: str
    operator: str
    value: Optional[Any] = None
    value2: Optional[Any] = None


class SelectRowsRequest(BaseModel):
    """Request body for select rows."""
    data_source: str
    conditions: List[SelectRowsCondition]
    purge_attributes: bool = False
    purge_classes: bool = False


@api_v1.post("/data/select-rows")
async def select_rows(request: SelectRowsRequest):
    """
    Filter data rows based on conditions.
    
    Returns matching and unmatched data counts.
    """
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Orange3 not available")
    
    try:
        from Orange.data import Table
        import Orange.data.filter as data_filter
        from Orange.data.filter import FilterContinuous, FilterString
        
        # Load data - handle various data source formats
        data_source = request.data_source
        
        # Try loading data in order of preference:
        # 1. Full file path (e.g., /Users/.../datasets/core/abalone.tab)
        # 2. Extract dataset name (e.g., abalone)
        data = None
        
        # First try: full path (most reliable for Datasets widget output)
        if data_source.startswith("/") or os.path.exists(data_source):
            try:
                data = Table(data_source)
            except (OSError, FileNotFoundError):
                pass
        
        # Second try: extract dataset name and use Orange3's built-in loader
        if data is None:
            if "/" in data_source:
                # Extract filename: "datasets/core/abalone.tab" -> "abalone"
                dataset_name = data_source.split("/")[-1].replace(".tab", "")
            else:
                dataset_name = data_source.replace(".tab", "")
            
            try:
                data = Table(dataset_name)
            except (OSError, FileNotFoundError) as e:
                raise HTTPException(status_code=404, detail=f"Dataset not found: {data_source}")
        
        total_count = len(data)
        
        if not request.conditions:
            return {
                "matching_count": total_count,
                "total_count": total_count,
                "unmatched_count": 0
            }
        
        # Build filters
        filters = []
        
        for cond in request.conditions:
            var_name = cond.variable
            op = cond.operator
            val = cond.value
            val2 = cond.value2
            
            # Handle "All" options
            if var_name.startswith('all_'):
                if op == 'is_defined':
                    filters.append(data_filter.IsDefined())
                continue
            
            # Find variable in domain
            try:
                var = data.domain[var_name]
                var_idx = data.domain.index(var)
            except KeyError:
                continue
            
            from Orange.data import ContinuousVariable, DiscreteVariable, StringVariable
            
            if isinstance(var, ContinuousVariable):
                # Numeric filter
                try:
                    float_val = float(val) if val else None
                    float_val2 = float(val2) if val2 else None
                except (ValueError, TypeError):
                    continue
                
                op_map = {
                    'equals': FilterContinuous.Equal,
                    'not_equals': FilterContinuous.NotEqual,
                    'less': FilterContinuous.Less,
                    'less_equal': FilterContinuous.LessEqual,
                    'greater': FilterContinuous.Greater,
                    'greater_equal': FilterContinuous.GreaterEqual,
                    'between': FilterContinuous.Between,
                    'outside': FilterContinuous.Outside,
                    'is_defined': FilterContinuous.IsDefined
                }
                
                if op in op_map:
                    if op == 'is_defined':
                        filters.append(data_filter.FilterContinuous(var_idx, op_map[op]))
                    elif op in ['between', 'outside']:
                        if float_val is not None and float_val2 is not None:
                            filters.append(data_filter.FilterContinuous(
                                var_idx, op_map[op], float_val, float_val2))
                    elif float_val is not None:
                        filters.append(data_filter.FilterContinuous(
                            var_idx, op_map[op], float_val))
                            
            elif isinstance(var, DiscreteVariable):
                # Discrete filter
                if op == 'is_defined':
                    filters.append(data_filter.FilterDiscrete(var_idx, None))
                elif op == 'equals' and val:
                    filters.append(data_filter.FilterDiscrete(var_idx, {val}))
                elif op == 'not_equals' and val:
                    other_vals = set(var.values) - {val}
                    filters.append(data_filter.FilterDiscrete(var_idx, other_vals))
                elif op == 'in' and isinstance(val, list):
                    filters.append(data_filter.FilterDiscrete(var_idx, set(val)))
                    
            elif isinstance(var, StringVariable):
                # String filter
                op_map = {
                    'equals': FilterString.Equal,
                    'not_equals': FilterString.NotEqual,
                    'less': FilterString.Less,
                    'greater': FilterString.Greater,
                    'contains': FilterString.Contains,
                    'not_contains': FilterString.NotContain,
                    'starts_with': FilterString.StartsWith,
                    'ends_with': FilterString.EndsWith,
                    'is_defined': FilterString.IsDefined
                }
                
                if op in op_map and val:
                    filters.append(data_filter.FilterString(
                        var_idx, op_map[op], str(val)))
        
        # Apply filters
        if filters:
            combined_filter = data_filter.Values(filters)
            matching_data = combined_filter(data)
            matching_count = len(matching_data)
        else:
            matching_data = data
            matching_count = total_count
        
        # Helper function to convert cell value
        def get_cell_value(var, val):
            import math
            if var.is_continuous:
                return float(val) if not math.isnan(val) else None
            else:
                # Discrete: use str_val to get actual category name
                return var.str_val(val)
        
        # Convert matching data to list of lists
        data_rows = []
        for row in matching_data:
            row_values = []
            # Attributes
            for var in matching_data.domain.attributes:
                val = row[var]
                row_values.append(get_cell_value(var, val))
            # Class variable
            if matching_data.domain.class_var:
                val = row[matching_data.domain.class_var]
                row_values.append(get_cell_value(matching_data.domain.class_var, val))
            # Metas
            for var in matching_data.domain.metas:
                val = row[var]
                if var.is_string:
                    row_values.append(str(val) if val else None)
                else:
                    row_values.append(get_cell_value(var, val))
            data_rows.append(row_values)
        
        # Build column info
        columns = []
        for var in matching_data.domain.attributes:
            columns.append({
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical",
                "role": "feature"
            })
        if matching_data.domain.class_var:
            columns.append({
                "name": matching_data.domain.class_var.name,
                "type": "numeric" if matching_data.domain.class_var.is_continuous else "categorical",
                "role": "target"
            })
        for var in matching_data.domain.metas:
            columns.append({
                "name": var.name,
                "type": "string" if var.is_string else ("numeric" if var.is_continuous else "categorical"),
                "role": "meta"
            })
        
        return {
            "matching_count": matching_count,
            "total_count": total_count,
            "unmatched_count": total_count - matching_count,
            "data": data_rows,
            "columns": columns,
            "instances": matching_count,
            "features": len(matching_data.domain.attributes)
        }
        
    except Exception as e:
        logger.error(f"Select rows error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Distributions Endpoint
# ============================================================================

class DistributionsRequest(BaseModel):
    """Request model for distributions endpoint."""
    data_path: Optional[str] = None
    variable: str  # Variable name for distribution
    split_by: Optional[str] = None  # Variable to split by (discrete only)
    number_of_bins: int = 5  # Bin count for continuous variables
    stacked: bool = False
    show_probs: bool = False
    cumulative: bool = False
    sort_by_freq: bool = False
    fitted_distribution: int = 0  # 0=None, 1=Normal, 2=Beta, 3=Gamma, etc.
    kde_smoothing: int = 10


@api_v1.post("/data/distributions")
async def get_distributions(request: DistributionsRequest):
    """
    Calculate distribution data for a variable.
    Returns histogram data, statistics, and fitted curve if requested.
    """
    try:
        from Orange.data import Table, DiscreteVariable, ContinuousVariable
        from Orange.statistics import distribution, contingency
        from Orange.preprocess.discretize import decimal_binnings
        import numpy as np
        from scipy.stats import norm, rayleigh, beta, gamma, pareto, expon
        
        # Load data
        data = None
        if request.data_path:
            data_path = request.data_path
            if data_path.startswith("datasets/"):
                dataset_name = data_path.replace("datasets/", "").replace(".tab", "")
                try:
                    data = Table(dataset_name)
                except:
                    data = Table(data_path)
            else:
                data = Table(data_path)
        
        if data is None:
            raise HTTPException(status_code=400, detail="No data provided")
        
        # Find the variable
        var = None
        domain = data.domain
        all_vars = list(domain.attributes) + list(domain.class_vars) + list(domain.metas)
        for v in all_vars:
            if v.name == request.variable:
                var = v
                break
        
        if var is None:
            raise HTTPException(status_code=400, detail=f"Variable '{request.variable}' not found")
        
        # Find split_by variable (discrete only)
        cvar = None
        if request.split_by:
            for v in all_vars:
                if v.name == request.split_by and isinstance(v, DiscreteVariable):
                    cvar = v
                    break
        
        # Get column data
        column = data.get_column(var)
        valid_mask = np.isfinite(column)
        
        if cvar:
            ccolumn = data.get_column(cvar)
            valid_mask = valid_mask & np.isfinite(ccolumn)
        
        valid_data = column[valid_mask]
        valid_group_data = ccolumn[valid_mask] if cvar else None
        
        if len(valid_data) == 0:
            return {
                "variable": var.name,
                "type": "discrete" if isinstance(var, DiscreteVariable) else "continuous",
                "bins": [],
                "total": 0,
                "error": "No valid data"
            }
        
        # Build response
        # Handle colors safely - convert numpy types to Python types
        split_colors = None
        split_values = None
        if cvar:
            # Convert values to Python strings
            split_values = [str(v) for v in cvar.values]
            
            # Convert colors (numpy uint8) to Python int
            try:
                if hasattr(cvar, 'colors') and cvar.colors is not None:
                    split_colors = [[int(c) for c in color] for color in cvar.colors]
            except Exception:
                pass  # Use default colors if colors attribute fails
        
        result = {
            "variable": var.name,
            "type": "discrete" if isinstance(var, DiscreteVariable) else "continuous",
            "split_by": cvar.name if cvar else None,
            "split_values": split_values,
            "split_colors": split_colors,
            "total": int(len(valid_data)),
            "bins": [],
            "fitted_curve": None,
            "statistics": {}
        }
        
        if isinstance(var, DiscreteVariable):
            # Discrete variable - category counts
            if cvar:
                # Split by another variable
                conts = contingency.get_contingency(data, cvar, var)
                conts = np.array(conts)
                
                if request.sort_by_freq:
                    order = np.argsort(conts.sum(axis=1))[::-1]
                else:
                    order = np.arange(len(conts))
                
                ordered_values = [str(var.values[i]) for i in order]
                
                for i, idx in enumerate(order):
                    freqs = [int(f) for f in conts[idx]]  # Convert numpy to int
                    total_freq = int(sum(freqs))
                    result["bins"].append({
                        "label": ordered_values[i],
                        "x": i,
                        "frequencies": freqs,
                        "total": total_freq,
                        "percentage": float(100 * total_freq / len(valid_data)) if len(valid_data) > 0 else 0.0
                    })
            else:
                # Single distribution
                dist = distribution.get_distribution(data, var)
                dist = np.array(dist)
                
                if request.sort_by_freq:
                    order = np.argsort(dist)[::-1]
                else:
                    order = np.arange(len(dist))
                
                ordered_values = [str(var.values[i]) for i in order]
                
                for i, idx in enumerate(order):
                    freq = int(dist[idx])
                    result["bins"].append({
                        "label": ordered_values[i],
                        "x": i,
                        "frequencies": [freq],
                        "total": freq,
                        "percentage": float(100 * freq / len(valid_data)) if len(valid_data) > 0 else 0.0
                    })
        else:
            # Continuous variable - histogram
            binnings = decimal_binnings(valid_data)
            if not binnings:
                # Fallback to simple binning
                bin_count = max(5, min(20, int(np.sqrt(len(valid_data)))))
                thresholds = np.linspace(np.min(valid_data), np.max(valid_data), bin_count + 1)
            else:
                bin_idx = min(request.number_of_bins, len(binnings) - 1)
                thresholds = binnings[bin_idx].thresholds
            
            if cvar:
                # Split by group
                nvalues = len(cvar.values)
                ys = []
                for val_idx in range(nvalues):
                    group_data = valid_data[valid_group_data == val_idx]
                    hist, _ = np.histogram(group_data, bins=thresholds)
                    ys.append(hist)
                
                cumulative_freqs = np.zeros(nvalues)
                for i in range(len(thresholds) - 1):
                    x0, x1 = thresholds[i], thresholds[i + 1]
                    freqs = [int(y[i]) for y in ys]
                    cumulative_freqs += np.array(freqs)
                    
                    if request.cumulative:
                        plot_freqs = cumulative_freqs.astype(int).tolist()
                    else:
                        plot_freqs = freqs
                    
                    result["bins"].append({
                        "label": f"{x0:.3g} - {x1:.3g}",
                        "x0": float(x0),
                        "x1": float(x1),
                        "x": float((x0 + x1) / 2),
                        "frequencies": plot_freqs,
                        "total": int(sum(plot_freqs)),
                        "percentage": float(100 * sum(freqs) / len(valid_data)) if len(valid_data) > 0 else 0.0
                    })
            else:
                # Single histogram
                hist, edges = np.histogram(valid_data, bins=thresholds)
                cumulative = 0
                
                for i in range(len(hist)):
                    x0, x1 = float(edges[i]), float(edges[i + 1])
                    freq = int(hist[i])
                    cumulative += freq
                    
                    plot_freq = cumulative if request.cumulative else freq
                    
                    result["bins"].append({
                        "label": f"{x0:.3g} - {x1:.3g}",
                        "x0": x0,
                        "x1": x1,
                        "x": float((x0 + x1) / 2),
                        "frequencies": [plot_freq],
                        "total": plot_freq,
                        "percentage": float(100 * freq / len(valid_data)) if len(valid_data) > 0 else 0.0
                    })
            
            # Fitted distribution (for continuous only)
            if request.fitted_distribution > 0 and not cvar:
                fitters = [
                    None,  # 0: None
                    norm,  # 1: Normal
                    beta,  # 2: Beta
                    gamma,  # 3: Gamma
                    rayleigh,  # 4: Rayleigh
                    pareto,  # 5: Pareto
                    expon,  # 6: Exponential
                ]
                
                if request.fitted_distribution < len(fitters):
                    fitter = fitters[request.fitted_distribution]
                    if fitter:
                        try:
                            params = fitter.fit(valid_data)
                            x_range = np.linspace(thresholds[0], thresholds[-1], 100)
                            y_pdf = fitter.pdf(x_range, *params)
                            # Scale to match histogram
                            bin_width = (thresholds[-1] - thresholds[0]) / (len(thresholds) - 1)
                            y_scaled = y_pdf * len(valid_data) * bin_width
                            
                            result["fitted_curve"] = {
                                "x": x_range.tolist(),
                                "y": y_scaled.tolist(),
                                "params": list(params),
                                "type": ["None", "Normal", "Beta", "Gamma", "Rayleigh", "Pareto", "Exponential"][request.fitted_distribution]
                            }
                        except Exception as fit_error:
                            logger.warning(f"Fitting error: {fit_error}")
            
            # Statistics
            result["statistics"] = {
                "mean": float(np.mean(valid_data)),
                "std": float(np.std(valid_data)),
                "min": float(np.min(valid_data)),
                "max": float(np.max(valid_data)),
                "median": float(np.median(valid_data))
            }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Distributions error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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
# Include Router (must be after all api_v1 routes are defined)
# ============================================================================

app.include_router(api_v1)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
