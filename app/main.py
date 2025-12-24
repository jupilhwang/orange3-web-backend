"""
Orange3 Web Backend - Multi-tenant FastAPI Application

This backend REUSES the existing orange-canvas-core and orange-widget-base code.
It only adds a thin web API layer on top.

Features:
- SQLite database for persistence
- Async locks for concurrent access protection
- Multi-tenant support via X-Tenant-ID header
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
import json
import uuid
from pydantic import BaseModel

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
    print("=" * 60)
    print("Starting Orange3 Web Backend...")
    print("=" * 60)
    
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
    print("=" * 60)
    
    yield
    
    # Cleanup
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

@api_v1.get("/widgets", tags=["Widgets"])
async def list_widgets(category: Optional[str] = None):
    """List all widgets. Uses existing WidgetRegistry from orange-canvas-core."""
    registry = get_registry()
    if registry and ORANGE_AVAILABLE:
        return registry.list_widgets(category)
    else:
        return []


@api_v1.get("/widgets/categories", tags=["Widgets"])
async def list_categories():
    """List widget categories. Uses existing CategoryDescription."""
    registry = get_registry()
    if registry and ORANGE_AVAILABLE:
        return registry.list_categories()
    else:
        return []


@api_v1.get("/widgets/{widget_id}", tags=["Widgets"])
async def get_widget(widget_id: str):
    """Get widget description. Uses existing WidgetDescription."""
    registry = get_registry()
    if registry and ORANGE_AVAILABLE:
        widget = registry.get_widget(widget_id)
        if widget:
            return widget
    raise HTTPException(status_code=404, detail="Widget not found")


@api_v1.post("/widgets/check-compatibility", tags=["Widgets"])
async def check_compatibility(source_types: List[str], sink_types: List[str]):
    """Check channel compatibility. Uses existing compatible_channels function."""
    registry = get_registry()
    if registry and ORANGE_AVAILABLE:
        return registry.check_channel_compatibility(source_types, sink_types)
    return {"compatible": True, "strict": True, "dynamic": False}


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


@api_v1.get("/data/load", tags=["Data"])
async def load_data_from_path(path: str):
    """Load data from a local file path."""
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Orange3 not available")
    
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
        raise HTTPException(status_code=400, detail=str(e))


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
# Include Router (must be after all api_v1 routes are defined)
# ============================================================================

app.include_router(api_v1)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
