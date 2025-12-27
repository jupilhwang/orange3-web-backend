"""
Workflow API Routes.
Handles Workflow CRUD, Nodes, Links, Annotations, and WebSocket.
"""

import json
import logging
import uuid
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel

from ..locks import lock_workflow, lock_tenant
from ..tenant import Tenant, get_current_tenant
from ..websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Workflows"])

# Check Orange3 availability
try:
    from ..orange_adapter import OrangeSchemeAdapter, OrangeRegistryAdapter
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False

# ============================================================================
# Workflow Adapter Management
# ============================================================================

# In-memory workflow adapters (for Orange3 widget instances)
# Key: tenant_id:workflow_id -> OrangeSchemeAdapter
_workflow_adapters: Dict[str, Any] = {}

# Widget registry reference (set by main.py)
_registry_getter = None

# WebSocket manager reference (set by main.py)
_websocket_manager: Optional[WebSocketManager] = None


def set_registry_getter(getter):
    """Set the registry getter function."""
    global _registry_getter
    _registry_getter = getter


def set_websocket_manager(manager: WebSocketManager):
    """Set the WebSocket manager instance."""
    global _websocket_manager
    _websocket_manager = manager


def get_workflow_adapters() -> Dict[str, Any]:
    """Get all workflow adapters (for listing)."""
    return _workflow_adapters


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
        
        if ORANGE_AVAILABLE and _registry_getter:
            registry = _registry_getter()
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
# Workflow CRUD
# ============================================================================

@router.get("/workflows")
async def list_workflows(tenant: Tenant = Depends(get_current_tenant)):
    """List all workflows for tenant."""
    result = []
    
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


@router.post("/workflows")
async def create_workflow(data: "WorkflowCreate", tenant: Tenant = Depends(get_current_tenant)):
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


@router.get("/workflows/{workflow_id}")
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


@router.delete("/workflows/{workflow_id}")
async def delete_workflow(workflow_id: str, tenant: Tenant = Depends(get_current_tenant)):
    """Delete a workflow."""
    deleted = await delete_workflow_adapter(tenant.id, workflow_id)
    if deleted:
        return {"message": "Workflow deleted"}
    raise HTTPException(status_code=404, detail="Workflow not found")


@router.get("/workflows/{workflow_id}/export")
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
            headers={"Content-Disposition": "attachment; filename=workflow.ows"}
        )
    
    raise HTTPException(status_code=501, detail="Export not available")


# ============================================================================
# Node CRUD
# ============================================================================

class NodeCreateRequest(BaseModel):
    """Request model for creating a node."""
    widget_id: str
    title: str
    x: float
    y: float


class NodePositionUpdate(BaseModel):
    """Request model for updating node position."""
    x: float
    y: float


@router.post("/workflows/{workflow_id}/nodes", tags=["Nodes"])
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
                if _websocket_manager:
                    await _websocket_manager.broadcast_to_workflow(
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


@router.delete("/workflows/{workflow_id}/nodes/{node_id}", tags=["Nodes"])
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
                if _websocket_manager:
                    await _websocket_manager.broadcast_to_workflow(
                        workflow_id,
                        {"type": "node_removed", "data": {"node_id": node_id}}
                    )
                return {"message": "Node deleted"}
    
    raise HTTPException(status_code=404, detail="Node not found")


@router.put("/workflows/{workflow_id}/nodes/{node_id}/position", tags=["Nodes"])
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
# Link CRUD
# ============================================================================

class LinkCreateRequest(BaseModel):
    """Request model for creating a link."""
    source_node_id: str
    source_channel: str
    sink_node_id: str
    sink_channel: str


@router.post("/workflows/{workflow_id}/links", tags=["Links"])
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
                if _websocket_manager:
                    await _websocket_manager.broadcast_to_workflow(
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


@router.delete("/workflows/{workflow_id}/links/{link_id}", tags=["Links"])
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
                if _websocket_manager:
                    await _websocket_manager.broadcast_to_workflow(
                        workflow_id,
                        {"type": "link_removed", "data": {"link_id": link_id}}
                    )
                return {"message": "Link deleted"}
    
    raise HTTPException(status_code=404, detail="Link not found")


# ============================================================================
# Annotations
# ============================================================================

class TextAnnotationCreate(BaseModel):
    """Request model for creating a text annotation."""
    x: float
    y: float
    width: float
    height: float
    content: str
    content_type: str = "text/plain"


class ArrowAnnotationCreate(BaseModel):
    """Request model for creating an arrow annotation."""
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    color: str = "#808080"


@router.post("/workflows/{workflow_id}/annotations/text", tags=["Annotations"])
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


@router.post("/workflows/{workflow_id}/annotations/arrow", tags=["Annotations"])
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

async def websocket_endpoint(websocket: WebSocket, workflow_id: str):
    """WebSocket for real-time updates."""
    if not _websocket_manager:
        await websocket.close(code=1011)
        return
    
    await _websocket_manager.connect(websocket, workflow_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Broadcast to other clients
            await _websocket_manager.broadcast_to_workflow(
                workflow_id,
                message,
                exclude=websocket
            )
    except WebSocketDisconnect:
        _websocket_manager.disconnect(websocket, workflow_id)


# ============================================================================
# Pydantic Models (imported from main for type hints)
# ============================================================================

class WorkflowCreate(BaseModel):
    """Request model for creating a workflow."""
    title: str = ""
    description: str = ""

