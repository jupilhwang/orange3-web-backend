"""
API Routes for Orange3 Web Backend.
Combines Workflow and Widget Registry routes.
"""

import json
import logging
import os
import sys
import site
import threading
import uuid
from typing import Dict, Any, Optional, List

import base64
import pickle
import io
from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel

from .core import lock_workflow, lock_tenant, get_current_tenant
from .core.models import Tenant
from .websocket_manager import TaskWebSocketManager

logger = logging.getLogger(__name__)


# ============================================================================
# Routers
# ============================================================================

workflow_router = APIRouter(tags=["Workflows"])
widget_registry_router = APIRouter(prefix="/widgets", tags=["Widgets"])


# ============================================================================
# Orange3 Availability Check
# ============================================================================

try:
    from .orange_adapter import (
        OrangeSchemeAdapter,
        OrangeRegistryAdapter,
        ORANGE3_AVAILABLE,
        discover_widgets,
    )

    ORANGE_AVAILABLE = ORANGE3_AVAILABLE
    DISCOVERY_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False
    DISCOVERY_AVAILABLE = False
    discover_widgets = None


# ============================================================================
# Shared State (set by main.py)
# ============================================================================

# In-memory workflow adapters (for Orange3 widget instances)
# Key: tenant_id:workflow_id -> OrangeSchemeAdapter | _WORKFLOW_MANAGER_SENTINEL
_workflow_adapters: Dict[str, Any] = {}

# Sentinel value: indicates that this workflow's data is managed by WorkflowManager
# (used when Orange3 is unavailable, replaces the previous inline fallback dict)
_WORKFLOW_MANAGER_SENTINEL = object()

# 어댑터 딕셔너리 접근 보호를 위한 락
import asyncio

_adapters_lock = asyncio.Lock()

# Widget registry reference (set by main.py)
_registry_getter = None

# WebSocket manager reference (set by main.py)
_websocket_manager: Optional[TaskWebSocketManager] = None

# WorkflowManager reference (set by main.py) — used as fallback when Orange3 is unavailable
_workflow_manager: Optional[Any] = None


def set_registry_getter(getter) -> None:
    """Set the registry getter function."""
    global _registry_getter
    _registry_getter = getter


def set_workflow_manager(manager: Any) -> None:
    """Set the WorkflowManager instance (injected by main.py).

    WorkflowManager handles the Pydantic-model-based workflow data store
    (non-Orange3 fallback path). When Orange3 is unavailable, workflow state
    is persisted here instead of the in-memory fallback dict.
    """
    global _workflow_manager
    _workflow_manager = manager


def set_websocket_manager(manager: TaskWebSocketManager) -> None:
    """Set the WebSocket manager instance."""
    global _websocket_manager
    _websocket_manager = manager


def get_workflow_adapters() -> Dict[str, Any]:
    """Get all workflow adapters (for listing)."""
    return _workflow_adapters


# ============================================================================
# Workflow Adapter Management
# ============================================================================


async def get_workflow_adapter(tenant_id: str, workflow_id: str) -> Optional[Any]:
    """Get workflow adapter with lock protection."""
    key = f"{tenant_id}:{workflow_id}"
    async with _adapters_lock:
        return _workflow_adapters.get(key)


async def create_workflow_adapter(tenant_id: str, workflow_id: str) -> Any:
    """Create a new workflow adapter with lock protection."""
    key = f"{tenant_id}:{workflow_id}"

    async with _adapters_lock:
        if key in _workflow_adapters:
            return _workflow_adapters[key]

        if ORANGE_AVAILABLE and _registry_getter:
            registry = _registry_getter()
            adapter = OrangeSchemeAdapter(
                registry=registry.registry if registry else None
            )
        else:
            # WorkflowManager handles non-Orange3 state — store sentinel
            adapter = _WORKFLOW_MANAGER_SENTINEL

        _workflow_adapters[key] = adapter
        return adapter


async def delete_workflow_adapter(tenant_id: str, workflow_id: str) -> bool:
    """Delete workflow adapter with lock protection."""
    key = f"{tenant_id}:{workflow_id}"

    async with _adapters_lock:
        if key in _workflow_adapters:
            del _workflow_adapters[key]
            return True
        return False


# ============================================================================
# Workflow CRUD Endpoints
# ============================================================================


class WorkflowCreate(BaseModel):
    """Request model for creating a workflow."""

    title: str = ""
    description: str = ""


@workflow_router.get("/workflows")
async def list_workflows(tenant: Tenant = Depends(get_current_tenant)) -> list:
    """List all workflows for tenant."""
    result = []

    async with _adapters_lock:
        adapters_snapshot = dict(_workflow_adapters)

    for key, adapter in adapters_snapshot.items():
        if key.startswith(f"{tenant.id}:"):
            workflow_id = key.split(":", 1)[1]

            if ORANGE_AVAILABLE and hasattr(adapter, "get_workflow_dict"):
                data = adapter.get_workflow_dict()
                result.append(
                    {
                        "id": workflow_id,
                        "title": data.get("title", ""),
                        "description": data.get("description", ""),
                        "node_count": len(data.get("nodes", [])),
                        "link_count": len(data.get("links", [])),
                    }
                )
            else:
                # WorkflowManager handles non-Orange3 state
                wf = None
                if _workflow_manager:
                    wf = _workflow_manager.get_workflow(tenant.id, workflow_id)
                if wf:
                    result.append(
                        {
                            "id": workflow_id,
                            "title": wf.title,
                            "description": getattr(wf, "description", ""),
                            "node_count": len(wf.nodes),
                            "link_count": len(wf.links),
                        }
                    )
                else:
                    result.append(
                        {
                            "id": workflow_id,
                            "title": "",
                            "node_count": 0,
                            "link_count": 0,
                        }
                    )

    return result


@workflow_router.post("/workflows")
async def create_workflow(
    data: WorkflowCreate, tenant: Tenant = Depends(get_current_tenant)
) -> dict:
    """Create a new workflow. Creates a new Scheme instance."""
    workflow_id = str(uuid.uuid4())

    async with lock_tenant(tenant.id):
        adapter = await create_workflow_adapter(tenant.id, workflow_id)

        if ORANGE_AVAILABLE and hasattr(adapter, "scheme"):
            adapter.scheme.title = data.title
            adapter.scheme.description = data.description
        elif adapter is _WORKFLOW_MANAGER_SENTINEL and _workflow_manager:
            _workflow_manager.create_workflow(
                tenant_id=tenant.id,
                workflow_id=workflow_id,
                title=data.title,
                description=data.description,
            )

    return {"id": workflow_id, "title": data.title, "description": data.description}


@workflow_router.get("/workflows/{workflow_id}")
async def get_workflow(
    workflow_id: str, tenant: Tenant = Depends(get_current_tenant)
) -> dict:
    """Get workflow details. Returns the full Scheme structure."""
    adapter = await get_workflow_adapter(tenant.id, workflow_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if ORANGE_AVAILABLE and hasattr(adapter, "get_workflow_dict"):
        data = adapter.get_workflow_dict()
        data["id"] = workflow_id
        return data
    elif adapter is _WORKFLOW_MANAGER_SENTINEL and _workflow_manager:
        wf = _workflow_manager.get_workflow(tenant.id, workflow_id)
        if not wf:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return {
            "id": workflow_id,
            "title": wf.title,
            "description": getattr(wf, "description", ""),
            "nodes": [n.model_dump() for n in wf.nodes],
            "links": [lk.model_dump() for lk in wf.links],
        }

    raise HTTPException(status_code=500, detail="Invalid workflow state")


@workflow_router.delete("/workflows/{workflow_id}", tags=["Workflows"])
async def delete_workflow(
    workflow_id: str, tenant: Tenant = Depends(get_current_tenant)
) -> dict:
    """Delete a workflow."""
    adapter = await get_workflow_adapter(tenant.id, workflow_id)
    deleted = await delete_workflow_adapter(tenant.id, workflow_id)

    # Clean up WorkflowManager state for non-Orange3 workflows
    if adapter is _WORKFLOW_MANAGER_SENTINEL and _workflow_manager:
        _workflow_manager.delete_workflow(tenant.id, workflow_id)

    if deleted:
        return {"message": "Workflow deleted"}
    raise HTTPException(status_code=404, detail="Workflow not found")


@workflow_router.get("/workflows/{workflow_id}/export")
async def export_workflow(
    workflow_id: str, tenant: Tenant = Depends(get_current_tenant)
) -> Response:
    """Export workflow as .ows. Uses existing scheme_to_ows_stream function."""
    adapter = await get_workflow_adapter(tenant.id, workflow_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if ORANGE_AVAILABLE and hasattr(adapter, "export_to_ows"):
        ows_content = adapter.export_to_ows()
        return Response(
            content=ows_content,
            media_type="application/xml",
            headers={"Content-Disposition": "attachment; filename=workflow.ows"},
        )

    raise HTTPException(status_code=501, detail="Export not available")


# ============================================================================
# OWS Pickle and Literal Decoding
# ============================================================================


class DecodePickleRequest(BaseModel):
    """Request model for decoding a base64 pickle string."""

    pickle_data: str


class DecodeLiteralRequest(BaseModel):
    """Request model for decoding a Python literal string."""

    literal_data: str


def _safe_serialize(obj: Any, allow_objects: bool = True) -> Any:
    """Safely serialize arbitrary Python objects to JSON-compatible types.

    Args:
        allow_objects: If True, tries to serialize objects with __dict__.
                      If False, falls back to str() for unknown types.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("utf-8")
    if isinstance(obj, (set, frozenset)):
        return [_safe_serialize(item, allow_objects) for item in obj]
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(item, allow_objects) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v, allow_objects) for k, v in obj.items()}
    if allow_objects and hasattr(obj, "__dict__"):
        return {
            k: _safe_serialize(v, allow_objects)
            for k, v in obj.__dict__.items()
            if not k.startswith("_")
        }
    return str(obj)


class _OrangeRestrictedUnpickler(pickle.Unpickler):
    """Unpickler that restricts deserialization to types used in Orange3 OWS widget
    settings only.

    OWS widget settings are persisted as pickle, but the set of allowed types is
    well-known and finite.  Blocking arbitrary globals prevents an RCE attack where
    a crafted OWS file calls os.system / subprocess / builtins.eval etc.

    Allowed categories:
    - Python builtins: int, float, bool, str, bytes, bytearray, list, tuple,
                       set, frozenset, dict, type(None)
    - datetime / collections / array
    - numpy scalar/array types (if numpy is installed)
    - Orange3 core value/domain types (if Orange3 is installed)
    """

    # (module, qualname) pairs that are unconditionally allowed
    _ALLOWED: frozenset = frozenset(
        {
            # builtins
            ("builtins", "int"),
            ("builtins", "float"),
            ("builtins", "bool"),
            ("builtins", "str"),
            ("builtins", "bytes"),
            ("builtins", "bytearray"),
            ("builtins", "list"),
            ("builtins", "tuple"),
            ("builtins", "set"),
            ("builtins", "frozenset"),
            ("builtins", "dict"),
            ("builtins", "NoneType"),
            ("builtins", "complex"),
            # datetime
            ("datetime", "datetime"),
            ("datetime", "date"),
            ("datetime", "time"),
            ("datetime", "timedelta"),
            # collections
            ("collections", "OrderedDict"),
            ("collections", "defaultdict"),
            ("collections", "namedtuple"),
            # array
            ("array", "array"),
            # copy_reg / pickle internals
            ("copy_reg", "_reconstructor"),
            ("__builtin__", "object"),
        }
    )

    # Module prefixes that are allowed (numpy, Orange3 value/domain types)
    _ALLOWED_PREFIXES: tuple = (
        "numpy",
        "Orange.data",
        "Orange.base",
        "Orange.preprocess",
    )

    def find_class(self, module: str, name: str):
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        for prefix in self._ALLOWED_PREFIXES:
            if module == prefix or module.startswith(prefix + "."):
                return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Blocked: pickle class '{module}.{name}' is not in the allowlist. "
            "Only Orange3 OWS widget setting types are permitted."
        )


def _safe_unpickle(data: bytes) -> object:
    """Deserialize *data* using the restricted unpickler."""
    import io

    return _OrangeRestrictedUnpickler(io.BytesIO(data)).load()


@workflow_router.post("/ows/decode_pickle")
async def decode_ows_pickle(data: DecodePickleRequest) -> dict:
    """Decode a base64-encoded Orange3 pickle string into a JSON-friendly dict.

    Uses a restricted unpickler that only allows Orange3 OWS widget setting
    types, preventing arbitrary code execution via crafted pickle payloads.
    """
    try:
        binary_data = base64.b64decode(data.pickle_data)
        obj = _safe_unpickle(binary_data)
        result = _safe_serialize(obj, allow_objects=True)
        return {"status": "success", "data": result}
    except pickle.UnpicklingError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to decode pickle: {str(e)}"
        )


@workflow_router.post("/ows/decode_literal")
async def decode_ows_literal(data: DecodeLiteralRequest) -> Any:
    """Decode a Python literal string (from OWS format='literal') into a JSON-friendly dict."""
    try:
        import ast

        # HTML 엔티티 디코딩
        literal_str = data.literal_data
        literal_str = literal_str.replace("&amp;", "&")
        literal_str = literal_str.replace("&lt;", "<")
        literal_str = literal_str.replace("&gt;", ">")
        literal_str = literal_str.replace("&quot;", '"')
        literal_str = literal_str.replace("&apos;", "'")

        # Python literal 파싱
        obj = ast.literal_eval(literal_str)

        # Convert to JSON-compatible structure
        result = _safe_serialize(obj, allow_objects=False)
        return result
    except Exception as e:
        logger.error(f"Literal decoding failed: {e}")
        return {"error": str(e), "fallback": True}


# ============================================================================
# Node CRUD Endpoints
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


@workflow_router.post("/workflows/{workflow_id}/nodes", tags=["Nodes"])
async def add_node(
    workflow_id: str,
    node: NodeCreateRequest,
    tenant: Tenant = Depends(get_current_tenant),
) -> dict:
    """Add a node. Uses existing SchemeNode class."""
    adapter = await get_workflow_adapter(tenant.id, workflow_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Workflow not found")

    async with lock_workflow(workflow_id):
        if ORANGE_AVAILABLE and hasattr(adapter, "add_node"):
            try:
                result = adapter.add_node(
                    widget_id=node.widget_id,
                    title=node.title,
                    position=(node.x, node.y),
                )

                # Broadcast to WebSocket
                if _websocket_manager:
                    await _websocket_manager.broadcast_to_workflow(
                        workflow_id, {"type": "node_added", "data": result}
                    )

                return result
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        # Fallback: delegate to WorkflowManager
        if adapter is _WORKFLOW_MANAGER_SENTINEL and _workflow_manager:
            from .core.models import WorkflowNode, Position as NodePosition

            node_id = str(uuid.uuid4())
            new_node = WorkflowNode(
                id=node_id,
                widget_id=node.widget_id,
                title=node.title or node.widget_id,
                position=NodePosition(x=node.x, y=node.y),
            )
            _workflow_manager.add_node(tenant.id, workflow_id, new_node)
            return new_node.model_dump()

        raise HTTPException(status_code=503, detail="Workflow adapter unavailable")


@workflow_router.delete("/workflows/{workflow_id}/nodes/{node_id}", tags=["Nodes"])
async def delete_node(
    workflow_id: str, node_id: str, tenant: Tenant = Depends(get_current_tenant)
) -> dict:
    """Delete a node. Uses existing Scheme.remove_node method."""
    adapter = await get_workflow_adapter(tenant.id, workflow_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Workflow not found")

    async with lock_workflow(workflow_id):
        if ORANGE_AVAILABLE and hasattr(adapter, "remove_node"):
            if adapter.remove_node(node_id):
                if _websocket_manager:
                    await _websocket_manager.broadcast_to_workflow(
                        workflow_id,
                        {"type": "node_removed", "data": {"node_id": node_id}},
                    )
                return {"message": "Node deleted"}
        elif adapter is _WORKFLOW_MANAGER_SENTINEL and _workflow_manager:
            if _workflow_manager.delete_node(tenant.id, workflow_id, node_id):
                if _websocket_manager:
                    await _websocket_manager.broadcast_to_workflow(
                        workflow_id,
                        {"type": "node_removed", "data": {"node_id": node_id}},
                    )
                return {"message": "Node deleted"}

    raise HTTPException(status_code=404, detail="Node not found")


@workflow_router.put(
    "/workflows/{workflow_id}/nodes/{node_id}/position", tags=["Nodes"]
)
async def update_node_position(
    workflow_id: str,
    node_id: str,
    position: NodePositionUpdate,
    tenant: Tenant = Depends(get_current_tenant),
) -> dict:
    """Update node position."""
    adapter = await get_workflow_adapter(tenant.id, workflow_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Workflow not found")

    async with lock_workflow(workflow_id):
        if ORANGE_AVAILABLE and hasattr(adapter, "update_node_position"):
            if adapter.update_node_position(node_id, (position.x, position.y)):
                return {"message": "Position updated"}

    raise HTTPException(status_code=404, detail="Node not found")


# ============================================================================
# Link CRUD Endpoints
# ============================================================================


class LinkCreateRequest(BaseModel):
    """Request model for creating a link."""

    source_node_id: str
    source_channel: str
    sink_node_id: str
    sink_channel: str


@workflow_router.post("/workflows/{workflow_id}/links", tags=["Links"])
async def add_link(
    workflow_id: str,
    link: LinkCreateRequest,
    tenant: Tenant = Depends(get_current_tenant),
) -> dict:
    """Add a link. Uses existing SchemeLink class and compatible_channels function."""
    adapter = await get_workflow_adapter(tenant.id, workflow_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Workflow not found")

    async with lock_workflow(workflow_id):
        if ORANGE_AVAILABLE and hasattr(adapter, "add_link"):
            result = adapter.add_link(
                source_node_id=link.source_node_id,
                source_channel=link.source_channel,
                sink_node_id=link.sink_node_id,
                sink_channel=link.sink_channel,
            )
            if result:
                if _websocket_manager:
                    await _websocket_manager.broadcast_to_workflow(
                        workflow_id, {"type": "link_added", "data": result}
                    )
                return result
            raise HTTPException(status_code=400, detail="Cannot create link")

        # Fallback: delegate to WorkflowManager
        if adapter is _WORKFLOW_MANAGER_SENTINEL and _workflow_manager:
            from .core.models import WorkflowLink

            link_id = str(uuid.uuid4())
            new_link = WorkflowLink(
                id=link_id,
                source_node_id=link.source_node_id,
                target_node_id=link.target_node_id,
                source_channel=link.source_channel,
                target_channel=link.target_channel,
            )
            _workflow_manager.add_link(tenant.id, workflow_id, new_link)
            return new_link.model_dump()

        raise HTTPException(status_code=503, detail="Workflow adapter unavailable")


@workflow_router.delete("/workflows/{workflow_id}/links/{link_id}", tags=["Links"])
async def delete_link(
    workflow_id: str, link_id: str, tenant: Tenant = Depends(get_current_tenant)
) -> dict:
    """Delete a link. Uses existing Scheme.remove_link method."""
    adapter = await get_workflow_adapter(tenant.id, workflow_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Workflow not found")

    async with lock_workflow(workflow_id):
        if ORANGE_AVAILABLE and hasattr(adapter, "remove_link"):
            if adapter.remove_link(link_id):
                if _websocket_manager:
                    await _websocket_manager.broadcast_to_workflow(
                        workflow_id,
                        {"type": "link_removed", "data": {"link_id": link_id}},
                    )
                return {"message": "Link deleted"}
        elif adapter is _WORKFLOW_MANAGER_SENTINEL and _workflow_manager:
            if _workflow_manager.delete_link(tenant.id, workflow_id, link_id):
                if _websocket_manager:
                    await _websocket_manager.broadcast_to_workflow(
                        workflow_id,
                        {"type": "link_removed", "data": {"link_id": link_id}},
                    )
                return {"message": "Link deleted"}

    raise HTTPException(status_code=404, detail="Link not found")


# ============================================================================
# Annotation Endpoints
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


@workflow_router.post("/workflows/{workflow_id}/annotations/text", tags=["Annotations"])
async def add_text_annotation(
    workflow_id: str,
    annotation: TextAnnotationCreate,
    tenant: Tenant = Depends(get_current_tenant),
) -> dict:
    """Add text annotation. Uses existing SchemeTextAnnotation class."""
    adapter = await get_workflow_adapter(tenant.id, workflow_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Workflow not found")

    async with lock_workflow(workflow_id):
        if ORANGE_AVAILABLE and hasattr(adapter, "add_text_annotation"):
            result = adapter.add_text_annotation(
                rect=(annotation.x, annotation.y, annotation.width, annotation.height),
                content=annotation.content,
                content_type=annotation.content_type,
            )
            return result

    raise HTTPException(status_code=501, detail="Annotations not available")


@workflow_router.post(
    "/workflows/{workflow_id}/annotations/arrow", tags=["Annotations"]
)
async def add_arrow_annotation(
    workflow_id: str,
    annotation: ArrowAnnotationCreate,
    tenant: Tenant = Depends(get_current_tenant),
) -> dict:
    """Add arrow annotation. Uses existing SchemeArrowAnnotation class."""
    adapter = await get_workflow_adapter(tenant.id, workflow_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Workflow not found")

    async with lock_workflow(workflow_id):
        if ORANGE_AVAILABLE and hasattr(adapter, "add_arrow_annotation"):
            result = adapter.add_arrow_annotation(
                start_pos=(annotation.start_x, annotation.start_y),
                end_pos=(annotation.end_x, annotation.end_y),
                color=annotation.color,
            )
            return result

    raise HTTPException(status_code=501, detail="Annotations not available")


# ============================================================================
# WebSocket Endpoint
# ============================================================================


async def websocket_endpoint(websocket: WebSocket, workflow_id: str) -> None:
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
                workflow_id, message, exclude=websocket
            )
    except WebSocketDisconnect:
        await _websocket_manager.disconnect(websocket, workflow_id)


# ============================================================================
# Widget Registry - Discovery Cache
# ============================================================================

# Cache for discovered widgets (refreshed on startup)
_discovered_widgets = None
_discovered_widgets_lock = threading.Lock()


def get_discovered_widgets() -> dict:
    """Get or discover widgets from Orange3 installation (thread-safe singleton).

    Priority:
    1. Use discover_widgets() (includes Orange3-Text and all add-ons)
    2. Fall back to OrangeRegistryAdapter if discover_widgets() fails
    """
    global _discovered_widgets
    if _discovered_widgets is not None:
        return _discovered_widgets
    with _discovered_widgets_lock:
        if _discovered_widgets is not None:  # double-check after acquiring lock
            return _discovered_widgets

        # Primary: Use discover_widgets() - includes Orange3-Text widgets
        if DISCOVERY_AVAILABLE:
            try:
                logger.info("Using discover_widgets() for widget discovery...")
                result = discover_widgets()
                total = result.get("total", 0)
                categories = result.get("categories", [])
                logger.info(
                    f"Discovered {total} widgets in {len(categories)} categories"
                )

                if total > 0:
                    _discovered_widgets = result
                    return _discovered_widgets
            except Exception as e:
                logger.warning(f"discover_widgets() failed: {e}")

        # Fallback: Use OrangeRegistryAdapter
        if _registry_getter and ORANGE_AVAILABLE:
            registry = _registry_getter()
            if registry:
                try:
                    categories = registry.list_categories()
                    widgets = registry.list_widgets()

                    _discovered_widgets = {
                        "categories": [
                            {
                                "name": cat["name"],
                                "color": cat.get("background", "#808080"),
                                "priority": cat.get("priority", 10),
                                "widgets": [
                                    w
                                    for w in widgets
                                    if w.get("category") == cat["name"]
                                ],
                            }
                            for cat in categories
                        ],
                        "widgets": widgets,
                        "total": len(widgets),
                    }
                    logger.info(
                        f"Fallback to OrangeRegistryAdapter: {len(widgets)} widgets"
                    )
                    return _discovered_widgets
                except Exception as e:
                    logger.warning(f"OrangeRegistryAdapter failed: {e}")

        # Last resort: empty result
        logger.warning("No widgets discovered")
        _discovered_widgets = {"categories": [], "widgets": [], "total": 0}

    return _discovered_widgets


# ============================================================================
# Widget Registry Endpoints
# ============================================================================


@widget_registry_router.get("")
async def list_widgets(category: Optional[str] = None) -> list:
    """List all widgets discovered from Orange3 installation."""
    discovered = get_discovered_widgets()
    widgets = discovered.get("widgets", [])

    if category:
        widgets = [w for w in widgets if w.get("category") == category]

    return widgets


@widget_registry_router.get("/categories")
async def list_categories() -> list:
    """List widget categories discovered from Orange3."""
    discovered = get_discovered_widgets()
    categories = discovered.get("categories", [])

    # Format for frontend compatibility
    return [
        {
            "name": cat["name"],
            "color": cat.get("color", "#999999"),
            "priority": cat.get("priority", 10),
            "widget_count": len(cat.get("widgets", [])),
        }
        for cat in categories
    ]


@widget_registry_router.get("/all")
async def get_all_widgets() -> dict:
    """Get all categories with their widgets (for frontend toolbox)."""
    discovered = get_discovered_widgets()
    return discovered


@widget_registry_router.get("/{widget_id}")
async def get_widget(widget_id: str) -> dict:
    """Get widget description by ID."""
    discovered = get_discovered_widgets()
    widgets = discovered.get("widgets", [])

    for widget in widgets:
        if widget.get("id") == widget_id:
            return widget

    raise HTTPException(status_code=404, detail="Widget not found")


@widget_registry_router.post("/check-compatibility")
async def check_compatibility(source_types: List[str], sink_types: List[str]) -> dict:
    """Check channel compatibility between source and sink types."""
    compatible = any(
        st == sk or st == "Any" or sk == "Any"
        for st in source_types
        for sk in sink_types
    )
    return {"compatible": compatible, "strict": True, "dynamic": False}


@widget_registry_router.post("/refresh")
async def refresh_widgets() -> dict:
    """Force refresh of widget discovery cache."""
    global _discovered_widgets
    _discovered_widgets = None
    discovered = get_discovered_widgets()
    return {"message": "Widget cache refreshed", "total": discovered.get("total", 0)}


# ============================================================================
# Legacy Endpoint Support
# ============================================================================


async def legacy_widgets_handler() -> dict:
    """
    Legacy endpoint handler for frontend compatibility.
    Returns widgets grouped by category.
    """
    if _registry_getter:
        registry = _registry_getter()
        if registry and ORANGE_AVAILABLE:
            categories = registry.list_categories()
            widgets = registry.list_widgets()

            # Group widgets by category
            result = []
            for cat in categories:
                cat_widgets = [w for w in widgets if w.get("category") == cat["name"]]
                result.append(
                    {
                        "name": cat["name"],
                        "color": cat.get("background", "#808080"),
                        "widgets": cat_widgets,
                    }
                )
            return {"categories": result}

    return {"categories": []}
