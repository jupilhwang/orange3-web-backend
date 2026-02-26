"""
Workflow management for Orange3 Web Backend
Based on orange-canvas-core scheme management

Concurrency notes:
    All public methods are synchronous and not thread-safe by themselves.
    Callers (e.g. HTTP route handlers) MUST hold the appropriate
    ``lock_workflow`` / ``lock_tenant`` context from ``core.locks`` before
    invoking any mutating method.  The internal ``_node_index``,
    ``_link_index``, and ``_annotation_index`` dicts are kept in sync with
    ``workflow.nodes / .links / .annotations`` lists, so the lists remain
    the authoritative serialisable representation while the dicts provide
    O(1) lookup.
"""

import logging
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

from .core.models import (
    ContentType,
    Workflow,
    WorkflowNode,
    WorkflowLink,
    WorkflowCreate,
    WorkflowUpdate,
    NodeCreate,
    NodeUpdate,
    LinkCreate,
    LinkUpdate,
    LinkValidation,
    LinkCompatibility,
    TextAnnotation,
    ArrowAnnotation,
    AnnotationCreate,
    AnnotationUpdate,
    Position,
    Rect,
    NodeState,
)

logger = logging.getLogger(__name__)


class WorkflowManager:
    """Manages workflows in memory (can be extended to use database).

    Index maintenance guarantee
    ---------------------------
    Every mutating operation that touches ``workflow.nodes``,
    ``workflow.links``, or ``workflow.annotations`` also updates the
    corresponding per-workflow lookup dict so that:

    * ``_node_index[wf_id][node_id]``  → O(1) node access
    * ``_link_index[wf_id][link_id]``  → O(1) link access
    * ``_annotation_index[wf_id][ann_id]`` → O(1) annotation access
    """

    def __init__(self, registry: Optional["OrangeRegistryAdapter"] = None) -> None:  # noqa: F821
        # In-memory storage: tenant_id -> workflow_id -> Workflow
        self._workflows: Dict[str, Dict[str, Workflow]] = {}

        # Per-workflow O(1) lookup indexes:  workflow_id -> {item_id -> item}
        self._node_index: Dict[str, Dict[str, WorkflowNode]] = {}
        self._link_index: Dict[str, Dict[str, WorkflowLink]] = {}
        self._annotation_index: Dict[
            str, Dict[str, Union[TextAnnotation, ArrowAnnotation]]
        ] = {}

        # Optional OrangeRegistryAdapter for type-compatibility checking
        self._registry = registry

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_tenant_workflows(self, tenant_id: str) -> Dict[str, Workflow]:
        """Get or create the workflow dict for a tenant."""
        if tenant_id not in self._workflows:
            self._workflows[tenant_id] = {}
        return self._workflows[tenant_id]

    def _ensure_indexes(self, workflow_id: str) -> None:
        """Create empty index dicts for a workflow if they don't exist yet."""
        if workflow_id not in self._node_index:
            self._node_index[workflow_id] = {}
        if workflow_id not in self._link_index:
            self._link_index[workflow_id] = {}
        if workflow_id not in self._annotation_index:
            self._annotation_index[workflow_id] = {}

    def _drop_indexes(self, workflow_id: str) -> None:
        """Remove all index entries for a deleted workflow."""
        self._node_index.pop(workflow_id, None)
        self._link_index.pop(workflow_id, None)
        self._annotation_index.pop(workflow_id, None)

    def _rebuild_indexes(self, workflow: Workflow) -> None:
        """Rebuild all indexes for *workflow* from its current lists.

        Used after bulk list replacement (e.g. during OWS import).
        """
        wf_id = workflow.id
        self._node_index[wf_id] = {n.id: n for n in workflow.nodes}
        self._link_index[wf_id] = {lk.id: lk for lk in workflow.links}
        self._annotation_index[wf_id] = {a.id: a for a in workflow.annotations}

    # ========================================================================
    # Workflow CRUD
    # ========================================================================

    def list_workflows(self, tenant_id: str) -> List[Workflow]:
        """List all workflows for a tenant."""
        return list(self._get_tenant_workflows(tenant_id).values())

    def create_workflow(self, tenant_id: str, data: WorkflowCreate) -> Workflow:
        """Create a new workflow."""
        workflow = Workflow(
            tenant_id=tenant_id, title=data.title, description=data.description
        )
        self._get_tenant_workflows(tenant_id)[workflow.id] = workflow
        self._ensure_indexes(workflow.id)
        return workflow

    def get_workflow(self, tenant_id: str, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID."""
        return self._get_tenant_workflows(tenant_id).get(workflow_id)

    def update_workflow(
        self, tenant_id: str, workflow_id: str, data: WorkflowUpdate
    ) -> Optional[Workflow]:
        """Update a workflow."""
        workflow = self.get_workflow(tenant_id, workflow_id)
        if not workflow:
            return None

        if data.title is not None:
            workflow.title = data.title
        if data.description is not None:
            workflow.description = data.description
        if data.env is not None:
            workflow.env = data.env
        workflow.updated_at = datetime.now(timezone.utc)

        return workflow

    def delete_workflow(self, tenant_id: str, workflow_id: str) -> bool:
        """Delete a workflow and clean up all associated index entries."""
        workflows = self._get_tenant_workflows(tenant_id)
        if workflow_id in workflows:
            del workflows[workflow_id]
            self._drop_indexes(workflow_id)
            return True
        return False

    # ========================================================================
    # Node Management
    # ========================================================================

    def add_node(
        self, tenant_id: str, workflow_id: str, data: NodeCreate
    ) -> Optional[WorkflowNode]:
        """Add a node to a workflow."""
        workflow = self.get_workflow(tenant_id, workflow_id)
        if not workflow:
            return None

        node = WorkflowNode(
            widget_id=data.widget_id,
            title=data.title,
            position=data.position,
            properties=data.properties,
        )
        workflow.nodes.append(node)
        self._ensure_indexes(workflow_id)
        self._node_index[workflow_id][node.id] = node
        workflow.updated_at = datetime.now(timezone.utc)

        return node

    def get_node(
        self, tenant_id: str, workflow_id: str, node_id: str
    ) -> Optional[WorkflowNode]:
        """Get a node by ID (O(1) dict lookup)."""
        workflow = self.get_workflow(tenant_id, workflow_id)
        if not workflow:
            return None
        return self._node_index.get(workflow_id, {}).get(node_id)

    def update_node(
        self, tenant_id: str, workflow_id: str, node_id: str, data: NodeUpdate
    ) -> Optional[WorkflowNode]:
        """Update a node."""
        node = self.get_node(tenant_id, workflow_id, node_id)
        if not node:
            return None

        if data.title is not None:
            node.title = data.title
        if data.position is not None:
            node.position = data.position
        if data.properties is not None:
            node.properties = data.properties
        if data.state is not None:
            node.state = data.state
        if data.progress is not None:
            node.progress = data.progress
        node.updated_at = datetime.now(timezone.utc)

        workflow = self.get_workflow(tenant_id, workflow_id)
        if workflow:
            workflow.updated_at = datetime.now(timezone.utc)

        return node

    def delete_node(self, tenant_id: str, workflow_id: str, node_id: str) -> bool:
        """Delete a node and its connected links."""
        workflow = self.get_workflow(tenant_id, workflow_id)
        if not workflow:
            return False

        node_index = self._node_index.get(workflow_id, {})
        if node_id not in node_index:
            return False

        # Remove from list and index
        del node_index[node_id]
        workflow.nodes = [n for n in workflow.nodes if n.id != node_id]

        # Remove connected links from both list and index
        link_index = self._link_index.get(workflow_id, {})
        dead_link_ids = [
            lk_id
            for lk_id, lk in link_index.items()
            if lk.source_node_id == node_id or lk.sink_node_id == node_id
        ]
        for lk_id in dead_link_ids:
            del link_index[lk_id]
        workflow.links = [
            lk
            for lk in workflow.links
            if lk.source_node_id != node_id and lk.sink_node_id != node_id
        ]

        workflow.updated_at = datetime.now(timezone.utc)
        return True

    # ========================================================================
    # Link Management
    # ========================================================================

    def add_link(
        self, tenant_id: str, workflow_id: str, data: LinkCreate
    ) -> Optional[WorkflowLink]:
        """Add a link between nodes."""
        workflow = self.get_workflow(tenant_id, workflow_id)
        if not workflow:
            return None

        # O(1) node existence check via index
        node_index = self._node_index.get(workflow_id, {})
        if data.source_node_id not in node_index or data.sink_node_id not in node_index:
            return None

        # Check for duplicate links (still O(n) over links — unavoidable without
        # a composite-key set, but links are typically few per workflow)
        for link in workflow.links:
            if (
                link.source_node_id == data.source_node_id
                and link.source_channel == data.source_channel
                and link.sink_node_id == data.sink_node_id
                and link.sink_channel == data.sink_channel
            ):
                return None  # Duplicate link

        link = WorkflowLink(
            source_node_id=data.source_node_id,
            source_channel=data.source_channel,
            sink_node_id=data.sink_node_id,
            sink_channel=data.sink_channel,
        )
        workflow.links.append(link)
        self._ensure_indexes(workflow_id)
        self._link_index[workflow_id][link.id] = link
        workflow.updated_at = datetime.now(timezone.utc)

        return link

    def get_link(
        self, tenant_id: str, workflow_id: str, link_id: str
    ) -> Optional[WorkflowLink]:
        """Get a link by ID (O(1) dict lookup)."""
        workflow = self.get_workflow(tenant_id, workflow_id)
        if not workflow:
            return None
        return self._link_index.get(workflow_id, {}).get(link_id)

    def update_link(
        self, tenant_id: str, workflow_id: str, link_id: str, data: LinkUpdate
    ) -> Optional[WorkflowLink]:
        """Update a link."""
        link = self.get_link(tenant_id, workflow_id, link_id)
        if not link:
            return None

        if data.enabled is not None:
            link.enabled = data.enabled

        workflow = self.get_workflow(tenant_id, workflow_id)
        if workflow:
            workflow.updated_at = datetime.now(timezone.utc)

        return link

    def delete_link(self, tenant_id: str, workflow_id: str, link_id: str) -> bool:
        """Delete a link."""
        workflow = self.get_workflow(tenant_id, workflow_id)
        if not workflow:
            return False

        link_index = self._link_index.get(workflow_id, {})
        if link_id not in link_index:
            return False

        del link_index[link_id]
        workflow.links = [lk for lk in workflow.links if lk.id != link_id]
        workflow.updated_at = datetime.now(timezone.utc)
        return True

    def validate_link(
        self, tenant_id: str, workflow_id: str, validation: LinkValidation
    ) -> LinkCompatibility:
        """Validate if a link can be created between two nodes."""
        workflow = self.get_workflow(tenant_id, workflow_id)
        if not workflow:
            return LinkCompatibility(compatible=False, reason="Workflow not found")

        # O(1) node lookup via index
        node_index = self._node_index.get(workflow_id, {})
        source_node = node_index.get(validation.source_node_id)
        sink_node = node_index.get(validation.sink_node_id)

        if not source_node:
            return LinkCompatibility(compatible=False, reason="Source node not found")
        if not sink_node:
            return LinkCompatibility(compatible=False, reason="Sink node not found")

        # Prevent self-loops
        if validation.source_node_id == validation.sink_node_id:
            return LinkCompatibility(compatible=False, reason="Self-loops not allowed")

        # Check for duplicate links
        for link in workflow.links:
            if (
                link.source_node_id == validation.source_node_id
                and link.source_channel == validation.source_channel
                and link.sink_node_id == validation.sink_node_id
                and link.sink_channel == validation.sink_channel
            ):
                return LinkCompatibility(compatible=False, reason="Duplicate link")

        # Type compatibility checking via widget registry
        if self._registry is not None:
            source_widget = self._registry.get_widget(source_node.widget_id)
            sink_widget = self._registry.get_widget(sink_node.widget_id)
            if source_widget and sink_widget:
                source_outputs = source_widget.get("outputs", [])
                sink_inputs = sink_widget.get("inputs", [])

                source_types = next(
                    (
                        ch["types"]
                        for ch in source_outputs
                        if ch["id"] == validation.source_channel
                    ),
                    None,
                )
                sink_types = next(
                    (
                        ch["types"]
                        for ch in sink_inputs
                        if ch["id"] == validation.sink_channel
                    ),
                    None,
                )

                if source_types is None:
                    return LinkCompatibility(
                        compatible=False,
                        reason=f"Source channel '{validation.source_channel}' not found on widget '{source_node.widget_id}'",
                    )
                if sink_types is None:
                    return LinkCompatibility(
                        compatible=False,
                        reason=f"Sink channel '{validation.sink_channel}' not found on widget '{sink_node.widget_id}'",
                    )

                result = self._registry.check_channel_compatibility(
                    source_types, sink_types
                )
                return LinkCompatibility(
                    compatible=result["compatible"],
                    strict=result.get("strict", result["compatible"]),
                    reason=None
                    if result["compatible"]
                    else "Incompatible channel types",
                )

        # Registry unavailable: allow connection (permissive fallback)
        return LinkCompatibility(compatible=True, strict=False)

    # ========================================================================
    # Annotation Management
    # ========================================================================

    def add_annotation(
        self, tenant_id: str, workflow_id: str, data: AnnotationCreate
    ) -> Optional[Union[TextAnnotation, ArrowAnnotation]]:
        """Add an annotation to a workflow."""
        workflow = self.get_workflow(tenant_id, workflow_id)
        if not workflow:
            return None

        if data.type == "text":
            if not data.rect:
                return None
            annotation: Union[TextAnnotation, ArrowAnnotation] = TextAnnotation(
                rect=data.rect,
                content=data.content or "",
                content_type=data.content_type
                if data.content_type is not None
                else ContentType.TextPlain,
                font=data.font or {},
            )
        elif data.type == "arrow":
            if not data.start_pos or not data.end_pos:
                return None
            annotation = ArrowAnnotation(
                start_pos=data.start_pos,
                end_pos=data.end_pos,
                color=data.color or "#808080",
            )
        else:
            return None

        workflow.annotations.append(annotation)
        self._ensure_indexes(workflow_id)
        self._annotation_index[workflow_id][annotation.id] = annotation
        workflow.updated_at = datetime.now(timezone.utc)

        return annotation

    def get_annotation(
        self, tenant_id: str, workflow_id: str, annotation_id: str
    ) -> Optional[Union[TextAnnotation, ArrowAnnotation]]:
        """Get an annotation by ID (O(1) dict lookup)."""
        workflow = self.get_workflow(tenant_id, workflow_id)
        if not workflow:
            return None
        return self._annotation_index.get(workflow_id, {}).get(annotation_id)

    def update_annotation(
        self,
        tenant_id: str,
        workflow_id: str,
        annotation_id: str,
        data: AnnotationUpdate,
    ) -> Optional[Union[TextAnnotation, ArrowAnnotation]]:
        """Update an annotation."""
        annotation = self.get_annotation(tenant_id, workflow_id, annotation_id)
        if not annotation:
            return None

        if isinstance(annotation, TextAnnotation):
            if data.rect is not None:
                annotation.rect = data.rect
            if data.content is not None:
                annotation.content = data.content
            if data.content_type is not None:
                annotation.content_type = data.content_type
            if data.font is not None:
                annotation.font = data.font
        elif isinstance(annotation, ArrowAnnotation):
            if data.start_pos is not None:
                annotation.start_pos = data.start_pos
            if data.end_pos is not None:
                annotation.end_pos = data.end_pos
            if data.color is not None:
                annotation.color = data.color

        workflow = self.get_workflow(tenant_id, workflow_id)
        if workflow:
            workflow.updated_at = datetime.now(timezone.utc)

        return annotation

    def delete_annotation(
        self, tenant_id: str, workflow_id: str, annotation_id: str
    ) -> bool:
        """Delete an annotation."""
        workflow = self.get_workflow(tenant_id, workflow_id)
        if not workflow:
            return False

        ann_index = self._annotation_index.get(workflow_id, {})
        if annotation_id not in ann_index:
            return False

        del ann_index[annotation_id]
        workflow.annotations = [
            a for a in workflow.annotations if a.id != annotation_id
        ]
        workflow.updated_at = datetime.now(timezone.utc)
        return True

    # ========================================================================
    # Import/Export (.ows format)
    # ========================================================================

    def export_to_ows(self, workflow: Workflow) -> str:
        """Export workflow to .ows XML format (Orange Workflow Scheme)."""
        root = ET.Element(
            "scheme",
            {
                "version": "2.0",
                "title": workflow.title,
                "description": workflow.description,
            },
        )

        # Nodes
        nodes_el = ET.SubElement(root, "nodes")
        for node in workflow.nodes:
            ET.SubElement(
                nodes_el,
                "node",
                {
                    "id": node.id,
                    "name": node.title,
                    "qualified_name": node.widget_id,
                    "position": f"({node.position.x}, {node.position.y})",
                },
            )

        # Links
        links_el = ET.SubElement(root, "links")
        for link in workflow.links:
            ET.SubElement(
                links_el,
                "link",
                {
                    "id": link.id,
                    "source_node_id": link.source_node_id,
                    "sink_node_id": link.sink_node_id,
                    "source_channel": link.source_channel,
                    "sink_channel": link.sink_channel,
                    "enabled": str(link.enabled).lower(),
                },
            )

        # Annotations
        annotations_el = ET.SubElement(root, "annotations")
        for annotation in workflow.annotations:
            if isinstance(annotation, TextAnnotation):
                ET.SubElement(
                    annotations_el,
                    "text",
                    {
                        "id": annotation.id,
                        "rect": f"({annotation.rect.x}, {annotation.rect.y}, {annotation.rect.width}, {annotation.rect.height})",
                    },
                ).text = annotation.content
            elif isinstance(annotation, ArrowAnnotation):
                ET.SubElement(
                    annotations_el,
                    "arrow",
                    {
                        "id": annotation.id,
                        "start": f"({annotation.start_pos.x}, {annotation.start_pos.y})",
                        "end": f"({annotation.end_pos.x}, {annotation.end_pos.y})",
                        "color": annotation.color,
                    },
                )

        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    def import_from_ows(
        self, tenant_id: str, workflow_id: str, ows_content: str
    ) -> Optional[Workflow]:
        """Import workflow from .ows XML format."""
        try:
            root = ET.fromstring(ows_content)

            workflow = self.get_workflow(tenant_id, workflow_id)
            if not workflow:
                workflow = Workflow(
                    id=workflow_id,
                    tenant_id=tenant_id,
                    title=root.get("title", "Imported Workflow"),
                    description=root.get("description", ""),
                )
                self._get_tenant_workflows(tenant_id)[workflow.id] = workflow

            # Clear existing content (lists and indexes rebuilt below)
            workflow.nodes = []
            workflow.links = []
            workflow.annotations = []

            # Parse nodes
            nodes_el = root.find("nodes")
            if nodes_el is not None:
                for node_el in nodes_el.findall("node"):
                    pos_str = node_el.get("position", "(0, 0)")
                    parts = [float(p.strip()) for p in pos_str.strip("()").split(",")]
                    if len(parts) >= 2:
                        x, y = parts[0], parts[1]
                    else:
                        x, y = 0.0, 0.0

                    node = WorkflowNode(
                        id=node_el.get("id", str(uuid.uuid4())),
                        widget_id=node_el.get("qualified_name", ""),
                        title=node_el.get("name", ""),
                        position=Position(x=x, y=y),
                    )
                    workflow.nodes.append(node)

            # Parse links
            links_el = root.find("links")
            if links_el is not None:
                for link_el in links_el.findall("link"):
                    link = WorkflowLink(
                        id=link_el.get("id", str(uuid.uuid4())),
                        source_node_id=link_el.get("source_node_id", ""),
                        sink_node_id=link_el.get("sink_node_id", ""),
                        source_channel=link_el.get("source_channel", ""),
                        sink_channel=link_el.get("sink_channel", ""),
                        enabled=link_el.get("enabled", "true").lower() == "true",
                    )
                    workflow.links.append(link)

            # Parse annotations
            annotations_el = root.find("annotations")
            if annotations_el is not None:
                for text_el in annotations_el.findall("text"):
                    rect_str = text_el.get("rect", "(0, 0, 100, 50)")
                    parts = [float(p.strip()) for p in rect_str.strip("()").split(",")]
                    if len(parts) >= 4:
                        rx, ry, rw, rh = parts[0], parts[1], parts[2], parts[3]
                    else:
                        rx, ry, rw, rh = 0.0, 0.0, 100.0, 50.0

                    annotation: Union[TextAnnotation, ArrowAnnotation] = TextAnnotation(
                        id=text_el.get("id", str(uuid.uuid4())),
                        rect=Rect(x=rx, y=ry, width=rw, height=rh),
                        content=text_el.text or "",
                    )
                    workflow.annotations.append(annotation)

                for arrow_el in annotations_el.findall("arrow"):
                    start_str = arrow_el.get("start", "(0, 0)")
                    end_str = arrow_el.get("end", "(100, 100)")

                    start_parts = [
                        float(p.strip()) for p in start_str.strip("()").split(",")
                    ]
                    end_parts = [
                        float(p.strip()) for p in end_str.strip("()").split(",")
                    ]

                    sx = start_parts[0] if len(start_parts) >= 1 else 0.0
                    sy = start_parts[1] if len(start_parts) >= 2 else 0.0
                    ex = end_parts[0] if len(end_parts) >= 1 else 100.0
                    ey = end_parts[1] if len(end_parts) >= 2 else 100.0

                    annotation = ArrowAnnotation(
                        id=arrow_el.get("id", str(uuid.uuid4())),
                        start_pos=Position(x=sx, y=sy),
                        end_pos=Position(x=ex, y=ey),
                        color=arrow_el.get("color", "#808080"),
                    )
                    workflow.annotations.append(annotation)

            # Rebuild indexes from the freshly-populated lists
            self._rebuild_indexes(workflow)

            workflow.updated_at = datetime.now(timezone.utc)
            return workflow

        except Exception as e:
            logger.error("Error importing OWS: %s", e, exc_info=True)
            return None
