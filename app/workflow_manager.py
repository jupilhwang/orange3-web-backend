"""
Workflow management for Orange3 Web Backend
Based on orange-canvas-core scheme management
"""
from typing import Dict, List, Optional, Union
from datetime import datetime
import uuid
import xml.etree.ElementTree as ET

from .core.models import (
    Workflow, WorkflowNode, WorkflowLink, WorkflowCreate, WorkflowUpdate,
    NodeCreate, NodeUpdate, LinkCreate, LinkUpdate, LinkValidation, LinkCompatibility,
    TextAnnotation, ArrowAnnotation, AnnotationCreate, AnnotationUpdate,
    Position, Rect, NodeState
)


class WorkflowManager:
    """Manages workflows in memory (can be extended to use database)."""
    
    def __init__(self):
        # In-memory storage: tenant_id -> workflow_id -> Workflow
        self._workflows: Dict[str, Dict[str, Workflow]] = {}
    
    def _get_tenant_workflows(self, tenant_id: str) -> Dict[str, Workflow]:
        """Get or create the workflow dict for a tenant."""
        if tenant_id not in self._workflows:
            self._workflows[tenant_id] = {}
        return self._workflows[tenant_id]
    
    # ========================================================================
    # Workflow CRUD
    # ========================================================================
    
    def list_workflows(self, tenant_id: str) -> List[Workflow]:
        """List all workflows for a tenant."""
        return list(self._get_tenant_workflows(tenant_id).values())
    
    def create_workflow(self, tenant_id: str, data: WorkflowCreate) -> Workflow:
        """Create a new workflow."""
        workflow = Workflow(
            tenant_id=tenant_id,
            title=data.title,
            description=data.description
        )
        self._get_tenant_workflows(tenant_id)[workflow.id] = workflow
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
        workflow.updated_at = datetime.utcnow()
        
        return workflow
    
    def delete_workflow(self, tenant_id: str, workflow_id: str) -> bool:
        """Delete a workflow."""
        workflows = self._get_tenant_workflows(tenant_id)
        if workflow_id in workflows:
            del workflows[workflow_id]
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
            properties=data.properties
        )
        workflow.nodes.append(node)
        workflow.updated_at = datetime.utcnow()
        
        return node
    
    def get_node(
        self, tenant_id: str, workflow_id: str, node_id: str
    ) -> Optional[WorkflowNode]:
        """Get a node by ID."""
        workflow = self.get_workflow(tenant_id, workflow_id)
        if not workflow:
            return None
        
        for node in workflow.nodes:
            if node.id == node_id:
                return node
        return None
    
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
        node.updated_at = datetime.utcnow()
        
        # Update workflow timestamp
        workflow = self.get_workflow(tenant_id, workflow_id)
        if workflow:
            workflow.updated_at = datetime.utcnow()
        
        return node
    
    def delete_node(
        self, tenant_id: str, workflow_id: str, node_id: str
    ) -> bool:
        """Delete a node and its connected links."""
        workflow = self.get_workflow(tenant_id, workflow_id)
        if not workflow:
            return False
        
        # Find and remove the node
        node_found = False
        for i, node in enumerate(workflow.nodes):
            if node.id == node_id:
                workflow.nodes.pop(i)
                node_found = True
                break
        
        if not node_found:
            return False
        
        # Remove connected links
        workflow.links = [
            link for link in workflow.links
            if link.source_node_id != node_id and link.sink_node_id != node_id
        ]
        
        workflow.updated_at = datetime.utcnow()
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
        
        # Verify source and sink nodes exist
        source_exists = any(n.id == data.source_node_id for n in workflow.nodes)
        sink_exists = any(n.id == data.sink_node_id for n in workflow.nodes)
        
        if not source_exists or not sink_exists:
            return None
        
        # Check for duplicate links
        for link in workflow.links:
            if (link.source_node_id == data.source_node_id and
                link.source_channel == data.source_channel and
                link.sink_node_id == data.sink_node_id and
                link.sink_channel == data.sink_channel):
                return None  # Duplicate link
        
        link = WorkflowLink(
            source_node_id=data.source_node_id,
            source_channel=data.source_channel,
            sink_node_id=data.sink_node_id,
            sink_channel=data.sink_channel
        )
        workflow.links.append(link)
        workflow.updated_at = datetime.utcnow()
        
        return link
    
    def get_link(
        self, tenant_id: str, workflow_id: str, link_id: str
    ) -> Optional[WorkflowLink]:
        """Get a link by ID."""
        workflow = self.get_workflow(tenant_id, workflow_id)
        if not workflow:
            return None
        
        for link in workflow.links:
            if link.id == link_id:
                return link
        return None
    
    def update_link(
        self, tenant_id: str, workflow_id: str, link_id: str, data: LinkUpdate
    ) -> Optional[WorkflowLink]:
        """Update a link."""
        link = self.get_link(tenant_id, workflow_id, link_id)
        if not link:
            return None
        
        if data.enabled is not None:
            link.enabled = data.enabled
        
        # Update workflow timestamp
        workflow = self.get_workflow(tenant_id, workflow_id)
        if workflow:
            workflow.updated_at = datetime.utcnow()
        
        return link
    
    def delete_link(
        self, tenant_id: str, workflow_id: str, link_id: str
    ) -> bool:
        """Delete a link."""
        workflow = self.get_workflow(tenant_id, workflow_id)
        if not workflow:
            return False
        
        for i, link in enumerate(workflow.links):
            if link.id == link_id:
                workflow.links.pop(i)
                workflow.updated_at = datetime.utcnow()
                return True
        
        return False
    
    def validate_link(
        self, tenant_id: str, workflow_id: str, validation: LinkValidation
    ) -> LinkCompatibility:
        """Validate if a link can be created between two nodes."""
        workflow = self.get_workflow(tenant_id, workflow_id)
        if not workflow:
            return LinkCompatibility(compatible=False, reason="Workflow not found")
        
        # Check if nodes exist
        source_node = None
        sink_node = None
        for node in workflow.nodes:
            if node.id == validation.source_node_id:
                source_node = node
            if node.id == validation.sink_node_id:
                sink_node = node
        
        if not source_node:
            return LinkCompatibility(compatible=False, reason="Source node not found")
        if not sink_node:
            return LinkCompatibility(compatible=False, reason="Sink node not found")
        
        # Check for cycles (simplified - just prevent self-loops)
        if validation.source_node_id == validation.sink_node_id:
            return LinkCompatibility(compatible=False, reason="Self-loops not allowed")
        
        # Check for duplicate links
        for link in workflow.links:
            if (link.source_node_id == validation.source_node_id and
                link.source_channel == validation.source_channel and
                link.sink_node_id == validation.sink_node_id and
                link.sink_channel == validation.sink_channel):
                return LinkCompatibility(compatible=False, reason="Duplicate link")
        
        # TODO: Add actual type compatibility checking using widget registry
        return LinkCompatibility(compatible=True, strict=True)
    
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
            annotation = TextAnnotation(
                rect=data.rect,
                content=data.content or "",
                content_type=data.content_type or "text/plain",
                font=data.font or {}
            )
        elif data.type == "arrow":
            if not data.start_pos or not data.end_pos:
                return None
            annotation = ArrowAnnotation(
                start_pos=data.start_pos,
                end_pos=data.end_pos,
                color=data.color or "#808080"
            )
        else:
            return None
        
        workflow.annotations.append(annotation)
        workflow.updated_at = datetime.utcnow()
        
        return annotation
    
    def get_annotation(
        self, tenant_id: str, workflow_id: str, annotation_id: str
    ) -> Optional[Union[TextAnnotation, ArrowAnnotation]]:
        """Get an annotation by ID."""
        workflow = self.get_workflow(tenant_id, workflow_id)
        if not workflow:
            return None
        
        for annotation in workflow.annotations:
            if annotation.id == annotation_id:
                return annotation
        return None
    
    def update_annotation(
        self, tenant_id: str, workflow_id: str, annotation_id: str, data: AnnotationUpdate
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
        
        # Update workflow timestamp
        workflow = self.get_workflow(tenant_id, workflow_id)
        if workflow:
            workflow.updated_at = datetime.utcnow()
        
        return annotation
    
    def delete_annotation(
        self, tenant_id: str, workflow_id: str, annotation_id: str
    ) -> bool:
        """Delete an annotation."""
        workflow = self.get_workflow(tenant_id, workflow_id)
        if not workflow:
            return False
        
        for i, annotation in enumerate(workflow.annotations):
            if annotation.id == annotation_id:
                workflow.annotations.pop(i)
                workflow.updated_at = datetime.utcnow()
                return True
        
        return False
    
    # ========================================================================
    # Import/Export (.ows format)
    # ========================================================================
    
    def export_to_ows(self, workflow: Workflow) -> str:
        """Export workflow to .ows XML format (Orange Workflow Scheme)."""
        root = ET.Element("scheme", {
            "version": "2.0",
            "title": workflow.title,
            "description": workflow.description
        })
        
        # Nodes
        nodes_el = ET.SubElement(root, "nodes")
        for node in workflow.nodes:
            ET.SubElement(nodes_el, "node", {
                "id": node.id,
                "name": node.title,
                "qualified_name": node.widget_id,
                "position": f"({node.position.x}, {node.position.y})"
            })
        
        # Links
        links_el = ET.SubElement(root, "links")
        for link in workflow.links:
            ET.SubElement(links_el, "link", {
                "id": link.id,
                "source_node_id": link.source_node_id,
                "sink_node_id": link.sink_node_id,
                "source_channel": link.source_channel,
                "sink_channel": link.sink_channel,
                "enabled": str(link.enabled).lower()
            })
        
        # Annotations
        annotations_el = ET.SubElement(root, "annotations")
        for annotation in workflow.annotations:
            if isinstance(annotation, TextAnnotation):
                ET.SubElement(annotations_el, "text", {
                    "id": annotation.id,
                    "rect": f"({annotation.rect.x}, {annotation.rect.y}, {annotation.rect.width}, {annotation.rect.height})"
                }).text = annotation.content
            elif isinstance(annotation, ArrowAnnotation):
                ET.SubElement(annotations_el, "arrow", {
                    "id": annotation.id,
                    "start": f"({annotation.start_pos.x}, {annotation.start_pos.y})",
                    "end": f"({annotation.end_pos.x}, {annotation.end_pos.y})",
                    "color": annotation.color
                })
        
        return ET.tostring(root, encoding="unicode", xml_declaration=True)
    
    def import_from_ows(
        self, tenant_id: str, workflow_id: str, ows_content: str
    ) -> Optional[Workflow]:
        """Import workflow from .ows XML format."""
        try:
            root = ET.fromstring(ows_content)
            
            workflow = self.get_workflow(tenant_id, workflow_id)
            if not workflow:
                # Create new workflow
                workflow = Workflow(
                    id=workflow_id,
                    tenant_id=tenant_id,
                    title=root.get("title", "Imported Workflow"),
                    description=root.get("description", "")
                )
                self._get_tenant_workflows(tenant_id)[workflow.id] = workflow
            
            # Clear existing content
            workflow.nodes = []
            workflow.links = []
            workflow.annotations = []
            
            # Parse nodes
            nodes_el = root.find("nodes")
            if nodes_el is not None:
                for node_el in nodes_el.findall("node"):
                    pos_str = node_el.get("position", "(0, 0)")
                    # Parse "(x, y)" format
                    pos_str = pos_str.strip("()")
                    parts = [float(p.strip()) for p in pos_str.split(",")]
                    
                    node = WorkflowNode(
                        id=node_el.get("id", str(uuid.uuid4())),
                        widget_id=node_el.get("qualified_name", ""),
                        title=node_el.get("name", ""),
                        position=Position(x=parts[0], y=parts[1])
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
                        enabled=link_el.get("enabled", "true").lower() == "true"
                    )
                    workflow.links.append(link)
            
            # Parse annotations
            annotations_el = root.find("annotations")
            if annotations_el is not None:
                for text_el in annotations_el.findall("text"):
                    rect_str = text_el.get("rect", "(0, 0, 100, 50)")
                    rect_str = rect_str.strip("()")
                    parts = [float(p.strip()) for p in rect_str.split(",")]
                    
                    annotation = TextAnnotation(
                        id=text_el.get("id", str(uuid.uuid4())),
                        rect=Rect(x=parts[0], y=parts[1], width=parts[2], height=parts[3]),
                        content=text_el.text or ""
                    )
                    workflow.annotations.append(annotation)
                
                for arrow_el in annotations_el.findall("arrow"):
                    start_str = arrow_el.get("start", "(0, 0)")
                    end_str = arrow_el.get("end", "(100, 100)")
                    
                    start_str = start_str.strip("()")
                    start_parts = [float(p.strip()) for p in start_str.split(",")]
                    
                    end_str = end_str.strip("()")
                    end_parts = [float(p.strip()) for p in end_str.split(",")]
                    
                    annotation = ArrowAnnotation(
                        id=arrow_el.get("id", str(uuid.uuid4())),
                        start_pos=Position(x=start_parts[0], y=start_parts[1]),
                        end_pos=Position(x=end_parts[0], y=end_parts[1]),
                        color=arrow_el.get("color", "#808080")
                    )
                    workflow.annotations.append(annotation)
            
            workflow.updated_at = datetime.utcnow()
            return workflow
            
        except Exception as e:
            print(f"Error importing OWS: {e}")
            return None


