"""
OrangeSchemeAdapter — wraps Orange3 workflow schemes for web API usage.

Manages nodes, links, and annotations in an Orange3 Scheme object,
converting between native Orange3 objects and web-friendly dicts.
"""

from typing import Dict, List, Optional, Tuple
import logging
import uuid

from app.core.orange_models import (
    ORANGE3_AVAILABLE,
    WebSchemeNode,
    WebSchemeLink,
    WebAnnotation,
)

logger = logging.getLogger(__name__)


class OrangeSchemeAdapter:
    """Adapter that wraps the existing Scheme class for web API usage."""

    def __init__(self, registry=None):
        if not ORANGE3_AVAILABLE:
            raise ImportError("Orange3 is required")

        from orangecanvas.scheme.scheme import Scheme

        self._scheme: "Scheme" = Scheme()
        self._registry = registry
        self._node_map: Dict[str, object] = {}
        self._link_map: Dict[str, object] = {}
        self._annotation_map: Dict[str, object] = {}
        # Reverse maps: id(obj) → uuid for O(1) lookup in get_workflow_dict
        self._node_reverse_map: Dict[int, str] = {}
        self._link_reverse_map: Dict[int, str] = {}
        self._annotation_reverse_map: Dict[int, str] = {}

    @property
    def scheme(self):
        return self._scheme

    def get_workflow_dict(self) -> Dict:
        """Convert scheme to web-friendly dictionary."""
        node_id_map: Dict[int, str] = {}
        nodes = []
        for node in self._scheme.nodes:
            existing_id = self._node_reverse_map.get(id(node))
            web_id = existing_id if existing_id is not None else str(uuid.uuid4())
            if web_id not in self._node_map:
                self._node_map[web_id] = node
                self._node_reverse_map[id(node)] = web_id
            node_id_map[id(node)] = web_id
            nodes.append(WebSchemeNode.from_scheme_node(node, web_id).to_dict())

        links = []
        for link in self._scheme.links:
            existing_id = self._link_reverse_map.get(id(link))
            web_id = existing_id if existing_id is not None else str(uuid.uuid4())
            if web_id not in self._link_map:
                self._link_map[web_id] = link
                self._link_reverse_map[id(link)] = web_id
            links.append(
                WebSchemeLink.from_scheme_link(link, web_id, node_id_map).to_dict()
            )

        annotations = []
        for annotation in self._scheme.annotations:
            existing_id = self._annotation_reverse_map.get(id(annotation))
            web_id = existing_id if existing_id is not None else str(uuid.uuid4())
            if web_id not in self._annotation_map:
                self._annotation_map[web_id] = annotation
                self._annotation_reverse_map[id(annotation)] = web_id
            annotations.append(
                WebAnnotation.from_scheme_annotation(annotation, web_id).to_dict()
            )

        return {
            "title": self._scheme.title,
            "description": self._scheme.description,
            "nodes": nodes,
            "links": links,
            "annotations": annotations,
        }

    def add_node(
        self, widget_id: str, title: str, position: Tuple[float, float]
    ) -> Dict:
        """Add a node using existing SchemeNode class."""
        if not self._registry:
            raise ValueError("Widget registry not set")

        from orangecanvas.scheme.node import SchemeNode

        widget_desc = self._registry.widget(widget_id)
        if not widget_desc:
            raise ValueError(f"Widget not found: {widget_id}")

        node = SchemeNode(description=widget_desc, title=title, position=position)

        self._scheme.add_node(node)

        web_id = str(uuid.uuid4())
        self._node_map[web_id] = node
        self._node_reverse_map[id(node)] = web_id
        return WebSchemeNode.from_scheme_node(node, web_id).to_dict()

    def remove_node(self, node_id: str) -> bool:
        """Remove a node using existing Scheme method."""
        node = self._node_map.get(node_id)
        if not node:
            return False

        self._scheme.remove_node(node)
        self._node_reverse_map.pop(id(node), None)
        del self._node_map[node_id]
        return True

    def update_node_position(self, node_id: str, position: Tuple[float, float]) -> bool:
        """Update node position."""
        node = self._node_map.get(node_id)
        if not node:
            return False
        node.position = position
        return True

    def add_link(
        self,
        source_node_id: str,
        source_channel: str,
        sink_node_id: str,
        sink_channel: str,
    ) -> Optional[Dict]:
        """Add a link using existing SchemeLink class."""
        from orangecanvas.scheme.link import SchemeLink, compatible_channels

        source_node = self._node_map.get(source_node_id)
        sink_node = self._node_map.get(sink_node_id)

        if not source_node or not sink_node:
            return None

        source_ch = None
        sink_ch = None

        if source_node.description:
            for out in source_node.description.outputs:
                if out.name == source_channel:
                    source_ch = out
                    break

        if sink_node.description:
            for inp in sink_node.description.inputs:
                if inp.name == sink_channel:
                    sink_ch = inp
                    break

        if not source_ch or not sink_ch:
            return None

        if not compatible_channels(source_ch, sink_ch):
            return None

        link = SchemeLink(
            source_node=source_node,
            source_channel=source_ch,
            sink_node=sink_node,
            sink_channel=sink_ch,
        )

        try:
            self._scheme.add_link(link)
        except Exception as e:
            logger.error(f"Error adding link: {e}")
            return None

        web_id = str(uuid.uuid4())
        self._link_map[web_id] = link
        self._link_reverse_map[id(link)] = web_id

        node_id_map = {id(n): k for k, n in self._node_map.items()}
        return WebSchemeLink.from_scheme_link(link, web_id, node_id_map).to_dict()

    def remove_link(self, link_id: str) -> bool:
        """Remove a link."""
        link = self._link_map.get(link_id)
        if not link:
            return False

        self._scheme.remove_link(link)
        self._link_reverse_map.pop(id(link), None)
        del self._link_map[link_id]
        return True

    def add_text_annotation(
        self,
        rect: Tuple[float, float, float, float],
        content: str,
        content_type: str = "text/plain",
    ) -> Dict:
        """Add text annotation."""
        from orangecanvas.scheme.annotations import SchemeTextAnnotation

        annotation = SchemeTextAnnotation(rect=rect, text=content)
        self._scheme.add_annotation(annotation)

        web_id = str(uuid.uuid4())
        self._annotation_map[web_id] = annotation
        self._annotation_reverse_map[id(annotation)] = web_id
        return WebAnnotation.from_scheme_annotation(annotation, web_id).to_dict()

    def add_arrow_annotation(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        color: str = "#808080",
    ) -> Dict:
        """Add arrow annotation."""
        from orangecanvas.scheme.annotations import SchemeArrowAnnotation

        annotation = SchemeArrowAnnotation(
            start_pos=start_pos, end_pos=end_pos, color=color
        )
        self._scheme.add_annotation(annotation)

        web_id = str(uuid.uuid4())
        self._annotation_map[web_id] = annotation
        self._annotation_reverse_map[id(annotation)] = web_id
        return WebAnnotation.from_scheme_annotation(annotation, web_id).to_dict()

    def remove_annotation(self, annotation_id: str) -> bool:
        """Remove annotation."""
        annotation = self._annotation_map.get(annotation_id)
        if not annotation:
            return False

        self._scheme.remove_annotation(annotation)
        self._annotation_reverse_map.pop(id(annotation), None)
        del self._annotation_map[annotation_id]
        return True

    def export_to_ows(self) -> str:
        """Export scheme to OWS format."""
        import io

        from orangecanvas.scheme.readwrite import scheme_to_ows_stream

        stream = io.BytesIO()
        scheme_to_ows_stream(self._scheme, stream)
        return stream.getvalue().decode("utf-8")

    def import_from_ows(self, ows_content: str) -> bool:
        """Import scheme from OWS format."""
        import io

        from orangecanvas.scheme.readwrite import scheme_load

        try:
            stream = io.BytesIO(ows_content.encode("utf-8"))
            self._scheme = scheme_load(stream, registry=self._registry)
            self._node_map = {str(uuid.uuid4()): n for n in self._scheme.nodes}
            self._link_map = {str(uuid.uuid4()): l for l in self._scheme.links}
            self._annotation_map = {
                str(uuid.uuid4()): a for a in self._scheme.annotations
            }
            # Rebuild reverse maps
            self._node_reverse_map = {id(v): k for k, v in self._node_map.items()}
            self._link_reverse_map = {id(v): k for k, v in self._link_map.items()}
            self._annotation_reverse_map = {
                id(v): k for k, v in self._annotation_map.items()
            }
            return True
        except Exception as e:
            logger.error(f"Error importing OWS: {e}")
            return False


__all__ = [
    "OrangeSchemeAdapter",
]
