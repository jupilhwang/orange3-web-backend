"""
Orange3 data models, type definitions, and shared constants.

This module contains:
- Orange3 import bootstrapping and the ORANGE3_AVAILABLE flag
- Web-friendly dataclasses (WebSchemeNode, WebSchemeLink, WebAnnotation)
- Category color / priority maps
- Parent-class port definitions for widget inheritance resolution
- The ``get_availability`` helper
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# ---- Module-level constant shared by adapters and discovery ----

WIDGET_NAME_OVERRIDES: Dict[str, str] = {
    "Column Statistics": "Feature Statistics",
}

# =====================================================================
# Orange3 Imports (includes canvas-core and widget-base as deps)
# =====================================================================

ORANGE3_AVAILABLE = False

try:
    # From Orange3
    from Orange.widgets import widget_discovery  # noqa: F401
    from Orange.data import Table, Domain, Variable  # noqa: F401
    from Orange.widgets.widget import OWWidget  # noqa: F401

    # From orange-canvas-core (installed with Orange3)
    from orangecanvas.scheme.scheme import Scheme  # noqa: F401
    from orangecanvas.scheme.node import SchemeNode  # noqa: F401
    from orangecanvas.scheme.link import SchemeLink, compatible_channels  # noqa: F401
    from orangecanvas.scheme.annotations import (  # noqa: F401
        BaseSchemeAnnotation,
        SchemeTextAnnotation,
        SchemeArrowAnnotation,
    )
    from orangecanvas.scheme.readwrite import (  # noqa: F401
        scheme_to_ows_stream,
        scheme_load,
    )
    from orangecanvas.registry import WidgetRegistry, WidgetDescription  # noqa: F401
    from orangecanvas.registry.description import (  # noqa: F401
        CategoryDescription,
        InputSignal,
        OutputSignal,
    )

    # From orange-widget-base (installed with Orange3)
    from orangewidget.widget import OWBaseWidget  # noqa: F401
    from orangewidget.settings import Setting, SettingsHandler  # noqa: F401
    from orangewidget.workflow.discovery import WidgetDiscovery  # noqa: F401

    ORANGE3_AVAILABLE = True
    logger.info("Orange3 loaded successfully")

except ImportError as e:
    logger.warning(f"Orange3 not available: {e}")
    logger.info("Install with: pip install Orange3")


# =====================================================================
# Web-friendly data classes
# =====================================================================


@dataclass
class WebSchemeNode:
    """Web-friendly wrapper for SchemeNode."""

    id: str
    widget_id: str
    title: str
    position: Tuple[float, float]
    properties: Dict[str, Any] = field(default_factory=dict)
    state: int = 0
    progress: float = -1

    @classmethod
    def from_scheme_node(cls, node: "SchemeNode", node_id: str) -> "WebSchemeNode":
        """Create from existing SchemeNode using a caller-supplied stable ID."""
        return cls(
            id=node_id,
            widget_id=node.description.qualified_name if node.description else "",
            title=node.title,
            position=node.position or (0, 0),
            properties=node.properties or {},
            state=int(node.state),
            progress=node.progress,
        )

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "widget_id": self.widget_id,
            "title": self.title,
            "position": {"x": self.position[0], "y": self.position[1]},
            "properties": self.properties,
            "state": self.state,
            "progress": self.progress,
        }


@dataclass
class WebSchemeLink:
    """Web-friendly wrapper for SchemeLink."""

    id: str
    source_node_id: str
    source_channel: str
    sink_node_id: str
    sink_channel: str
    enabled: bool = True

    @classmethod
    def from_scheme_link(
        cls, link: "SchemeLink", link_id: str, node_id_map: Dict
    ) -> "WebSchemeLink":
        """Create from existing SchemeLink using a caller-supplied stable ID.

        node_id_map maps Python object id (int) to stable UUID string.
        """
        return cls(
            id=link_id,
            source_node_id=node_id_map.get(id(link.source_node), ""),
            source_channel=link.source_channel.name if link.source_channel else "",
            sink_node_id=node_id_map.get(id(link.sink_node), ""),
            sink_channel=link.sink_channel.name if link.sink_channel else "",
            enabled=link.enabled,
        )

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "source_node_id": self.source_node_id,
            "source_channel": self.source_channel,
            "sink_node_id": self.sink_node_id,
            "sink_channel": self.sink_channel,
            "enabled": self.enabled,
        }


@dataclass
class WebAnnotation:
    """Web-friendly wrapper for annotations."""

    id: str
    type: str
    rect: Optional[Tuple[float, float, float, float]] = None
    content: Optional[str] = None
    content_type: str = "text/plain"
    font: Optional[Dict] = None
    start_pos: Optional[Tuple[float, float]] = None
    end_pos: Optional[Tuple[float, float]] = None
    color: str = "#808080"

    @classmethod
    def from_scheme_annotation(
        cls, annotation: "BaseSchemeAnnotation", annotation_id: str
    ) -> "WebAnnotation":
        """Create from existing annotation using a caller-supplied stable ID."""
        if ORANGE3_AVAILABLE:
            from orangecanvas.scheme.annotations import (
                SchemeTextAnnotation,
                SchemeArrowAnnotation,
            )

            if isinstance(annotation, SchemeTextAnnotation):
                return cls(
                    id=annotation_id,
                    type="text",
                    rect=annotation.rect,
                    content=annotation.content,
                    content_type=annotation.content_type,
                    font=annotation.font,
                )
            elif isinstance(annotation, SchemeArrowAnnotation):
                return cls(
                    id=annotation_id,
                    type="arrow",
                    start_pos=annotation.start_pos,
                    end_pos=annotation.end_pos,
                    color=annotation.color,
                )
        return cls(id=annotation_id, type="unknown")

    def to_dict(self) -> Dict:
        if self.type == "text":
            return {
                "id": self.id,
                "type": "text",
                "rect": {
                    "x": self.rect[0],
                    "y": self.rect[1],
                    "width": self.rect[2],
                    "height": self.rect[3],
                }
                if self.rect
                else None,
                "content": self.content,
                "content_type": self.content_type,
                "font": self.font,
            }
        return {
            "id": self.id,
            "type": "arrow",
            "start_pos": {"x": self.start_pos[0], "y": self.start_pos[1]}
            if self.start_pos
            else None,
            "end_pos": {"x": self.end_pos[0], "y": self.end_pos[1]}
            if self.end_pos
            else None,
            "color": self.color,
        }


# =====================================================================
# Category colors and priorities (core Orange3 widgets)
# =====================================================================

CATEGORY_COLORS = {
    "Data": "#FFD39F",
    "Transform": "#FF9D5E",
    "Visualize": "#FFB7B1",
    "Model": "#FAC1D9",
    "Evaluate": "#C3F3F3",
    "Unsupervised": "#CAE1EF",
}

CATEGORY_PRIORITIES = {
    "Data": 1,
    "Transform": 2,
    "Visualize": 3,
    "Model": 4,
    "Evaluate": 5,
    "Unsupervised": 6,
}

# Default icon as Base64 (simple gray circle SVG)
_DEFAULT_ICON_BASE64 = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9"
    "IjAgMCA0OCA0OCI+PGNpcmNsZSBjeD0iMjQiIGN5PSIyNCIgcj0iMjAiIGZpbGw9"
    "IiM5OTkiLz48L3N2Zz4="
)

# Parent class port definitions — automatically inherited by child widgets
PARENT_CLASS_PORTS = {
    "OWBaseLearner": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "preprocessor", "name": "Preprocessor", "type": "Preprocessor"},
        ],
        "outputs": [
            {"id": "learner", "name": "Learner", "type": "Learner"},
            {"id": "model", "name": "Model", "type": "Model"},
        ],
    },
    "OWProvidesLearner": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "preprocessor", "name": "Preprocessor", "type": "Preprocessor"},
        ],
        "outputs": [
            {"id": "learner", "name": "Learner", "type": "Learner"},
            {"id": "model", "name": "Model", "type": "Model"},
        ],
    },
    "OWDataProjectionWidget": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "data_subset", "name": "Data Subset", "type": "Data"},
        ],
        "outputs": [
            {"id": "selected_data", "name": "Selected Data", "type": "Data"},
            {"id": "annotated_data", "name": "Annotated Data", "type": "Data"},
        ],
    },
    "OWProjectionWidgetBase": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "data_subset", "name": "Data Subset", "type": "Data"},
        ],
        "outputs": [
            {"id": "selected_data", "name": "Selected Data", "type": "Data"},
            {"id": "annotated_data", "name": "Annotated Data", "type": "Data"},
        ],
    },
    "OWAnchorProjectionWidget": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "data_subset", "name": "Data Subset", "type": "Data"},
        ],
        "outputs": [
            {"id": "selected_data", "name": "Selected Data", "type": "Data"},
            {"id": "annotated_data", "name": "Annotated Data", "type": "Data"},
            {"id": "components", "name": "Components", "type": "Data"},
        ],
    },
    "OWTextBaseWidget": {
        "inputs": [{"id": "corpus", "name": "Corpus", "type": "Corpus"}],
        "outputs": [{"id": "corpus", "name": "Corpus", "type": "Corpus"}],
    },
    "OWBaseVectorizer": {
        "inputs": [{"id": "corpus", "name": "Corpus", "type": "Corpus"}],
        "outputs": [{"id": "corpus", "name": "Corpus", "type": "Corpus"}],
    },
}

# Static mapping from class name patterns to parent class (fallback)
CLASS_INHERITANCE_MAP: Dict[str, str] = {}

# Dynamic class inheritance cache (populated at runtime)
_DYNAMIC_INHERITANCE_CACHE: Dict[str, List[str]] = {}


# =====================================================================
# Availability check
# =====================================================================


def get_availability() -> Dict[str, bool]:
    """Check if Orange3 is available."""
    return {"orange3": ORANGE3_AVAILABLE}


__all__ = [
    # Orange3 availability
    "ORANGE3_AVAILABLE",
    "get_availability",
    # Data classes
    "WebSchemeNode",
    "WebSchemeLink",
    "WebAnnotation",
    # Constants
    "WIDGET_NAME_OVERRIDES",
    "CATEGORY_COLORS",
    "CATEGORY_PRIORITIES",
    "PARENT_CLASS_PORTS",
    "CLASS_INHERITANCE_MAP",
    "_DYNAMIC_INHERITANCE_CACHE",
    "_DEFAULT_ICON_BASE64",
]
