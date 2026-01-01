"""
Orange3 Adapter - Wraps existing Orange3 code for web API usage.

This module imports and uses EXISTING Orange3 classes directly,
only adding a minimal web API layer on top.

Orange3 includes orange-canvas-core and orange-widget-base as dependencies,
so we only need to install Orange3.
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json

# =============================================================================
# Orange3 Imports (includes canvas-core and widget-base as dependencies)
# =============================================================================
ORANGE3_AVAILABLE = False

try:
    # From Orange3
    from Orange.widgets import widget_discovery
    from Orange.data import Table, Domain, Variable
    from Orange.widgets.widget import OWWidget
    
    # From orange-canvas-core (installed with Orange3)
    from orangecanvas.scheme.scheme import Scheme
    from orangecanvas.scheme.node import SchemeNode
    from orangecanvas.scheme.link import SchemeLink, compatible_channels
    from orangecanvas.scheme.annotations import (
        BaseSchemeAnnotation, SchemeTextAnnotation, SchemeArrowAnnotation
    )
    from orangecanvas.scheme.readwrite import scheme_to_ows_stream, scheme_load
    from orangecanvas.registry import WidgetRegistry, WidgetDescription
    from orangecanvas.registry.description import (
        CategoryDescription, InputSignal, OutputSignal
    )
    
    # From orange-widget-base (installed with Orange3)
    from orangewidget.widget import OWBaseWidget
    from orangewidget.settings import Setting, SettingsHandler
    from orangewidget.workflow.discovery import WidgetDiscovery
    
    ORANGE3_AVAILABLE = True
    print("Orange3 loaded successfully")
    
except ImportError as e:
    print(f"Warning: Orange3 not available: {e}")
    print("Install with: pip install Orange3")


# =============================================================================
# Web-friendly data classes
# =============================================================================

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
    def from_scheme_node(cls, node: 'SchemeNode') -> "WebSchemeNode":
        """Create from existing SchemeNode."""
        return cls(
            id=str(id(node)),
            widget_id=node.description.qualified_name if node.description else "",
            title=node.title,
            position=node.position or (0, 0),
            properties=node.properties or {},
            state=int(node.state),
            progress=node.progress
        )
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "widget_id": self.widget_id,
            "title": self.title,
            "position": {"x": self.position[0], "y": self.position[1]},
            "properties": self.properties,
            "state": self.state,
            "progress": self.progress
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
    def from_scheme_link(cls, link: 'SchemeLink', node_id_map: Dict) -> "WebSchemeLink":
        """Create from existing SchemeLink."""
        return cls(
            id=str(id(link)),
            source_node_id=node_id_map.get(id(link.source_node), ""),
            source_channel=link.source_channel.name if link.source_channel else "",
            sink_node_id=node_id_map.get(id(link.sink_node), ""),
            sink_channel=link.sink_channel.name if link.sink_channel else "",
            enabled=link.enabled
        )
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "source_node_id": self.source_node_id,
            "source_channel": self.source_channel,
            "sink_node_id": self.sink_node_id,
            "sink_channel": self.sink_channel,
            "enabled": self.enabled
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
    def from_scheme_annotation(cls, annotation: 'BaseSchemeAnnotation') -> "WebAnnotation":
        if isinstance(annotation, SchemeTextAnnotation):
            return cls(
                id=str(id(annotation)),
                type="text",
                rect=annotation.rect,
                content=annotation.content,
                content_type=annotation.content_type,
                font=annotation.font
            )
        elif isinstance(annotation, SchemeArrowAnnotation):
            return cls(
                id=str(id(annotation)),
                type="arrow",
                start_pos=annotation.start_pos,
                end_pos=annotation.end_pos,
                color=annotation.color
            )
        return cls(id=str(id(annotation)), type="unknown")
    
    def to_dict(self) -> Dict:
        if self.type == "text":
            return {
                "id": self.id,
                "type": "text",
                "rect": {"x": self.rect[0], "y": self.rect[1], 
                        "width": self.rect[2], "height": self.rect[3]} if self.rect else None,
                "content": self.content,
                "content_type": self.content_type,
                "font": self.font
            }
        return {
            "id": self.id,
            "type": "arrow",
            "start_pos": {"x": self.start_pos[0], "y": self.start_pos[1]} if self.start_pos else None,
            "end_pos": {"x": self.end_pos[0], "y": self.end_pos[1]} if self.end_pos else None,
            "color": self.color
        }


# =============================================================================
# Orange3 Registry Adapter (uses Orange3's widget_discovery)
# =============================================================================

class OrangeRegistryAdapter:
    """
    Adapter that uses Orange3's widget discovery.
    """
    
    def __init__(self):
        self._registry = WidgetRegistry() if ORANGE3_AVAILABLE else None
        self._categories: List[Dict] = []
        self._widgets: Dict[str, Dict] = {}
        self._loaded = False
    
    @property
    def registry(self) -> Optional['WidgetRegistry']:
        return self._registry
    
    def discover_widgets(self):
        """Discover widgets using Orange3's widget_discovery."""
        if self._loaded:
            return
        
        try:
            if ORANGE3_AVAILABLE:
                print("Using Orange3 widget discovery...")
                
                discovery = WidgetDiscovery(self._registry)
                widget_discovery(discovery)
                self._process_registry()
            else:
                print("Orange3 not available, using fallback...")
                self._manual_discovery()
            
            self._loaded = True
            print(f"Discovered {len(self._widgets)} widgets in {len(self._categories)} categories")
            
        except Exception as e:
            print(f"Error discovering widgets: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_registry(self):
        """Process the registry after discovery."""
        if not self._registry:
            return
        
        # Extract categories
        for cat in self._registry.categories():
            self._categories.append({
                "name": cat.name,
                "description": cat.description or "",
                "background": cat.background or "#808080",
                "priority": cat.priority or 0,
                "icon": cat.icon or ""
            })
        
        # Extract widgets
        for widget in self._registry.widgets():
            widget_dict = self._widget_to_dict(widget)
            self._widgets[widget.qualified_name] = widget_dict
    
    def _manual_discovery(self):
        """Manual widget discovery when Orange3 is not available."""
        # Define basic categories
        self._categories = [
            {"name": "Data", "background": "#FFD39F", "priority": 1},
            {"name": "Visualize", "background": "#6FA8DC", "priority": 2},
            {"name": "Model", "background": "#E69138", "priority": 3},
            {"name": "Evaluate", "background": "#93C47D", "priority": 4},
            {"name": "Unsupervised", "background": "#8E7CC3", "priority": 5},
        ]
    
    # 위젯 이름 매핑 (Orange3 원래 이름 -> 표시 이름)
    WIDGET_NAME_OVERRIDES = {
        "Column Statistics": "Feature Statistics",
    }
    
    def _widget_to_dict(self, widget: 'WidgetDescription') -> Dict:
        """Convert WidgetDescription to dict."""
        inputs = []
        for inp in (widget.inputs or []):
            inputs.append({
                "id": inp.name,
                "name": inp.name,
                "types": list(inp.types) if inp.types else [],
                "flags": inp.flags if hasattr(inp, 'flags') else 0,
                "multiple": getattr(inp, 'single', 1) == 0
            })
        
        outputs = []
        for out in (widget.outputs or []):
            outputs.append({
                "id": out.name,
                "name": out.name,
                "types": list(out.types) if out.types else [],
                "flags": out.flags if hasattr(out, 'flags') else 0,
                "dynamic": getattr(out, 'dynamic', False)
            })
        
        # 위젯 이름 오버라이드 적용
        display_name = self.WIDGET_NAME_OVERRIDES.get(widget.name, widget.name)
        
        return {
            "id": widget.qualified_name,
            "name": display_name,
            "description": widget.description or "",
            "icon": widget.icon or "",
            "category": widget.category or "",
            "keywords": list(widget.keywords) if widget.keywords else [],
            "inputs": inputs,
            "outputs": outputs,
            "background": widget.background or ""
        }
    
    def list_categories(self) -> List[Dict]:
        """List all categories."""
        return self._categories
    
    def list_widgets(self, category: str = None) -> List[Dict]:
        """List all widgets, optionally filtered by category."""
        widgets = list(self._widgets.values())
        if category:
            widgets = [w for w in widgets if w.get("category") == category]
        return widgets
    
    def get_widget(self, widget_id: str) -> Optional[Dict]:
        """Get a specific widget."""
        return self._widgets.get(widget_id)
    
    def check_channel_compatibility(
        self, 
        source_types: List[str], 
        sink_types: List[str]
    ) -> Dict:
        """Check if channels are compatible using existing function."""
        if not ORANGE3_AVAILABLE:
            return {"compatible": True, "strict": True, "dynamic": False}
        
        source = OutputSignal(name="source", types=tuple(source_types))
        sink = InputSignal(name="sink", types=tuple(sink_types))
        
        compatible = compatible_channels(source, sink)
        
        return {
            "compatible": compatible,
            "strict": compatible,
            "dynamic": False
        }


# =============================================================================
# Orange3 Scheme Adapter (workflow management)
# =============================================================================

class OrangeSchemeAdapter:
    """
    Adapter that wraps the existing Scheme class for web API usage.
    """
    
    def __init__(self, registry: 'WidgetRegistry' = None):
        if not ORANGE3_AVAILABLE:
            raise ImportError("Orange3 is required")
        
        self._scheme: 'Scheme' = Scheme()
        self._registry = registry
        self._node_map: Dict[str, 'SchemeNode'] = {}
        self._link_map: Dict[str, 'SchemeLink'] = {}
        self._annotation_map: Dict[str, 'BaseSchemeAnnotation'] = {}
    
    @property
    def scheme(self) -> 'Scheme':
        return self._scheme
    
    def get_workflow_dict(self) -> Dict:
        """Convert scheme to web-friendly dictionary."""
        node_id_map = {}
        nodes = []
        for node in self._scheme.nodes:
            web_id = str(id(node))
            node_id_map[id(node)] = web_id
            self._node_map[web_id] = node
            nodes.append(WebSchemeNode.from_scheme_node(node).to_dict())
        
        links = []
        for link in self._scheme.links:
            web_id = str(id(link))
            self._link_map[web_id] = link
            links.append(WebSchemeLink.from_scheme_link(link, node_id_map).to_dict())
        
        annotations = []
        for annotation in self._scheme.annotations:
            web_id = str(id(annotation))
            self._annotation_map[web_id] = annotation
            annotations.append(WebAnnotation.from_scheme_annotation(annotation).to_dict())
        
        return {
            "title": self._scheme.title,
            "description": self._scheme.description,
            "nodes": nodes,
            "links": links,
            "annotations": annotations
        }
    
    def add_node(self, widget_id: str, title: str, position: Tuple[float, float]) -> Dict:
        """Add a node using existing SchemeNode class."""
        if not self._registry:
            raise ValueError("Widget registry not set")
        
        widget_desc = self._registry.widget(widget_id)
        if not widget_desc:
            raise ValueError(f"Widget not found: {widget_id}")
        
        node = SchemeNode(
            description=widget_desc,
            title=title,
            position=position
        )
        
        self._scheme.add_node(node)
        
        web_id = str(id(node))
        self._node_map[web_id] = node
        return WebSchemeNode.from_scheme_node(node).to_dict()
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node using existing Scheme method."""
        node = self._node_map.get(node_id)
        if not node:
            return False
        
        self._scheme.remove_node(node)
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
        sink_channel: str
    ) -> Optional[Dict]:
        """Add a link using existing SchemeLink class."""
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
            sink_channel=sink_ch
        )
        
        try:
            self._scheme.add_link(link)
        except Exception as e:
            print(f"Error adding link: {e}")
            return None
        
        web_id = str(id(link))
        self._link_map[web_id] = link
        
        node_id_map = {id(n): str(id(n)) for n in self._scheme.nodes}
        return WebSchemeLink.from_scheme_link(link, node_id_map).to_dict()
    
    def remove_link(self, link_id: str) -> bool:
        """Remove a link."""
        link = self._link_map.get(link_id)
        if not link:
            return False
        
        self._scheme.remove_link(link)
        del self._link_map[link_id]
        return True
    
    def add_text_annotation(
        self, 
        rect: Tuple[float, float, float, float], 
        content: str,
        content_type: str = "text/plain"
    ) -> Dict:
        """Add text annotation."""
        annotation = SchemeTextAnnotation(rect=rect, text=content)
        self._scheme.add_annotation(annotation)
        
        web_id = str(id(annotation))
        self._annotation_map[web_id] = annotation
        return WebAnnotation.from_scheme_annotation(annotation).to_dict()
    
    def add_arrow_annotation(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        color: str = "#808080"
    ) -> Dict:
        """Add arrow annotation."""
        annotation = SchemeArrowAnnotation(start_pos=start_pos, end_pos=end_pos, color=color)
        self._scheme.add_annotation(annotation)
        
        web_id = str(id(annotation))
        self._annotation_map[web_id] = annotation
        return WebAnnotation.from_scheme_annotation(annotation).to_dict()
    
    def remove_annotation(self, annotation_id: str) -> bool:
        """Remove annotation."""
        annotation = self._annotation_map.get(annotation_id)
        if not annotation:
            return False
        
        self._scheme.remove_annotation(annotation)
        del self._annotation_map[annotation_id]
        return True
    
    def export_to_ows(self) -> str:
        """Export scheme to OWS format."""
        import io
        stream = io.BytesIO()
        scheme_to_ows_stream(self._scheme, stream)
        return stream.getvalue().decode('utf-8')
    
    def import_from_ows(self, ows_content: str) -> bool:
        """Import scheme from OWS format."""
        import io
        try:
            stream = io.BytesIO(ows_content.encode('utf-8'))
            self._scheme = scheme_load(stream, registry=self._registry)
            self._node_map = {str(id(n)): n for n in self._scheme.nodes}
            self._link_map = {str(id(l)): l for l in self._scheme.links}
            self._annotation_map = {str(id(a)): a for a in self._scheme.annotations}
            return True
        except Exception as e:
            print(f"Error importing OWS: {e}")
            return False


# =============================================================================
# Availability check
# =============================================================================

def get_availability() -> Dict[str, bool]:
    """Check if Orange3 is available."""
    return {
        "orange3": ORANGE3_AVAILABLE
    }


# =============================================================================
# Widget Discovery
# Automatically discovers Orange3 widgets by scanning widget directories
# and parsing Python files using AST (no import required).
# =============================================================================

import os
import ast
import re
import sys
import site
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_orange3_path_from_import() -> Optional[str]:
    """Try to get Orange3 widgets path by importing Orange module."""
    try:
        import Orange
        orange_dir = os.path.dirname(Orange.__file__)
        widgets_path = os.path.join(orange_dir, 'widgets')
        if os.path.exists(widgets_path):
            return widgets_path
    except ImportError:
        pass
    return None


def _get_orange3_text_path_from_import() -> Optional[str]:
    """Try to get Orange3-Text widgets path."""
    try:
        import orangecontrib.text
        text_dir = os.path.dirname(orangecontrib.text.__file__)
        widgets_path = os.path.join(text_dir, 'widgets')
        if os.path.exists(widgets_path):
            return widgets_path
    except ImportError:
        pass
    return None


def _get_site_packages_paths() -> List[str]:
    """Get all possible site-packages paths dynamically."""
    paths = []
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    
    for p in sys.path:
        if 'site-packages' in p and os.path.isdir(p):
            paths.append(p)
    
    try:
        for sp in site.getsitepackages():
            if sp and os.path.isdir(sp):
                paths.append(sp)
    except Exception:
        pass
    
    venv_path = os.environ.get('VIRTUAL_ENV')
    if venv_path:
        linux_path = os.path.join(venv_path, 'lib', py_version, 'site-packages')
        if os.path.isdir(linux_path):
            paths.append(linux_path)
    
    return list(dict.fromkeys(paths))


# Category colors and priorities
CATEGORY_COLORS = {
    "Data": "#FFD39F",
    "Transform": "#FF9D5E",
    "Visualize": "#FFB7B1",
    "Model": "#FAC1D9",
    "Evaluate": "#C3F3F3",
    "Unsupervised": "#CAE1EF",
    "Text Mining": "#B8E0D2",
}

CATEGORY_PRIORITIES = {
    "Data": 1,
    "Transform": 2,
    "Visualize": 3,
    "Model": 4,
    "Evaluate": 5,
    "Unsupervised": 6,
    "Text Mining": 7,
}

# Fallback port definitions for widgets that inherit from parent classes
# These are used when AST parsing cannot detect inherited ports
WIDGET_PORT_FALLBACKS = {
    # Visualization widgets (inherit from OWDataProjectionWidget)
    "scatter-plot": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "data_subset", "name": "Data Subset", "type": "Data"},
            {"id": "features", "name": "Features", "type": "AttributeList"}
        ],
        "outputs": [
            {"id": "selected_data", "name": "Selected Data", "type": "Data"},
            {"id": "annotated_data", "name": "Annotated Data", "type": "Data"},
            {"id": "features", "name": "Features", "type": "AttributeList"}
        ]
    },
    "linear-projection": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "data_subset", "name": "Data Subset", "type": "Data"}
        ],
        "outputs": [
            {"id": "selected_data", "name": "Selected Data", "type": "Data"},
            {"id": "annotated_data", "name": "Annotated Data", "type": "Data"},
            {"id": "components", "name": "Components", "type": "Data"}
        ]
    },
    "radviz": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "data_subset", "name": "Data Subset", "type": "Data"}
        ],
        "outputs": [
            {"id": "selected_data", "name": "Selected Data", "type": "Data"},
            {"id": "annotated_data", "name": "Annotated Data", "type": "Data"},
            {"id": "components", "name": "Components", "type": "Data"}
        ]
    },
    "freeviz": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "data_subset", "name": "Data Subset", "type": "Data"}
        ],
        "outputs": [
            {"id": "selected_data", "name": "Selected Data", "type": "Data"},
            {"id": "annotated_data", "name": "Annotated Data", "type": "Data"}
        ]
    },
    "t-sne": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "data_subset", "name": "Data Subset", "type": "Data"}
        ],
        "outputs": [
            {"id": "selected_data", "name": "Selected Data", "type": "Data"},
            {"id": "annotated_data", "name": "Annotated Data", "type": "Data"}
        ]
    },
    "mds": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "distances", "name": "Distances", "type": "DistMatrix"},
            {"id": "data_subset", "name": "Data Subset", "type": "Data"}
        ],
        "outputs": [
            {"id": "selected_data", "name": "Selected Data", "type": "Data"},
            {"id": "annotated_data", "name": "Annotated Data", "type": "Data"}
        ]
    },
    # Model widgets (inherit from OWBaseLearner)
    "knn": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "preprocessor", "name": "Preprocessor", "type": "Preprocessor"}
        ],
        "outputs": [
            {"id": "learner", "name": "Learner", "type": "Learner"},
            {"id": "model", "name": "Model", "type": "Model"}
        ]
    },
    "tree": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "preprocessor", "name": "Preprocessor", "type": "Preprocessor"}
        ],
        "outputs": [
            {"id": "learner", "name": "Learner", "type": "Learner"},
            {"id": "model", "name": "Model", "type": "Model"}
        ]
    },
    "naive-bayes": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "preprocessor", "name": "Preprocessor", "type": "Preprocessor"}
        ],
        "outputs": [
            {"id": "learner", "name": "Learner", "type": "Learner"},
            {"id": "model", "name": "Model", "type": "Model"}
        ]
    },
    "logistic-regression": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "preprocessor", "name": "Preprocessor", "type": "Preprocessor"}
        ],
        "outputs": [
            {"id": "learner", "name": "Learner", "type": "Learner"},
            {"id": "model", "name": "Model", "type": "Model"},
            {"id": "coefficients", "name": "Coefficients", "type": "Data"}
        ]
    },
    "random-forest": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "preprocessor", "name": "Preprocessor", "type": "Preprocessor"}
        ],
        "outputs": [
            {"id": "learner", "name": "Learner", "type": "Learner"},
            {"id": "model", "name": "Model", "type": "Model"}
        ]
    },
    "linear-regression": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "preprocessor", "name": "Preprocessor", "type": "Preprocessor"}
        ],
        "outputs": [
            {"id": "learner", "name": "Learner", "type": "Learner"},
            {"id": "model", "name": "Model", "type": "Model"},
            {"id": "coefficients", "name": "Coefficients", "type": "Data"}
        ]
    },
    "svm": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "preprocessor", "name": "Preprocessor", "type": "Preprocessor"}
        ],
        "outputs": [
            {"id": "learner", "name": "Learner", "type": "Learner"},
            {"id": "model", "name": "Model", "type": "Model"},
            {"id": "support_vectors", "name": "Support Vectors", "type": "Data"}
        ]
    },
    "neural-network": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "preprocessor", "name": "Preprocessor", "type": "Preprocessor"}
        ],
        "outputs": [
            {"id": "learner", "name": "Learner", "type": "Learner"},
            {"id": "model", "name": "Model", "type": "Model"}
        ]
    },
    "gradient-boosting": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "preprocessor", "name": "Preprocessor", "type": "Preprocessor"}
        ],
        "outputs": [
            {"id": "learner", "name": "Learner", "type": "Learner"},
            {"id": "model", "name": "Model", "type": "Model"}
        ]
    },
    "adaboost": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "preprocessor", "name": "Preprocessor", "type": "Preprocessor"},
            {"id": "base_learner", "name": "Base Learner", "type": "Learner"}
        ],
        "outputs": [
            {"id": "learner", "name": "Learner", "type": "Learner"},
            {"id": "model", "name": "Model", "type": "Model"}
        ]
    },
    "stochastic-gradient-descent": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "preprocessor", "name": "Preprocessor", "type": "Preprocessor"}
        ],
        "outputs": [
            {"id": "learner", "name": "Learner", "type": "Learner"},
            {"id": "model", "name": "Model", "type": "Model"}
        ]
    },
    "cn2-rule-induction": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "preprocessor", "name": "Preprocessor", "type": "Preprocessor"}
        ],
        "outputs": [
            {"id": "learner", "name": "Learner", "type": "Learner"},
            {"id": "model", "name": "Model", "type": "Model"}
        ]
    },
    "constant": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "preprocessor", "name": "Preprocessor", "type": "Preprocessor"}
        ],
        "outputs": [
            {"id": "learner", "name": "Learner", "type": "Learner"},
            {"id": "model", "name": "Model", "type": "Model"}
        ]
    },
    "calibrated-learner": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "base_learner", "name": "Base Learner", "type": "Learner"}
        ],
        "outputs": [
            {"id": "learner", "name": "Learner", "type": "Learner"},
            {"id": "model", "name": "Model", "type": "Model"}
        ]
    },
    "stacking": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "learners", "name": "Learners", "type": "Learner", "multiple": True},
            {"id": "aggregate", "name": "Aggregate", "type": "Learner"}
        ],
        "outputs": [
            {"id": "learner", "name": "Learner", "type": "Learner"},
            {"id": "model", "name": "Model", "type": "Model"}
        ]
    },
    "scoring-sheet": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "preprocessor", "name": "Preprocessor", "type": "Preprocessor"}
        ],
        "outputs": [
            {"id": "learner", "name": "Learner", "type": "Learner"},
            {"id": "model", "name": "Model", "type": "Model"}
        ]
    },
    # Text Mining widgets
    "bag-of-words": {
        "inputs": [
            {"id": "corpus", "name": "Corpus", "type": "Corpus"}
        ],
        "outputs": [
            {"id": "bow", "name": "Bag of Words", "type": "Data"}
        ]
    },
    "similarity-hashing": {
        "inputs": [
            {"id": "corpus", "name": "Corpus", "type": "Corpus"}
        ],
        "outputs": [
            {"id": "corpus", "name": "Corpus", "type": "Corpus"}
        ]
    },
    # Multi-input widgets
    "venn-diagram": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data", "multiple": True}
        ],
        "outputs": [
            {"id": "selected_data", "name": "Selected Data", "type": "Data"},
            {"id": "annotated_data", "name": "Annotated Data", "type": "Data"}
        ]
    },
    "python-script": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data", "multiple": True},
            {"id": "learner", "name": "Learner", "type": "Learner", "multiple": True},
            {"id": "classifier", "name": "Classifier", "type": "Model", "multiple": True},
            {"id": "object", "name": "Object", "type": "Object", "multiple": True}
        ],
        "outputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "learner", "name": "Learner", "type": "Learner"},
            {"id": "classifier", "name": "Classifier", "type": "Model"},
            {"id": "object", "name": "Object", "type": "Object"}
        ]
    },
    # PLS widget
    "pls": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "preprocessor", "name": "Preprocessor", "type": "Preprocessor"}
        ],
        "outputs": [
            {"id": "learner", "name": "Learner", "type": "Learner"},
            {"id": "model", "name": "Model", "type": "Model"},
            {"id": "components", "name": "Components", "type": "Data"}
        ]
    },
    # Curve Fit widget
    "curve-fit": {
        "inputs": [
            {"id": "data", "name": "Data", "type": "Data"},
            {"id": "preprocessor", "name": "Preprocessor", "type": "Preprocessor"}
        ],
        "outputs": [
            {"id": "learner", "name": "Learner", "type": "Learner"},
            {"id": "model", "name": "Model", "type": "Model"}
        ]
    },
    # Document Embedding widget
    "document-embedding": {
        "inputs": [
            {"id": "corpus", "name": "Corpus", "type": "Corpus"}
        ],
        "outputs": [
            {"id": "corpus", "name": "Corpus", "type": "Corpus"}
        ]
    },
}


class WidgetDiscovery:
    """Discovers Orange3 widgets from the filesystem using AST parsing."""
    
    WIDGET_NAME_OVERRIDES = {
        "Column Statistics": "Feature Statistics",
    }
    
    def __init__(self, orange3_path: Optional[str] = None, orange3_text_path: Optional[str] = None):
        self.orange3_path = orange3_path or self._find_orange3_path()
        self.orange3_text_path = orange3_text_path or _get_orange3_text_path_from_import()
        self.categories: Dict[str, Dict] = {}
        self.widgets: List[Dict] = []
    
    def _find_orange3_path(self) -> Optional[str]:
        """Find Orange3 installation path."""
        env_path = os.environ.get('ORANGE3_WIDGETS_PATH')
        if env_path and os.path.isdir(env_path):
            return env_path
        
        import_path = _get_orange3_path_from_import()
        if import_path:
            return import_path
        
        for sp in _get_site_packages_paths():
            path = os.path.join(sp, "Orange", "widgets")
            if os.path.isdir(path) and os.path.isdir(os.path.join(path, "data")):
                return path
        
        return None
    
    def discover(self) -> Dict[str, Any]:
        """Discover all widgets and categories."""
        self.categories = {}
        self.widgets = []
        
        if self.orange3_path and os.path.exists(self.orange3_path):
            self._discover_orange3_widgets()
        
        if self.orange3_text_path and os.path.exists(self.orange3_text_path):
            self._discover_text_widgets()
        
        if not self.widgets:
            return {"categories": [], "widgets": [], "total": 0}
        
        return self._format_result()
    
    def _discover_orange3_widgets(self):
        """Discover Orange3 core widgets."""
        widget_dirs = ['data', 'visualize', 'model', 'evaluate', 'unsupervised']
        
        for subdir in widget_dirs:
            dir_path = os.path.join(self.orange3_path, subdir)
            if not os.path.exists(dir_path):
                continue
            
            cat_info = self._read_category_info(dir_path, subdir)
            self._scan_widget_directory(dir_path, cat_info)
    
    def _discover_text_widgets(self):
        """Discover Orange3-Text widgets."""
        cat_info = {
            'name': 'Text Mining',
            'background': CATEGORY_COLORS.get('Text Mining', '#B8E0D2'),
            'priority': CATEGORY_PRIORITIES.get('Text Mining', 7)
        }
        self._scan_widget_directory(self.orange3_text_path, cat_info, icon_prefix='text/')
    
    def _scan_widget_directory(self, dir_path: str, cat_info: Dict, icon_prefix: str = ''):
        """Scan a widget directory."""
        for filename in sorted(os.listdir(dir_path)):
            if filename.startswith('ow') and filename.endswith('.py'):
                filepath = os.path.join(dir_path, filename)
                widget_info = self._extract_widget_info(filepath)
                
                if widget_info and widget_info.get('name'):
                    widget_category = widget_info.get('category') or cat_info['name']
                    
                    if widget_category not in self.categories:
                        self.categories[widget_category] = {
                            'name': widget_category,
                            'background': CATEGORY_COLORS.get(widget_category, cat_info['background']),
                            'priority': CATEGORY_PRIORITIES.get(widget_category, 10),
                            'widgets': []
                        }
                    
                    icon = widget_info.get('icon', 'icons/Unknown.svg')
                    if icon_prefix and not icon.startswith('http'):
                        icon = icon_prefix + icon.replace('icons/', '')
                    
                    display_name = self.WIDGET_NAME_OVERRIDES.get(widget_info['name'], widget_info['name'])
                    widget_id = self._generate_widget_id(widget_info['name'])
                    
                    # Get inputs/outputs from parsing, fallback to predefined if empty
                    parsed_inputs = widget_info.get('inputs', [])
                    parsed_outputs = widget_info.get('outputs', [])
                    
                    # Apply fallback if parsed ports are empty or incomplete
                    if widget_id in WIDGET_PORT_FALLBACKS:
                        fallback = WIDGET_PORT_FALLBACKS[widget_id]
                        if not parsed_inputs or len(parsed_inputs) < len(fallback.get('inputs', [])):
                            parsed_inputs = fallback.get('inputs', parsed_inputs)
                        if not parsed_outputs or len(parsed_outputs) < len(fallback.get('outputs', [])):
                            parsed_outputs = fallback.get('outputs', parsed_outputs)
                    
                    widget_data = {
                        'id': widget_id,
                        'name': display_name,
                        'description': widget_info.get('description', ''),
                        'icon': icon,
                        'category': widget_category,
                        'priority': widget_info.get('priority', 9999),
                        'inputs': parsed_inputs,
                        'outputs': parsed_outputs,
                        'keywords': widget_info.get('keywords', []),
                        'source': 'orange3-text' if icon_prefix else 'orange3',
                    }
                    
                    self.categories[widget_category]['widgets'].append(widget_data)
                    self.widgets.append(widget_data)
    
    def _read_category_info(self, dir_path: str, default_name: str) -> Dict:
        """Read category info from __init__.py."""
        cat_name = default_name.capitalize()
        cat_bg = CATEGORY_COLORS.get(cat_name, '#999999')
        cat_priority = CATEGORY_PRIORITIES.get(cat_name, 10)
        
        init_file = os.path.join(dir_path, '__init__.py')
        if os.path.exists(init_file):
            try:
                with open(init_file, 'r') as f:
                    content = f.read()
                
                name_match = re.search(r'NAME\s*=\s*["\']([^"\']+)["\']', content)
                bg_match = re.search(r'BACKGROUND\s*=\s*["\']([^"\']+)["\']', content)
                priority_match = re.search(r'PRIORITY\s*=\s*(\d+)', content)
                
                if name_match:
                    cat_name = name_match.group(1)
                if bg_match:
                    cat_bg = bg_match.group(1)
                if priority_match:
                    cat_priority = int(priority_match.group(1))
            except Exception:
                pass
        
        return {'name': cat_name, 'background': cat_bg, 'priority': cat_priority}
    
    def _extract_widget_info(self, filepath: str) -> Optional[Dict]:
        """Extract widget info from Python file using AST parsing."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            info = {
                'name': None,
                'description': None,
                'icon': None,
                'category': None,
                'priority': 9999,
                'inputs': [],
                'outputs': [],
                'keywords': [],
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and self._is_widget_class(node):
                    for item in node.body:
                        if isinstance(item, ast.Assign):
                            self._extract_assign(item, info)
                        elif isinstance(item, ast.ClassDef):
                            if item.name == 'Inputs':
                                info['inputs'] = self._extract_io_class(item)
                            elif item.name == 'Outputs':
                                info['outputs'] = self._extract_io_class(item)
                    
                    if info['name']:
                        return info
            
            return None
        except Exception:
            return None
    
    def _is_widget_class(self, node: ast.ClassDef) -> bool:
        """Check if a class is a widget class."""
        for base in node.bases:
            base_name = ''
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            
            if 'Widget' in base_name or base_name.startswith('OW'):
                return True
        return False
    
    def _extract_assign(self, item: ast.Assign, info: Dict):
        """Extract info from simple assignment."""
        for target in item.targets:
            if isinstance(target, ast.Name):
                name = target.id
                value = self._get_constant_value(item.value)
                
                if name == 'name' and value:
                    info['name'] = value
                elif name == 'description' and value:
                    info['description'] = value
                elif name == 'icon' and value:
                    info['icon'] = value
                elif name == 'category' and value:
                    info['category'] = value
                elif name == 'priority' and isinstance(item.value, ast.Constant):
                    if isinstance(item.value.value, (int, float)):
                        info['priority'] = int(item.value.value)
    
    def _extract_io_class(self, class_node: ast.ClassDef) -> List[Dict]:
        """Extract inputs or outputs from nested class."""
        ports = []
        
        for item in class_node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        port_id = target.id
                        port_info = self._parse_io_call(item.value, port_id)
                        if port_info:
                            ports.append(port_info)
        
        return ports
    
    def _parse_io_call(self, node, port_id: str) -> Optional[Dict]:
        """Parse Input(...) or Output(...) call."""
        if not isinstance(node, ast.Call):
            return None
        
        func_name = ''
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        
        if func_name not in ('Input', 'Output'):
            return None
        
        port_name = port_id
        if node.args and len(node.args) >= 1:
            name_val = self._get_constant_value(node.args[0])
            if name_val:
                port_name = name_val
        
        port_type = 'Data'
        if node.args and len(node.args) >= 2:
            type_node = node.args[1]
            if isinstance(type_node, ast.Name):
                port_type = self._simplify_type_name(type_node.id)
            elif isinstance(type_node, ast.Attribute):
                port_type = self._simplify_type_name(type_node.attr)
        
        return {'id': port_id, 'name': port_name, 'type': port_type}
    
    def _simplify_type_name(self, type_name: str) -> str:
        """Simplify Orange3 type names."""
        type_map = {
            'Table': 'Data',
            'Domain': 'Data',
            'Learner': 'Learner',
            'Model': 'Model',
            'DistMatrix': 'Distance',
            'Corpus': 'Corpus',
        }
        return type_map.get(type_name, type_name)
    
    def _get_constant_value(self, node) -> Optional[str]:
        """Get constant value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        if hasattr(ast, 'Str') and isinstance(node, ast.Str):
            return node.s
        
        # Handle Orange3 i18n format
        if isinstance(node, ast.Subscript):
            slice_node = node.slice
            if isinstance(slice_node, ast.Tuple) and len(slice_node.elts) >= 2:
                for elt in slice_node.elts:
                    val = self._get_constant_value(elt)
                    if isinstance(val, str):
                        return val
        
        return None
    
    def _generate_widget_id(self, name: str) -> str:
        """Generate a URL-friendly widget ID from name."""
        widget_id = name.lower()
        widget_id = re.sub(r'[^a-z0-9]+', '-', widget_id)
        return widget_id.strip('-')
    
    def _format_result(self) -> Dict[str, Any]:
        """Format the discovery result."""
        sorted_categories = sorted(
            self.categories.values(),
            key=lambda c: c.get('priority', 10)
        )
        
        formatted_categories = []
        for cat in sorted_categories:
            if not cat['widgets']:
                continue
            
            sorted_widgets = sorted(cat['widgets'], key=lambda w: (w.get('priority', 9999), w['name']))
            
            formatted_categories.append({
                'name': cat['name'],
                'color': cat['background'],
                'priority': cat['priority'],
                'widgets': sorted_widgets
            })
        
        return {
            'categories': formatted_categories,
            'widgets': self.widgets,
            'total': len(self.widgets)
        }


# Singleton instance
_discovery_instance: Optional[WidgetDiscovery] = None


def get_widget_discovery(orange3_path: Optional[str] = None) -> WidgetDiscovery:
    """Get or create the widget discovery instance."""
    global _discovery_instance
    if _discovery_instance is None:
        _discovery_instance = WidgetDiscovery(orange3_path)
    return _discovery_instance


def discover_widgets(orange3_path: Optional[str] = None) -> Dict[str, Any]:
    """Discover all Orange3 widgets."""
    discovery = get_widget_discovery(orange3_path)
    return discovery.discover()


__all__ = [
    # Orange3 availability
    'ORANGE3_AVAILABLE',
    'get_availability',
    # Adapters
    'OrangeRegistryAdapter',
    'OrangeSchemeAdapter',
    # Data classes
    'WebSchemeNode',
    'WebSchemeLink',
    'WebAnnotation',
    # Widget discovery
    'WidgetDiscovery',
    'discover_widgets',
    'get_widget_discovery',
    'CATEGORY_COLORS',
    'CATEGORY_PRIORITIES',
]

