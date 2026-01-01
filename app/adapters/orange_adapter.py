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


