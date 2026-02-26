"""
Orange3 Adapter - Wraps existing Orange3 code for web API usage.

This module imports and uses EXISTING Orange3 classes directly,
only adding a minimal web API layer on top.

Orange3 includes orange-canvas-core and orange-widget-base as dependencies,
so we only need to install Orange3.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import ast
import base64
import importlib.metadata
import json
import logging
import os
import re
import site
import sys
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# Module-level constant shared by OrangeRegistryAdapter and FilesystemWidgetDiscovery
WIDGET_NAME_OVERRIDES: Dict[str, str] = {
    "Column Statistics": "Feature Statistics",
}

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
        BaseSchemeAnnotation,
        SchemeTextAnnotation,
        SchemeArrowAnnotation,
    )
    from orangecanvas.scheme.readwrite import scheme_to_ows_stream, scheme_load
    from orangecanvas.registry import WidgetRegistry, WidgetDescription
    from orangecanvas.registry.description import (
        CategoryDescription,
        InputSignal,
        OutputSignal,
    )

    # From orange-widget-base (installed with Orange3)
    from orangewidget.widget import OWBaseWidget
    from orangewidget.settings import Setting, SettingsHandler
    from orangewidget.workflow.discovery import WidgetDiscovery

    ORANGE3_AVAILABLE = True
    logger.info("Orange3 loaded successfully")

except ImportError as e:
    logger.warning(f"Orange3 not available: {e}")
    logger.info("Install with: pip install Orange3")


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
    def registry(self) -> Optional["WidgetRegistry"]:
        return self._registry

    def discover_widgets(self) -> None:
        """Discover widgets using Orange3's widget_discovery."""
        if self._loaded:
            return

        try:
            if ORANGE3_AVAILABLE:
                logger.info("Using Orange3 widget discovery...")

                try:
                    # Try using WidgetDiscovery with widget_discovery function
                    discovery = WidgetDiscovery(self._registry)
                    widget_discovery(discovery)
                    self._process_registry()
                except AttributeError as ae:
                    # Fallback: WidgetDiscovery API changed in newer Orange3 versions
                    logger.warning(
                        f"WidgetDiscovery API changed, using alternative method: {ae}"
                    )
                    self._alternative_discovery()
            else:
                logger.info("Orange3 not available, using fallback...")
                self._manual_discovery()

            self._loaded = True
            logger.info(
                f"Discovered {len(self._widgets)} widgets in {len(self._categories)} categories"
            )

        except Exception as e:
            logger.error(f"Error discovering widgets: {e}")
            import traceback

            traceback.print_exc()
            # Use manual discovery as last resort
            logger.info("Using manual discovery as fallback...")
            self._manual_discovery()
            self._loaded = True

    def _process_registry(self):
        """Process the registry after discovery."""
        if not self._registry:
            return

        # Extract categories
        for cat in self._registry.categories():
            self._categories.append(
                {
                    "name": cat.name,
                    "description": cat.description or "",
                    "background": cat.background or "#808080",
                    "priority": cat.priority or 0,
                    "icon": cat.icon or "",
                }
            )

        # Extract widgets
        for widget in self._registry.widgets():
            widget_dict = self._widget_to_dict(widget)
            self._widgets[widget.qualified_name] = widget_dict

    def _alternative_discovery(self):
        """Alternative widget discovery using dynamic entry_points."""
        try:
            import pkgutil
            import importlib

            # Core Orange3 widget packages
            core_packages = [
                ("Orange.widgets.data", "Data"),
                ("Orange.widgets.visualize", "Visualize"),
                ("Orange.widgets.model", "Model"),
                ("Orange.widgets.evaluate", "Evaluate"),
                ("Orange.widgets.unsupervised", "Unsupervised"),
            ]

            # Dynamically discover add-ons via entry_points
            addon_packages = []
            for addon in _discover_addon_entry_points():
                # Convert path to module name
                module_name = None
                if "orangecontrib" in addon["path"]:
                    # Extract module path from filesystem path
                    parts = addon["path"].split("orangecontrib")
                    if len(parts) > 1:
                        subpath = parts[1].strip(os.sep).replace(os.sep, ".")
                        module_name = f"orangecontrib.{subpath}"

                if module_name:
                    addon_packages.append(
                        (
                            module_name,
                            addon["name"],
                            addon.get("background"),
                            addon.get("priority", 1000),
                        )
                    )

            discovered_categories = set()
            widget_count = 0

            # Process core packages
            for pkg_name, category_name in core_packages:
                try:
                    pkg = importlib.import_module(pkg_name)
                    pkg_path = getattr(pkg, "__path__", None)
                    if not pkg_path:
                        continue

                    cat_info = {
                        "background": CATEGORY_COLORS.get(category_name, "#808080"),
                        "priority": CATEGORY_PRIORITIES.get(category_name, 99),
                    }

                    if category_name not in discovered_categories:
                        self._categories.append(
                            {
                                "name": category_name,
                                "description": f"{category_name} widgets",
                                "background": cat_info["background"],
                                "priority": cat_info["priority"],
                                "icon": "",
                            }
                        )
                        discovered_categories.add(category_name)

                    widget_count += self._scan_package_for_widgets(
                        pkg_name, pkg_path[0], category_name
                    )
                except ImportError:
                    continue

            # Process dynamically discovered add-ons
            for addon_info in addon_packages:
                pkg_name, category_name = addon_info[0], addon_info[1]
                background = addon_info[2] if len(addon_info) > 2 else None
                priority = addon_info[3] if len(addon_info) > 3 else 1000

                try:
                    pkg = importlib.import_module(pkg_name)
                    pkg_path = getattr(pkg, "__path__", None)
                    if not pkg_path:
                        continue

                    if category_name not in discovered_categories:
                        self._categories.append(
                            {
                                "name": category_name,
                                "description": f"{category_name} widgets",
                                "background": _normalize_color(background),
                                "priority": priority,
                                "icon": "",
                            }
                        )
                        discovered_categories.add(category_name)

                    widget_count += self._scan_package_for_widgets(
                        pkg_name, pkg_path[0], category_name
                    )
                except ImportError:
                    continue

            logger.info(
                f"Alternative discovery found {widget_count} widgets in {len(discovered_categories)} categories"
            )

        except Exception as e:
            logger.error(f"Alternative discovery failed: {e}")
            import traceback

            traceback.print_exc()
            # Fall back to manual discovery
            self._manual_discovery()

    def _scan_package_for_widgets(
        self, pkg_name: str, pkg_path: str, category_name: str
    ) -> int:
        """Scan a package for widgets and return count."""
        import pkgutil
        import importlib

        widget_count = 0
        for importer, modname, ispkg in pkgutil.iter_modules([pkg_path]):
            if modname.startswith("ow") or modname.startswith("OW"):
                try:
                    module = importlib.import_module(f"{pkg_name}.{modname}")
                    # Find OWWidget subclasses
                    for name, obj in vars(module).items():
                        if (
                            isinstance(obj, type)
                            and name.startswith("OW")
                            and hasattr(obj, "name")
                            and hasattr(obj, "inputs")
                            and hasattr(obj, "outputs")
                        ):
                            widget_dict = self._class_to_widget_dict(
                                obj, category_name, pkg_name, modname, pkg_path
                            )
                            if widget_dict:
                                self._widgets[widget_dict["qualified_name"]] = (
                                    widget_dict
                                )
                                widget_count += 1
                except Exception as e:
                    logger.debug(f"Skipped {modname}: {e}")  # Skip problematic modules
        return widget_count

    def _class_to_widget_dict(
        self,
        widget_class,
        category_name: str,
        pkg_name: str,
        modname: str,
        pkg_path: str = None,
    ) -> Optional[Dict]:
        """Convert a widget class to widget dict with Base64 icon."""
        try:
            name = getattr(widget_class, "name", widget_class.__name__)
            qualified_name = f"{pkg_name}.{modname}.{widget_class.__name__}"

            # Get inputs
            inputs = []
            for inp in getattr(widget_class, "inputs", []):
                inp_dict = {
                    "id": getattr(inp, "name", str(inp)),
                    "name": getattr(inp, "name", str(inp)),
                    "types": [],
                    "flags": 0,
                    "multiple": False,
                }
                inputs.append(inp_dict)

            # Get outputs
            outputs = []
            for out in getattr(widget_class, "outputs", []):
                out_dict = {
                    "id": getattr(out, "name", str(out)),
                    "name": getattr(out, "name", str(out)),
                    "types": [],
                    "flags": 0,
                }
                outputs.append(out_dict)

            # Resolve icon path and encode as Base64
            icon_relative = getattr(widget_class, "icon", "")
            icon_base64 = _DEFAULT_ICON_BASE64

            if icon_relative and pkg_path:
                icon_full_path = os.path.join(pkg_path, icon_relative)
                encoded = _read_icon_as_base64(icon_full_path)
                if encoded:
                    icon_base64 = encoded

            return {
                "id": qualified_name,
                "qualified_name": qualified_name,
                "name": self.WIDGET_NAME_OVERRIDES.get(name, name),
                "description": getattr(widget_class, "description", "") or "",
                "category": category_name,
                "background": getattr(widget_class, "background", None),
                "icon": icon_base64,  # Base64 data URL
                "priority": getattr(widget_class, "priority", 0),
                "inputs": inputs,
                "outputs": outputs,
                "keywords": list(getattr(widget_class, "keywords", [])),
                "replaces": list(getattr(widget_class, "replaces", [])),
            }
        except Exception as e:
            return None

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

    # 위젯 이름 매핑 (Orange3 원래 이름 -> 표시 이름) — module-level constant
    WIDGET_NAME_OVERRIDES = WIDGET_NAME_OVERRIDES

    def _widget_to_dict(self, widget: "WidgetDescription") -> Dict:
        """Convert WidgetDescription to dict."""
        inputs = []
        for inp in widget.inputs or []:
            inputs.append(
                {
                    "id": inp.name,
                    "name": inp.name,
                    "types": list(inp.types) if inp.types else [],
                    "flags": inp.flags if hasattr(inp, "flags") else 0,
                    "multiple": getattr(inp, "single", 1) == 0,
                }
            )

        outputs = []
        for out in widget.outputs or []:
            outputs.append(
                {
                    "id": out.name,
                    "name": out.name,
                    "types": list(out.types) if out.types else [],
                    "flags": out.flags if hasattr(out, "flags") else 0,
                    "dynamic": getattr(out, "dynamic", False),
                }
            )

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
            "background": widget.background or "",
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
        self, source_types: List[str], sink_types: List[str]
    ) -> Dict:
        """Check if channels are compatible using existing function."""
        if not ORANGE3_AVAILABLE:
            return {"compatible": True, "strict": True, "dynamic": False}

        source = OutputSignal(name="source", types=tuple(source_types))
        sink = InputSignal(name="sink", types=tuple(sink_types))

        compatible = compatible_channels(source, sink)

        return {"compatible": compatible, "strict": compatible, "dynamic": False}


# =============================================================================
# Orange3 Scheme Adapter (workflow management)
# =============================================================================


class OrangeSchemeAdapter:
    """
    Adapter that wraps the existing Scheme class for web API usage.
    """

    def __init__(self, registry: "WidgetRegistry" = None):
        if not ORANGE3_AVAILABLE:
            raise ImportError("Orange3 is required")

        self._scheme: "Scheme" = Scheme()
        self._registry = registry
        self._node_map: Dict[str, "SchemeNode"] = {}
        self._link_map: Dict[str, "SchemeLink"] = {}
        self._annotation_map: Dict[str, "BaseSchemeAnnotation"] = {}

    @property
    def scheme(self) -> "Scheme":
        return self._scheme

    def get_workflow_dict(self) -> Dict:
        """Convert scheme to web-friendly dictionary."""
        # Build reverse lookup: Python object id -> stable UUID
        # Reuse any UUID already registered for this object; assign a new one otherwise.
        node_id_map: Dict[int, str] = {}
        nodes = []
        for node in self._scheme.nodes:
            existing_id = next(
                (k for k, v in self._node_map.items() if v is node), None
            )
            web_id = existing_id if existing_id is not None else str(uuid.uuid4())
            if web_id not in self._node_map:
                self._node_map[web_id] = node
            node_id_map[id(node)] = web_id
            nodes.append(WebSchemeNode.from_scheme_node(node, web_id).to_dict())

        links = []
        for link in self._scheme.links:
            existing_id = next(
                (k for k, v in self._link_map.items() if v is link), None
            )
            web_id = existing_id if existing_id is not None else str(uuid.uuid4())
            if web_id not in self._link_map:
                self._link_map[web_id] = link
            links.append(
                WebSchemeLink.from_scheme_link(link, web_id, node_id_map).to_dict()
            )

        annotations = []
        for annotation in self._scheme.annotations:
            existing_id = next(
                (k for k, v in self._annotation_map.items() if v is annotation), None
            )
            web_id = existing_id if existing_id is not None else str(uuid.uuid4())
            if web_id not in self._annotation_map:
                self._annotation_map[web_id] = annotation
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

        widget_desc = self._registry.widget(widget_id)
        if not widget_desc:
            raise ValueError(f"Widget not found: {widget_id}")

        node = SchemeNode(description=widget_desc, title=title, position=position)

        self._scheme.add_node(node)

        web_id = str(uuid.uuid4())
        self._node_map[web_id] = node
        return WebSchemeNode.from_scheme_node(node, web_id).to_dict()

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
        sink_channel: str,
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
            sink_channel=sink_ch,
        )

        try:
            self._scheme.add_link(link)
        except Exception as e:
            logger.error(f"Error adding link: {e}")
            return None

        web_id = str(uuid.uuid4())
        self._link_map[web_id] = link

        # Build node_id_map using stable UUIDs already in _node_map
        node_id_map = {id(n): k for k, n in self._node_map.items()}
        return WebSchemeLink.from_scheme_link(link, web_id, node_id_map).to_dict()

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
        content_type: str = "text/plain",
    ) -> Dict:
        """Add text annotation."""
        annotation = SchemeTextAnnotation(rect=rect, text=content)
        self._scheme.add_annotation(annotation)

        web_id = str(uuid.uuid4())
        self._annotation_map[web_id] = annotation
        return WebAnnotation.from_scheme_annotation(annotation, web_id).to_dict()

    def add_arrow_annotation(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        color: str = "#808080",
    ) -> Dict:
        """Add arrow annotation."""
        annotation = SchemeArrowAnnotation(
            start_pos=start_pos, end_pos=end_pos, color=color
        )
        self._scheme.add_annotation(annotation)

        web_id = str(uuid.uuid4())
        self._annotation_map[web_id] = annotation
        return WebAnnotation.from_scheme_annotation(annotation, web_id).to_dict()

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
        return stream.getvalue().decode("utf-8")

    def import_from_ows(self, ows_content: str) -> bool:
        """Import scheme from OWS format."""
        import io

        try:
            stream = io.BytesIO(ows_content.encode("utf-8"))
            self._scheme = scheme_load(stream, registry=self._registry)
            self._node_map = {str(uuid.uuid4()): n for n in self._scheme.nodes}
            self._link_map = {str(uuid.uuid4()): l for l in self._scheme.links}
            self._annotation_map = {
                str(uuid.uuid4()): a for a in self._scheme.annotations
            }
            return True
        except Exception as e:
            logger.error(f"Error importing OWS: {e}")
            return False


# =============================================================================
# Availability check
# =============================================================================


def get_availability() -> Dict[str, bool]:
    """Check if Orange3 is available."""
    return {"orange3": ORANGE3_AVAILABLE}


# =============================================================================
# Widget Discovery
# Automatically discovers Orange3 widgets by scanning widget directories
# and parsing Python files using AST (no import required).
# =============================================================================


def _get_orange3_path_from_import() -> Optional[str]:
    """Try to get Orange3 widgets path by importing Orange module."""
    try:
        import Orange

        orange_dir = os.path.dirname(Orange.__file__)
        widgets_path = os.path.join(orange_dir, "widgets")
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
        if "site-packages" in p and os.path.isdir(p):
            paths.append(p)

    try:
        for sp in site.getsitepackages():
            if sp and os.path.isdir(sp):
                paths.append(sp)
    except Exception as e:
        logger.debug(f"Skipped site.getsitepackages(): {e}")

    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path:
        linux_path = os.path.join(venv_path, "lib", py_version, "site-packages")
        if os.path.isdir(linux_path):
            paths.append(linux_path)

    return list(dict.fromkeys(paths))


# =============================================================================
# Dynamic Add-on Discovery and Icon Utilities
# =============================================================================


def _read_icon_as_base64(icon_path: str) -> Optional[str]:
    """Read SVG/PNG icon and return as Base64 data URL."""
    if not icon_path or not os.path.exists(icon_path):
        return None
    try:
        with open(icon_path, "rb") as f:
            content = f.read()
        b64 = base64.b64encode(content).decode("utf-8")

        # Determine MIME type
        if icon_path.lower().endswith(".svg"):
            mime_type = "image/svg+xml"
        elif icon_path.lower().endswith(".png"):
            mime_type = "image/png"
        else:
            mime_type = "image/svg+xml"  # Default to SVG

        return f"data:{mime_type};base64,{b64}"
    except Exception as e:
        logger.debug(f"Skipped {icon_path}: {e}")
        return None


def _normalize_color(color: Optional[str]) -> str:
    """Normalize color value to hex format."""
    if not color:
        return "#999999"

    # Handle named colors
    color_map = {
        "light-blue": "#B8E0D2",
        "lightblue": "#B8E0D2",
        "light-green": "#93C47D",
        "lightgreen": "#93C47D",
        "light-yellow": "#F7F5A8",
        "lightyellow": "#F7F5A8",
    }

    color_lower = color.lower().strip()
    if color_lower in color_map:
        return color_map[color_lower]

    # Already a hex color
    if color.startswith("#"):
        return color

    return "#999999"


def _discover_addon_entry_points() -> List[Dict]:
    """Dynamically discover all Orange3 add-ons via entry_points."""
    addons = []
    try:
        eps = importlib.metadata.entry_points(group="orange.widgets")
        for ep in eps:
            # Skip core Orange3 widgets (handled separately)
            if ep.name == "Orange Widgets":
                continue
            try:
                module = importlib.import_module(ep.value)
                module_path = (
                    os.path.dirname(module.__file__)
                    if hasattr(module, "__file__")
                    else None
                )

                if module_path:
                    # Extract source name from module path (e.g., 'text' from 'orangecontrib.text.widgets')
                    parts = ep.value.split(".")
                    source = (
                        parts[1] if len(parts) > 1 else ep.name.lower().replace(" ", "")
                    )

                    addons.append(
                        {
                            "name": getattr(module, "NAME", ep.name),
                            "path": module_path,
                            "background": getattr(module, "BACKGROUND", None),
                            "priority": getattr(module, "PRIORITY", 1000),
                            "source": source,
                            "entry_point_name": ep.name,
                        }
                    )
            except ImportError as e:
                logger.warning(f"Could not import add-on {ep.name}: {e}")
    except Exception as e:
        logger.warning(f"Error discovering add-ons: {e}")

    return addons


# Default icon as Base64 (simple gray circle SVG)
_DEFAULT_ICON_BASE64 = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0OCA0OCI+PGNpcmNsZSBjeD0iMjQiIGN5PSIyNCIgcj0iMjAiIGZpbGw9IiM5OTkiLz48L3N2Zz4="


# Category colors and priorities (for core Orange3 widgets)
# Add-on categories are discovered dynamically via entry_points
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

# Parent class port definitions - automatically inherited by child widgets
# Orange3 widgets inherit Inputs/Outputs from parent classes
PARENT_CLASS_PORTS = {
    # Base learner widgets (OWBaseLearner, OWProvidesLearner)
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
    # Data projection widgets (Scatter Plot, t-SNE, MDS, etc.)
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
    # Anchor projection widgets (Radviz, FreeViz, Linear Projection)
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
    # Text corpus widgets
    "OWTextBaseWidget": {
        "inputs": [{"id": "corpus", "name": "Corpus", "type": "Corpus"}],
        "outputs": [{"id": "corpus", "name": "Corpus", "type": "Corpus"}],
    },
    # Text vectorizer widgets (Bag of Words, etc.)
    "OWBaseVectorizer": {
        "inputs": [{"id": "corpus", "name": "Corpus", "type": "Corpus"}],
        "outputs": [{"id": "corpus", "name": "Corpus", "type": "Corpus"}],
    },
}

# Static mapping from class name patterns to parent class for inheritance resolution
# This is a fallback when dynamic resolution fails
CLASS_INHERITANCE_MAP = {
    # Add explicit mappings only for special cases
    # Most widgets are auto-detected via dynamic class inspection
}

# Dynamic class inheritance cache (populated at runtime)
_DYNAMIC_INHERITANCE_CACHE: Dict[str, List[str]] = {}


def _get_dynamic_inheritance(class_name: str) -> List[str]:
    """Get inheritance chain for a widget class by importing it at runtime.

    Returns list of parent class names that exist in PARENT_CLASS_PORTS.
    """
    if class_name in _DYNAMIC_INHERITANCE_CACHE:
        return _DYNAMIC_INHERITANCE_CACHE[class_name]

    result = []

    # Try to import the widget class and inspect its MRO
    try:
        # Try Orange3 core widgets
        import importlib

        # Common Orange3 widget module patterns
        module_patterns = [
            f"Orange.widgets.model.ow{class_name[2:].lower()}",
            f"Orange.widgets.data.ow{class_name[2:].lower()}",
            f"Orange.widgets.visualize.ow{class_name[2:].lower()}",
            f"Orange.widgets.evaluate.ow{class_name[2:].lower()}",
            f"Orange.widgets.unsupervised.ow{class_name[2:].lower()}",
        ]

        widget_class = None
        for pattern in module_patterns:
            try:
                module = importlib.import_module(pattern)
                if hasattr(module, class_name):
                    widget_class = getattr(module, class_name)
                    break
            except (ImportError, ModuleNotFoundError):
                continue

        if widget_class:
            # Get MRO and find matching parent classes
            for parent in widget_class.__mro__:
                parent_name = parent.__name__
                if parent_name in PARENT_CLASS_PORTS:
                    result.append(parent_name)
    except Exception as e:
        logger.debug(f"Skipped {class_name}: {e}")

    _DYNAMIC_INHERITANCE_CACHE[class_name] = result
    return result


class FilesystemWidgetDiscovery:
    """Discovers Orange3 widgets from the filesystem using AST parsing.

    Dynamically discovers all installed Orange3 add-ons via entry_points
    and includes widget icons as Base64 data URLs.
    """

    # 위젯 이름 매핑 — module-level constant
    WIDGET_NAME_OVERRIDES = WIDGET_NAME_OVERRIDES

    def __init__(self, orange3_path: Optional[str] = None):
        self.orange3_path = orange3_path or self._find_orange3_path()
        # Dynamically discover all installed add-ons
        self.addon_paths = _discover_addon_entry_points()
        self.categories: Dict[str, Dict] = {}
        self.widgets: List[Dict] = []

    def _find_orange3_path(self) -> Optional[str]:
        """Find Orange3 installation path."""
        env_path = os.environ.get("ORANGE3_WIDGETS_PATH")
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

        # Discover core Orange3 widgets
        if self.orange3_path and os.path.exists(self.orange3_path):
            self._discover_orange3_widgets()

        # Dynamically discover all installed add-ons
        for addon in self.addon_paths:
            if os.path.exists(addon["path"]):
                self._discover_addon_widgets(addon)

        if not self.widgets:
            return {"categories": [], "widgets": [], "total": 0}

        return self._format_result()

    def _discover_orange3_widgets(self):
        """Discover Orange3 core widgets."""
        widget_dirs = ["data", "visualize", "model", "evaluate", "unsupervised"]

        for subdir in widget_dirs:
            dir_path = os.path.join(self.orange3_path, subdir)
            if not os.path.exists(dir_path):
                continue

            cat_info = self._read_category_info(dir_path, subdir)
            self._scan_widget_directory(dir_path, cat_info, source="orange3")

    def _discover_addon_widgets(self, addon: Dict):
        """Discover widgets from a dynamically discovered add-on."""
        cat_info = {
            "name": addon["name"],
            "background": _normalize_color(addon.get("background"))
            or CATEGORY_COLORS.get(addon["name"], "#999999"),
            "priority": addon.get("priority", 1000),
        }
        self._scan_widget_directory(addon["path"], cat_info, source=addon["source"])

    def _scan_widget_directory(
        self, dir_path: str, cat_info: Dict, source: str = "orange3"
    ):
        """Scan a widget directory and extract widgets with Base64 icons."""
        for filename in sorted(os.listdir(dir_path)):
            if filename.startswith("ow") and filename.endswith(".py"):
                filepath = os.path.join(dir_path, filename)
                widget_info = self._extract_widget_info(filepath)

                if widget_info and widget_info.get("name"):
                    widget_category = widget_info.get("category") or cat_info["name"]

                    if widget_category not in self.categories:
                        self.categories[widget_category] = {
                            "name": widget_category,
                            "background": CATEGORY_COLORS.get(
                                widget_category, cat_info["background"]
                            ),
                            "priority": CATEGORY_PRIORITIES.get(
                                widget_category, cat_info.get("priority", 10)
                            ),
                            "widgets": [],
                        }

                    # Resolve icon path and encode as Base64
                    icon_relative = widget_info.get("icon", "icons/Unknown.svg")
                    icon_full_path = os.path.join(dir_path, icon_relative)
                    icon_base64 = _read_icon_as_base64(icon_full_path)

                    # Use default icon if not found
                    if not icon_base64:
                        icon_base64 = _DEFAULT_ICON_BASE64

                    display_name = self.WIDGET_NAME_OVERRIDES.get(
                        widget_info["name"], widget_info["name"]
                    )
                    widget_id = self._generate_widget_id(widget_info["name"])

                    # Ports are already merged with inherited ports in _extract_widget_info
                    widget_data = {
                        "id": widget_id,
                        "name": display_name,
                        "description": widget_info.get("description", ""),
                        "icon": icon_base64,  # Base64 data URL
                        "category": widget_category,
                        "priority": widget_info.get("priority", 9999),
                        "inputs": widget_info.get("inputs", []),
                        "outputs": widget_info.get("outputs", []),
                        "keywords": widget_info.get("keywords", []),
                        "source": source,
                    }

                    self.categories[widget_category]["widgets"].append(widget_data)
                    self.widgets.append(widget_data)

    def _read_category_info(self, dir_path: str, default_name: str) -> Dict:
        """Read category info from __init__.py."""
        cat_name = default_name.capitalize()
        cat_bg = CATEGORY_COLORS.get(cat_name, "#999999")
        cat_priority = CATEGORY_PRIORITIES.get(cat_name, 10)

        init_file = os.path.join(dir_path, "__init__.py")
        if os.path.exists(init_file):
            try:
                with open(init_file, "r") as f:
                    content = f.read()

                name_match = re.search(r'NAME\s*=\s*["\']([^"\']+)["\']', content)
                bg_match = re.search(r'BACKGROUND\s*=\s*["\']([^"\']+)["\']', content)
                priority_match = re.search(r"PRIORITY\s*=\s*(\d+)", content)

                if name_match:
                    cat_name = name_match.group(1)
                if bg_match:
                    cat_bg = bg_match.group(1)
                if priority_match:
                    cat_priority = int(priority_match.group(1))
            except Exception as e:
                logger.debug(f"Skipped {init_file}: {e}")

        return {"name": cat_name, "background": cat_bg, "priority": cat_priority}

    def _extract_widget_info(self, filepath: str) -> Optional[Dict]:
        """Extract widget info from Python file using AST parsing."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            info = {
                "name": None,
                "description": None,
                "icon": None,
                "category": None,
                "priority": 9999,
                "inputs": [],
                "outputs": [],
                "keywords": [],
                "class_name": None,
                "parent_classes": [],
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and self._is_widget_class(node):
                    # Extract class name and parent classes
                    info["class_name"] = node.name
                    info["parent_classes"] = self._extract_parent_classes(node)

                    for item in node.body:
                        if isinstance(item, ast.Assign):
                            self._extract_assign(item, info)
                        elif isinstance(item, ast.ClassDef):
                            if item.name == "Inputs":
                                info["inputs"] = self._extract_io_class(item)
                            elif item.name == "Outputs":
                                info["outputs"] = self._extract_io_class(item)

                    if info["name"]:
                        # Apply inherited ports from parent classes
                        self._apply_inherited_ports(info)
                        return info

            return None
        except Exception as e:
            logger.debug(f"Skipped {filepath}: {e}")
            return None

    def _extract_parent_classes(self, node: ast.ClassDef) -> List[str]:
        """Extract parent class names from a class definition."""
        parents = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                parents.append(base.id)
            elif isinstance(base, ast.Attribute):
                parents.append(base.attr)
        return parents

    def _apply_inherited_ports(self, info: Dict):
        """Apply inherited ports from parent classes.

        Automatically resolves parent class ports by:
        1. Checking CLASS_INHERITANCE_MAP for explicit mappings
        2. Directly matching parent class names in PARENT_CLASS_PORTS
        3. Recursively checking parent class hierarchy
        """
        class_name = info.get("class_name", "")
        parent_classes = info.get("parent_classes", [])

        # Find all matching parent port definitions
        resolved_parents = self._resolve_parent_ports(class_name, parent_classes)

        if not resolved_parents:
            return

        # Merge all inherited ports (earlier in list = higher priority)
        all_inherited_inputs = []
        all_inherited_outputs = []
        seen_input_ids = set()
        seen_output_ids = set()

        for parent_name in resolved_parents:
            parent_ports = PARENT_CLASS_PORTS.get(parent_name, {})
            for inp in parent_ports.get("inputs", []):
                if inp["id"] not in seen_input_ids:
                    all_inherited_inputs.append(inp.copy())
                    seen_input_ids.add(inp["id"])
            for out in parent_ports.get("outputs", []):
                if out["id"] not in seen_output_ids:
                    all_inherited_outputs.append(out.copy())
                    seen_output_ids.add(out["id"])

        # Merge with widget's own ports (inherited first, then own)
        existing_input_ids = {p["id"] for p in info["inputs"]}
        merged_inputs = [
            p for p in all_inherited_inputs if p["id"] not in existing_input_ids
        ]
        merged_inputs.extend(info["inputs"])
        info["inputs"] = merged_inputs

        existing_output_ids = {p["id"] for p in info["outputs"]}
        merged_outputs = [
            p for p in all_inherited_outputs if p["id"] not in existing_output_ids
        ]
        merged_outputs.extend(info["outputs"])
        info["outputs"] = merged_outputs

    def _resolve_parent_ports(
        self, class_name: str, parent_classes: List[str]
    ) -> List[str]:
        """Resolve parent class names that have port definitions.

        Uses multiple strategies:
        1. Dynamic class inspection (imports actual class and checks MRO)
        2. Explicit CLASS_INHERITANCE_MAP
        3. Direct parent class matching in PARENT_CLASS_PORTS
        4. Pattern matching on parent class names

        Returns a list of parent class names that exist in PARENT_CLASS_PORTS,
        in order of inheritance priority.
        """
        resolved = []
        checked = set()

        # Try dynamic inheritance first (most accurate)
        dynamic_parents = _get_dynamic_inheritance(class_name)
        if dynamic_parents:
            resolved.extend(dynamic_parents)
            return resolved

        def check_class(name: str):
            if name in checked:
                return
            checked.add(name)

            # Check explicit mapping first
            if name in CLASS_INHERITANCE_MAP:
                mapped = CLASS_INHERITANCE_MAP[name]
                if mapped in PARENT_CLASS_PORTS and mapped not in resolved:
                    resolved.append(mapped)
                check_class(mapped)
                return

            # Check if directly in PARENT_CLASS_PORTS
            if name in PARENT_CLASS_PORTS and name not in resolved:
                resolved.append(name)

        # Check the widget class itself
        check_class(class_name)

        # Check all parent classes from AST
        for parent in parent_classes:
            check_class(parent)

            # Pattern matching: find matching parent in PARENT_CLASS_PORTS
            for known_parent in PARENT_CLASS_PORTS.keys():
                # Match patterns like "BaseLearner" in "OWBaseLearner"
                base_name = known_parent.replace("OW", "")
                if base_name in parent and known_parent not in resolved:
                    resolved.append(known_parent)

        return resolved

    def _is_widget_class(self, node: ast.ClassDef) -> bool:
        """Check if a class is a widget class."""
        for base in node.bases:
            base_name = ""
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr

            if "Widget" in base_name or base_name.startswith("OW"):
                return True
        return False

    def _extract_assign(self, item: ast.Assign, info: Dict):
        """Extract info from simple assignment."""
        for target in item.targets:
            if isinstance(target, ast.Name):
                name = target.id
                value = self._get_constant_value(item.value)

                if name == "name" and value:
                    info["name"] = value
                elif name == "description" and value:
                    info["description"] = value
                elif name == "icon" and value:
                    info["icon"] = value
                elif name == "category" and value:
                    info["category"] = value
                elif name == "priority" and isinstance(item.value, ast.Constant):
                    if isinstance(item.value.value, (int, float)):
                        info["priority"] = int(item.value.value)

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
        """Parse Input(...), Output(...), or MultiInput(...) call."""
        if not isinstance(node, ast.Call):
            return None

        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name not in ("Input", "Output", "MultiInput"):
            return None

        port_name = port_id
        if node.args and len(node.args) >= 1:
            name_val = self._get_constant_value(node.args[0])
            if name_val:
                port_name = name_val

        port_type = "Data"
        if node.args and len(node.args) >= 2:
            type_node = node.args[1]
            if isinstance(type_node, ast.Name):
                port_type = self._simplify_type_name(type_node.id)
            elif isinstance(type_node, ast.Attribute):
                port_type = self._simplify_type_name(type_node.attr)

        result = {"id": port_id, "name": port_name, "type": port_type}

        # Mark as multiple if it's a MultiInput
        if func_name == "MultiInput":
            result["multiple"] = True

        return result

    def _simplify_type_name(self, type_name: str) -> str:
        """Simplify Orange3 type names."""
        type_map = {
            "Table": "Data",
            "Domain": "Data",
            "Learner": "Learner",
            "Model": "Model",
            "DistMatrix": "Distance",
            "Corpus": "Corpus",
        }
        return type_map.get(type_name, type_name)

    def _get_constant_value(self, node) -> Optional[str]:
        """Get constant value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        if hasattr(ast, "Str") and isinstance(node, ast.Str):
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
        widget_id = re.sub(r"[^a-z0-9]+", "-", widget_id)
        return widget_id.strip("-")

    def _format_result(self) -> Dict[str, Any]:
        """Format the discovery result."""
        sorted_categories = sorted(
            self.categories.values(), key=lambda c: c.get("priority", 10)
        )

        formatted_categories = []
        for cat in sorted_categories:
            if not cat["widgets"]:
                continue

            sorted_widgets = sorted(
                cat["widgets"], key=lambda w: (w.get("priority", 9999), w["name"])
            )

            formatted_categories.append(
                {
                    "name": cat["name"],
                    "color": cat["background"],
                    "priority": cat["priority"],
                    "widgets": sorted_widgets,
                }
            )

        return {
            "categories": formatted_categories,
            "widgets": self.widgets,
            "total": len(self.widgets),
        }


# Singleton instance
_discovery_instance: Optional[FilesystemWidgetDiscovery] = None


def get_widget_discovery(
    orange3_path: Optional[str] = None,
) -> FilesystemWidgetDiscovery:
    """Get or create the widget discovery instance."""
    global _discovery_instance
    if _discovery_instance is None:
        _discovery_instance = FilesystemWidgetDiscovery(orange3_path)
    return _discovery_instance


def discover_widgets(orange3_path: Optional[str] = None) -> Dict[str, Any]:
    """Discover all Orange3 widgets."""
    discovery = get_widget_discovery(orange3_path)
    return discovery.discover()


__all__ = [
    # Orange3 availability
    "ORANGE3_AVAILABLE",
    "get_availability",
    # Adapters
    "OrangeRegistryAdapter",
    "OrangeSchemeAdapter",
    # Data classes
    "WebSchemeNode",
    "WebSchemeLink",
    "WebAnnotation",
    # Widget discovery
    "FilesystemWidgetDiscovery",
    "discover_widgets",
    "get_widget_discovery",
    "CATEGORY_COLORS",
    "CATEGORY_PRIORITIES",
]
