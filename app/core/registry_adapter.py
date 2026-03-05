"""
OrangeRegistryAdapter — wraps the Orange3 widget registry for web API usage.

Uses Orange3's built-in widget discovery and provides a web-friendly
interface for listing categories, widgets, and checking channel compatibility.
"""

from typing import Dict, List, Optional
import logging
import os

from app.core.orange_models import (
    ORANGE3_AVAILABLE,
    WIDGET_NAME_OVERRIDES,
    CATEGORY_COLORS,
    CATEGORY_PRIORITIES,
    _DEFAULT_ICON_BASE64,
)
from app.core.orange_utils import (
    _read_icon_as_base64,
    _normalize_color,
    _discover_addon_entry_points,
)

logger = logging.getLogger(__name__)


class OrangeRegistryAdapter:
    """Adapter that uses Orange3's widget discovery."""

    WIDGET_NAME_OVERRIDES = WIDGET_NAME_OVERRIDES

    def __init__(self):
        if ORANGE3_AVAILABLE:
            from orangecanvas.registry import WidgetRegistry

            self._registry = WidgetRegistry()
        else:
            self._registry = None
        self._categories: List[Dict] = []
        self._widgets: Dict[str, Dict] = {}
        self._loaded = False

    @property
    def registry(self):
        return self._registry

    def discover_widgets(self) -> None:
        """Discover widgets using Orange3's widget_discovery."""
        if self._loaded:
            return

        try:
            if ORANGE3_AVAILABLE:
                logger.info("Using Orange3 widget discovery...")

                try:
                    from Orange.widgets import widget_discovery
                    from orangewidget.workflow.discovery import WidgetDiscovery

                    discovery = WidgetDiscovery(self._registry)
                    widget_discovery(discovery)
                    self._process_registry()
                except AttributeError as ae:
                    logger.warning(
                        f"WidgetDiscovery API changed, using alternative method: {ae}"
                    )
                    self._alternative_discovery()
            else:
                logger.info("Orange3 not available, using fallback...")
                self._manual_discovery()

            self._loaded = True
            logger.info(
                f"Discovered {len(self._widgets)} widgets "
                f"in {len(self._categories)} categories"
            )

        except Exception as e:
            logger.error(f"Error discovering widgets: {e}")
            import traceback

            traceback.print_exc()
            logger.info("Using manual discovery as fallback...")
            self._manual_discovery()
            self._loaded = True

    def _process_registry(self):
        """Process the registry after discovery."""
        if not self._registry:
            return

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

        for widget in self._registry.widgets():
            widget_dict = self._widget_to_dict(widget)
            self._widgets[widget.qualified_name] = widget_dict

    def _alternative_discovery(self):
        """Alternative widget discovery using dynamic entry_points."""
        try:
            import pkgutil
            import importlib

            core_packages = [
                ("Orange.widgets.data", "Data"),
                ("Orange.widgets.visualize", "Visualize"),
                ("Orange.widgets.model", "Model"),
                ("Orange.widgets.evaluate", "Evaluate"),
                ("Orange.widgets.unsupervised", "Unsupervised"),
            ]

            addon_packages = []
            for addon in _discover_addon_entry_points():
                module_name = None
                if "orangecontrib" in addon["path"]:
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

            discovered_categories: set = set()
            widget_count = 0

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
                f"Alternative discovery found {widget_count} widgets "
                f"in {len(discovered_categories)} categories"
            )

        except Exception as e:
            logger.error(f"Alternative discovery failed: {e}")
            import traceback

            traceback.print_exc()
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
                    logger.debug(f"Skipped {modname}: {e}")
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

            outputs = []
            for out in getattr(widget_class, "outputs", []):
                out_dict = {
                    "id": getattr(out, "name", str(out)),
                    "name": getattr(out, "name", str(out)),
                    "types": [],
                    "flags": 0,
                }
                outputs.append(out_dict)

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
                "icon": icon_base64,
                "priority": getattr(widget_class, "priority", 0),
                "inputs": inputs,
                "outputs": outputs,
                "keywords": list(getattr(widget_class, "keywords", [])),
                "replaces": list(getattr(widget_class, "replaces", [])),
            }
        except Exception as e:
            logger.debug(f"Suppressed error: {e}")
            return None

    def _manual_discovery(self):
        """Manual widget discovery when Orange3 is not available."""
        self._categories = [
            {"name": "Data", "background": "#FFD39F", "priority": 1},
            {"name": "Visualize", "background": "#6FA8DC", "priority": 2},
            {"name": "Model", "background": "#E69138", "priority": 3},
            {"name": "Evaluate", "background": "#93C47D", "priority": 4},
            {"name": "Unsupervised", "background": "#8E7CC3", "priority": 5},
        ]

    def _widget_to_dict(self, widget) -> Dict:
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

        from orangecanvas.registry.description import OutputSignal, InputSignal
        from orangecanvas.scheme.link import compatible_channels

        source = OutputSignal(name="source", types=tuple(source_types))
        sink = InputSignal(name="sink", types=tuple(sink_types))

        compatible = compatible_channels(source, sink)

        return {"compatible": compatible, "strict": compatible, "dynamic": False}


__all__ = [
    "OrangeRegistryAdapter",
]
