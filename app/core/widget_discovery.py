"""
Filesystem-based Orange3 widget discovery using AST parsing.

Contains:
- ``FilesystemWidgetDiscovery`` class
- Singleton accessor ``get_widget_discovery`` / ``discover_widgets``
"""

from typing import Dict, List, Optional, Any
import ast
import logging
import os
import re

from app.core.orange_models import (
    WIDGET_NAME_OVERRIDES,
    CATEGORY_COLORS,
    CATEGORY_PRIORITIES,
    PARENT_CLASS_PORTS,
    CLASS_INHERITANCE_MAP,
    _DEFAULT_ICON_BASE64,
)
from app.core.orange_utils import (
    _read_icon_as_base64,
    _normalize_color,
    _discover_addon_entry_points,
    _get_orange3_path_from_import,
    _get_site_packages_paths,
    _get_dynamic_inheritance,
)

logger = logging.getLogger(__name__)


class FilesystemWidgetDiscovery:
    """Discovers Orange3 widgets from the filesystem using AST parsing.

    Dynamically discovers all installed Orange3 add-ons via entry_points
    and includes widget icons as Base64 data URLs.
    """

    WIDGET_NAME_OVERRIDES = WIDGET_NAME_OVERRIDES

    def __init__(self, orange3_path: Optional[str] = None):
        self.orange3_path = orange3_path or self._find_orange3_path()
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

        if self.orange3_path and os.path.exists(self.orange3_path):
            self._discover_orange3_widgets()

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

                    icon_relative = widget_info.get("icon", "icons/Unknown.svg")
                    icon_full_path = os.path.join(dir_path, icon_relative)
                    icon_base64 = _read_icon_as_base64(icon_full_path)

                    if not icon_base64:
                        icon_base64 = _DEFAULT_ICON_BASE64

                    display_name = self.WIDGET_NAME_OVERRIDES.get(
                        widget_info["name"], widget_info["name"]
                    )
                    widget_id = self._generate_widget_id(widget_info["name"])

                    widget_data = {
                        "id": widget_id,
                        "name": display_name,
                        "description": widget_info.get("description", ""),
                        "icon": icon_base64,
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
        """Apply inherited ports from parent classes."""
        class_name = info.get("class_name", "")
        parent_classes = info.get("parent_classes", [])

        resolved_parents = self._resolve_parent_ports(class_name, parent_classes)

        if not resolved_parents:
            return

        all_inherited_inputs: List[Dict] = []
        all_inherited_outputs: List[Dict] = []
        seen_input_ids: set = set()
        seen_output_ids: set = set()

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
        """Resolve parent class names that have port definitions."""
        resolved: List[str] = []
        checked: set = set()

        dynamic_parents = _get_dynamic_inheritance(class_name)
        if dynamic_parents:
            resolved.extend(dynamic_parents)
            return resolved

        def check_class(name: str):
            if name in checked:
                return
            checked.add(name)

            if name in CLASS_INHERITANCE_MAP:
                mapped = CLASS_INHERITANCE_MAP[name]
                if mapped in PARENT_CLASS_PORTS and mapped not in resolved:
                    resolved.append(mapped)
                check_class(mapped)
                return

            if name in PARENT_CLASS_PORTS and name not in resolved:
                resolved.append(name)

        check_class(class_name)

        for parent in parent_classes:
            check_class(parent)

            for known_parent in PARENT_CLASS_PORTS.keys():
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


# =====================================================================
# Singleton instance
# =====================================================================

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
    "FilesystemWidgetDiscovery",
    "get_widget_discovery",
    "discover_widgets",
]
