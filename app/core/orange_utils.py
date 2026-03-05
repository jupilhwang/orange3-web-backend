"""
Orange3 utility functions shared across adapter and discovery modules.

Contains:
- Icon reading and Base64 encoding
- Color normalization
- Add-on entry-point discovery
- Orange3 path resolution helpers
- Dynamic class inheritance resolution
"""

from typing import Dict, List, Optional
import base64
import importlib.metadata
import logging
import os
import site
import sys

from app.core.orange_models import (
    PARENT_CLASS_PORTS,
    CLASS_INHERITANCE_MAP,
    _DYNAMIC_INHERITANCE_CACHE,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Icon utilities
# =====================================================================


def _read_icon_as_base64(icon_path: str) -> Optional[str]:
    """Read SVG/PNG icon and return as Base64 data URL."""
    if not icon_path or not os.path.exists(icon_path):
        return None
    try:
        with open(icon_path, "rb") as f:
            content = f.read()
        b64 = base64.b64encode(content).decode("utf-8")

        if icon_path.lower().endswith(".svg"):
            mime_type = "image/svg+xml"
        elif icon_path.lower().endswith(".png"):
            mime_type = "image/png"
        else:
            mime_type = "image/svg+xml"

        return f"data:{mime_type};base64,{b64}"
    except Exception as e:
        logger.debug(f"Skipped {icon_path}: {e}")
        return None


# =====================================================================
# Color utilities
# =====================================================================


def _normalize_color(color: Optional[str]) -> str:
    """Normalize color value to hex format."""
    if not color:
        return "#999999"

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

    if color.startswith("#"):
        return color

    return "#999999"


# =====================================================================
# Add-on discovery
# =====================================================================


def _discover_addon_entry_points() -> List[Dict]:
    """Dynamically discover all Orange3 add-ons via entry_points."""
    addons = []
    try:
        eps = importlib.metadata.entry_points(group="orange.widgets")
        for ep in eps:
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


# =====================================================================
# Orange3 path helpers
# =====================================================================


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


# =====================================================================
# Dynamic inheritance resolution
# =====================================================================


def _get_dynamic_inheritance(class_name: str) -> List[str]:
    """Get inheritance chain for a widget class by importing it at runtime.

    Returns list of parent class names that exist in PARENT_CLASS_PORTS.
    """
    if class_name in _DYNAMIC_INHERITANCE_CACHE:
        return _DYNAMIC_INHERITANCE_CACHE[class_name]

    result: List[str] = []

    try:
        import importlib as _importlib

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
                module = _importlib.import_module(pattern)
                if hasattr(module, class_name):
                    widget_class = getattr(module, class_name)
                    break
            except (ImportError, ModuleNotFoundError):
                continue

        if widget_class:
            for parent in widget_class.__mro__:
                parent_name = parent.__name__
                if parent_name in PARENT_CLASS_PORTS:
                    result.append(parent_name)
    except Exception as e:
        logger.debug(f"Skipped {class_name}: {e}")

    _DYNAMIC_INHERITANCE_CACHE[class_name] = result
    return result


__all__ = [
    "_read_icon_as_base64",
    "_normalize_color",
    "_discover_addon_entry_points",
    "_get_orange3_path_from_import",
    "_get_site_packages_paths",
    "_get_dynamic_inheritance",
]
