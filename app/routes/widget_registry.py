"""
Widget Registry API Routes.
Handles widget discovery, listing, and compatibility checking.
"""

import logging
import os
import sys
import site
from typing import List, Optional

from fastapi import APIRouter, HTTPException


def _get_orange3_path_from_import() -> Optional[str]:
    """Try to get Orange3 widgets path by importing Orange module."""
    try:
        import Orange
        orange_dir = os.path.dirname(Orange.__file__)
        widgets_path = os.path.join(orange_dir, 'widgets')
        if os.path.exists(widgets_path):
            logger.info(f"Found Orange3 widgets via import: {widgets_path}")
            return widgets_path
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Error finding Orange path: {e}")
    return None


def _get_site_packages_paths() -> List[str]:
    """
    Get all possible site-packages paths dynamically.
    Supports: venv, virtualenv, uv, conda, pyenv, system Python
    """
    paths = []
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    
    # 1. From sys.path (most reliable for any environment)
    for p in sys.path:
        if 'site-packages' in p and os.path.isdir(p):
            paths.append(p)
        elif 'dist-packages' in p and os.path.isdir(p):
            paths.append(p)
    
    # 2. From site module
    try:
        for sp in site.getsitepackages():
            if sp and os.path.isdir(sp):
                paths.append(sp)
    except Exception:
        pass
    
    # 3. User site-packages
    try:
        user_site = site.getusersitepackages()
        if user_site and os.path.isdir(user_site):
            paths.append(user_site)
    except Exception:
        pass
    
    # 4. Virtual environment (venv, virtualenv, uv)
    venv_path = os.environ.get('VIRTUAL_ENV')
    if venv_path:
        linux_path = os.path.join(venv_path, 'lib', py_version, 'site-packages')
        if os.path.isdir(linux_path):
            paths.append(linux_path)
        win_path = os.path.join(venv_path, 'Lib', 'site-packages')
        if os.path.isdir(win_path):
            paths.append(win_path)
    
    # 5. Conda environment
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        conda_sp = os.path.join(conda_prefix, 'lib', py_version, 'site-packages')
        if os.path.isdir(conda_sp):
            paths.append(conda_sp)
    
    # Remove duplicates
    return list(dict.fromkeys(paths))

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/widgets", tags=["Widgets"])

# Check Orange3 availability
try:
    from ..adapters import OrangeRegistryAdapter, ORANGE3_AVAILABLE
    ORANGE_AVAILABLE = ORANGE3_AVAILABLE
except ImportError:
    ORANGE_AVAILABLE = False

# Widget registry reference (set by main.py)
_registry_getter = None

# Widget discovery module
try:
    from ..adapters import discover_widgets
    DISCOVERY_AVAILABLE = True
except ImportError:
    DISCOVERY_AVAILABLE = False
    discover_widgets = None


def set_registry_getter(getter):
    """Set the registry getter function."""
    global _registry_getter
    _registry_getter = getter


# Cache for discovered widgets (refreshed on startup)
_discovered_widgets = None


def get_discovered_widgets():
    """Get or discover widgets from Orange3 installation.
    
    Priority:
    1. Use OrangeRegistryAdapter if available (most complete - 105+ widgets)
    2. Fall back to discover_widgets() if registry not available
    """
    global _discovered_widgets
    if _discovered_widgets is None:
        # Try to use the registry from main.py first (has all 105 widgets)
        if _registry_getter and ORANGE_AVAILABLE:
            registry = _registry_getter()
            if registry:
                try:
                    categories = registry.list_categories()
                    widgets = registry.list_widgets()
                    
                    # Convert to discovery format
                    _discovered_widgets = {
                        "categories": [
                            {
                                "name": cat["name"],
                                "color": cat.get("background", "#808080"),
                                "priority": cat.get("priority", 10),
                                "widgets": [w for w in widgets if w.get("category") == cat["name"]]
                            }
                            for cat in categories
                        ],
                        "widgets": widgets,
                        "total": len(widgets)
                    }
                    logger.info(f"Using OrangeRegistryAdapter: {len(widgets)} widgets in {len(categories)} categories")
                    return _discovered_widgets
                except Exception as e:
                    logger.warning(f"Failed to use registry: {e}")
        
        # Fallback to discover_widgets()
        orange3_path = None
        
        # 1. Environment variable (highest priority)
        env_path = os.environ.get('ORANGE3_WIDGETS_PATH')
        if env_path and os.path.isdir(env_path):
            orange3_path = env_path
            logger.info(f"Using ORANGE3_WIDGETS_PATH: {orange3_path}")
        
        # 2. Try importing Orange module directly (most reliable)
        if not orange3_path:
            orange3_path = _get_orange3_path_from_import()
        
        # 3. Search in site-packages as fallback
        if not orange3_path:
            possible_paths = []
            for sp in _get_site_packages_paths():
                possible_paths.append(os.path.join(sp, "Orange", "widgets"))
            possible_paths.append(os.path.expanduser("~/works/test/orange3/orange3/Orange/widgets"))
            
            for path in possible_paths:
                if os.path.isdir(path) and os.path.isdir(os.path.join(path, "data")):
                    orange3_path = path
                    logger.info(f"Found Orange3 widgets at: {orange3_path}")
                    break
        
        if orange3_path and DISCOVERY_AVAILABLE:
            logger.info(f"Discovering widgets from: {orange3_path}")
            _discovered_widgets = discover_widgets(orange3_path)
            logger.info(f"Discovered {_discovered_widgets.get('total', 0)} widgets")
        else:
            logger.warning("Orange3 widgets path not found or discovery not available")
            _discovered_widgets = {"categories": [], "widgets": [], "total": 0}
    
    return _discovered_widgets


# ============================================================================
# Widget Registry Endpoints
# ============================================================================

@router.get("")
async def list_widgets(category: Optional[str] = None):
    """List all widgets discovered from Orange3 installation."""
    discovered = get_discovered_widgets()
    widgets = discovered.get("widgets", [])
    
    if category:
        widgets = [w for w in widgets if w.get("category") == category]
    
    return widgets


@router.get("/categories")
async def list_categories():
    """List widget categories discovered from Orange3."""
    discovered = get_discovered_widgets()
    categories = discovered.get("categories", [])
    
    # Format for frontend compatibility
    return [
        {
            "name": cat["name"],
            "color": cat.get("color", "#999999"),
            "priority": cat.get("priority", 10),
            "widget_count": len(cat.get("widgets", []))
        }
        for cat in categories
    ]


@router.get("/all")
async def get_all_widgets():
    """Get all categories with their widgets (for frontend toolbox)."""
    discovered = get_discovered_widgets()
    return discovered


@router.get("/{widget_id}")
async def get_widget(widget_id: str):
    """Get widget description by ID."""
    discovered = get_discovered_widgets()
    widgets = discovered.get("widgets", [])
    
    for widget in widgets:
        if widget.get("id") == widget_id:
            return widget
    
    raise HTTPException(status_code=404, detail="Widget not found")


@router.post("/check-compatibility")
async def check_compatibility(source_types: List[str], sink_types: List[str]):
    """Check channel compatibility between source and sink types."""
    # Simple compatibility check (can be enhanced later)
    compatible = any(
        st == sk or st == "Any" or sk == "Any"
        for st in source_types
        for sk in sink_types
    )
    return {"compatible": compatible, "strict": True, "dynamic": False}


@router.post("/refresh")
async def refresh_widgets():
    """Force refresh of widget discovery cache."""
    global _discovered_widgets
    _discovered_widgets = None
    discovered = get_discovered_widgets()
    return {"message": "Widget cache refreshed", "total": discovered.get("total", 0)}


# ============================================================================
# Legacy Endpoint Support
# ============================================================================

async def legacy_widgets_handler():
    """
    Legacy endpoint handler for frontend compatibility.
    Returns widgets grouped by category.
    """
    if _registry_getter:
        registry = _registry_getter()
        if registry and ORANGE_AVAILABLE:
            categories = registry.list_categories()
            widgets = registry.list_widgets()
            
            # Group widgets by category
            result = []
            for cat in categories:
                cat_widgets = [w for w in widgets if w.get("category") == cat["name"]]
                result.append({
                    "name": cat["name"],
                    "color": cat.get("background", "#808080"),
                    "widgets": cat_widgets
                })
            return {"categories": result}
    
    return {"categories": []}

