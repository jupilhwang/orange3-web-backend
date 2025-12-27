"""
Widget Registry API Routes.
Handles widget discovery, listing, and compatibility checking.
"""

import logging
import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException

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
    """Get or discover widgets from Orange3 installation."""
    global _discovered_widgets
    if _discovered_widgets is None:
        # Try to find Orange3 widgets path
        possible_paths = [
            os.path.expanduser("~/works/test/orange3/orange3/Orange/widgets"),
            os.path.join(os.path.dirname(__file__), "..", "..", ".venv", "lib", "python3.11", "site-packages", "Orange", "widgets"),
        ]
        
        orange3_path = None
        for path in possible_paths:
            if os.path.exists(path):
                orange3_path = path
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

