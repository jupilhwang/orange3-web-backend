"""
Geo Widget API endpoints.
Geographic data visualization with lat/lon coordinates.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/geo", tags=["Geo"])

from app.core.orange_compat import ORANGE_AVAILABLE


class GeoMapRequest(BaseModel):
    """Request model for geo map data."""

    data_path: str
    lat_column: Optional[str] = None
    lon_column: Optional[str] = None
    color_column: Optional[str] = None
    size_column: Optional[str] = None
    tooltip_columns: Optional[List[str]] = None
    selected_indices: Optional[List[int]] = None


# Color palette for categorical values (matches scatter_plot.py)
_COLORS = [
    "#5dade2",
    "#e74c3c",
    "#82e0aa",
    "#f39c12",
    "#9b59b6",
    "#1abc9c",
    "#e67e22",
    "#3498db",
]


def _find_lat_lon_columns(variables: list) -> tuple:
    """Heuristically detect latitude/longitude columns from variable names."""
    lat_candidates = ["lat", "latitude", "y", "lat_", "latitude_"]
    lon_candidates = ["lon", "longitude", "lng", "x", "lon_", "lng_", "longitude_"]

    lat_var = None
    lon_var = None

    numeric_vars = [v for v in variables if v["type"] == "numeric"]

    for var in numeric_vars:
        name_lower = var["name"].lower()
        if any(name_lower == c or name_lower.startswith(c) for c in lat_candidates):
            lat_var = var["name"]
            break

    for var in numeric_vars:
        name_lower = var["name"].lower()
        if any(name_lower == c or name_lower.startswith(c) for c in lon_candidates):
            lon_var = var["name"]
            break

    return lat_var, lon_var


@router.post("/map")
async def get_geo_map_data(
    request: GeoMapRequest, x_session_id: Optional[str] = Header(None)
):
    """Generate geo map data with coordinates, markers, colors, and sizes."""
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Orange3 not available")

    try:
        import numpy as np
        from app.core.data_utils import async_load_data

        logger.info(
            f"Loading geo map data from: {request.data_path} (session: {x_session_id})"
        )
        data = await async_load_data(request.data_path, session_id=x_session_id)
        if data is None:
            raise HTTPException(
                status_code=400, detail=f"Failed to load data: {request.data_path}"
            )

        logger.info(f"Loaded data: {len(data)} instances, domain: {data.domain}")

        # Filter by selected indices if provided
        if request.selected_indices and len(request.selected_indices) > 0:
            valid_indices = [i for i in request.selected_indices if 0 <= i < len(data)]
            if valid_indices:
                data = data[valid_indices]

        # Collect all variables (attributes + class + metas)
        all_vars = (
            list(data.domain.attributes)
            + ([data.domain.class_var] if data.domain.class_var else [])
            + list(data.domain.metas)
        )

        variables = []
        for var in all_vars:
            var_info = {
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical",
            }
            if hasattr(var, "values") and var.values:
                var_info["values"] = list(var.values)
            variables.append(var_info)

        # Determine lat/lon columns — use auto-detection if not specified
        lat_col_name, lon_col_name = _find_lat_lon_columns(variables)
        lat_col_name = request.lat_column or lat_col_name
        lon_col_name = request.lon_column or lon_col_name

        if not lat_col_name or not lon_col_name:
            # Return metadata-only response so frontend can ask user to pick columns
            return {
                "markers": [],
                "variables": variables,
                "totalMarkers": 0,
                "instanceCount": len(data),
                "latColumn": None,
                "lonColumn": None,
                "colorColumn": None,
                "sizeColumn": None,
                "bounds": None,
                "error": "No latitude/longitude columns detected. Please select columns.",
            }

        # Resolve variable objects for requested column names
        var_by_name = {v.name: v for v in all_vars}

        lat_var = var_by_name.get(lat_col_name)
        lon_var = var_by_name.get(lon_col_name)
        color_var = (
            var_by_name.get(request.color_column) if request.color_column else None
        )
        size_var = var_by_name.get(request.size_column) if request.size_column else None

        if lat_var is None or lon_var is None:
            available = list(var_by_name.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Lat/lon columns not found. lat={lat_col_name}, lon={lon_col_name}. Available: {available}",
            )

        if not lat_var.is_continuous or not lon_var.is_continuous:
            raise HTTPException(
                status_code=400,
                detail="Latitude and longitude columns must be numeric.",
            )

        # Extract column arrays
        lat_array = data.get_column(lat_var)
        lon_array = data.get_column(lon_var)
        valid_mask = ~(np.isnan(lat_array) | np.isnan(lon_array))

        # Validate coordinate ranges
        lat_valid = lat_array[valid_mask]
        lon_valid = lon_array[valid_mask]
        if len(lat_valid) > 0:
            if np.any(np.abs(lat_valid) > 90) or np.any(np.abs(lon_valid) > 180):
                logger.warning("Some coordinates are outside valid lat/lon ranges.")

        # Color setup
        color_array = None
        class_names = []
        class_colors = {}
        if color_var:
            try:
                color_array = data.get_column(color_var)
                if hasattr(color_var, "values") and color_var.values:
                    class_names = list(color_var.values)
                    class_colors = {
                        cls: _COLORS[i % len(_COLORS)]
                        for i, cls in enumerate(class_names)
                    }
            except Exception as e:
                logger.warning(f"Could not get color column: {e}")

        # Size setup
        size_array = None
        size_min = size_max = None
        if size_var and size_var.is_continuous:
            try:
                size_array = data.get_column(size_var)
                valid_sizes = size_array[valid_mask & ~np.isnan(size_array)]
                if len(valid_sizes) > 0:
                    size_min = float(np.nanmin(valid_sizes))
                    size_max = float(np.nanmax(valid_sizes))
            except Exception as e:
                logger.warning(f"Could not get size column: {e}")

        # Tooltip columns
        tooltip_vars = []
        if request.tooltip_columns:
            tooltip_vars = [
                var_by_name[col]
                for col in request.tooltip_columns
                if col in var_by_name
            ]

        # Build markers
        markers = []
        for i in range(len(data)):
            if not valid_mask[i]:
                continue

            lat = float(lat_array[i])
            lon = float(lon_array[i])

            marker = {
                "lat": lat,
                "lon": lon,
                "index": i,
            }

            # Color
            if color_var and color_array is not None and not np.isnan(color_array[i]):
                if class_names:
                    class_idx = int(color_array[i])
                    if class_idx < len(class_names):
                        cls_name = class_names[class_idx]
                        marker["colorValue"] = cls_name
                        marker["color"] = class_colors.get(cls_name, _COLORS[0])
                else:
                    # Continuous color: store raw value, frontend maps to gradient
                    marker["colorValue"] = float(color_array[i])

            # Size (normalized 5–20px radius)
            if size_array is not None and not np.isnan(size_array[i]):
                raw = float(size_array[i])
                if size_max is not None and size_max != size_min:
                    normalized = (raw - size_min) / (size_max - size_min)
                    marker["size"] = 5 + normalized * 15  # 5–20px
                else:
                    marker["size"] = 10
                marker["sizeValue"] = raw

            # Tooltip data
            if tooltip_vars:
                tooltip = {}
                for tv in tooltip_vars:
                    try:
                        val = data.get_column(tv)[i]
                        if not np.isnan(val):
                            if hasattr(tv, "values") and tv.values:
                                idx = int(val)
                                tooltip[tv.name] = (
                                    tv.values[idx] if idx < len(tv.values) else str(val)
                                )
                            else:
                                tooltip[tv.name] = float(val)
                    except Exception as e:
                        logger.debug(f"Suppressed error: {e}")
                marker["tooltip"] = tooltip

            markers.append(marker)

        # Map bounds for auto-fit
        bounds = None
        if len(lat_valid) > 0:
            bounds = {
                "minLat": float(np.nanmin(lat_valid)),
                "maxLat": float(np.nanmax(lat_valid)),
                "minLon": float(np.nanmin(lon_valid)),
                "maxLon": float(np.nanmax(lon_valid)),
            }

        return {
            "markers": markers,
            "variables": variables,
            "totalMarkers": len(markers),
            "instanceCount": len(data),
            "latColumn": lat_col_name,
            "lonColumn": lon_col_name,
            "colorColumn": request.color_column,
            "sizeColumn": request.size_column,
            "bounds": bounds,
            "classes": class_names if class_names else None,
            "classColors": list(class_colors.values()) if class_colors else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
