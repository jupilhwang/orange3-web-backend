"""
Box Plot Widget API endpoints.
Shows distribution of data by category (min, Q1, median, Q3, max, outliers).
"""

import logging
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Visualize"])

from app.core.orange_compat import ORANGE_AVAILABLE


class BoxPlotRequest(BaseModel):
    """Request model for box plot endpoint."""

    data_path: str
    x_axis: Optional[str] = None  # Categorical variable for X axis (groups)
    y_axis: Optional[str] = None  # Numeric variable for Y axis (values)
    group_by: Optional[str] = None  # Optional categorical variable for color grouping
    selected_indices: Optional[List[int]] = None


def _compute_box_stats(values: "np.ndarray") -> dict:
    """Compute box plot statistics: min, q1, median, q3, max, mean, outliers."""
    if len(values) == 0:
        return None

    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1
    median = float(np.median(values))
    mean = float(np.mean(values))

    # Tukey whiskers: 1.5 * IQR
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    whisker_low = float(np.min(values[values >= lower_fence]))
    whisker_high = float(np.max(values[values <= upper_fence]))

    outliers = values[(values < lower_fence) | (values > upper_fence)]

    return {
        "min": float(np.min(values)),
        "q1": q1,
        "median": median,
        "mean": mean,
        "q3": q3,
        "max": float(np.max(values)),
        "whisker_low": whisker_low,
        "whisker_high": whisker_high,
        "outliers": [float(v) for v in outliers],
        "count": int(len(values)),
    }


@router.post("/box-plot")
async def get_box_plot_data(
    request: BoxPlotRequest, x_session_id: Optional[str] = Header(None)
):
    """
    Generate box plot data for visualization.
    Returns box statistics (min, Q1, median, Q3, max, outliers) per category.
    Similar to Orange3's Box Plot widget.
    """
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Orange3 not available")

    try:
        from app.core.data_utils import async_load_data

        logger.info(
            f"Loading box plot data from: {request.data_path} (session: {x_session_id})"
        )
        data = await async_load_data(request.data_path, session_id=x_session_id)
        if data is None:
            raise HTTPException(
                status_code=400, detail=f"Failed to load data: {request.data_path}"
            )

        original_len = len(data)

        # Filter by selected indices if provided
        if request.selected_indices and len(request.selected_indices) > 0:
            valid_indices = [i for i in request.selected_indices if 0 <= i < len(data)]
            if valid_indices:
                data = data[valid_indices]
                logger.info(
                    f"BoxPlot: filtered to {len(data)} of {original_len} instances"
                )

        domain = data.domain
        all_vars = (
            list(domain.attributes) + list(domain.class_vars) + list(domain.metas)
        )

        # Classify variables
        numeric_vars = [v for v in all_vars if v.is_continuous]
        categorical_vars = [v for v in all_vars if v.is_discrete]

        variables = [
            {
                "name": v.name,
                "type": "numeric" if v.is_continuous else "categorical",
                "values": list(v.values) if hasattr(v, "values") and v.values else None,
            }
            for v in all_vars
        ]

        # Determine y_axis (numeric, required)
        y_var = None
        if request.y_axis:
            y_var = next((v for v in numeric_vars if v.name == request.y_axis), None)
        if y_var is None and numeric_vars:
            y_var = numeric_vars[0]
        if y_var is None:
            raise HTTPException(
                status_code=400, detail="No numeric variable available for Y axis"
            )

        # Determine x_axis (categorical, optional — None means all data in one box)
        x_var = None
        if request.x_axis:
            x_var = next(
                (v for v in categorical_vars if v.name == request.x_axis), None
            )
        if x_var is None and categorical_vars:
            x_var = categorical_vars[0]

        # Determine group_by (categorical, optional)
        group_var = None
        if request.group_by:
            group_var = next(
                (
                    v
                    for v in categorical_vars
                    if v.name != (x_var.name if x_var else None)
                    and v.name == request.group_by
                ),
                None,
            )

        y_col = data.get_column(y_var)

        # Build boxes
        colors = [
            "#5dade2",
            "#e74c3c",
            "#82e0aa",
            "#f39c12",
            "#9b59b6",
            "#1abc9c",
            "#e67e22",
            "#3498db",
        ]

        boxes = []

        if x_var is not None:
            x_col = data.get_column(x_var)
            x_categories = list(x_var.values)

            if group_var is not None:
                g_col = data.get_column(group_var)
                g_categories = list(group_var.values)

                for g_idx, g_cat in enumerate(g_categories):
                    for x_idx, x_cat in enumerate(x_categories):
                        mask = (
                            (~np.isnan(y_col))
                            & (~np.isnan(x_col))
                            & (~np.isnan(g_col))
                            & (x_col == x_idx)
                            & (g_col == g_idx)
                        )
                        values = y_col[mask]
                        stats = _compute_box_stats(values)
                        if stats is not None:
                            boxes.append(
                                {
                                    "label": f"{x_cat} / {g_cat}",
                                    "x_category": x_cat,
                                    "group": g_cat,
                                    "color": colors[g_idx % len(colors)],
                                    **stats,
                                }
                            )
            else:
                for x_idx, x_cat in enumerate(x_categories):
                    mask = (~np.isnan(y_col)) & (~np.isnan(x_col)) & (x_col == x_idx)
                    values = y_col[mask]
                    stats = _compute_box_stats(values)
                    if stats is not None:
                        boxes.append(
                            {
                                "label": x_cat,
                                "x_category": x_cat,
                                "group": None,
                                "color": colors[x_idx % len(colors)],
                                **stats,
                            }
                        )
        else:
            # No x_axis: single box for all valid values
            mask = ~np.isnan(y_col)
            values = y_col[mask]
            stats = _compute_box_stats(values)
            if stats is not None:
                boxes.append(
                    {
                        "label": "All",
                        "x_category": "All",
                        "group": None,
                        "color": colors[0],
                        **stats,
                    }
                )

        # Overall statistics
        valid_y = y_col[~np.isnan(y_col)]
        y_min = float(np.min(valid_y)) if len(valid_y) > 0 else 0
        y_max = float(np.max(valid_y)) if len(valid_y) > 0 else 1
        y_range = y_max - y_min if y_max != y_min else 1

        return {
            "boxes": boxes,
            "y_axis": y_var.name,
            "x_axis": x_var.name if x_var else None,
            "group_by": group_var.name if group_var else None,
            "y_min": y_min - y_range * 0.05,
            "y_max": y_max + y_range * 0.05,
            "variables": variables,
            "total_instances": original_len,
            "displayed_instances": len(data),
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(f"BoxPlot error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
