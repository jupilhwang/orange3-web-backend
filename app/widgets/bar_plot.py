"""
Bar Plot Widget API endpoints.
"""

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Visualize"])

# Check Orange3 availability
try:
    from Orange.data import Table
    import numpy as np
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False

# Datasets cache directory
DATASETS_CACHE_DIR = Path("./datasets")


class BarPlotRequest(BaseModel):
    """Request model for bar plot endpoint."""
    dataset_path: str
    value_var: str  # Continuous variable for bar heights
    group_var: Optional[str] = None  # Discrete variable for grouping
    annot_var: Optional[str] = None  # Variable for x-axis annotations
    color_var: Optional[str] = None  # Discrete variable for coloring
    selected_indices: Optional[List[int]] = None


@router.post("/barplot")
async def get_barplot_data(request: BarPlotRequest):
    """
    Generate bar plot data for visualization.
    Similar to Orange3's Bar Plot widget.
    """
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Orange3 not available")
    
    try:
        from Orange.data import Table
        import numpy as np
        
        # Load dataset
        dataset_path = request.dataset_path
        
        # Handle different path formats
        if dataset_path.startswith('/'):
            data = Table(dataset_path)
        elif 'datasets/' in dataset_path or dataset_path.endswith('.tab') or dataset_path.endswith('.csv'):
            try:
                data = Table(dataset_path)
            except:
                possible_paths = [
                    Path(dataset_path),
                    DATASETS_CACHE_DIR / dataset_path,
                    Path("datasets") / dataset_path,
                ]
                data = None
                for p in possible_paths:
                    if p.exists():
                        data = Table(str(p))
                        break
                if data is None:
                    raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")
        else:
            data = Table(dataset_path)
        
        if data is None:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_path}")
        
        original_len = len(data)
        
        # Filter by selected indices if provided
        if request.selected_indices and len(request.selected_indices) > 0:
            valid_indices = [i for i in request.selected_indices if 0 <= i < len(data)]
            if valid_indices:
                data = data[valid_indices]
                logger.info(f"BarPlot: filtered to {len(data)} of {original_len} instances")
        
        MAX_INSTANCES = 200
        truncated = False
        if len(data) > MAX_INSTANCES:
            data = data[:MAX_INSTANCES]
            truncated = True
        
        # Find value variable (must be continuous)
        value_var = None
        for var in list(data.domain.attributes) + ([data.domain.class_var] if data.domain.class_var else []):
            if var.name == request.value_var and var.is_continuous:
                value_var = var
                break
        
        if value_var is None:
            raise HTTPException(status_code=400, detail=f"Continuous variable '{request.value_var}' not found")
        
        # Find group variable (discrete, optional)
        group_var = None
        if request.group_var:
            for var in list(data.domain.attributes) + ([data.domain.class_var] if data.domain.class_var else []) + list(data.domain.metas):
                if var.name == request.group_var and var.is_discrete:
                    group_var = var
                    break
        
        # Find annotation variable
        annot_var = None
        if request.annot_var and request.annot_var != "Enumeration":
            for var in list(data.domain.attributes) + ([data.domain.class_var] if data.domain.class_var else []) + list(data.domain.metas):
                if var.name == request.annot_var:
                    annot_var = var
                    break
        
        # Find color variable (discrete, optional)
        color_var = None
        if request.color_var:
            for var in list(data.domain.attributes) + ([data.domain.class_var] if data.domain.class_var else []) + list(data.domain.metas):
                if var.name == request.color_var and var.is_discrete:
                    color_var = var
                    break
        
        # Sort by group if specified
        indices = np.arange(len(data))
        if group_var:
            group_col = data.get_column(group_var)
            indices = np.argsort(group_col, kind="mergesort")
        
        # Get values
        values_col = data.get_column(value_var)
        values = [float(values_col[i]) if not np.isnan(values_col[i]) else None for i in indices]
        
        # Get annotations (x-axis labels)
        labels = []
        if request.annot_var == "Enumeration":
            labels = [str(i + 1) for i in range(len(data))]
        elif annot_var:
            for i in indices:
                val = data[int(i)][annot_var]
                if annot_var.is_discrete:
                    labels.append(annot_var.str_val(val))
                else:
                    labels.append(str(val) if val is not None else "")
        
        # Get group labels
        group_labels = []
        if group_var:
            for i in indices:
                val = data[int(i)][group_var]
                group_labels.append(group_var.str_val(val))
        
        # Get colors
        colors = []
        color_map = {}
        legend_items = []
        
        if color_var:
            color_col = data.get_column(color_var)
            var_colors = color_var.colors if hasattr(color_var, 'colors') else None
            
            for idx, val in enumerate(color_var.values):
                if var_colors is not None:
                    c = var_colors[idx]
                    color_rgb = [int(c[0]), int(c[1]), int(c[2])]
                else:
                    default_colors = [
                        [70, 190, 250],
                        [237, 70, 47],
                        [170, 242, 43],
                        [245, 174, 50],
                        [157, 118, 196],
                    ]
                    color_rgb = default_colors[idx % len(default_colors)]
                
                color_map[idx] = color_rgb
                legend_items.append({
                    "value": val,
                    "color": color_rgb
                })
            
            for i in indices:
                c_val = color_col[i]
                if np.isnan(c_val):
                    colors.append([128, 128, 128])
                else:
                    colors.append(color_map.get(int(c_val), [128, 128, 128]))
        else:
            colors = [[128, 128, 128]] * len(data)
        
        # Calculate group separators
        group_separators = []
        if group_labels:
            prev_label = None
            for idx, label in enumerate(group_labels):
                if prev_label is not None and label != prev_label:
                    group_separators.append(idx)
                prev_label = label
        
        # Build bars data
        bars = []
        for idx in range(len(values)):
            bar = {
                "index": idx,
                "value": values[idx],
                "color": colors[idx] if idx < len(colors) else [128, 128, 128],
            }
            if labels:
                bar["label"] = labels[idx] if idx < len(labels) else ""
            if group_labels:
                bar["group"] = group_labels[idx] if idx < len(group_labels) else ""
            bars.append(bar)
        
        # Statistics
        valid_values = [v for v in values if v is not None]
        statistics = {}
        if valid_values:
            statistics = {
                "min": float(np.min(valid_values)),
                "max": float(np.max(valid_values)),
                "mean": float(np.mean(valid_values)),
                "std": float(np.std(valid_values))
            }
        
        return {
            "bars": bars,
            "value_var": value_var.name,
            "group_var": group_var.name if group_var else None,
            "annot_var": annot_var.name if annot_var else (request.annot_var if request.annot_var == "Enumeration" else None),
            "color_var": color_var.name if color_var else None,
            "legend": legend_items,
            "group_separators": group_separators,
            "statistics": statistics,
            "total_instances": len(data),
            "truncated": truncated,
            "max_instances": MAX_INSTANCES
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bar plot error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

