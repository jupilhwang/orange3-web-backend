"""
Scatter Plot Widget API endpoints.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/visualize", tags=["Visualize"])

from app.core.orange_compat import ORANGE_AVAILABLE, Table


class ScatterPlotRequest(BaseModel):
    """Request model for scatter plot data."""
    data_path: str
    axis_x: Optional[str] = None
    axis_y: Optional[str] = None
    color_attr: Optional[str] = None
    size_attr: Optional[str] = None
    shape_attr: Optional[str] = None
    jittering: float = 0
    subset_indices: Optional[List[int]] = None
    selected_indices: Optional[List[int]] = None


class ScatterPlotSelectionRequest(BaseModel):
    """Request model for scatter plot selection."""
    data_path: str
    selected_indices: List[int]


@router.post("/scatter-plot")
async def get_scatter_plot_data(
    request: ScatterPlotRequest,
    x_session_id: Optional[str] = Header(None)
):
    """Generate scatter plot data for visualization."""
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Orange3 not available")
    
    try:
        from Orange.data import Table
        import numpy as np
        from app.core.data_utils import async_load_data
        
        # Load data (supports datasets, uploads, kmeans results)
        logger.info(f"Loading scatter plot data from: {request.data_path} (session: {x_session_id})")
        data = await async_load_data(request.data_path, session_id=x_session_id)
        if data is None:
            raise HTTPException(status_code=400, detail=f"Failed to load data: {request.data_path}")
        original_len = len(data)
        logger.info(f"Loaded data: {len(data)} instances, domain: {data.domain}")
        
        # Filter by selected indices if provided
        if request.selected_indices and len(request.selected_indices) > 0:
            valid_indices = [i for i in request.selected_indices if 0 <= i < len(data)]
            if valid_indices:
                data = data[valid_indices]
                logger.info(f"Scatter plot: filtered to {len(data)} of {original_len} instances")
        
        # Get variables
        variables = []
        numeric_vars = []
        categorical_vars = []
        
        for var in data.domain.attributes:
            var_info = {
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical"
            }
            if hasattr(var, 'values') and var.values:
                var_info["values"] = list(var.values)
            variables.append(var_info)
            
            if var.is_continuous:
                numeric_vars.append(var)
            else:
                categorical_vars.append(var)
        
        # Add class variable
        if data.domain.class_var:
            var = data.domain.class_var
            var_info = {
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical"
            }
            if hasattr(var, 'values') and var.values:
                var_info["values"] = list(var.values)
            variables.append(var_info)
            
            if var.is_continuous:
                numeric_vars.append(var)
            else:
                categorical_vars.append(var)
        
        # Determine axis variables - fallback to available vars if requested ones not found
        default_x = numeric_vars[0].name if len(numeric_vars) > 0 else None
        default_y = numeric_vars[1].name if len(numeric_vars) > 1 else default_x
        
        axis_x_name = request.axis_x if request.axis_x else default_x
        axis_y_name = request.axis_y if request.axis_y else default_y
        
        # Validate that requested axis variables exist in data
        all_var_names = [v.name for v in list(data.domain.attributes) + 
                         ([data.domain.class_var] if data.domain.class_var else []) +
                         list(data.domain.metas)]
        
        # If requested axis_x doesn't exist, use default
        if axis_x_name and axis_x_name not in all_var_names:
            logger.warning(f"Requested axis_x '{axis_x_name}' not found, using default '{default_x}'")
            axis_x_name = default_x
        
        # If requested axis_y doesn't exist, use default  
        if axis_y_name and axis_y_name not in all_var_names:
            logger.warning(f"Requested axis_y '{axis_y_name}' not found, using default '{default_y}'")
            axis_y_name = default_y
        
        if not axis_x_name or not axis_y_name:
            raise HTTPException(status_code=400, detail="Not enough numeric variables for scatter plot")
        
        # Find variable objects (include metas for k-Means results)
        axis_x_var = None
        axis_y_var = None
        color_var = None
        
        all_vars = list(data.domain.attributes) + \
                   ([data.domain.class_var] if data.domain.class_var else []) + \
                   list(data.domain.metas)
        
        logger.info(f"Looking for axis_x={axis_x_name}, axis_y={axis_y_name} in {[v.name for v in all_vars]}")
        
        for var in all_vars:
            if var.name == axis_x_name:
                axis_x_var = var
            if var.name == axis_y_name:
                axis_y_var = var
            if request.color_attr and var.name == request.color_attr:
                color_var = var
        
        # If color_attr is "Cluster" (from k-Means), try to find it in metas
        if request.color_attr and not color_var:
            for var in data.domain.metas:
                if var.name == request.color_attr:
                    color_var = var
                    break
        
        if not axis_x_var or not axis_y_var:
            available = [v.name for v in all_vars]
            raise HTTPException(
                status_code=400, 
                detail=f"Axis variables not found. Looking for x={axis_x_name}, y={axis_y_name}. Available: {available}"
            )
        
        # Get data values
        x_col = data.get_column(axis_x_var)
        y_col = data.get_column(axis_y_var)
        
        # Handle missing values
        valid_mask = ~(np.isnan(x_col) | np.isnan(y_col))
        
        # Build points
        points = []
        
        # Color mapping
        colors = ['#5dade2', '#e74c3c', '#82e0aa', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#3498db']
        class_names = []
        class_colors = {}
        
        if color_var and hasattr(color_var, 'values') and color_var.values:
            class_names = list(color_var.values)
            for i, cls in enumerate(class_names):
                class_colors[cls] = colors[i % len(colors)]
        
        # Get color column (may be in metas for k-Means Cluster)
        color_col = None
        if color_var:
            try:
                color_col = data.get_column(color_var)
            except Exception as e:
                logger.warning(f"Could not get color column: {e}")
        
        for i in range(len(data)):
            if not valid_mask[i]:
                continue
            
            x_val = float(x_col[i])
            y_val = float(y_col[i])
            
            # Apply jittering
            if request.jittering > 0:
                jitter_amount = request.jittering / 100
                x_range = float(np.nanmax(x_col) - np.nanmin(x_col))
                y_range = float(np.nanmax(y_col) - np.nanmin(y_col))
                x_val += (np.random.random() - 0.5) * x_range * jitter_amount
                y_val += (np.random.random() - 0.5) * y_range * jitter_amount
            
            point = {
                "x": x_val,
                "y": y_val,
                "index": i
            }
            
            # Add class info (using "classValue" for frontend compatibility)
            if color_var and color_col is not None:
                class_idx = int(color_col[i]) if not np.isnan(color_col[i]) else 0
                if class_names and class_idx < len(class_names):
                    cls_name = class_names[class_idx]
                    point["classValue"] = cls_name  # Changed from "class" to "classValue"
                    point["color"] = class_colors.get(cls_name, colors[0])
            
            # Add range data (all numeric variable values for error bars)
            range_data = {}
            for attr in data.domain.attributes:
                if attr.is_continuous:
                    try:
                        col_idx = data.domain.index(attr)
                        val = data.X[i, col_idx]
                        if not np.isnan(val):
                            range_data[attr.name] = float(val)
                    except:
                        pass
            point["rangeData"] = range_data
            
            points.append(point)
        
        # Calculate ranges
        valid_x = x_col[valid_mask]
        valid_y = y_col[valid_mask]
        
        # Handle empty data case
        if len(valid_x) == 0 or len(valid_y) == 0:
            return {
                "points": [],
                "xLabel": axis_x_name,
                "yLabel": axis_y_name,
                "xMin": 0,
                "xMax": 1,
                "yMin": 0,
                "yMax": 1,
                "classes": class_names if class_names else None,
                "classColors": list(class_colors.values()) if class_colors else None,
                "variables": variables,
                "totalPoints": 0,
                "instanceCount": len(data),
                "error": "No valid data points"
            }
        
        x_min = float(np.nanmin(valid_x))
        x_max = float(np.nanmax(valid_x))
        y_min = float(np.nanmin(valid_y))
        y_max = float(np.nanmax(valid_y))
        
        # Add padding (handle case where min == max)
        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1
        x_padding = x_range * 0.05
        y_padding = y_range * 0.05
        
        return {
            "points": points,
            "xLabel": axis_x_name,
            "yLabel": axis_y_name,
            "xMin": x_min - x_padding,
            "xMax": x_max + x_padding,
            "yMin": y_min - y_padding,
            "yMax": y_max + y_padding,
            "classes": class_names if class_names else None,
            "classColors": list(class_colors.values()) if class_colors else None,
            "variables": variables,
            "totalPoints": len(points),
            "instanceCount": len(data)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scatter-plot/select")
async def select_scatter_plot_data(
    request: ScatterPlotSelectionRequest,
    x_session_id: Optional[str] = Header(None)
):
    """Return selected data subset from scatter plot."""
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Orange3 not available")
    
    try:
        from app.core.data_utils import async_load_data
        
        logger.info(f"Loading scatter plot selection from: {request.data_path} (session: {x_session_id})")
        data = await async_load_data(request.data_path, session_id=x_session_id)
        if data is None:
            raise HTTPException(status_code=400, detail=f"Failed to load data: {request.data_path}")
        
        if not request.selected_indices:
            return {"selected_count": 0, "data": None}
        
        # Filter to selected indices
        selected_data = data[request.selected_indices]
        
        return {
            "selected_count": len(selected_data),
            "instances": len(selected_data),
            "features": len(selected_data.domain.attributes)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class ProjectionRequest(BaseModel):
    """Request model for finding informative projections."""
    path: str
    color_attr: Optional[str] = None


@router.post("/scatter-plot/projections")
async def find_informative_projections(
    request: ProjectionRequest,
    x_session_id: Optional[str] = Header(None)
):
    """Find informative projections for scatter plot (Score Plot)."""
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Orange3 not available")
    
    try:
        from app.core.data_utils import async_load_data
        import numpy as np
        
        logger.info(f"Finding informative projections for: {request.path}")
        data = await async_load_data(request.path, session_id=x_session_id)
        if data is None:
            raise HTTPException(status_code=400, detail=f"Failed to load data: {request.path}")
        
        # Get numeric attributes only
        numeric_attrs = [attr for attr in data.domain.attributes 
                        if attr.is_continuous]
        
        if len(numeric_attrs) < 2:
            return {"projections": [], "message": "Need at least 2 numeric variables"}
        
        # Get class variable if exists for scoring
        class_var = data.domain.class_var if data.domain.class_var else None
        
        # Calculate projection scores using variance-based scoring
        # Higher score = better separation of classes (if class exists) or higher variance spread
        projections = []
        
        for i, attr1 in enumerate(numeric_attrs):
            for attr2 in numeric_attrs[i+1:]:
                try:
                    # Get data for these two attributes
                    col1_idx = data.domain.index(attr1)
                    col2_idx = data.domain.index(attr2)
                    
                    x_data = data.X[:, col1_idx]
                    y_data = data.X[:, col2_idx]
                    
                    # Remove NaN values
                    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
                    x_valid = x_data[valid_mask]
                    y_valid = y_data[valid_mask]
                    
                    if len(x_valid) < 2:
                        continue
                    
                    # Calculate score based on data spread and correlation
                    x_std = np.std(x_valid) if np.std(x_valid) > 0 else 1e-10
                    y_std = np.std(y_valid) if np.std(y_valid) > 0 else 1e-10
                    
                    # Correlation (lower is better for projections - we want independent axes)
                    if len(x_valid) > 1:
                        correlation = abs(np.corrcoef(x_valid, y_valid)[0, 1])
                        if np.isnan(correlation):
                            correlation = 0
                    else:
                        correlation = 0
                    
                    # Score: higher variance spread + lower correlation = better
                    variance_score = (x_std * y_std) ** 0.5
                    independence_score = 1 - correlation
                    
                    # If we have class variable, add class separation score
                    class_score = 0
                    if class_var and class_var.is_discrete:
                        try:
                            classes = data.Y[valid_mask]
                            unique_classes = np.unique(classes[~np.isnan(classes)])
                            if len(unique_classes) > 1:
                                # Calculate between-class variance
                                class_means_x = []
                                class_means_y = []
                                for c in unique_classes:
                                    mask = classes == c
                                    if np.sum(mask) > 0:
                                        class_means_x.append(np.mean(x_valid[mask]))
                                        class_means_y.append(np.mean(y_valid[mask]))
                                
                                if len(class_means_x) > 1:
                                    class_score = np.std(class_means_x) + np.std(class_means_y)
                        except:
                            pass
                    
                    # Combined score (normalized)
                    score = variance_score * independence_score * (1 + class_score)
                    
                    projections.append({
                        "attr1": attr1.name,
                        "attr2": attr2.name,
                        "score": float(score),
                        "correlation": float(correlation),
                        "variance": float(variance_score)
                    })
                    
                except Exception as e:
                    logger.warning(f"Error calculating projection score for {attr1.name}/{attr2.name}: {e}")
                    continue
        
        # Sort by score descending
        projections.sort(key=lambda x: x["score"], reverse=True)
        
        # Normalize scores to 0-1 range
        if projections:
            max_score = projections[0]["score"] if projections[0]["score"] > 0 else 1
            for p in projections:
                p["score"] = p["score"] / max_score
        
        # Add rank
        for idx, p in enumerate(projections):
            p["rank"] = idx + 1
        
        # Return top 20 projections
        return {
            "projections": projections[:20],
            "total": len(projections)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


