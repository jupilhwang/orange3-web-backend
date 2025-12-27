"""
Heat Map Widget API endpoints.
"""

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/visualize", tags=["Visualize"])

# Check Orange3 availability
try:
    from Orange.data import Table
    import numpy as np
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False

# Datasets cache directory
DATASETS_CACHE_DIR = Path("./datasets")


class HeatMapRequest(BaseModel):
    """Request model for heatmap endpoint."""
    data_path: str
    color_scheme: str = "Blue-Green-Yellow"
    threshold_low: float = 0.0
    threshold_high: float = 1.0
    merge_kmeans: bool = False
    merge_kmeans_k: int = 50
    clustering_rows: str = "None"
    clustering_cols: str = "None"
    split_by_rows: Optional[str] = None
    split_by_cols: Optional[str] = None
    show_legend: bool = True
    show_averages: bool = True
    row_annotation_text: Optional[str] = None
    row_annotation_color: Optional[str] = None
    col_annotation_position: str = "Top"
    col_annotation_color: Optional[str] = None
    keep_aspect_ratio: bool = False
    selected_indices: Optional[List[int]] = None


@router.post("/heatmap")
async def get_heatmap_data(request: HeatMapRequest):
    """
    Generate heatmap data for visualization.
    Similar to Orange3's Heat Map widget.
    """
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Orange3 not available")
    
    try:
        from Orange.data import Table
        import numpy as np
        
        # Load dataset
        data_path = request.data_path
        
        if data_path.startswith('/'):
            data = Table(data_path)
        elif 'datasets/' in data_path or data_path.endswith('.tab') or data_path.endswith('.csv'):
            try:
                data = Table(data_path)
            except:
                possible_paths = [
                    Path(data_path),
                    DATASETS_CACHE_DIR / data_path,
                    Path("datasets") / data_path,
                ]
                data = None
                for p in possible_paths:
                    if p.exists():
                        data = Table(str(p))
                        break
                if data is None:
                    raise HTTPException(status_code=404, detail=f"Dataset not found: {data_path}")
        else:
            data = Table(data_path)
        
        if data is None:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {request.data_path}")
        
        original_len = len(data)
        
        # Filter by selected indices
        if request.selected_indices and len(request.selected_indices) > 0:
            valid_indices = [i for i in request.selected_indices if 0 <= i < len(data)]
            if valid_indices:
                data = data[valid_indices]
                logger.info(f"HeatMap: filtered to {len(data)} of {original_len} instances")
        
        # Get continuous variables only
        continuous_vars = [var for var in data.domain.attributes if var.is_continuous]
        
        if not continuous_vars:
            raise HTTPException(status_code=400, detail="No continuous variables found for heatmap")
        
        # Get discrete variables
        discrete_vars = [var for var in data.domain.attributes if var.is_discrete]
        if data.domain.class_var and data.domain.class_var.is_discrete:
            discrete_vars.append(data.domain.class_var)
        
        # Build data matrix
        col_names = [var.name for var in continuous_vars]
        data_matrix = np.column_stack([data.get_column(var) for var in continuous_vars])
        
        # Handle missing values
        for col_idx in range(data_matrix.shape[1]):
            col = data_matrix[:, col_idx]
            mask = np.isnan(col)
            if mask.any():
                col[mask] = np.nanmean(col) if not np.all(mask) else 0
        
        # Calculate data range
        data_min = float(np.nanmin(data_matrix))
        data_max = float(np.nanmax(data_matrix))
        
        # Normalize data
        norm_matrix = data_matrix.copy()
        for col_idx in range(norm_matrix.shape[1]):
            col = norm_matrix[:, col_idx]
            col_min, col_max = np.min(col), np.max(col)
            if col_max > col_min:
                norm_matrix[:, col_idx] = (col - col_min) / (col_max - col_min)
            else:
                norm_matrix[:, col_idx] = 0.5
        
        # Apply threshold range
        norm_matrix = np.clip(
            (norm_matrix - request.threshold_low) / (request.threshold_high - request.threshold_low + 1e-10),
            0, 1
        )
        
        # K-means clustering for row merging
        row_indices = list(range(len(data)))
        row_labels = None
        cluster_data = None
        
        if request.merge_kmeans and len(data) > request.merge_kmeans_k:
            try:
                from sklearn.cluster import KMeans
                
                kmeans = KMeans(n_clusters=request.merge_kmeans_k, random_state=42, n_init=10)
                row_labels = kmeans.fit_predict(norm_matrix)
                
                cluster_data = []
                for cluster_id in range(request.merge_kmeans_k):
                    cluster_mask = row_labels == cluster_id
                    if cluster_mask.any():
                        cluster_mean = np.mean(norm_matrix[cluster_mask], axis=0)
                        cluster_count = np.sum(cluster_mask)
                        cluster_data.append({
                            "cluster_id": cluster_id,
                            "values": cluster_mean.tolist(),
                            "count": int(cluster_count)
                        })
                
                norm_matrix = np.array([c["values"] for c in cluster_data])
                row_indices = list(range(len(cluster_data)))
                logger.info(f"HeatMap: merged {original_len} rows into {len(cluster_data)} clusters")
            except ImportError:
                logger.warning("sklearn not available for k-means clustering")
        
        # Row clustering (hierarchical)
        row_order = list(range(len(norm_matrix)))
        row_dendrogram = None
        
        if request.clustering_rows != "None" and len(norm_matrix) > 1:
            try:
                from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
                from scipy.spatial.distance import pdist
                
                if len(norm_matrix) > 2:
                    dist_matrix = pdist(norm_matrix, metric='euclidean')
                    linkage_matrix = linkage(dist_matrix, method='average')
                    row_order = leaves_list(linkage_matrix).tolist()
                    
                    dendro = dendrogram(linkage_matrix, no_plot=True)
                    row_dendrogram = {
                        "icoord": dendro["icoord"],
                        "dcoord": dendro["dcoord"],
                        "leaves": dendro["leaves"]
                    }
            except ImportError:
                logger.warning("scipy not available for hierarchical clustering")
        
        # Column clustering
        col_order = list(range(len(col_names)))
        col_dendrogram = None
        
        if request.clustering_cols != "None" and len(col_names) > 1:
            try:
                from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
                from scipy.spatial.distance import pdist
                
                col_data = norm_matrix.T
                if len(col_data) > 2:
                    dist_matrix = pdist(col_data, metric='euclidean')
                    linkage_matrix = linkage(dist_matrix, method='average')
                    col_order = leaves_list(linkage_matrix).tolist()
                    
                    dendro = dendrogram(linkage_matrix, no_plot=True)
                    col_dendrogram = {
                        "icoord": dendro["icoord"],
                        "dcoord": dendro["dcoord"],
                        "leaves": dendro["leaves"]
                    }
            except ImportError:
                logger.warning("scipy not available for hierarchical clustering")
        
        # Reorder matrix
        norm_matrix = norm_matrix[row_order, :]
        norm_matrix = norm_matrix[:, col_order]
        ordered_col_names = [col_names[i] for i in col_order]
        
        # Calculate row averages
        row_averages = np.mean(norm_matrix, axis=1).tolist() if request.show_averages else None
        
        # Split by discrete variable
        split_groups = None
        if request.split_by_rows:
            split_var = None
            for var in discrete_vars + list(data.domain.metas):
                if var.name == request.split_by_rows and var.is_discrete:
                    split_var = var
                    break
            
            if split_var:
                split_col = data.get_column(split_var)
                
                new_row_order = []
                split_groups = []
                current_offset = 0
                
                for val_idx, val_name in enumerate(split_var.values):
                    group_indices = []
                    for i, orig_idx in enumerate(row_order):
                        if row_indices[orig_idx] < len(split_col):
                            if int(split_col[row_indices[orig_idx]]) == val_idx:
                                group_indices.append(i)
                    
                    if group_indices:
                        new_row_order.extend([row_order[i] for i in group_indices])
                        split_groups.append({
                            "name": val_name,
                            "start_row": current_offset,
                            "row_count": len(group_indices),
                            "indices": group_indices
                        })
                        current_offset += len(group_indices)
                
                if new_row_order:
                    row_order = new_row_order
                    norm_matrix = norm_matrix[row_order, :]
                    if row_averages:
                        row_averages = [row_averages[i] for i in row_order]
        
        # Row annotations
        row_annotations = None
        if request.row_annotation_text:
            annot_var = None
            for var in list(data.domain.attributes) + ([data.domain.class_var] if data.domain.class_var else []) + list(data.domain.metas):
                if var.name == request.row_annotation_text:
                    annot_var = var
                    break
            
            if annot_var:
                if cluster_data:
                    row_annotations = [f"Cluster {i+1} (n={cluster_data[i]['count']})" for i in row_order]
                else:
                    annot_col = data.get_column(annot_var)
                    row_annotations = []
                    for i in row_order:
                        if annot_var.is_discrete:
                            row_annotations.append(annot_var.str_val(annot_col[row_indices[i]]))
                        else:
                            row_annotations.append(str(annot_col[row_indices[i]]))
        
        # Row annotation colors
        row_annotation_colors = None
        if request.row_annotation_color:
            color_var = None
            for var in discrete_vars:
                if var.name == request.row_annotation_color:
                    color_var = var
                    break
            
            if color_var:
                color_col = data.get_column(color_var)
                var_colors = color_var.colors if hasattr(color_var, 'colors') else None
                
                row_annotation_colors = []
                for i in row_order:
                    if cluster_data:
                        row_annotation_colors.append([128, 128, 128])
                    else:
                        val = color_col[row_indices[i]]
                        if np.isnan(val):
                            row_annotation_colors.append([128, 128, 128])
                        elif var_colors is not None:
                            c = var_colors[int(val)]
                            row_annotation_colors.append([int(c[0]), int(c[1]), int(c[2])])
                        else:
                            default_colors = [
                                [70, 190, 250], [237, 70, 47], [170, 242, 43],
                                [245, 174, 50], [157, 118, 196]
                            ]
                            row_annotation_colors.append(default_colors[int(val) % len(default_colors)])
        
        return {
            "matrix": norm_matrix.tolist(),
            "columns": ordered_col_names,
            "row_count": len(norm_matrix),
            "col_count": len(ordered_col_names),
            "data_min": data_min,
            "data_max": data_max,
            "color_scheme": request.color_scheme,
            "row_dendrogram": row_dendrogram,
            "col_dendrogram": col_dendrogram,
            "row_averages": row_averages,
            "split_groups": split_groups,
            "row_annotations": row_annotations,
            "row_annotation_colors": row_annotation_colors,
            "is_clustered": request.merge_kmeans and cluster_data is not None,
            "cluster_count": len(cluster_data) if cluster_data else 0,
            "total_instances": original_len,
            "discrete_vars": [{"name": v.name, "values": list(v.values)} for v in discrete_vars],
            "continuous_vars": [v.name for v in continuous_vars]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"HeatMap error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


