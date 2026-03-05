"""
k-Means Widget API endpoints.
k-Means clustering algorithm.
"""

import logging
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

from app.core.data_utils import resolve_data_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model", tags=["Model"])

from app.core.orange_compat import ORANGE_AVAILABLE, Domain, DiscreteVariable

# Upload directory
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"


class KMeansRequest(BaseModel):
    """Request model for k-Means clustering."""

    data_path: str
    n_clusters: int = 3
    cluster_range_from: Optional[int] = None  # If set, find optimal k in range
    cluster_range_to: Optional[int] = None
    use_fixed: bool = True  # True: use n_clusters, False: use range
    normalize: bool = True
    init_method: str = "k-means++"  # "k-means++", "random"
    n_init: int = 10  # Re-runs
    max_iter: int = 300  # Maximum iterations
    name: str = "k-Means"
    selected_indices: Optional[List[int]] = None


class KMeansResponse(BaseModel):
    """Response model for k-Means clustering."""

    success: bool
    cluster_id: Optional[str] = None
    data_path: Optional[str] = None  # Path to clustered data
    cluster_info: Optional[dict] = None
    error: Optional[str] = None


# In-memory storage for clustering results
_kmeans_results = {}


@router.post("/kmeans/cluster")
async def cluster_kmeans(
    request: KMeansRequest, x_session_id: Optional[str] = Header(None)
) -> KMeansResponse:
    """
    Perform k-Means clustering.

    Parameters:
    - data_path: Path to the data
    - n_clusters: Number of clusters (when use_fixed=True)
    - cluster_range_from, cluster_range_to: Range for finding optimal k
    - use_fixed: True to use fixed n_clusters, False to find optimal k in range
    - normalize: Normalize columns before clustering
    - init_method: Initialization method ("k-means++", "random")
    - n_init: Number of re-runs
    - max_iter: Maximum iterations
    - selected_indices: Optional list of row indices to use
    """
    if not ORANGE_AVAILABLE:
        return KMeansResponse(success=False, error="Orange3 not available")

    try:
        import numpy as np

        # Validate parameters
        if request.use_fixed:
            if request.n_clusters < 2 or request.n_clusters > 100:
                raise HTTPException(
                    status_code=400,
                    detail="Number of clusters must be between 2 and 100",
                )
        else:
            if request.cluster_range_from is None or request.cluster_range_to is None:
                raise HTTPException(
                    status_code=400,
                    detail="Cluster range must be specified when not using fixed k",
                )
            if request.cluster_range_from < 2 or request.cluster_range_to > 100:
                raise HTTPException(
                    status_code=400, detail="Cluster range must be between 2 and 100"
                )
            if request.cluster_range_from >= request.cluster_range_to:
                raise HTTPException(
                    status_code=400, detail="Range 'from' must be less than 'to'"
                )

        valid_init_methods = ["k-means++", "random"]
        if request.init_method not in valid_init_methods:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid init method. Must be one of: {valid_init_methods}",
            )

        # Load data using common utility (supports sampler, kmeans, uploads, datasets)
        from app.core.data_utils import async_load_data

        data = await async_load_data(request.data_path, session_id=x_session_id)
        logger.debug(
            f"[k-Means] Loading data from {request.data_path} with session_id={x_session_id}"
        )

        if data is None:
            raise HTTPException(
                status_code=404, detail=f"Data not found: {request.data_path}"
            )

        # Filter by selected indices if provided
        if request.selected_indices and len(request.selected_indices) > 0:
            data = data[request.selected_indices]

        if len(data) == 0:
            raise HTTPException(status_code=400, detail="No data to cluster")

        # Preprocess: normalize if requested
        if request.normalize:
            normalizer = Normalize()
            data_for_clustering = normalizer(data)
        else:
            data_for_clustering = data

        # Determine number of clusters
        silhouette_scores = []  # List of {k, score} for range mode

        if request.use_fixed:
            k = request.n_clusters
            silhouette = None
        else:
            # Find optimal k using silhouette score
            import numpy as np
            from sklearn.metrics import silhouette_score

            best_k = request.cluster_range_from
            best_score = -1

            for k_test in range(
                request.cluster_range_from, request.cluster_range_to + 1
            ):
                if k_test >= len(data):
                    break

                kmeans_test = KMeans(
                    n_clusters=k_test,
                    init=request.init_method,
                    n_init=request.n_init,
                    max_iter=request.max_iter,
                )

                # Get cluster assignments - KMeans returns cluster labels directly
                cluster_labels_test = kmeans_test(data_for_clustering)

                # Calculate silhouette score
                if len(np.unique(cluster_labels_test)) > 1:
                    score = silhouette_score(data_for_clustering.X, cluster_labels_test)
                    silhouette_scores.append({"k": k_test, "score": round(score, 3)})
                    if score > best_score:
                        best_score = score
                        best_k = k_test

            k = best_k
            silhouette = best_score

        # Ensure k is valid
        if k >= len(data):
            k = max(2, len(data) - 1)

        # Create k-Means model
        kmeans = KMeans(
            n_clusters=k,
            init=request.init_method,
            n_init=request.n_init,
            max_iter=request.max_iter,
        )

        # Fit and get cluster labels - KMeans returns cluster labels directly
        cluster_labels = kmeans(data_for_clustering).astype(int)

        # Create annotated data with cluster column
        cluster_var = DiscreteVariable(
            "Cluster", values=[f"C{i + 1}" for i in range(k)]
        )

        new_domain = Domain(
            data.domain.attributes,
            data.domain.class_vars,
            list(data.domain.metas) + [cluster_var],
        )

        # Create new data with cluster column
        annotated_data = data.transform(new_domain)
        annotated_data[:, cluster_var] = cluster_labels.reshape(-1, 1)

        # Generate cluster ID
        import uuid

        cluster_id = str(uuid.uuid4())[:8]

        # Store result in session-based storage (preferred) or legacy storage
        from app.core.data_utils import DataSessionManager

        data_path = f"kmeans/{cluster_id}"
        if x_session_id:
            # Store in session-based storage
            await DataSessionManager.store(x_session_id, data_path, annotated_data)
            logger.debug(
                f"[k-Means] Stored result in session {x_session_id}: {data_path}"
            )
        else:
            # Fallback to legacy storage
            _kmeans_results[cluster_id] = {
                "data": annotated_data,
                "original_data": data,
                "model": kmeans,
                "k": k,
                "cluster_labels": cluster_labels,
            }
            logger.debug(f"[k-Means] Stored result in legacy storage: {cluster_id}")

        # Calculate cluster sizes
        unique, counts = np.unique(cluster_labels, return_counts=True)
        cluster_sizes = {f"C{i + 1}": int(counts[i]) for i in range(len(counts))}

        # Cluster info
        cluster_info = {
            "name": request.name,
            "n_clusters": k,
            "instances": len(data),
            "features": len(data.domain.attributes),
            "cluster_sizes": cluster_sizes,
            "init_method": request.init_method,
            "n_init": request.n_init,
            "max_iter": request.max_iter,
            "normalized": request.normalize,
        }

        if silhouette is not None:
            cluster_info["silhouette_score"] = round(silhouette, 4)

        # Add silhouette scores for range mode
        if silhouette_scores:
            cluster_info["silhouette_scores"] = silhouette_scores
            cluster_info["best_k"] = k

        return KMeansResponse(
            success=True,
            cluster_id=cluster_id,
            data_path=f"kmeans/{cluster_id}",
            cluster_info=cluster_info,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        return KMeansResponse(success=False, error=str(e))


@router.get("/kmeans/data/{cluster_id}")
async def get_kmeans_data(cluster_id: str, limit: int = 1000, offset: int = 0):
    """
    Get clustered data.

    Returns data with cluster assignments.
    """
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Orange3 not available")

    if cluster_id not in _kmeans_results:
        raise HTTPException(status_code=404, detail="Cluster result not found")

    try:
        import numpy as np

        result = _kmeans_results[cluster_id]
        data = result["data"]
        cluster_labels = result["cluster_labels"]

        # Apply limit and offset
        total = len(data)
        start = min(offset, total)
        end = min(start + limit, total)

        # Get slice
        subset = data[start:end]
        labels_subset = cluster_labels[start:end]

        # Build response
        columns = [attr.name for attr in data.domain.attributes]
        if data.domain.class_vars:
            columns.extend([cls.name for cls in data.domain.class_vars])
        columns.append("Cluster")

        rows = []
        for i, row in enumerate(subset):
            row_data = []
            for attr in data.domain.attributes:
                val = row[attr]
                if hasattr(val, "value"):
                    row_data.append(val.value)
                else:
                    row_data.append(float(val) if not np.isnan(val) else None)

            for cls in data.domain.class_vars:
                val = row[cls]
                if hasattr(val, "value"):
                    row_data.append(val.value)
                else:
                    row_data.append(float(val) if not np.isnan(val) else None)

            # Cluster label
            row_data.append(f"C{int(labels_subset[i]) + 1}")
            rows.append(row_data)

        return {
            "success": True,
            "columns": columns,
            "data": rows,
            "total": total,
            "offset": start,
            "limit": limit,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/kmeans/info/{cluster_id}")
async def get_kmeans_info(cluster_id: str):
    """Get information about clustering result."""
    if cluster_id not in _kmeans_results:
        raise HTTPException(status_code=404, detail="Cluster result not found")

    result = _kmeans_results[cluster_id]
    data = result["data"]
    cluster_labels = result["cluster_labels"]
    k = result["k"]

    import numpy as np

    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = {f"C{i + 1}": int(counts[i]) for i in range(len(counts))}

    return {
        "cluster_id": cluster_id,
        "n_clusters": k,
        "instances": len(data),
        "features": len(data.domain.attributes) if hasattr(data, "domain") else 0,
        "cluster_sizes": cluster_sizes,
    }


@router.delete("/kmeans/{cluster_id}")
async def delete_kmeans_result(cluster_id: str):
    """Delete a clustering result."""
    if cluster_id in _kmeans_results:
        del _kmeans_results[cluster_id]
    return {"message": f"Cluster result {cluster_id} deleted"}


@router.get("/kmeans/options")
async def get_kmeans_options():
    """Get available options for k-Means configuration."""
    return {
        "init_methods": [
            {"value": "k-means++", "label": "Initialize with KMeans++"},
            {"value": "random", "label": "Random initialization"},
        ],
        "n_clusters": {"min": 2, "max": 100, "default": 3},
        "n_init": {"min": 1, "max": 100, "default": 10},
        "max_iter": {"min": 1, "max": 1000, "default": 300},
    }
