"""
PCA Widget API endpoints.
Principal Component Analysis for dimensionality reduction.
"""

import logging
import uuid
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

from app.core.data_utils import resolve_data_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model", tags=["Model"])

from app.core.orange_compat import ORANGE_AVAILABLE, Domain, ContinuousVariable

# In-memory storage for PCA results
_pca_results: Dict[str, Any] = {}


class PCARequest(BaseModel):
    """Request model for PCA."""

    data_path: str
    n_components: int = 2  # Number of components to show/output
    max_components: Optional[int] = None  # Max components to compute (None = auto)
    standardize: bool = (
        True  # Standardize features before PCA (zero mean, unit variance)
    )
    use_correlation: bool = False  # Use correlation matrix (vs covariance)
    color_column: Optional[str] = None  # Column name for coloring in biplot
    name: str = "PCA"
    selected_indices: Optional[List[int]] = None


class PCAResponse(BaseModel):
    """Response model for PCA."""

    success: bool
    pca_id: Optional[str] = None
    data_path: Optional[str] = None  # Path to transformed data
    pca_info: Optional[dict] = None
    error: Optional[str] = None


@router.post("/pca/transform")
async def transform_pca(
    request: PCARequest, x_session_id: Optional[str] = Header(None)
) -> PCAResponse:
    """
    Perform PCA transformation.

    Parameters:
    - data_path: Path to the input data
    - n_components: Number of principal components to output
    - max_components: Max components to compute for variance analysis (default: min(features, instances))
    - standardize: Standardize features (zero mean, unit variance)
    - use_correlation: Use correlation matrix instead of covariance
    - color_column: Column name for biplot coloring
    - selected_indices: Optional list of row indices to use
    """
    if not ORANGE_AVAILABLE:
        return PCAResponse(success=False, error="Orange3 not available")

    try:
        import numpy as np

        # Validate n_components
        if request.n_components < 1:
            raise HTTPException(
                status_code=400, detail="Number of components must be at least 1"
            )

        # Load data
        from app.core.data_utils import async_load_data

        logger.debug(
            f"[PCA] Loading data from {request.data_path} with session_id={x_session_id}"
        )
        data = await async_load_data(request.data_path, session_id=x_session_id)

        if data is None:
            raise HTTPException(
                status_code=404, detail=f"Data not found: {request.data_path}"
            )

        # Filter by selected indices if provided
        if request.selected_indices and len(request.selected_indices) > 0:
            data = data[request.selected_indices]

        if len(data) == 0:
            raise HTTPException(status_code=400, detail="No data to transform")

        # Get numeric features only
        numeric_attrs = [attr for attr in data.domain.attributes if attr.is_continuous]

        if len(numeric_attrs) == 0:
            raise HTTPException(
                status_code=400,
                detail="Data must have at least one numeric feature for PCA",
            )

        # Determine max components to compute
        n_numeric = len(numeric_attrs)
        n_samples = len(data)
        max_possible = min(n_numeric, n_samples)

        max_to_compute = request.max_components
        if max_to_compute is None:
            max_to_compute = max_possible
        else:
            max_to_compute = min(max_to_compute, max_possible)

        # Clamp n_components
        n_output = min(request.n_components, max_to_compute)

        # Build numeric-only domain for PCA
        numeric_domain = Domain(numeric_attrs)
        numeric_data = data.transform(numeric_domain)

        # Extract X matrix
        X = numeric_data.X.copy()

        # Handle missing values by mean imputation
        col_means = np.nanmean(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        # Standardize if requested (or use correlation matrix)
        if request.standardize or request.use_correlation:
            col_stds = np.std(X, axis=0, ddof=1)
            # Avoid division by zero for constant columns
            col_stds[col_stds == 0] = 1.0
            X_scaled = (X - np.mean(X, axis=0)) / col_stds
        else:
            X_scaled = X - np.mean(X, axis=0)

        # Perform PCA using numpy SVD (more control than Orange3's PCA wrapper)
        # This gives us full control over variance explained calculation
        cov_matrix = np.dot(X_scaled.T, X_scaled) / (len(X_scaled) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by descending eigenvalue
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # Clamp to max_to_compute
        eigenvalues = eigenvalues[:max_to_compute]
        eigenvectors = eigenvectors[:, :max_to_compute]

        # Explained variance ratio
        total_variance = np.sum(np.abs(eigenvalues))
        if total_variance > 0:
            explained_variance_ratio = np.abs(eigenvalues) / total_variance
        else:
            explained_variance_ratio = np.zeros(len(eigenvalues))

        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Project data onto principal components
        X_transformed = np.dot(X_scaled, eigenvectors[:, :n_output])

        # Build loadings matrix (component x feature)
        loadings = eigenvectors[:, :n_output].T  # shape: (n_output, n_numeric)

        # Build output data with PC columns
        pc_vars = [ContinuousVariable(f"PC{i + 1}") for i in range(n_output)]

        new_domain = Domain(
            data.domain.attributes,
            data.domain.class_vars,
            list(data.domain.metas) + pc_vars,
        )

        transformed_data = data.transform(new_domain)
        # Set PC values in meta columns
        for i, pc_var in enumerate(pc_vars):
            transformed_data[:, pc_var] = X_transformed[:, i : i + 1]

        # Generate unique ID
        pca_id = str(uuid.uuid4())[:8]
        data_path_key = f"pca/{pca_id}"

        # Store result in session-based storage
        from app.core.data_utils import DataSessionManager

        if x_session_id:
            await DataSessionManager.store(
                x_session_id, data_path_key, transformed_data
            )
            logger.debug(
                f"[PCA] Stored result in session {x_session_id}: {data_path_key}"
            )
        else:
            _pca_results[pca_id] = {
                "data": transformed_data,
                "original_data": data,
                "eigenvalues": eigenvalues,
                "eigenvectors": eigenvectors,
                "loadings": loadings,
                "n_output": n_output,
            }
            logger.debug(f"[PCA] Stored result in legacy storage: {pca_id}")

        # Build scree plot data (all computed components)
        scree_data = [
            {
                "component": i + 1,
                "explained_variance": float(
                    round(explained_variance_ratio[i] * 100, 2)
                ),
                "cumulative_variance": float(round(cumulative_variance[i] * 100, 2)),
                "eigenvalue": float(round(eigenvalues[i], 4)),
            }
            for i in range(len(eigenvalues))
        ]

        # Build loadings table for biplot (features x n_output components)
        feature_names = [attr.name for attr in numeric_attrs]
        loadings_data = []
        for fi, feat_name in enumerate(feature_names):
            row = {"feature": feat_name}
            for ci in range(n_output):
                row[f"PC{ci + 1}"] = float(round(float(loadings[ci, fi]), 4))
            loadings_data.append(row)

        pca_info = {
            "name": request.name,
            "n_components": n_output,
            "max_components": max_to_compute,
            "instances": len(data),
            "features": n_numeric,
            "total_features": len(data.domain.attributes),
            "standardized": request.standardize,
            "use_correlation": request.use_correlation,
            "scree_data": scree_data,
            "loadings": loadings_data,
            "feature_names": feature_names,
            "component_names": [f"PC{i + 1}" for i in range(n_output)],
        }

        return PCAResponse(
            success=True, pca_id=pca_id, data_path=data_path_key, pca_info=pca_info
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        return PCAResponse(success=False, error=str(e))


@router.get("/pca/options")
async def get_pca_options():
    """Get available options for PCA configuration."""
    return {
        "n_components": {"min": 1, "max": 100, "default": 2},
        "max_components": {
            "min": 1,
            "max": 100,
            "default": None,  # auto
        },
        "standardize": {
            "default": True,
            "description": "Standardize features to zero mean and unit variance",
        },
        "use_correlation": {
            "default": False,
            "description": "Use correlation matrix (standardizes automatically)",
        },
    }


@router.get("/pca/info/{pca_id}")
async def get_pca_info(pca_id: str):
    """Get information about a PCA result."""
    if pca_id not in _pca_results:
        raise HTTPException(status_code=404, detail="PCA result not found")

    result = _pca_results[pca_id]
    data = result["data"]
    eigenvalues = result["eigenvalues"]

    import numpy as np

    total_variance = np.sum(np.abs(eigenvalues))
    explained_ratios = (
        np.abs(eigenvalues) / total_variance
        if total_variance > 0
        else np.zeros_like(eigenvalues)
    )

    return {
        "pca_id": pca_id,
        "n_components": result["n_output"],
        "instances": len(data),
        "features": len(data.domain.attributes),
        "explained_variance_ratio": [
            round(float(r), 4) for r in explained_ratios[: result["n_output"]]
        ],
        "total_explained_variance": round(
            float(np.sum(explained_ratios[: result["n_output"]])) * 100, 2
        ),
    }


@router.delete("/pca/{pca_id}")
async def delete_pca_result(pca_id: str):
    """Delete a PCA result."""
    if pca_id in _pca_results:
        del _pca_results[pca_id]
    return {"message": f"PCA result {pca_id} deleted"}
