"""
kNN Widget API endpoints.
K-Nearest Neighbors classification and regression.
"""

import logging
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model", tags=["Model"])

# Check Orange3 availability
try:
    from Orange.data import Table
    from Orange.modelling import KNNLearner
    import numpy as np
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False

# Upload directory
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"


class KNNRequest(BaseModel):
    """Request model for kNN training."""
    data_path: str
    n_neighbors: int = 5
    metric: str = "euclidean"  # euclidean, manhattan, chebyshev, mahalanobis
    weights: str = "uniform"   # uniform, distance
    selected_indices: Optional[List[int]] = None


class KNNResponse(BaseModel):
    """Response model for kNN training."""
    success: bool
    model_id: Optional[str] = None
    learner_params: Optional[dict] = None
    model_info: Optional[dict] = None
    error: Optional[str] = None


# In-memory storage for trained models
_knn_models = {}
_knn_learners = {}


def resolve_data_path(data_path: str) -> str:
    """Resolve data path to actual file path."""
    if data_path.startswith("uploads/"):
        return str(UPLOAD_DIR / data_path.replace("uploads/", ""))
    elif data_path.startswith("datasets/"):
        # Built-in Orange3 datasets
        dataset_name = data_path.replace("datasets/", "").split(".")[0]
        return dataset_name
    return data_path


@router.post("/knn/train")
async def train_knn(request: KNNRequest) -> KNNResponse:
    """
    Train a kNN model.
    
    Parameters:
    - data_path: Path to the training data
    - n_neighbors: Number of neighbors (1-100)
    - metric: Distance metric (euclidean, manhattan, chebyshev, mahalanobis)
    - weights: Weight function (uniform, distance)
    - selected_indices: Optional list of row indices to use
    """
    if not ORANGE_AVAILABLE:
        return KNNResponse(
            success=False,
            error="Orange3 not available"
        )
    
    try:
        # Validate parameters
        if request.n_neighbors < 1 or request.n_neighbors > 100:
            raise HTTPException(
                status_code=400,
                detail="Number of neighbors must be between 1 and 100"
            )
        
        valid_metrics = ["euclidean", "manhattan", "chebyshev", "mahalanobis"]
        if request.metric not in valid_metrics:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid metric. Must be one of: {valid_metrics}"
            )
        
        valid_weights = ["uniform", "distance"]
        if request.weights not in valid_weights:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid weight. Must be one of: {valid_weights}"
            )
        
        # Load data
        data_path = resolve_data_path(request.data_path)
        data = Table(data_path)
        
        # Filter by selected indices if provided
        if request.selected_indices and len(request.selected_indices) > 0:
            data = data[request.selected_indices]
        
        if len(data) == 0:
            raise HTTPException(status_code=400, detail="No data to train on")
        
        if not data.domain.class_var:
            raise HTTPException(
                status_code=400,
                detail="Data must have a target variable for kNN"
            )
        
        # Check if we have enough samples
        if len(data) < request.n_neighbors:
            raise HTTPException(
                status_code=400,
                detail=f"Number of neighbors ({request.n_neighbors}) cannot be greater than number of samples ({len(data)})"
            )
        
        # Create learner
        learner = KNNLearner(
            n_neighbors=request.n_neighbors,
            metric=request.metric,
            weights=request.weights
        )
        
        # Train model
        model = learner(data)
        
        # Generate model ID
        import uuid
        model_id = str(uuid.uuid4())[:8]
        
        # Store model and learner
        _knn_models[model_id] = model
        _knn_learners[model_id] = learner
        
        # Prepare response
        learner_params = {
            "n_neighbors": request.n_neighbors,
            "metric": request.metric,
            "weights": request.weights
        }
        
        # Model info
        is_classification = not data.domain.class_var.is_continuous
        model_info = {
            "type": "classification" if is_classification else "regression",
            "training_instances": len(data),
            "features": len(data.domain.attributes),
            "target": data.domain.class_var.name,
            "target_values": list(data.domain.class_var.values) if is_classification else None
        }
        
        return KNNResponse(
            success=True,
            model_id=model_id,
            learner_params=learner_params,
            model_info=model_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return KNNResponse(
            success=False,
            error=str(e)
        )


@router.post("/knn/predict")
async def predict_knn(model_id: str, data_path: str, selected_indices: Optional[List[int]] = None):
    """
    Predict using a trained kNN model.
    """
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Orange3 not available")
    
    if model_id not in _knn_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        model = _knn_models[model_id]
        
        # Load data
        resolved_path = resolve_data_path(data_path)
        data = Table(resolved_path)
        
        # Filter by selected indices if provided
        if selected_indices and len(selected_indices) > 0:
            data = data[selected_indices]
        
        # Predict
        predictions = model(data)
        
        # Convert predictions to list
        pred_list = []
        for pred in predictions:
            if hasattr(pred, 'value'):
                pred_list.append(pred.value)
            else:
                pred_list.append(float(pred))
        
        return {
            "success": True,
            "predictions": pred_list,
            "count": len(pred_list)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/knn/info/{model_id}")
async def get_knn_info(model_id: str):
    """Get information about a trained kNN model."""
    if model_id not in _knn_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = _knn_models[model_id]
    learner = _knn_learners.get(model_id)
    
    info = {
        "model_id": model_id,
        "model_type": type(model).__name__,
    }
    
    if learner:
        try:
            params = learner.get_params() if hasattr(learner, 'get_params') else {}
            info["learner_params"] = {
                "n_neighbors": params.get("n_neighbors", 5),
                "metric": params.get("metric", "euclidean"),
                "weights": params.get("weights", "uniform")
            }
        except Exception:
            info["learner_params"] = {
                "n_neighbors": 5,
                "metric": "euclidean",
                "weights": "uniform"
            }
    
    return info


@router.delete("/knn/{model_id}")
async def delete_knn_model(model_id: str):
    """Delete a trained kNN model."""
    if model_id in _knn_models:
        del _knn_models[model_id]
    if model_id in _knn_learners:
        del _knn_learners[model_id]
    return {"message": f"Model {model_id} deleted"}


@router.get("/knn/options")
async def get_knn_options():
    """Get available options for kNN configuration."""
    return {
        "metrics": [
            {"value": "euclidean", "label": "Euclidean"},
            {"value": "manhattan", "label": "Manhattan"},
            {"value": "chebyshev", "label": "Chebyshev"},
            {"value": "mahalanobis", "label": "Mahalanobis"}
        ],
        "weights": [
            {"value": "uniform", "label": "Uniform"},
            {"value": "distance", "label": "By Distances"}
        ],
        "n_neighbors": {
            "min": 1,
            "max": 100,
            "default": 5
        }
    }

