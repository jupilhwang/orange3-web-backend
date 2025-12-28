"""
Random Forest Learner Widget API endpoints.
Ensemble of decision trees for classification and regression.
"""

import logging
import uuid
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model", tags=["Model"])

# Check Orange3 availability
try:
    from Orange.data import Table
    from Orange.modelling import RandomForestLearner
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False

# Model storage
_rf_models: Dict[str, Any] = {}
_rf_learners: Dict[str, Any] = {}


class RandomForestTrainRequest(BaseModel):
    """Request model for Random Forest training."""
    data_path: str
    n_estimators: int = 10  # Number of trees
    max_features: Optional[int] = None  # Number of features per split
    max_depth: Optional[int] = None  # Maximum tree depth
    min_samples_split: int = 2  # Minimum samples to split
    class_weight: bool = False  # Balance class distribution
    random_state: Optional[int] = None  # For reproducibility


class RandomForestTrainResponse(BaseModel):
    """Response model for Random Forest training."""
    success: bool
    model_id: Optional[str] = None
    message: Optional[str] = None
    feature_importances: Optional[List[dict]] = None
    error: Optional[str] = None


class RandomForestOptionsResponse(BaseModel):
    """Response model for Random Forest options."""
    default_n_estimators: int = 10
    default_min_samples_split: int = 2
    max_n_estimators: int = 10000
    max_max_depth: int = 100


@router.get("/random-forest/options", response_model=RandomForestOptionsResponse)
async def get_random_forest_options():
    """Get Random Forest learner options and defaults."""
    return RandomForestOptionsResponse()


@router.post("/random-forest/train", response_model=RandomForestTrainResponse)
async def train_random_forest(
    request: RandomForestTrainRequest,
    x_session_id: Optional[str] = Header(None)
):
    """Train a Random Forest model."""
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Orange3 not available")
    
    try:
        # Validate parameters
        if request.n_estimators < 1:
            return RandomForestTrainResponse(
                success=False,
                error="Number of estimators must be at least 1"
            )
        
        # Load data using common utility
        from .data_utils import load_data
        logger.info(f"Loading Random Forest data from: {request.data_path} (session: {x_session_id})")
        data = load_data(request.data_path, session_id=x_session_id)
        
        if data is None:
            raise HTTPException(status_code=404, detail=f"Data not found: {request.data_path}")
        
        if data is None or len(data) == 0:
            return RandomForestTrainResponse(
                success=False,
                error="Dataset is empty or could not be loaded"
            )
        
        # Build learner kwargs
        learner_kwargs = {
            "n_estimators": request.n_estimators,
            "min_samples_split": request.min_samples_split,
        }
        
        if request.max_features is not None:
            if request.max_features > len(data.domain.attributes):
                return RandomForestTrainResponse(
                    success=False,
                    error=f"max_features ({request.max_features}) exceeds number of attributes ({len(data.domain.attributes)})"
                )
            learner_kwargs["max_features"] = request.max_features
        
        if request.max_depth is not None:
            learner_kwargs["max_depth"] = request.max_depth
        
        if request.random_state is not None:
            learner_kwargs["random_state"] = request.random_state
        
        if request.class_weight:
            learner_kwargs["class_weight"] = "balanced"
        
        # Create Random Forest learner
        learner = RandomForestLearner(**learner_kwargs)
        
        # Train model
        model = learner(data)
        
        # Extract feature importances if available
        feature_importances = None
        try:
            if hasattr(model, 'skl_model') and hasattr(model.skl_model, 'feature_importances_'):
                importances = model.skl_model.feature_importances_
                feature_importances = [
                    {
                        "feature": attr.name,
                        "importance": float(importances[i])
                    }
                    for i, attr in enumerate(data.domain.attributes)
                ]
                # Sort by importance
                feature_importances.sort(key=lambda x: x["importance"], reverse=True)
        except Exception as e:
            logger.warning(f"Could not extract feature importances: {e}")
        
        # Store model
        model_id = str(uuid.uuid4())[:8]
        _rf_models[model_id] = model
        _rf_learners[model_id] = {
            "learner": learner,
            "type": "random_forest",
            "training_instances": len(data),
            "features": len(data.domain.attributes),
            "target": data.domain.class_var.name if data.domain.class_var else None,
            "n_estimators": request.n_estimators,
        }
        
        return RandomForestTrainResponse(
            success=True,
            model_id=model_id,
            message=f"Random Forest model trained successfully with {request.n_estimators} trees on {len(data)} instances",
            feature_importances=feature_importances
        )
        
    except Exception as e:
        logger.error(f"Random Forest training error: {e}")
        return RandomForestTrainResponse(success=False, error=str(e))


@router.get("/random-forest/info/{model_id}")
async def get_random_forest_info(model_id: str):
    """Get Random Forest model information."""
    if model_id not in _rf_learners:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_data = _rf_learners[model_id]
    
    return {
        "success": True,
        "model_id": model_id,
        "type": model_data.get("type"),
        "training_instances": model_data.get("training_instances"),
        "features": model_data.get("features"),
        "target": model_data.get("target"),
        "n_estimators": model_data.get("n_estimators"),
    }


@router.delete("/random-forest/{model_id}")
async def delete_random_forest_model(model_id: str):
    """Delete a Random Forest model."""
    if model_id not in _rf_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    del _rf_models[model_id]
    if model_id in _rf_learners:
        del _rf_learners[model_id]
    
    return {"success": True, "message": "Model deleted"}

