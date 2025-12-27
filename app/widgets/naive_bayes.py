"""
Naive Bayes Learner Widget API endpoints.
Fast probabilistic classifier based on Bayes' theorem.
"""

import logging
import uuid
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model", tags=["Model"])

# Check Orange3 availability
try:
    from Orange.data import Table
    from Orange.classification.naive_bayes import NaiveBayesLearner
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False

# Model storage
_nb_models: Dict[str, Any] = {}
_nb_learners: Dict[str, Any] = {}

# Upload directory
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"


def resolve_data_path(data_path: str) -> str:
    """Resolve data path to loadable format."""
    if data_path.startswith("uploads/"):
        return str(UPLOAD_DIR / data_path.replace("uploads/", ""))
    elif data_path.startswith("datasets/"):
        return data_path.replace("datasets/", "").split(".")[0]
    return data_path


class NaiveBayesTrainRequest(BaseModel):
    """Request model for Naive Bayes training."""
    data_path: str


class NaiveBayesTrainResponse(BaseModel):
    """Response model for Naive Bayes training."""
    success: bool
    model_id: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None


@router.get("/naive-bayes/options")
async def get_naive_bayes_options():
    """Get Naive Bayes learner options (minimal - no hyperparameters)."""
    return {
        "description": "Naive Bayes classifier - no hyperparameters required"
    }


@router.post("/naive-bayes/train", response_model=NaiveBayesTrainResponse)
async def train_naive_bayes(request: NaiveBayesTrainRequest):
    """Train a Naive Bayes model."""
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Orange3 not available")
    
    try:
        # Load data
        data_path = resolve_data_path(request.data_path)
        data = Table(data_path)
        
        if data is None or len(data) == 0:
            return NaiveBayesTrainResponse(
                success=False,
                error="Dataset is empty or could not be loaded"
            )
        
        if data.domain.class_var is None:
            return NaiveBayesTrainResponse(
                success=False,
                error="Dataset must have a class variable for classification"
            )
        
        if data.domain.class_var.is_continuous:
            return NaiveBayesTrainResponse(
                success=False,
                error="Naive Bayes is for classification only (discrete target required)"
            )
        
        # Create Naive Bayes learner
        learner = NaiveBayesLearner()
        
        # Train model
        model = learner(data)
        
        # Store model
        model_id = str(uuid.uuid4())[:8]
        _nb_models[model_id] = model
        _nb_learners[model_id] = {
            "learner": learner,
            "type": "naive_bayes",
            "training_instances": len(data),
            "features": len(data.domain.attributes),
            "target": data.domain.class_var.name,
            "target_values": list(data.domain.class_var.values) if hasattr(data.domain.class_var, 'values') else None,
        }
        
        return NaiveBayesTrainResponse(
            success=True,
            model_id=model_id,
            message=f"Naive Bayes model trained successfully on {len(data)} instances"
        )
        
    except Exception as e:
        logger.error(f"Naive Bayes training error: {e}")
        return NaiveBayesTrainResponse(success=False, error=str(e))


@router.get("/naive-bayes/info/{model_id}")
async def get_naive_bayes_info(model_id: str):
    """Get Naive Bayes model information."""
    if model_id not in _nb_learners:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_data = _nb_learners[model_id]
    
    return {
        "success": True,
        "model_id": model_id,
        "type": model_data.get("type"),
        "training_instances": model_data.get("training_instances"),
        "features": model_data.get("features"),
        "target": model_data.get("target"),
        "target_values": model_data.get("target_values"),
    }


@router.delete("/naive-bayes/{model_id}")
async def delete_naive_bayes_model(model_id: str):
    """Delete a Naive Bayes model."""
    if model_id not in _nb_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    del _nb_models[model_id]
    if model_id in _nb_learners:
        del _nb_learners[model_id]
    
    return {"success": True, "message": "Model deleted"}

