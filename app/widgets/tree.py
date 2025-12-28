"""
Tree Learner Widget API endpoints.
Decision Tree for classification and regression.
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
    from Orange.modelling.tree import TreeLearner
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False

# Model storage
_tree_models: Dict[str, Any] = {}
_tree_learners: Dict[str, Any] = {}

# Upload directory
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"


def resolve_data_path(data_path: str) -> str:
    """Resolve data path to loadable format."""
    if data_path.startswith("uploads/"):
        return str(UPLOAD_DIR / data_path.replace("uploads/", ""))
    elif data_path.startswith("datasets/"):
        return data_path.replace("datasets/", "").split(".")[0]
    return data_path


class TreeTrainRequest(BaseModel):
    """Request model for Tree training."""
    data_path: str
    binary_trees: bool = True
    max_depth: Optional[int] = 100
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    sufficient_majority: float = 0.95


class TreeTrainResponse(BaseModel):
    """Response model for Tree training."""
    success: bool
    model_id: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None


class TreeOptionsResponse(BaseModel):
    """Response model for Tree options."""
    default_binary_trees: bool = True
    default_max_depth: int = 100
    default_min_samples_split: int = 5
    default_min_samples_leaf: int = 2
    default_sufficient_majority: float = 0.95


@router.get("/tree/options", response_model=TreeOptionsResponse)
async def get_tree_options():
    """Get Tree learner options and defaults."""
    return TreeOptionsResponse()


@router.post("/tree/train", response_model=TreeTrainResponse)
async def train_tree(request: TreeTrainRequest):
    """Train a Decision Tree model."""
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Orange3 not available")
    
    try:
        # Load data using common utility
        from .data_utils import load_data
        data = load_data(request.data_path)
        
        if data is None:
            raise HTTPException(status_code=404, detail=f"Data not found: {request.data_path}")
        
        if data is None or len(data) == 0:
            return TreeTrainResponse(
                success=False,
                error="Dataset is empty or could not be loaded"
            )
        
        # Create Tree learner
        learner = TreeLearner(
            binarize=request.binary_trees,
            max_depth=request.max_depth,
            min_samples_split=request.min_samples_split,
            min_samples_leaf=request.min_samples_leaf,
            sufficient_majority=request.sufficient_majority
        )
        
        # Train model
        model = learner(data)
        
        # Store model
        model_id = str(uuid.uuid4())[:8]
        _tree_models[model_id] = model
        _tree_learners[model_id] = {
            "learner": learner,
            "type": "tree",
            "training_instances": len(data),
            "features": len(data.domain.attributes),
            "target": data.domain.class_var.name if data.domain.class_var else None,
        }
        
        return TreeTrainResponse(
            success=True,
            model_id=model_id,
            message=f"Tree model trained successfully on {len(data)} instances"
        )
        
    except Exception as e:
        logger.error(f"Tree training error: {e}")
        return TreeTrainResponse(success=False, error=str(e))


@router.get("/tree/info/{model_id}")
async def get_tree_info(model_id: str):
    """Get Tree model information."""
    if model_id not in _tree_learners:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_data = _tree_learners[model_id]
    
    return {
        "success": True,
        "model_id": model_id,
        "type": model_data.get("type"),
        "training_instances": model_data.get("training_instances"),
        "features": model_data.get("features"),
        "target": model_data.get("target"),
    }


@router.delete("/tree/{model_id}")
async def delete_tree_model(model_id: str):
    """Delete a Tree model."""
    if model_id not in _tree_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    del _tree_models[model_id]
    if model_id in _tree_learners:
        del _tree_learners[model_id]
    
    return {"success": True, "message": "Model deleted"}

