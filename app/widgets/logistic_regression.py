"""
Logistic Regression Learner Widget API endpoints.
Classification with LASSO (L1) or ridge (L2) regularization.
"""

import logging
import uuid
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model", tags=["Model"])

# Check Orange3 availability
try:
    from Orange.data import Table
    from Orange.classification.logistic_regression import LogisticRegressionLearner
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False

# Model storage
_lr_models: Dict[str, Any] = {}
_lr_learners: Dict[str, Any] = {}

# Upload directory
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"


def resolve_data_path(data_path: str) -> str:
    """Resolve data path to loadable format."""
    if data_path.startswith("uploads/"):
        return str(UPLOAD_DIR / data_path.replace("uploads/", ""))
    elif data_path.startswith("datasets/"):
        return data_path.replace("datasets/", "").split(".")[0]
    return data_path


class LogisticRegressionTrainRequest(BaseModel):
    """Request model for Logistic Regression training."""
    data_path: str
    penalty: str = "l2"  # l1, l2, or none
    C: float = 1.0  # Inverse of regularization strength
    class_weight: bool = False  # Balance class distribution
    max_iter: int = 10000


class LogisticRegressionTrainResponse(BaseModel):
    """Response model for Logistic Regression training."""
    success: bool
    model_id: Optional[str] = None
    message: Optional[str] = None
    coefficients: Optional[List[dict]] = None
    error: Optional[str] = None


class LogisticRegressionOptionsResponse(BaseModel):
    """Response model for Logistic Regression options."""
    penalties: List[dict]
    default_C: float = 1.0
    default_max_iter: int = 10000


@router.get("/logistic-regression/options", response_model=LogisticRegressionOptionsResponse)
async def get_logistic_regression_options():
    """Get Logistic Regression learner options and defaults."""
    return LogisticRegressionOptionsResponse(
        penalties=[
            {"value": "l1", "label": "Lasso (L1)"},
            {"value": "l2", "label": "Ridge (L2)"},
            {"value": "none", "label": "None"},
        ],
        default_C=1.0,
        default_max_iter=10000
    )


@router.post("/logistic-regression/train", response_model=LogisticRegressionTrainResponse)
async def train_logistic_regression(
    request: LogisticRegressionTrainRequest,
    x_session_id: Optional[str] = Header(None)
):
    """Train a Logistic Regression model."""
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Orange3 not available")
    
    try:
        # Load data using common utility
        from .data_utils import load_data
        logger.info(f"Loading Logistic Regression data from: {request.data_path} (session: {x_session_id})")
        data = load_data(request.data_path, session_id=x_session_id)
        
        if data is None:
            raise HTTPException(status_code=404, detail=f"Data not found: {request.data_path}")
        
        if data is None or len(data) == 0:
            return LogisticRegressionTrainResponse(
                success=False,
                error="Dataset is empty or could not be loaded"
            )
        
        if data.domain.class_var is None:
            return LogisticRegressionTrainResponse(
                success=False,
                error="Dataset must have a class variable for classification"
            )
        
        if data.domain.class_var.is_continuous:
            return LogisticRegressionTrainResponse(
                success=False,
                error="Logistic Regression is for classification only (discrete target required)"
            )
        
        # Parse penalty
        penalty = request.penalty.lower()
        if penalty == "none":
            penalty = None
        
        # Create Logistic Regression learner
        learner = LogisticRegressionLearner(
            penalty=penalty,
            C=request.C,
            class_weight="balanced" if request.class_weight else None,
            max_iter=request.max_iter,
            random_state=0
        )
        
        # Train model
        model = learner(data)
        
        # Extract coefficients if available
        coefficients = None
        try:
            if hasattr(model, 'coefficients') and hasattr(model, 'domain'):
                coefs = []
                for i, attr in enumerate(model.domain.attributes):
                    coefs.append({
                        "feature": attr.name,
                        "coefficient": float(model.coefficients[0][i]) if len(model.coefficients.shape) > 1 else float(model.coefficients[i])
                    })
                if hasattr(model, 'intercept'):
                    coefs.insert(0, {"feature": "intercept", "coefficient": float(model.intercept[0]) if hasattr(model.intercept, '__len__') else float(model.intercept)})
                coefficients = coefs
        except Exception as e:
            logger.warning(f"Could not extract coefficients: {e}")
        
        # Store model
        model_id = str(uuid.uuid4())[:8]
        _lr_models[model_id] = model
        _lr_learners[model_id] = {
            "learner": learner,
            "type": "logistic_regression",
            "training_instances": len(data),
            "features": len(data.domain.attributes),
            "target": data.domain.class_var.name,
            "target_values": list(data.domain.class_var.values) if hasattr(data.domain.class_var, 'values') else None,
            "penalty": request.penalty,
            "C": request.C,
        }
        
        return LogisticRegressionTrainResponse(
            success=True,
            model_id=model_id,
            message=f"Logistic Regression model trained successfully on {len(data)} instances",
            coefficients=coefficients
        )
        
    except Exception as e:
        logger.error(f"Logistic Regression training error: {e}")
        return LogisticRegressionTrainResponse(success=False, error=str(e))


@router.get("/logistic-regression/info/{model_id}")
async def get_logistic_regression_info(model_id: str):
    """Get Logistic Regression model information."""
    if model_id not in _lr_learners:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_data = _lr_learners[model_id]
    
    return {
        "success": True,
        "model_id": model_id,
        "type": model_data.get("type"),
        "training_instances": model_data.get("training_instances"),
        "features": model_data.get("features"),
        "target": model_data.get("target"),
        "target_values": model_data.get("target_values"),
        "penalty": model_data.get("penalty"),
        "C": model_data.get("C"),
    }


@router.delete("/logistic-regression/{model_id}")
async def delete_logistic_regression_model(model_id: str):
    """Delete a Logistic Regression model."""
    if model_id not in _lr_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    del _lr_models[model_id]
    if model_id in _lr_learners:
        del _lr_learners[model_id]
    
    return {"success": True, "message": "Model deleted"}

