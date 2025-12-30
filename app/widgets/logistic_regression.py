"""
Logistic Regression Learner Widget API endpoints.
Classification with LASSO (L1) or ridge (L2) regularization.
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
    from Orange.classification.logistic_regression import LogisticRegressionLearner
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False

# Model storage
_lr_models: Dict[str, Any] = {}
_lr_learners: Dict[str, Any] = {}


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
    error_type: Optional[str] = None  # target_type, no_target, exception


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
                error="Data has no target variable.",
                error_type="no_target"
            )
        
        if data.domain.class_var.is_continuous:
            return LogisticRegressionTrainResponse(
                success=False,
                error="Categorical target variable expected.",
                error_type="target_type"
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


@router.get("/logistic-regression/coefficients/{model_id}")
async def get_logistic_regression_coefficients(
    model_id: str,
    sort_by: str = "name",  # name, coefficient, abs_coefficient
    class_index: int = 0  # For multi-class, which class coefficients to return
):
    """
    Get detailed coefficient information from a trained Logistic Regression model.
    
    Parameters:
    - model_id: ID of the trained model
    - sort_by: How to sort coefficients (name, coefficient, abs_coefficient)
    - class_index: For multi-class classification, which class's coefficients to show
    """
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Orange3 not available")
    
    if model_id not in _lr_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        import numpy as np
        
        model = _lr_models[model_id]
        model_info = _lr_learners.get(model_id, {})
        
        # Try to extract coefficients
        coefficients = []
        n_classes = 1
        class_names = model_info.get("target_values", [])
        
        # Get feature names
        if hasattr(model, 'domain'):
            feature_names = [attr.name for attr in model.domain.attributes]
        else:
            feature_names = [f"feature_{i}" for i in range(model_info.get("features", 0))]
        
        # Try different ways to access coefficients
        coefs = None
        intercept = None
        
        if hasattr(model, 'coefficients'):
            coefs = model.coefficients
            if hasattr(model, 'intercept'):
                intercept = model.intercept
        elif hasattr(model, 'skl_model'):
            if hasattr(model.skl_model, 'coef_'):
                coefs = model.skl_model.coef_
            if hasattr(model.skl_model, 'intercept_'):
                intercept = model.skl_model.intercept_
        
        if coefs is None:
            return {
                "success": False,
                "error": "Coefficients not available for this model"
            }
        
        # Handle multi-class vs binary
        if len(coefs.shape) > 1:
            n_classes = coefs.shape[0]
            class_coefs = coefs[min(class_index, n_classes - 1)]
        else:
            class_coefs = coefs
        
        # Add intercept
        if intercept is not None:
            if hasattr(intercept, '__len__') and len(intercept) > 0:
                int_val = float(intercept[min(class_index, len(intercept) - 1)])
            else:
                int_val = float(intercept)
            
            coefficients.append({
                "feature": "(intercept)",
                "coefficient": int_val,
                "abs_coefficient": abs(int_val),
                "is_intercept": True,
                "odds_ratio": np.exp(int_val) if abs(int_val) < 100 else None
            })
        
        # Add feature coefficients
        for name, coef in zip(feature_names, class_coefs):
            coef_val = float(coef)
            coefficients.append({
                "feature": name,
                "coefficient": coef_val,
                "abs_coefficient": abs(coef_val),
                "is_intercept": False,
                "odds_ratio": np.exp(coef_val) if abs(coef_val) < 100 else None
            })
        
        # Sort coefficients
        if sort_by == "coefficient":
            coefficients.sort(key=lambda x: x["coefficient"], reverse=True)
        elif sort_by == "abs_coefficient":
            coefficients.sort(key=lambda x: x["abs_coefficient"], reverse=True)
        else:
            # Sort by name, but keep intercept first
            non_intercept = [c for c in coefficients if not c.get("is_intercept")]
            intercept_list = [c for c in coefficients if c.get("is_intercept")]
            non_intercept.sort(key=lambda x: x["feature"])
            coefficients = intercept_list + non_intercept
        
        # Calculate statistics
        coef_values = [c["coefficient"] for c in coefficients if not c.get("is_intercept")]
        
        stats = {
            "n_features": len(coef_values),
            "mean": float(np.mean(coef_values)) if coef_values else 0,
            "std": float(np.std(coef_values)) if coef_values else 0,
            "max": float(max(coef_values)) if coef_values else 0,
            "min": float(min(coef_values)) if coef_values else 0,
            "n_positive": sum(1 for c in coef_values if c > 0),
            "n_negative": sum(1 for c in coef_values if c < 0),
            "n_zero": sum(1 for c in coef_values if abs(c) < 1e-10)
        }
        
        return {
            "success": True,
            "model_id": model_id,
            "coefficients": coefficients,
            "statistics": stats,
            "n_classes": n_classes,
            "class_names": class_names,
            "current_class_index": min(class_index, n_classes - 1),
            "sorted_by": sort_by
        }
        
    except Exception as e:
        logger.error(f"Coefficients error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

