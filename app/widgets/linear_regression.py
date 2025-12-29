"""
Linear Regression Widget API endpoints.
Ordinary Least Squares regression with optional regularization.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model", tags=["Model"])

# Check Orange3 availability
try:
    from Orange.data import Table
    from Orange.regression import LinearRegressionLearner, RidgeRegressionLearner, LassoRegressionLearner, ElasticNetLearner
    import numpy as np
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False


class LinearRegressionRequest(BaseModel):
    """Request model for Linear Regression training."""
    data_path: str
    fit_intercept: bool = True
    regularization_type: str = "none"  # none, ridge, lasso, elastic_net
    alpha: float = 0.0001  # Regularization strength
    l1_ratio: float = 0.5  # Elastic net mixing (0=L2, 1=L1)
    name: str = "Linear Regression"  # Model name
    selected_indices: Optional[List[int]] = None


class LinearRegressionResponse(BaseModel):
    """Response model for Linear Regression training."""
    success: bool
    model_id: Optional[str] = None
    learner_params: Optional[dict] = None
    model_info: Optional[dict] = None
    coefficients: Optional[List[dict]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None  # For categorizing errors (e.g., 'target_type')


# In-memory storage for trained models
_linear_models = {}
_linear_learners = {}


@router.post("/linear-regression/train")
async def train_linear_regression(
    request: LinearRegressionRequest,
    x_session_id: Optional[str] = Header(None)
) -> LinearRegressionResponse:
    """
    Train a Linear Regression model.
    
    Parameters:
    - data_path: Path to the training data
    - fit_intercept: Whether to fit the intercept
    - regularization_type: Type of regularization (none, ridge, lasso, elastic_net)
    - alpha: Regularization strength (0.0001 to 1000)
    - l1_ratio: Elastic net mixing ratio (0 to 1, 0=L2 only, 1=L1 only)
    - name: Model name
    - selected_indices: Optional list of row indices to use
    """
    if not ORANGE_AVAILABLE:
        return LinearRegressionResponse(
            success=False,
            error="Orange3 not available"
        )
    
    try:
        # Validate parameters
        valid_regularization = ["none", "ridge", "lasso", "elastic_net"]
        if request.regularization_type not in valid_regularization:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid regularization type. Must be one of: {valid_regularization}"
            )
        
        if request.alpha < 0:
            raise HTTPException(
                status_code=400,
                detail="Alpha must be non-negative"
            )
        
        if request.l1_ratio < 0 or request.l1_ratio > 1:
            raise HTTPException(
                status_code=400,
                detail="L1 ratio must be between 0 and 1"
            )
        
        # Load data using common utility
        from .data_utils import load_data
        logger.info(f"Loading Linear Regression data from: {request.data_path} (session: {x_session_id})")
        data = load_data(request.data_path, session_id=x_session_id)
        
        if data is None:
            raise HTTPException(status_code=404, detail=f"Data not found: {request.data_path}")
        
        # Filter by selected indices if provided
        if request.selected_indices and len(request.selected_indices) > 0:
            data = data[request.selected_indices]
        
        if len(data) == 0:
            raise HTTPException(status_code=400, detail="No data to train on")
        
        if not data.domain.class_var:
            return LinearRegressionResponse(
                success=False,
                error="Data has no target variable.",
                error_type="no_target"
            )
        
        # Check if target is continuous (regression) - Orange3 standard check
        if data.domain.class_var.is_discrete:
            return LinearRegressionResponse(
                success=False,
                error="Numeric target variable expected.",
                error_type="target_type"
            )
        
        # Create learner based on regularization type
        if request.regularization_type == "none":
            learner = LinearRegressionLearner(fit_intercept=request.fit_intercept)
        elif request.regularization_type == "ridge":
            learner = RidgeRegressionLearner(
                alpha=request.alpha,
                fit_intercept=request.fit_intercept
            )
        elif request.regularization_type == "lasso":
            learner = LassoRegressionLearner(
                alpha=request.alpha,
                fit_intercept=request.fit_intercept
            )
        elif request.regularization_type == "elastic_net":
            learner = ElasticNetLearner(
                alpha=request.alpha,
                l1_ratio=request.l1_ratio,
                fit_intercept=request.fit_intercept
            )
        
        learner.name = request.name
        
        # Train model
        model = learner(data)
        model.name = request.name
        
        # Generate model ID
        import uuid
        model_id = str(uuid.uuid4())[:8]
        
        # Store model and learner
        _linear_models[model_id] = model
        _linear_learners[model_id] = learner
        
        # Prepare response
        learner_params = {
            "fit_intercept": request.fit_intercept,
            "regularization_type": request.regularization_type,
            "alpha": request.alpha,
            "l1_ratio": request.l1_ratio
        }
        
        # Extract coefficients
        coefficients = []
        try:
            if hasattr(model, 'skl_model') and hasattr(model.skl_model, 'coef_'):
                coefs = model.skl_model.coef_
                intercept = model.skl_model.intercept_ if request.fit_intercept else 0
                
                # Feature names
                feature_names = [attr.name for attr in data.domain.attributes]
                
                for name, coef in zip(feature_names, coefs):
                    coefficients.append({
                        "feature": name,
                        "coefficient": float(coef)
                    })
                
                if request.fit_intercept:
                    coefficients.insert(0, {
                        "feature": "(intercept)",
                        "coefficient": float(intercept)
                    })
        except Exception as e:
            logger.warning(f"Could not extract coefficients: {e}")
        
        # Model info
        model_info = {
            "name": request.name,
            "type": "regression",
            "training_instances": len(data),
            "features": len(data.domain.attributes),
            "target": data.domain.class_var.name,
            "regularization": request.regularization_type
        }
        
        return LinearRegressionResponse(
            success=True,
            model_id=model_id,
            learner_params=learner_params,
            model_info=model_info,
            coefficients=coefficients
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return LinearRegressionResponse(
            success=False,
            error=str(e)
        )


@router.post("/linear-regression/predict")
async def predict_linear_regression(
    model_id: str,
    data_path: str,
    selected_indices: Optional[List[int]] = None,
    x_session_id: Optional[str] = Header(None)
):
    """
    Predict using a trained Linear Regression model.
    """
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Orange3 not available")
    
    if model_id not in _linear_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        model = _linear_models[model_id]
        
        # Load data using common utility
        from .data_utils import load_data
        data = load_data(data_path, session_id=x_session_id)
        
        if data is None:
            raise HTTPException(status_code=404, detail=f"Data not found: {data_path}")
        
        # Filter by selected indices if provided
        if selected_indices and len(selected_indices) > 0:
            data = data[selected_indices]
        
        # Predict
        predictions = model(data)
        
        # Convert predictions to list
        pred_list = []
        for pred in predictions:
            if hasattr(pred, 'value'):
                pred_list.append(float(pred.value))
            else:
                pred_list.append(float(pred))
        
        return {
            "success": True,
            "predictions": pred_list,
            "count": len(pred_list)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/linear-regression/info/{model_id}")
async def get_linear_regression_info(model_id: str):
    """Get information about a trained Linear Regression model."""
    if model_id not in _linear_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = _linear_models[model_id]
    learner = _linear_learners.get(model_id)
    
    info = {
        "model_id": model_id,
        "model_type": type(model).__name__,
        "name": getattr(model, 'name', 'Linear Regression')
    }
    
    # Extract coefficients
    try:
        if hasattr(model, 'skl_model') and hasattr(model.skl_model, 'coef_'):
            info["coefficients"] = model.skl_model.coef_.tolist()
            if hasattr(model.skl_model, 'intercept_'):
                info["intercept"] = float(model.skl_model.intercept_)
    except Exception:
        pass
    
    return info


@router.delete("/linear-regression/{model_id}")
async def delete_linear_regression_model(model_id: str):
    """Delete a trained Linear Regression model."""
    if model_id in _linear_models:
        del _linear_models[model_id]
    if model_id in _linear_learners:
        del _linear_learners[model_id]
    return {"message": f"Model {model_id} deleted"}


@router.get("/linear-regression/options")
async def get_linear_regression_options():
    """Get available options for Linear Regression configuration."""
    return {
        "regularization_types": [
            {"value": "none", "label": "No regularization"},
            {"value": "ridge", "label": "Ridge regression (L2)"},
            {"value": "lasso", "label": "Lasso regression (L1)"},
            {"value": "elastic_net", "label": "Elastic net regression"}
        ],
        "alpha": {
            "min": 0.0001,
            "max": 1000,
            "default": 0.0001
        },
        "l1_ratio": {
            "min": 0,
            "max": 1,
            "default": 0.5
        }
    }


# Export models dict for predictions widget
def get_model(model_id: str):
    """Get a trained model by ID."""
    return _linear_models.get(model_id)


def get_learner(model_id: str):
    """Get a learner by model ID."""
    return _linear_learners.get(model_id)

