"""
Predictions Widget API endpoints.
Display predictions of models for an input dataset.
Based on Orange3's OWPredictions widget.
"""

import logging
import uuid
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluate", tags=["Evaluate"])

# Check Orange3 availability
try:
    from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
    from Orange.evaluation import Results
    import numpy as np
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False

# In-memory storage for predictions
_predictions_cache: Dict[str, Any] = {}


def calculate_error(actual: float, predicted: float, error_type: str) -> Optional[float]:
    """Calculate error based on type."""
    if actual is None or predicted is None:
        return None
    
    diff = predicted - actual
    
    if error_type == "difference":
        return round(diff, 4)
    elif error_type == "absolute_difference":
        return round(abs(diff), 4)
    elif error_type == "relative":
        if actual != 0:
            return round(diff / actual, 4)
        return None
    elif error_type == "absolute_relative":
        if actual != 0:
            return round(abs(diff / actual), 4)
        return None
    else:
        return None


def calculate_regression_metrics(actuals: np.ndarray, predictions: np.ndarray, model_name: str) -> Dict:
    """Calculate regression performance metrics: MSE, RMSE, MAE, MAPE, R2."""
    n = len(actuals)
    if n == 0:
        return {"model": model_name}
    
    # MSE (Mean Squared Error)
    mse = np.mean((actuals - predictions) ** 2)
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(actuals - predictions))
    
    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    nonzero_mask = actuals != 0
    if np.any(nonzero_mask):
        mape = np.mean(np.abs((actuals[nonzero_mask] - predictions[nonzero_mask]) / actuals[nonzero_mask])) * 100
    else:
        mape = None
    
    # R2 (Coefficient of Determination)
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        "model": model_name,
        "mse": round(float(mse), 3),
        "rmse": round(float(rmse), 3),
        "mae": round(float(mae), 3),
        "mape": round(float(mape), 3) if mape is not None else None,
        "r2": round(float(r2), 3)
    }


class PredictRequest(BaseModel):
    """Request model for predictions."""
    data_path: str
    model_ids: List[str]  # List of model IDs to use for prediction
    show_probabilities: bool = True
    selected_indices: Optional[List[int]] = None
    regression_error_type: str = "none"  # none, difference, absolute_difference, relative, absolute_relative


class PredictResponse(BaseModel):
    """Response model for predictions."""
    success: bool
    prediction_id: Optional[str] = None
    predictions: Optional[List[Dict]] = None
    columns: Optional[List[Dict]] = None
    model_names: Optional[List[str]] = None
    target_variable: Optional[str] = None
    target_values: Optional[List[str]] = None
    is_regression: bool = False
    performance_scores: Optional[List[Dict]] = None  # MSE, RMSE, MAE, MAPE, R2
    instances: int = 0
    error: Optional[str] = None


@router.post("/predictions/predict")
async def make_predictions(
    request: PredictRequest,
    x_session_id: Optional[str] = Header(None)
) -> PredictResponse:
    """
    Make predictions using trained models.
    
    Parameters:
    - data_path: Path to the data
    - model_ids: List of model IDs (from kNN, Tree, etc.)
    - show_probabilities: Whether to include probability columns
    - selected_indices: Optional list of row indices to predict
    """
    if not ORANGE_AVAILABLE:
        return PredictResponse(
            success=False,
            error="Orange3 not available"
        )
    
    try:
        # Import model storage from other widgets
        from .knn import _knn_models, _knn_learners
        from .tree import _tree_models, _tree_learners
        from .naive_bayes import _nb_models, _nb_learners
        from .logistic_regression import _lr_models, _lr_learners
        from .random_forest import _rf_models, _rf_learners
        
        # Collect all model and learner storages
        all_models = {
            **_knn_models,
            **_tree_models,
            **_nb_models,
            **_lr_models,
            **_rf_models
        }
        
        # Map model_id to model type/name
        model_type_map = {}
        for model_id in _knn_models:
            model_type_map[model_id] = "kNN"
        for model_id in _tree_models:
            model_type_map[model_id] = "Tree"
        for model_id in _nb_models:
            model_type_map[model_id] = "Naive Bayes"
        for model_id in _lr_models:
            model_type_map[model_id] = "Logistic Regression"
        for model_id in _rf_models:
            model_type_map[model_id] = "Random Forest"
        
        # Load data using common utility
        from .data_utils import load_data
        logger.info(f"Loading Predictions data from: {request.data_path} (session: {x_session_id})")
        data = load_data(request.data_path, session_id=x_session_id)
        
        if data is None:
            raise HTTPException(status_code=404, detail=f"Data not found: {request.data_path}")
        
        # Filter by selected indices if provided
        if request.selected_indices and len(request.selected_indices) > 0:
            data = data[request.selected_indices]
        
        if len(data) == 0:
            return PredictResponse(
                success=False,
                error="No data to predict"
            )
        
        # Get models
        models = []
        model_names = []
        for model_id in request.model_ids:
            if model_id not in all_models:
                return PredictResponse(
                    success=False,
                    error=f"Model not found: {model_id}"
                )
            models.append(all_models[model_id])
            # Use actual model type name
            model_names.append(model_type_map.get(model_id, f"Model_{model_id[:4]}"))
        
        if not models:
            return PredictResponse(
                success=False,
                error="No models provided"
            )
        
        # Make predictions
        predictions_data = []
        columns = []
        
        # Add original data columns
        for attr in data.domain.attributes:
            columns.append({
                "id": attr.name,
                "name": attr.name,
                "type": "continuous" if attr.is_continuous else "discrete"
            })
        
        # Add target column if exists
        target_var = data.domain.class_var
        target_values = None
        if target_var:
            columns.append({
                "id": target_var.name,
                "name": target_var.name,
                "type": "target"
            })
            if target_var.is_discrete:
                target_values = list(target_var.values)
        
        # Determine if regression or classification
        is_regression = target_var is not None and target_var.is_continuous
        
        # Add prediction columns for each model
        for i, (model, model_name) in enumerate(zip(models, model_names)):
            columns.append({
                "id": f"pred_{i}",
                "name": model_name,
                "type": "prediction"
            })
            
            # Add error column for regression
            if is_regression and request.regression_error_type != "none":
                columns.append({
                    "id": f"error_{i}",
                    "name": "error",
                    "type": "error"
                })
            
            # Add probability columns if requested and classification
            if request.show_probabilities and target_var and target_var.is_discrete:
                for class_val in target_var.values:
                    columns.append({
                        "id": f"prob_{i}_{class_val}",
                        "name": f"{model_name} P({class_val})",
                        "type": "probability"
                    })
        
        # Storage for performance metrics calculation
        model_actuals = [[] for _ in models]
        model_predictions_arr = [[] for _ in models]
        
        # Generate predictions for each row
        for row_idx in range(len(data)):
            row_data = {}
            
            # Original features
            for attr in data.domain.attributes:
                val = data[row_idx][attr]
                if attr.is_continuous:
                    row_data[attr.name] = float(val) if not np.isnan(val) else None
                else:
                    row_data[attr.name] = str(val) if val is not None else None
            
            # Target value
            actual_val = None
            if target_var:
                target_val = data[row_idx][target_var]
                if target_var.is_discrete:
                    row_data[target_var.name] = str(target_val) if target_val is not None else None
                else:
                    actual_val = float(target_val) if not np.isnan(target_val) else None
                    row_data[target_var.name] = actual_val
            
            # Predictions from each model
            for i, model in enumerate(models):
                try:
                    pred = model(data[row_idx])
                    
                    if target_var and target_var.is_discrete:
                        # Classification
                        row_data[f"pred_{i}"] = str(target_var.values[int(pred)])
                        
                        # Probabilities
                        if request.show_probabilities:
                            try:
                                probs = model(data[row_idx], model.Probs)
                                for j, class_val in enumerate(target_var.values):
                                    row_data[f"prob_{i}_{class_val}"] = round(float(probs[j]), 4)
                            except:
                                for class_val in target_var.values:
                                    row_data[f"prob_{i}_{class_val}"] = None
                    else:
                        # Regression
                        pred_val = round(float(pred), 4)
                        row_data[f"pred_{i}"] = pred_val
                        
                        # Store for metrics calculation
                        if actual_val is not None:
                            model_actuals[i].append(actual_val)
                            model_predictions_arr[i].append(pred_val)
                        
                        # Calculate error if requested
                        if is_regression and request.regression_error_type != "none" and actual_val is not None:
                            error_val = calculate_error(actual_val, pred_val, request.regression_error_type)
                            row_data[f"error_{i}"] = error_val
                        elif is_regression and request.regression_error_type != "none":
                            row_data[f"error_{i}"] = None
                            
                except Exception as e:
                    row_data[f"pred_{i}"] = f"Error: {str(e)}"
            
            predictions_data.append(row_data)
        
        # Calculate performance scores for regression
        performance_scores = None
        if is_regression and len(model_actuals[0]) > 0:
            performance_scores = []
            for i, model_name in enumerate(model_names):
                actuals = np.array(model_actuals[i])
                preds = np.array(model_predictions_arr[i])
                
                scores = calculate_regression_metrics(actuals, preds, model_name)
                performance_scores.append(scores)
        
        # Generate prediction ID
        prediction_id = str(uuid.uuid4())[:8]
        
        # Cache predictions
        _predictions_cache[prediction_id] = {
            "data": predictions_data,
            "columns": columns,
            "model_names": model_names,
            "data_path": request.data_path
        }
        
        return PredictResponse(
            success=True,
            prediction_id=prediction_id,
            predictions=predictions_data,
            columns=columns,
            model_names=model_names,
            target_variable=target_var.name if target_var else None,
            target_values=target_values,
            is_regression=is_regression,
            performance_scores=performance_scores,
            instances=len(predictions_data)
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return PredictResponse(
            success=False,
            error=str(e)
        )


@router.get("/predictions/{prediction_id}")
async def get_predictions(prediction_id: str):
    """Get cached predictions by ID."""
    if prediction_id not in _predictions_cache:
        raise HTTPException(status_code=404, detail="Predictions not found")
    
    return _predictions_cache[prediction_id]


@router.delete("/predictions/{prediction_id}")
async def delete_predictions(prediction_id: str):
    """Delete cached predictions."""
    if prediction_id in _predictions_cache:
        del _predictions_cache[prediction_id]
    return {"message": f"Predictions {prediction_id} deleted"}

