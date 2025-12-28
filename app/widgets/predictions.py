"""
Predictions Widget API endpoints.
Display predictions of models for an input dataset.
Based on Orange3's OWPredictions widget.
"""

import logging
import uuid
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException
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

# Upload directory
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"

# In-memory storage for predictions
_predictions_cache: Dict[str, Any] = {}


class PredictRequest(BaseModel):
    """Request model for predictions."""
    data_path: str
    model_ids: List[str]  # List of model IDs to use for prediction
    show_probabilities: bool = True
    selected_indices: Optional[List[int]] = None


class PredictResponse(BaseModel):
    """Response model for predictions."""
    success: bool
    prediction_id: Optional[str] = None
    predictions: Optional[List[Dict]] = None
    columns: Optional[List[Dict]] = None
    model_names: Optional[List[str]] = None
    target_variable: Optional[str] = None
    target_values: Optional[List[str]] = None
    instances: int = 0
    error: Optional[str] = None


def resolve_data_path(data_path: str) -> str:
    """Resolve data path to actual file path."""
    if data_path.startswith("uploads/"):
        return str(UPLOAD_DIR / data_path.replace("uploads/", ""))
    elif data_path.startswith("datasets/"):
        dataset_name = data_path.replace("datasets/", "").split(".")[0]
        return dataset_name
    return data_path


@router.post("/predictions/predict")
async def make_predictions(request: PredictRequest) -> PredictResponse:
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
        from .knn import _knn_models
        from .tree import _tree_models
        from .naive_bayes import _nb_models
        from .logistic_regression import _lr_models
        from .random_forest import _rf_models
        
        # Collect all model storages
        all_models = {
            **_knn_models,
            **_tree_models,
            **_nb_models,
            **_lr_models,
            **_rf_models
        }
        
        # Load data using common utility
        from .data_utils import load_data
        data = load_data(request.data_path)
        
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
            model_names.append(f"Model_{model_id[:4]}")
        
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
        
        # Add prediction columns for each model
        for i, (model, model_name) in enumerate(zip(models, model_names)):
            columns.append({
                "id": f"pred_{i}",
                "name": f"{model_name} Prediction",
                "type": "prediction"
            })
            
            # Add probability columns if requested and classification
            if request.show_probabilities and target_var and target_var.is_discrete:
                for class_val in target_var.values:
                    columns.append({
                        "id": f"prob_{i}_{class_val}",
                        "name": f"{model_name} P({class_val})",
                        "type": "probability"
                    })
        
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
            if target_var:
                target_val = data[row_idx][target_var]
                if target_var.is_discrete:
                    row_data[target_var.name] = str(target_val) if target_val is not None else None
                else:
                    row_data[target_var.name] = float(target_val) if not np.isnan(target_val) else None
            
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
                        row_data[f"pred_{i}"] = round(float(pred), 4)
                except Exception as e:
                    row_data[f"pred_{i}"] = f"Error: {str(e)}"
            
            predictions_data.append(row_data)
        
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

