"""
Neural Network Widget API endpoints.
Multi-Layer Perceptron (MLP) classifier.
"""

import logging
import uuid
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model", tags=["Model"])

from app.core.orange_compat import ORANGE_AVAILABLE

if ORANGE_AVAILABLE:
    from Orange.classification import NNClassificationLearner

# Model storage
_nn_models: Dict[str, Any] = {}
_nn_learners: Dict[str, Any] = {}


class NeuralNetworkTrainRequest(BaseModel):
    """Request model for Neural Network training."""

    data_path: str
    hidden_layers: List[int] = [100]  # neuron counts per hidden layer
    activation: str = "relu"  # relu, logistic, tanh
    alpha: float = 0.0001  # L2 regularization term
    learning_rate_init: float = 0.001  # initial learning rate
    max_iter: int = 200  # maximum number of iterations
    early_stopping: bool = False  # use early stopping
    name: str = "Neural Network"  # model name
    selected_indices: Optional[List[int]] = None


class NeuralNetworkTrainResponse(BaseModel):
    """Response model for Neural Network training."""

    success: bool
    model_id: Optional[str] = None
    learner_params: Optional[dict] = None
    model_info: Optional[dict] = None
    error: Optional[str] = None


class NeuralNetworkOptionsResponse(BaseModel):
    """Response model for Neural Network options."""

    activations: list
    default_activation: str
    alpha_range: dict
    learning_rate_range: dict
    max_iter_range: dict


@router.get("/neural-network/options", response_model=NeuralNetworkOptionsResponse)
async def get_neural_network_options():
    """Get available options for Neural Network configuration."""
    return NeuralNetworkOptionsResponse(
        activations=[
            {"value": "relu", "label": "ReLU"},
            {"value": "logistic", "label": "Logistic"},
            {"value": "tanh", "label": "Tanh"},
        ],
        default_activation="relu",
        alpha_range={"min": 0.00001, "max": 1.0, "default": 0.0001},
        learning_rate_range={"min": 0.0001, "max": 0.5, "default": 0.001},
        max_iter_range={"min": 10, "max": 1000, "default": 200},
    )


@router.post("/neural-network/train", response_model=NeuralNetworkTrainResponse)
async def train_neural_network(
    request: NeuralNetworkTrainRequest,
    x_session_id: Optional[str] = Header(None),
):
    """Train a Neural Network (MLP) classifier."""
    if not ORANGE_AVAILABLE:
        return NeuralNetworkTrainResponse(success=False, error="Orange3 not available")

    try:
        # Validate activation function
        valid_activations = ["relu", "logistic", "tanh"]
        if request.activation not in valid_activations:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid activation. Must be one of: {valid_activations}",
            )

        # Validate hidden layers
        if not request.hidden_layers:
            raise HTTPException(
                status_code=400, detail="hidden_layers must have at least one layer"
            )
        for layer_size in request.hidden_layers:
            if layer_size < 1:
                raise HTTPException(
                    status_code=400,
                    detail="Each hidden layer must have at least 1 neuron",
                )

        # Validate alpha (L2 regularization)
        if request.alpha < 0:
            raise HTTPException(
                status_code=400, detail="Alpha (regularization) must be non-negative"
            )

        # Validate learning rate
        if request.learning_rate_init <= 0:
            raise HTTPException(
                status_code=400, detail="Learning rate must be positive"
            )

        # Validate max_iter
        if request.max_iter < 1:
            raise HTTPException(status_code=400, detail="max_iter must be at least 1")

        # Load data
        from app.core.data_utils import async_load_data

        logger.info(
            f"Loading Neural Network data from: {request.data_path} (session: {x_session_id})"
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
            raise HTTPException(status_code=400, detail="No data to train on")

        if not data.domain.class_var:
            raise HTTPException(
                status_code=400,
                detail="Data must have a target variable for Neural Network",
            )

        # Build hidden layer sizes tuple
        hidden_layer_sizes = tuple(request.hidden_layers)

        # Create and train learner
        # Orange3's NNClassificationLearner wraps sklearn's MLPClassifier
        learner = NNClassificationLearner(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=request.activation,
            alpha=request.alpha,
            learning_rate_init=request.learning_rate_init,
            max_iter=request.max_iter,
            early_stopping=request.early_stopping,
        )
        learner.name = request.name

        model = learner(data)
        model.name = request.name

        # Store model
        model_id = str(uuid.uuid4())[:8]
        _nn_models[model_id] = model
        _nn_learners[model_id] = {
            "learner": learner,
            "hidden_layers": request.hidden_layers,
            "activation": request.activation,
            "alpha": request.alpha,
            "learning_rate_init": request.learning_rate_init,
            "max_iter": request.max_iter,
            "early_stopping": request.early_stopping,
        }

        # Build response
        learner_params = {
            "hidden_layers": request.hidden_layers,
            "activation": request.activation,
            "alpha": request.alpha,
            "learning_rate_init": request.learning_rate_init,
            "max_iter": request.max_iter,
            "early_stopping": request.early_stopping,
        }

        is_classification = not data.domain.class_var.is_continuous

        # Extract training score from the underlying sklearn model if available
        training_score = None
        n_iter_actual = None
        try:
            import numpy as np

            skl = getattr(model, "skl_model", None)
            if skl is not None:
                if hasattr(skl, "score"):
                    # Compute training accuracy on the converted data
                    X = np.array(
                        [[float(v) if v == v else 0.0 for v in inst.x] for inst in data]
                    )
                    y = np.array([float(inst.y) for inst in data])
                    training_score = float(skl.score(X, y))
                if hasattr(skl, "n_iter_"):
                    n_iter_actual = int(skl.n_iter_)
        except Exception as e:
            logger.debug(f"Suppressed error: {e}")

        model_info: Dict[str, Any] = {
            "name": request.name,
            "type": "classification" if is_classification else "regression",
            "training_instances": len(data),
            "features": len(data.domain.attributes),
            "target": data.domain.class_var.name,
            "target_values": list(data.domain.class_var.values)
            if is_classification
            else None,
            "hidden_layers": request.hidden_layers,
            "training_score": training_score,
            "n_iter": n_iter_actual,
        }

        return NeuralNetworkTrainResponse(
            success=True,
            model_id=model_id,
            learner_params=learner_params,
            model_info=model_info,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        return NeuralNetworkTrainResponse(success=False, error=str(e))


@router.get("/neural-network/info/{model_id}")
async def get_neural_network_info(model_id: str):
    """Get information about a trained Neural Network model."""
    if model_id not in _nn_models:
        raise HTTPException(status_code=404, detail="Model not found")

    model = _nn_models[model_id]
    meta = _nn_learners.get(model_id, {})

    info: Dict[str, Any] = {
        "model_id": model_id,
        "model_type": type(model).__name__,
        "learner_params": {
            "hidden_layers": meta.get("hidden_layers", [100]),
            "activation": meta.get("activation", "relu"),
            "alpha": meta.get("alpha", 0.0001),
            "learning_rate_init": meta.get("learning_rate_init", 0.001),
            "max_iter": meta.get("max_iter", 200),
            "early_stopping": meta.get("early_stopping", False),
        },
    }

    # Attach iteration count if available
    try:
        skl = getattr(model, "skl_model", None)
        if skl is not None and hasattr(skl, "n_iter_"):
            info["n_iter"] = int(skl.n_iter_)
    except Exception as e:
        logger.debug(f"Suppressed error: {e}")

    return info


@router.delete("/neural-network/{model_id}")
async def delete_neural_network_model(model_id: str):
    """Delete a trained Neural Network model."""
    if model_id not in _nn_models:
        raise HTTPException(status_code=404, detail="Model not found")

    del _nn_models[model_id]
    if model_id in _nn_learners:
        del _nn_learners[model_id]

    return {"success": True, "message": f"Neural Network model {model_id} deleted"}
