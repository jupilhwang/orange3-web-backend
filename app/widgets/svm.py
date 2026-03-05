"""
SVM Widget API endpoints.
Support Vector Machine for classification.
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
    from Orange.classification import SVMLearner

# Model storage
_svm_models: Dict[str, Any] = {}
_svm_learners: Dict[str, Any] = {}


class SVMTrainRequest(BaseModel):
    """Request model for SVM training."""

    data_path: str
    kernel: str = "RBF"  # linear, RBF, polynomial, sigmoid
    C: float = 1.0  # Regularization parameter
    gamma: str = "auto"  # auto, scale, or float value as string
    degree: int = 3  # Degree for polynomial kernel
    coef0: float = 0.0  # Coef0 for polynomial/sigmoid kernels
    probability: bool = False  # Enable probability estimates
    shrinking: bool = True  # Use shrinking heuristic
    tol: float = 0.001  # Tolerance for stopping criterion
    max_iter: int = -1  # Max iterations (-1 = unlimited)
    name: str = "SVM"  # Model name
    selected_indices: Optional[List[int]] = None


class SVMTrainResponse(BaseModel):
    """Response model for SVM training."""

    success: bool
    model_id: Optional[str] = None
    learner_params: Optional[dict] = None
    model_info: Optional[dict] = None
    error: Optional[str] = None


class SVMOptionsResponse(BaseModel):
    """Response model for SVM options."""

    kernels: list
    default_kernel: str
    C_range: dict
    gamma_options: list
    degree_range: dict


@router.get("/svm/options", response_model=SVMOptionsResponse)
async def get_svm_options():
    """Get available options for SVM configuration."""
    return SVMOptionsResponse(
        kernels=[
            {"value": "RBF", "label": "RBF"},
            {"value": "linear", "label": "Linear"},
            {"value": "polynomial", "label": "Polynomial"},
            {"value": "sigmoid", "label": "Sigmoid"},
        ],
        default_kernel="RBF",
        C_range={"min": 0.01, "max": 100.0, "default": 1.0},
        gamma_options=[
            {"value": "auto", "label": "Auto"},
            {"value": "scale", "label": "Scale"},
        ],
        degree_range={"min": 1, "max": 10, "default": 3},
    )


@router.post("/svm/train", response_model=SVMTrainResponse)
async def train_svm(
    request: SVMTrainRequest, x_session_id: Optional[str] = Header(None)
):
    """Train a Support Vector Machine model."""
    if not ORANGE_AVAILABLE:
        return SVMTrainResponse(success=False, error="Orange3 not available")

    try:
        # Validate kernel
        valid_kernels = ["RBF", "linear", "polynomial", "sigmoid"]
        if request.kernel not in valid_kernels:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid kernel. Must be one of: {valid_kernels}",
            )

        # Validate C
        if request.C <= 0:
            raise HTTPException(
                status_code=400, detail="C (regularization) must be positive"
            )

        # Validate degree for polynomial kernel
        if request.degree < 1 or request.degree > 10:
            raise HTTPException(
                status_code=400, detail="Degree must be between 1 and 10"
            )

        # Resolve gamma value
        gamma_value: Any
        if request.gamma in ("auto", "scale"):
            gamma_value = request.gamma
        else:
            try:
                gamma_value = float(request.gamma)
                if gamma_value <= 0:
                    raise HTTPException(
                        status_code=400, detail="Gamma value must be positive"
                    )
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=400,
                    detail="Gamma must be 'auto', 'scale', or a positive float",
                )

        # Load data
        from app.core.data_utils import async_load_data

        logger.info(
            f"Loading SVM data from: {request.data_path} (session: {x_session_id})"
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
                status_code=400, detail="Data must have a target variable for SVM"
            )

        # Map kernel name to Orange3 convention
        kernel_map = {
            "RBF": "rbf",
            "linear": "linear",
            "polynomial": "poly",
            "sigmoid": "sigmoid",
        }
        orange_kernel = kernel_map[request.kernel]

        # Build learner kwargs
        learner_kwargs: Dict[str, Any] = {
            "C": request.C,
            "kernel": orange_kernel,
            "gamma": gamma_value,
            "degree": request.degree,
            "coef0": request.coef0,
            "probability": request.probability,
            "shrinking": request.shrinking,
            "tol": request.tol,
        }
        if request.max_iter != -1:
            learner_kwargs["max_iter"] = request.max_iter

        # Create and train learner
        learner = SVMLearner(**learner_kwargs)
        learner.name = request.name

        model = learner(data)
        model.name = request.name

        # Store model
        model_id = str(uuid.uuid4())[:8]
        _svm_models[model_id] = model
        _svm_learners[model_id] = {
            "learner": learner,
            "kernel": request.kernel,
            "C": request.C,
            "gamma": request.gamma,
            "degree": request.degree,
            "probability": request.probability,
        }

        # Build response
        learner_params = {
            "kernel": request.kernel,
            "C": request.C,
            "gamma": request.gamma,
            "degree": request.degree,
            "coef0": request.coef0,
            "probability": request.probability,
        }

        is_classification = not data.domain.class_var.is_continuous

        # Count support vectors when accessible
        support_vectors_count = None
        try:
            skl = getattr(model, "skl_model", None)
            if skl is not None and hasattr(skl, "support_vectors_"):
                support_vectors_count = int(skl.support_vectors_.shape[0])
        except Exception as e:
            logger.debug(f"Suppressed error: {e}")

        model_info = {
            "name": request.name,
            "type": "classification" if is_classification else "regression",
            "training_instances": len(data),
            "features": len(data.domain.attributes),
            "target": data.domain.class_var.name,
            "target_values": list(data.domain.class_var.values)
            if is_classification
            else None,
            "support_vectors": support_vectors_count,
        }

        return SVMTrainResponse(
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
        return SVMTrainResponse(success=False, error=str(e))


@router.get("/svm/info/{model_id}")
async def get_svm_info(model_id: str):
    """Get information about a trained SVM model."""
    if model_id not in _svm_models:
        raise HTTPException(status_code=404, detail="Model not found")

    model = _svm_models[model_id]
    meta = _svm_learners.get(model_id, {})

    info: Dict[str, Any] = {
        "model_id": model_id,
        "model_type": type(model).__name__,
        "learner_params": {
            "kernel": meta.get("kernel", "RBF"),
            "C": meta.get("C", 1.0),
            "gamma": meta.get("gamma", "auto"),
            "degree": meta.get("degree", 3),
            "probability": meta.get("probability", False),
        },
    }

    # Attach support vector count if available
    try:
        skl = getattr(model, "skl_model", None)
        if skl is not None and hasattr(skl, "support_vectors_"):
            info["support_vectors"] = int(skl.support_vectors_.shape[0])
            info["n_support"] = [int(x) for x in skl.n_support_]
    except Exception as e:
        logger.debug(f"Suppressed error: {e}")

    return info


@router.delete("/svm/{model_id}")
async def delete_svm_model(model_id: str):
    """Delete a trained SVM model."""
    if model_id not in _svm_models:
        raise HTTPException(status_code=404, detail="Model not found")

    del _svm_models[model_id]
    if model_id in _svm_learners:
        del _svm_learners[model_id]

    return {"success": True, "message": f"SVM model {model_id} deleted"}
