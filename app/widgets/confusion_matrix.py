"""
Confusion Matrix Widget API endpoints.
Display a confusion matrix from classifier evaluation results.
"""

import logging
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluate", tags=["Evaluate"])

# Check Orange3 availability
try:
    import numpy as np
    from Orange.data import Table
    from Orange.evaluation import Results
    import sklearn.metrics as skl_metrics
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False


class ConfusionMatrixRequest(BaseModel):
    """Request model for confusion matrix computation."""
    results_id: str
    learner_index: int = 0
    quantity: str = "instances"  # instances, predicted, actual, probabilities


class ConfusionMatrixResponse(BaseModel):
    """Response model for confusion matrix."""
    success: bool
    matrix: Optional[List[List[Any]]] = None
    headers: Optional[List[str]] = None
    learners: Optional[List[str]] = None
    row_sums: Optional[List[int]] = None
    col_sums: Optional[List[int]] = None
    total: Optional[int] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class SelectionRequest(BaseModel):
    """Request model for selecting data from confusion matrix."""
    results_id: str
    learner_index: int
    selected_cells: List[List[int]]  # [[row, col], ...]
    append_predictions: bool = True
    append_probabilities: bool = False


class SelectionResponse(BaseModel):
    """Response model for selected data."""
    success: bool
    selected_count: int = 0
    data_path: Optional[str] = None
    error: Optional[str] = None


# Store for evaluation results
_eval_results: Dict[str, Any] = {}


def compute_confusion_matrix(results, learner_index: int):
    """Compute confusion matrix from evaluation results."""
    labels = np.arange(len(results.domain.class_var.values))
    if not results.actual.size:
        return np.zeros((len(labels), len(labels)))
    else:
        return skl_metrics.confusion_matrix(
            results.actual, results.predicted[learner_index], labels=labels)


@router.post("/confusion-matrix/compute", response_model=ConfusionMatrixResponse)
async def compute_matrix(
    request: ConfusionMatrixRequest,
    x_session_id: Optional[str] = Header(None)
):
    """Compute confusion matrix from evaluation results."""
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Orange3 not available")
    
    try:
        # Get results from cache
        from .test_and_score import _test_results
        
        results_data = _test_results.get(request.results_id)
        if not results_data:
            return ConfusionMatrixResponse(
                success=False,
                error="Evaluation results not found",
                error_type="not_found"
            )
        
        results = results_data.get('results')
        if results is None:
            return ConfusionMatrixResponse(
                success=False,
                error="Invalid evaluation results",
                error_type="invalid_results"
            )
        
        # Check for discrete class
        if not results.domain.has_discrete_class:
            return ConfusionMatrixResponse(
                success=False,
                error="Confusion Matrix cannot show regression results.",
                error_type="regression"
            )
        
        # Get class values
        class_values = list(results.domain.class_var.values)
        
        # Get learner names
        learner_names = getattr(results, 'learner_names', 
                               [f"Learner #{i+1}" for i in range(results.predicted.shape[0])])
        
        # Compute confusion matrix
        learner_idx = min(request.learner_index, len(learner_names) - 1)
        
        if request.quantity == "probabilities" and results.probabilities is not None:
            # Sum of probabilities
            probabilities = results.probabilities[learner_idx]
            n = probabilities.shape[1]
            cmatrix = np.zeros((n, n), dtype=float)
            for index in np.unique(results.actual).astype(int):
                mask = results.actual == index
                cmatrix[index] = np.sum(probabilities[mask], axis=0)
        else:
            cmatrix = compute_confusion_matrix(results, learner_idx)
        
        # Calculate sums
        col_sums = cmatrix.sum(axis=0).astype(int).tolist()
        row_sums = cmatrix.sum(axis=1).astype(int).tolist()
        total = int(cmatrix.sum())
        
        # Format matrix based on quantity
        if request.quantity == "instances":
            matrix = cmatrix.astype(int).tolist()
        elif request.quantity == "predicted":
            # Proportion of predicted (column-wise)
            col_sums_arr = cmatrix.sum(axis=0)
            col_sums_arr[col_sums_arr == 0] = 1
            matrix = (100 * cmatrix / col_sums_arr).round(1).tolist()
        elif request.quantity == "actual":
            # Proportion of actual (row-wise)
            row_sums_arr = cmatrix.sum(axis=1)[:, np.newaxis]
            row_sums_arr[row_sums_arr == 0] = 1
            matrix = (100 * cmatrix / row_sums_arr).round(1).tolist()
        elif request.quantity == "probabilities":
            matrix = cmatrix.round(1).tolist()
        else:
            matrix = cmatrix.astype(int).tolist()
        
        return ConfusionMatrixResponse(
            success=True,
            matrix=matrix,
            headers=class_values,
            learners=learner_names,
            row_sums=row_sums,
            col_sums=col_sums,
            total=total
        )
        
    except Exception as e:
        logger.error(f"Confusion matrix computation error: {e}")
        return ConfusionMatrixResponse(
            success=False,
            error=str(e),
            error_type="exception"
        )


@router.post("/confusion-matrix/select", response_model=SelectionResponse)
async def select_data(
    request: SelectionRequest,
    x_session_id: Optional[str] = Header(None)
):
    """Select data instances based on confusion matrix cell selection."""
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Orange3 not available")
    
    try:
        from .test_and_score import _test_results
        from .data_utils import save_data
        
        results_data = _test_results.get(request.results_id)
        if not results_data:
            return SelectionResponse(success=False, error="Results not found")
        
        results = results_data.get('results')
        data = results.data[results.row_indices] if results.data is not None else None
        
        if data is None:
            return SelectionResponse(success=False, error="No data available")
        
        # Get selected indices based on cell selection
        actual = results.actual
        predicted = results.predicted[request.learner_index]
        
        selected_cells = set(tuple(cell) for cell in request.selected_cells)
        selected_indices = [
            i for i, (a, p) in enumerate(zip(actual, predicted))
            if (int(a), int(p)) in selected_cells
        ]
        
        if not selected_indices:
            return SelectionResponse(success=True, selected_count=0)
        
        # Create selected data
        selected_data = data[selected_indices]
        
        # Save to session
        import uuid
        data_id = f"confusion_selection_{uuid.uuid4().hex[:8]}"
        save_data(data_id, selected_data, session_id=x_session_id)
        
        return SelectionResponse(
            success=True,
            selected_count=len(selected_indices),
            data_path=data_id
        )
        
    except Exception as e:
        logger.error(f"Selection error: {e}")
        return SelectionResponse(success=False, error=str(e))


@router.get("/confusion-matrix/options")
async def get_options():
    """Get confusion matrix display options."""
    return {
        "quantities": [
            {"id": "instances", "label": "Number of instances"},
            {"id": "predicted", "label": "Proportion of predicted"},
            {"id": "actual", "label": "Proportion of actual"},
            {"id": "probabilities", "label": "Sum of probabilities"}
        ]
    }


