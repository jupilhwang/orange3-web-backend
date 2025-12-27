"""
Test and Score Widget API endpoints.
Cross-validation accuracy estimation.
Based on Orange3's OWTestAndScore widget.
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
    from Orange.data import Table
    from Orange.evaluation import (
        CrossValidation, ShuffleSplit, LeaveOneOut, 
        TestOnTrainingData, TestOnTestData, Results
    )
    from Orange.evaluation.scoring import (
        CA, AUC, F1, Precision, Recall, 
        MAE, MSE, RMSE, R2
    )
    from Orange.modelling import KNNLearner, TreeLearner
    import numpy as np
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False

# Upload directory
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"

# In-memory storage for evaluation results
_evaluation_cache: Dict[str, Any] = {}

# Resampling methods
RESAMPLING_METHODS = {
    "cross_validation": "Cross Validation",
    "random_sampling": "Random Sampling", 
    "leave_one_out": "Leave One Out",
    "test_on_train": "Test on Train Data",
    "test_on_test": "Test on Test Data"
}

# Number of folds options
N_FOLDS_OPTIONS = [2, 3, 5, 10, 20]

# Number of repeats for random sampling
N_REPEATS_OPTIONS = [2, 3, 5, 10, 20, 50, 100]

# Sample sizes for random sampling (percentage)
SAMPLE_SIZES = [5, 10, 20, 25, 30, 33, 40, 50, 60, 66, 70, 75, 80, 90, 95]


class EvaluateRequest(BaseModel):
    """Request model for evaluation."""
    data_path: str
    test_data_path: Optional[str] = None
    learner_configs: List[Dict[str, Any]]  # List of learner configurations
    resampling: str = "cross_validation"  # cross_validation, random_sampling, leave_one_out, test_on_train, test_on_test
    n_folds: int = 5
    stratified: bool = True
    n_repeats: int = 10
    sample_size: int = 66  # percentage
    selected_indices: Optional[List[int]] = None


class EvaluateResponse(BaseModel):
    """Response model for evaluation."""
    success: bool
    evaluation_id: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    scores: Optional[List[Dict]] = None
    learner_names: Optional[List[str]] = None
    target_variable: Optional[str] = None
    is_classification: bool = True
    error: Optional[str] = None


def resolve_data_path(data_path: str) -> str:
    """Resolve data path to actual file path."""
    if data_path.startswith("uploads/"):
        return str(UPLOAD_DIR / data_path.replace("uploads/", ""))
    elif data_path.startswith("datasets/"):
        dataset_name = data_path.replace("datasets/", "").split(".")[0]
        return dataset_name
    return data_path


def create_learner(config: Dict[str, Any]):
    """Create a learner from configuration."""
    learner_type = config.get("type", "knn")
    
    if learner_type == "knn":
        return KNNLearner(
            n_neighbors=config.get("n_neighbors", 5),
            metric=config.get("metric", "euclidean"),
            weights=config.get("weights", "uniform")
        )
    elif learner_type == "tree":
        return TreeLearner(
            max_depth=config.get("max_depth", None),
            min_samples_split=config.get("min_samples_split", 2)
        )
    else:
        # Default to KNN
        return KNNLearner()


@router.post("/test_and_score/evaluate")
async def evaluate_models(request: EvaluateRequest) -> EvaluateResponse:
    """
    Evaluate models using various resampling methods.
    
    Parameters:
    - data_path: Path to training data
    - test_data_path: Path to test data (for test_on_test)
    - learner_configs: List of learner configurations
    - resampling: Resampling method
    - n_folds: Number of folds for cross validation
    - stratified: Whether to use stratified sampling
    - n_repeats: Number of repeats for random sampling
    - sample_size: Training set size percentage
    """
    if not ORANGE_AVAILABLE:
        return EvaluateResponse(
            success=False,
            error="Orange3 not available"
        )
    
    try:
        # Load data
        data_path = resolve_data_path(request.data_path)
        data = Table(data_path)
        
        # Filter by selected indices if provided
        if request.selected_indices and len(request.selected_indices) > 0:
            data = data[request.selected_indices]
        
        if len(data) == 0:
            return EvaluateResponse(
                success=False,
                error="No data to evaluate"
            )
        
        if not data.domain.class_var:
            return EvaluateResponse(
                success=False,
                error="Data must have a target variable"
            )
        
        is_classification = data.domain.class_var.is_discrete
        
        # Create learners
        learners = []
        learner_names = []
        for i, config in enumerate(request.learner_configs):
            learner = create_learner(config)
            learners.append(learner)
            learner_names.append(config.get("name", f"Model {i+1}"))
        
        if not learners:
            return EvaluateResponse(
                success=False,
                error="No learners provided"
            )
        
        # Create sampler based on resampling method
        if request.resampling == "cross_validation":
            sampler = CrossValidation(
                k=request.n_folds,
                stratified=request.stratified if is_classification else False
            )
        elif request.resampling == "random_sampling":
            sampler = ShuffleSplit(
                n_resamples=request.n_repeats,
                train_size=request.sample_size / 100,
                stratified=request.stratified if is_classification else False
            )
        elif request.resampling == "leave_one_out":
            sampler = LeaveOneOut()
        elif request.resampling == "test_on_train":
            sampler = TestOnTrainingData(store_models=True)
        elif request.resampling == "test_on_test":
            if not request.test_data_path:
                return EvaluateResponse(
                    success=False,
                    error="Test data path required for test_on_test"
                )
            test_data_path = resolve_data_path(request.test_data_path)
            test_data = Table(test_data_path)
            
            # Use TestOnTestData
            evaluator = TestOnTestData(store_data=True, store_models=True)
            results = evaluator(data, test_data, learners)
        else:
            return EvaluateResponse(
                success=False,
                error=f"Unknown resampling method: {request.resampling}"
            )
        
        # Run evaluation (except for test_on_test which was handled above)
        if request.resampling != "test_on_test":
            sampler.store_data = True
            results = sampler(data, learners)
        
        # Calculate scores
        scores = []
        
        if is_classification:
            # Classification scores
            score_funcs = [
                ("CA", CA),
                ("AUC", AUC),
                ("F1", F1),
                ("Precision", Precision),
                ("Recall", Recall)
            ]
        else:
            # Regression scores
            score_funcs = [
                ("MAE", MAE),
                ("MSE", MSE),
                ("RMSE", RMSE),
                ("R2", R2)
            ]
        
        for learner_idx, learner_name in enumerate(learner_names):
            learner_scores = {"name": learner_name}
            
            for score_name, score_func in score_funcs:
                try:
                    score_values = score_func(results)
                    if isinstance(score_values, np.ndarray):
                        learner_scores[score_name] = round(float(score_values[learner_idx]), 4)
                    else:
                        learner_scores[score_name] = round(float(score_values), 4)
                except Exception as e:
                    logger.warning(f"Score {score_name} failed: {e}")
                    learner_scores[score_name] = None
            
            scores.append(learner_scores)
        
        # Generate evaluation ID
        evaluation_id = str(uuid.uuid4())[:8]
        
        # Cache results
        _evaluation_cache[evaluation_id] = {
            "scores": scores,
            "learner_names": learner_names,
            "resampling": request.resampling,
            "data_path": request.data_path,
            "is_classification": is_classification
        }
        
        return EvaluateResponse(
            success=True,
            evaluation_id=evaluation_id,
            scores=scores,
            learner_names=learner_names,
            target_variable=data.domain.class_var.name,
            is_classification=is_classification
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return EvaluateResponse(
            success=False,
            error=str(e)
        )


@router.get("/test_and_score/options")
async def get_evaluation_options():
    """Get available options for evaluation configuration."""
    return {
        "resampling_methods": RESAMPLING_METHODS,
        "n_folds_options": N_FOLDS_OPTIONS,
        "n_repeats_options": N_REPEATS_OPTIONS,
        "sample_sizes": SAMPLE_SIZES,
        "classification_scores": ["CA", "AUC", "F1", "Precision", "Recall"],
        "regression_scores": ["MAE", "MSE", "RMSE", "R2"]
    }


@router.get("/test_and_score/{evaluation_id}")
async def get_evaluation_results(evaluation_id: str):
    """Get cached evaluation results by ID."""
    if evaluation_id not in _evaluation_cache:
        raise HTTPException(status_code=404, detail="Evaluation results not found")
    
    return _evaluation_cache[evaluation_id]


@router.delete("/test_and_score/{evaluation_id}")
async def delete_evaluation_results(evaluation_id: str):
    """Delete cached evaluation results."""
    if evaluation_id in _evaluation_cache:
        del _evaluation_cache[evaluation_id]
    return {"message": f"Evaluation {evaluation_id} deleted"}


@router.post("/test_and_score/compare")
async def compare_models(
    evaluation_id: str,
    comparison_method: str = "rope",
    rope_threshold: float = 0.1
):
    """
    Compare models using statistical tests.
    
    Parameters:
    - evaluation_id: ID of cached evaluation
    - comparison_method: Method for comparison (rope, bayesian)
    - rope_threshold: Threshold for Region of Practical Equivalence
    """
    if evaluation_id not in _evaluation_cache:
        raise HTTPException(status_code=404, detail="Evaluation results not found")
    
    cached = _evaluation_cache[evaluation_id]
    scores = cached["scores"]
    
    # Simple comparison matrix
    n_models = len(scores)
    comparison_matrix = []
    
    for i in range(n_models):
        row = []
        for j in range(n_models):
            if i == j:
                row.append("-")
            else:
                # Compare main score (CA for classification, R2 for regression)
                score_key = "CA" if cached.get("is_classification", True) else "R2"
                score_i = scores[i].get(score_key, 0) or 0
                score_j = scores[j].get(score_key, 0) or 0
                
                diff = score_i - score_j
                if abs(diff) < rope_threshold:
                    row.append("≈")  # Practically equivalent
                elif diff > 0:
                    row.append(">")  # Model i is better
                else:
                    row.append("<")  # Model j is better
        comparison_matrix.append(row)
    
    return {
        "comparison_matrix": comparison_matrix,
        "model_names": cached["learner_names"],
        "method": comparison_method,
        "rope_threshold": rope_threshold
    }

