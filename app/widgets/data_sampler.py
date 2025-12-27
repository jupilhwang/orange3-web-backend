"""
Data Sampler Widget API endpoints.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["Data"])

# Check Orange3 availability
try:
    from Orange.data import Table
    import numpy as np
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False

# Upload directory
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"


class SampleDataRequest(BaseModel):
    """Request model for data sampling."""
    data_path: str
    sampling_type: int = 0  # 0=FixedProportion, 1=FixedSize, 2=CrossValidation, 3=Bootstrap
    sample_percentage: int = 70
    sample_size: int = 1
    replacement: bool = False
    number_of_folds: int = 10
    selected_fold: int = 1
    use_seed: bool = True
    stratify: bool = False


@router.post("/sample")
async def sample_data(request: SampleDataRequest):
    """
    Sample data using Orange3's sampling algorithms.
    
    Sampling Types:
    - 0: Fixed proportion (percentage)
    - 1: Fixed sample size (number of instances)
    - 2: Cross validation (k-fold)
    - 3: Bootstrap
    """
    if not ORANGE_AVAILABLE:
        return calculate_sample_fallback(request)
    
    try:
        from Orange.data import Table
        import numpy as np
        import math
        
        # Constants
        RANDOM_SEED = 42
        FixedProportion, FixedSize, CrossValidation, Bootstrap = range(4)
        
        # Resolve data path
        data_path = request.data_path
        if data_path.startswith("uploads/"):
            data_path = str(UPLOAD_DIR / data_path.replace("uploads/", ""))
        elif data_path.startswith("datasets/"):
            dataset_name = data_path.replace("datasets/", "").split(".")[0]
            data_path = dataset_name
        
        # Load data
        data = Table(data_path)
        data_length = len(data)
        
        if data_length == 0:
            raise HTTPException(status_code=400, detail="Dataset is empty")
        
        # Determine random state
        rnd = RANDOM_SEED if request.use_seed else None
        
        # Perform sampling based on type
        sample_indices = None
        remaining_indices = None
        
        if request.sampling_type == FixedProportion:
            size = int(math.ceil(request.sample_percentage / 100 * data_length))
            sample_indices, remaining_indices = sample_random_n(
                data, size, stratified=request.stratify, replace=False, random_state=rnd
            )
            
        elif request.sampling_type == FixedSize:
            size = request.sample_size
            if not request.replacement and size > data_length:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Sample size ({size}) cannot be larger than data size ({data_length}) without replacement"
                )
            sample_indices, remaining_indices = sample_random_n(
                data, size, stratified=request.stratify, replace=request.replacement, random_state=rnd
            )
            
        elif request.sampling_type == CrossValidation:
            if data_length < request.number_of_folds:
                raise HTTPException(
                    status_code=400,
                    detail=f"Number of folds ({request.number_of_folds}) exceeds data size ({data_length})"
                )
            folds = sample_fold_indices(
                data, request.number_of_folds, stratified=request.stratify, random_state=rnd
            )
            sample_indices, remaining_indices = folds[request.selected_fold - 1]
            
        elif request.sampling_type == Bootstrap:
            sample_indices, remaining_indices = sample_bootstrap(data_length, random_state=rnd)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown sampling type: {request.sampling_type}")
        
        # Calculate counts
        sample_count = len(sample_indices) if sample_indices is not None else 0
        remaining_count = len(remaining_indices) if remaining_indices is not None else 0
        
        return {
            "success": True,
            "sample_count": sample_count,
            "remaining_count": remaining_count,
            "total_count": data_length,
            "sampling_type": request.sampling_type,
            "sample_indices": sample_indices.tolist() if sample_indices is not None else [],
            "remaining_indices": remaining_indices.tolist() if remaining_indices is not None else []
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


def sample_random_n(data, n, stratified=False, replace=False, random_state=None):
    """Sample n instances from data."""
    import numpy as np
    import sklearn.model_selection as skl
    
    data_length = len(data)
    
    if replace:
        rgen = np.random.RandomState(random_state)
        sample = rgen.randint(0, data_length, n)
        others = np.ones(data_length)
        others[sample] = 0
        remaining = np.nonzero(others)[0]
        return sample, remaining
    
    if n == 0:
        rgen = np.random.RandomState(random_state)
        shuffled = np.arange(data_length)
        rgen.shuffle(shuffled)
        return np.array([], dtype=int), shuffled
    
    if n >= data_length:
        rgen = np.random.RandomState(random_state)
        shuffled = np.arange(data_length)
        rgen.shuffle(shuffled)
        return shuffled, np.array([], dtype=int)
    
    if stratified and data.domain.has_discrete_class:
        try:
            test_size = max(len(data.domain.class_var.values), n)
            splitter = skl.StratifiedShuffleSplit(
                n_splits=1, test_size=test_size,
                train_size=data_length - test_size,
                random_state=random_state
            )
            splitter.get_n_splits(data.X, data.Y)
            ind = splitter.split(data.X, data.Y)
            remaining, sample = next(iter(ind))
            return sample, remaining
        except:
            pass
    
    splitter = skl.ShuffleSplit(n_splits=1, test_size=n, random_state=random_state)
    splitter.get_n_splits(data)
    ind = splitter.split(data)
    remaining, sample = next(iter(ind))
    return sample, remaining


def sample_fold_indices(data, folds, stratified=False, random_state=None):
    """Generate k-fold cross validation indices."""
    import sklearn.model_selection as skl
    
    if stratified and data.domain.has_discrete_class:
        splitter = skl.StratifiedKFold(folds, shuffle=True, random_state=random_state)
        splitter.get_n_splits(data.X, data.Y)
        ind = splitter.split(data.X, data.Y)
    else:
        splitter = skl.KFold(folds, shuffle=True, random_state=random_state)
        splitter.get_n_splits(data)
        ind = splitter.split(data)
    
    return tuple(ind)


def sample_bootstrap(size, random_state=None):
    """Bootstrap sampling indices."""
    import numpy as np
    
    rgen = np.random.RandomState(random_state)
    sample = rgen.randint(0, size, size)
    sample.sort()
    
    insample = np.ones((size,), dtype=bool)
    insample[sample] = False
    remaining = np.flatnonzero(insample)
    
    return sample, remaining


def calculate_sample_fallback(request: SampleDataRequest):
    """Fallback sampling calculation when Orange3 is not available."""
    total = 100
    
    if request.sampling_type == 0:
        sample_count = int(total * request.sample_percentage / 100)
    elif request.sampling_type == 1:
        sample_count = min(request.sample_size, total) if not request.replacement else request.sample_size
    elif request.sampling_type == 2:
        fold_size = total // request.number_of_folds
        sample_count = total - fold_size
    else:
        sample_count = total
    
    remaining_count = total - sample_count if request.sampling_type != 3 else int(total * 0.368)
    
    return {
        "success": True,
        "sample_count": max(0, sample_count),
        "remaining_count": max(0, remaining_count),
        "total_count": total,
        "sampling_type": request.sampling_type,
        "sample_indices": [],
        "remaining_indices": [],
        "note": "Fallback calculation (Orange3 not available)"
    }

