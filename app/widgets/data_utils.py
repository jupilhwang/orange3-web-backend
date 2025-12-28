"""
Common data loading utilities for widget backends.
Handles various data path formats including k-Means clustered data.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Upload directory
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"

# Check Orange3 availability
try:
    from Orange.data import Table
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False


def load_data(data_path: str) -> Optional['Table']:
    """
    Load data from various path formats.
    
    Supported formats:
    - "datasets/iris" - Orange3 built-in datasets
    - "uploads/xxx" - Uploaded files
    - "kmeans/{cluster_id}" - k-Means clustered data
    - "sampler/{sample_id}" - Data sampler results
    - Direct file paths
    
    Returns:
        Orange.data.Table or None if loading fails
    """
    if not ORANGE_AVAILABLE:
        logger.error("Orange3 not available")
        return None
    
    try:
        # Handle k-Means clustered data
        if data_path.startswith("kmeans/"):
            return _load_kmeans_data(data_path)
        
        # Handle Data Sampler results
        if data_path.startswith("sampler/"):
            return _load_sampler_data(data_path)
        
        # Handle uploads
        if data_path.startswith("uploads/"):
            resolved_path = str(UPLOAD_DIR / data_path.replace("uploads/", ""))
            return Table(resolved_path)
        
        # Handle built-in datasets
        if data_path.startswith("datasets/"):
            dataset_name = data_path.replace("datasets/", "").split(".")[0]
            return Table(dataset_name)
        
        # Direct path or dataset name
        return Table(data_path)
        
    except Exception as e:
        logger.error(f"Failed to load data from {data_path}: {e}")
        return None


def _load_kmeans_data(data_path: str) -> Optional['Table']:
    """
    Load k-Means clustered data.
    
    Args:
        data_path: "kmeans/{cluster_id}" format
        
    Returns:
        Annotated Table with cluster column
    """
    try:
        # Import k-Means results storage
        from .kmeans import _kmeans_results
        
        cluster_id = data_path.replace("kmeans/", "")
        
        if cluster_id not in _kmeans_results:
            logger.error(f"K-Means result not found: {cluster_id}")
            return None
        
        result = _kmeans_results[cluster_id]
        return result.get("data")
        
    except Exception as e:
        logger.error(f"Failed to load k-Means data: {e}")
        return None


def _load_sampler_data(data_path: str) -> Optional['Table']:
    """
    Load Data Sampler results.
    
    Args:
        data_path: "sampler/{sampler_id}" format
        
    Returns:
        Sampled or remaining data Table
    """
    try:
        # Import Data Sampler results storage
        from .data_sampler import _sampler_results
        
        sampler_id = data_path.replace("sampler/", "")
        
        if sampler_id not in _sampler_results:
            logger.error(f"Sampler result not found: {sampler_id}")
            return None
        
        result = _sampler_results[sampler_id]
        return result.get("data")
        
    except Exception as e:
        logger.error(f"Failed to load sampler data: {e}")
        return None


def resolve_data_path(data_path: str) -> str:
    """
    Resolve data path to actual file path (for legacy compatibility).
    
    Note: For k-Means data, use load_data() instead.
    """
    if data_path.startswith("uploads/"):
        return str(UPLOAD_DIR / data_path.replace("uploads/", ""))
    elif data_path.startswith("datasets/"):
        dataset_name = data_path.replace("datasets/", "").split(".")[0]
        return dataset_name
    return data_path

