"""
Tree Learner Widget API endpoints.
Decision Tree for classification and regression.
"""

import logging
import uuid
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model", tags=["Model"])

# Check Orange3 availability
try:
    from Orange.data import Table
    from Orange.modelling.tree import TreeLearner
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False

# Model storage
_tree_models: Dict[str, Any] = {}
_tree_learners: Dict[str, Any] = {}

class TreeTrainRequest(BaseModel):
    """Request model for Tree training."""
    data_path: str
    binary_trees: bool = True
    max_depth: Optional[int] = 100
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    sufficient_majority: float = 0.95


class TreeTrainResponse(BaseModel):
    """Response model for Tree training."""
    success: bool
    model_id: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None


class TreeOptionsResponse(BaseModel):
    """Response model for Tree options."""
    default_binary_trees: bool = True
    default_max_depth: int = 100
    default_min_samples_split: int = 5
    default_min_samples_leaf: int = 2
    default_sufficient_majority: float = 0.95


@router.get("/tree/options", response_model=TreeOptionsResponse)
async def get_tree_options():
    """Get Tree learner options and defaults."""
    return TreeOptionsResponse()


@router.post("/tree/train", response_model=TreeTrainResponse)
async def train_tree(
    request: TreeTrainRequest,
    x_session_id: Optional[str] = Header(None)
):
    """Train a Decision Tree model."""
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Orange3 not available")
    
    try:
        # Load data using common utility
        from app.core.data_utils import load_data
        logger.info(f"Loading Tree data from: {request.data_path} (session: {x_session_id})")
        data = load_data(request.data_path, session_id=x_session_id)
        
        if data is None:
            raise HTTPException(status_code=404, detail=f"Data not found: {request.data_path}")
        
        if data is None or len(data) == 0:
            return TreeTrainResponse(
                success=False,
                error="Dataset is empty or could not be loaded"
            )
        
        # Create Tree learner
        learner = TreeLearner(
            binarize=request.binary_trees,
            max_depth=request.max_depth,
            min_samples_split=request.min_samples_split,
            min_samples_leaf=request.min_samples_leaf,
            sufficient_majority=request.sufficient_majority
        )
        
        # Train model
        model = learner(data)
        
        # Store model
        model_id = str(uuid.uuid4())[:8]
        _tree_models[model_id] = model
        _tree_learners[model_id] = {
            "learner": learner,
            "type": "tree",
            "training_instances": len(data),
            "features": len(data.domain.attributes),
            "target": data.domain.class_var.name if data.domain.class_var else None,
        }
        
        return TreeTrainResponse(
            success=True,
            model_id=model_id,
            message=f"Tree model trained successfully on {len(data)} instances"
        )
        
    except Exception as e:
        logger.error(f"Tree training error: {e}")
        return TreeTrainResponse(success=False, error=str(e))


@router.get("/tree/info/{model_id}")
async def get_tree_info(model_id: str):
    """Get Tree model information."""
    if model_id not in _tree_learners:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_data = _tree_learners[model_id]
    
    return {
        "success": True,
        "model_id": model_id,
        "type": model_data.get("type"),
        "training_instances": model_data.get("training_instances"),
        "features": model_data.get("features"),
        "target": model_data.get("target"),
    }


@router.delete("/tree/{model_id}")
async def delete_tree_model(model_id: str):
    """Delete a Tree model."""
    if model_id not in _tree_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    del _tree_models[model_id]
    if model_id in _tree_learners:
        del _tree_learners[model_id]
    
    return {"success": True, "message": "Model deleted"}


@router.get("/tree/visualize/{model_id}")
async def visualize_tree(model_id: str, max_depth: int = 3):
    """
    Get tree structure for visualization.
    Returns a hierarchical JSON representation of the tree.
    """
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Orange3 not available")
    
    if model_id not in _tree_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        model = _tree_models[model_id]
        
        def node_to_dict(node, depth=0):
            """Convert tree node to dictionary for visualization."""
            if node is None or depth > max_depth:
                return None
            
            result = {
                "id": id(node),
                "depth": depth,
                "samples": int(node.n_node_samples) if hasattr(node, 'n_node_samples') else 0,
            }
            
            # Get split information
            if hasattr(node, 'attr') and node.attr is not None:
                result["split_attr"] = node.attr.name
                result["split_value"] = float(node.value) if hasattr(node, 'value') else None
                result["is_leaf"] = False
            else:
                result["is_leaf"] = True
            
            # Get class distribution for classification
            if hasattr(node, 'value') and hasattr(model, 'domain'):
                if model.domain.class_var.is_discrete:
                    class_values = list(model.domain.class_var.values)
                    if hasattr(node, 'value') and isinstance(node.value, (list, tuple)):
                        result["class_distribution"] = {
                            class_values[i]: int(v) for i, v in enumerate(node.value) if i < len(class_values)
                        }
                    # Majority class
                    if hasattr(node, 'majority'):
                        result["majority_class"] = str(model.domain.class_var.values[int(node.majority)])
                else:
                    # Regression - mean value
                    if hasattr(node, 'value'):
                        result["mean_value"] = float(node.value) if isinstance(node.value, (int, float)) else None
            
            # Get children
            if hasattr(node, 'children') and node.children:
                result["children"] = []
                for child in node.children:
                    child_dict = node_to_dict(child, depth + 1)
                    if child_dict:
                        result["children"].append(child_dict)
            
            return result
        
        # Get tree root
        tree_root = None
        if hasattr(model, 'model'):
            tree_root = model.model
        elif hasattr(model, 'tree_'):
            tree_root = model.tree_
        elif hasattr(model, 'root'):
            tree_root = model.root
        
        if tree_root is None:
            # Try sklearn tree structure
            if hasattr(model, 'skl_model') and hasattr(model.skl_model, 'tree_'):
                skl_tree = model.skl_model.tree_
                return convert_sklearn_tree(skl_tree, model, max_depth)
        
        tree_data = node_to_dict(tree_root)
        
        if tree_data is None:
            return {
                "success": False,
                "error": "Could not extract tree structure"
            }
        
        return {
            "success": True,
            "tree": tree_data,
            "max_depth": max_depth,
            "model_id": model_id
        }
        
    except Exception as e:
        logger.error(f"Tree visualization error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def convert_sklearn_tree(skl_tree, model, max_depth):
    """Convert sklearn tree to visualization format."""
    import numpy as np
    
    def build_node(node_id, depth=0):
        if depth > max_depth:
            return None
        
        left = skl_tree.children_left[node_id]
        right = skl_tree.children_right[node_id]
        
        result = {
            "id": int(node_id),
            "depth": depth,
            "samples": int(skl_tree.n_node_samples[node_id]),
            "is_leaf": left == -1
        }
        
        if left != -1:
            # Split node
            feature_idx = skl_tree.feature[node_id]
            threshold = skl_tree.threshold[node_id]
            
            if hasattr(model, 'domain') and feature_idx < len(model.domain.attributes):
                result["split_attr"] = model.domain.attributes[feature_idx].name
            else:
                result["split_attr"] = f"feature_{feature_idx}"
            result["split_value"] = float(threshold)
            
            result["children"] = []
            left_child = build_node(left, depth + 1)
            right_child = build_node(right, depth + 1)
            if left_child:
                left_child["edge_label"] = f"≤ {threshold:.2f}"
                result["children"].append(left_child)
            if right_child:
                right_child["edge_label"] = f"> {threshold:.2f}"
                result["children"].append(right_child)
        else:
            # Leaf node
            value = skl_tree.value[node_id]
            if hasattr(model, 'domain') and model.domain.class_var.is_discrete:
                class_values = list(model.domain.class_var.values)
                if len(value.shape) > 1:
                    value = value[0]
                result["class_distribution"] = {
                    class_values[i]: int(v) for i, v in enumerate(value) if i < len(class_values)
                }
                result["majority_class"] = class_values[int(np.argmax(value))]
            else:
                result["mean_value"] = float(value[0][0]) if len(value.shape) > 1 else float(value[0])
        
        return result
    
    tree_data = build_node(0)
    
    return {
        "success": True,
        "tree": tree_data,
        "max_depth": max_depth,
        "n_nodes": int(skl_tree.node_count),
        "n_features": int(skl_tree.n_features)
    }

