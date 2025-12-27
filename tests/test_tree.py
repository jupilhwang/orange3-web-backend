"""
Unit tests for Tree Learner Widget API.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestTreeOptions:
    """Test Tree options endpoint."""
    
    def test_get_tree_options(self):
        """Test getting Tree options."""
        response = client.get("/api/v1/model/tree/options")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "default_binary_trees" in data
        assert "default_max_depth" in data
        assert "default_min_samples_split" in data
        assert "default_min_samples_leaf" in data


class TestTreeTrainBasic:
    """Basic Tree training tests."""
    
    def test_train_tree_iris(self):
        """Test training Tree on iris dataset."""
        request_data = {
            "data_path": "iris",
            "binary_trees": True,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2
        }
        response = client.post("/api/v1/model/tree/train", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "model_id" in data
    
    def test_train_tree_default_params(self):
        """Test training Tree with default parameters."""
        request_data = {
            "data_path": "iris"
        }
        response = client.post("/api/v1/model/tree/train", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True


class TestTreeParameters:
    """Test different Tree parameters."""
    
    def test_train_tree_shallow(self):
        """Test Tree with shallow depth."""
        request_data = {
            "data_path": "iris",
            "max_depth": 3
        }
        response = client.post("/api/v1/model/tree/train", json=request_data)
        
        assert response.status_code == 200
        assert response.json()["success"] == True
    
    def test_train_tree_deep(self):
        """Test Tree with deep depth."""
        request_data = {
            "data_path": "iris",
            "max_depth": 50
        }
        response = client.post("/api/v1/model/tree/train", json=request_data)
        
        assert response.status_code == 200
        assert response.json()["success"] == True
    
    def test_train_tree_binary_false(self):
        """Test Tree without binary splits."""
        request_data = {
            "data_path": "iris",
            "binary_trees": False
        }
        response = client.post("/api/v1/model/tree/train", json=request_data)
        
        assert response.status_code == 200
        assert response.json()["success"] == True


class TestTreeModelInfo:
    """Test Tree model info."""
    
    def test_get_tree_info(self):
        """Test getting Tree model info."""
        # Train first
        train_response = client.post("/api/v1/model/tree/train", json={"data_path": "iris"})
        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]
        
        # Get info
        info_response = client.get(f"/api/v1/model/tree/info/{model_id}")
        assert info_response.status_code == 200
        
        data = info_response.json()
        assert data["success"] == True
        assert data["type"] == "tree"
    
    def test_get_nonexistent_tree_info(self):
        """Test getting info for non-existent model."""
        response = client.get("/api/v1/model/tree/info/nonexistent_id")
        assert response.status_code == 404


class TestTreeDeletion:
    """Test Tree model deletion."""
    
    def test_delete_tree_model(self):
        """Test deleting a Tree model."""
        # Train first
        train_response = client.post("/api/v1/model/tree/train", json={"data_path": "iris"})
        model_id = train_response.json()["model_id"]
        
        # Delete
        delete_response = client.delete(f"/api/v1/model/tree/{model_id}")
        assert delete_response.status_code == 200
        
        # Verify deleted
        info_response = client.get(f"/api/v1/model/tree/info/{model_id}")
        assert info_response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


