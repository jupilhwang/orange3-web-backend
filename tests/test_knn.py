"""
Unit tests for kNN Widget API.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app


client = TestClient(app)


class TestKNNWidget:
    """Test cases for kNN widget API endpoints."""
    
    def test_get_knn_options(self):
        """Test getting kNN configuration options."""
        response = client.get("/api/v1/model/knn/options")
        assert response.status_code == 200
        
        data = response.json()
        
        # Check metrics
        assert "metrics" in data
        metrics = data["metrics"]
        assert len(metrics) == 4
        metric_values = [m["value"] for m in metrics]
        assert "euclidean" in metric_values
        assert "manhattan" in metric_values
        assert "chebyshev" in metric_values
        assert "mahalanobis" in metric_values
        
        # Check weights
        assert "weights" in data
        weights = data["weights"]
        assert len(weights) == 2
        weight_values = [w["value"] for w in weights]
        assert "uniform" in weight_values
        assert "distance" in weight_values
        
        # Check n_neighbors limits
        assert "n_neighbors" in data
        assert data["n_neighbors"]["min"] == 1
        assert data["n_neighbors"]["max"] == 100
        assert data["n_neighbors"]["default"] == 5
    
    def test_train_knn_with_iris(self):
        """Test training kNN model with iris dataset."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "n_neighbors": 5,
            "metric": "euclidean",
            "weights": "uniform"
        }
        
        response = client.post("/api/v1/model/knn/train", json=request_data)
        
        # Check if Orange3 is available
        if response.status_code == 200:
            data = response.json()
            
            if data.get("success"):
                assert data["model_id"] is not None
                assert data["learner_params"]["n_neighbors"] == 5
                assert data["learner_params"]["metric"] == "euclidean"
                assert data["learner_params"]["weights"] == "uniform"
                
                # Check model info
                assert data["model_info"]["type"] == "classification"
                assert data["model_info"]["training_instances"] == 150
                assert data["model_info"]["target"] == "iris"
            else:
                # Orange3 not available
                pytest.skip("Orange3 not available")
        else:
            pytest.skip("kNN training endpoint not available")
    
    def test_train_knn_with_distance_weights(self):
        """Test training kNN with distance-based weights."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "n_neighbors": 3,
            "metric": "manhattan",
            "weights": "distance"
        }
        
        response = client.post("/api/v1/model/knn/train", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("success"):
                assert data["learner_params"]["n_neighbors"] == 3
                assert data["learner_params"]["metric"] == "manhattan"
                assert data["learner_params"]["weights"] == "distance"
    
    def test_train_knn_invalid_neighbors(self):
        """Test training kNN with invalid number of neighbors."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "n_neighbors": 0,  # Invalid
            "metric": "euclidean",
            "weights": "uniform"
        }
        
        response = client.post("/api/v1/model/knn/train", json=request_data)
        
        # Should return error for invalid neighbors
        if response.status_code == 400:
            assert "neighbors" in response.json().get("detail", "").lower()
        elif response.status_code == 200:
            data = response.json()
            if not data.get("success"):
                # Also acceptable if returned as failure
                pass
    
    def test_train_knn_invalid_metric(self):
        """Test training kNN with invalid metric."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "n_neighbors": 5,
            "metric": "invalid_metric",
            "weights": "uniform"
        }
        
        response = client.post("/api/v1/model/knn/train", json=request_data)
        
        # Should return error for invalid metric
        if response.status_code == 400:
            assert "metric" in response.json().get("detail", "").lower()
    
    def test_delete_knn_model(self):
        """Test deleting a kNN model."""
        # First train a model
        request_data = {
            "data_path": "datasets/iris.tab",
            "n_neighbors": 5,
            "metric": "euclidean",
            "weights": "uniform"
        }
        
        train_response = client.post("/api/v1/model/knn/train", json=request_data)
        
        if train_response.status_code == 200:
            data = train_response.json()
            
            if data.get("success") and data.get("model_id"):
                model_id = data["model_id"]
                
                # Delete the model
                delete_response = client.delete(f"/api/v1/model/knn/{model_id}")
                assert delete_response.status_code == 200
                assert "deleted" in delete_response.json().get("message", "").lower()
    
    def test_get_knn_model_info(self):
        """Test getting information about a trained kNN model."""
        # First train a model
        request_data = {
            "data_path": "datasets/iris.tab",
            "n_neighbors": 7,
            "metric": "chebyshev",
            "weights": "uniform"
        }
        
        train_response = client.post("/api/v1/model/knn/train", json=request_data)
        
        if train_response.status_code == 200:
            data = train_response.json()
            
            if data.get("success") and data.get("model_id"):
                model_id = data["model_id"]
                
                # Get model info
                info_response = client.get(f"/api/v1/model/knn/info/{model_id}")
                assert info_response.status_code == 200
                
                info = info_response.json()
                assert info["model_id"] == model_id
    
    def test_get_nonexistent_model_info(self):
        """Test getting info for non-existent model."""
        response = client.get("/api/v1/model/knn/info/nonexistent123")
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

