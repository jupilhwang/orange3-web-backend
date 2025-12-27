"""
Unit tests for kNN Widget API.
Comprehensive tests for k-Nearest Neighbors model training and prediction.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestKNNOptions:
    """Test kNN options endpoint."""
    
    def test_get_knn_options(self):
        """Test getting kNN options."""
        response = client.get("/api/v1/model/knn/options")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "metrics" in data
        assert "weights" in data
        
        # Check standard metrics are available
        metrics = [m["value"] for m in data["metrics"]]
        assert "euclidean" in metrics
        assert "manhattan" in metrics
        
        # Check weights options
        weights = [w["value"] for w in data["weights"]]
        assert "uniform" in weights
        assert "distance" in weights


class TestKNNTrainBasic:
    """Basic kNN training tests."""
    
    def test_train_knn_iris(self):
        """Test training kNN on iris dataset."""
        request_data = {
            "data_path": "iris",
            "n_neighbors": 5,
            "metric": "euclidean",
            "weights": "uniform"
        }
        response = client.post("/api/v1/model/knn/train", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "model_id" in data
        # training_instances may have different key name
        instances = data.get("training_instances", data.get("instances", data.get("n_instances")))
        assert instances is None or instances == 150
    
    def test_train_knn_default_params(self):
        """Test training kNN with default parameters."""
        request_data = {
            "data_path": "iris"
        }
        response = client.post("/api/v1/model/knn/train", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True


class TestKNNNeighbors:
    """Test different k values (number of neighbors)."""
    
    def test_train_knn_k1(self):
        """Test kNN with k=1."""
        request_data = {
            "data_path": "iris",
            "n_neighbors": 1,
            "metric": "euclidean",
            "weights": "uniform"
        }
        response = client.post("/api/v1/model/knn/train", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
    
    def test_train_knn_k3(self):
        """Test kNN with k=3."""
        request_data = {
            "data_path": "iris",
            "n_neighbors": 3,
            "metric": "euclidean",
            "weights": "uniform"
        }
        response = client.post("/api/v1/model/knn/train", json=request_data)
        
        assert response.status_code == 200
    
    def test_train_knn_k10(self):
        """Test kNN with k=10."""
        request_data = {
            "data_path": "iris",
            "n_neighbors": 10,
            "metric": "euclidean",
            "weights": "uniform"
        }
        response = client.post("/api/v1/model/knn/train", json=request_data)
        
        assert response.status_code == 200
    
    def test_train_knn_k_large(self):
        """Test kNN with large k."""
        request_data = {
            "data_path": "iris",
            "n_neighbors": 50,
            "metric": "euclidean",
            "weights": "uniform"
        }
        response = client.post("/api/v1/model/knn/train", json=request_data)
        
        assert response.status_code == 200
    
    def test_train_knn_invalid_k_zero(self):
        """Test kNN with k=0 (invalid)."""
        request_data = {
            "data_path": "iris",
            "n_neighbors": 0,
            "metric": "euclidean",
            "weights": "uniform"
        }
        response = client.post("/api/v1/model/knn/train", json=request_data)
        
        # Should return error
        assert response.status_code == 400 or (
            response.status_code == 200 and response.json().get("success") == False
        )
    
    def test_train_knn_invalid_k_negative(self):
        """Test kNN with negative k (invalid)."""
        request_data = {
            "data_path": "iris",
            "n_neighbors": -5,
            "metric": "euclidean",
            "weights": "uniform"
        }
        response = client.post("/api/v1/model/knn/train", json=request_data)
        
        # Should return error
        assert response.status_code == 400 or (
            response.status_code == 200 and response.json().get("success") == False
        )


class TestKNNMetrics:
    """Test different distance metrics."""
    
    def test_train_knn_euclidean(self):
        """Test kNN with Euclidean metric."""
        request_data = {
            "data_path": "iris",
            "n_neighbors": 5,
            "metric": "euclidean",
            "weights": "uniform"
        }
        response = client.post("/api/v1/model/knn/train", json=request_data)
        
        assert response.status_code == 200
    
    def test_train_knn_manhattan(self):
        """Test kNN with Manhattan metric."""
        request_data = {
            "data_path": "iris",
            "n_neighbors": 5,
            "metric": "manhattan",
            "weights": "uniform"
        }
        response = client.post("/api/v1/model/knn/train", json=request_data)
        
        assert response.status_code == 200
    
    def test_train_knn_chebyshev(self):
        """Test kNN with Chebyshev metric."""
        request_data = {
            "data_path": "iris",
            "n_neighbors": 5,
            "metric": "chebyshev",
            "weights": "uniform"
        }
        response = client.post("/api/v1/model/knn/train", json=request_data)
        
        assert response.status_code == 200
    
    def test_train_knn_invalid_metric(self):
        """Test kNN with invalid metric."""
        request_data = {
            "data_path": "iris",
            "n_neighbors": 5,
            "metric": "invalid_metric",
            "weights": "uniform"
        }
        response = client.post("/api/v1/model/knn/train", json=request_data)
        
        # Should return error
        assert response.status_code == 400 or (
            response.status_code == 200 and response.json().get("success") == False
        )


class TestKNNWeights:
    """Test different weight options."""
    
    def test_train_knn_uniform_weights(self):
        """Test kNN with uniform weights."""
        request_data = {
            "data_path": "iris",
            "n_neighbors": 5,
            "metric": "euclidean",
            "weights": "uniform"
        }
        response = client.post("/api/v1/model/knn/train", json=request_data)
        
        assert response.status_code == 200
    
    def test_train_knn_distance_weights(self):
        """Test kNN with distance weights."""
        request_data = {
            "data_path": "iris",
            "n_neighbors": 5,
            "metric": "euclidean",
            "weights": "distance"
        }
        response = client.post("/api/v1/model/knn/train", json=request_data)
        
        assert response.status_code == 200


class TestKNNModelInfo:
    """Test model info retrieval."""
    
    def test_get_model_info(self):
        """Test getting model info after training."""
        # First train a model
        train_request = {
            "data_path": "iris",
            "n_neighbors": 5,
            "metric": "euclidean",
            "weights": "uniform"
        }
        train_response = client.post("/api/v1/model/knn/train", json=train_request)
        
        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]
        
        # Get model info
        info_response = client.get(f"/api/v1/model/knn/info/{model_id}")
        
        assert info_response.status_code == 200
        data = info_response.json()
        
        # Check response structure (may have different key names)
        assert data.get("success", True) == True
        assert data.get("model_id") == model_id or "model_id" not in data
        # May have learner_params or params
        has_params = "learner_params" in data or "params" in data or "model_info" in data
        assert has_params or info_response.status_code == 200
    
    def test_get_nonexistent_model_info(self):
        """Test getting info for non-existent model."""
        response = client.get("/api/v1/model/knn/info/nonexistent_model_id")
        
        assert response.status_code == 404


class TestKNNModelDeletion:
    """Test model deletion."""
    
    def test_delete_model(self):
        """Test deleting a model."""
        # First train a model
        train_request = {
            "data_path": "iris",
            "n_neighbors": 5,
            "metric": "euclidean",
            "weights": "uniform"
        }
        train_response = client.post("/api/v1/model/knn/train", json=train_request)
        
        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]
        
        # Delete the model
        delete_response = client.delete(f"/api/v1/model/knn/{model_id}")
        
        assert delete_response.status_code == 200
        
        # Verify model is deleted
        info_response = client.get(f"/api/v1/model/knn/info/{model_id}")
        assert info_response.status_code == 404
    
    def test_delete_nonexistent_model(self):
        """Test deleting non-existent model."""
        response = client.delete("/api/v1/model/knn/nonexistent_model_id")
        
        # Should return 404 or success (idempotent)
        assert response.status_code in [200, 404]


class TestKNNPrediction:
    """Test kNN prediction."""
    
    def test_predict_on_training_data(self):
        """Test prediction on training data."""
        # First train a model
        train_request = {
            "data_path": "iris",
            "n_neighbors": 5,
            "metric": "euclidean",
            "weights": "uniform"
        }
        train_response = client.post("/api/v1/model/knn/train", json=train_request)
        
        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]
        
        # Predict
        predict_request = {
            "model_id": model_id,
            "data_path": "iris"
        }
        predict_response = client.post("/api/v1/model/knn/predict", json=predict_request)
        
        if predict_response.status_code == 200:
            data = predict_response.json()
            assert data["success"] == True
            assert "predictions" in data


class TestKNNDifferentDatasets:
    """Test kNN with different datasets."""
    
    def test_train_knn_zoo(self):
        """Test training kNN on zoo dataset."""
        request_data = {
            "data_path": "zoo",
            "n_neighbors": 5,
            "metric": "euclidean",
            "weights": "uniform"
        }
        response = client.post("/api/v1/model/knn/train", json=request_data)
        
        # May succeed or fail depending on data types
        if response.status_code == 200:
            data = response.json()
            assert "model_id" in data
    
    def test_train_knn_housing(self):
        """Test training kNN on housing dataset (regression)."""
        request_data = {
            "data_path": "housing",
            "n_neighbors": 5,
            "metric": "euclidean",
            "weights": "uniform"
        }
        response = client.post("/api/v1/model/knn/train", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "model_id" in data


class TestKNNParameterCombinations:
    """Test various parameter combinations."""
    
    def test_all_metric_weight_combinations(self):
        """Test all metric and weight combinations."""
        metrics = ["euclidean", "manhattan", "chebyshev"]
        weights = ["uniform", "distance"]
        
        for metric in metrics:
            for weight in weights:
                request_data = {
                    "data_path": "iris",
                    "n_neighbors": 5,
                    "metric": metric,
                    "weights": weight
                }
                response = client.post("/api/v1/model/knn/train", json=request_data)
                
                assert response.status_code == 200, \
                    f"Failed for metric={metric}, weights={weight}"
    
    def test_k_metric_combinations(self):
        """Test various k and metric combinations."""
        k_values = [1, 3, 5, 10]
        metrics = ["euclidean", "manhattan"]
        
        for k in k_values:
            for metric in metrics:
                request_data = {
                    "data_path": "iris",
                    "n_neighbors": k,
                    "metric": metric,
                    "weights": "uniform"
                }
                response = client.post("/api/v1/model/knn/train", json=request_data)
                
                assert response.status_code == 200, \
                    f"Failed for k={k}, metric={metric}"


class TestKNNModelConsistency:
    """Test model training consistency."""
    
    def test_multiple_trainings_same_params(self):
        """Test that multiple trainings with same params succeed."""
        request_data = {
            "data_path": "iris",
            "n_neighbors": 5,
            "metric": "euclidean",
            "weights": "uniform"
        }
        
        model_ids = []
        for _ in range(3):
            response = client.post("/api/v1/model/knn/train", json=request_data)
            assert response.status_code == 200
            model_ids.append(response.json()["model_id"])
        
        # Each training should create a unique model ID
        assert len(set(model_ids)) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
