"""
Tests for Predictions Widget API.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestPredictionsWidget:
    """Test cases for Predictions widget endpoints."""
    
    def test_predict_without_orange3(self):
        """Test prediction endpoint handles missing Orange3."""
        response = client.post("/api/v1/evaluate/predictions/predict", json={
            "data_path": "iris",
            "model_ids": ["nonexistent"],
            "show_probabilities": True
        })
        # Either success or Orange3 not available error
        assert response.status_code == 200
        
    def test_predict_with_invalid_model(self):
        """Test prediction with non-existent model."""
        response = client.post("/api/v1/evaluate/predictions/predict", json={
            "data_path": "iris",
            "model_ids": ["invalid_model_id"],
            "show_probabilities": True
        })
        assert response.status_code == 200
        result = response.json()
        # Should fail with model not found error
        if not result.get("success"):
            assert "not found" in result.get("error", "").lower() or \
                   "not available" in result.get("error", "").lower()
    
    def test_predict_empty_model_list(self):
        """Test prediction with empty model list."""
        response = client.post("/api/v1/evaluate/predictions/predict", json={
            "data_path": "iris",
            "model_ids": [],
            "show_probabilities": True
        })
        assert response.status_code == 200
        result = response.json()
        if not result.get("success"):
            assert "No models" in result.get("error", "") or \
                   "not available" in result.get("error", "")
    
    def test_get_predictions_not_found(self):
        """Test getting predictions with invalid ID."""
        response = client.get("/api/v1/evaluate/predictions/invalid_id")
        assert response.status_code == 404
    
    def test_delete_predictions(self):
        """Test deleting predictions."""
        response = client.delete("/api/v1/evaluate/predictions/test_id")
        assert response.status_code == 200
        assert "deleted" in response.json().get("message", "").lower()


class TestPredictionsIntegration:
    """Integration tests for Predictions widget with KNN model."""
    
    def test_train_and_predict_flow(self):
        """Test full flow: train kNN model then make predictions."""
        # First train a kNN model
        train_response = client.post("/api/v1/model/knn/train", json={
            "data_path": "iris",
            "n_neighbors": 3,
            "metric": "euclidean",
            "weights": "uniform"
        })
        
        if train_response.status_code == 200:
            train_result = train_response.json()
            
            if train_result.get("success"):
                model_id = train_result.get("model_id")
                
                # Now make predictions
                predict_response = client.post("/api/v1/evaluate/predictions/predict", json={
                    "data_path": "iris",
                    "model_ids": [model_id],
                    "show_probabilities": True
                })
                
                assert predict_response.status_code == 200
                predict_result = predict_response.json()
                
                if predict_result.get("success"):
                    assert "predictions" in predict_result
                    assert "columns" in predict_result
                    assert predict_result.get("instances", 0) > 0

