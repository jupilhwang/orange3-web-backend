"""
Unit tests for Logistic Regression Learner Widget API.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestLogisticRegressionOptions:
    """Test Logistic Regression options endpoint."""
    
    def test_get_logistic_regression_options(self):
        """Test getting Logistic Regression options."""
        response = client.get("/api/v1/model/logistic-regression/options")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "penalties" in data
        assert "default_C" in data
        
        # Check penalties
        penalty_values = [p["value"] for p in data["penalties"]]
        assert "l1" in penalty_values
        assert "l2" in penalty_values


class TestLogisticRegressionTrainBasic:
    """Basic Logistic Regression training tests."""
    
    def test_train_logistic_regression_iris(self):
        """Test training Logistic Regression on iris dataset."""
        request_data = {
            "data_path": "iris",
            "penalty": "l2",
            "C": 1.0
        }
        response = client.post("/api/v1/model/logistic-regression/train", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "model_id" in data
    
    def test_train_logistic_regression_default_params(self):
        """Test training Logistic Regression with default parameters."""
        request_data = {
            "data_path": "iris"
        }
        response = client.post("/api/v1/model/logistic-regression/train", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True


class TestLogisticRegressionPenalties:
    """Test different regularization penalties."""
    
    def test_train_with_l1_penalty(self):
        """Test Logistic Regression with L1 penalty (Lasso)."""
        request_data = {
            "data_path": "iris",
            "penalty": "l1",
            "C": 1.0
        }
        response = client.post("/api/v1/model/logistic-regression/train", json=request_data)
        
        assert response.status_code == 200
    
    def test_train_with_l2_penalty(self):
        """Test Logistic Regression with L2 penalty (Ridge)."""
        request_data = {
            "data_path": "iris",
            "penalty": "l2",
            "C": 1.0
        }
        response = client.post("/api/v1/model/logistic-regression/train", json=request_data)
        
        assert response.status_code == 200
        assert response.json()["success"] == True
    
    def test_train_without_penalty(self):
        """Test Logistic Regression without regularization."""
        request_data = {
            "data_path": "iris",
            "penalty": "none",
            "C": 1.0
        }
        response = client.post("/api/v1/model/logistic-regression/train", json=request_data)
        
        assert response.status_code == 200


class TestLogisticRegressionC:
    """Test different regularization strengths (C)."""
    
    def test_train_with_strong_regularization(self):
        """Test with strong regularization (small C)."""
        request_data = {
            "data_path": "iris",
            "penalty": "l2",
            "C": 0.01
        }
        response = client.post("/api/v1/model/logistic-regression/train", json=request_data)
        
        assert response.status_code == 200
        assert response.json()["success"] == True
    
    def test_train_with_weak_regularization(self):
        """Test with weak regularization (large C)."""
        request_data = {
            "data_path": "iris",
            "penalty": "l2",
            "C": 100.0
        }
        response = client.post("/api/v1/model/logistic-regression/train", json=request_data)
        
        assert response.status_code == 200
        assert response.json()["success"] == True


class TestLogisticRegressionCoefficients:
    """Test coefficient extraction."""
    
    def test_coefficients_returned(self):
        """Test that coefficients are returned."""
        request_data = {
            "data_path": "iris",
            "penalty": "l2",
            "C": 1.0
        }
        response = client.post("/api/v1/model/logistic-regression/train", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Coefficients should be returned
        if "coefficients" in data and data["coefficients"]:
            assert len(data["coefficients"]) > 0
            # Check structure
            coef = data["coefficients"][0]
            assert "feature" in coef
            assert "coefficient" in coef


class TestLogisticRegressionModelInfo:
    """Test Logistic Regression model info."""
    
    def test_get_logistic_regression_info(self):
        """Test getting Logistic Regression model info."""
        # Train first
        train_response = client.post("/api/v1/model/logistic-regression/train", json={
            "data_path": "iris",
            "penalty": "l2",
            "C": 1.0
        })
        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]
        
        # Get info
        info_response = client.get(f"/api/v1/model/logistic-regression/info/{model_id}")
        assert info_response.status_code == 200
        
        data = info_response.json()
        assert data["success"] == True
        assert data["type"] == "logistic_regression"
        assert data["penalty"] == "l2"
        assert data["C"] == 1.0


class TestLogisticRegressionDeletion:
    """Test Logistic Regression model deletion."""
    
    def test_delete_logistic_regression_model(self):
        """Test deleting a Logistic Regression model."""
        # Train first
        train_response = client.post("/api/v1/model/logistic-regression/train", json={"data_path": "iris"})
        model_id = train_response.json()["model_id"]
        
        # Delete
        delete_response = client.delete(f"/api/v1/model/logistic-regression/{model_id}")
        assert delete_response.status_code == 200
        
        # Verify deleted
        info_response = client.get(f"/api/v1/model/logistic-regression/info/{model_id}")
        assert info_response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


