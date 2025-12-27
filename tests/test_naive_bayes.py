"""
Unit tests for Naive Bayes Learner Widget API.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestNaiveBayesOptions:
    """Test Naive Bayes options endpoint."""
    
    def test_get_naive_bayes_options(self):
        """Test getting Naive Bayes options."""
        response = client.get("/api/v1/model/naive-bayes/options")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "description" in data


class TestNaiveBayesTrainBasic:
    """Basic Naive Bayes training tests."""
    
    def test_train_naive_bayes_iris(self):
        """Test training Naive Bayes on iris dataset."""
        request_data = {
            "data_path": "iris"
        }
        response = client.post("/api/v1/model/naive-bayes/train", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "model_id" in data
    
    def test_train_naive_bayes_zoo(self):
        """Test training Naive Bayes on zoo dataset."""
        request_data = {
            "data_path": "zoo"
        }
        response = client.post("/api/v1/model/naive-bayes/train", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True


class TestNaiveBayesModelInfo:
    """Test Naive Bayes model info."""
    
    def test_get_naive_bayes_info(self):
        """Test getting Naive Bayes model info."""
        # Train first
        train_response = client.post("/api/v1/model/naive-bayes/train", json={"data_path": "iris"})
        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]
        
        # Get info
        info_response = client.get(f"/api/v1/model/naive-bayes/info/{model_id}")
        assert info_response.status_code == 200
        
        data = info_response.json()
        assert data["success"] == True
        assert data["type"] == "naive_bayes"
        assert data["target"] == "iris"
    
    def test_get_nonexistent_naive_bayes_info(self):
        """Test getting info for non-existent model."""
        response = client.get("/api/v1/model/naive-bayes/info/nonexistent_id")
        assert response.status_code == 404


class TestNaiveBayesDeletion:
    """Test Naive Bayes model deletion."""
    
    def test_delete_naive_bayes_model(self):
        """Test deleting a Naive Bayes model."""
        # Train first
        train_response = client.post("/api/v1/model/naive-bayes/train", json={"data_path": "iris"})
        model_id = train_response.json()["model_id"]
        
        # Delete
        delete_response = client.delete(f"/api/v1/model/naive-bayes/{model_id}")
        assert delete_response.status_code == 200
        
        # Verify deleted
        info_response = client.get(f"/api/v1/model/naive-bayes/info/{model_id}")
        assert info_response.status_code == 404


class TestNaiveBayesErrors:
    """Test Naive Bayes error handling."""
    
    def test_train_on_regression_data(self):
        """Test training on regression data (should fail)."""
        request_data = {
            "data_path": "housing"
        }
        response = client.post("/api/v1/model/naive-bayes/train", json=request_data)
        
        # Should fail because Naive Bayes is classification only
        if response.status_code == 200:
            data = response.json()
            assert data["success"] == False or "error" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

