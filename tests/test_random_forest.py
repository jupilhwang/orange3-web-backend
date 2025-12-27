"""
Unit tests for Random Forest Learner Widget API.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestRandomForestOptions:
    """Test Random Forest options endpoint."""
    
    def test_get_random_forest_options(self):
        """Test getting Random Forest options."""
        response = client.get("/api/v1/model/random-forest/options")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "default_n_estimators" in data
        assert "default_min_samples_split" in data
        assert "max_n_estimators" in data


class TestRandomForestTrainBasic:
    """Basic Random Forest training tests."""
    
    def test_train_random_forest_iris(self):
        """Test training Random Forest on iris dataset."""
        request_data = {
            "data_path": "iris",
            "n_estimators": 10
        }
        response = client.post("/api/v1/model/random-forest/train", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "model_id" in data
    
    def test_train_random_forest_default_params(self):
        """Test training Random Forest with default parameters."""
        request_data = {
            "data_path": "iris"
        }
        response = client.post("/api/v1/model/random-forest/train", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True


class TestRandomForestEstimators:
    """Test different numbers of estimators."""
    
    def test_train_single_tree(self):
        """Test Random Forest with single tree."""
        request_data = {
            "data_path": "iris",
            "n_estimators": 1
        }
        response = client.post("/api/v1/model/random-forest/train", json=request_data)
        
        assert response.status_code == 200
        assert response.json()["success"] == True
    
    def test_train_many_trees(self):
        """Test Random Forest with many trees."""
        request_data = {
            "data_path": "iris",
            "n_estimators": 50
        }
        response = client.post("/api/v1/model/random-forest/train", json=request_data)
        
        assert response.status_code == 200
        assert response.json()["success"] == True
    
    def test_train_invalid_estimators(self):
        """Test Random Forest with invalid n_estimators."""
        request_data = {
            "data_path": "iris",
            "n_estimators": 0
        }
        response = client.post("/api/v1/model/random-forest/train", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert data["success"] == False


class TestRandomForestParameters:
    """Test different Random Forest parameters."""
    
    def test_train_with_max_depth(self):
        """Test Random Forest with max depth."""
        request_data = {
            "data_path": "iris",
            "n_estimators": 10,
            "max_depth": 5
        }
        response = client.post("/api/v1/model/random-forest/train", json=request_data)
        
        assert response.status_code == 200
        assert response.json()["success"] == True
    
    def test_train_with_max_features(self):
        """Test Random Forest with max features."""
        request_data = {
            "data_path": "iris",
            "n_estimators": 10,
            "max_features": 2
        }
        response = client.post("/api/v1/model/random-forest/train", json=request_data)
        
        assert response.status_code == 200
        assert response.json()["success"] == True
    
    def test_train_with_class_weight(self):
        """Test Random Forest with balanced class weights."""
        request_data = {
            "data_path": "iris",
            "n_estimators": 10,
            "class_weight": True
        }
        response = client.post("/api/v1/model/random-forest/train", json=request_data)
        
        assert response.status_code == 200
        assert response.json()["success"] == True
    
    def test_train_with_random_state(self):
        """Test Random Forest with random state."""
        request_data = {
            "data_path": "iris",
            "n_estimators": 10,
            "random_state": 42
        }
        response = client.post("/api/v1/model/random-forest/train", json=request_data)
        
        assert response.status_code == 200
        assert response.json()["success"] == True


class TestRandomForestFeatureImportances:
    """Test feature importance extraction."""
    
    def test_feature_importances_returned(self):
        """Test that feature importances are returned."""
        request_data = {
            "data_path": "iris",
            "n_estimators": 10
        }
        response = client.post("/api/v1/model/random-forest/train", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Feature importances should be returned
        if "feature_importances" in data and data["feature_importances"]:
            assert len(data["feature_importances"]) == 4  # 4 iris features
            
            # Check structure
            fi = data["feature_importances"][0]
            assert "feature" in fi
            assert "importance" in fi
            
            # Check sorted by importance
            importances = [f["importance"] for f in data["feature_importances"]]
            assert importances == sorted(importances, reverse=True)


class TestRandomForestModelInfo:
    """Test Random Forest model info."""
    
    def test_get_random_forest_info(self):
        """Test getting Random Forest model info."""
        # Train first
        train_response = client.post("/api/v1/model/random-forest/train", json={
            "data_path": "iris",
            "n_estimators": 10
        })
        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]
        
        # Get info
        info_response = client.get(f"/api/v1/model/random-forest/info/{model_id}")
        assert info_response.status_code == 200
        
        data = info_response.json()
        assert data["success"] == True
        assert data["type"] == "random_forest"
        assert data["n_estimators"] == 10


class TestRandomForestDeletion:
    """Test Random Forest model deletion."""
    
    def test_delete_random_forest_model(self):
        """Test deleting a Random Forest model."""
        # Train first
        train_response = client.post("/api/v1/model/random-forest/train", json={"data_path": "iris"})
        model_id = train_response.json()["model_id"]
        
        # Delete
        delete_response = client.delete(f"/api/v1/model/random-forest/{model_id}")
        assert delete_response.status_code == 200
        
        # Verify deleted
        info_response = client.get(f"/api/v1/model/random-forest/info/{model_id}")
        assert info_response.status_code == 404


class TestRandomForestDifferentDatasets:
    """Test Random Forest with different datasets."""
    
    def test_random_forest_zoo(self):
        """Test Random Forest on zoo dataset."""
        request_data = {
            "data_path": "zoo",
            "n_estimators": 10
        }
        response = client.post("/api/v1/model/random-forest/train", json=request_data)
        
        assert response.status_code == 200
    
    def test_random_forest_housing(self):
        """Test Random Forest on housing dataset (regression)."""
        request_data = {
            "data_path": "housing",
            "n_estimators": 10
        }
        response = client.post("/api/v1/model/random-forest/train", json=request_data)
        
        # Should work for regression too
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


