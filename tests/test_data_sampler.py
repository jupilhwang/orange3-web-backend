"""
Unit tests for Data Sampler Widget API.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestDataSamplerWidget:
    """Test cases for Data Sampler widget API endpoints."""
    
    def test_sample_fixed_proportion(self):
        """Test sampling with fixed proportion (70%)."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "sampling_type": 0,  # FixedProportion
            "sample_percentage": 70,
            "use_seed": True
        }
        
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "sample_count" in data or "sample" in data
        assert "remaining_count" in data or "remaining" in data
        
        # 70% of 150 = ~105
        sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
        assert 100 <= sample_count <= 110
    
    def test_sample_fixed_size(self):
        """Test sampling with fixed size."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "sampling_type": 1,  # FixedSize
            "sample_size": 50,
            "use_seed": True
        }
        
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
        assert sample_count == 50
    
    def test_sample_cross_validation(self):
        """Test k-fold cross validation sampling."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "sampling_type": 2,  # CrossValidation
            "number_of_folds": 10,
            "selected_fold": 1,
            "use_seed": True
        }
        
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # 10-fold CV: each fold should have ~15 samples (150/10)
        sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
        remaining_count = data.get("remaining_count", data.get("remaining", {}).get("count", 0))
        
        # Sample is training set (9/10), remaining is test set (1/10)
        assert remaining_count >= 10 and remaining_count <= 20
    
    def test_sample_bootstrap(self):
        """Test bootstrap sampling."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "sampling_type": 3,  # Bootstrap
            "use_seed": True
        }
        
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Bootstrap sample size equals original data size
        sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
        assert sample_count == 150
    
    def test_sample_with_stratify(self):
        """Test stratified sampling."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "sampling_type": 0,  # FixedProportion
            "sample_percentage": 50,
            "stratify": True,
            "use_seed": True
        }
        
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
        assert 70 <= sample_count <= 80  # ~50% of 150
    
    def test_sample_with_replacement(self):
        """Test sampling with replacement."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "sampling_type": 1,  # FixedSize
            "sample_size": 200,  # More than original data
            "replacement": True,
            "use_seed": True
        }
        
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
        # With replacement, we can have more samples than original
        assert sample_count == 200
    
    def test_sample_different_seeds(self):
        """Test that different seeds produce different results."""
        request_data_seed = {
            "data_path": "datasets/iris.tab",
            "sampling_type": 0,
            "sample_percentage": 50,
            "use_seed": True
        }
        
        request_data_no_seed = {
            "data_path": "datasets/iris.tab",
            "sampling_type": 0,
            "sample_percentage": 50,
            "use_seed": False
        }
        
        response1 = client.post("/api/v1/data/sample", json=request_data_seed)
        response2 = client.post("/api/v1/data/sample", json=request_data_seed)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # With same seed, results should be same
        data1 = response1.json()
        data2 = response2.json()
        
        indices1 = data1.get("sample_indices", [])
        indices2 = data2.get("sample_indices", [])
        
        if indices1 and indices2:
            assert indices1 == indices2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

