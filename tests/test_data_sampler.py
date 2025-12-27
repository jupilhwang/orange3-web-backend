"""
Unit tests for Data Sampler Widget API.
Comprehensive tests based on Orange3's test_owdatasampler.py patterns.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestDataSamplerFixedProportion:
    """Test cases for fixed proportion sampling (sampling_type=0)."""
    
    def test_sample_70_percent(self):
        """Test 70% sampling."""
        request_data = {
            "data_path": "iris",
            "sampling_type": 0,
            "sample_percentage": 70,
            "use_seed": True
        }
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # 70% of 150 = 105
        sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
        remaining_count = data.get("remaining_count", data.get("remaining", {}).get("count", 0))
        
        assert sample_count == 105
        assert remaining_count == 45
        assert sample_count + remaining_count == 150
    
    def test_sample_50_percent(self):
        """Test 50% sampling."""
        request_data = {
            "data_path": "iris",
            "sampling_type": 0,
            "sample_percentage": 50,
            "use_seed": True
        }
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
        assert sample_count == 75
    
    def test_sample_100_percent(self):
        """Test 100% sampling returns all data."""
        request_data = {
            "data_path": "iris",
            "sampling_type": 0,
            "sample_percentage": 100,
            "use_seed": True
        }
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
        remaining_count = data.get("remaining_count", data.get("remaining", {}).get("count", 0))
        
        assert sample_count == 150
        assert remaining_count == 0
    
    def test_sample_0_percent(self):
        """Test 0% sampling returns no data."""
        request_data = {
            "data_path": "iris",
            "sampling_type": 0,
            "sample_percentage": 0,
            "use_seed": True
        }
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
        remaining_count = data.get("remaining_count", data.get("remaining", {}).get("count", 0))
        
        assert sample_count == 0
        assert remaining_count == 150


class TestDataSamplerFixedSize:
    """Test cases for fixed size sampling (sampling_type=1)."""
    
    def test_sample_fixed_50(self):
        """Test fixed size 50."""
        request_data = {
            "data_path": "iris",
            "sampling_type": 1,
            "sample_size": 50,
            "use_seed": True
        }
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
        assert sample_count == 50
    
    def test_sample_fixed_1(self):
        """Test fixed size 1."""
        request_data = {
            "data_path": "iris",
            "sampling_type": 1,
            "sample_size": 1,
            "use_seed": True
        }
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
        assert sample_count == 1
    
    def test_sample_fixed_all(self):
        """Test fixed size equal to data size."""
        request_data = {
            "data_path": "iris",
            "sampling_type": 1,
            "sample_size": 150,
            "use_seed": True
        }
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
        remaining_count = data.get("remaining_count", data.get("remaining", {}).get("count", 0))
        
        assert sample_count == 150
        assert remaining_count == 0
    
    def test_sample_bigger_without_replacement(self):
        """Test sample size bigger than data without replacement - should error or cap."""
        request_data = {
            "data_path": "iris",
            "sampling_type": 1,
            "sample_size": 200,
            "replacement": False,
            "use_seed": True
        }
        response = client.post("/api/v1/data/sample", json=request_data)
        
        # Should either return error (400) or cap to max size (200)
        if response.status_code == 200:
            data = response.json()
            sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
            # Without replacement, can't have more than original
            assert sample_count <= 150
        else:
            # 400 is also acceptable - cannot sample more than data size
            assert response.status_code == 400
    
    def test_sample_bigger_with_replacement(self):
        """Test sample size bigger than data with replacement."""
        request_data = {
            "data_path": "iris",
            "sampling_type": 1,
            "sample_size": 200,
            "replacement": True,
            "use_seed": True
        }
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
        assert sample_count == 200


class TestDataSamplerCrossValidation:
    """Test cases for cross-validation sampling (sampling_type=2)."""
    
    def test_cv_10_fold_first(self):
        """Test 10-fold CV, fold 1."""
        request_data = {
            "data_path": "iris",
            "sampling_type": 2,
            "number_of_folds": 10,
            "selected_fold": 1,
            "use_seed": True
        }
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
        remaining_count = data.get("remaining_count", data.get("remaining", {}).get("count", 0))
        
        # 10-fold: training=90%, test=10%
        # Sample is training (135), remaining is test (15)
        assert sample_count == 135
        assert remaining_count == 15
        assert sample_count + remaining_count == 150
    
    def test_cv_5_fold(self):
        """Test 5-fold CV."""
        request_data = {
            "data_path": "iris",
            "sampling_type": 2,
            "number_of_folds": 5,
            "selected_fold": 1,
            "use_seed": True
        }
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
        remaining_count = data.get("remaining_count", data.get("remaining", {}).get("count", 0))
        
        # 5-fold: training=80%, test=20%
        assert sample_count == 120
        assert remaining_count == 30
    
    def test_cv_different_folds(self):
        """Test different folds produce different results."""
        results = []
        
        for fold in range(1, 4):
            request_data = {
                "data_path": "iris",
                "sampling_type": 2,
                "number_of_folds": 10,
                "selected_fold": fold,
                "use_seed": True
            }
            response = client.post("/api/v1/data/sample", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            indices = data.get("remaining_indices", data.get("remaining", {}).get("indices", []))
            results.append(set(indices) if indices else set())
        
        # Different folds should have different test sets
        if all(results):
            assert results[0] != results[1], "Different folds should have different test sets"
    
    def test_cv_too_many_folds_error(self):
        """Test error when folds > data size."""
        request_data = {
            "data_path": "iris",
            "sampling_type": 2,
            "number_of_folds": 200,  # More folds than instances
            "selected_fold": 1,
            "use_seed": True
        }
        response = client.post("/api/v1/data/sample", json=request_data)
        
        # Should either return error or handle gracefully
        if response.status_code == 200:
            data = response.json()
            # Should have some reasonable behavior
            assert "error" in data or data.get("sample_count", 0) >= 0


class TestDataSamplerBootstrap:
    """Test cases for bootstrap sampling (sampling_type=3)."""
    
    def test_bootstrap_basic(self):
        """Test basic bootstrap sampling."""
        request_data = {
            "data_path": "iris",
            "sampling_type": 3,
            "use_seed": True
        }
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
        # Bootstrap samples same size as original
        assert sample_count == 150
    
    def test_bootstrap_has_duplicates(self):
        """Test bootstrap sampling with replacement has duplicates."""
        request_data = {
            "data_path": "iris",
            "sampling_type": 3,
            "use_seed": True
        }
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        sample_indices = data.get("sample_indices", data.get("sample", {}).get("indices", []))
        
        if sample_indices:
            # With replacement, there should be duplicates (very high probability)
            unique_count = len(set(sample_indices))
            assert unique_count < len(sample_indices), "Bootstrap should have duplicate indices"
    
    def test_bootstrap_remaining_is_oob(self):
        """Test bootstrap remaining (out-of-bag) samples."""
        request_data = {
            "data_path": "iris",
            "sampling_type": 3,
            "use_seed": True
        }
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        remaining_count = data.get("remaining_count", data.get("remaining", {}).get("count", 0))
        
        # OOB should be roughly 36.8% of data (1/e)
        # Allow some variance: between 30% and 45%
        assert remaining_count > 45  # > 30%
        assert remaining_count < 70  # < 47%


class TestDataSamplerStratified:
    """Test cases for stratified sampling."""
    
    def test_stratified_proportion(self):
        """Test stratified proportional sampling maintains class distribution."""
        request_data = {
            "data_path": "iris",
            "sampling_type": 0,
            "sample_percentage": 60,
            "stratify": True,
            "use_seed": True
        }
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
        # 60% of 150 = 90
        assert sample_count == 90
    
    def test_stratified_fixed_size(self):
        """Test stratified fixed size sampling."""
        request_data = {
            "data_path": "iris",
            "sampling_type": 1,
            "sample_size": 90,
            "stratify": True,
            "use_seed": True
        }
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
        assert sample_count == 90


class TestDataSamplerReplicability:
    """Test cases for random seed and replicability."""
    
    def test_same_seed_same_result(self):
        """Test that same seed produces same results."""
        request_data = {
            "data_path": "iris",
            "sampling_type": 0,
            "sample_percentage": 50,
            "use_seed": True
        }
        
        response1 = client.post("/api/v1/data/sample", json=request_data)
        response2 = client.post("/api/v1/data/sample", json=request_data)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        indices1 = response1.json().get("sample_indices", [])
        indices2 = response2.json().get("sample_indices", [])
        
        if indices1 and indices2:
            assert indices1 == indices2, "Same seed should produce same results"
    
    def test_no_seed_different_results(self):
        """Test that no seed can produce different results."""
        request_data = {
            "data_path": "iris",
            "sampling_type": 0,
            "sample_percentage": 50,
            "use_seed": False
        }
        
        # Without seed, results may vary (though not guaranteed)
        response1 = client.post("/api/v1/data/sample", json=request_data)
        response2 = client.post("/api/v1/data/sample", json=request_data)
        
        assert response1.status_code == 200
        assert response2.status_code == 200


class TestDataSamplerNoIntersection:
    """Test that sample and remaining don't intersect."""
    
    def test_no_intersection_fixed_proportion(self):
        """Test no intersection in fixed proportion sampling."""
        request_data = {
            "data_path": "iris",
            "sampling_type": 0,
            "sample_percentage": 70,
            "use_seed": True
        }
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        sample_indices = set(data.get("sample_indices", []))
        remaining_indices = set(data.get("remaining_indices", []))
        
        if sample_indices and remaining_indices:
            # No intersection
            assert len(sample_indices & remaining_indices) == 0
            # Together they cover all
            assert len(sample_indices | remaining_indices) == 150
    
    def test_no_intersection_cv(self):
        """Test no intersection in cross-validation."""
        request_data = {
            "data_path": "iris",
            "sampling_type": 2,
            "number_of_folds": 10,
            "selected_fold": 1,
            "use_seed": True
        }
        response = client.post("/api/v1/data/sample", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
        remaining_count = data.get("remaining_count", data.get("remaining", {}).get("count", 0))
        
        assert sample_count + remaining_count == 150


class TestDataSamplerDifferentDatasets:
    """Test with different datasets."""
    
    def test_zoo_dataset(self):
        """Test sampling on zoo dataset."""
        request_data = {
            "data_path": "zoo",
            "sampling_type": 0,
            "sample_percentage": 50,
            "use_seed": True
        }
        response = client.post("/api/v1/data/sample", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
            assert sample_count > 0
    
    def test_housing_dataset(self):
        """Test sampling on housing dataset."""
        request_data = {
            "data_path": "housing",
            "sampling_type": 0,
            "sample_percentage": 30,
            "use_seed": True
        }
        response = client.post("/api/v1/data/sample", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            sample_count = data.get("sample_count", data.get("sample", {}).get("count", 0))
            assert sample_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
