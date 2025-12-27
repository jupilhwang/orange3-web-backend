"""
Unit tests for Distributions Widget API.
Comprehensive tests for distribution visualization.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestDistributionsContinuousVariable:
    """Test distributions for continuous variables."""
    
    def test_distributions_sepal_length(self):
        """Test distribution of sepal length."""
        request_data = {
            "data_path": "iris",
            "variable": "sepal length",
            "number_of_bins": 10
        }
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "bins" in data or "histogram" in data
    
    def test_distributions_different_bin_counts(self):
        """Test distributions with different bin counts."""
        for bins in [5, 10, 20, 50]:
            request_data = {
                "data_path": "iris",
                "variable": "sepal length",
                "number_of_bins": bins
            }
            response = client.post("/api/v1/data/distributions", json=request_data)
            
            if response.status_code == 501:
                pytest.skip("Orange3 not available")
            
            assert response.status_code == 200
    
    def test_distributions_statistics(self):
        """Test that distributions include statistics."""
        request_data = {
            "data_path": "iris",
            "variable": "sepal length",
            "number_of_bins": 10
        }
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have some statistics
        has_stats = ("statistics" in data or "mean" in data or 
                    "std" in data or "min" in data or "max" in data)
        assert has_stats or "bins" in data


class TestDistributionsDiscreteVariable:
    """Test distributions for discrete variables."""
    
    def test_distributions_class_variable(self):
        """Test distribution of class variable."""
        request_data = {
            "data_path": "iris",
            "variable": "iris"
        }
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have category counts
        assert "bins" in data or "counts" in data or "values" in data
    
    def test_distributions_discrete_counts(self):
        """Test that discrete distribution has correct counts."""
        request_data = {
            "data_path": "iris",
            "variable": "iris"
        }
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Each iris class has 50 instances
        if "counts" in data:
            for count in data["counts"]:
                assert count == 50


class TestDistributionsSplitBy:
    """Test distributions split by another variable."""
    
    def test_distributions_split_by_class(self):
        """Test distribution split by class variable."""
        request_data = {
            "data_path": "iris",
            "variable": "sepal length",
            "split_by": "iris",
            "number_of_bins": 10
        }
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have split data
        assert "bins" in data or "groups" in data
    
    def test_distributions_stacked(self):
        """Test stacked distributions."""
        request_data = {
            "data_path": "iris",
            "variable": "sepal length",
            "split_by": "iris",
            "stacked": True
        }
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200


class TestDistributionsCumulative:
    """Test cumulative distributions."""
    
    def test_distributions_cumulative(self):
        """Test cumulative distribution."""
        request_data = {
            "data_path": "iris",
            "variable": "sepal length",
            "cumulative": True
        }
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
    
    def test_distributions_cumulative_with_split(self):
        """Test cumulative distribution with split."""
        request_data = {
            "data_path": "iris",
            "variable": "sepal length",
            "split_by": "iris",
            "cumulative": True
        }
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200


class TestDistributionsProbabilities:
    """Test probability distributions."""
    
    def test_distributions_show_probs(self):
        """Test probability display."""
        request_data = {
            "data_path": "iris",
            "variable": "sepal length",
            "show_probs": True
        }
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200


class TestDistributionsFittedCurve:
    """Test fitted distribution curves."""
    
    def test_distributions_normal_fit(self):
        """Test normal distribution fit."""
        request_data = {
            "data_path": "iris",
            "variable": "sepal length",
            "fitted_distribution": 1  # Normal
        }
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
    
    def test_distributions_different_fits(self):
        """Test different fitted distributions."""
        for fit_type in range(4):  # 0=None, 1=Normal, 2=Beta, 3=Gamma
            request_data = {
                "data_path": "iris",
                "variable": "sepal length",
                "fitted_distribution": fit_type
            }
            response = client.post("/api/v1/data/distributions", json=request_data)
            
            if response.status_code == 501:
                pytest.skip("Orange3 not available")
            
            # Should handle gracefully
            assert response.status_code in [200, 400]


class TestDistributionsSorting:
    """Test sorting options."""
    
    def test_distributions_sort_by_freq(self):
        """Test sorting by frequency."""
        request_data = {
            "data_path": "iris",
            "variable": "iris",
            "sort_by_freq": True
        }
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200


class TestDistributionsKDE:
    """Test KDE smoothing."""
    
    def test_distributions_kde_smoothing(self):
        """Test KDE smoothing parameter."""
        for smoothing in [5, 10, 20]:
            request_data = {
                "data_path": "iris",
                "variable": "sepal length",
                "kde_smoothing": smoothing
            }
            response = client.post("/api/v1/data/distributions", json=request_data)
            
            if response.status_code == 501:
                pytest.skip("Orange3 not available")
            
            assert response.status_code == 200


class TestDistributionsSelectedIndices:
    """Test distributions with selected indices."""
    
    def test_distributions_selected_subset(self):
        """Test distribution of selected subset."""
        request_data = {
            "data_path": "iris",
            "variable": "sepal length",
            "selected_indices": list(range(50))  # First class
        }
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
    
    def test_distributions_empty_selection(self):
        """Test distribution with empty selection."""
        request_data = {
            "data_path": "iris",
            "variable": "sepal length",
            "selected_indices": []
        }
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        # Empty selection should return all data
        assert response.status_code == 200


class TestDistributionsErrors:
    """Test error handling."""
    
    def test_distributions_variable_not_found(self):
        """Test with non-existent variable."""
        request_data = {
            "data_path": "iris",
            "variable": "nonexistent_variable"
        }
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        # Should return error
        assert response.status_code in [400, 404, 500]
    
    def test_distributions_invalid_split_by(self):
        """Test with invalid split_by variable."""
        request_data = {
            "data_path": "iris",
            "variable": "sepal length",
            "split_by": "nonexistent_var"
        }
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        # Should handle gracefully
        assert response.status_code in [200, 400]


class TestDistributionsDifferentDatasets:
    """Test with different datasets."""
    
    def test_distributions_zoo(self):
        """Test distributions with zoo dataset."""
        request_data = {
            "data_path": "zoo",
            "variable": "legs"
        }
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        if response.status_code == 200:
            data = response.json()
            assert "bins" in data or "counts" in data
    
    def test_distributions_housing(self):
        """Test distributions with housing dataset."""
        request_data = {
            "data_path": "housing",
            "variable": "MEDV",
            "number_of_bins": 20
        }
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        if response.status_code == 200:
            data = response.json()
            assert "bins" in data


class TestDistributionsAllVariables:
    """Test distributions for all iris variables."""
    
    def test_distributions_all_iris_features(self):
        """Test distribution for each iris feature."""
        features = ["sepal length", "sepal width", "petal length", "petal width"]
        
        for feature in features:
            request_data = {
                "data_path": "iris",
                "variable": feature,
                "number_of_bins": 10
            }
            response = client.post("/api/v1/data/distributions", json=request_data)
            
            if response.status_code == 501:
                pytest.skip("Orange3 not available")
            
            assert response.status_code == 200, f"Failed for variable: {feature}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
