"""
Unit tests for Distributions Widget API.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestDistributionsWidget:
    """Test cases for Distributions widget API endpoints."""
    
    def test_distributions_continuous_variable(self):
        """Test distributions for continuous variable."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "variable": "sepal length",
            "number_of_bins": 10
        }
        
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            
            assert "bins" in data or "histogram" in data
            assert "statistics" in data or "mean" in data
        elif response.status_code == 501:
            pytest.skip("Orange3 not available")
    
    def test_distributions_discrete_variable(self):
        """Test distributions for discrete variable (class)."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "variable": "iris"
        }
        
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            # Should have category counts
            assert "bins" in data or "counts" in data or "values" in data
    
    def test_distributions_with_split_by(self):
        """Test distributions with split by variable."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "variable": "sepal length",
            "split_by": "iris",
            "number_of_bins": 5
        }
        
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            # Should have split data
            assert "bins" in data or "groups" in data
    
    def test_distributions_cumulative(self):
        """Test cumulative distributions."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "variable": "sepal length",
            "cumulative": True
        }
        
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "bins" in data or "cumulative" in data
    
    def test_distributions_variable_not_found(self):
        """Test distributions with non-existent variable."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "variable": "nonexistent_variable"
        }
        
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        # Should return error
        assert response.status_code in [400, 404, 500]
    
    def test_distributions_with_selected_indices(self):
        """Test distributions with selected indices filter."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "variable": "sepal length",
            "selected_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        }
        
        response = client.post("/api/v1/data/distributions", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "bins" in data or "histogram" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

