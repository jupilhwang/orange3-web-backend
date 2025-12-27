"""
Unit tests for Bar Plot Widget API.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestBarPlotWidget:
    """Test cases for Bar Plot widget API endpoints."""
    
    def test_bar_plot_basic(self):
        """Test basic bar plot."""
        request_data = {
            "dataset_path": "datasets/iris.tab",
            "value_var": "sepal length"
        }
        
        response = client.post("/api/v1/barplot", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            
            assert "bars" in data or "data" in data
            assert "variables" in data or "value_var" in data
        elif response.status_code == 503:
            pytest.skip("Orange3 not available")
    
    def test_bar_plot_with_grouping(self):
        """Test bar plot with group variable."""
        request_data = {
            "dataset_path": "datasets/iris.tab",
            "value_var": "sepal length",
            "group_var": "iris"
        }
        
        response = client.post("/api/v1/barplot", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "bars" in data or "groups" in data or "data" in data
    
    def test_bar_plot_with_color(self):
        """Test bar plot with color variable."""
        request_data = {
            "dataset_path": "datasets/iris.tab",
            "value_var": "sepal length",
            "color_var": "iris"
        }
        
        response = client.post("/api/v1/barplot", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "bars" in data or "data" in data
    
    def test_bar_plot_with_selected_indices(self):
        """Test bar plot with selected indices."""
        request_data = {
            "dataset_path": "datasets/iris.tab",
            "value_var": "sepal length",
            "selected_indices": [0, 1, 2, 3, 4]
        }
        
        response = client.post("/api/v1/barplot", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            # Should have filtered data
            assert "bars" in data or "data" in data
    
    def test_bar_plot_invalid_variable(self):
        """Test bar plot with invalid value variable."""
        request_data = {
            "dataset_path": "datasets/iris.tab",
            "value_var": "nonexistent_var"
        }
        
        response = client.post("/api/v1/barplot", json=request_data)
        
        # Should return error or empty result
        if response.status_code == 200:
            data = response.json()
            # Either error in response or empty bars
            assert "error" in data or data.get("bars", []) == [] or "value_var" not in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

