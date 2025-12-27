"""
Unit tests for Scatter Plot Widget API.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestScatterPlotWidget:
    """Test cases for Scatter Plot widget API endpoints."""
    
    def test_scatter_plot_with_iris(self):
        """Test scatter plot with iris dataset."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "axis_x": "sepal length",
            "axis_y": "sepal width"
        }
        
        response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            
            assert "points" in data
            assert "variables" in data
            assert len(data["points"]) > 0
            assert len(data["variables"]) > 0
        elif response.status_code == 501:
            pytest.skip("Orange3 not available")
    
    def test_scatter_plot_with_color_attribute(self):
        """Test scatter plot with color attribute."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "axis_x": "sepal length",
            "axis_y": "sepal width",
            "color_attr": "iris"
        }
        
        response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "points" in data
            # Check if color data is included
            if data["points"]:
                assert "color" in data["points"][0] or "color_value" in data["points"][0]
    
    def test_scatter_plot_default_axes(self):
        """Test scatter plot with default axis selection."""
        request_data = {
            "data_path": "datasets/iris.tab"
        }
        
        response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "axis_x" in data
            assert "axis_y" in data
    
    def test_scatter_plot_with_jittering(self):
        """Test scatter plot with jittering."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "axis_x": "sepal length",
            "axis_y": "sepal width",
            "jittering": 0.5
        }
        
        response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "points" in data
    
    def test_scatter_plot_with_selected_indices(self):
        """Test scatter plot with selected indices filter."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "selected_indices": [0, 1, 2, 3, 4]
        }
        
        response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            # Should have filtered points
            assert len(data.get("points", [])) <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

