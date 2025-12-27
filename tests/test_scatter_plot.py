"""
Unit tests for Scatter Plot Widget API.
Comprehensive tests for scatter plot visualization.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestScatterPlotBasic:
    """Basic scatter plot functionality tests."""
    
    def test_scatter_plot_iris(self):
        """Test scatter plot with iris dataset."""
        request_data = {
            "data_path": "iris",
            "axis_x": "sepal length",
            "axis_y": "sepal width"
        }
        response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "points" in data
        assert len(data["points"]) == 150
    
    def test_scatter_plot_default_axes(self):
        """Test scatter plot with default axis selection."""
        request_data = {
            "data_path": "iris"
        }
        response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have points (axes may not be returned explicitly)
        assert "points" in data
        assert len(data["points"]) == 150
    
    def test_scatter_plot_all_axes_combinations(self):
        """Test scatter plot with different axis combinations."""
        axes = ["sepal length", "sepal width", "petal length", "petal width"]
        
        for ax_x in axes[:2]:
            for ax_y in axes[2:]:
                request_data = {
                    "data_path": "iris",
                    "axis_x": ax_x,
                    "axis_y": ax_y
                }
                response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
                
                if response.status_code == 501:
                    pytest.skip("Orange3 not available")
                
                assert response.status_code == 200


class TestScatterPlotColorAttribute:
    """Color attribute tests."""
    
    def test_scatter_plot_with_color(self):
        """Test scatter plot with color attribute."""
        request_data = {
            "data_path": "iris",
            "axis_x": "sepal length",
            "axis_y": "sepal width",
            "color_attr": "iris"
        }
        response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Points should have color information
        assert "points" in data
        if data["points"]:
            point = data["points"][0]
            # Check for color-related fields
            has_color = "color" in point or "color_value" in point or "class" in point
            assert has_color or "iris" in str(point)
    
    def test_scatter_plot_color_with_continuous(self):
        """Test scatter plot with continuous color attribute."""
        request_data = {
            "data_path": "iris",
            "axis_x": "sepal length",
            "axis_y": "sepal width",
            "color_attr": "petal length"
        }
        response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200


class TestScatterPlotSizeAttribute:
    """Size attribute tests."""
    
    def test_scatter_plot_with_size(self):
        """Test scatter plot with size attribute."""
        request_data = {
            "data_path": "iris",
            "axis_x": "sepal length",
            "axis_y": "sepal width",
            "size_attr": "petal length"
        }
        response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200


class TestScatterPlotJittering:
    """Jittering tests."""
    
    def test_scatter_plot_no_jitter(self):
        """Test scatter plot without jittering."""
        request_data = {
            "data_path": "iris",
            "axis_x": "sepal length",
            "axis_y": "sepal width",
            "jittering": 0
        }
        response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
    
    def test_scatter_plot_with_jitter(self):
        """Test scatter plot with jittering."""
        request_data = {
            "data_path": "iris",
            "axis_x": "sepal length",
            "axis_y": "sepal width",
            "jittering": 0.5
        }
        response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
    
    def test_scatter_plot_max_jitter(self):
        """Test scatter plot with maximum jittering."""
        request_data = {
            "data_path": "iris",
            "axis_x": "sepal length",
            "axis_y": "sepal width",
            "jittering": 1.0
        }
        response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200


class TestScatterPlotSelection:
    """Selection and subset tests."""
    
    def test_scatter_plot_with_selection(self):
        """Test scatter plot with selected indices."""
        request_data = {
            "data_path": "iris",
            "axis_x": "sepal length",
            "axis_y": "sepal width",
            "selected_indices": [0, 1, 2, 3, 4]
        }
        response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have filtered points
        assert len(data.get("points", [])) == 5
    
    def test_scatter_plot_subset_indices(self):
        """Test scatter plot with subset indices."""
        request_data = {
            "data_path": "iris",
            "axis_x": "sepal length",
            "axis_y": "sepal width",
            "subset_indices": list(range(50))  # First class
        }
        response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
    
    def test_scatter_plot_empty_selection(self):
        """Test scatter plot with empty selection."""
        request_data = {
            "data_path": "iris",
            "axis_x": "sepal length",
            "axis_y": "sepal width",
            "selected_indices": []
        }
        response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Empty selection should return all points
        assert len(data.get("points", [])) == 150


class TestScatterPlotVariables:
    """Variable listing tests."""
    
    def test_scatter_plot_returns_variables(self):
        """Test that scatter plot returns variable info."""
        request_data = {
            "data_path": "iris"
        }
        response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "variables" in data
        assert len(data["variables"]) >= 4  # At least 4 iris features
    
    def test_scatter_plot_variable_types(self):
        """Test variable type information."""
        request_data = {
            "data_path": "iris"
        }
        response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check variable structure
        for var in data["variables"]:
            assert "name" in var
            assert "type" in var


class TestScatterPlotPointData:
    """Point data structure tests."""
    
    def test_scatter_plot_point_coordinates(self):
        """Test that points have x and y coordinates."""
        request_data = {
            "data_path": "iris",
            "axis_x": "sepal length",
            "axis_y": "sepal width"
        }
        response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check point structure
        if data["points"]:
            point = data["points"][0]
            assert "x" in point
            assert "y" in point
    
    def test_scatter_plot_point_values_in_range(self):
        """Test that point values are in expected range."""
        request_data = {
            "data_path": "iris",
            "axis_x": "sepal length",
            "axis_y": "sepal width"
        }
        response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        for point in data["points"]:
            # Iris sepal length: 4.3 - 7.9
            assert 4.0 <= point["x"] <= 8.0
            # Iris sepal width: 2.0 - 4.4
            assert 2.0 <= point["y"] <= 5.0


class TestScatterPlotDifferentDatasets:
    """Test with different datasets."""
    
    def test_scatter_plot_zoo(self):
        """Test scatter plot with zoo dataset."""
        request_data = {
            "data_path": "zoo"
        }
        response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        # Zoo has mostly discrete variables, may not be ideal for scatter
        assert response.status_code in [200, 400]
    
    def test_scatter_plot_housing(self):
        """Test scatter plot with housing dataset."""
        request_data = {
            "data_path": "housing",
            "axis_x": "CRIM",
            "axis_y": "MEDV"
        }
        response = client.post("/api/v1/visualize/scatter-plot", json=request_data)
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        if response.status_code == 200:
            data = response.json()
            assert len(data["points"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
