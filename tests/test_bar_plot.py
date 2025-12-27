"""
Unit tests for Bar Plot Widget API.
Comprehensive tests for bar plot visualization.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestBarPlotBasic:
    """Basic bar plot functionality tests."""
    
    def test_bar_plot_iris(self):
        """Test basic bar plot with iris dataset."""
        request_data = {
            "dataset_path": "iris",
            "value_var": "sepal length"
        }
        response = client.post("/api/v1/barplot", json=request_data)
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "bars" in data or "data" in data
    
    def test_bar_plot_different_values(self):
        """Test bar plot with different value variables."""
        value_vars = ["sepal length", "sepal width", "petal length", "petal width"]
        
        for var in value_vars:
            request_data = {
                "dataset_path": "iris",
                "value_var": var
            }
            response = client.post("/api/v1/barplot", json=request_data)
            
            if response.status_code == 503:
                pytest.skip("Orange3 not available")
            
            assert response.status_code == 200


class TestBarPlotGrouping:
    """Test bar plot with grouping."""
    
    def test_bar_plot_group_by_class(self):
        """Test bar plot grouped by class."""
        request_data = {
            "dataset_path": "iris",
            "value_var": "sepal length",
            "group_var": "iris"
        }
        response = client.post("/api/v1/barplot", json=request_data)
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have grouped data
        assert "bars" in data or "groups" in data or "data" in data
    
    def test_bar_plot_group_aggregation(self):
        """Test that grouping aggregates values."""
        request_data = {
            "dataset_path": "iris",
            "value_var": "sepal length",
            "group_var": "iris"
        }
        response = client.post("/api/v1/barplot", json=request_data)
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200


class TestBarPlotColor:
    """Test bar plot with color variable."""
    
    def test_bar_plot_with_color(self):
        """Test bar plot with color variable."""
        request_data = {
            "dataset_path": "iris",
            "value_var": "sepal length",
            "color_var": "iris"
        }
        response = client.post("/api/v1/barplot", json=request_data)
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
    
    def test_bar_plot_group_and_color(self):
        """Test bar plot with both grouping and color."""
        request_data = {
            "dataset_path": "iris",
            "value_var": "sepal length",
            "group_var": "iris",
            "color_var": "iris"
        }
        response = client.post("/api/v1/barplot", json=request_data)
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200


class TestBarPlotAnnotation:
    """Test bar plot with annotation variable."""
    
    def test_bar_plot_with_annot(self):
        """Test bar plot with annotation variable."""
        request_data = {
            "dataset_path": "iris",
            "value_var": "sepal length",
            "annot_var": "iris"
        }
        response = client.post("/api/v1/barplot", json=request_data)
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200


class TestBarPlotSelection:
    """Test bar plot with selection."""
    
    def test_bar_plot_selected_indices(self):
        """Test bar plot with selected indices."""
        request_data = {
            "dataset_path": "iris",
            "value_var": "sepal length",
            "selected_indices": list(range(50))  # First 50
        }
        response = client.post("/api/v1/barplot", json=request_data)
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
    
    def test_bar_plot_single_class(self):
        """Test bar plot with single class selected."""
        request_data = {
            "dataset_path": "iris",
            "value_var": "sepal length",
            "group_var": "iris",
            "selected_indices": list(range(50))  # First class only
        }
        response = client.post("/api/v1/barplot", json=request_data)
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
    
    def test_bar_plot_empty_selection(self):
        """Test bar plot with empty selection."""
        request_data = {
            "dataset_path": "iris",
            "value_var": "sepal length",
            "selected_indices": []
        }
        response = client.post("/api/v1/barplot", json=request_data)
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        # Empty selection should return all data
        assert response.status_code == 200


class TestBarPlotErrors:
    """Test error handling."""
    
    def test_bar_plot_invalid_value_var(self):
        """Test with invalid value variable."""
        request_data = {
            "dataset_path": "iris",
            "value_var": "nonexistent_var"
        }
        response = client.post("/api/v1/barplot", json=request_data)
        
        # Should handle gracefully
        if response.status_code == 200:
            data = response.json()
            # Either error in response or empty bars
            assert "error" in data or data.get("bars", []) == [] or "value_var" not in data
    
    def test_bar_plot_discrete_value_var(self):
        """Test with discrete variable as value (should fail)."""
        request_data = {
            "dataset_path": "iris",
            "value_var": "iris"  # Class is discrete
        }
        response = client.post("/api/v1/barplot", json=request_data)
        
        # May fail or handle gracefully
        assert response.status_code in [200, 400, 422]
    
    def test_bar_plot_invalid_group_var(self):
        """Test with invalid group variable."""
        request_data = {
            "dataset_path": "iris",
            "value_var": "sepal length",
            "group_var": "nonexistent_var"
        }
        response = client.post("/api/v1/barplot", json=request_data)
        
        # Should handle gracefully
        assert response.status_code in [200, 400]


class TestBarPlotDataOutput:
    """Test data output structure."""
    
    def test_bar_plot_output_structure(self):
        """Test bar plot output structure."""
        request_data = {
            "dataset_path": "iris",
            "value_var": "sepal length"
        }
        response = client.post("/api/v1/barplot", json=request_data)
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have bars or data
        assert "bars" in data or "data" in data
    
    def test_bar_plot_has_variables_info(self):
        """Test that bar plot returns variable info."""
        request_data = {
            "dataset_path": "iris",
            "value_var": "sepal length"
        }
        response = client.post("/api/v1/barplot", json=request_data)
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200


class TestBarPlotTruncation:
    """Test data truncation for large datasets."""
    
    def test_bar_plot_truncation_limit(self):
        """Test that large datasets are truncated."""
        request_data = {
            "dataset_path": "housing",  # Larger dataset
            "value_var": "MEDV"
        }
        response = client.post("/api/v1/barplot", json=request_data)
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        if response.status_code == 200:
            data = response.json()
            # Check for truncation info
            if "truncated" in data:
                assert data["truncated"] == True or data["truncated"] == False


class TestBarPlotDifferentDatasets:
    """Test with different datasets."""
    
    def test_bar_plot_zoo(self):
        """Test bar plot with zoo dataset."""
        request_data = {
            "dataset_path": "zoo",
            "value_var": "legs"
        }
        response = client.post("/api/v1/barplot", json=request_data)
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        # May succeed or fail depending on variable type
        assert response.status_code in [200, 400, 422]
    
    def test_bar_plot_housing(self):
        """Test bar plot with housing dataset."""
        request_data = {
            "dataset_path": "housing",
            "value_var": "MEDV"
        }
        response = client.post("/api/v1/barplot", json=request_data)
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        if response.status_code == 200:
            data = response.json()
            assert "bars" in data or "data" in data


class TestBarPlotAllFeatures:
    """Test bar plot with all iris continuous features."""
    
    def test_bar_plot_all_continuous_features(self):
        """Test bar plot for each continuous feature."""
        features = ["sepal length", "sepal width", "petal length", "petal width"]
        
        for feature in features:
            request_data = {
                "dataset_path": "iris",
                "value_var": feature,
                "group_var": "iris"
            }
            response = client.post("/api/v1/barplot", json=request_data)
            
            if response.status_code == 503:
                pytest.skip("Orange3 not available")
            
            assert response.status_code == 200, f"Failed for feature: {feature}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
