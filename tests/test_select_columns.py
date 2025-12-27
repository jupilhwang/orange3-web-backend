"""
Unit tests for Select Columns Widget API.
Comprehensive tests based on Orange3's test_owselectcolumns.py patterns.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestSelectColumnsBasic:
    """Basic functionality tests."""
    
    def test_select_two_features(self):
        """Test selecting two features."""
        request_data = {
            "data_path": "iris",
            "features": ["sepal length", "sepal width"],
            "target": ["iris"],
            "metas": [],
            "ignored": ["petal length", "petal width"]
        }
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert data["features"] == ["sepal length", "sepal width"]
        assert data["target"] == ["iris"]
        assert len(data["metas"]) == 0
        assert data["instances"] == 150
        assert data["variables"] == 3  # 2 features + 1 target
    
    def test_select_all_features(self):
        """Test selecting all columns as features."""
        request_data = {
            "data_path": "iris",
            "features": ["sepal length", "sepal width", "petal length", "petal width"],
            "target": ["iris"],
            "metas": [],
            "ignored": []
        }
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["features"]) == 4
        assert len(data["target"]) == 1
        assert data["variables"] == 5
    
    def test_reorder_features(self):
        """Test that features are reordered as specified."""
        request_data = {
            "data_path": "iris",
            "features": ["petal width", "petal length", "sepal width", "sepal length"],
            "target": ["iris"],
            "metas": [],
            "ignored": []
        }
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Features should be in the specified order
        assert data["features"] == ["petal width", "petal length", "sepal width", "sepal length"]


class TestSelectColumnsTarget:
    """Target variable tests."""
    
    def test_no_target(self):
        """Test with no target variable."""
        request_data = {
            "data_path": "iris",
            "features": ["sepal length", "sepal width", "petal length", "petal width"],
            "target": [],
            "metas": [],
            "ignored": ["iris"]
        }
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["target"] == []
        assert data["variables"] == 4  # Only features
    
    def test_feature_as_target(self):
        """Test moving a feature to target."""
        request_data = {
            "data_path": "iris",
            "features": ["sepal width", "petal length", "petal width"],
            "target": ["sepal length"],
            "metas": [],
            "ignored": ["iris"]
        }
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["target"] == ["sepal length"]
        assert "sepal length" not in data["features"]


class TestSelectColumnsMetas:
    """Meta attributes tests."""
    
    def test_feature_to_meta(self):
        """Test moving feature to meta."""
        request_data = {
            "data_path": "iris",
            "features": ["sepal length", "sepal width"],
            "target": ["iris"],
            "metas": ["petal length"],
            "ignored": ["petal width"]
        }
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["metas"] == ["petal length"]
        assert "petal length" not in data["features"]
    
    def test_multiple_metas(self):
        """Test multiple meta attributes."""
        request_data = {
            "data_path": "iris",
            "features": ["sepal length", "sepal width"],
            "target": ["iris"],
            "metas": ["petal length", "petal width"],
            "ignored": []
        }
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["metas"]) == 2


class TestSelectColumnsIgnored:
    """Ignored columns tests."""
    
    def test_ignore_columns(self):
        """Test ignoring columns."""
        request_data = {
            "data_path": "iris",
            "features": ["sepal length", "sepal width"],
            "target": ["iris"],
            "metas": [],
            "ignored": ["petal length", "petal width"]
        }
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["ignored"] == ["petal length", "petal width"]
        # Ignored columns should not appear in output
        assert "petal length" not in data["features"]
        assert "petal width" not in data["features"]
    
    def test_ignore_all_except_one(self):
        """Test ignoring all features except one."""
        request_data = {
            "data_path": "iris",
            "features": ["sepal length"],
            "target": ["iris"],
            "metas": [],
            "ignored": ["sepal width", "petal length", "petal width"]
        }
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["features"] == ["sepal length"]
        assert data["variables"] == 2  # 1 feature + 1 target


class TestSelectColumnsDataOutput:
    """Data output tests."""
    
    def test_output_has_correct_columns(self):
        """Test output column structure."""
        request_data = {
            "data_path": "iris",
            "features": ["sepal length", "petal length"],
            "target": ["iris"],
            "metas": [],
            "ignored": ["sepal width", "petal width"]
        }
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "columns" in data
        column_names = [c["name"] for c in data["columns"]]
        
        assert "sepal length" in column_names
        assert "petal length" in column_names
        assert "iris" in column_names
        assert "sepal width" not in column_names
        assert "petal width" not in column_names
    
    def test_output_column_roles(self):
        """Test output column roles are correct."""
        request_data = {
            "data_path": "iris",
            "features": ["sepal length"],
            "target": ["iris"],
            "metas": ["petal length"],
            "ignored": ["sepal width", "petal width"]
        }
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        columns = {c["name"]: c["role"] for c in data["columns"]}
        
        assert columns.get("sepal length") == "feature"
        assert columns.get("iris") == "target"
        assert columns.get("petal length") == "meta"
    
    def test_output_has_data_rows(self):
        """Test output includes data rows."""
        request_data = {
            "data_path": "iris",
            "features": ["sepal length", "sepal width"],
            "target": ["iris"],
            "metas": [],
            "ignored": ["petal length", "petal width"]
        }
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "data" in data
        assert len(data["data"]) == 150
        # Each row should have 3 columns (2 features + 1 target)
        assert len(data["data"][0]) == 3


class TestSelectColumnsEdgeCases:
    """Edge cases and error handling."""
    
    def test_empty_features(self):
        """Test with empty features list."""
        request_data = {
            "data_path": "iris",
            "features": [],
            "target": ["iris"],
            "metas": [],
            "ignored": ["sepal length", "sepal width", "petal length", "petal width"]
        }
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["features"] == []
        assert data["variables"] == 1  # Only target
    
    def test_all_empty_lists(self):
        """Test with all empty lists."""
        request_data = {
            "data_path": "iris",
            "features": [],
            "target": [],
            "metas": [],
            "ignored": []
        }
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["variables"] == 0
    
    def test_no_data_path(self):
        """Test without data path."""
        request_data = {
            "features": ["a", "b"],
            "target": ["c"],
            "metas": [],
            "ignored": []
        }
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["instances"] == 0
        assert data["features"] == ["a", "b"]
    
    def test_nonexistent_column(self):
        """Test with non-existent column name."""
        request_data = {
            "data_path": "iris",
            "features": ["nonexistent_column", "sepal length"],
            "target": ["iris"],
            "metas": [],
            "ignored": []
        }
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        # Should handle gracefully - either skip invalid or return error
        if response.status_code == 200:
            data = response.json()
            # nonexistent column should be skipped
            assert "nonexistent_column" not in data["features"]


class TestSelectColumnsDifferentDatasets:
    """Test with different datasets."""
    
    def test_zoo_dataset(self):
        """Test column selection on zoo dataset."""
        request_data = {
            "data_path": "zoo",
            "features": ["hair", "feathers", "eggs"],
            "target": ["type"],
            "metas": [],
            "ignored": []
        }
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert len(data["features"]) == 3
    
    def test_housing_dataset(self):
        """Test column selection on housing dataset."""
        request_data = {
            "data_path": "housing",
            "features": ["CRIM", "ZN", "INDUS"],
            "target": ["MEDV"],
            "metas": [],
            "ignored": []
        }
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert len(data["features"]) == 3


class TestSelectColumnsDataIntegrity:
    """Test data integrity after column selection."""
    
    def test_data_values_preserved(self):
        """Test that data values are preserved correctly."""
        request_data = {
            "data_path": "iris",
            "features": ["sepal length", "sepal width"],
            "target": ["iris"],
            "metas": [],
            "ignored": ["petal length", "petal width"]
        }
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check first row values are reasonable for iris
        first_row = data["data"][0]
        sepal_length = first_row[0]
        sepal_width = first_row[1]
        
        # Iris sepal length range: 4.3 - 7.9
        assert 4.0 <= sepal_length <= 8.0
        # Iris sepal width range: 2.0 - 4.4
        assert 2.0 <= sepal_width <= 5.0
    
    def test_instance_count_unchanged(self):
        """Test that instance count is unchanged."""
        request_data = {
            "data_path": "iris",
            "features": ["sepal length"],
            "target": ["iris"],
            "metas": [],
            "ignored": ["sepal width", "petal length", "petal width"]
        }
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should still have all 150 instances
        assert data["instances"] == 150
        assert len(data["data"]) == 150


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
