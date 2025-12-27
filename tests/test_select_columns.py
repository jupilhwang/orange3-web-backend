"""
Unit tests for Select Columns Widget API.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestSelectColumnsWidget:
    """Test cases for Select Columns widget API endpoints."""
    
    def test_select_columns_basic(self):
        """Test basic column selection."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "features": ["sepal length", "sepal width"],
            "target": ["iris"],
            "metas": [],
            "ignored": ["petal length", "petal width"]
        }
        
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "features" in data
        assert "target" in data
    
    def test_select_columns_all_features(self):
        """Test with all columns as features."""
        request_data = {
            "data_path": "datasets/iris.tab",
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
    
    def test_select_columns_no_target(self):
        """Test with no target variable."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "features": ["sepal length", "sepal width"],
            "target": [],
            "metas": [],
            "ignored": []
        }
        
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["target"]) == 0
    
    def test_select_columns_with_metas(self):
        """Test with meta attributes."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "features": ["sepal length", "sepal width"],
            "target": ["iris"],
            "metas": ["petal length"],
            "ignored": ["petal width"]
        }
        
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["metas"]) == 1
    
    def test_select_columns_empty_request(self):
        """Test with empty column lists."""
        request_data = {
            "data_path": "datasets/iris.tab",
            "features": [],
            "target": [],
            "metas": [],
            "ignored": []
        }
        
        response = client.post("/api/v1/data/select-columns", json=request_data)
        
        assert response.status_code == 200
    
    def test_select_columns_no_data_path(self):
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

