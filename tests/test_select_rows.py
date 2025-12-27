"""
Unit tests for Select Rows Widget API.
Comprehensive tests based on Orange3's test_owselectrows.py patterns.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestSelectRowsContinuousVariable:
    """Test cases for continuous variable filtering."""
    
    def test_filter_continuous_equal(self):
        """Test equals operator on continuous variable."""
        request_data = {
            "data_source": "iris",
            "conditions": [{"variable": "sepal length", "operator": "=", "value": 5.0}]
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["matching_count"] == 10  # Exactly 10 iris have sepal length = 5.0
        assert data["total_count"] == 150
    
    def test_filter_continuous_not_equal(self):
        """Test not equals operator on continuous variable."""
        request_data = {
            "data_source": "iris",
            "conditions": [{"variable": "sepal length", "operator": "!=", "value": 5.0}]
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["matching_count"] == 140  # 150 - 10
    
    def test_filter_continuous_less(self):
        """Test less than operator on continuous variable."""
        request_data = {
            "data_source": "iris",
            "conditions": [{"variable": "sepal length", "operator": "<", "value": 5.0}]
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["matching_count"] > 0
        assert data["matching_count"] < 150
    
    def test_filter_continuous_less_equal(self):
        """Test less than or equal operator."""
        request_data = {
            "data_source": "iris",
            "conditions": [{"variable": "sepal length", "operator": "<=", "value": 5.0}]
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        # Should include both < 5.0 and = 5.0
        less_request = {
            "data_source": "iris",
            "conditions": [{"variable": "sepal length", "operator": "<", "value": 5.0}]
        }
        less_response = client.post("/api/v1/data/select-rows", json=less_request)
        less_count = less_response.json()["matching_count"]
        
        assert data["matching_count"] == less_count + 10  # + 10 that equal 5.0
    
    def test_filter_continuous_greater(self):
        """Test greater than operator on continuous variable."""
        request_data = {
            "data_source": "iris",
            "conditions": [{"variable": "sepal length", "operator": ">", "value": 6.0}]
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["matching_count"] == 61  # Known value for iris dataset
    
    def test_filter_continuous_greater_equal(self):
        """Test greater than or equal operator."""
        request_data = {
            "data_source": "iris",
            "conditions": [{"variable": "sepal length", "operator": ">=", "value": 6.0}]
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["matching_count"] > 61  # > 6.0 plus == 6.0
    
    def test_filter_continuous_between(self):
        """Test between operator on continuous variable."""
        request_data = {
            "data_source": "iris",
            "conditions": [{
                "variable": "sepal length", 
                "operator": "between", 
                "value": 5.0, 
                "value2": 6.0
            }]
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["matching_count"] > 0
        assert data["matching_count"] < 150
        
        # Verify all results are within range
        if data.get("data"):
            for row in data["data"]:
                sepal_length = row[0]
                assert 5.0 <= sepal_length <= 6.0
    
    def test_filter_continuous_outside(self):
        """Test outside operator on continuous variable."""
        request_data = {
            "data_source": "iris",
            "conditions": [{
                "variable": "sepal length",
                "operator": "outside",
                "value": 5.0,
                "value2": 6.0
            }]
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Outside + between should equal total
        between_request = {
            "data_source": "iris",
            "conditions": [{
                "variable": "sepal length",
                "operator": "between",
                "value": 5.0,
                "value2": 6.0
            }]
        }
        between_response = client.post("/api/v1/data/select-rows", json=between_request)
        between_count = between_response.json()["matching_count"]
        
        assert data["matching_count"] + between_count == 150


class TestSelectRowsDiscreteVariable:
    """Test cases for discrete (categorical) variable filtering."""
    
    def test_filter_discrete_is(self):
        """Test 'is' operator on discrete variable."""
        request_data = {
            "data_source": "iris",
            "conditions": [{"variable": "iris", "operator": "is", "value": "Iris-setosa"}]
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["matching_count"] == 50  # Each class has 50 instances
    
    def test_filter_discrete_is_not(self):
        """Test 'is_not' operator on discrete variable."""
        request_data = {
            "data_source": "iris",
            "conditions": [{"variable": "iris", "operator": "is_not", "value": "Iris-setosa"}]
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["matching_count"] == 100  # 150 - 50
    
    def test_filter_discrete_equals(self):
        """Test '=' operator on discrete variable (alias for 'is')."""
        request_data = {
            "data_source": "iris",
            "conditions": [{"variable": "iris", "operator": "=", "value": "Iris-versicolor"}]
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["matching_count"] == 50
    
    def test_filter_discrete_all_classes(self):
        """Test filtering for each class separately."""
        classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        
        for cls in classes:
            request_data = {
                "data_source": "iris",
                "conditions": [{"variable": "iris", "operator": "is", "value": cls}]
            }
            response = client.post("/api/v1/data/select-rows", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["matching_count"] == 50, f"Class {cls} should have 50 instances"


class TestSelectRowsMultipleConditions:
    """Test cases for multiple conditions (AND logic)."""
    
    def test_multiple_continuous_conditions(self):
        """Test multiple continuous conditions."""
        request_data = {
            "data_source": "iris",
            "conditions": [
                {"variable": "sepal length", "operator": ">", "value": 5.0},
                {"variable": "sepal width", "operator": "<", "value": 3.5}
            ]
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        # Result should be less than either condition alone
        assert data["matching_count"] < 150
    
    def test_mixed_conditions(self):
        """Test mix of continuous and discrete conditions."""
        request_data = {
            "data_source": "iris",
            "conditions": [
                {"variable": "sepal length", "operator": ">", "value": 5.5},
                {"variable": "iris", "operator": "is", "value": "Iris-setosa"}
            ]
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        # Setosa are generally smaller, so few should pass
        assert data["matching_count"] < 50
    
    def test_three_conditions(self):
        """Test three conditions combined."""
        request_data = {
            "data_source": "iris",
            "conditions": [
                {"variable": "sepal length", "operator": ">", "value": 5.0},
                {"variable": "sepal width", "operator": "<", "value": 3.5},
                {"variable": "petal length", "operator": ">", "value": 1.5}
            ]
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["matching_count"] >= 0


class TestSelectRowsEdgeCases:
    """Test edge cases and error handling."""
    
    def test_no_conditions_returns_all(self):
        """Test that no conditions returns all rows."""
        request_data = {
            "data_source": "iris",
            "conditions": []
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["matching_count"] == 150
        assert data["unmatched_count"] == 0
    
    def test_impossible_condition(self):
        """Test condition that matches no rows."""
        request_data = {
            "data_source": "iris",
            "conditions": [{"variable": "sepal length", "operator": "<", "value": 0}]
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["matching_count"] == 0
        assert data["unmatched_count"] == 150
    
    def test_dataset_not_found(self):
        """Test with non-existent dataset."""
        request_data = {
            "data_source": "nonexistent_dataset_12345",
            "conditions": []
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        assert response.status_code == 404
    
    def test_unknown_variable(self):
        """Test with unknown variable name."""
        request_data = {
            "data_source": "iris",
            "conditions": [{"variable": "unknown_var", "operator": "=", "value": 5.0}]
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        # Should either return all data (condition ignored) or handle gracefully
        assert response.status_code in [200, 400]
    
    def test_word_operators(self):
        """Test word-based operators."""
        operators = [
            ("equals", "="),
            ("greater", ">"),
            ("less", "<"),
            ("greater_equal", ">="),
            ("less_equal", "<="),
        ]
        
        for word_op, symbol_op in operators:
            word_request = {
                "data_source": "iris",
                "conditions": [{"variable": "sepal length", "operator": word_op, "value": 5.5}]
            }
            symbol_request = {
                "data_source": "iris", 
                "conditions": [{"variable": "sepal length", "operator": symbol_op, "value": 5.5}]
            }
            
            word_response = client.post("/api/v1/data/select-rows", json=word_request)
            symbol_response = client.post("/api/v1/data/select-rows", json=symbol_request)
            
            if word_response.status_code == 200 and symbol_response.status_code == 200:
                assert word_response.json()["matching_count"] == symbol_response.json()["matching_count"], \
                    f"Operator '{word_op}' should equal '{symbol_op}'"


class TestSelectRowsDataOutput:
    """Test data output format."""
    
    def test_output_has_data_rows(self):
        """Test that output includes actual data rows."""
        request_data = {
            "data_source": "iris",
            "conditions": [{"variable": "iris", "operator": "is", "value": "Iris-setosa"}]
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "data" in data
        assert len(data["data"]) == data["matching_count"]
    
    def test_output_has_columns_info(self):
        """Test that output includes column metadata when data is filtered."""
        request_data = {
            "data_source": "iris",
            "conditions": [{"variable": "sepal length", "operator": ">", "value": 5.0}]
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Columns are returned when conditions are applied
        if "columns" in data:
            assert len(data["columns"]) == 5  # 4 features + 1 target
            # Check column structure
            for col in data["columns"]:
                assert "name" in col
                assert "type" in col
                assert "role" in col
    
    def test_output_counts_consistency(self):
        """Test that counts are consistent."""
        request_data = {
            "data_source": "iris",
            "conditions": [{"variable": "sepal length", "operator": ">", "value": 5.5}]
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["matching_count"] + data["unmatched_count"] == data["total_count"]
        assert data["total_count"] == 150


class TestSelectRowsDifferentDatasets:
    """Test with different built-in datasets."""
    
    def test_with_zoo_dataset(self):
        """Test filtering on zoo dataset."""
        request_data = {
            "data_source": "zoo",
            "conditions": [{"variable": "legs", "operator": "=", "value": 4}]
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert data["matching_count"] > 0
    
    def test_with_housing_dataset(self):
        """Test filtering on housing dataset."""
        request_data = {
            "data_source": "housing",
            "conditions": [{"variable": "CRIM", "operator": "<", "value": 1.0}]
        }
        response = client.post("/api/v1/data/select-rows", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert data["matching_count"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
