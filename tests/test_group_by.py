"""
Unit tests for Group By Widget API.
Tests follow the pattern from test_select_columns.py.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestGroupByBasic:
    """Basic functionality tests."""

    def test_group_by_iris_species(self):
        """Group iris by species, aggregate sepal length mean."""
        response = client.post(
            "/api/v1/data/group-by",
            json={
                "data_path": "iris",
                "group_by_column": "iris",
                "aggregations": [
                    {"column": "sepal length", "function": "mean"},
                    {"column": "sepal width", "function": "mean"},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        # Iris has 3 species
        assert data["instances"] == 3
        # group column + 2 agg columns
        assert len(data["columns"]) == 3
        assert len(data["data"]) == 3

    def test_no_data_path_returns_empty(self):
        """Without data_path returns empty result."""
        response = client.post(
            "/api/v1/data/group-by",
            json={"group_by_column": "some_col", "aggregations": []},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["instances"] == 0
        assert data["columns"] == []
        assert data["data"] == []

    def test_empty_aggregations_returns_counts(self):
        """With no aggregations, returns group counts."""
        response = client.post(
            "/api/v1/data/group-by",
            json={"data_path": "iris", "group_by_column": "iris", "aggregations": []},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["instances"] == 3
        col_names = [c["name"] for c in data["columns"]]
        assert "count" in col_names
        assert data["group_by_column"] == "iris"


class TestGroupByAggregationFunctions:
    """Tests for each supported aggregation function."""

    @pytest.mark.parametrize(
        "fn", ["mean", "sum", "count", "min", "max", "std", "median"]
    )
    def test_all_supported_functions(self, fn):
        """All 7 supported functions return successful responses."""
        response = client.post(
            "/api/v1/data/group-by",
            json={
                "data_path": "iris",
                "group_by_column": "iris",
                "aggregations": [{"column": "sepal length", "function": fn}],
            },
        )
        assert response.status_code == 200, f"Failed for function: {fn}"
        data = response.json()
        assert data["success"] is True
        assert data["instances"] == 3

    def test_invalid_function_returns_422(self):
        """Unsupported aggregation function returns 422."""
        response = client.post(
            "/api/v1/data/group-by",
            json={
                "data_path": "iris",
                "group_by_column": "iris",
                "aggregations": [{"column": "sepal length", "function": "variance"}],
            },
        )
        assert response.status_code == 422

    def test_multiple_functions_same_column(self):
        """Multiple aggregations on the same column produce distinct output columns."""
        response = client.post(
            "/api/v1/data/group-by",
            json={
                "data_path": "iris",
                "group_by_column": "iris",
                "aggregations": [
                    {"column": "sepal length", "function": "mean"},
                    {"column": "sepal length", "function": "min"},
                    {"column": "sepal length", "function": "max"},
                    {"column": "sepal length", "function": "std"},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        # group col + 4 agg cols
        assert len(data["columns"]) == 5

    def test_multiple_columns_different_functions(self):
        """Aggregations across different columns."""
        response = client.post(
            "/api/v1/data/group-by",
            json={
                "data_path": "iris",
                "group_by_column": "iris",
                "aggregations": [
                    {"column": "sepal length", "function": "mean"},
                    {"column": "sepal width", "function": "sum"},
                    {"column": "petal length", "function": "count"},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        # group col + 3 agg cols
        assert len(data["columns"]) == 4


class TestGroupByColumnNaming:
    """Tests for output column naming conventions."""

    def test_aggregated_column_name_format(self):
        """Aggregated columns are named as column__function."""
        response = client.post(
            "/api/v1/data/group-by",
            json={
                "data_path": "iris",
                "group_by_column": "iris",
                "aggregations": [{"column": "sepal length", "function": "mean"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        col_names = [c["name"] for c in data["columns"]]
        assert "sepal length__mean" in col_names

    def test_group_column_type_is_categorical(self):
        """Group-by column metadata type is 'categorical'."""
        response = client.post(
            "/api/v1/data/group-by",
            json={
                "data_path": "iris",
                "group_by_column": "iris",
                "aggregations": [{"column": "sepal length", "function": "mean"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        group_col = next(c for c in data["columns"] if c["name"] == "iris")
        assert group_col["type"] == "categorical"

    def test_aggregated_column_type_is_numeric(self):
        """Aggregated column metadata type is 'numeric'."""
        response = client.post(
            "/api/v1/data/group-by",
            json={
                "data_path": "iris",
                "group_by_column": "iris",
                "aggregations": [{"column": "sepal length", "function": "mean"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        agg_col = next(c for c in data["columns"] if "sepal length" in c["name"])
        assert agg_col["type"] == "numeric"


class TestGroupByDataOutput:
    """Tests for correctness of output data."""

    def test_data_rows_match_instances(self):
        """Number of data rows equals reported instances."""
        response = client.post(
            "/api/v1/data/group-by",
            json={
                "data_path": "iris",
                "group_by_column": "iris",
                "aggregations": [{"column": "sepal length", "function": "mean"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == data["instances"]

    def test_mean_sepal_length_reasonable_range(self):
        """Mean sepal lengths for iris species fall in expected range."""
        response = client.post(
            "/api/v1/data/group-by",
            json={
                "data_path": "iris",
                "group_by_column": "iris",
                "aggregations": [{"column": "sepal length", "function": "mean"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        for row in data["data"]:
            mean_val = row[1]  # second column is sepal length__mean
            if mean_val is not None:
                # Iris sepal length mean should be between 4 and 8
                assert 4.0 <= mean_val <= 8.0, f"Unexpected mean value: {mean_val}"

    def test_count_sums_to_total(self):
        """Sum of group counts equals total dataset rows."""
        response = client.post(
            "/api/v1/data/group-by",
            json={
                "data_path": "iris",
                "group_by_column": "iris",
                "aggregations": [{"column": "sepal length", "function": "count"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        total = sum(row[1] for row in data["data"] if row[1] is not None)
        assert total == 150  # iris has 150 rows

    def test_row_values_length_matches_columns(self):
        """Each data row has exactly as many values as there are columns."""
        response = client.post(
            "/api/v1/data/group-by",
            json={
                "data_path": "iris",
                "group_by_column": "iris",
                "aggregations": [
                    {"column": "sepal length", "function": "mean"},
                    {"column": "petal length", "function": "max"},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        n_cols = len(data["columns"])
        for row in data["data"]:
            assert len(row) == n_cols


class TestGroupByErrorHandling:
    """Tests for error handling and edge cases."""

    def test_missing_group_column_returns_422(self):
        """Non-existent group-by column returns 422."""
        response = client.post(
            "/api/v1/data/group-by",
            json={
                "data_path": "iris",
                "group_by_column": "nonexistent_column",
                "aggregations": [],
            },
        )
        assert response.status_code == 422

    def test_nonexistent_dataset_returns_404(self):
        """Non-existent dataset path returns 404."""
        response = client.post(
            "/api/v1/data/group-by",
            json={
                "data_path": "this_dataset_does_not_exist_12345",
                "group_by_column": "col",
                "aggregations": [],
            },
        )
        assert response.status_code in (404, 500)

    def test_invalid_agg_column_is_skipped(self):
        """Aggregation on non-existent column is silently skipped."""
        response = client.post(
            "/api/v1/data/group-by",
            json={
                "data_path": "iris",
                "group_by_column": "iris",
                "aggregations": [
                    {"column": "nonexistent_col", "function": "mean"},
                    {"column": "sepal length", "function": "mean"},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        col_names = [c["name"] for c in data["columns"]]
        # nonexistent col skipped, only sepal length__mean present
        assert "sepal length__mean" in col_names
        assert not any("nonexistent_col" in n for n in col_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
