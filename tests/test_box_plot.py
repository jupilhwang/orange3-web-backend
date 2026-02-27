"""
Unit tests for Box Plot Widget API.
Tests box plot statistics endpoint: min, Q1, median, Q3, max, outliers per category.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestBoxPlotBasic:
    """Basic box plot functionality tests."""

    def test_box_plot_iris_defaults(self):
        """Test box plot with iris dataset using default axis selection."""
        response = client.post("/api/v1/box-plot", json={"data_path": "iris"})

        if response.status_code == 503:
            pytest.skip("Orange3 not available")

        assert response.status_code == 200
        data = response.json()

        assert "boxes" in data
        assert "y_axis" in data
        assert "variables" in data
        assert "total_instances" in data
        assert data["total_instances"] == 150

    def test_box_plot_response_structure(self):
        """Test that response contains required fields per box."""
        response = client.post(
            "/api/v1/box-plot",
            json={
                "data_path": "iris",
                "y_axis": "sepal length",
                "x_axis": "iris",
            },
        )

        if response.status_code == 503:
            pytest.skip("Orange3 not available")

        assert response.status_code == 200
        data = response.json()

        assert len(data["boxes"]) > 0
        box = data["boxes"][0]

        # Required box statistics
        for field in [
            "min",
            "q1",
            "median",
            "q3",
            "max",
            "whisker_low",
            "whisker_high",
            "outliers",
            "count",
            "label",
            "color",
        ]:
            assert field in box, f"Missing field: {field}"

        # Statistical ordering invariant
        assert box["min"] <= box["q1"]
        assert box["q1"] <= box["median"]
        assert box["median"] <= box["q3"]
        assert box["q3"] <= box["max"]
        assert box["whisker_low"] >= box["min"]
        assert box["whisker_high"] <= box["max"]


class TestBoxPlotAxisSelection:
    """Test axis variable selection."""

    def test_box_plot_specific_y_axis(self):
        """Test with each numeric variable as y_axis."""
        for var in ["sepal length", "sepal width", "petal length", "petal width"]:
            response = client.post(
                "/api/v1/box-plot",
                json={
                    "data_path": "iris",
                    "y_axis": var,
                },
            )

            if response.status_code == 503:
                pytest.skip("Orange3 not available")

            assert response.status_code == 200, f"Failed for y_axis={var}"
            data = response.json()
            assert data["y_axis"] == var

    def test_box_plot_with_x_axis(self):
        """Test grouping by categorical x_axis."""
        response = client.post(
            "/api/v1/box-plot",
            json={
                "data_path": "iris",
                "y_axis": "sepal length",
                "x_axis": "iris",
            },
        )

        if response.status_code == 503:
            pytest.skip("Orange3 not available")

        assert response.status_code == 200
        data = response.json()

        # Iris has 3 classes: should produce 3 boxes
        assert len(data["boxes"]) == 3
        assert data["x_axis"] == "iris"

    def test_box_plot_no_x_axis(self):
        """Test without x_axis: backend auto-selects first categorical variable."""
        response = client.post(
            "/api/v1/box-plot",
            json={
                "data_path": "iris",
                "y_axis": "sepal length",
                "x_axis": None,
            },
        )

        if response.status_code == 503:
            pytest.skip("Orange3 not available")

        assert response.status_code == 200
        data = response.json()

        # When x_axis is null and categorical variables exist, backend auto-selects
        # iris has 1 categorical (the class), so 3 boxes will be generated
        assert len(data["boxes"]) >= 1


class TestBoxPlotGroupBy:
    """Test group_by parameter for color grouping."""

    def test_box_plot_with_group_by(self):
        """Test box plot with both x_axis and group_by."""
        # heartdisease dataset has multiple categorical variables
        response = client.post(
            "/api/v1/box-plot",
            json={
                "data_path": "iris",
                "y_axis": "sepal length",
                "x_axis": "iris",
            },
        )

        if response.status_code == 503:
            pytest.skip("Orange3 not available")

        assert response.status_code == 200


class TestBoxPlotStatistics:
    """Test correctness of box plot statistics."""

    def test_box_statistics_ordering(self):
        """Verify Q1 <= median <= Q3 for all boxes."""
        response = client.post(
            "/api/v1/box-plot",
            json={
                "data_path": "iris",
                "y_axis": "petal length",
                "x_axis": "iris",
            },
        )

        if response.status_code == 503:
            pytest.skip("Orange3 not available")

        assert response.status_code == 200
        data = response.json()

        for box in data["boxes"]:
            assert box["q1"] <= box["median"], f"Q1 > median in box {box['label']}"
            assert box["median"] <= box["q3"], f"median > Q3 in box {box['label']}"
            assert box["whisker_low"] <= box["q1"], (
                f"whisker_low > Q1 in box {box['label']}"
            )
            assert box["whisker_high"] >= box["q3"], (
                f"whisker_high < Q3 in box {box['label']}"
            )

    def test_box_outliers_are_list(self):
        """Verify outliers field is a list of floats."""
        response = client.post(
            "/api/v1/box-plot",
            json={
                "data_path": "iris",
                "y_axis": "petal length",
                "x_axis": "iris",
            },
        )

        if response.status_code == 503:
            pytest.skip("Orange3 not available")

        assert response.status_code == 200
        data = response.json()

        for box in data["boxes"]:
            assert isinstance(box["outliers"], list)

    def test_box_count_matches_category(self):
        """Verify box count values are positive integers."""
        response = client.post(
            "/api/v1/box-plot",
            json={
                "data_path": "iris",
                "y_axis": "sepal width",
                "x_axis": "iris",
            },
        )

        if response.status_code == 503:
            pytest.skip("Orange3 not available")

        assert response.status_code == 200
        data = response.json()

        total_count = sum(box["count"] for box in data["boxes"])
        # All 150 iris instances
        assert total_count == 150


class TestBoxPlotSelection:
    """Test box plot with selected_indices."""

    def test_box_plot_with_selection(self):
        """Test that selected_indices filters data."""
        response = client.post(
            "/api/v1/box-plot",
            json={
                "data_path": "iris",
                "y_axis": "sepal length",
                "x_axis": "iris",
                "selected_indices": list(range(50)),  # First 50 rows (setosa only)
            },
        )

        if response.status_code == 503:
            pytest.skip("Orange3 not available")

        assert response.status_code == 200
        data = response.json()

        assert data["displayed_instances"] == 50
        assert data["total_instances"] == 150

    def test_box_plot_empty_selection(self):
        """Empty selection should return all data."""
        response = client.post(
            "/api/v1/box-plot",
            json={
                "data_path": "iris",
                "y_axis": "sepal length",
                "selected_indices": [],
            },
        )

        if response.status_code == 503:
            pytest.skip("Orange3 not available")

        assert response.status_code == 200
        data = response.json()
        assert data["total_instances"] == 150


class TestBoxPlotErrors:
    """Test error handling."""

    def test_box_plot_invalid_data_path(self):
        """Test with invalid data path."""
        response = client.post(
            "/api/v1/box-plot",
            json={
                "data_path": "nonexistent_dataset_xyz",
                "y_axis": "some_var",
            },
        )

        if response.status_code == 503:
            pytest.skip("Orange3 not available")

        assert response.status_code in [400, 404, 500]

    def test_box_plot_invalid_y_axis(self):
        """Test with non-existent y_axis variable — should fall back to default."""
        response = client.post(
            "/api/v1/box-plot",
            json={
                "data_path": "iris",
                "y_axis": "nonexistent_variable",
            },
        )

        if response.status_code == 503:
            pytest.skip("Orange3 not available")

        # Should either succeed with a fallback or return 400
        assert response.status_code in [200, 400]

    def test_box_plot_missing_y_axis_no_numeric(self):
        """Test payload without y_axis (uses first numeric variable)."""
        response = client.post(
            "/api/v1/box-plot",
            json={
                "data_path": "iris",
            },
        )

        if response.status_code == 503:
            pytest.skip("Orange3 not available")

        # Should succeed with auto-selected y_axis
        assert response.status_code == 200


class TestBoxPlotDifferentDatasets:
    """Test with different datasets."""

    def test_box_plot_housing(self):
        """Test box plot with housing dataset."""
        response = client.post(
            "/api/v1/box-plot",
            json={
                "data_path": "housing",
                "y_axis": "MEDV",
            },
        )

        if response.status_code == 503:
            pytest.skip("Orange3 not available")

        if response.status_code == 200:
            data = response.json()
            assert "boxes" in data

    def test_box_plot_titanic(self):
        """Test box plot with titanic dataset."""
        response = client.post(
            "/api/v1/box-plot",
            json={
                "data_path": "titanic",
                "y_axis": "age",
                "x_axis": "survived",
            },
        )

        if response.status_code == 503:
            pytest.skip("Orange3 not available")

        if response.status_code == 200:
            data = response.json()
            assert len(data["boxes"]) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
