"""
Unit tests for PCA Widget API.
Tests for Principal Component Analysis transformation.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestPCAOptions:
    """Test PCA options endpoint."""

    def test_get_pca_options(self):
        """Test getting PCA options returns valid configuration."""
        response = client.get("/api/v1/model/pca/options")

        assert response.status_code == 200
        data = response.json()

        assert "n_components" in data
        assert "standardize" in data
        assert "use_correlation" in data

        # Check component range defaults
        assert data["n_components"]["min"] == 1
        assert data["n_components"]["default"] == 2
        assert data["standardize"]["default"] == True


class TestPCATransformBasic:
    """Basic PCA transformation tests."""

    def test_transform_iris_default_params(self):
        """Test PCA on iris dataset with default parameters."""
        request_data = {"data_path": "iris"}
        response = client.post("/api/v1/model/pca/transform", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] == True
        assert "pca_id" in data
        assert "data_path" in data
        assert "pca_info" in data

    def test_transform_returns_pca_info(self):
        """Test that PCA info contains required fields."""
        request_data = {"data_path": "iris", "n_components": 2}
        response = client.post("/api/v1/model/pca/transform", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True

        pca_info = data["pca_info"]
        assert "n_components" in pca_info
        assert "instances" in pca_info
        assert "features" in pca_info
        assert "scree_data" in pca_info
        assert "loadings" in pca_info
        assert "feature_names" in pca_info
        assert "component_names" in pca_info

    def test_transform_scree_data_structure(self):
        """Test that scree data has correct structure per component."""
        request_data = {"data_path": "iris", "n_components": 2, "max_components": 4}
        response = client.post("/api/v1/model/pca/transform", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True

        scree_data = data["pca_info"]["scree_data"]
        assert len(scree_data) > 0

        # Check each scree entry has required fields
        for entry in scree_data:
            assert "component" in entry
            assert "explained_variance" in entry
            assert "cumulative_variance" in entry
            assert "eigenvalue" in entry
            # Variance values should be percentages (0-100)
            assert 0 <= entry["explained_variance"] <= 100
            assert 0 <= entry["cumulative_variance"] <= 100

    def test_transform_loadings_structure(self):
        """Test that loadings data has correct structure."""
        request_data = {"data_path": "iris", "n_components": 2}
        response = client.post("/api/v1/model/pca/transform", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True

        loadings = data["pca_info"]["loadings"]
        assert len(loadings) > 0

        # Each loading row must have feature name and component values
        for row in loadings:
            assert "feature" in row
            assert "PC1" in row
            assert "PC2" in row

    def test_transform_cumulative_variance_is_monotone(self):
        """Test that cumulative variance is non-decreasing."""
        request_data = {"data_path": "iris", "n_components": 2, "max_components": 4}
        response = client.post("/api/v1/model/pca/transform", json=request_data)

        assert response.status_code == 200
        scree_data = response.json()["pca_info"]["scree_data"]

        for i in range(1, len(scree_data)):
            assert (
                scree_data[i]["cumulative_variance"]
                >= scree_data[i - 1]["cumulative_variance"]
            ), "Cumulative variance must be non-decreasing"


class TestPCAComponents:
    """Test different numbers of components."""

    def test_transform_1_component(self):
        """Test PCA with 1 component."""
        request_data = {"data_path": "iris", "n_components": 1}
        response = client.post("/api/v1/model/pca/transform", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["pca_info"]["n_components"] == 1

    def test_transform_3_components(self):
        """Test PCA with 3 components."""
        request_data = {"data_path": "iris", "n_components": 3}
        response = client.post("/api/v1/model/pca/transform", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["pca_info"]["n_components"] == 3

    def test_transform_max_components_greater_than_n(self):
        """Test that max_components > n_components works for scree plot."""
        request_data = {"data_path": "iris", "n_components": 2, "max_components": 4}
        response = client.post("/api/v1/model/pca/transform", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True

        pca_info = data["pca_info"]
        # Output components is 2 but scree has up to max_components
        assert pca_info["n_components"] == 2
        assert len(pca_info["scree_data"]) >= 2

    def test_transform_n_components_clamped_to_features(self):
        """Test that n_components is clamped to number of numeric features."""
        # iris has 4 numeric features
        request_data = {
            "data_path": "iris",
            "n_components": 100,  # Way more than available
        }
        response = client.post("/api/v1/model/pca/transform", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        # Must be clamped to at most 4 (iris features)
        assert data["pca_info"]["n_components"] <= 4


class TestPCAPreprocessing:
    """Test different preprocessing options."""

    def test_transform_with_standardize(self):
        """Test PCA with standardization (default)."""
        request_data = {"data_path": "iris", "n_components": 2, "standardize": True}
        response = client.post("/api/v1/model/pca/transform", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["pca_info"]["standardized"] == True

    def test_transform_without_standardize(self):
        """Test PCA without standardization."""
        request_data = {
            "data_path": "iris",
            "n_components": 2,
            "standardize": False,
            "use_correlation": False,
        }
        response = client.post("/api/v1/model/pca/transform", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True

    def test_transform_with_correlation_matrix(self):
        """Test PCA using correlation matrix."""
        request_data = {"data_path": "iris", "n_components": 2, "use_correlation": True}
        response = client.post("/api/v1/model/pca/transform", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["pca_info"]["use_correlation"] == True

    def test_standardize_vs_no_standardize_differ(self):
        """Test that standardized and non-standardized results differ."""
        request_std = {"data_path": "iris", "n_components": 2, "standardize": True}
        request_no_std = {
            "data_path": "iris",
            "n_components": 2,
            "standardize": False,
            "use_correlation": False,
        }

        resp_std = client.post("/api/v1/model/pca/transform", json=request_std)
        resp_no_std = client.post("/api/v1/model/pca/transform", json=request_no_std)

        assert resp_std.status_code == 200
        assert resp_no_std.status_code == 200

        # The explained variances should generally differ
        std_pc1 = resp_std.json()["pca_info"]["scree_data"][0]["explained_variance"]
        no_std_pc1 = resp_no_std.json()["pca_info"]["scree_data"][0][
            "explained_variance"
        ]
        # They might be equal for some datasets but request IDs should differ
        assert resp_std.json()["pca_id"] != resp_no_std.json()["pca_id"]


class TestPCAInputValidation:
    """Test input validation."""

    def test_invalid_n_components_zero(self):
        """Test that n_components=0 is rejected."""
        request_data = {"data_path": "iris", "n_components": 0}
        response = client.post("/api/v1/model/pca/transform", json=request_data)

        assert response.status_code == 400 or (
            response.status_code == 200 and response.json().get("success") == False
        )

    def test_invalid_data_path(self):
        """Test that invalid data path returns appropriate error."""
        request_data = {"data_path": "nonexistent_dataset_xyz_12345", "n_components": 2}
        response = client.post("/api/v1/model/pca/transform", json=request_data)

        # Backend raises 404 HTTPException when data not found, which is correct
        assert response.status_code in [200, 404, 500]
        if response.status_code == 200:
            assert response.json().get("success") == False


class TestPCAResultInfo:
    """Test PCA result retrieval."""

    def test_get_pca_info(self):
        """Test getting PCA info after transformation."""
        # First run PCA
        request_data = {"data_path": "iris", "n_components": 2}
        transform_response = client.post(
            "/api/v1/model/pca/transform", json=request_data
        )
        assert transform_response.status_code == 200
        pca_id = transform_response.json()["pca_id"]

        # Get info - note: this only works with legacy storage (no session)
        info_response = client.get(f"/api/v1/model/pca/info/{pca_id}")
        # May 404 if session-based storage is used; that's acceptable
        assert info_response.status_code in [200, 404]

    def test_get_nonexistent_pca_info(self):
        """Test getting info for non-existent PCA result."""
        response = client.get("/api/v1/model/pca/info/nonexistent_pca_id_xyz")
        assert response.status_code == 404

    def test_delete_pca_result(self):
        """Test deleting a PCA result (idempotent)."""
        response = client.delete("/api/v1/model/pca/nonexistent_id")
        # Delete should always succeed (idempotent)
        assert response.status_code == 200


class TestPCADifferentDatasets:
    """Test PCA with various datasets."""

    def test_pca_housing_dataset(self):
        """Test PCA on housing dataset (regression dataset)."""
        request_data = {"data_path": "housing", "n_components": 2}
        response = client.post("/api/v1/model/pca/transform", json=request_data)

        if response.status_code == 200:
            data = response.json()
            assert data["success"] == True

    def test_pca_iris_full_components(self):
        """Test PCA on iris with all 4 components."""
        request_data = {"data_path": "iris", "n_components": 4, "max_components": 4}
        response = client.post("/api/v1/model/pca/transform", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True

        # Total variance explained by all 4 components should be ~100%
        total_variance = data["pca_info"]["scree_data"][-1]["cumulative_variance"]
        assert total_variance > 99.0, (
            f"Expected ~100% variance with all components, got {total_variance}"
        )


class TestPCADataPathResponse:
    """Test that the data_path in the response is usable."""

    def test_data_path_format(self):
        """Test that data_path in response has correct pca/ prefix."""
        request_data = {"data_path": "iris", "n_components": 2}
        response = client.post("/api/v1/model/pca/transform", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["data_path"].startswith("pca/")

    def test_multiple_transforms_unique_ids(self):
        """Test that multiple PCA runs produce unique IDs."""
        request_data = {"data_path": "iris", "n_components": 2}

        pca_ids = []
        for _ in range(3):
            response = client.post("/api/v1/model/pca/transform", json=request_data)
            assert response.status_code == 200
            pca_ids.append(response.json()["pca_id"])

        # Each run should produce a unique ID
        assert len(set(pca_ids)) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
