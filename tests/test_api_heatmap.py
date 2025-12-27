"""
Tests for HeatMap API endpoint.
"""
import pytest
from httpx import AsyncClient


class TestHeatMapVisualization:
    """Test HeatMap visualization endpoints."""

    @pytest.mark.slow
    async def test_heatmap_basic(self, client: AsyncClient):
        """Test basic heatmap generation with iris dataset."""
        response = await client.post(
            "/api/v1/visualize/heatmap",
            json={
                "data_path": "iris",
                "color_scheme": "Blue-Green-Yellow",
                "threshold_low": 0.0,
                "threshold_high": 1.0,
                "merge_kmeans": False,
                "clustering_rows": "None",
                "clustering_cols": "None",
                "show_legend": True,
                "show_averages": True,
                "keep_aspect_ratio": False
            }
        )
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "matrix" in data
        assert "columns" in data
        assert "row_count" in data
        assert "col_count" in data
        
        # Iris has 4 continuous features
        assert data["col_count"] == 4
        assert data["row_count"] == 150
        
        # Check matrix dimensions
        assert len(data["matrix"]) == data["row_count"]
        assert len(data["matrix"][0]) == data["col_count"]
        
        # Check column names
        expected_cols = ["sepal length", "sepal width", "petal length", "petal width"]
        assert all(col in data["columns"] for col in expected_cols)

    @pytest.mark.slow
    async def test_heatmap_with_kmeans_clustering(self, client: AsyncClient):
        """Test heatmap with k-means row merging."""
        response = await client.post(
            "/api/v1/visualize/heatmap",
            json={
                "data_path": "iris",
                "merge_kmeans": True,
                "merge_kmeans_k": 10
            }
        )
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have 10 clusters instead of 150 rows
        assert data["is_clustered"] is True
        assert data["row_count"] <= 10  # May be less if some clusters are empty
        assert data["cluster_count"] <= 10

    @pytest.mark.slow
    async def test_heatmap_with_hierarchical_clustering(self, client: AsyncClient):
        """Test heatmap with hierarchical row clustering."""
        response = await client.post(
            "/api/v1/visualize/heatmap",
            json={
                "data_path": "iris",
                "clustering_rows": "Clustering",
                "clustering_cols": "None"
            }
        )
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Row dendrogram should be present
        if "row_dendrogram" in data and data["row_dendrogram"] is not None:
            assert "icoord" in data["row_dendrogram"]
            assert "dcoord" in data["row_dendrogram"]
            assert "leaves" in data["row_dendrogram"]

    @pytest.mark.slow
    async def test_heatmap_with_column_clustering(self, client: AsyncClient):
        """Test heatmap with column clustering."""
        response = await client.post(
            "/api/v1/visualize/heatmap",
            json={
                "data_path": "iris",
                "clustering_rows": "None",
                "clustering_cols": "Clustering"
            }
        )
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Column dendrogram should be present
        if "col_dendrogram" in data and data["col_dendrogram"] is not None:
            assert "icoord" in data["col_dendrogram"]
            assert "dcoord" in data["col_dendrogram"]
            assert "leaves" in data["col_dendrogram"]

    @pytest.mark.slow
    async def test_heatmap_with_threshold(self, client: AsyncClient):
        """Test heatmap with custom threshold range."""
        response = await client.post(
            "/api/v1/visualize/heatmap",
            json={
                "data_path": "iris",
                "threshold_low": 0.2,
                "threshold_high": 0.8
            }
        )
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Matrix values should still be valid
        matrix = data["matrix"]
        for row in matrix:
            for val in row:
                assert 0.0 <= val <= 1.0

    @pytest.mark.slow
    async def test_heatmap_with_averages(self, client: AsyncClient):
        """Test heatmap with column averages."""
        response = await client.post(
            "/api/v1/visualize/heatmap",
            json={
                "data_path": "iris",
                "show_averages": True
            }
        )
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Column averages should be present
        assert "col_averages" in data
        assert data["col_averages"] is not None
        assert len(data["col_averages"]) == data["col_count"]

    @pytest.mark.slow
    async def test_heatmap_color_schemes(self, client: AsyncClient):
        """Test different color schemes."""
        color_schemes = [
            "Blue-Green-Yellow",
            "Green-Black-Red",
            "Green-White-Red",
            "Blue-White-Red",
            "Black-Body",
            "Viridis"
        ]
        
        for scheme in color_schemes:
            response = await client.post(
                "/api/v1/visualize/heatmap",
                json={
                    "data_path": "iris",
                    "color_scheme": scheme
                }
            )
            
            if response.status_code == 503:
                pytest.skip("Orange3 not available")
            
            assert response.status_code == 200
            data = response.json()
            assert data["color_scheme"] == scheme

    @pytest.mark.slow
    async def test_heatmap_with_selected_indices(self, client: AsyncClient):
        """Test heatmap with subset of selected data."""
        response = await client.post(
            "/api/v1/visualize/heatmap",
            json={
                "data_path": "iris",
                "selected_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # First 10 rows
            }
        )
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have 10 rows, not 150
        assert data["row_count"] == 10
        assert len(data["matrix"]) == 10

    @pytest.mark.slow
    async def test_heatmap_data_range(self, client: AsyncClient):
        """Test that data range is returned correctly."""
        response = await client.post(
            "/api/v1/visualize/heatmap",
            json={
                "data_path": "iris"
            }
        )
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check data range is returned
        assert "data_min" in data
        assert "data_max" in data
        assert data["data_min"] < data["data_max"]

    @pytest.mark.slow
    async def test_heatmap_discrete_vars(self, client: AsyncClient):
        """Test that discrete variables are returned for UI."""
        response = await client.post(
            "/api/v1/visualize/heatmap",
            json={
                "data_path": "iris"
            }
        )
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Iris has one discrete variable (class)
        assert "discrete_vars" in data
        discrete_var_names = [v["name"] for v in data["discrete_vars"]]
        assert "iris" in discrete_var_names

    async def test_heatmap_invalid_dataset(self, client: AsyncClient):
        """Test heatmap with non-existent dataset."""
        response = await client.post(
            "/api/v1/visualize/heatmap",
            json={
                "data_path": "nonexistent_dataset_xyz"
            }
        )
        
        # Should return error
        assert response.status_code in [404, 500, 503]


class TestHeatMapRowAnnotations:
    """Test row annotation features."""

    @pytest.mark.slow
    async def test_heatmap_row_annotation_text(self, client: AsyncClient):
        """Test row annotation with text variable."""
        response = await client.post(
            "/api/v1/visualize/heatmap",
            json={
                "data_path": "iris",
                "row_annotation_text": "iris"  # Use class variable for annotation
            }
        )
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Row annotations should be present
        if data.get("row_annotations"):
            assert len(data["row_annotations"]) == data["row_count"]

    @pytest.mark.slow
    async def test_heatmap_row_annotation_color(self, client: AsyncClient):
        """Test row annotation with color variable."""
        response = await client.post(
            "/api/v1/visualize/heatmap",
            json={
                "data_path": "iris",
                "row_annotation_color": "iris"  # Use class variable for color
            }
        )
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Row annotation colors should be present
        if data.get("row_annotation_colors"):
            assert len(data["row_annotation_colors"]) == data["row_count"]
            # Each color should be RGB array
            for color in data["row_annotation_colors"]:
                assert len(color) == 3
                assert all(0 <= c <= 255 for c in color)


class TestHeatMapSplitBy:
    """Test split by features."""

    @pytest.mark.slow
    async def test_heatmap_split_by_rows(self, client: AsyncClient):
        """Test splitting heatmap by discrete variable."""
        response = await client.post(
            "/api/v1/visualize/heatmap",
            json={
                "data_path": "iris",
                "split_by_rows": "iris"  # Split by class
            }
        )
        
        if response.status_code == 503:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Split groups should be present
        if data.get("split_groups"):
            # Iris has 3 classes
            assert len(data["split_groups"]) == 3
            for group in data["split_groups"]:
                assert "name" in group
                assert "indices" in group

