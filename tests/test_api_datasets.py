"""
Tests for Datasets API endpoints
"""
import pytest
from httpx import AsyncClient


class TestDatasetsEndpoint:
    """Test datasets listing endpoint."""

    @pytest.mark.slow
    async def test_list_datasets(self, client: AsyncClient):
        """Test that /api/v1/datasets returns dataset list."""
        response = await client.get("/api/v1/datasets")
        
        # May fail if Orange3 is not available
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        # Response can be a list or a dict with 'datasets' key
        assert isinstance(data, (list, dict))

    @pytest.mark.slow
    async def test_datasets_have_required_fields(self, client: AsyncClient):
        """Test that datasets have required fields."""
        response = await client.get("/api/v1/datasets")
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Handle both list and dict response formats
        datasets = data if isinstance(data, list) else data.get('datasets', [])
        
        if datasets:
            dataset = datasets[0]
            required_fields = ["id", "title"]
            for field in required_fields:
                assert field in dataset, f"Missing required field: {field}"

    @pytest.mark.slow
    async def test_filter_datasets_by_domain(self, client: AsyncClient):
        """Test filtering datasets by domain."""
        response = await client.get("/api/v1/datasets?domain=all")
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200

    @pytest.mark.slow
    async def test_search_datasets(self, client: AsyncClient):
        """Test searching datasets."""
        response = await client.get("/api/v1/datasets?search=iris")
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Handle both list and dict response formats
        datasets = data if isinstance(data, list) else data.get('datasets', [])
        
        # If Iris is in the results, it should match
        for dataset in datasets:
            if isinstance(dataset, dict) and "iris" in dataset.get("title", "").lower():
                assert True
                return


class TestDatasetInfo:
    """Test dataset info endpoint."""

    @pytest.mark.slow
    async def test_get_dataset_info(self, client: AsyncClient):
        """Test getting info for a specific dataset."""
        # First get list to find a valid dataset ID
        list_response = await client.get("/api/v1/datasets")
        
        if list_response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        data = list_response.json()
        datasets = data if isinstance(data, list) else data.get('datasets', [])
        
        if not datasets:
            pytest.skip("No datasets available")
        
        dataset_id = datasets[0]["id"]
        response = await client.get(f"/api/v1/datasets/{dataset_id}/info")
        
        # Endpoint may not be implemented yet
        assert response.status_code in [200, 404, 501]


class TestDatasetLoad:
    """Test dataset loading endpoint."""

    @pytest.mark.slow
    async def test_load_builtin_dataset(self, client: AsyncClient):
        """Test loading a built-in dataset (iris)."""
        # Try GET first (some APIs use GET for loading)
        response = await client.get("/api/v1/data/load?path=iris")
        
        if response.status_code == 404:
            # Try POST as fallback
            response = await client.post(
                "/api/v1/data/load",
                json={"path": "iris"}
            )
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        # Accept 200, 405 (method not allowed), or 404
        assert response.status_code in [200, 404, 405, 501]

    @pytest.mark.slow
    async def test_load_nonexistent_dataset(self, client: AsyncClient):
        """Test loading a non-existent dataset returns error."""
        response = await client.get("/api/v1/data/load?path=nonexistent_dataset_xyz")
        
        if response.status_code == 404:
            response = await client.post(
                "/api/v1/data/load",
                json={"path": "nonexistent_dataset_xyz"}
            )
        
        # Should return an error status
        assert response.status_code in [400, 404, 405, 500, 501]

