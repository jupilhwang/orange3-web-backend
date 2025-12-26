"""
Dedicated tests for Datasets widget functionality
"""
import pytest
from httpx import AsyncClient


class TestDatasetsListing:
    """Test dataset listing and filtering."""

    @pytest.mark.slow
    async def test_list_returns_many_datasets(self, client: AsyncClient):
        """Test that we get a substantial list of datasets."""
        response = await client.get("/api/v1/datasets")
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have many datasets (Orange3 has 100+)
        datasets = data if isinstance(data, list) else data.get('datasets', [])
        assert len(datasets) > 50, f"Expected many datasets, got {len(datasets)}"

    @pytest.mark.slow
    async def test_iris_in_datasets(self, client: AsyncClient):
        """Test that Iris dataset exists."""
        response = await client.get("/api/v1/datasets?search=iris")
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        datasets = data if isinstance(data, list) else data.get('datasets', [])
        
        # Find Iris
        iris_datasets = [d for d in datasets if 'iris' in d.get('title', '').lower()]
        assert len(iris_datasets) >= 1, "Iris dataset should exist"

    @pytest.mark.slow
    async def test_dataset_has_metadata(self, client: AsyncClient):
        """Test that datasets have expected metadata fields."""
        response = await client.get("/api/v1/datasets")
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        datasets = data if isinstance(data, list) else data.get('datasets', [])
        
        if not datasets:
            pytest.skip("No datasets available")
        
        # Check first dataset has expected fields
        dataset = datasets[0]
        expected_fields = ['id', 'title', 'instances', 'variables']
        
        for field in expected_fields:
            assert field in dataset, f"Dataset missing field: {field}"

    @pytest.mark.slow
    async def test_filter_by_search_term(self, client: AsyncClient):
        """Test search filter returns relevant results."""
        response = await client.get("/api/v1/datasets?search=heart")
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        datasets = data if isinstance(data, list) else data.get('datasets', [])
        
        # All results should contain 'heart' in title
        for dataset in datasets:
            title = dataset.get('title', '').lower()
            assert 'heart' in title, f"Search result '{title}' doesn't match 'heart'"


class TestDatasetLoading:
    """Test dataset download and loading."""

    @pytest.mark.slow
    async def test_load_iris_dataset(self, client: AsyncClient):
        """Test loading Iris dataset returns valid data."""
        # First, find the Iris dataset ID
        list_response = await client.get("/api/v1/datasets?search=iris")
        
        if list_response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        data = list_response.json()
        datasets = data if isinstance(data, list) else data.get('datasets', [])
        
        iris = next((d for d in datasets if 'iris' in d.get('filename', '').lower()), None)
        
        if not iris:
            pytest.skip("Iris dataset not found in list")
        
        # Load the dataset
        dataset_id = iris['id']
        load_response = await client.post(f"/api/v1/datasets/{dataset_id}/load")
        
        # May not be implemented yet
        if load_response.status_code == 404:
            pytest.skip("Dataset load endpoint not implemented")
        
        if load_response.status_code == 200:
            result = load_response.json()
            # Iris has 150 instances and 5 variables
            assert result.get('instances', 0) == 150
            assert result.get('variables', 0) >= 4


class TestDatasetIntegration:
    """Test Datasets widget integration with workflow."""

    @pytest.mark.slow
    async def test_datasets_to_data_table_flow(self, client: AsyncClient):
        """Test that loaded dataset can be used in Data Table."""
        # This tests the data flow: Datasets -> Data Table
        
        # Step 1: Load a dataset
        load_response = await client.post(
            "/api/v1/data/load",
            json={"path": "iris"}
        )
        
        if load_response.status_code in [404, 405, 501]:
            pytest.skip("Data load not available")
        
        if load_response.status_code == 200:
            data = load_response.json()
            
            # Should have data
            assert data.get('instances', 0) > 0
            
            # Should have columns
            columns = data.get('columns', [])
            assert len(columns) > 0, "Data should have columns"

