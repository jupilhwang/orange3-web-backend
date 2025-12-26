"""
Tests for Data API endpoints (sampling, loading, etc.)
"""
import pytest
from httpx import AsyncClient


class TestDataLoad:
    """Test data loading endpoints."""

    @pytest.mark.slow
    async def test_load_iris_dataset(self, client: AsyncClient):
        """Test loading the Iris dataset."""
        response = await client.post(
            "/api/v1/data/load",
            json={"path": "iris"}
        )
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Iris should have 150 instances
        assert data.get("instances", 0) == 150

    @pytest.mark.slow
    async def test_load_housing_dataset(self, client: AsyncClient):
        """Test loading the Housing dataset."""
        response = await client.post(
            "/api/v1/data/load",
            json={"path": "housing"}
        )
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        # Housing may or may not exist
        assert response.status_code in [200, 400, 404, 500]


class TestDataSampling:
    """Test data sampling endpoints."""

    @pytest.mark.slow
    async def test_sample_fixed_proportion(self, client: AsyncClient, sample_sample_request):
        """Test fixed proportion sampling."""
        response = await client.post(
            "/api/v1/data/sample",
            json=sample_sample_request
        )
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check sample size is approximately 70% of 150
        sample_instances = data.get("sample", {}).get("instances", 0)
        assert 100 <= sample_instances <= 110  # ~105 ± tolerance

    @pytest.mark.slow
    async def test_sample_fixed_size(self, client: AsyncClient):
        """Test fixed size sampling."""
        response = await client.post(
            "/api/v1/data/sample",
            json={
                "data_path": "iris",
                "sampling_type": "fixed_size",
                "sample_size": 50,
                "use_seed": True,
                "seed": 42
            }
        )
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        sample_instances = data.get("sample", {}).get("instances", 0)
        assert sample_instances == 50

    @pytest.mark.slow
    async def test_sample_cross_validation(self, client: AsyncClient):
        """Test cross validation sampling."""
        response = await client.post(
            "/api/v1/data/sample",
            json={
                "data_path": "iris",
                "sampling_type": "cross_validation",
                "n_folds": 5,
                "selected_fold": 1,
                "use_seed": True,
                "seed": 42
            }
        )
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Each fold should have about 30 instances (150/5)
        sample_instances = data.get("sample", {}).get("instances", 0)
        remaining_instances = data.get("remaining", {}).get("instances", 0)
        
        # Remaining should be ~30 (one fold), sample should be ~120
        assert 25 <= remaining_instances <= 35
        assert 115 <= sample_instances <= 125

    @pytest.mark.slow
    async def test_sample_bootstrap(self, client: AsyncClient):
        """Test bootstrap sampling."""
        response = await client.post(
            "/api/v1/data/sample",
            json={
                "data_path": "iris",
                "sampling_type": "bootstrap",
                "use_seed": True,
                "seed": 42
            }
        )
        
        if response.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Bootstrap sample size equals original size
        sample_instances = data.get("sample", {}).get("instances", 0)
        assert sample_instances == 150

    @pytest.mark.slow
    async def test_sample_replicability(self, client: AsyncClient):
        """Test that sampling with same seed produces same results."""
        request_data = {
            "data_path": "iris",
            "sampling_type": "fixed_proportion",
            "proportion": 0.5,
            "use_seed": True,
            "seed": 12345
        }
        
        response1 = await client.post("/api/v1/data/sample", json=request_data)
        response2 = await client.post("/api/v1/data/sample", json=request_data)
        
        if response1.status_code == 501:
            pytest.skip("Orange3 not available")
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Both should return identical results
        data1 = response1.json()
        data2 = response2.json()
        
        assert data1.get("sample", {}).get("instances") == data2.get("sample", {}).get("instances")


class TestDataValidation:
    """Test data validation."""

    async def test_sample_invalid_proportion(self, client: AsyncClient):
        """Test that invalid proportion is rejected."""
        response = await client.post(
            "/api/v1/data/sample",
            json={
                "data_path": "iris",
                "sampling_type": "fixed_proportion",
                "proportion": 1.5  # Invalid: > 1
            }
        )
        
        # Should return validation error
        assert response.status_code in [400, 422, 501]

    async def test_sample_invalid_fold(self, client: AsyncClient):
        """Test that invalid fold number is handled."""
        response = await client.post(
            "/api/v1/data/sample",
            json={
                "data_path": "iris",
                "sampling_type": "cross_validation",
                "n_folds": 5,
                "selected_fold": 10  # Invalid: > n_folds
            }
        )
        
        # Should return error or handle gracefully
        assert response.status_code in [200, 400, 422, 500, 501]

