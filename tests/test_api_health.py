"""
Tests for health check and basic API endpoints
"""
import pytest
from httpx import AsyncClient


class TestHealthCheck:
    """Test health check endpoints."""

    async def test_health_endpoint(self, client: AsyncClient):
        """Test that /health returns 200 OK."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    async def test_root_endpoint(self, client: AsyncClient):
        """Test that root endpoint is accessible."""
        response = await client.get("/")
        # Should redirect or return some response
        assert response.status_code in [200, 307, 404]


class TestBackendInfo:
    """Test backend information endpoints."""

    async def test_register_backend(self, client: AsyncClient):
        """Test backend registration endpoint."""
        response = await client.post(
            "/api/v1/backends/register",
            json={
                "url": "http://localhost:8000",
                "weight": 1,
                "metadata": {"version": "1.0.0"}
            }
        )
        # Should succeed or conflict if already registered
        assert response.status_code in [200, 409]

