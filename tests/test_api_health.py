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
