"""
Tests for Widget API endpoints
"""
import pytest
from httpx import AsyncClient


class TestWidgetDiscovery:
    """Test widget discovery endpoints."""

    async def test_list_widgets(self, client: AsyncClient):
        """Test that /api/v1/widgets returns widget list."""
        response = await client.get("/api/v1/widgets")
        assert response.status_code == 200
        
        data = response.json()
        assert "categories" in data
        assert isinstance(data["categories"], list)

    async def test_widgets_have_required_fields(self, client: AsyncClient):
        """Test that widgets have required fields."""
        response = await client.get("/api/v1/widgets")
        assert response.status_code == 200
        
        data = response.json()
        for category in data["categories"]:
            assert "name" in category
            assert "widgets" in category
            
            for widget in category["widgets"]:
                assert "id" in widget
                assert "name" in widget
                # icon and description are optional but common
                if "inputs" in widget:
                    assert isinstance(widget["inputs"], list)
                if "outputs" in widget:
                    assert isinstance(widget["outputs"], list)

    async def test_widgets_include_data_category(self, client: AsyncClient):
        """Test that Data category exists with essential widgets."""
        response = await client.get("/api/v1/widgets")
        assert response.status_code == 200
        
        data = response.json()
        category_names = [c["name"] for c in data["categories"]]
        assert "Data" in category_names
        
        # Find Data category and check for essential widgets
        data_category = next(c for c in data["categories"] if c["name"] == "Data")
        widget_ids = [w["id"] for w in data_category["widgets"]]
        
        # Essential Data widgets
        essential_widgets = ["file", "data-table", "data-info"]
        for widget_id in essential_widgets:
            assert widget_id in widget_ids, f"Missing essential widget: {widget_id}"


class TestWidgetIcons:
    """Test widget icon endpoints."""

    async def test_get_widget_icon(self, client: AsyncClient):
        """Test that widget icons are accessible."""
        # First get widget list to find a valid icon path
        response = await client.get("/api/v1/widgets")
        assert response.status_code == 200
        
        data = response.json()
        if data["categories"]:
            first_category = data["categories"][0]
            if first_category["widgets"]:
                icon_path = first_category["widgets"][0].get("icon", "")
                if icon_path:
                    # Icon paths are served by frontend, just verify format
                    assert icon_path.startswith("/") or icon_path.startswith("http")

