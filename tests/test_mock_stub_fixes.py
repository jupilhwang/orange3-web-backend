"""
Tests for mock/stub removal fixes:
1. Data load failure now raises HTTP 500 instead of returning mock data
2. Link validation uses registry-based type checking instead of always-compatible stub
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from httpx import AsyncClient, ASGITransport

from app.workflow_manager import WorkflowManager
from app.core.models import (
    WorkflowCreate,
    NodeCreate,
    LinkValidation,
    LinkCompatibility,
    Position,
)


# ============================================================================
# Test 1: Data load error raises HTTP 500 (not mock fallback)
# ============================================================================


class TestDataLoadErrorRaisesHTTPException:
    """Verify that load_data_from_path raises HTTP 500 on failure."""

    @pytest.mark.asyncio
    async def test_load_data_error_raises_http_exception(self, client: AsyncClient):
        """
        When Orange3 Table() raises an exception, the endpoint should return
        HTTP 500 with a descriptive error instead of silently returning mock data.
        """
        # We patch Orange3 as "available" and force Table() to raise.
        # Table is imported locally inside load_data_from_path via
        # `from Orange.data import Table`, so we patch the module-level source.
        with (
            patch("app.core.orange_compat.ORANGE_AVAILABLE", True),
            patch(
                "Orange.data.Table",
                side_effect=FileNotFoundError("No such file: 'nonexistent_dataset'"),
            ),
        ):
            response = await client.get(
                "/api/v1/data/load",
                params={"path": "nonexistent_dataset"},
            )

        assert response.status_code == 500, (
            f"Expected HTTP 500 but got {response.status_code}. Body: {response.text}"
        )
        body = response.json()
        assert "detail" in body
        assert "Failed to load data from" in body["detail"]
        assert "nonexistent_dataset" in body["detail"]


# ============================================================================
# Tests 2-5: Link validation with registry-based type checking
# ============================================================================


def _make_manager_with_workflow(registry=None):
    """Helper: create a WorkflowManager with a workflow containing two nodes."""
    mgr = WorkflowManager(registry=registry)
    tenant = "test-tenant"

    wf = mgr.create_workflow(tenant, WorkflowCreate(title="Test WF"))

    source_node = mgr.add_node(
        tenant,
        wf.id,
        NodeCreate(
            widget_id="org.orange.widget.File",
            title="File",
            position=Position(x=0, y=0),
        ),
    )
    sink_node = mgr.add_node(
        tenant,
        wf.id,
        NodeCreate(
            widget_id="org.orange.widget.DataTable",
            title="Data Table",
            position=Position(x=200, y=0),
        ),
    )
    return mgr, tenant, wf, source_node, sink_node


class TestValidateLinkWithRegistry:
    """Tests for validate_link with an injected registry."""

    def test_validate_link_with_registry_compatible(self):
        """
        When registry reports compatible types, validate_link should
        return compatible=True.
        """
        mock_registry = MagicMock()

        # get_widget returns widget dicts with matching channel types
        mock_registry.get_widget.side_effect = lambda wid: {
            "org.orange.widget.File": {
                "outputs": [
                    {"id": "data_out", "name": "Data", "types": ["Orange.data.Table"]}
                ],
                "inputs": [],
            },
            "org.orange.widget.DataTable": {
                "outputs": [],
                "inputs": [
                    {"id": "data_in", "name": "Data", "types": ["Orange.data.Table"]}
                ],
            },
        }.get(wid)

        mock_registry.check_channel_compatibility.return_value = {
            "compatible": True,
            "strict": True,
            "dynamic": False,
        }

        mgr, tenant, wf, src, snk = _make_manager_with_workflow(registry=mock_registry)

        result = mgr.validate_link(
            tenant,
            wf.id,
            LinkValidation(
                source_node_id=src.id,
                source_channel="data_out",
                sink_node_id=snk.id,
                sink_channel="data_in",
            ),
        )

        assert result.compatible is True
        assert result.strict is True
        assert result.reason is None
        mock_registry.check_channel_compatibility.assert_called_once_with(
            ["Orange.data.Table"], ["Orange.data.Table"]
        )

    def test_validate_link_with_registry_incompatible(self):
        """
        When registry reports incompatible types, validate_link should
        return compatible=False with reason.
        """
        mock_registry = MagicMock()

        mock_registry.get_widget.side_effect = lambda wid: {
            "org.orange.widget.File": {
                "outputs": [
                    {"id": "data_out", "name": "Data", "types": ["Orange.data.Table"]}
                ],
                "inputs": [],
            },
            "org.orange.widget.DataTable": {
                "outputs": [],
                "inputs": [
                    {"id": "model_in", "name": "Model", "types": ["Orange.base.Model"]}
                ],
            },
        }.get(wid)

        mock_registry.check_channel_compatibility.return_value = {
            "compatible": False,
            "strict": False,
            "dynamic": False,
        }

        mgr, tenant, wf, src, snk = _make_manager_with_workflow(registry=mock_registry)

        result = mgr.validate_link(
            tenant,
            wf.id,
            LinkValidation(
                source_node_id=src.id,
                source_channel="data_out",
                sink_node_id=snk.id,
                sink_channel="model_in",
            ),
        )

        assert result.compatible is False
        assert result.reason is not None
        assert "Incompatible" in result.reason

    def test_validate_link_missing_channel(self):
        """
        When a channel ID does not exist on the widget, validate_link should
        return compatible=False with a descriptive reason.
        """
        mock_registry = MagicMock()

        mock_registry.get_widget.side_effect = lambda wid: {
            "org.orange.widget.File": {
                "outputs": [
                    {"id": "data_out", "name": "Data", "types": ["Orange.data.Table"]}
                ],
                "inputs": [],
            },
            "org.orange.widget.DataTable": {
                "outputs": [],
                "inputs": [
                    {"id": "data_in", "name": "Data", "types": ["Orange.data.Table"]}
                ],
            },
        }.get(wid)

        mgr, tenant, wf, src, snk = _make_manager_with_workflow(registry=mock_registry)

        # Use a non-existent source channel
        result = mgr.validate_link(
            tenant,
            wf.id,
            LinkValidation(
                source_node_id=src.id,
                source_channel="nonexistent_channel",
                sink_node_id=snk.id,
                sink_channel="data_in",
            ),
        )

        assert result.compatible is False
        assert result.reason is not None
        assert "not found" in result.reason.lower()

        # Also test missing sink channel
        result2 = mgr.validate_link(
            tenant,
            wf.id,
            LinkValidation(
                source_node_id=src.id,
                source_channel="data_out",
                sink_node_id=snk.id,
                sink_channel="nonexistent_sink",
            ),
        )

        assert result2.compatible is False
        assert result2.reason is not None
        assert "not found" in result2.reason.lower()

    def test_validate_link_no_registry_fallback(self):
        """
        When no registry is provided, validate_link should use permissive
        fallback: compatible=True, strict=False.
        """
        mgr, tenant, wf, src, snk = _make_manager_with_workflow(registry=None)

        result = mgr.validate_link(
            tenant,
            wf.id,
            LinkValidation(
                source_node_id=src.id,
                source_channel="any_output",
                sink_node_id=snk.id,
                sink_channel="any_input",
            ),
        )

        assert result.compatible is True
        assert result.strict is False
        assert result.reason is None
