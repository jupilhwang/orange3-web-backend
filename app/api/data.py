"""
Data loading endpoints.

Provides /data/load, /data/load-url with SSRF protection,
file path resolution, pagination, and domain column extraction.
"""

import asyncio
import ipaddress
import json
import logging
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from ..core.config import get_upload_dir
from ..core.globals import ORANGE_AVAILABLE, get_availability
from ..core.mock_data import get_mock_data_info

logger = logging.getLogger(__name__)

data_router = APIRouter(tags=["Data"])


# ============================================================================
# SSRF Validation
# ============================================================================


async def validate_url_for_ssrf(url: str) -> tuple[bool, str]:
    """
    Validate URL to prevent SSRF attacks.

    Returns:
        (is_valid, error_message)
    """
    try:
        parsed = urlparse(url)

        if not parsed.scheme:
            return False, "URL must have a scheme (http/https)"

        if parsed.scheme not in ("http", "https"):
            return False, f"Scheme '{parsed.scheme}' not allowed. Only http/https."

        if not parsed.hostname:
            return False, "URL must have a hostname"

        blocked_hostnames = [
            "169.254.169.254",  # AWS/Azure/GCP metadata
            "metadata.google.internal",
            "metadata",
        ]
        if parsed.hostname.lower() in blocked_hostnames:
            return False, f"Blocked hostname: {parsed.hostname}"

        try:
            ip = ipaddress.ip_address(parsed.hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
                return (
                    False,
                    f"Private/internal IP addresses not allowed: {parsed.hostname}",
                )
        except ValueError:
            import socket

            try:
                resolved_ip = await asyncio.to_thread(
                    socket.gethostbyname, parsed.hostname
                )
                ip = ipaddress.ip_address(resolved_ip)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    return False, f"Hostname resolves to private IP: {resolved_ip}"
            except socket.gaierror:
                return False, f"Cannot resolve hostname: {parsed.hostname}"

        return True, ""

    except Exception as e:
        return False, f"Invalid URL: {str(e)}"


# ============================================================================
# Helpers
# ============================================================================


class UrlLoadRequest(BaseModel):
    url: str


def _extract_domain_columns(domain) -> list[dict]:
    """Extract column metadata from an Orange3 domain."""
    columns = []
    for var in domain.attributes:
        columns.append(
            {
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical",
                "role": "feature",
                "values": ", ".join(var.values)
                if hasattr(var, "values") and var.values
                else "",
            }
        )
    if domain.class_var:
        var = domain.class_var
        columns.append(
            {
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical",
                "role": "target",
                "values": ", ".join(var.values)
                if hasattr(var, "values") and var.values
                else "",
            }
        )
    for var in domain.metas:
        columns.append(
            {
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical",
                "role": "meta",
                "values": "",
            }
        )
    return columns


async def _resolve_file_path(
    path: str, tenant_id: Optional[str]
) -> tuple[str, Any, Any]:
    """Resolve path to actual filesystem path.

    Returns (actual_path, temp_file, metadata).
    Handles file: prefix (storage lookup), uploads/ prefix, and datasets/ prefix.
    Callers are responsible for cleaning up temp_file if not None.
    """
    import tempfile

    from ..core.file_storage import get_file, get_file_metadata

    actual_path = path
    temp_file = None
    metadata = None

    if path.startswith("file:"):
        file_id = path.replace("file:", "")
        logger.info(f"Loading file by ID: {file_id}, tenant: {tenant_id}")

        metadata = await get_file_metadata(file_id, tenant_id)
        if not metadata:
            return path, None, None

        content = await get_file(file_id, tenant_id)
        if not content:
            return path, None, metadata

        suffix = Path(metadata.filename).suffix or ".tab"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(content)
        temp_file.close()
        actual_path = temp_file.name
        logger.info(f"Created temp file: {actual_path} for {metadata.filename}")

    elif path.startswith("uploads/"):
        from ..core.config import get_upload_dir as _get_upload_dir

        upload_dir = _get_upload_dir()
        full_path = upload_dir / path.replace("uploads/", "")
        if full_path.exists():
            actual_path = str(full_path)

    elif path.startswith("datasets/"):
        actual_path = path.replace("datasets/", "").split(".")[0]

    elif "/" in path and not path.startswith(("/", ".")):
        from ..core.config import get_datasets_cache_dir as _get_datasets_cache_dir

        datasets_cache_dir = _get_datasets_cache_dir()
        full_path = datasets_cache_dir / path
        if full_path.exists():
            actual_path = str(full_path)

    return actual_path, temp_file, metadata


async def _apply_metadata_overrides(
    columns: list, path: str, tenant_id: Optional[str]
) -> list:
    """Read .metadata.json sidecar and apply column type/role overrides."""
    from ..core.config import get_tenant_upload_dir

    file_id_for_meta = path
    if path.startswith("file:"):
        file_id_for_meta = path.replace("file:", "")
    elif path.startswith("uploads/"):
        file_id_for_meta = path.replace("uploads/", "")

    upload_dir = get_tenant_upload_dir(tenant_id or "default")
    metadata_path = upload_dir / f"{file_id_for_meta}.metadata.json"

    if not metadata_path.exists():
        return columns

    try:

        def _read_metadata():
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)

        metadata_overrides = await asyncio.to_thread(_read_metadata)

        meta_cols = {col["name"]: col for col in metadata_overrides.get("columns", [])}

        for col in columns:
            if col["name"] in meta_cols:
                override = meta_cols[col["name"]]
                col["type"] = override.get("type", col["type"])
                col["role"] = override.get("role", col["role"])

        logger.info(f"Applied column metadata overrides from {metadata_path}")
    except Exception as e:
        logger.warning(f"Failed to load metadata overrides: {e}")

    return columns


def _paginate_orange_data(
    data: Any, offset: int, limit: int
) -> tuple[list | None, dict | None]:
    """Slice Orange Table rows and build a pagination dict."""
    total_rows = len(data)

    if limit is None or limit <= 0:
        return None, None

    end_idx = min(offset + limit, total_rows)
    paginated_rows = data[offset:end_idx]

    paginated_data = []
    for row in paginated_rows:
        row_data = []
        for val in row:
            if hasattr(val, "is_nan") and val.is_nan():
                row_data.append(None)
            else:
                row_data.append(float(val) if not isinstance(val, str) else val)
        if data.domain.class_var:
            class_val = row.get_class()
            if hasattr(class_val, "is_nan") and class_val.is_nan():
                row_data.append(None)
            else:
                row_data.append(str(class_val))
        paginated_data.append(row_data)

    pagination = {
        "offset": offset,
        "limit": limit,
        "total": total_rows,
        "hasMore": end_idx < total_rows,
    }

    return paginated_data, pagination


# ============================================================================
# Endpoints
# ============================================================================


@data_router.get("/data/load")
async def load_data_from_path(
    path: str,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    offset: int = 0,
    limit: Optional[int] = None,
) -> dict:
    """Load data from a local file path or file ID.

    [BE-PERF-001] Pagination support added for large datasets.
    """
    if not ORANGE_AVAILABLE:
        return get_mock_data_info(path)

    actual_path, temp_file, metadata = await _resolve_file_path(path, x_tenant_id)

    if path.startswith("file:") and metadata is None:
        return {
            "name": "File Not Found",
            "description": "파일을 찾을 수 없습니다. 파일을 다시 업로드해주세요.",
            "path": path,
            "instances": 0,
            "features": 0,
            "missingValues": False,
            "classType": "None",
            "classValues": None,
            "metaAttributes": 0,
            "columns": [],
            "error": f"File not found: {path.replace('file:', '')}",
        }

    if path.startswith("file:") and temp_file is None and metadata is not None:
        file_id = path.replace("file:", "")
        return {
            "name": metadata.original_filename or metadata.filename,
            "description": "파일 내용을 읽을 수 없습니다.",
            "path": path,
            "instances": 0,
            "features": 0,
            "missingValues": False,
            "classType": "None",
            "classValues": None,
            "metaAttributes": 0,
            "columns": [],
            "error": f"File content not found: {file_id}",
        }

    try:
        from Orange.data import Table

        data = await asyncio.to_thread(Table, actual_path)

        columns = _extract_domain_columns(data.domain)
        columns = await _apply_metadata_overrides(columns, path, x_tenant_id)

        if path.startswith("file:") and metadata:
            original_name = metadata.original_filename or metadata.filename
            display_name = Path(original_name).stem
        else:
            display_name = data.name or path.split("/")[-1].split(":")[-1]

        paginated_data, pagination = _paginate_orange_data(data, offset, limit or 0)

        response = {
            "name": display_name,
            "description": "",
            "path": path,
            "instances": len(data),
            "features": len(data.domain.attributes),
            "missingValues": data.has_missing(),
            "classType": "Classification"
            if data.domain.class_var and not data.domain.class_var.is_continuous
            else "Regression"
            if data.domain.class_var
            else "None",
            "classValues": len(data.domain.class_var.values)
            if data.domain.class_var and hasattr(data.domain.class_var, "values")
            else None,
            "metaAttributes": len(data.domain.metas),
            "columns": columns,
        }

        if pagination:
            response["data"] = paginated_data
            response["pagination"] = pagination

        return response
    except Exception as e:
        logger.error(f"Failed to load data from {actual_path}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to load data from '{path}': {str(e)}"
        )
    finally:
        if temp_file and Path(temp_file.name).exists():
            try:
                Path(temp_file.name).unlink()
            except Exception as e:
                logger.debug(f"Suppressed error: {e}")


@data_router.post("/data/load-url")
async def load_data_from_url(request: UrlLoadRequest) -> dict:
    """Load data from a URL with SSRF protection."""
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Orange3 not available")

    url = request.url
    is_valid, error_msg = await validate_url_for_ssrf(url)
    if not is_valid:
        logger.warning(f"SSRF attempt blocked: {url} - {error_msg}")
        raise HTTPException(status_code=403, detail=f"URL not allowed: {error_msg}")

    try:
        from Orange.data import Table
        import tempfile
        import httpx
        import os

        logger.info(f"Downloading from validated URL: {url}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp_path = tmp.name

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                with open(tmp_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        f.write(chunk)

        try:
            data = await asyncio.to_thread(Table, tmp_path)
            columns = _extract_domain_columns(data.domain)

            return {
                "name": url.split("/")[-1],
                "description": f"Loaded from {url}",
                "instances": len(data),
                "features": len(data.domain.attributes),
                "missingValues": data.has_missing(),
                "classType": "Classification"
                if data.domain.class_var and not data.domain.class_var.is_continuous
                else "Regression"
                if data.domain.class_var
                else "None",
                "classValues": len(data.domain.class_var.values)
                if data.domain.class_var and hasattr(data.domain.class_var, "values")
                else None,
                "metaAttributes": len(data.domain.metas),
                "columns": columns,
            }
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
