"""
Data loading endpoint handlers.

Provides /data/load and /data/load-url with SSRF protection,
file path resolution, pagination, and domain column extraction.
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from ..core.globals import ORANGE_AVAILABLE
from ..core.mock_data import get_mock_data_info
from ..core.ssrf import fetch_ssrf_safe, validate_url_for_ssrf
from .data_helpers import (
    apply_metadata_overrides,
    build_data_response,
    extract_domain_columns,
    file_content_missing_response,
    file_not_found_response,
    paginate_orange_data,
    resolve_file_path,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Data"])


# ── Request models ───────────────────────────────────────────────


class UrlLoadRequest(BaseModel):
    url: str


# ── Endpoints ────────────────────────────────────────────────────


@router.get("/data/load")
async def load_data_from_path(
    path: str,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    offset: int = 0,
    limit: Optional[int] = None,
) -> dict:
    """Load data from a local file path or file ID."""
    if not ORANGE_AVAILABLE:
        return get_mock_data_info(path)

    actual_path, temp_file, metadata = await resolve_file_path(path, x_tenant_id)

    if path.startswith("file:") and metadata is None:
        return file_not_found_response(path)

    if path.startswith("file:") and temp_file is None and metadata is not None:
        return file_content_missing_response(path, metadata)

    try:
        from Orange.data import Table

        data = await asyncio.to_thread(Table, actual_path)

        columns = extract_domain_columns(data.domain)
        columns = await apply_metadata_overrides(columns, path, x_tenant_id)

        if path.startswith("file:") and metadata:
            original_name = metadata.original_filename or metadata.filename
            display_name = Path(original_name).stem
        else:
            display_name = data.name or path.split("/")[-1].split(":")[-1]

        paginated_data, pagination = paginate_orange_data(data, offset, limit or 0)

        response = build_data_response(data, display_name, path, columns)

        if pagination:
            response["data"] = paginated_data
            response["pagination"] = pagination

        return response
    except Exception as e:
        logger.exception("Failed to load data from path '%s'", path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load data from '{path}'",
        )
    finally:
        if temp_file and Path(temp_file.name).exists():
            try:
                Path(temp_file.name).unlink()
            except Exception as e:
                logger.debug("Suppressed error: %s", e)


@router.post("/data/load-url")
async def load_data_from_url(request: UrlLoadRequest) -> dict:
    """Load data from a URL with SSRF protection."""
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Orange3 not available")

    url = request.url
    is_valid, error_msg, _resolved_ip = await validate_url_for_ssrf(url)
    if not is_valid:
        logger.warning("SSRF attempt blocked: %s - %s", url, error_msg)
        raise HTTPException(status_code=403, detail=f"URL not allowed: {error_msg}")

    try:
        from Orange.data import Table

        logger.info("Downloading from validated URL: %s", url)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp_path = tmp.name

        response = await fetch_ssrf_safe(url)
        with open(tmp_path, "wb") as f:
            f.write(response.content)

        try:
            data = await asyncio.to_thread(Table, tmp_path)
            columns = extract_domain_columns(data.domain)

            return build_data_response(
                data,
                url.split("/")[-1],
                url,
                columns,
                description=f"Loaded from {url}",
            )
        finally:
            os.unlink(tmp_path)

    except ValueError as e:
        logger.exception("SSRF validation error for URL '%s'", url)
        raise HTTPException(status_code=403, detail="URL not allowed")
    except Exception as e:
        logger.exception("Failed to load data from URL '%s'", url)
        raise HTTPException(status_code=400, detail="Failed to load data from URL")
