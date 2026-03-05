"""
Data loading helper functions.

Domain extraction, file resolution, metadata overrides, pagination,
and response builders used by the data endpoints.
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)


# ── Domain helpers ───────────────────────────────────────────────


def _make_column_dict(var, role: str) -> dict:
    """Build a single column metadata dict from an Orange3 variable."""
    return {
        "name": var.name,
        "type": "numeric" if var.is_continuous else "categorical",
        "role": role,
        "values": (
            ", ".join(var.values) if hasattr(var, "values") and var.values else ""
        ),
    }


def extract_domain_columns(domain) -> list[dict]:
    """Extract column metadata from an Orange3 domain."""
    columns = [_make_column_dict(var, "feature") for var in domain.attributes]
    if domain.class_var:
        columns.append(_make_column_dict(domain.class_var, "target"))
    columns.extend(_make_column_dict(var, "meta") for var in domain.metas)
    return columns


# ── File resolution ──────────────────────────────────────────────


async def resolve_file_path(
    path: str, tenant_id: Optional[str]
) -> tuple[str, Any, Any]:
    """Resolve *path* to an actual filesystem path.

    Returns ``(actual_path, temp_file, metadata)``.
    Callers must clean up *temp_file* if it is not ``None``.
    """
    from ..core.file_storage import get_file, get_file_metadata

    actual_path = path
    temp_file = None
    metadata = None

    if path.startswith("file:"):
        file_id = path.replace("file:", "")
        logger.info("Loading file by ID: %s, tenant: %s", file_id, tenant_id)

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
        logger.info("Created temp file: %s for %s", actual_path, metadata.filename)

    elif path.startswith("uploads/"):
        from ..core.config import get_upload_dir as _get_upload_dir

        upload_dir = _get_upload_dir()
        full_path = upload_dir / path.replace("uploads/", "")
        resolved = full_path.resolve()
        if not resolved.is_relative_to(upload_dir.resolve()):
            raise HTTPException(status_code=400, detail="Invalid file path")
        full_path = resolved
        if full_path.exists():
            actual_path = str(full_path)

    elif path.startswith("datasets/"):
        actual_path = path.replace("datasets/", "").split(".")[0]

    elif "/" in path and not path.startswith(("/", ".")):
        from ..core.config import get_datasets_cache_dir as _get_datasets_cache_dir

        datasets_cache_dir = _get_datasets_cache_dir()
        full_path = datasets_cache_dir / path
        resolved = full_path.resolve()
        if not resolved.is_relative_to(datasets_cache_dir.resolve()):
            raise HTTPException(status_code=400, detail="Invalid file path")
        full_path = resolved
        if full_path.exists():
            actual_path = str(full_path)

    return actual_path, temp_file, metadata


# ── Metadata overrides ───────────────────────────────────────────


async def apply_metadata_overrides(
    columns: list, path: str, tenant_id: Optional[str]
) -> list:
    """Read ``.metadata.json`` sidecar and apply column type/role overrides."""
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

        logger.info("Applied column metadata overrides from %s", metadata_path)
    except Exception as e:
        logger.warning("Failed to load metadata overrides: %s", e)

    return columns


# ── Pagination ───────────────────────────────────────────────────


def paginate_orange_data(
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


# ── Response builders ────────────────────────────────────────────


def file_not_found_response(path: str) -> dict:
    """Build response for missing file."""
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


def file_content_missing_response(path: str, metadata) -> dict:
    """Build response when file metadata exists but content is gone."""
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


def build_data_response(
    data: Any,
    display_name: str,
    path: str,
    columns: list[dict],
    *,
    description: str = "",
) -> dict:
    """Build a standard data-loaded response dict."""
    domain = data.domain
    return {
        "name": display_name,
        "description": description,
        "path": path,
        "instances": len(data),
        "features": len(domain.attributes),
        "missingValues": data.has_missing(),
        "classType": (
            "Classification"
            if domain.class_var and not domain.class_var.is_continuous
            else "Regression"
            if domain.class_var
            else "None"
        ),
        "classValues": (
            len(domain.class_var.values)
            if domain.class_var and hasattr(domain.class_var, "values")
            else None
        ),
        "metaAttributes": len(domain.metas),
        "columns": columns,
    }
