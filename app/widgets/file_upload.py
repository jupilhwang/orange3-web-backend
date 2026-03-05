"""
File Upload Widget API endpoints.
Multi-tenant support: files are stored in tenant-specific directories or database.

Storage Backend:
    STORAGE_TYPE='filesystem' (default): Files stored on local filesystem
    STORAGE_TYPE='database': Files stored in database (for multi-server)
"""

import logging
import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Header

from ..core.file_storage import (
    save_file,
    get_file,
    get_file_metadata,
    delete_file,
    list_files,
    STORAGE_TYPE,
    StoredFile,
)
from ..core.config import get_tenant_upload_dir
from ..core.rate_limit import limiter, LIMIT_UPLOAD

# Pickle deserialization can execute arbitrary code (RCE risk).
# Disabled by default; set ALLOW_PICKLE_UPLOAD=true to re-enable in trusted environments.
ALLOW_PICKLE: bool = os.environ.get("ALLOW_PICKLE_UPLOAD", "false").lower() == "true"

from pydantic import BaseModel
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["Data"])


# --- Models for Phase 4: Backend Persistence ---
class ColumnUpdate(BaseModel):
    name: str
    type: str
    role: str
    values: Optional[str] = ""


class ColumnMetadataUpdate(BaseModel):
    path: str  # File path or file ID
    columns: List[ColumnUpdate]


@router.post("/columns/save")
async def save_column_metadata(
    data: ColumnMetadataUpdate,
    x_tenant_id: str = Header(default="default", alias="X-Tenant-ID"),
):
    """
    Save column metadata overrides for a file.
    This enables persistent column names, types, and roles.
    """
    import json

    file_id = data.path
    if file_id.startswith("file:"):
        file_id = file_id.replace("file:", "")

    # Define metadata filename
    # For simplicity, we store it in the tenant's upload directory as {file_id}.metadata.json
    upload_dir = get_tenant_upload_dir(x_tenant_id)
    metadata_path = upload_dir / f"{file_id}.metadata.json"

    # Ensure parent directory exists (file_id might contain subdirectories like 'datasets/')
    if not metadata_path.parent.exists():
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        metadata_content = {
            "path": data.path,
            "columns": [col.model_dump() for col in data.columns],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_content, f, indent=4, ensure_ascii=False)

        logger.info(f"Saved column metadata for {file_id} to {metadata_path}")
        return {"success": True, "path": str(metadata_path)}
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to save metadata: {str(e)}"
        )


from app.core.orange_compat import ORANGE_AVAILABLE, Table


def _parse_with_orange3(file_path: str, filename: str) -> dict:
    """Parse data file with Orange3 and return metadata."""
    try:
        from Orange.data import Table

        data = Table(file_path)

        # Get column info
        columns = []

        # Features
        for var in data.domain.attributes:
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

        # Target
        if data.domain.class_var:
            var = data.domain.class_var
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

        # Meta
        for var in data.domain.metas:
            columns.append(
                {
                    "name": var.name,
                    "type": "numeric" if var.is_continuous else "categorical",
                    "role": "meta",
                    "values": "",
                }
            )

        # 항상 원본 파일명 사용 (임시파일명이 data.name에 들어갈 수 있음)
        display_name = Path(filename).stem  # 확장자 제거한 파일명
        return {
            "name": display_name,
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
    except Exception as e:
        logger.warning(f"Orange3 parsing failed: {e}")
        return None


from ..core.concurrency import run_in_threadpool


@router.post("/upload")
@limiter.limit(LIMIT_UPLOAD)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    x_tenant_id: str = Header(default="default", alias="X-Tenant-ID"),
):
    """
    Upload a data file from local PC.
    Supports: CSV, TSV, TAB, XLSX, PKL files

    Storage location depends on STORAGE_TYPE:
    - filesystem: uploads/{tenant_id}/{uuid}_{filename}
    - database: stored in file_storage table
    """
    # Validate file extension (Orange3 compatible formats)
    allowed_extensions = {
        ".csv",  # Comma-separated values
        ".tsv",  # Tab-separated values
        ".tab",  # Tab-separated values (Orange3 native)
        ".xlsx",  # Excel 2007+
        ".xls",  # Excel 97-2003
        ".txt",  # Text files
        ".dat",  # Data files
        ".data",  # Data files
        ".arff",  # Weka ARFF format
        ".basket",  # Basket format
        ".sparse",  # Sparse format
    }
    # Pickle formats are disabled by default (RCE risk via arbitrary code execution).
    # Enable only in trusted, controlled environments via ALLOW_PICKLE_UPLOAD=true.
    if ALLOW_PICKLE:
        allowed_extensions.update({".pkl", ".pickle"})
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed types: {', '.join(sorted(allowed_extensions))}",
        )

    try:
        # Read file content
        content = await file.read()

        # Determine content type
        content_type_map = {
            ".csv": "text/csv",
            ".tsv": "text/tab-separated-values",
            ".tab": "text/tab-separated-values",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls": "application/vnd.ms-excel",
            ".pkl": "application/octet-stream",
            ".pickle": "application/octet-stream",
            ".txt": "text/plain",
            ".dat": "text/plain",
            ".data": "text/plain",
            ".arff": "text/plain",
            ".basket": "text/plain",
            ".sparse": "text/plain",
        }
        content_type = content_type_map.get(file_ext, "application/octet-stream")

        # Save file using storage backend
        stored_file = await save_file(
            tenant_id=x_tenant_id,
            filename=file.filename,
            content=content,
            content_type=content_type,
            category="upload",
            original_filename=file.filename,
        )

        # Parse with Orange3 if available
        orange_metadata = None
        if ORANGE_AVAILABLE:
            if STORAGE_TYPE == "filesystem" and stored_file.file_path:
                # Filesystem mode: use stored file path directly
                # Run CPU-bound parsing in thread pool
                orange_metadata = await run_in_threadpool(
                    _parse_with_orange3, stored_file.file_path, file.filename
                )
            else:
                # Database mode: create temp file for Orange3 parsing
                with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                try:
                    # Run CPU-bound parsing in thread pool
                    orange_metadata = await run_in_threadpool(
                        _parse_with_orange3, tmp_path, file.filename
                    )
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

        # Build response
        response = {
            "success": True,
            "fileId": stored_file.id,
            "filename": file.filename,
            "storedFilename": stored_file.filename,
            "tenantId": x_tenant_id,
            "storageType": STORAGE_TYPE,
            "fileSize": stored_file.file_size,
            "originalSize": stored_file.original_size,
            "isCompressed": stored_file.is_compressed,
            "contentType": content_type,
            "uploadedAt": datetime.now().isoformat(),
        }

        # Add file path for filesystem storage
        if stored_file.file_path:
            response["savedPath"] = stored_file.file_path
            response["relativePath"] = f"uploads/{x_tenant_id}/{stored_file.filename}"

        # Add Orange3 parsing results
        if orange_metadata:
            response.update(
                {
                    "name": orange_metadata["name"],
                    "description": f"Uploaded file: {file.filename}",
                    "instances": orange_metadata["instances"],
                    "features": orange_metadata["features"],
                    "missingValues": orange_metadata["missingValues"],
                    "classType": orange_metadata["classType"],
                    "classValues": orange_metadata["classValues"],
                    "metaAttributes": orange_metadata["metaAttributes"],
                    "columns": orange_metadata["columns"],
                }
            )
        else:
            response.update(
                {
                    "name": file.filename,
                    "description": f"Uploaded file: {file.filename}",
                    "instances": 0,
                    "features": 0,
                    "missingValues": False,
                    "classType": "Unknown",
                    "classValues": None,
                    "metaAttributes": 0,
                    "columns": [],
                    "parseError": "Orange3 not available or parsing failed",
                }
            )

        return response

    except ValueError as e:
        # File size limit exceeded
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to upload file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@router.get("/uploaded")
async def list_uploaded_files(
    x_tenant_id: str = Header(default="default", alias="X-Tenant-ID"),
):
    """List all uploaded files for the current tenant."""
    files = await list_files(x_tenant_id, category="upload")

    file_list = []
    for f in sorted(files, key=lambda x: x.created_at or datetime.min, reverse=True):
        file_info = {
            "fileId": f.id,
            "filename": f.filename,
            "originalName": f.original_filename,
            "size": f.file_size,
            "contentType": f.content_type,
            "uploadedAt": f.created_at.isoformat() if f.created_at else None,
            "tenantId": x_tenant_id,
            "storageType": STORAGE_TYPE,
        }
        if f.file_path:
            file_info["path"] = f"uploads/{x_tenant_id}/{f.filename}"
        file_list.append(file_info)

    return {
        "files": file_list,
        "tenantId": x_tenant_id,
        "totalCount": len(file_list),
        "storageType": STORAGE_TYPE,
    }


@router.get("/uploaded/{file_id}")
async def get_uploaded_file_info(
    file_id: str, x_tenant_id: str = Header(default="default", alias="X-Tenant-ID")
):
    """Get metadata for a specific uploaded file."""
    metadata = await get_file_metadata(file_id, x_tenant_id)

    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    response = {
        "fileId": metadata.id,
        "filename": metadata.filename,
        "originalName": metadata.original_filename,
        "size": metadata.file_size,
        "contentType": metadata.content_type,
        "checksum": metadata.checksum,
        "uploadedAt": metadata.created_at.isoformat() if metadata.created_at else None,
        "tenantId": x_tenant_id,
        "storageType": STORAGE_TYPE,
    }

    if metadata.file_path:
        response["path"] = metadata.file_path

    return response


@router.get("/uploaded/{file_id}/download")
@limiter.limit(LIMIT_UPLOAD)
async def download_uploaded_file(
    request: Request,
    file_id: str,
    x_tenant_id: str = Header(default="default", alias="X-Tenant-ID"),
):
    """Download an uploaded file."""
    from fastapi.responses import Response

    metadata = await get_file_metadata(file_id, x_tenant_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    content = await get_file(file_id, x_tenant_id)
    if not content:
        raise HTTPException(status_code=404, detail="File content not found")

    return Response(
        content=content,
        media_type=metadata.content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{metadata.original_filename}"'
        },
    )


@router.delete("/uploaded/{file_id}")
async def delete_uploaded_file(
    file_id: str, x_tenant_id: str = Header(default="default", alias="X-Tenant-ID")
):
    """Delete an uploaded file for the current tenant."""
    deleted = await delete_file(file_id, x_tenant_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="File not found")

    return {
        "success": True,
        "message": f"File {file_id} deleted",
        "tenantId": x_tenant_id,
    }


@router.get("/uploaded/stats")
async def get_upload_stats(
    x_tenant_id: str = Header(default="default", alias="X-Tenant-ID"),
):
    """Get upload statistics for the current tenant."""
    files = await list_files(x_tenant_id, category="upload")

    total_size = sum(f.file_size for f in files)

    return {
        "tenantId": x_tenant_id,
        "totalFiles": len(files),
        "totalSize": total_size,
        "totalSizeHuman": _format_size(total_size),
        "storageType": STORAGE_TYPE,
    }


def _format_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
