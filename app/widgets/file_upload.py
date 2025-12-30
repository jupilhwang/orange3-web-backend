"""
File Upload Widget API endpoints.
Multi-tenant support: files are stored in tenant-specific directories.
"""

import logging
import shutil
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Header, Depends

from ..core.tenant import get_current_tenant
from ..models import Tenant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["Data"])

# Check Orange3 availability
try:
    from Orange.data import Table
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False

# Base upload directory configuration
BASE_UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"
BASE_UPLOAD_DIR.mkdir(exist_ok=True)


def get_tenant_upload_dir(tenant_id: str) -> Path:
    """Get the upload directory for a specific tenant."""
    tenant_dir = BASE_UPLOAD_DIR / tenant_id
    tenant_dir.mkdir(parents=True, exist_ok=True)
    return tenant_dir


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    x_tenant_id: str = Header(default="default", alias="X-Tenant-ID")
):
    """
    Upload a data file from local PC.
    Supports: CSV, TSV, TAB, XLSX, PKL files
    
    Files are stored in tenant-specific directories:
    uploads/{tenant_id}/{uuid}_{filename}
    """
    # Validate file extension
    allowed_extensions = {'.csv', '.tsv', '.tab', '.xlsx', '.xls', '.pkl', '.pickle', '.txt'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not supported. Allowed types: {', '.join(sorted(allowed_extensions))}"
        )
    
    # Get tenant-specific upload directory
    tenant_dir = get_tenant_upload_dir(x_tenant_id)
    
    # Generate unique filename to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    unique_filename = f"{timestamp}_{unique_id}_{file.filename}"
    file_path = tenant_dir / unique_filename
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Relative path for client reference
        relative_path = f"uploads/{x_tenant_id}/{unique_filename}"
        
        # Try to load and parse the data with Orange3
        if ORANGE_AVAILABLE:
            try:
                from Orange.data import Table
                data = Table(str(file_path))
                
                # Get column info
                columns = []
                
                # Features
                for var in data.domain.attributes:
                    columns.append({
                        "name": var.name,
                        "type": "numeric" if var.is_continuous else "categorical",
                        "role": "feature",
                        "values": ", ".join(var.values) if hasattr(var, 'values') and var.values else ""
                    })
                
                # Target
                if data.domain.class_var:
                    var = data.domain.class_var
                    columns.append({
                        "name": var.name,
                        "type": "numeric" if var.is_continuous else "categorical",
                        "role": "target",
                        "values": ", ".join(var.values) if hasattr(var, 'values') and var.values else ""
                    })
                
                # Meta
                for var in data.domain.metas:
                    columns.append({
                        "name": var.name,
                        "type": "numeric" if var.is_continuous else "categorical",
                        "role": "meta",
                        "values": ""
                    })
                
                return {
                    "success": True,
                    "filename": file.filename,
                    "savedPath": str(file_path),
                    "relativePath": relative_path,
                    "tenantId": x_tenant_id,
                    "name": data.name or file.filename,
                    "description": f"Uploaded file: {file.filename}",
                    "instances": len(data),
                    "features": len(data.domain.attributes),
                    "missingValues": data.has_missing(),
                    "classType": "Classification" if data.domain.class_var and not data.domain.class_var.is_continuous else "Regression" if data.domain.class_var else "None",
                    "classValues": len(data.domain.class_var.values) if data.domain.class_var and hasattr(data.domain.class_var, 'values') else None,
                    "metaAttributes": len(data.domain.metas),
                    "columns": columns,
                    "uploadedAt": datetime.now().isoformat()
                }
            except Exception as e:
                logger.warning(f"Orange3 parsing failed: {e}")
        
        # Fallback: Return basic file info without parsing
        return {
            "success": True,
            "filename": file.filename,
            "savedPath": str(file_path),
            "relativePath": relative_path,
            "tenantId": x_tenant_id,
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
            "uploadedAt": datetime.now().isoformat()
        }
        
    except Exception as e:
        # Clean up on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@router.get("/uploaded")
async def list_uploaded_files(
    x_tenant_id: str = Header(default="default", alias="X-Tenant-ID")
):
    """List all uploaded files for the current tenant."""
    tenant_dir = get_tenant_upload_dir(x_tenant_id)
    
    files = []
    if tenant_dir.exists():
        for f in sorted(tenant_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if f.is_file():
                stat = f.stat()
                files.append({
                    "filename": f.name,
                    "originalName": "_".join(f.name.split("_")[2:]) if len(f.name.split("_")) > 2 else f.name,
                    "path": f"uploads/{x_tenant_id}/{f.name}",
                    "size": stat.st_size,
                    "uploadedAt": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "tenantId": x_tenant_id
                })
    
    return {
        "files": files,
        "tenantId": x_tenant_id,
        "totalCount": len(files)
    }


@router.delete("/uploaded/{filename}")
async def delete_uploaded_file(
    filename: str,
    x_tenant_id: str = Header(default="default", alias="X-Tenant-ID")
):
    """Delete an uploaded file for the current tenant."""
    tenant_dir = get_tenant_upload_dir(x_tenant_id)
    file_path = tenant_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Security check: ensure file is within tenant directory
    try:
        file_path.resolve().relative_to(tenant_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        file_path.unlink()
        return {
            "success": True,
            "message": f"File {filename} deleted",
            "tenantId": x_tenant_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


@router.get("/uploaded/stats")
async def get_upload_stats(
    x_tenant_id: str = Header(default="default", alias="X-Tenant-ID")
):
    """Get upload statistics for the current tenant."""
    tenant_dir = get_tenant_upload_dir(x_tenant_id)
    
    total_files = 0
    total_size = 0
    
    if tenant_dir.exists():
        for f in tenant_dir.iterdir():
            if f.is_file():
                total_files += 1
                total_size += f.stat().st_size
    
    return {
        "tenantId": x_tenant_id,
        "totalFiles": total_files,
        "totalSize": total_size,
        "totalSizeHuman": _format_size(total_size)
    }


def _format_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
