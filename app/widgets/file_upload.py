"""
File Upload Widget API endpoints.
"""

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["Data"])

# Check Orange3 availability
try:
    from Orange.data import Table
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False

# Upload directory configuration
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a data file from local PC.
    Supports: CSV, TSV, TAB, XLSX, PKL files
    """
    # Validate file extension
    allowed_extensions = {'.csv', '.tsv', '.tab', '.xlsx', '.xls', '.pkl', '.pickle', '.txt'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not supported. Allowed types: {', '.join(sorted(allowed_extensions))}"
        )
    
    # Generate unique filename to avoid conflicts
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = UPLOAD_DIR / unique_filename
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
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
                    "relativePath": f"uploads/{unique_filename}",
                    "name": data.name or file.filename,
                    "description": f"Uploaded file: {file.filename}",
                    "instances": len(data),
                    "features": len(data.domain.attributes),
                    "missingValues": data.has_missing(),
                    "classType": "Classification" if data.domain.class_var and not data.domain.class_var.is_continuous else "Regression" if data.domain.class_var else "None",
                    "classValues": len(data.domain.class_var.values) if data.domain.class_var and hasattr(data.domain.class_var, 'values') else None,
                    "metaAttributes": len(data.domain.metas),
                    "columns": columns
                }
            except Exception as e:
                print(f"Orange3 parsing failed: {e}")
        
        # Fallback: Return basic file info without parsing
        return {
            "success": True,
            "filename": file.filename,
            "savedPath": str(file_path),
            "relativePath": f"uploads/{unique_filename}",
            "name": file.filename,
            "description": f"Uploaded file: {file.filename}",
            "instances": 0,
            "features": 0,
            "missingValues": False,
            "classType": "Unknown",
            "classValues": None,
            "metaAttributes": 0,
            "columns": [],
            "parseError": "Orange3 not available or parsing failed"
        }
        
    except Exception as e:
        # Clean up on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@router.get("/uploaded")
async def list_uploaded_files():
    """List all uploaded files."""
    files = []
    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.iterdir():
            if f.is_file():
                files.append({
                    "filename": f.name,
                    "path": f"uploads/{f.name}",
                    "size": f.stat().st_size
                })
    return {"files": files}


@router.delete("/uploaded/{filename}")
async def delete_uploaded_file(filename: str):
    """Delete an uploaded file."""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        file_path.unlink()
        return {"message": f"File {filename} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

