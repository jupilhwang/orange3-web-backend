"""
Corpus Widget API endpoints.
Load and manage text corpus data.
Multi-tenant support for corpus files.
"""

import logging
import uuid
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

from fastapi import APIRouter, HTTPException, Header, UploadFile, File
from pydantic import BaseModel

from .text_mining_utils import (
    ORANGE_TEXT_AVAILABLE, get_text_cache, set_cache_item, get_cache_item
)
from ..core.paths import get_corpus_dir, get_tenant_corpus_dir

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/text", tags=["Text Mining - Corpus"])

# Base upload directory for corpus files (from centralized config)
BASE_CORPUS_DIR = get_corpus_dir()


# ============================================================================
# Models
# ============================================================================

class CorpusLoadRequest(BaseModel):
    """Request to load a corpus file."""
    file_path: Optional[str] = None
    data_path: Optional[str] = None  # For Data input port (from Table)
    title_variable: Optional[str] = None
    language: str = "English"
    used_text_features: Optional[List[str]] = None  # Selected text features


class CorpusResponse(BaseModel):
    """Response with corpus information."""
    success: bool
    corpus_id: Optional[str] = None
    documents: int = 0
    text_features: List[str] = []
    meta_features: List[str] = []
    title_variable: Optional[str] = None
    language: str = "English"
    preview: Optional[List[Dict]] = None
    error: Optional[str] = None


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/corpus/available")
async def get_available_corpora():
    """Get list of available sample corpora."""
    corpora = []
    
    if ORANGE_TEXT_AVAILABLE:
        try:
            # Try to find sample corpora from Orange3-Text
            import orangecontrib.text.datasets as text_datasets
            datasets_path = os.path.dirname(text_datasets.__file__)
            
            for filename in sorted(os.listdir(datasets_path)):
                if filename.endswith('.tab') or filename.endswith('.csv'):
                    corpora.append({
                        'name': filename,
                        'path': os.path.join(datasets_path, filename),
                        'source': 'orange3-text'
                    })
        except Exception as e:
            logger.warning(f"Could not list text datasets: {e}")
    
    # Add common sample corpora
    sample_corpora = [
        {'name': 'book-excerpts.tab', 'description': 'Book excerpts with genres'},
        {'name': 'deerwester.tab', 'description': 'Deerwester toy corpus for LSI'},
        {'name': 'friends-transcripts.tab', 'description': 'Friends TV show transcripts'},
        {'name': 'andersen.tab', 'description': 'Andersen fairy tales'},
    ]
    
    return {
        'corpora': corpora,
        'samples': sample_corpora,
        'orange_text_available': ORANGE_TEXT_AVAILABLE
    }


@router.post("/corpus/load")
async def load_corpus(
    request: CorpusLoadRequest,
    x_session_id: Optional[str] = Header(None)
) -> CorpusResponse:
    """Load a corpus from file or from Data input."""
    if not ORANGE_TEXT_AVAILABLE:
        return CorpusResponse(
            success=False,
            error="Orange3-Text is not installed"
        )
    
    try:
        from orangecontrib.text import Corpus
        from Orange.data import StringVariable
        
        corpus = None
        
        # Option 1: Load from Data input (Table)
        if request.data_path:
            from app.widgets.data_utils import load_data
            table = load_data(request.data_path)
            
            if table is None:
                return CorpusResponse(
                    success=False,
                    error=f"Failed to load data from: {request.data_path}"
                )
            
            # Convert Table to Corpus
            # Find text features (StringVariable in metas or attributes)
            text_vars = []
            for var in list(table.domain.metas) + list(table.domain.attributes):
                if isinstance(var, StringVariable):
                    text_vars.append(var)
            
            if not text_vars:
                return CorpusResponse(
                    success=False,
                    error="No text features found in input data"
                )
            
            # Select used text features
            used_text_features = request.used_text_features or [text_vars[0].name]
            selected_text_vars = [v for v in text_vars if v.name in used_text_features]
            
            if not selected_text_vars:
                selected_text_vars = [text_vars[0]]
            
            # Create Corpus from Table
            corpus = Corpus.from_table(table.domain, table)
            corpus.set_text_features(selected_text_vars)
        
        # Option 2: Load from file
        elif request.file_path:
            corpus = Corpus.from_file(request.file_path)
        
        else:
            return CorpusResponse(
                success=False,
                error="Either file_path or data_path must be provided"
            )
        
        if corpus is None:
            return CorpusResponse(
                success=False,
                error="Failed to load corpus"
            )
        
        # Get text features
        text_features = [var.name for var in corpus.text_features] if hasattr(corpus, 'text_features') else []
        
        # Get meta features
        meta_features = [var.name for var in corpus.domain.metas] if corpus.domain.metas else []
        
        # Generate corpus ID
        corpus_id = f"corpus_{uuid.uuid4().hex[:8]}"
        
        # Store in cache
        set_cache_item(corpus_id, {
            'corpus': corpus,
            'file_path': request.file_path,
            'data_path': request.data_path,
            'title_variable': request.title_variable,
            'language': request.language,
            'used_text_features': request.used_text_features
        })
        
        # Create preview
        preview = []
        for i in range(min(5, len(corpus))):
            row = {}
            for var in corpus.domain.metas:
                try:
                    val = corpus[i, var]
                    row[var.name] = str(val) if val is not None else ""
                except:
                    row[var.name] = ""
            preview.append(row)
        
        return CorpusResponse(
            success=True,
            corpus_id=corpus_id,
            documents=len(corpus),
            text_features=text_features,
            meta_features=meta_features,
            title_variable=request.title_variable,
            language=request.language,
            preview=preview
        )
        
    except Exception as e:
        logger.error(f"Error loading corpus: {e}")
        return CorpusResponse(
            success=False,
            error=str(e)
        )


@router.get("/corpus/{corpus_id}")
async def get_corpus_info(corpus_id: str) -> CorpusResponse:
    """Get information about a corpus."""
    cached = get_cache_item(corpus_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    corpus = cached['corpus']
    
    text_features = [var.name for var in corpus.text_features] if hasattr(corpus, 'text_features') else []
    meta_features = [var.name for var in corpus.domain.metas] if corpus.domain.metas else []
    
    return CorpusResponse(
        success=True,
        corpus_id=corpus_id,
        documents=len(corpus),
        text_features=text_features,
        meta_features=meta_features,
        title_variable=cached.get('title_variable'),
        language=cached.get('language', 'English')
    )


# ============================================================================
# Corpus File Upload (Multi-tenant)
# ============================================================================

@router.post("/corpus/upload")
async def upload_corpus_file(
    file: UploadFile = File(...),
    x_tenant_id: str = Header(default="default", alias="X-Tenant-ID")
):
    """
    Upload a corpus file (tab, csv, txt).
    Files are stored in tenant-specific directories.
    """
    allowed_extensions = {'.tab', '.csv', '.tsv', '.txt', '.xlsx'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed: {', '.join(sorted(allowed_extensions))}"
        )
    
    # Get tenant-specific directory
    tenant_dir = get_tenant_corpus_dir(x_tenant_id)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    unique_filename = f"{timestamp}_{unique_id}_{file.filename}"
    file_path = tenant_dir / unique_filename
    
    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = {
            "success": True,
            "filename": file.filename,
            "savedPath": str(file_path),
            "relativePath": f"uploads/corpus/{x_tenant_id}/{unique_filename}",
            "tenantId": x_tenant_id,
            "uploadedAt": datetime.now().isoformat()
        }
        
        # Try to load as corpus and get info
        if ORANGE_TEXT_AVAILABLE:
            try:
                from orangecontrib.text import Corpus
                corpus = Corpus.from_file(str(file_path))
                
                result.update({
                    "documents": len(corpus),
                    "textFeatures": [v.name for v in corpus.text_features] if hasattr(corpus, 'text_features') else [],
                    "metaFeatures": [v.name for v in corpus.domain.metas] if corpus.domain.metas else []
                })
            except Exception as e:
                logger.warning(f"Could not parse corpus: {e}")
                result["parseWarning"] = str(e)
        
        return result
        
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/corpus/files")
async def list_corpus_files(
    x_tenant_id: str = Header(default="default", alias="X-Tenant-ID")
):
    """List all corpus files for the current tenant."""
    tenant_dir = get_tenant_corpus_dir(x_tenant_id)
    
    files = []
    if tenant_dir.exists():
        for f in sorted(tenant_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if f.is_file():
                stat = f.stat()
                # Extract original name from unique filename
                parts = f.name.split("_", 2)
                original_name = parts[2] if len(parts) > 2 else f.name
                
                files.append({
                    "filename": f.name,
                    "originalName": original_name,
                    "path": str(f),
                    "relativePath": f"uploads/corpus/{x_tenant_id}/{f.name}",
                    "size": stat.st_size,
                    "uploadedAt": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
    
    return {
        "files": files,
        "tenantId": x_tenant_id,
        "totalCount": len(files)
    }


@router.delete("/corpus/files/{filename}")
async def delete_corpus_file(
    filename: str,
    x_tenant_id: str = Header(default="default", alias="X-Tenant-ID")
):
    """Delete a corpus file."""
    tenant_dir = get_tenant_corpus_dir(x_tenant_id)
    file_path = tenant_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Security: ensure file is in tenant directory
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
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

