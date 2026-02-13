"""
Corpus Widget API endpoints.
Load and manage text corpus data.
Multi-tenant support for corpus files.

Storage Backend:
    STORAGE_TYPE='filesystem' (default): Files stored on local filesystem
    STORAGE_TYPE='database': Files stored in database (for multi-server)
"""

import logging
import uuid
import os
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

from fastapi import APIRouter, HTTPException, Header, UploadFile, File
from pydantic import BaseModel

from app.core.text_mining_utils import (
    ORANGE_TEXT_AVAILABLE,
    get_text_cache,
    set_cache_item,
    get_cache_item,
)
from ..core.file_storage import (
    save_file,
    get_file,
    get_file_metadata,
    delete_file,
    list_files,
    STORAGE_TYPE,
    StoredFile,
    validate_and_sanitize_path,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/text", tags=["Text Mining - Corpus"])


# ============================================================================
# Models
# ============================================================================


class CorpusLoadRequest(BaseModel):
    """Request to load a corpus file."""

    file_path: Optional[str] = None
    file_id: Optional[str] = None  # For database storage
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
                if filename.endswith(".tab") or filename.endswith(".csv"):
                    corpora.append(
                        {
                            "name": filename,
                            "path": os.path.join(datasets_path, filename),
                            "source": "orange3-text",
                        }
                    )
        except Exception as e:
            logger.warning(f"Could not list text datasets: {e}")

    # Add common sample corpora
    sample_corpora = [
        {"name": "book-excerpts.tab", "description": "Book excerpts with genres"},
        {"name": "deerwester.tab", "description": "Deerwester toy corpus for LSI"},
        {
            "name": "friends-transcripts.tab",
            "description": "Friends TV show transcripts",
        },
        {"name": "andersen.tab", "description": "Andersen fairy tales"},
    ]

    return {
        "corpora": corpora,
        "samples": sample_corpora,
        "orange_text_available": ORANGE_TEXT_AVAILABLE,
    }


@router.post("/corpus/load")
async def load_corpus(
    request: CorpusLoadRequest,
    x_session_id: Optional[str] = Header(None),
    x_tenant_id: str = Header(default="default", alias="X-Tenant-ID"),
) -> CorpusResponse:
    """Load a corpus from file, file_id, or from Data input."""
    if not ORANGE_TEXT_AVAILABLE:
        return CorpusResponse(success=False, error="Orange3-Text is not installed")

    try:
        from orangecontrib.text import Corpus
        from Orange.data import StringVariable

        corpus = None

        # Option 1: Load from Data input (Table)
        if request.data_path:
            from app.core.data_utils import load_data

            table = load_data(request.data_path)

            if table is None:
                return CorpusResponse(
                    success=False,
                    error=f"Failed to load data from: {request.data_path}",
                )

            # Convert Table to Corpus
            # Find text features (StringVariable in metas or attributes)
            text_vars = []
            for var in list(table.domain.metas) + list(table.domain.attributes):
                if isinstance(var, StringVariable):
                    text_vars.append(var)

            if not text_vars:
                return CorpusResponse(
                    success=False, error="No text features found in input data"
                )

            # Select used text features
            used_text_features = request.used_text_features or [text_vars[0].name]
            selected_text_vars = [v for v in text_vars if v.name in used_text_features]

            if not selected_text_vars:
                selected_text_vars = [text_vars[0]]

            # Create Corpus from Table
            corpus = Corpus.from_table(table.domain, table)
            corpus.set_text_features(selected_text_vars)

        # Option 2: Load from file_id (database storage)
        elif request.file_id:
            content = await get_file(request.file_id, x_tenant_id)
            if not content:
                return CorpusResponse(
                    success=False, error=f"File not found: {request.file_id}"
                )

            # Get metadata for file extension
            metadata = await get_file_metadata(request.file_id, x_tenant_id)
            file_ext = (
                Path(metadata.original_filename).suffix.lower() if metadata else ".tab"
            )

            # Write to temp file for Orange3 to parse
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                corpus = Corpus.from_file(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        # Option 3: Load from file path (restricted to tenant corpus directory)
        elif request.file_path:
            from app.core.config import get_tenant_corpus_dir

            corpus_dir = get_tenant_corpus_dir(x_tenant_id)
            try:
                safe_path = validate_and_sanitize_path(
                    request.file_path, corpus_dir, "corpus file path"
                )
                if not safe_path.exists():
                    return CorpusResponse(
                        success=False, error=f"File not found: {request.file_path}"
                    )
                corpus = Corpus.from_file(str(safe_path))
            except ValueError as e:
                return CorpusResponse(success=False, error=str(e))

        else:
            return CorpusResponse(
                success=False,
                error="Either file_path, file_id, or data_path must be provided",
            )

        if corpus is None:
            return CorpusResponse(success=False, error="Failed to load corpus")

        # Get text features
        text_features = (
            [var.name for var in corpus.text_features]
            if hasattr(corpus, "text_features")
            else []
        )

        # Get meta features
        meta_features = (
            [var.name for var in corpus.domain.metas] if corpus.domain.metas else []
        )

        # Generate corpus ID
        corpus_id = f"corpus_{uuid.uuid4().hex[:8]}"

        # Store in cache
        set_cache_item(
            corpus_id,
            {
                "corpus": corpus,
                "file_path": request.file_path,
                "file_id": request.file_id,
                "data_path": request.data_path,
                "title_variable": request.title_variable,
                "language": request.language,
                "used_text_features": request.used_text_features,
            },
        )

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
            preview=preview,
        )

    except Exception as e:
        logger.error(f"Error loading corpus: {e}")
        return CorpusResponse(success=False, error=str(e))


@router.get("/corpus/info/{corpus_id}")
async def get_corpus_info(corpus_id: str) -> CorpusResponse:
    """Get information about a corpus."""
    cached = get_cache_item(corpus_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Corpus not found")

    corpus = cached["corpus"]

    text_features = (
        [var.name for var in corpus.text_features]
        if hasattr(corpus, "text_features")
        else []
    )
    meta_features = (
        [var.name for var in corpus.domain.metas] if corpus.domain.metas else []
    )

    return CorpusResponse(
        success=True,
        corpus_id=corpus_id,
        documents=len(corpus),
        text_features=text_features,
        meta_features=meta_features,
        title_variable=cached.get("title_variable"),
        language=cached.get("language", "English"),
    )


# ============================================================================
# Corpus File Upload (Multi-tenant with Hybrid Storage)
# ============================================================================


@router.post("/corpus/upload")
async def upload_corpus_file(
    file: UploadFile = File(...),
    x_tenant_id: str = Header(default="default", alias="X-Tenant-ID"),
):
    """
    Upload a corpus file (tab, csv, txt).

    Storage location depends on STORAGE_TYPE:
    - filesystem: uploads/corpus/{tenant_id}/{uuid}_{filename}
    - database: stored in file_storage table
    """
    allowed_extensions = {".tab", ".csv", ".tsv", ".txt", ".xlsx"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed: {', '.join(sorted(allowed_extensions))}",
        )

    try:
        # Read file content
        content = await file.read()

        # Determine content type
        content_type_map = {
            ".tab": "text/tab-separated-values",
            ".csv": "text/csv",
            ".tsv": "text/tab-separated-values",
            ".txt": "text/plain",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }
        content_type = content_type_map.get(file_ext, "application/octet-stream")

        # Save file using storage backend
        stored_file = await save_file(
            tenant_id=x_tenant_id,
            filename=file.filename,
            content=content,
            content_type=content_type,
            category="corpus",
            original_filename=file.filename,
        )

        result = {
            "success": True,
            "fileId": stored_file.id,
            "filename": file.filename,
            "storedFilename": stored_file.filename,
            "tenantId": x_tenant_id,
            "storageType": STORAGE_TYPE,
            "fileSize": stored_file.file_size,
            "originalSize": stored_file.original_size,
            "isCompressed": stored_file.is_compressed,
            "uploadedAt": datetime.now().isoformat(),
        }

        # Add file path for filesystem storage
        if stored_file.file_path:
            result["savedPath"] = stored_file.file_path
            result["relativePath"] = (
                f"uploads/corpus/{x_tenant_id}/{stored_file.filename}"
            )

        # Try to load as corpus and get info
        if ORANGE_TEXT_AVAILABLE:
            try:
                from orangecontrib.text import Corpus

                if STORAGE_TYPE == "filesystem" and stored_file.file_path:
                    corpus = Corpus.from_file(stored_file.file_path)
                else:
                    # Database mode: create temp file
                    with tempfile.NamedTemporaryFile(
                        suffix=file_ext, delete=False
                    ) as tmp:
                        tmp.write(content)
                        tmp_path = tmp.name
                    try:
                        corpus = Corpus.from_file(tmp_path)
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)

                result.update(
                    {
                        "documents": len(corpus),
                        "textFeatures": [v.name for v in corpus.text_features]
                        if hasattr(corpus, "text_features")
                        else [],
                        "metaFeatures": [v.name for v in corpus.domain.metas]
                        if corpus.domain.metas
                        else [],
                    }
                )
            except Exception as e:
                logger.warning(f"Could not parse corpus: {e}")
                result["parseWarning"] = str(e)

        return result

    except ValueError as e:
        # File size limit exceeded
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/corpus/uploaded")
async def list_corpus_files(
    x_tenant_id: str = Header(default="default", alias="X-Tenant-ID"),
):
    """List all corpus files for the current tenant."""
    files = await list_files(x_tenant_id, category="corpus")

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
            file_info["path"] = f.file_path
            file_info["relativePath"] = f"uploads/corpus/{x_tenant_id}/{f.filename}"
        file_list.append(file_info)

    return {
        "files": file_list,
        "tenantId": x_tenant_id,
        "totalCount": len(file_list),
        "storageType": STORAGE_TYPE,
    }


@router.get("/corpus/uploaded/{file_id}")
async def get_corpus_file_info(
    file_id: str, x_tenant_id: str = Header(default="default", alias="X-Tenant-ID")
):
    """Get metadata for a specific corpus file."""
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


@router.get("/corpus/uploaded/{file_id}/download")
async def download_corpus_file(
    file_id: str, x_tenant_id: str = Header(default="default", alias="X-Tenant-ID")
):
    """Download a corpus file."""
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


@router.delete("/corpus/uploaded/{file_id}")
async def delete_corpus_file(
    file_id: str, x_tenant_id: str = Header(default="default", alias="X-Tenant-ID")
):
    """Delete a corpus file."""
    deleted = await delete_file(file_id, x_tenant_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="File not found")

    return {
        "success": True,
        "message": f"File {file_id} deleted",
        "tenantId": x_tenant_id,
    }
