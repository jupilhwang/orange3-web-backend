"""
Datasets Widget API endpoints.
Online dataset repository integration.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["Datasets"])

# Datasets server URL (Orange3's official dataset repository)
DATASETS_INDEX_URL = "https://datasets.biolab.si/"
DATASETS_CACHE_DIR = Path(__file__).parent.parent.parent / "datasets_cache"
DATASETS_CACHE_DIR.mkdir(exist_ok=True)

# In-memory cache for datasets list
_datasets_cache = None
_datasets_cache_time = None
DATASETS_CACHE_TTL = 3600  # 1 hour cache


async def fetch_datasets_list():
    """Fetch datasets list from Orange3 server with caching."""
    global _datasets_cache, _datasets_cache_time
    
    # Return cached if available and fresh
    if _datasets_cache and _datasets_cache_time:
        if time.time() - _datasets_cache_time < DATASETS_CACHE_TTL:
            logger.debug("Using cached datasets list")
            return _datasets_cache
    
    try:
        import asyncio
        from serverfiles import ServerFiles, LocalFiles
        
        logger.info(f"Fetching datasets from {DATASETS_INDEX_URL}...")
        
        # Run blocking operation in executor with timeout
        def get_server_info():
            client = ServerFiles(server=DATASETS_INDEX_URL)
            return client.allinfo()
        
        loop = asyncio.get_event_loop()
        try:
            allinfo = await asyncio.wait_for(
                loop.run_in_executor(None, get_server_info),
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Timeout fetching datasets from server, using fallback")
            return get_fallback_datasets()
        
        logger.info(f"Fetched {len(allinfo)} items from server")
        
        # Also get local cached files
        def get_local_info():
            local = LocalFiles(str(DATASETS_CACHE_DIR))
            return local.allinfo()
        
        local_info = await loop.run_in_executor(None, get_local_info)
        
        datasets = []
        for file_path, info in allinfo.items():
            prefix = '/'.join(file_path[:-1]) if len(file_path) > 1 else ''
            filename = file_path[-1]
            
            if not filename.endswith(('.tab', '.csv', '.xlsx', '.pkl')):
                continue
            
            islocal = file_path in local_info
            
            datasets.append({
                "id": '/'.join(file_path),
                "file_path": list(file_path),
                "prefix": prefix,
                "filename": filename,
                "title": info.get('title', filename),
                "description": info.get('description', ''),
                "size": info.get('size', 0),
                "instances": info.get('instances'),
                "variables": info.get('variables'),
                "target": info.get('target'),
                "tags": info.get('tags', []),
                "source": info.get('source', ''),
                "year": info.get('year'),
                "references": info.get('references', []),
                "seealso": info.get('seealso', []),
                "language": info.get('language', 'English'),
                "domain": info.get('domain'),
                "islocal": islocal,
                "version": info.get('version', '')
            })
        
        datasets.sort(key=lambda x: x['title'].lower())
        
        _datasets_cache = datasets
        _datasets_cache_time = time.time()
        
        logger.info(f"Successfully fetched {len(datasets)} datasets")
        return datasets
        
    except Exception as e:
        logger.error(f"Error fetching datasets: {e}")
        import traceback
        traceback.print_exc()
        logger.warning("Using fallback datasets")
        return get_fallback_datasets()


def get_fallback_datasets():
    """Fallback dataset list when server is unavailable."""
    return [
        {
            "id": "core/iris.tab",
            "file_path": ["core", "iris.tab"],
            "prefix": "core",
            "filename": "iris.tab",
            "title": "Iris",
            "description": "Fisher's Iris data with 150 instances and 4 features.",
            "size": 4625,
            "instances": 150,
            "variables": 5,
            "target": "categorical",
            "tags": [],
            "source": "",
            "year": None,
            "references": [],
            "seealso": [],
            "language": "English",
            "domain": None,
            "islocal": True,
            "version": ""
        },
        {
            "id": "core/titanic.tab",
            "file_path": ["core", "titanic.tab"],
            "prefix": "core",
            "filename": "titanic.tab",
            "title": "Titanic",
            "description": "Titanic survival data.",
            "size": 77400,
            "instances": 2201,
            "variables": 4,
            "target": "categorical",
            "tags": [],
            "source": "",
            "year": None,
            "references": [],
            "seealso": [],
            "language": "English",
            "domain": None,
            "islocal": True,
            "version": ""
        },
        {
            "id": "core/housing.tab",
            "file_path": ["core", "housing.tab"],
            "prefix": "core",
            "filename": "housing.tab",
            "title": "Housing",
            "description": "Boston housing dataset.",
            "size": 52500,
            "instances": 506,
            "variables": 14,
            "target": "numeric",
            "tags": ["economy"],
            "source": "",
            "year": None,
            "references": [],
            "seealso": [],
            "language": "English",
            "domain": None,
            "islocal": True,
            "version": ""
        }
    ]


@router.get("")
async def list_datasets(
    language: Optional[str] = None,
    domain: Optional[str] = None,
    search: Optional[str] = None
):
    """
    List available datasets from the Orange3 online repository.
    
    Supports filtering by:
    - language: e.g., 'English', 'Slovenian'
    - domain: e.g., 'biology', 'economy'
    - search: text search in title
    """
    datasets = await fetch_datasets_list()
    
    filtered = datasets
    
    if language:
        filtered = [d for d in filtered if d.get('language') == language]
    
    if domain:
        if domain == "(General)":
            filtered = [d for d in filtered if d.get('domain') is None]
        elif domain != "(Show all)":
            filtered = [d for d in filtered if d.get('domain') == domain]
    
    if search and len(search) >= 2:
        search_lower = search.lower()
        filtered = [d for d in filtered if search_lower in d['title'].lower()]
    
    all_languages = sorted(set(d.get('language', 'English') for d in datasets))
    all_domains = sorted(set(d.get('domain') for d in datasets if d.get('domain')))
    
    return {
        "datasets": filtered,
        "total": len(filtered),
        "languages": all_languages,
        "domains": all_domains
    }


@router.get("/{dataset_id:path}/info")
async def get_dataset_info(dataset_id: str):
    """Get detailed information about a specific dataset."""
    datasets = await fetch_datasets_list()
    
    for d in datasets:
        if d['id'] == dataset_id:
            return d
    
    raise HTTPException(status_code=404, detail="Dataset not found")


@router.post("/{dataset_id:path}/download")
async def download_dataset(dataset_id: str):
    """
    Download a dataset from the online repository to local cache.
    Returns the local file path after download.
    """
    datasets = await fetch_datasets_list()
    
    dataset = None
    for d in datasets:
        if d['id'] == dataset_id:
            dataset = d
            break
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        from serverfiles import LocalFiles, ServerFiles
        
        file_path = tuple(dataset['file_path'])
        
        localfiles = LocalFiles(
            str(DATASETS_CACHE_DIR),
            serverfiles=ServerFiles(server=DATASETS_INDEX_URL)
        )
        local_path = localfiles.localpath_download(*file_path)
        
        return {
            "success": True,
            "local_path": local_path,
            "dataset_id": dataset_id,
            "title": dataset['title']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.post("/{dataset_id:path}/load")
async def load_dataset(dataset_id: str):
    """
    Load a dataset and return its data information.
    Downloads the file first if not cached locally.
    """
    datasets = await fetch_datasets_list()
    
    dataset = None
    for d in datasets:
        if d['id'] == dataset_id:
            dataset = d
            break
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        from serverfiles import LocalFiles, ServerFiles
        from Orange.data import Table
        import math
        
        file_path = tuple(dataset['file_path'])
        
        localfiles = LocalFiles(
            str(DATASETS_CACHE_DIR),
            serverfiles=ServerFiles(server=DATASETS_INDEX_URL)
        )
        local_path = localfiles.localpath_download(*file_path)
        
        data = Table(local_path)
        
        columns = []
        
        for var in data.domain.attributes:
            columns.append({
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical",
                "role": "feature",
                "values": ", ".join(var.values) if hasattr(var, 'values') and var.values else ""
            })
        
        if data.domain.class_var:
            var = data.domain.class_var
            columns.append({
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical",
                "role": "target",
                "values": ", ".join(var.values) if hasattr(var, 'values') and var.values else ""
            })
        
        for var in data.domain.metas:
            columns.append({
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical",
                "role": "meta",
                "values": ""
            })
        
        def get_cell_value(var, val):
            """Convert Orange3 cell value to proper format."""
            if val is None:
                return None
            if isinstance(val, float) and math.isnan(val):
                return None
            
            if var.is_continuous:
                return float(val) if not math.isnan(val) else None
            else:
                str_repr = var.str_val(val)
                if str_repr == '?' or str_repr == '':
                    return None
                return str_repr
        
        data_rows = []
        for row in data:
            row_values = []
            for var in data.domain.attributes:
                val = row[var]
                row_values.append(get_cell_value(var, val))
            if data.domain.class_var:
                var = data.domain.class_var
                val = row[var]
                row_values.append(get_cell_value(var, val))
            for var in data.domain.metas:
                val = row[var]
                row_values.append(get_cell_value(var, val))
            data_rows.append(row_values)
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "local_path": local_path,
            "name": data.name or dataset['title'],
            "description": dataset.get('description', ''),
            "instances": len(data),
            "features": len(data.domain.attributes),
            "missingValues": data.has_missing(),
            "classType": "Classification" if data.domain.class_var and not data.domain.class_var.is_continuous else "Regression" if data.domain.class_var else "None",
            "classValues": len(data.domain.class_var.values) if data.domain.class_var and hasattr(data.domain.class_var, 'values') else None,
            "metaAttributes": len(data.domain.metas),
            "columns": columns,
            "data": data_rows
        }
        
    except ImportError:
        raise HTTPException(status_code=501, detail="Orange3 not available")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")


def sizeformat(size):
    """Format file size in human-readable format."""
    for unit in ['bytes', 'KB', 'MB', 'GB']:
        if abs(size) < 1024:
            return f"{size:.1f} {unit}" if unit != 'bytes' else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


