"""
Data Info API - provides information about loaded data.
"""

import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["Data"])


class DataInfoRequest(BaseModel):
    """Request model for data info."""
    data_path: str


class ColumnInfo(BaseModel):
    """Column information."""
    name: str
    type: str  # 'numeric', 'categorical', 'string', 'datetime'
    role: str  # 'feature', 'target', 'meta'
    is_target: bool = False


class DataInfoResponse(BaseModel):
    """Response model for data info."""
    instances: int
    features: int
    n_rows: int = 0  # Alias for instances
    n_cols: int = 0  # Total columns count
    columns: List[ColumnInfo]
    target: Optional[str] = None
    metas: int = 0


@router.get("/info")
async def get_data_info_get(data_path: str) -> DataInfoResponse:
    """
    Get information about a dataset (GET method).
    Supports datasets, uploads, and kmeans results.
    """
    return await _get_data_info(data_path)


@router.post("/info")
async def get_data_info_post(request: DataInfoRequest) -> DataInfoResponse:
    """
    Get information about a dataset (POST method).
    Supports datasets, uploads, and kmeans results.
    """
    return await _get_data_info(request.data_path)


async def _get_data_info(data_path: str) -> DataInfoResponse:
    """
    Internal function to get data info.
    """
    try:
        from .data_utils import load_data
        
        data = load_data(data_path)
        
        if data is None:
            raise HTTPException(status_code=404, detail=f"Data not found: {data_path}")
        
        # Build column info
        columns = []
        
        # Attributes (features)
        for var in data.domain.attributes:
            col_type = 'numeric' if var.is_continuous else 'categorical'
            columns.append(ColumnInfo(
                name=var.name,
                type=col_type,
                role='feature',
                is_target=False
            ))
        
        # Class variable (target)
        target_name = None
        if data.domain.class_var:
            var = data.domain.class_var
            col_type = 'numeric' if var.is_continuous else 'categorical'
            columns.append(ColumnInfo(
                name=var.name,
                type=col_type,
                role='target',
                is_target=True
            ))
            target_name = var.name
        
        # Metas
        for var in data.domain.metas:
            if var.is_continuous:
                col_type = 'numeric'
            elif var.is_discrete:
                col_type = 'categorical'
            else:
                col_type = 'string'
            columns.append(ColumnInfo(
                name=var.name,
                type=col_type,
                role='meta',
                is_target=False
            ))
        
        n_rows = len(data)
        n_features = len(data.domain.attributes)
        n_metas = len(data.domain.metas)
        n_cols = n_features + (1 if data.domain.class_var else 0) + n_metas
        
        return DataInfoResponse(
            instances=n_rows,
            features=n_features,
            n_rows=n_rows,
            n_cols=n_cols,
            columns=columns,
            target=target_name,
            metas=n_metas
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data info error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

