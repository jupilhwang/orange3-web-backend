"""
Select Columns Widget API endpoints.
"""

import logging
import math
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["Data"])

# Check Orange3 availability
try:
    from Orange.data import Table, Domain
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False

# Upload directory
UPLOAD_DIR = Path("./uploads")


class SelectColumnsRequest(BaseModel):
    """Request model for select columns."""
    data_path: Optional[str] = None
    features: List[str] = []
    target: List[str] = []
    metas: List[str] = []
    ignored: List[str] = []


@router.post("/select-columns")
async def select_columns(request: SelectColumnsRequest):
    """
    Select and reorder columns in a dataset.
    
    This endpoint allows you to:
    - Assign columns as features, target, metas, or ignored
    - Reorder columns within each category
    - Create a new domain with the specified column assignments
    """
    if not ORANGE_AVAILABLE:
        return {
            "success": True,
            "features": request.features,
            "target": request.target,
            "metas": request.metas,
            "ignored": request.ignored,
            "instances": 0,
            "variables": len(request.features) + len(request.target) + len(request.metas),
            "note": "Fallback response (Orange3 not available)"
        }
    
    try:
        from Orange.data import Table, Domain
        
        data_path = request.data_path
        if not data_path:
            return {
                "success": True,
                "features": request.features,
                "target": request.target,
                "metas": request.metas,
                "ignored": request.ignored,
                "instances": 0,
                "variables": len(request.features) + len(request.target) + len(request.metas)
            }
        
        if data_path.startswith("uploads/"):
            data_path = str(UPLOAD_DIR / data_path.replace("uploads/", ""))
        elif data_path.startswith("datasets/"):
            dataset_name = data_path.replace("datasets/", "").split(".")[0]
            data_path = dataset_name
        
        original_data = Table(data_path)
        
        all_vars = {}
        for var in original_data.domain.attributes:
            all_vars[var.name] = var
        if original_data.domain.class_var:
            all_vars[original_data.domain.class_var.name] = original_data.domain.class_var
        for var in original_data.domain.metas:
            all_vars[var.name] = var
        
        new_features = [all_vars[name] for name in request.features if name in all_vars]
        new_target = all_vars.get(request.target[0]) if request.target else None
        new_metas = [all_vars[name] for name in request.metas if name in all_vars]
        
        new_domain = Domain(new_features, new_target, new_metas)
        new_data = original_data.transform(new_domain)
        
        def get_cell_value(var, val):
            if var.is_continuous:
                return float(val) if not math.isnan(val) else None
            else:
                return var.str_val(val)
        
        data_rows = []
        for row in new_data:
            row_values = []
            for var in new_domain.attributes:
                val = row[var]
                row_values.append(get_cell_value(var, val))
            if new_domain.class_var:
                val = row[new_domain.class_var]
                row_values.append(get_cell_value(new_domain.class_var, val))
            for var in new_domain.metas:
                val = row[var]
                if var.is_string:
                    row_values.append(str(val) if val else None)
                else:
                    row_values.append(get_cell_value(var, val))
            data_rows.append(row_values)
        
        columns = []
        for var in new_domain.attributes:
            columns.append({
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical",
                "role": "feature"
            })
        if new_domain.class_var:
            columns.append({
                "name": new_domain.class_var.name,
                "type": "numeric" if new_domain.class_var.is_continuous else "categorical",
                "role": "target"
            })
        for var in new_domain.metas:
            columns.append({
                "name": var.name,
                "type": "string" if var.is_string else ("numeric" if var.is_continuous else "categorical"),
                "role": "meta"
            })
        
        return {
            "success": True,
            "features": [v.name for v in new_domain.attributes],
            "target": [new_domain.class_var.name] if new_domain.class_var else [],
            "metas": [v.name for v in new_domain.metas],
            "ignored": request.ignored,
            "instances": len(new_data),
            "variables": len(new_domain.attributes) + (1 if new_domain.class_var else 0) + len(new_domain.metas),
            "columns": columns,
            "data": data_rows
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

