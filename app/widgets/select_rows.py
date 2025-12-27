"""
Select Rows Widget API endpoints.
"""

import logging
import math
import os
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["Data"])

# Check Orange3 availability
try:
    from Orange.data import Table, ContinuousVariable, DiscreteVariable, StringVariable
    import Orange.data.filter as data_filter
    from Orange.data.filter import FilterContinuous, FilterString
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False


class SelectRowsCondition(BaseModel):
    """Single condition for row filtering."""
    variable: str
    operator: str
    value: Optional[Any] = None
    value2: Optional[Any] = None


class SelectRowsRequest(BaseModel):
    """Request body for select rows."""
    data_source: str
    conditions: List[SelectRowsCondition]
    purge_attributes: bool = False
    purge_classes: bool = False


@router.post("/select-rows")
async def select_rows(request: SelectRowsRequest):
    """
    Filter data rows based on conditions.
    
    Returns matching and unmatched data counts.
    """
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Orange3 not available")
    
    try:
        from Orange.data import Table, ContinuousVariable, DiscreteVariable, StringVariable
        import Orange.data.filter as data_filter
        from Orange.data.filter import FilterContinuous, FilterString
        
        data_source = request.data_source
        data = None
        
        if data_source.startswith("/") or os.path.exists(data_source):
            try:
                data = Table(data_source)
            except (OSError, FileNotFoundError):
                pass
        
        if data is None:
            if "/" in data_source:
                dataset_name = data_source.split("/")[-1].replace(".tab", "")
            else:
                dataset_name = data_source.replace(".tab", "")
            
            try:
                data = Table(dataset_name)
            except (OSError, FileNotFoundError):
                raise HTTPException(status_code=404, detail=f"Dataset not found: {data_source}")
        
        total_count = len(data)
        
        if not request.conditions:
            return {
                "matching_count": total_count,
                "total_count": total_count,
                "unmatched_count": 0
            }
        
        filters = []
        
        for cond in request.conditions:
            var_name = cond.variable
            op = cond.operator
            val = cond.value
            val2 = cond.value2
            
            if var_name.startswith('all_'):
                if op == 'is_defined':
                    filters.append(data_filter.IsDefined())
                continue
            
            try:
                var = data.domain[var_name]
                var_idx = data.domain.index(var)
            except KeyError:
                continue
            
            if isinstance(var, ContinuousVariable):
                try:
                    float_val = float(val) if val else None
                    float_val2 = float(val2) if val2 else None
                except (ValueError, TypeError):
                    continue
                
                op_map = {
                    'equals': FilterContinuous.Equal,
                    'not_equals': FilterContinuous.NotEqual,
                    'less': FilterContinuous.Less,
                    'less_equal': FilterContinuous.LessEqual,
                    'greater': FilterContinuous.Greater,
                    'greater_equal': FilterContinuous.GreaterEqual,
                    'between': FilterContinuous.Between,
                    'outside': FilterContinuous.Outside,
                    'is_defined': FilterContinuous.IsDefined
                }
                
                if op in op_map:
                    if op == 'is_defined':
                        filters.append(data_filter.FilterContinuous(var_idx, op_map[op]))
                    elif op in ['between', 'outside']:
                        if float_val is not None and float_val2 is not None:
                            filters.append(data_filter.FilterContinuous(
                                var_idx, op_map[op], float_val, float_val2))
                    elif float_val is not None:
                        filters.append(data_filter.FilterContinuous(
                            var_idx, op_map[op], float_val))
                            
            elif isinstance(var, DiscreteVariable):
                if op == 'is_defined':
                    filters.append(data_filter.FilterDiscrete(var_idx, None))
                elif op == 'equals' and val:
                    filters.append(data_filter.FilterDiscrete(var_idx, {val}))
                elif op == 'not_equals' and val:
                    other_vals = set(var.values) - {val}
                    filters.append(data_filter.FilterDiscrete(var_idx, other_vals))
                elif op == 'in' and isinstance(val, list):
                    filters.append(data_filter.FilterDiscrete(var_idx, set(val)))
                    
            elif isinstance(var, StringVariable):
                op_map = {
                    'equals': FilterString.Equal,
                    'not_equals': FilterString.NotEqual,
                    'less': FilterString.Less,
                    'greater': FilterString.Greater,
                    'contains': FilterString.Contains,
                    'not_contains': FilterString.NotContain,
                    'starts_with': FilterString.StartsWith,
                    'ends_with': FilterString.EndsWith,
                    'is_defined': FilterString.IsDefined
                }
                
                if op in op_map and val:
                    filters.append(data_filter.FilterString(
                        var_idx, op_map[op], str(val)))
        
        if filters:
            combined_filter = data_filter.Values(filters)
            matching_data = combined_filter(data)
            matching_count = len(matching_data)
        else:
            matching_data = data
            matching_count = total_count
        
        def get_cell_value(var, val):
            if var.is_continuous:
                return float(val) if not math.isnan(val) else None
            else:
                return var.str_val(val)
        
        data_rows = []
        for row in matching_data:
            row_values = []
            for var in matching_data.domain.attributes:
                val = row[var]
                row_values.append(get_cell_value(var, val))
            if matching_data.domain.class_var:
                val = row[matching_data.domain.class_var]
                row_values.append(get_cell_value(matching_data.domain.class_var, val))
            for var in matching_data.domain.metas:
                val = row[var]
                if var.is_string:
                    row_values.append(str(val) if val else None)
                else:
                    row_values.append(get_cell_value(var, val))
            data_rows.append(row_values)
        
        columns = []
        for var in matching_data.domain.attributes:
            columns.append({
                "name": var.name,
                "type": "numeric" if var.is_continuous else "categorical",
                "role": "feature"
            })
        if matching_data.domain.class_var:
            columns.append({
                "name": matching_data.domain.class_var.name,
                "type": "numeric" if matching_data.domain.class_var.is_continuous else "categorical",
                "role": "target"
            })
        for var in matching_data.domain.metas:
            columns.append({
                "name": var.name,
                "type": "string" if var.is_string else ("numeric" if var.is_continuous else "categorical"),
                "role": "meta"
            })
        
        return {
            "matching_count": matching_count,
            "total_count": total_count,
            "unmatched_count": total_count - matching_count,
            "data": data_rows,
            "columns": columns,
            "instances": matching_count,
            "features": len(matching_data.domain.attributes)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Select rows error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

