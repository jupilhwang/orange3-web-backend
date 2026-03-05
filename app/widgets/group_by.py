"""
Group By Widget API endpoints.
"""

import logging
import math
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["Data"])

from app.core.orange_compat import ORANGE_AVAILABLE

SUPPORTED_FUNCTIONS = {"mean", "sum", "count", "min", "max", "std", "median"}


class AggregationSpec(BaseModel):
    """Single aggregation specification."""

    column: str
    function: str  # one of SUPPORTED_FUNCTIONS


class GroupByRequest(BaseModel):
    """Request model for group by."""

    data_path: Optional[str] = None
    group_by_column: str
    aggregations: List[AggregationSpec] = []


@router.post("/group-by")
async def group_by(request: GroupByRequest, x_session_id: Optional[str] = Header(None)):
    """
    Aggregate data by grouping on a categorical column.

    Returns aggregated data table with one row per unique group value.
    Aggregated column names follow the pattern: {column}__{function}.
    """
    if not ORANGE_AVAILABLE:
        return {
            "success": True,
            "group_by_column": request.group_by_column,
            "aggregations": [a.model_dump() for a in request.aggregations],
            "instances": 0,
            "columns": [],
            "data": [],
            "note": "Fallback response (Orange3 not available)",
        }

    if not request.data_path:
        return {
            "success": True,
            "group_by_column": request.group_by_column,
            "aggregations": [a.model_dump() for a in request.aggregations],
            "instances": 0,
            "columns": [],
            "data": [],
        }

    # Validate aggregation functions before loading data
    for agg in request.aggregations:
        if agg.function not in SUPPORTED_FUNCTIONS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unsupported aggregation function: '{agg.function}'. "
                    f"Supported: {sorted(SUPPORTED_FUNCTIONS)}"
                ),
            )

    try:
        import pandas as pd
        from app.core.data_utils import async_load_data as async_load_data_util

        logger.info(
            f"Loading Group By data from: {request.data_path} (session: {x_session_id})"
        )
        data = await async_load_data_util(request.data_path, session_id=x_session_id)

        if data is None:
            raise HTTPException(
                status_code=404, detail=f"Data not found: {request.data_path}"
            )

        df = _orange_to_dataframe(data)

        if request.group_by_column not in df.columns:
            raise HTTPException(
                status_code=422,
                detail=f"Column '{request.group_by_column}' not found in data",
            )

        if not request.aggregations:
            # Return group counts when no aggregations specified
            grouped = (
                df.groupby(request.group_by_column).size().reset_index(name="count")
            )
            columns = [
                {
                    "name": request.group_by_column,
                    "type": "categorical",
                    "role": "feature",
                },
                {"name": "count", "type": "numeric", "role": "feature"},
            ]
            data_rows = [
                [_clean_val(row[c["name"]]) for c in columns]
                for _, row in grouped.iterrows()
            ]
            return {
                "success": True,
                "group_by_column": request.group_by_column,
                "instances": len(grouped),
                "columns": columns,
                "data": data_rows,
            }

        # Build named aggregations for pandas groupby
        named_aggs = {}
        valid_aggs = []
        for agg in request.aggregations:
            if agg.column not in df.columns:
                continue
            col_key = f"{agg.column}__{agg.function}"
            fn = "median" if agg.function == "median" else agg.function
            named_aggs[col_key] = pd.NamedAgg(column=agg.column, aggfunc=fn)
            valid_aggs.append((col_key, agg))

        if not named_aggs:
            # All aggregation columns were invalid - fall back to counts
            grouped = (
                df.groupby(request.group_by_column).size().reset_index(name="count")
            )
            columns = [
                {
                    "name": request.group_by_column,
                    "type": "categorical",
                    "role": "feature",
                },
                {"name": "count", "type": "numeric", "role": "feature"},
            ]
            data_rows = [
                [_clean_val(row[c["name"]]) for c in columns]
                for _, row in grouped.iterrows()
            ]
            return {
                "success": True,
                "group_by_column": request.group_by_column,
                "instances": len(grouped),
                "columns": columns,
                "data": data_rows,
            }

        grouped = df.groupby(request.group_by_column).agg(**named_aggs).reset_index()

        # Build column metadata
        columns = [
            {"name": request.group_by_column, "type": "categorical", "role": "feature"}
        ]
        for col_key, _ in valid_aggs:
            if col_key in grouped.columns:
                columns.append({"name": col_key, "type": "numeric", "role": "feature"})

        data_rows = [
            [_clean_val(row[c["name"]]) for c in columns]
            for _, row in grouped.iterrows()
        ]

        return {
            "success": True,
            "group_by_column": request.group_by_column,
            "instances": len(grouped),
            "columns": columns,
            "data": data_rows,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _orange_to_dataframe(data):
    """Convert an Orange3 Table to a pandas DataFrame."""
    import pandas as pd
    import numpy as np

    domain = data.domain
    all_vars = list(domain.attributes)
    if domain.class_var:
        all_vars.append(domain.class_var)
    all_vars.extend(domain.metas)

    col_names = [v.name for v in all_vars]
    rows = []

    for row in data:
        r = []
        for var in domain.attributes:
            val = row[var]
            if var.is_continuous:
                fv = float(val)
                r.append(None if math.isnan(fv) else fv)
            else:
                r.append(var.str_val(val))
        if domain.class_var:
            var = domain.class_var
            val = row[var]
            if var.is_continuous:
                fv = float(val)
                r.append(None if math.isnan(fv) else fv)
            else:
                r.append(var.str_val(val))
        for var in domain.metas:
            val = row[var]
            if var.is_string:
                r.append(str(val) if val else None)
            elif var.is_continuous:
                fv = float(val)
                r.append(None if math.isnan(fv) else fv)
            else:
                r.append(var.str_val(val))
        rows.append(r)

    return pd.DataFrame(rows, columns=col_names)


def _clean_val(v):
    """Normalize a value: convert NaN to None, keep other types as-is."""
    if v is None:
        return None
    try:
        if math.isnan(float(v)):
            return None
    except (TypeError, ValueError):
        pass
    return v
