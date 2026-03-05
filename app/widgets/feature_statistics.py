"""
Feature Statistics 위젯 API

Orange3의 OWFeatureStatistics 위젯을 웹 버전으로 구현합니다.
각 피처(변수)의 통계 정보를 제공합니다.

Input:
    - Data (Table): 통계를 계산할 데이터

Output:
    - Reduced Data (Table): 선택된 컬럼만 포함된 데이터
    - Statistics (Table): 컬럼별 통계 테이블
"""

from typing import List, Optional, Any
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
import numpy as np

router = APIRouter(prefix="/data/feature-statistics", tags=["Feature Statistics"])


class FeatureStatisticsRequest(BaseModel):
    """Feature Statistics 요청 모델"""

    data_path: str
    color_var: Optional[str] = None  # 분포 색상 변수
    selected_indices: Optional[List[int]] = None  # 선택된 행 인덱스


class ColumnStatistics(BaseModel):
    """컬럼별 통계 모델"""

    index: int
    name: str
    type: str  # 'continuous', 'discrete', 'time', 'string'
    role: str  # 'attribute', 'class', 'meta'
    mean: Optional[float] = None
    mode: Optional[Any] = None
    median: Optional[float] = None
    dispersion: Optional[float] = None  # CV for continuous, entropy for discrete
    min: Optional[Any] = None
    max: Optional[Any] = None
    missing: int = 0
    missing_percent: float = 0.0
    distribution: Optional[dict] = None  # 히스토그램 데이터


class FeatureStatisticsResponse(BaseModel):
    """Feature Statistics 응답 모델"""

    success: bool
    n_instances: int
    n_features: int
    columns: List[ColumnStatistics]
    color_var: Optional[str] = None
    color_values: Optional[List[str]] = None
    error: Optional[str] = None


class ReducedDataRequest(BaseModel):
    """Reduced Data 요청 모델"""

    data_path: str
    selected_columns: List[str]


class ReducedDataResponse(BaseModel):
    """Reduced Data 응답 모델"""

    success: bool
    data_path: str  # 결과 데이터 경로
    n_instances: int
    n_features: int
    error: Optional[str] = None


def _get_variable_type(var) -> str:
    """변수 타입을 문자열로 반환"""
    from Orange.data import (
        ContinuousVariable,
        DiscreteVariable,
        TimeVariable,
        StringVariable,
    )

    if isinstance(var, TimeVariable):
        return "time"
    elif isinstance(var, ContinuousVariable):
        return "continuous"
    elif isinstance(var, DiscreteVariable):
        return "discrete"
    elif isinstance(var, StringVariable):
        return "string"
    return "unknown"


def _get_variable_role(var, domain) -> str:
    """변수 역할을 문자열로 반환"""
    if var in domain.attributes:
        return "attribute"
    elif var in domain.class_vars:
        return "class"
    elif var in domain.metas:
        return "meta"
    return "unknown"


def _compute_dispersion(x: np.ndarray, var_type: str) -> Optional[float]:
    """분산 계산 - 연속형: coefficient of variation, 범주형: entropy"""
    import scipy.stats as ss

    if var_type == "continuous":
        # Coefficient of variation
        valid = x[~np.isnan(x)]
        if len(valid) == 0:
            return None
        mean = np.mean(valid)
        if np.isclose(mean, 0, atol=1e-12):
            return float("inf") if np.std(valid) > 0 else 0.0
        return float(np.std(valid) / abs(mean))
    elif var_type == "discrete":
        # Entropy
        valid = x[~np.isnan(x)].astype(int)
        if len(valid) == 0:
            return None
        counts = np.bincount(valid)
        p = counts / counts.sum()
        p = p[p > 0]  # 0 제외
        return float(ss.entropy(p))
    return None


def _get_palette_colors(color_var, count: int) -> Optional[list]:
    """QColor palette에서 hex 색상 추출"""
    if not hasattr(color_var, "palette") or color_var.palette is None:
        return None

    colors = []
    try:
        palette = color_var.palette
        for i in range(min(count, len(palette))):
            c = palette[i]
            # QColor 객체인 경우 RGB 값 추출
            if hasattr(c, "red") and hasattr(c, "green") and hasattr(c, "blue"):
                # QColor
                colors.append(f"#{c.red():02x}{c.green():02x}{c.blue():02x}")
            elif hasattr(c, "__iter__") and len(c) >= 3:
                # tuple/list (r, g, b)
                colors.append(f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}")
            elif isinstance(c, int):
                # integer color
                colors.append(f"#{c:06x}")
            else:
                # 기본 색상
                default_colors = ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]
                colors.append(default_colors[i % len(default_colors)])
    except Exception as e:  # noqa: F841
        return None

    return colors if colors else None


def _compute_distribution(
    x: np.ndarray, var, var_type: str, color_data=None, color_var=None
) -> Optional[dict]:
    """분포 히스토그램 데이터 계산"""
    from Orange.data import DiscreteVariable

    valid_mask = (
        ~np.isnan(x) if np.issubdtype(x.dtype, np.number) else np.array([True] * len(x))
    )
    valid = x[valid_mask]

    if len(valid) == 0:
        return None

    if var_type == "discrete":
        # 범주형: 각 값별 카운트
        counts = np.bincount(valid.astype(int), minlength=len(var.values))

        # 색상 변수가 있는 경우 분리
        if (
            color_data is not None
            and color_var is not None
            and isinstance(color_var, DiscreteVariable)
        ):
            color_valid = color_data[valid_mask]
            stacked = []
            for c_idx in range(len(color_var.values)):
                c_mask = color_valid == c_idx
                c_counts = np.bincount(
                    valid[c_mask].astype(int), minlength=len(var.values)
                )
                stacked.append(c_counts.tolist())
            return {
                "type": "bar",
                "labels": list(var.values),
                "stacked": stacked,
                "colors": _get_palette_colors(color_var, len(color_var.values)),
            }

        return {"type": "bar", "labels": list(var.values), "counts": counts.tolist()}
    elif var_type in ("continuous", "time"):
        # 연속형: 히스토그램
        try:
            hist, bin_edges = np.histogram(valid, bins="auto")

            # 색상 변수가 있는 경우
            if (
                color_data is not None
                and color_var is not None
                and isinstance(color_var, DiscreteVariable)
            ):
                color_valid = color_data[valid_mask]
                stacked = []
                for c_idx in range(len(color_var.values)):
                    c_mask = color_valid == c_idx
                    c_hist, _ = np.histogram(valid[c_mask], bins=bin_edges)
                    stacked.append(c_hist.tolist())
                return {
                    "type": "histogram",
                    "bins": bin_edges.tolist(),
                    "stacked": stacked,
                    "colors": _get_palette_colors(color_var, len(color_var.values)),
                }

            return {
                "type": "histogram",
                "bins": bin_edges.tolist(),
                "counts": hist.tolist(),
            }
        except Exception as e:  # noqa: F841
            return None
    return None


@router.post("/compute", response_model=FeatureStatisticsResponse)
async def compute_feature_statistics(
    request: FeatureStatisticsRequest, x_session_id: Optional[str] = Header(None)
):
    """
    피처 통계 계산

    각 피처(변수)에 대해 다음 통계를 계산합니다:
    - Mean (평균) - 연속형만
    - Mode (최빈값)
    - Median (중앙값) - 연속형만
    - Dispersion (분산) - 연속형: CV, 범주형: entropy
    - Min/Max
    - Missing (결측치)
    - Distribution (분포 히스토그램)
    """
    try:
        from Orange.data import Table, StringVariable
        from app.core.data_utils import async_load_data

        # 데이터 로드
        data = await async_load_data(request.data_path, session_id=x_session_id)
        if data is None:
            return FeatureStatisticsResponse(
                success=False,
                n_instances=0,
                n_features=0,
                columns=[],
                error="Data not found",
            )

        n_instances = len(data)
        domain = data.domain

        # Color 변수 찾기
        color_var = None
        color_data = None
        if request.color_var:
            try:
                color_var = domain[request.color_var]
                col_idx = list(domain.variables).index(color_var)
                if col_idx < len(domain.attributes):
                    color_data = data.X[:, col_idx]
                elif col_idx < len(domain.attributes) + len(domain.class_vars):
                    y_idx = col_idx - len(domain.attributes)
                    color_data = data.Y if data.Y.ndim == 1 else data.Y[:, y_idx]
            except Exception as e:  # noqa: F841
                color_var = None

        # 모든 변수 수집 (StringVariable 제외)
        all_vars = []
        for var in domain.attributes:
            if not isinstance(var, StringVariable):
                all_vars.append((var, "attribute"))
        for var in domain.class_vars:
            if not isinstance(var, StringVariable):
                all_vars.append((var, "class"))
        for var in domain.metas:
            if not isinstance(var, StringVariable):
                all_vars.append((var, "meta"))

        columns = []
        for idx, (var, role) in enumerate(all_vars):
            var_type = _get_variable_type(var)

            # 데이터 추출
            if role == "attribute":
                col_idx = list(domain.attributes).index(var)
                x = data.X[:, col_idx]
            elif role == "class":
                if data.Y.ndim == 1:
                    x = data.Y
                else:
                    col_idx = list(domain.class_vars).index(var)
                    x = data.Y[:, col_idx]
            else:  # meta
                col_idx = list(domain.metas).index(var)
                x = data.metas[:, col_idx]

            # 숫자형으로 변환
            if not np.issubdtype(x.dtype, np.number):
                try:
                    x = x.astype(np.float64)
                except (ValueError, TypeError):
                    # 변환 불가능한 경우
                    columns.append(
                        ColumnStatistics(
                            index=idx,
                            name=var.name,
                            type=var_type,
                            role=role,
                            missing=n_instances,
                            missing_percent=100.0,
                        )
                    )
                    continue

            # 결측치 계산
            missing = int(np.isnan(x).sum())
            missing_percent = (
                round(100 * missing / n_instances, 1) if n_instances > 0 else 0.0
            )

            # 유효 데이터
            valid = x[~np.isnan(x)]

            # 통계 계산
            col_stats = ColumnStatistics(
                index=idx,
                name=var.name,
                type=var_type,
                role=role,
                missing=missing,
                missing_percent=missing_percent,
            )

            if len(valid) > 0:
                # Mean (연속형만)
                if var_type in ("continuous", "time"):
                    col_stats.mean = float(np.mean(valid))

                # Mode
                try:
                    from scipy import stats

                    mode_result = stats.mode(valid, keepdims=False)
                    mode_val = mode_result.mode
                    if var_type == "discrete" and hasattr(var, "values"):
                        col_stats.mode = (
                            var.values[int(mode_val)]
                            if int(mode_val) < len(var.values)
                            else str(mode_val)
                        )
                    else:
                        col_stats.mode = (
                            float(mode_val) if np.isfinite(mode_val) else None
                        )
                except Exception as e:  # noqa: F841
                    pass

                # Median (연속형만)
                if var_type in ("continuous", "time"):
                    col_stats.median = float(np.median(valid))

                # Dispersion
                col_stats.dispersion = _compute_dispersion(x, var_type)

                # Min/Max
                if var_type == "discrete" and hasattr(var, "values"):
                    col_stats.min = (
                        var.values[int(np.min(valid))]
                        if int(np.min(valid)) < len(var.values)
                        else None
                    )
                    col_stats.max = (
                        var.values[int(np.max(valid))]
                        if int(np.max(valid)) < len(var.values)
                        else None
                    )
                elif var_type != "string":
                    col_stats.min = float(np.min(valid))
                    col_stats.max = float(np.max(valid))

                # Distribution
                col_stats.distribution = _compute_distribution(
                    x, var, var_type, color_data, color_var
                )

            columns.append(col_stats)

        # Color 변수 정보
        color_values = None
        if color_var is not None and hasattr(color_var, "values"):
            color_values = list(color_var.values)

        return FeatureStatisticsResponse(
            success=True,
            n_instances=n_instances,
            n_features=len(columns),
            columns=columns,
            color_var=request.color_var,
            color_values=color_values,
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return FeatureStatisticsResponse(
            success=False, n_instances=0, n_features=0, columns=[], error=str(e)
        )


@router.post("/reduced-data", response_model=ReducedDataResponse)
async def get_reduced_data(
    request: ReducedDataRequest, x_session_id: Optional[str] = Header(None)
):
    """
    선택된 컬럼만 포함된 데이터 반환
    """
    try:
        from Orange.data import Table
        from app.core.data_utils import async_load_data, DataSessionManager

        # 데이터 로드
        data = await async_load_data(request.data_path, session_id=x_session_id)
        if data is None:
            return ReducedDataResponse(
                success=False,
                data_path="",
                n_instances=0,
                n_features=0,
                error="Data not found",
            )

        # 선택된 컬럼으로 필터링
        domain = data.domain
        selected_vars = []
        for col_name in request.selected_columns:
            try:
                var = domain[col_name]
                selected_vars.append(var)
            except Exception as e:  # noqa: F841
                pass

        if not selected_vars:
            return ReducedDataResponse(
                success=False,
                data_path="",
                n_instances=0,
                n_features=0,
                error="No valid columns selected",
            )

        # Reduced data 생성
        reduced_data = data[:, selected_vars]

        # 세션에 저장
        import uuid

        result_id = f"feature_stats_{uuid.uuid4().hex[:8]}"
        result_path = f"feature_stats/{result_id}"
        session_id = x_session_id or "default"
        await DataSessionManager.store(session_id, result_path, reduced_data)

        return ReducedDataResponse(
            success=True,
            data_path=result_path,
            n_instances=len(reduced_data),
            n_features=len(selected_vars),
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return ReducedDataResponse(
            success=False, data_path="", n_instances=0, n_features=0, error=str(e)
        )


@router.get("/options")
async def get_options():
    """위젯 옵션 목록 반환"""
    return {
        "columns": [
            {"id": "icon", "name": "", "sortable": False},
            {"id": "name", "name": "Name", "sortable": True},
            {"id": "distribution", "name": "Distribution", "sortable": False},
            {"id": "mean", "name": "Mean", "sortable": True},
            {"id": "mode", "name": "Mode", "sortable": True},
            {"id": "median", "name": "Median", "sortable": True},
            {"id": "dispersion", "name": "Dispersion", "sortable": True},
            {"id": "min", "name": "Min.", "sortable": True},
            {"id": "max", "name": "Max.", "sortable": True},
            {"id": "missing", "name": "Missing", "sortable": True},
        ]
    }
