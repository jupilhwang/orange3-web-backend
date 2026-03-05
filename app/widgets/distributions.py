"""
Distributions Widget API endpoints.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["Data"])

from app.core.orange_compat import (
    ORANGE_AVAILABLE,
    Table,
    DiscreteVariable,
    ContinuousVariable,
)


class DistributionsRequest(BaseModel):
    """Request model for distributions endpoint."""

    data_path: Optional[str] = None
    variable: str  # Variable name for distribution
    split_by: Optional[str] = None  # Variable to split by (discrete only)
    number_of_bins: int = 5  # Bin count for continuous variables
    stacked: bool = False
    show_probs: bool = False
    cumulative: bool = False
    sort_by_freq: bool = False
    fitted_distribution: int = 0  # 0=None, 1=Normal, 2=Beta, 3=Gamma, etc.
    kde_smoothing: int = 10
    selected_indices: Optional[List[int]] = None


@router.post("/distributions")
async def get_distributions(
    request: DistributionsRequest, x_session_id: Optional[str] = Header(None)
):
    """
    Calculate distribution data for a variable.
    Returns histogram data, statistics, and fitted curve if requested.
    """
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Orange3 not available")

    try:
        from Orange.data import Table, DiscreteVariable, ContinuousVariable
        from Orange.statistics import distribution, contingency
        from Orange.preprocess.discretize import decimal_binnings
        import numpy as np
        from scipy.stats import norm, rayleigh, beta, gamma, pareto, expon

        # Load data (supports datasets, uploads, kmeans results)
        from app.core.data_utils import async_load_data

        data = None
        if request.data_path:
            logger.info(
                f"Loading distributions data from: {request.data_path} (session: {x_session_id})"
            )
            data = await async_load_data(request.data_path, session_id=x_session_id)

        if data is None:
            raise HTTPException(
                status_code=400,
                detail=f"No data provided or failed to load: {request.data_path}",
            )

        # Filter by selected indices if provided
        original_len = len(data)
        if request.selected_indices and len(request.selected_indices) > 0:
            valid_indices = [i for i in request.selected_indices if 0 <= i < len(data)]
            if valid_indices:
                data = data[valid_indices]
                logger.info(
                    f"Distributions: filtered to {len(data)} of {original_len} instances"
                )

        # Find the variable
        var = None
        domain = data.domain
        all_vars = (
            list(domain.attributes) + list(domain.class_vars) + list(domain.metas)
        )
        for v in all_vars:
            if v.name == request.variable:
                var = v
                break

        if var is None:
            raise HTTPException(
                status_code=400, detail=f"Variable '{request.variable}' not found"
            )

        # Find split_by variable (discrete only)
        cvar = None
        if request.split_by:
            for v in all_vars:
                if v.name == request.split_by and isinstance(v, DiscreteVariable):
                    cvar = v
                    break

        # Get column data
        column = data.get_column(var)
        valid_mask = np.isfinite(column)

        if cvar:
            ccolumn = data.get_column(cvar)
            valid_mask = valid_mask & np.isfinite(ccolumn)

        valid_data = column[valid_mask]
        valid_group_data = ccolumn[valid_mask] if cvar else None

        if len(valid_data) == 0:
            return {
                "variable": var.name,
                "type": "discrete"
                if isinstance(var, DiscreteVariable)
                else "continuous",
                "bins": [],
                "total": 0,
                "error": "No valid data",
            }

        # Build response
        split_colors = None
        split_values = None
        if cvar:
            split_values = [str(v) for v in cvar.values]
            try:
                if hasattr(cvar, "colors") and cvar.colors is not None:
                    split_colors = [[int(c) for c in color] for color in cvar.colors]
            except Exception as e:
                logger.debug(f"Suppressed error: {e}")

        result = {
            "variable": var.name,
            "type": "discrete" if isinstance(var, DiscreteVariable) else "continuous",
            "split_by": cvar.name if cvar else None,
            "split_values": split_values,
            "split_colors": split_colors,
            "total": int(len(valid_data)),
            "bins": [],
            "fitted_curve": None,
            "statistics": {},
        }

        if isinstance(var, DiscreteVariable):
            # Discrete variable - category counts
            if cvar:
                conts = contingency.get_contingency(data, cvar, var)
                conts = np.array(conts)

                if request.sort_by_freq:
                    order = np.argsort(conts.sum(axis=1))[::-1]
                else:
                    order = np.arange(len(conts))

                ordered_values = [str(var.values[i]) for i in order]

                for i, idx in enumerate(order):
                    freqs = [int(f) for f in conts[idx]]
                    total_freq = int(sum(freqs))
                    result["bins"].append(
                        {
                            "label": ordered_values[i],
                            "x": i,
                            "frequencies": freqs,
                            "total": total_freq,
                            "percentage": float(100 * total_freq / len(valid_data))
                            if len(valid_data) > 0
                            else 0.0,
                        }
                    )
            else:
                dist = distribution.get_distribution(data, var)
                dist = np.array(dist)

                if request.sort_by_freq:
                    order = np.argsort(dist)[::-1]
                else:
                    order = np.arange(len(dist))

                ordered_values = [str(var.values[i]) for i in order]

                for i, idx in enumerate(order):
                    freq = int(dist[idx])
                    result["bins"].append(
                        {
                            "label": ordered_values[i],
                            "x": i,
                            "frequencies": [freq],
                            "total": freq,
                            "percentage": float(100 * freq / len(valid_data))
                            if len(valid_data) > 0
                            else 0.0,
                        }
                    )
        else:
            # Continuous variable - histogram
            binnings = decimal_binnings(valid_data)
            if not binnings:
                bin_count = max(5, min(20, int(np.sqrt(len(valid_data)))))
                thresholds = np.linspace(
                    np.min(valid_data), np.max(valid_data), bin_count + 1
                )
            else:
                bin_idx = min(request.number_of_bins, len(binnings) - 1)
                thresholds = binnings[bin_idx].thresholds

            if cvar:
                nvalues = len(cvar.values)
                ys = []
                for val_idx in range(nvalues):
                    group_data = valid_data[valid_group_data == val_idx]
                    hist, _ = np.histogram(group_data, bins=thresholds)
                    ys.append(hist)

                cumulative_freqs = np.zeros(nvalues)
                for i in range(len(thresholds) - 1):
                    x0, x1 = thresholds[i], thresholds[i + 1]
                    freqs = [int(y[i]) for y in ys]
                    cumulative_freqs += np.array(freqs)

                    if request.cumulative:
                        plot_freqs = cumulative_freqs.astype(int).tolist()
                    else:
                        plot_freqs = freqs

                    result["bins"].append(
                        {
                            "label": f"{x0:.3g} - {x1:.3g}",
                            "x0": float(x0),
                            "x1": float(x1),
                            "x": float((x0 + x1) / 2),
                            "frequencies": plot_freqs,
                            "total": int(sum(plot_freqs)),
                            "percentage": float(100 * sum(freqs) / len(valid_data))
                            if len(valid_data) > 0
                            else 0.0,
                        }
                    )
            else:
                hist, edges = np.histogram(valid_data, bins=thresholds)
                cumulative = 0

                for i in range(len(hist)):
                    x0, x1 = float(edges[i]), float(edges[i + 1])
                    freq = int(hist[i])
                    cumulative += freq

                    plot_freq = cumulative if request.cumulative else freq

                    result["bins"].append(
                        {
                            "label": f"{x0:.3g} - {x1:.3g}",
                            "x0": x0,
                            "x1": x1,
                            "x": float((x0 + x1) / 2),
                            "frequencies": [plot_freq],
                            "total": plot_freq,
                            "percentage": float(100 * freq / len(valid_data))
                            if len(valid_data) > 0
                            else 0.0,
                        }
                    )

            # Fitted distribution
            if request.fitted_distribution > 0 and not cvar:
                fitters = [
                    None,  # 0: None
                    norm,  # 1: Normal
                    beta,  # 2: Beta
                    gamma,  # 3: Gamma
                    rayleigh,  # 4: Rayleigh
                    pareto,  # 5: Pareto
                    expon,  # 6: Exponential
                ]

                if request.fitted_distribution < len(fitters):
                    fitter = fitters[request.fitted_distribution]
                    if fitter:
                        try:
                            params = fitter.fit(valid_data)
                            x_range = np.linspace(thresholds[0], thresholds[-1], 100)
                            y_pdf = fitter.pdf(x_range, *params)
                            bin_width = (thresholds[-1] - thresholds[0]) / (
                                len(thresholds) - 1
                            )
                            y_scaled = y_pdf * len(valid_data) * bin_width

                            result["fitted_curve"] = {
                                "x": x_range.tolist(),
                                "y": y_scaled.tolist(),
                                "params": list(params),
                                "type": [
                                    "None",
                                    "Normal",
                                    "Beta",
                                    "Gamma",
                                    "Rayleigh",
                                    "Pareto",
                                    "Exponential",
                                ][request.fitted_distribution],
                            }
                        except Exception as fit_error:
                            logger.warning(f"Fitting error: {fit_error}")

            # Kernel Density Estimation (KDE)
            if request.kde_smoothing > 0:
                try:
                    from scipy.stats import gaussian_kde

                    # Calculate bandwidth based on kde_smoothing (1-50)
                    # Lower value = more smoothing, higher = less smoothing
                    # Convert kde_smoothing to bandwidth factor
                    bw_factor = max(0.1, request.kde_smoothing / 50.0)

                    kde = gaussian_kde(valid_data, bw_method="silverman")
                    kde.set_bandwidth(kde.factor * bw_factor)

                    x_range = np.linspace(
                        np.min(valid_data) - np.std(valid_data) * 0.5,
                        np.max(valid_data) + np.std(valid_data) * 0.5,
                        200,
                    )
                    y_kde = kde(x_range)

                    # Scale to match histogram
                    bin_width = (thresholds[-1] - thresholds[0]) / (len(thresholds) - 1)
                    y_scaled = y_kde * len(valid_data) * bin_width

                    result["kde_curve"] = {
                        "x": x_range.tolist(),
                        "y": y_scaled.tolist(),
                        "bandwidth": float(kde.factor),
                        "smoothing": request.kde_smoothing,
                    }
                except Exception as kde_error:
                    logger.warning(f"KDE error: {kde_error}")

            # Statistics
            result["statistics"] = {
                "mean": float(np.mean(valid_data)),
                "std": float(np.std(valid_data)),
                "min": float(np.min(valid_data)),
                "max": float(np.max(valid_data)),
                "median": float(np.median(valid_data)),
                "q1": float(np.percentile(valid_data, 25)),
                "q3": float(np.percentile(valid_data, 75)),
                "iqr": float(
                    np.percentile(valid_data, 75) - np.percentile(valid_data, 25)
                ),
                "skewness": float(
                    ((valid_data - np.mean(valid_data)) ** 3).mean()
                    / (np.std(valid_data) ** 3)
                )
                if np.std(valid_data) > 0
                else 0,
                "kurtosis": float(
                    ((valid_data - np.mean(valid_data)) ** 4).mean()
                    / (np.std(valid_data) ** 4)
                    - 3
                )
                if np.std(valid_data) > 0
                else 0,
            }

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Distributions error: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
