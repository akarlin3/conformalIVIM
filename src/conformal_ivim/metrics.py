"""Coverage, sharpness, proper scoring rules, and marginal/conditional diagnostics
for IVIM prediction intervals.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


def empirical_coverage(lower, upper, true) -> float:
    """Fraction of points whose ground truth lies within ``[lower, upper]`` (inclusive)."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    true = np.asarray(true, dtype=float)
    return float(np.mean((true >= lower) & (true <= upper)))


def interval_width(lower, upper) -> np.ndarray:
    """Elementwise prediction interval width ``upper - lower``."""
    return np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)


def interval_score(lower, upper, true, alpha: float) -> np.ndarray:
    """Gneiting & Raftery (2007) proper interval score for a central ``(1 - alpha)`` band.

    Lower is better: rewards sharp intervals while adding a ``2 / alpha`` penalty
    proportional to distance outside ``[lower, upper]``:

        IS = (u - l) + (2 / alpha) * (l - y) * 1{y < l} + (2 / alpha) * (y - u) * 1{y > u}.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    true = np.asarray(true, dtype=float)
    width = upper - lower
    below = np.maximum(lower - true, 0.0)
    above = np.maximum(true - upper, 0.0)
    return width + (2.0 / float(alpha)) * (below + above)


def sharpness_summary(lower, upper) -> Dict[str, Any]:
    """Summarize interval sharpness (median, mean, IQR, and 90th-percentile width).

    Operates along axis 0 so ``(N,)`` inputs return scalar floats and ``(N, P)``
    inputs return length-``P`` NumPy arrays.
    """
    w = interval_width(lower, upper)
    q25, q50, q75, q90 = np.quantile(w, [0.25, 0.50, 0.75, 0.90], axis=0)
    mean_w = np.mean(w, axis=0)
    if w.ndim == 1:
        return {
            "median_width": float(q50),
            "mean_width": float(mean_w),
            "iqr_width": float(q75 - q25),
            "q90_width": float(q90),
        }
    return {
        "median_width": q50,
        "mean_width": mean_w,
        "iqr_width": q75 - q25,
        "q90_width": q90,
    }


def directional_miss_rates(lower, upper, true) -> Dict[str, float]:
    """Decompose miscoverage into lower-tail (``y < L``) and upper-tail (``y > U``) misses."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    true = np.asarray(true, dtype=float)
    lo_miss = float(np.mean(true < lower))
    hi_miss = float(np.mean(true > upper))
    cov = float(np.mean((true >= lower) & (true <= upper)))
    tot_miss = lo_miss + hi_miss
    upper_share = float(hi_miss / tot_miss) if tot_miss > 0.0 else 0.0
    return {
        "coverage": cov,
        "lower_miss": lo_miss,
        "upper_miss": hi_miss,
        "upper_share_of_misses": upper_share,
    }


def coverage_by_stratum(
    lower,
    upper,
    true,
    strata,
    keys: Optional[Iterable[Any]] = None,
) -> Dict[Any, Dict[str, Any]]:
    """Compute empirical coverage, sample count, and median width per stratum."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    true = np.asarray(true, dtype=float)
    strata_arr = np.asarray(strata)
    if keys is None:
        keys = list(np.unique(strata_arr))
    out: Dict[Any, Dict[str, Any]] = {}
    for k in keys:
        mask = strata_arr == k
        n = int(np.sum(mask))
        if n == 0:
            out[k] = {"coverage": float("nan"), "count": 0, "median_width": float("nan")}
        else:
            out[k] = {
                "coverage": empirical_coverage(lower[mask], upper[mask], true[mask]),
                "count": n,
                "median_width": float(
                    np.median(interval_width(lower[mask], upper[mask]))
                ),
            }
    return out


def regime_terciles(
    values,
    n: int = 3,
    edges: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign regime labels (e.g. low / mid / high true D* terciles)."""
    values_arr = np.asarray(values, dtype=float)
    if edges is None:
        edges_arr = np.quantile(values_arr, [(i + 1) / n for i in range(n - 1)])
    else:
        edges_arr = np.asarray(edges, dtype=float)
    return np.digitize(values_arr, edges_arr), edges_arr


def conditional_coverage_grid(
    lower,
    upper,
    true,
    regime,
    snr,
    snr_levels: Optional[Sequence[float]] = None,
    n_regime: int = 3,
) -> Dict[str, Any]:
    """2D conditional coverage map across parameter-regime x SNR strata.

    Returns a dictionary containing the ``(n_regime, n_snr)`` coverage ``grid``,
    stratum ``counts``, top-regime (e.g. high-D*) marginal coverage ``hi_marg``,
    worst SNR cell within the top regime ``hi_worst``, overall worst cell
    ``worst_overall``, and top-regime median interval width ``hi_width``.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    true = np.asarray(true, dtype=float)
    regime = np.asarray(regime, dtype=int)
    snr = np.asarray(snr)
    if snr_levels is None:
        snr_levels = list(np.unique(snr))

    grid = np.full((n_regime, len(snr_levels)), np.nan, dtype=float)
    counts = np.zeros((n_regime, len(snr_levels)), dtype=int)
    for r in range(n_regime):
        for k, s in enumerate(snr_levels):
            m = (regime == r) & (snr == s)
            counts[r, k] = int(m.sum())
            if m.sum() > 0:
                grid[r, k] = empirical_coverage(lower[m], upper[m], true[m])

    hi_mask = regime == (n_regime - 1)
    hi_marg = (
        empirical_coverage(lower[hi_mask], upper[hi_mask], true[hi_mask])
        if hi_mask.any()
        else float("nan")
    )
    hi_cells = grid[n_regime - 1, :]
    hi_worst = (
        float(np.nanmin(hi_cells)) if not np.all(np.isnan(hi_cells)) else float("nan")
    )
    worst_overall = (
        float(np.nanmin(grid)) if not np.all(np.isnan(grid)) else float("nan")
    )
    hi_width = (
        float(np.median(interval_width(lower[hi_mask], upper[hi_mask])))
        if hi_mask.any()
        else float("nan")
    )
    return {
        "grid": grid,
        "counts": counts,
        "hi_marg": hi_marg,
        "hi_worst": hi_worst,
        "worst_overall": worst_overall,
        "hi_counts": counts[n_regime - 1, :],
        "hi_width": hi_width,
    }


def evaluate_intervals(
    lower,
    upper,
    true,
    alpha: float = 0.10,
    param_names: Sequence[str] = ("D", "Dstar", "f"),
    snr: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Compute marginal coverage, sharpness, interval scores, and miss diagnostics
    across IVIM parameters ``(D, Dstar, f)``.
    """
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    y = np.asarray(true, dtype=float)
    nominal = 1.0 - float(alpha)

    if lo.ndim == 1:
        lo, hi, y = lo[:, None], hi[:, None], y[:, None]
        names = [param_names[0]]
    else:
        names = list(param_names[: lo.shape[1]])

    per_param: Dict[str, Dict[str, Any]] = {}
    for j, name in enumerate(names):
        miss = directional_miss_rates(lo[:, j], hi[:, j], y[:, j])
        sharp = sharpness_summary(lo[:, j], hi[:, j])
        iscore = float(np.mean(interval_score(lo[:, j], hi[:, j], y[:, j], alpha)))
        entry: Dict[str, Any] = {
            "nominal": nominal,
            "coverage": miss["coverage"],
            "coverage_gap": miss["coverage"] - nominal,
            "abs_coverage_gap": abs(miss["coverage"] - nominal),
            "median_width": sharp["median_width"],
            "mean_width": sharp["mean_width"],
            "iqr_width": sharp["iqr_width"],
            "mean_interval_score": iscore,
            "lower_miss": miss["lower_miss"],
            "upper_miss": miss["upper_miss"],
            "upper_share_of_misses": miss["upper_share_of_misses"],
        }
        if snr is not None:
            entry["by_snr"] = coverage_by_stratum(lo[:, j], hi[:, j], y[:, j], snr)
        per_param[name] = entry

    return {
        "alpha": float(alpha),
        "nominal": nominal,
        "parameters": per_param,
    }
