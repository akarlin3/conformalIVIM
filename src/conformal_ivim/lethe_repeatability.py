"""Lethe / Echo scan-rescan repeatability scale diagnostics & Reverb precision-vs-coverage counterexample.

Implements:
- ``evaluate_echo_repeatability_scale``: evaluates test-retest repeatability coverage,
  standardized residual scale ratio :math:`\\sigma_{\\text{repeat}} / \\text{width}_{\\text{conformal}}`,
  Spearman rank correlation between local conformal interval width and empirical
  scan-rescan error, and bias-corrected and accelerated (BCa) bootstrap 95% CIs.
- ``run_reverb_precision_coverage_counterexample``: constructive synthetic & analytical
  proof demonstrating how conformal intervals achieve exact :math:`1 - \\alpha`
  ground-truth coverage while systematically under-covering scan-rescan repeatability.

License: MIT
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from scipy.stats import norm, spearmanr

from conformal_ivim.conformal import split_conformal
from conformal_ivim.forward import CLINICAL_B, simulate_ivim_signal

__all__ = [
    "LetheRepeatabilityReport",
    "bca_bootstrap_ci",
    "evaluate_echo_repeatability_scale",
    "run_reverb_precision_coverage_counterexample",
]


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if x_arr.size < 3 or np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return 0.0, 1.0
    res = spearmanr(x_arr, y_arr)
    r_val = float(res.statistic) if np.isfinite(res.statistic) else 0.0
    p_val = float(res.pvalue) if np.isfinite(res.pvalue) else 1.0
    return r_val, p_val


def bca_bootstrap_ci(
    samples: np.ndarray,
    stat_fn: Any,
    *,
    confidence: float = 0.95,
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute point estimate and BCa (bias-corrected and accelerated) bootstrap CI via ``PCG64``."""
    arr = np.asarray(samples, dtype=np.float64)
    n = arr.shape[0]
    theta_hat = float(stat_fn(arr))
    if n < 4:
        return theta_hat, theta_hat, theta_hat

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    boot_vals = np.empty(int(n_boot), dtype=np.float64)
    for b_idx in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boot_vals[b_idx] = float(stat_fn(arr[idx]))

    if np.allclose(boot_vals, boot_vals[0]):
        return theta_hat, float(boot_vals[0]), float(boot_vals[0])

    # Bias-correction factor z0
    prop_less = float(np.clip(np.mean(boot_vals < theta_hat), 1.0 / (2.0 * n_boot), 1.0 - 1.0 / (2.0 * n_boot)))
    z0 = float(norm.ppf(prop_less))

    # Jackknife acceleration factor a (subsampled to <= 120 blocks for speed on large N)
    n_jack = min(n, 120)
    jack_vals = np.empty(n_jack, dtype=np.float64)
    if n_jack == n:
        full_idx = np.arange(n)
        for i in range(n):
            jack_vals[i] = float(stat_fn(arr[full_idx != i]))
    else:
        blocks = np.array_split(np.arange(n), n_jack)
        for i, blk in enumerate(blocks):
            keep = np.ones(n, dtype=bool)
            keep[blk] = False
            jack_vals[i] = float(stat_fn(arr[keep]))

    jack_mean = float(np.mean(jack_vals))
    diff = jack_mean - jack_vals
    denom = 6.0 * (float(np.sum(diff**2)) ** 1.5)
    accel = float(np.sum(diff**3) / denom) if denom > 1e-18 else 0.0

    alpha_tail = (1.0 - float(confidence)) / 2.0
    z_lo = float(norm.ppf(alpha_tail))
    z_hi = float(norm.ppf(1.0 - alpha_tail))

    def _adj_quantile(z_alpha: float) -> float:
        num = z0 + z_alpha
        den = 1.0 - accel * num
        if abs(den) < 1e-9:
            return 0.5
        return float(np.clip(norm.cdf(z0 + num / den), 0.0, 1.0))

    q_lo = _adj_quantile(z_lo)
    q_hi = _adj_quantile(z_hi)
    ci_lo, ci_hi = np.quantile(boot_vals, [min(q_lo, q_hi), max(q_lo, q_hi)])
    return theta_hat, float(ci_lo), float(ci_hi)


def _unpack_intervals_and_params(
    intervals_scan1: Any,
    params_scan1: Any,
    params_scan2: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p1 = np.asarray(params_scan1, dtype=np.float64)
    p2 = np.asarray(params_scan2, dtype=np.float64)
    if p1.shape != p2.shape:
        raise ValueError(f"`params_scan1` shape {p1.shape} must match `params_scan2` shape {p2.shape}.")
    if p1.ndim == 1:
        p1 = p1[:, None]
        p2 = p2[:, None]
    elif p1.ndim != 2:
        raise ValueError(f"`params_scan1` must be 1D or 2D; got shape {p1.shape}.")

    if isinstance(intervals_scan1, Mapping):
        lo = np.asarray(intervals_scan1["lower"], dtype=np.float64)
        hi = np.asarray(intervals_scan1["upper"], dtype=np.float64)
    elif isinstance(intervals_scan1, tuple) and len(intervals_scan1) == 2:
        lo = np.asarray(intervals_scan1[0], dtype=np.float64)
        hi = np.asarray(intervals_scan1[1], dtype=np.float64)
    else:
        arr = np.asarray(intervals_scan1, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 2 and p1.shape[1] == 1:
            lo = arr[:, 0:1]
            hi = arr[:, 1:2]
        elif arr.ndim == 3 and arr.shape[-1] == 2:
            lo = arr[..., 0]
            hi = arr[..., 1]
        else:
            raise ValueError(
                "Unsupported `intervals_scan1` format; pass (lower, upper), {'lower', 'upper'}, or (..., 2) array."
            )

    if lo.ndim == 1:
        lo = lo[:, None]
        hi = hi[:, None]
    if lo.shape != p1.shape or hi.shape != p1.shape:
        raise ValueError(
            f"Interval bounds shape {lo.shape} must match parameter matrix shape {p1.shape}."
        )
    return lo, hi, p1, p2


@dataclass(frozen=True)
class LetheRepeatabilityReport(Mapping[str, Any]):
    """Immutable Lethe / Echo scan-rescan repeatability scale diagnostic report."""

    n_voxels: int
    n_parameters: int
    seed: int
    repeatability_coverage: float
    repeatability_coverage_bca_ci: tuple[float, float]
    sigma_repeat_over_width: float
    sigma_repeat_over_width_bca_ci: tuple[float, float]
    standardized_residual_rms: float
    spearman_r: float
    spearman_p: float
    per_parameter: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to a JSON-compatible dictionary."""
        return {
            "n_voxels": int(self.n_voxels),
            "n_parameters": int(self.n_parameters),
            "seed": int(self.seed),
            "repeatability_coverage": float(self.repeatability_coverage),
            "repeatability_coverage_bca_ci": [
                float(self.repeatability_coverage_bca_ci[0]),
                float(self.repeatability_coverage_bca_ci[1]),
            ],
            "sigma_repeat_over_width": float(self.sigma_repeat_over_width),
            "sigma_repeat_over_width_bca_ci": [
                float(self.sigma_repeat_over_width_bca_ci[0]),
                float(self.sigma_repeat_over_width_bca_ci[1]),
            ],
            "standardized_residual_rms": float(self.standardized_residual_rms),
            "spearman_r": float(self.spearman_r),
            "spearman_p": float(self.spearman_p),
            "per_parameter": dict(self.per_parameter),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def evaluate_echo_repeatability_scale(
    intervals_scan1: Any,
    params_scan1: Any,
    params_scan2: Any,
    seed: int = 42,
    *,
    alpha: float = 0.10,
    n_boot: int = 500,
    param_names: Sequence[str] | None = None,
) -> LetheRepeatabilityReport:
    """Evaluate scan-rescan repeatability coverage, scale ratio, Spearman correlation, and BCa 95% CIs."""
    lo, hi, p1, p2 = _unpack_intervals_and_params(intervals_scan1, params_scan1, params_scan2)
    n_vox, n_par = p1.shape
    if n_vox < 2:
        raise ValueError(f"At least 2 voxels are required; got {n_vox}.")

    if param_names is None:
        default_names = ("D", "Dstar", "f")
        names = default_names[:n_par] if n_par <= 3 else tuple(f"p{j}" for j in range(n_par))
    else:
        names = tuple(str(x) for x in param_names)

    z_crit = float(norm.ppf(1.0 - float(alpha) / 2.0))
    seed_int = int(seed)
    per_param: dict[str, dict[str, Any]] = {}

    cov_list: list[float] = []
    ratio_list: list[float] = []
    rms_list: list[float] = []
    spearman_r_list: list[float] = []
    spearman_p_list: list[float] = []

    for j in range(n_par):
        lo_j = lo[:, j]
        hi_j = hi[:, j]
        w_j = np.maximum(hi_j - lo_j, 1e-12)
        diff_j = p2[:, j] - p1[:, j]
        abs_diff_j = np.abs(diff_j)
        in_iv_j = ((p2[:, j] >= lo_j) & (p2[:, j] <= hi_j)).astype(np.float64)

        cov_pt, cov_lo, cov_hi = bca_bootstrap_ci(
            in_iv_j,
            np.mean,
            confidence=0.95,
            n_boot=n_boot,
            seed=seed_int + 17 * j,
        )

        # Pair [diff_j, w_j] for BCa CI on sigma_repeat / median(width_conformal)
        paired = np.column_stack([diff_j, w_j])

        def _scale_ratio_fn(mat: np.ndarray) -> float:
            # Within-subject repeatability SD: sigma_w = sqrt(mean((theta2 - theta1)^2) / 2)
            sigma_rep = float(np.sqrt(np.mean(mat[:, 0] ** 2) / 2.0))
            med_w = float(np.maximum(np.median(mat[:, 1]), 1e-12))
            return sigma_rep / med_w

        rat_pt, rat_lo, rat_hi = bca_bootstrap_ci(
            paired,
            _scale_ratio_fn,
            confidence=0.95,
            n_boot=n_boot,
            seed=seed_int + 31 * j + 5,
        )

        eff_sigma_conf = w_j / (2.0 * z_crit)
        std_rms = float(np.sqrt(np.mean((diff_j / np.maximum(eff_sigma_conf, 1e-12)) ** 2)))
        sp_r, sp_p = _safe_spearman(w_j, abs_diff_j)

        per_param[names[j]] = {
            "repeatability_coverage": cov_pt,
            "repeatability_coverage_bca_ci": [cov_lo, cov_hi],
            "sigma_repeat_over_width": rat_pt,
            "sigma_repeat_over_width_bca_ci": [rat_lo, rat_hi],
            "standardized_residual_rms": std_rms,
            "spearman_r": sp_r,
            "spearman_p": sp_p,
        }
        cov_list.append(cov_pt)
        ratio_list.append(rat_pt)
        rms_list.append(std_rms)
        spearman_r_list.append(sp_r)
        spearman_p_list.append(sp_p)

    # Primary/first parameter BCa CI if 1D, else voxel-averaged indicators across parameters
    if n_par == 1:
        p0 = per_param[names[0]]
        agg_cov_ci = (float(p0["repeatability_coverage_bca_ci"][0]), float(p0["repeatability_coverage_bca_ci"][1]))
        agg_rat_ci = (float(p0["sigma_repeat_over_width_bca_ci"][0]), float(p0["sigma_repeat_over_width_bca_ci"][1]))
    else:
        voxel_cov = np.mean((p2 >= lo) & (p2 <= hi), axis=1).astype(np.float64)
        _, c_lo, c_hi = bca_bootstrap_ci(voxel_cov, np.mean, n_boot=n_boot, seed=seed_int + 99)
        agg_cov_ci = (c_lo, c_hi)
        agg_rat_ci = (
            float(np.mean([per_param[nm]["sigma_repeat_over_width_bca_ci"][0] for nm in names])),
            float(np.mean([per_param[nm]["sigma_repeat_over_width_bca_ci"][1] for nm in names])),
        )

    return LetheRepeatabilityReport(
        n_voxels=n_vox,
        n_parameters=n_par,
        seed=seed_int,
        repeatability_coverage=float(np.mean(cov_list)),
        repeatability_coverage_bca_ci=agg_cov_ci,
        sigma_repeat_over_width=float(np.mean(ratio_list)),
        sigma_repeat_over_width_bca_ci=agg_rat_ci,
        standardized_residual_rms=float(np.mean(rms_list)),
        spearman_r=float(np.mean(spearman_r_list)),
        spearman_p=float(np.min(spearman_p_list)),
        per_parameter=per_param,
    )


def _fast_segmented_df_fit(signals: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fast vectorized two-step log-linear IVIM point estimator returning (D, Dstar, f)."""
    s = np.maximum(np.asarray(signals, dtype=np.float64), 1e-6)
    s0 = s[:, 0:1]
    hi_m = b >= 200.0
    b_hi = b[hi_m]
    y_hi = np.log(s[:, hi_m] / s0)
    b_mean = float(np.mean(b_hi))
    denom = float(np.sum((b_hi - b_mean) ** 2))
    d_hat = -np.sum((b_hi[None, :] - b_mean) * (y_hi - np.mean(y_hi, axis=1, keepdims=True)), axis=1) / denom
    d_hat = np.clip(d_hat, 1.0e-4, 4.0e-3)
    ln_a = np.mean(y_hi, axis=1) + d_hat * b_mean
    f_hat = np.clip(1.0 - np.exp(ln_a), 0.01, 0.65)

    low_m = (b > 0.0) & (b <= 80.0)
    b_lo = b[low_m]
    rem = (s[:, low_m] / s0 - (1.0 - f_hat[:, None]) * np.exp(-b_lo[None, :] * d_hat[:, None])) / np.maximum(
        f_hat[:, None], 1e-3
    )
    rem_clip = np.clip(rem, 0.05, 0.99)
    ds_hat = np.median(-np.log(rem_clip) / b_lo[None, :], axis=1)
    ds_hat = np.clip(ds_hat, 3.0e-3, 0.15)
    return np.column_stack([d_hat, ds_hat, f_hat])


def run_reverb_precision_coverage_counterexample(
    n_voxels: int = 1000,
    seed: int = 42,
    *,
    alpha: float = 0.10,
    snr: float = 28.0,
) -> dict[str, Any]:
    """Prove how exact (1 - alpha) ground-truth conformal coverage under-covers scan-rescan repeatability.

    Under independent scan-1 and scan-2 noise realizations, :math:`\\hat{\\theta}_2 - \\hat{\\theta}_1`
    carries variance :math:`2\\sigma^2` relative to single-scan estimation error variance :math:`\\sigma^2`,
    so a centered :math:`(1 - \\alpha) = 90\\%` truth interval covers a repeat point estimate with
    theoretical Gaussian probability :math:`2\\Phi(z_{1-\\alpha/2}/\\sqrt{2}) - 1 \\approx 0.755 < 0.90`.
    """
    n_vox = int(n_voxels)
    if n_vox < 20:
        raise ValueError(f"`n_voxels` must be >= 20; got {n_vox}.")
    seed_int = int(seed)
    rng = np.random.Generator(np.random.PCG64(seed_int))

    n_cal = max(400, n_vox // 2)
    b = CLINICAL_B.copy()

    d_cal = rng.uniform(0.7e-3, 1.9e-3, size=n_cal)
    ds_cal = rng.uniform(15.0e-3, 75.0e-3, size=n_cal)
    f_cal = rng.uniform(0.08, 0.34, size=n_cal)

    d_te = rng.uniform(0.7e-3, 1.9e-3, size=n_vox)
    ds_te = rng.uniform(15.0e-3, 75.0e-3, size=n_vox)
    f_te = rng.uniform(0.08, 0.34, size=n_vox)

    sig_cal = simulate_ivim_signal(b, D=d_cal, Dstar=ds_cal, f=f_cal, snr=snr, noise="rician", rng=rng)
    sig_scan1 = simulate_ivim_signal(b, D=d_te, Dstar=ds_te, f=f_te, snr=snr, noise="rician", rng=rng)
    sig_scan2 = simulate_ivim_signal(b, D=d_te, Dstar=ds_te, f=f_te, snr=snr, noise="rician", rng=rng)

    hat_cal = _fast_segmented_df_fit(sig_cal, b)
    hat_scan1 = _fast_segmented_df_fit(sig_scan1, b)
    hat_scan2 = _fast_segmented_df_fit(sig_scan2, b)

    y_cal = np.column_stack([d_cal, ds_cal, f_cal])
    y_te = np.column_stack([d_te, ds_te, f_te])

    # Evaluate on slow diffusivity D (well-identified, nearly unbiased Gaussian-like error)
    lo_d, hi_d, q_d = split_conformal(hat_cal[:, 0], y_cal[:, 0], hat_scan1[:, 0], float(alpha))
    truth_cov_d = float(np.mean((y_te[:, 0] >= lo_d) & (y_te[:, 0] <= hi_d)))

    lethe_rep = evaluate_echo_repeatability_scale(
        (lo_d, hi_d),
        hat_scan1[:, 0],
        hat_scan2[:, 0],
        seed=seed_int,
        alpha=float(alpha),
        n_boot=300,
        param_names=("D",),
    )
    retest_cov_d = float(lethe_rep.repeatability_coverage)
    z_crit = float(norm.ppf(1.0 - float(alpha) / 2.0))
    theoretical_gaussian_retest_cov = float(2.0 * norm.cdf(z_crit / np.sqrt(2.0)) - 1.0)

    return {
        "n_voxels": n_vox,
        "seed": seed_int,
        "alpha": float(alpha),
        "nominal_truth_coverage": 1.0 - float(alpha),
        "empirical_truth_coverage": truth_cov_d,
        "empirical_scan_rescan_coverage": retest_cov_d,
        "theoretical_gaussian_retest_coverage": theoretical_gaussian_retest_cov,
        "repeatability_coverage_deficit": float(truth_cov_d - retest_cov_d),
        "conformal_halfwidth_D": float(q_d),
        "counterexample_verified": bool(
            (truth_cov_d >= 1.0 - float(alpha) - 0.04) and (truth_cov_d - retest_cov_d >= 0.08)
        ),
        "lethe_report": lethe_rep.to_dict(),
    }
