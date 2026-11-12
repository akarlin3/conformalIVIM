"""Single-voxel Out-of-Distribution (OOD) and repeat-scan detector for IVIM MRI.

Complements the cohort-level two-sample :class:`~conformal_ivim.monitor.DeploymentValidityMonitor`
by computing localized **per-voxel** nonconformity OOD scores from unlabeled DWI
decay curves and calibrating exact finite-sample conformal OOD p-values:

    p_{OOD, i} = (1 + sum_{j=1}^{n_cal} 1{ T_cal,j >= T_test,i }) / (n_cal + 1)

Under exchangeability with the calibration reference distribution, the per-voxel
false-alarm rate satisfies ``P(p_{OOD, i} <= alpha_ood) <= alpha_ood`` in finite
samples without distributional assumptions.

Three complementary observable anomaly axes are combined per voxel:
  1. **Signal-space bi-exponential projection residuals**: noise-standardized and
     relative L2 projection residuals catching RF/motion spikes, susceptibility
     dropouts, and multi-compartment model breakdown.
  2. **Physical monotonicity & positivity violations**: negative/zero signal
     dropouts, non-physical signal rebounds across increasing ``b``-values beyond
     the Rician noise floor, and necrotic free-fluid saturation.
  3. **9-feature summary-space k-NN & Mahalanobis leverage**: local Euclidean
     k-nearest-neighbor distance (``scipy.spatial.cKDTree``) and regularized
     Mahalanobis leverage in the 9-dimensional observable summary space.

Flagged voxels receive conservative interval adjustments alongside explicit
finite-sample marginal and selective coverage guarantees.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from conformal_ivim.conformal import ConformalIVIM
from conformal_ivim.estimators import PARAM_NAMES
from conformal_ivim.forward import DEFAULT_B_VALUES, D_FREE_FLUID
from conformal_ivim.monitor import extract_observable_features

DEFAULT_PHYS_BOUNDS: Tuple[Tuple[float, float], ...] = (
    (0.1e-3, 4.0e-3),
    (3.0e-3, 300.0e-3),
    (0.0, 0.70),
)


def _robust_location_scale(x: np.ndarray, floor: float = 1e-6) -> Tuple[float, float]:
    """Compute median and robust dispersion scale (MAD / IQR / SD fallback)."""
    arr = np.asarray(x, dtype=float).ravel()
    med = float(np.median(arr))
    mad = 1.4826 * float(np.median(np.abs(arr - med)))
    q75, q25 = np.percentile(arr, [75.0, 25.0])
    iqr_sd = float(q75 - q25) / 1.349
    sd = float(np.std(arr))
    scale = max(mad, iqr_sd, sd, float(floor))
    return med, scale


def _physical_violation_score(
    signals: np.ndarray,
    b: np.ndarray,
    theta_hat: np.ndarray,
    sigma_hat: np.ndarray,
) -> np.ndarray:
    """Compute per-voxel physical monotonicity, positivity, and dropout violations."""
    sig = np.atleast_2d(np.asarray(signals, dtype=float))
    b_arr = np.asarray(b, dtype=float)
    N = sig.shape[0]

    # Group and sort unique b-values to inspect physical signal attenuation order
    uniq_b = np.unique(b_arr)
    sig_sorted = np.column_stack(
        [np.mean(sig[:, b_arr == bv], axis=1) for bv in uniq_b]
    )
    s0_obs = np.maximum(sig_sorted[:, 0], 1e-6)
    s_norm = sig_sorted / s0_obs[:, None]
    noise_rel = np.clip(sigma_hat / s0_obs, 1e-4, 1.0)

    # 1) Positivity & catastrophic susceptibility/RF signal dropout
    neg_violation = np.maximum(0.0, -np.min(sig, axis=1)) / np.maximum(s0_obs, 1.0)
    b0_collapse = np.maximum(0.0, 0.05 - sig_sorted[:, 0]) * 10.0

    low_mid_mask = (uniq_b > 0.0) & (uniq_b <= 300.0)
    if np.any(low_mid_mask):
        # Complete susceptibility dropout at low/mid b where parenchyma cannot have decayed to 0
        expected_min = 0.20 * np.exp(-uniq_b[low_mid_mask] * 3.2e-3)
        dropout_gap = np.maximum(0.0, expected_min[None, :] - s_norm[:, low_mid_mask])
        dropout_score = np.sum(dropout_gap, axis=1) * 5.0
    else:
        dropout_score = np.zeros(N, dtype=float)

    # 2) Non-physical signal rebounds across increasing b-values beyond Rician fluctuations
    if uniq_b.size >= 2:
        diffs = s_norm[:, 1:] - s_norm[:, :-1]
        rebound_excess = np.maximum(0.0, diffs - 2.2 * noise_rel[:, None])
        max_jump = np.max(np.maximum(0.0, diffs - 3.0 * noise_rel[:, None]), axis=1)
        rebound_score = np.sum(rebound_excess, axis=1) * 4.0 + max_jump * 6.0
    else:
        rebound_score = np.zeros(N, dtype=float)

    # 3) Extreme motion/RF intensity spike (b > 0 signal exceeding b = 0 baseline)
    if uniq_b.size >= 2:
        spike_excess = np.maximum(
            0.0, np.max(s_norm[:, 1:], axis=1) - (1.0 + 2.5 * noise_rel)
        )
        spike_score = spike_excess * 8.0
    else:
        spike_score = np.zeros(N, dtype=float)

    # 4) Necrotic / free-fluid unphysical regime (D >= D_free_fluid and vanishing perfusion)
    d_hat = theta_hat[:, 0]
    f_hat = theta_hat[:, 2]
    necrotic_excess = np.maximum(0.0, (d_hat - 0.98 * D_FREE_FLUID) / 1e-3) * np.where(
        f_hat < 0.03, 3.0, 1.0
    )

    return (
        neg_violation * 10.0
        + b0_collapse
        + dropout_score
        + rebound_score
        + spike_score
        + necrotic_excess
    )


class VoxelOODDetector:
    """Single-voxel conformal Out-of-Distribution (OOD) and repeat-scan detector.

    Parameters
    ----------
    alpha_ood : float, default=0.05
        Target per-voxel false-positive (Type-I error) rate in ``(0, 1)``. Voxels
        with conformal p-value ``p_ood <= alpha_ood`` are flagged in ``ood_mask``.
    b : sequence of float, default=DEFAULT_B_VALUES
        Diffusion-weighting b-values in ``s/mm^2``.
    n_neighbors : int, default=10
        Number of nearest neighbors ``k`` in standardized 9-feature summary space.
    weights : tuple of (float, float, float), default=(0.40, 0.25, 0.35)
        Relative weights ``(w_residual, w_physical, w_leverage)`` combining
        bi-exponential projection residuals, physical monotonicity/positivity
        violations, and 9-feature k-NN / Mahalanobis leverage.
    phys_bounds : tuple of (low, high) pairs, default=DEFAULT_PHYS_BOUNDS
        Physiological support bounds ``((D_lo, D_hi), (Dstar_lo, Dstar_hi), (f_lo, f_hi))``
        used to conservatively widen intervals on flagged OOD voxels.
    ood_action : {'widen_to_bounds', 'inflate', 'abstain'}, default='widen_to_bounds'
        Interval adjustment policy applied to voxels with ``ood_mask[i] == True``:
        - ``'widen_to_bounds'``: widens flagged voxel intervals to ``phys_bounds``,
          guaranteeing 100% coverage for any bounded parameter inside physical support.
        - ``'inflate'``: inflates flagged voxel interval half-widths by ``inflate_factor``.
        - ``'abstain'``: sets flagged voxel bounds to ``(-inf, +inf)`` (explicit abstention).
    inflate_factor : float, default=2.5
        Half-width multiplier used when ``ood_action='inflate'``.
    repeat_scan_threshold : float, default=0.20
        Corrupted voxel fraction above which ``repeat_scan`` is flagged ``True``.
    fdr_control : bool, default=False
        If ``True``, applies Benjamini–Hochberg conformal outlier FDR control at
        level ``alpha_ood`` instead of marginal per-voxel thresholding.
    seed : int, default=0
        Deterministic ``PCG64`` seed.
    """

    def __init__(
        self,
        alpha_ood: float = 0.05,
        *,
        alpha: Optional[float] = None,
        b: Sequence[float] = DEFAULT_B_VALUES,
        n_neighbors: int = 10,
        weights: Tuple[float, float, float] = (0.40, 0.25, 0.35),
        phys_bounds: Sequence[Tuple[float, float]] = DEFAULT_PHYS_BOUNDS,
        ood_action: str = "widen_to_bounds",
        inflate_factor: float = 2.5,
        repeat_scan_threshold: float = 0.20,
        fdr_control: bool = False,
        seed: int = 0,
    ) -> None:
        eff_alpha = float(alpha) if alpha is not None else float(alpha_ood)
        if not (0.0 < eff_alpha < 1.0):
            raise ValueError(f"alpha_ood must lie strictly in (0, 1), got {eff_alpha!r}")
        if int(n_neighbors) < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors!r}")
        if ood_action not in ("widen_to_bounds", "inflate", "abstain"):
            raise ValueError(
                f"ood_action must be 'widen_to_bounds', 'inflate', or 'abstain', got {ood_action!r}"
            )
        w = np.asarray(weights, dtype=float)
        if w.shape != (3,) or np.any(w < 0.0) or float(np.sum(w)) <= 0.0:
            raise ValueError(f"weights must be 3 non-negative numbers with positive sum, got {weights!r}")

        self.alpha_ood = eff_alpha
        self.alpha = eff_alpha
        self.b = np.asarray(b, dtype=float)
        self.n_neighbors = int(n_neighbors)
        self.weights = (float(w[0]), float(w[1]), float(w[2]))
        self.phys_bounds = tuple((float(lo), float(hi)) for lo, hi in phys_bounds)
        self.ood_action = ood_action
        self.inflate_factor = float(inflate_factor)
        self.repeat_scan_threshold = float(repeat_scan_threshold)
        self.fdr_control = bool(fdr_control)
        self.seed = int(seed)

    def _raw_components_from_signals(
        self,
        signals: np.ndarray,
        resid: Optional[np.ndarray] = None,
        *,
        b: Optional[Sequence[float]] = None,
        is_calibration: bool = False,
    ) -> Dict[str, np.ndarray]:
        b_arr = self.b if b is None else np.asarray(b, dtype=float)
        if resid is not None:
            feat_raw = np.atleast_2d(np.asarray(signals, dtype=float))
            res_raw = np.asarray(resid, dtype=float).ravel()
            nonfinite_mask = (~np.all(np.isfinite(feat_raw), axis=1)) | (~np.isfinite(res_raw))
            feat9 = np.where(np.isfinite(feat_raw), feat_raw, 0.0)
            resid_norm = np.where(np.isfinite(res_raw), res_raw, 0.0)
            sigma_hat = (
                np.exp(np.clip(feat9[:, -1], -20.0, 20.0))
                if feat9.shape[1] >= 9
                else np.ones(feat9.shape[0], dtype=float)
            )
            comp_resid = resid_norm / np.maximum(sigma_hat, 1e-6)
            comp_phys = np.zeros(feat9.shape[0], dtype=float)
            theta_hat = np.zeros((feat9.shape[0], 3), dtype=float)
        else:
            sig = np.atleast_2d(np.asarray(signals, dtype=float))
            if sig.shape[1] != b_arr.size:
                raise ValueError(
                    f"Expected signal curves with {b_arr.size} b-values, got shape {sig.shape}."
                )
            # Sanitize non-finite inputs before NLLS feature extraction while recording corruption
            nonfinite_mask = ~np.all(np.isfinite(sig), axis=1)
            sig_clean = np.where(np.isfinite(sig), sig, 0.0)

            theta_hat, feat9, resid_norm = extract_observable_features(sig_clean, b=b_arr)
            sigma_hat = np.exp(np.clip(feat9[:, -1], -20.0, 20.0))
            sig_l2 = np.maximum(np.linalg.norm(sig_clean, axis=1), 1e-6)

            # Component 1: Signal-space bi-exponential projection residual
            std_resid = resid_norm / np.maximum(sigma_hat, 1e-6)
            rel_resid = resid_norm / sig_l2
            comp_resid = std_resid + 4.0 * rel_resid

            # Component 2: Physical monotonicity, positivity & dropout violations
            comp_phys = _physical_violation_score(sig_clean, b_arr, theta_hat, sigma_hat)

        if is_calibration:
            self.feat_mu_ = feat9.mean(axis=0)
            self.feat_sd_ = feat9.std(axis=0) + 1e-8
            Z = (feat9 - self.feat_mu_) / self.feat_sd_
            cov = np.atleast_2d(np.cov(Z, rowvar=False)) + np.eye(Z.shape[1]) * 1e-5
            self.inv_cov_ = np.linalg.inv(cov)
            self.Z_cal_ = Z
            self.kdtree_ = cKDTree(Z)

            # LOO Mahalanobis distance via Sherman-Morrison
            d2_in = np.einsum("ni,ij,nj->n", Z, self.inv_cov_, Z)
            n = Z.shape[0]
            if n > Z.shape[1] + 3:
                nm1_sq = float((n - 1) * (n - 1))
                denom = np.clip(1.0 - (float(n) / nm1_sq) * d2_in, 1e-3, None)
                d2_loo = (float(n * (n - 2)) / nm1_sq) * (d2_in / denom)
                maha = np.sqrt(np.maximum(d2_loo, 0.0))
            else:
                maha = np.sqrt(np.maximum(d2_in, 0.0))

            # LOO k-NN distance (drop self at index 0)
            k_eff = min(self.n_neighbors, max(1, n - 1))
            dists, _ = self.kdtree_.query(Z, k=k_eff + 1)
            dists_2d = np.atleast_2d(dists)
            knn_dist = np.mean(dists_2d[:, 1 : k_eff + 1], axis=1)
        else:
            Z = (feat9 - self.feat_mu_) / self.feat_sd_
            d2 = np.einsum("ni,ij,nj->n", Z, self.inv_cov_, Z)
            maha = np.sqrt(np.maximum(d2, 0.0))

            k_eff = min(self.n_neighbors, max(1, self.Z_cal_.shape[0]))
            dists, _ = self.kdtree_.query(Z, k=k_eff)
            if k_eff == 1:
                knn_dist = np.asarray(dists, dtype=float).ravel()
            else:
                knn_dist = np.mean(np.atleast_2d(dists), axis=1)

        if np.any(nonfinite_mask):
            comp_resid = np.where(nonfinite_mask, 1e6, comp_resid)
            comp_phys = np.where(nonfinite_mask, 1e6, comp_phys)
            maha = np.where(nonfinite_mask, 1e6, maha)
            knn_dist = np.where(nonfinite_mask, 1e6, knn_dist)

        return {
            "residual": comp_resid,
            "physical": comp_phys,
            "mahalanobis": maha,
            "knn": knn_dist,
            "features": feat9,
            "theta_hat": theta_hat,
        }

    def _combine_components(self, raw: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        z_res = np.maximum(0.0, (raw["residual"] - self.res_med_) / self.res_scale_)
        # Physical violations are near zero on clean inliers; preserve sensitivity when MAD is tiny
        z_phys = np.maximum(
            0.0,
            (raw["physical"] - self.phys_med_) / max(self.phys_scale_, 0.15),
        )
        z_maha = np.maximum(0.0, (raw["mahalanobis"] - self.maha_med_) / self.maha_scale_)
        z_knn = np.maximum(0.0, (raw["knn"] - self.knn_med_) / self.knn_scale_)
        z_lev = 0.5 * (z_maha + z_knn)

        w_res, w_phys, w_lev = self.weights
        w_sum = w_res + w_phys + w_lev
        composite = (w_res * z_res + w_phys * z_phys + w_lev * z_lev) / w_sum
        standardized = {
            "residual": z_res,
            "physical": z_phys,
            "leverage": z_lev,
            "mahalanobis": z_maha,
            "knn": z_knn,
        }
        return composite, standardized

    def fit(
        self,
        cal_signals: np.ndarray,
        cal_resid: Optional[np.ndarray] = None,
        *,
        b: Optional[Sequence[float]] = None,
    ) -> "VoxelOODDetector":
        """Calibrate reference per-voxel OOD score distribution on ``cal_signals``."""
        if b is not None:
            self.b = np.asarray(b, dtype=float)
        raw = self._raw_components_from_signals(
            cal_signals, cal_resid, b=self.b, is_calibration=True
        )
        self.res_med_, self.res_scale_ = _robust_location_scale(raw["residual"], floor=1e-4)
        self.phys_med_, self.phys_scale_ = _robust_location_scale(raw["physical"], floor=0.10)
        self.maha_med_, self.maha_scale_ = _robust_location_scale(raw["mahalanobis"], floor=1e-4)
        self.knn_med_, self.knn_scale_ = _robust_location_scale(raw["knn"], floor=1e-4)

        cal_scores, cal_std = self._combine_components(raw)
        self.cal_scores_ = cal_scores
        self.cal_scores_sorted_ = np.sort(cal_scores)
        self.cal_components_ = cal_std
        self.n_cal_ = int(cal_scores.size)

        # Finite-sample (1 - alpha_ood) conformal OOD threshold
        k = int(np.ceil((self.n_cal_ + 1) * (1.0 - self.alpha_ood)))
        if k > self.n_cal_:
            self.threshold_ = float("inf")
        else:
            self.threshold_ = float(self.cal_scores_sorted_[max(0, k - 1)])
        return self

    calibrate = fit

    def score_voxels(
        self,
        signals: np.ndarray,
        test_resid: Optional[np.ndarray] = None,
        *,
        b: Optional[Sequence[float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Compute composite OOD scores and standardized component scores per voxel."""
        if not hasattr(self, "cal_scores_sorted_"):
            raise RuntimeError(
                "VoxelOODDetector must be fitted via .fit(cal_signals) before scoring."
            )
        raw = self._raw_components_from_signals(
            signals, test_resid, b=b, is_calibration=False
        )
        return self._combine_components(raw)

    def predict_ood(
        self,
        signals: np.ndarray,
        test_resid: Optional[np.ndarray] = None,
        *,
        b: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Compute conformal OOD p-values, boolean mask, and flagged voxel fraction."""
        scores, comp_scores = self.score_voxels(signals, test_resid=test_resid, b=b)
        p_vals = self._p_values_from_scores(scores)
        ood_mask = self._flag_from_p_values(p_vals)
        ood_frac = float(np.mean(ood_mask)) if ood_mask.size else 0.0
        return {
            "ood_mask": ood_mask,
            "p_values": p_vals,
            "scores": scores,
            "ood_scores": scores,
            "component_scores": comp_scores,
            "threshold": self.threshold_,
            "flagged_fraction": ood_frac,
            "ood_fraction": ood_frac,
        }

    def conformal_p_values(self, signals: np.ndarray) -> np.ndarray:
        """Compute exact finite-sample conformal p-values ``p_{OOD, i}`` per voxel."""
        scores, _ = self.score_voxels(signals)
        return self._p_values_from_scores(scores)

    def _p_values_from_scores(self, scores: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.cal_scores_sorted_, scores, side="left")
        n_geq = self.n_cal_ - idx
        return (1.0 + n_geq.astype(float)) / float(self.n_cal_ + 1)

    def _flag_from_p_values(self, p_values: np.ndarray) -> np.ndarray:
        if not self.fdr_control:
            return p_values <= self.alpha_ood
        # Benjamini-Hochberg conformal outlier procedure (Bates et al., 2023)
        m = p_values.size
        if m == 0:
            return np.zeros(0, dtype=bool)
        order = np.argsort(p_values, kind="mergesort")
        p_sorted = p_values[order]
        crit = self.alpha_ood * np.arange(1, m + 1, dtype=float) / float(m)
        passing = np.where(p_sorted <= crit)[0]
        mask = np.zeros(m, dtype=bool)
        if passing.size > 0:
            max_idx = int(passing[-1])
            mask[order[: max_idx + 1]] = True
        return mask

    def adjust_intervals(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        ood_mask: np.ndarray,
        *,
        nominal_alpha: float = 0.10,
    ) -> Dict[str, Any]:
        """Adjust prediction intervals on flagged OOD voxels and report coverage guarantees."""
        lo = np.asarray(lower, dtype=float).copy()
        hi = np.asarray(upper, dtype=float).copy()
        mask = np.asarray(ood_mask, dtype=bool).ravel()

        squeeze = lo.ndim == 1
        if squeeze:
            lo = lo[:, None]
            hi = hi[:, None]

        if lo.shape[0] != mask.size:
            raise ValueError(
                f"Interval batch size ({lo.shape[0]}) != ood_mask length ({mask.size})."
            )
        P = lo.shape[1]

        if np.any(mask):
            if self.ood_action == "widen_to_bounds":
                for j in range(P):
                    b_lo, b_hi = self.phys_bounds[min(j, len(self.phys_bounds) - 1)]
                    lo[mask, j] = np.minimum(lo[mask, j], b_lo)
                    hi[mask, j] = np.maximum(hi[mask, j], b_hi)
            elif self.ood_action == "inflate":
                mid = 0.5 * (lo[mask] + hi[mask])
                half = 0.5 * (hi[mask] - lo[mask]) * self.inflate_factor
                lo[mask] = mid - half
                hi[mask] = mid + half
            else:  # "abstain"
                lo[mask] = -np.inf
                hi[mask] = np.inf

        # Finite-sample theoretical coverage lower bounds:
        # - Under widen_to_bounds or abstain (-inf, +inf), any bounded physiological parameter
        #   on an OOD-flagged voxel is covered with probability 1, while exchangeable inliers
        #   satisfy P(Y in C_base(X)) >= 1 - nominal_alpha => marginal bound >= 1 - nominal_alpha.
        # - Selective inlier-only reporting without widening carries union-bound slack alpha_ood.
        selective_bound = max(0.0, 1.0 - float(nominal_alpha) - self.alpha_ood)
        if self.ood_action in ("widen_to_bounds", "abstain"):
            adjusted_marginal_bound = 1.0 - float(nominal_alpha)
        else:
            adjusted_marginal_bound = selective_bound

        lo_out = lo[:, 0] if squeeze else lo
        hi_out = hi[:, 0] if squeeze else hi
        return {
            "lower": lo_out,
            "upper": hi_out,
            "width": hi_out - lo_out,
            "ood_action": self.ood_action,
            "nominal_coverage": 1.0 - float(nominal_alpha),
            "alpha_ood": self.alpha_ood,
            "adjusted_marginal_coverage_bound": float(adjusted_marginal_bound),
            "selective_inlier_coverage_bound": float(selective_bound),
            "n_adjusted": int(np.sum(mask)),
        }

    def detect(
        self,
        signals: np.ndarray,
        *,
        conformal_model: Optional[ConformalIVIM] = None,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        nominal_alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Score voxels, compute conformal OOD p-values, flag corrupted voxels, and adjust intervals.

        Parameters
        ----------
        signals : array-like of shape (N, n_b)
            Unlabeled test DWI signal curves.
        conformal_model : ConformalIVIM or None, optional
            Optional calibrated ``ConformalIVIM`` instance used to predict baseline
            intervals before applying conservative OOD interval adjustments.
        lower, upper : np.ndarray or None, optional
            Optional pre-computed interval arrays of shape ``(N, 3)`` (or ``(N,)``).
        nominal_alpha : float or None, optional
            Nominal miscoverage rate ``alpha`` of the supplied intervals (inferred from
            ``conformal_model.alpha`` if available, else ``0.10``).
        """
        scores, comp_scores = self.score_voxels(signals)
        p_ood = self._p_values_from_scores(scores)
        ood_mask = self._flag_from_p_values(p_ood)
        ood_frac = float(np.mean(ood_mask)) if ood_mask.size > 0 else 0.0

        alpha_int = (
            float(nominal_alpha)
            if nominal_alpha is not None
            else (float(conformal_model.alpha) if conformal_model is not None else 0.10)
        )

        if conformal_model is not None and (lower is None or upper is None):
            lower, upper = conformal_model.predict_interval(signals, alpha=alpha_int)

        if lower is not None and upper is not None:
            adj = self.adjust_intervals(
                lower, upper, ood_mask, nominal_alpha=alpha_int
            )
        else:
            selective_bound = max(0.0, 1.0 - alpha_int - self.alpha_ood)
            adj_bound = (
                1.0 - alpha_int
                if self.ood_action in ("widen_to_bounds", "abstain")
                else selective_bound
            )
            adj = {
                "lower": None,
                "upper": None,
                "width": None,
                "ood_action": self.ood_action,
                "nominal_coverage": 1.0 - alpha_int,
                "alpha_ood": self.alpha_ood,
                "adjusted_marginal_coverage_bound": float(adj_bound),
                "selective_inlier_coverage_bound": float(selective_bound),
                "n_adjusted": int(np.sum(ood_mask)),
            }

        out: Dict[str, Any] = {
            "ood_scores": scores,
            "component_scores": comp_scores,
            "p_values": p_ood,
            "p_ood": p_ood,
            "ood_mask": ood_mask,
            "inlier_mask": ~ood_mask,
            "ood_fraction": ood_frac,
            "threshold": self.threshold_,
            "alpha_ood": self.alpha_ood,
            "repeat_scan": bool(ood_frac > self.repeat_scan_threshold),
            "adjusted_intervals": adj,
            "guarantees": {
                "finite_sample_fpr_bound": self.alpha_ood,
                "min_possible_p_value": 1.0 / float(self.n_cal_ + 1),
                "adjusted_marginal_coverage_bound": adj["adjusted_marginal_coverage_bound"],
                "selective_inlier_coverage_bound": adj["selective_inlier_coverage_bound"],
                "n_cal": self.n_cal_,
            },
        }
        if adj["lower"] is not None and np.ndim(adj["lower"]) == 2:
            for j, nm in enumerate(PARAM_NAMES[: adj["lower"].shape[1]]):
                out[nm] = (adj["lower"][:, j], adj["upper"][:, j])
        return out

    evaluate = detect
