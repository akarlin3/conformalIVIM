"""Label-free deployment-validity monitor for conformal IVIM parameter mapping.

Combines two observable two-sample shift-detection families on unlabeled test
batches (never accessing ground-truth IVIM parameters):

  * **Family 1 -- Summary-Space Mahalanobis & Hotelling T^2**: fits the
    calibration distribution of observable summary features (model-free decay
    slopes/curvature + plug-in NLLS parameters + noise scale) and tests
    deployment batches via per-voxel Mahalanobis distances, FPR-controlled batch
    mean subsampling, and multivariate two-sample Hotelling T^2.
  * **Family 2 -- Signal-Space Residual Conformal**: fits the calibration
    distribution of bi-exponential NLLS residual norms ``||S - S_hat||_2`` (and
    noise-standardized residuals ``||S - S_hat||_2 / sigma_hat``) to detect
    forward-model misspecification and acquisition/noise shifts.

By construction, the monitor fires on *observable* covariate, SNR, noise-model,
and multi-compartment shifts before conformal coverage silently degrades, while
remaining blind to purely *latent* within-distribution subgroup shifts where X
is exchangeable between calibration and deployment.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import chi2, ncx2, norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from conformal_ivim.estimators import nlls_fit_and_noise
from conformal_ivim.forward import DEFAULT_B_VALUES, ivim_signal


def signal_shape_features(signals, b=DEFAULT_B_VALUES) -> np.ndarray:
    """Compute 4 model-free observable signal-shape features from magnitude curves.

    Returns an ``(N, 4)`` array with columns
    ``(slope_low, slope_high, curvature, early_drop)``.
    """
    signals = np.atleast_2d(np.asarray(signals, dtype=float))
    b = np.asarray(b, dtype=float)
    s0 = signals[:, int(np.argmin(b))]
    s0 = np.where(s0 > 0, s0, signals.max(1))
    eps = 1e-6
    logs = np.log(np.clip(signals, eps, None))

    def _slope(mask):
        bb = b[mask]
        A = np.column_stack([bb, np.ones_like(bb)])
        coef, *_ = np.linalg.lstsq(A, logs[:, mask].T, rcond=None)
        return -coef[0]

    low = b <= 60.0
    high = b >= 200.0
    slope_low = _slope(low) if low.sum() >= 2 else np.zeros(signals.shape[0])
    slope_high = _slope(high) if high.sum() >= 2 else np.zeros(signals.shape[0])
    j30 = int(np.argmin(np.abs(b - 30.0)))
    early_drop = 1.0 - signals[:, j30] / np.clip(s0, eps, None)
    curvature = slope_low - slope_high
    return np.column_stack([slope_low, slope_high, curvature, early_drop])


def extract_observable_features(
    signals, b=DEFAULT_B_VALUES
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract label-free observable summary features and NLLS fit-residual norms.

    Returns ``(theta_hat, features, resid_norm)`` where:
      - ``theta_hat`` has shape ``(N, 3)`` with plug-in ``(D_hat, Dstar_hat, f_hat)``,
      - ``features`` has shape ``(N, 9)`` stacking the 4 signal-shape features,
        plug-in ``(D_hat, Dstar_hat, f_hat)``, estimated ``SNR_hat``, and
        ``log(sigma_hat)``,
      - ``resid_norm`` has shape ``(N,)`` with ``||S - S_hat||_2``.
    """
    signals = np.atleast_2d(np.asarray(signals, dtype=float))
    b_arr = np.asarray(b, dtype=float)
    theta, s0, sigma = nlls_fit_and_noise(signals, b_arr)
    model = ivim_signal(
        b_arr[None, :],
        theta[:, 0:1],
        theta[:, 1:2],
        theta[:, 2:3],
        S0=s0[:, None],
    )
    resid_norm = np.linalg.norm(signals - model, axis=1)
    shape = signal_shape_features(signals, b_arr)
    snr_hat = s0 / np.clip(sigma, 1e-6, None)
    feat = np.column_stack(
        [
            shape,
            theta[:, 0],
            theta[:, 1],
            theta[:, 2],
            snr_hat,
            np.log(np.clip(sigma, 1e-6, None)),
        ]
    )
    return theta, feat, resid_norm


def estimate_importance_weights(
    feat_cal,
    feat_test,
    seed: int = 0,
    clip: Tuple[float, float] = (1e-2, 1e2),
) -> Tuple[np.ndarray, np.ndarray, LogisticRegression]:
    """Estimate likelihood-ratio importance weights ``dP_test / dP_cal`` on
    observable features via a balanced logistic domain classifier.
    """
    feat_cal = np.atleast_2d(np.asarray(feat_cal, dtype=float))
    feat_test = np.atleast_2d(np.asarray(feat_test, dtype=float))
    X = np.vstack([feat_cal, feat_test])
    y = np.r_[np.zeros(len(feat_cal)), np.ones(len(feat_test))]
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xs = (X - mu) / sd
    clf = LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced", random_state=int(seed)
    )
    clf.fit(Xs, y)
    p = clf.predict_proba(Xs)[:, 1]
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    w = np.clip(p / (1.0 - p), clip[0], clip[1])
    return w[: len(feat_cal)], w[len(feat_cal) :], clf


class DeploymentValidityMonitor:
    """Two-sample aggregate shift detector combining Family-1 summary Mahalanobis /
    Hotelling T^2 testing and Family-2 residual-conformal testing to flag
    exchangeability breaks on unlabeled IVIM deployment batches.

    Supports two input modes seamlessly:
      1. **Precomputed observables**: ``fit(cal_feat, cal_resid)`` and
         ``evaluate(test_feat, test_resid)``.
      2. **Raw unlabeled DWI signals**: ``fit(cal_signals, b=b_values)`` (or
         ``fit_signals``) and ``evaluate(test_signals)`` (or ``evaluate_signals``),
         automatically extracting observable summary features and NLLS residual norms.
    """

    def __init__(
        self,
        fpr: float = 0.05,
        n_null: int = 400,
        null_subsample: Optional[int] = None,
        seed: int = 0,
    ):
        self.fpr = float(fpr)
        self.n_null = int(n_null)
        self.null_subsample = null_subsample
        self.seed = int(seed)

    def fit(
        self,
        cal_feat,
        cal_resid=None,
        *,
        b: Sequence[float] = DEFAULT_B_VALUES,
    ) -> "DeploymentValidityMonitor":
        """Fit calibration reference statistics at target false-positive rate ``fpr``."""
        if cal_resid is None:
            self.b_ = np.asarray(b, dtype=float)
            self.input_mode_ = "signals"
            _, cal_feat, cal_resid = extract_observable_features(cal_feat, b=self.b_)
        else:
            self.b_ = np.asarray(b, dtype=float)
            self.input_mode_ = "features"

        cal_feat = np.atleast_2d(np.asarray(cal_feat, dtype=float))
        cal_resid = np.asarray(cal_resid, dtype=float).ravel()

        self.mu_ = cal_feat.mean(0)
        self.sd_ = cal_feat.std(0) + 1e-12
        Z = (cal_feat - self.mu_) / self.sd_
        cov = np.cov(Z, rowvar=False)
        cov = np.atleast_2d(cov) + np.eye(Z.shape[1]) * 1e-6
        self.inv_cov_ = np.linalg.inv(cov)
        self.cal_feat_ = cal_feat
        self.cal_resid_ = np.sort(cal_resid)

        self.cal_maha_ = self._loo_maha(cal_feat)
        self.cal_resid_score_ = cal_resid
        self.cal_std_resid_ = self._std_resid(cal_feat, cal_resid)
        self.Z_cal_ = Z

        m = self.null_subsample or min(2000, cal_feat.shape[0])
        self.thr_maha_ = self._null_threshold(self.cal_maha_, m)
        self.thr_resid_ = self._null_threshold(self.cal_resid_score_, m)
        self.thr_std_resid_ = self._null_threshold(self.cal_std_resid_, m)
        return self

    def fit_signals(
        self, cal_signals, b: Sequence[float] = DEFAULT_B_VALUES
    ) -> "DeploymentValidityMonitor":
        """Fit directly on raw unlabeled calibration DWI signals ``(N_cal, n_b)``."""
        return self.fit(cal_signals, cal_resid=None, b=b)

    def _maha(self, feat) -> np.ndarray:
        Z = (np.atleast_2d(np.asarray(feat, dtype=float)) - self.mu_) / self.sd_
        d = np.einsum("ni,ij,nj->n", Z, self.inv_cov_, Z)
        return np.sqrt(np.maximum(d, 0.0))

    def _loo_maha(self, cal_feat) -> np.ndarray:
        """Leave-one-out Mahalanobis distances on calibration features via Sherman-Morrison."""
        d_in = self._maha(cal_feat)
        n = cal_feat.shape[0]
        if n <= cal_feat.shape[1] + 3:
            return d_in
        d2 = d_in * d_in
        nm1_sq = float((n - 1) * (n - 1))
        denom = np.clip(1.0 - (float(n) / nm1_sq) * d2, 1e-3, None)
        d2_loo = (float(n * (n - 2)) / nm1_sq) * (d2 / denom)
        return np.sqrt(np.maximum(d2_loo, 0.0))

    @staticmethod
    def _std_resid(feat, resid) -> np.ndarray:
        """Noise-standardized NLLS residual ``||S - S_hat||_2 / sigma_hat``."""
        feat = np.atleast_2d(np.asarray(feat, dtype=float))
        resid = np.asarray(resid, dtype=float).ravel()
        if feat.shape[1] >= 9:
            sigma_hat = np.exp(np.clip(feat[:, -1], -20.0, 20.0))
            return resid / np.clip(sigma_hat, 1e-6, None)
        return resid.copy()

    def _null_threshold(self, id_scores: np.ndarray, m: int) -> float:
        """Upper ``(1 - fpr)`` quantile of the two-sample batch mean at size ``m``."""
        rng = np.random.default_rng(self.seed)
        n = id_scores.size
        m = max(1, int(m))
        mu_cal = float(np.mean(id_scores))
        scale = float(np.sqrt(1.0 + m / n))
        stats = np.empty(self.n_null, dtype=float)
        for k in range(self.n_null):
            idx = rng.choice(n, size=m, replace=True)
            stats[k] = mu_cal + scale * (float(np.mean(id_scores[idx])) - mu_cal)
        return float(np.quantile(stats, 1.0 - self.fpr))

    def _two_sample_batch_stats(
        self, id_scores, test_scores, two_sided: bool = False
    ) -> Dict[str, float]:
        """Two-sample batch mean comparison calibrated via bootstrap at ``alpha = fpr``."""
        id_s = np.asarray(id_scores, dtype=float).ravel()
        te_s = np.asarray(test_scores, dtype=float).ravel()
        n_cal, n_test = id_s.size, te_s.size
        mu_cal = float(np.mean(id_s))
        sd_cal = float(np.std(id_s)) + 1e-12
        stat = float(np.mean(te_s))
        se = float(
            np.sqrt(sd_cal**2 / n_cal + (float(np.var(te_s)) + 1e-12) / n_test)
        )
        se_null = sd_cal * float(np.sqrt(1.0 / n_cal + 1.0 / n_test))
        z = (stat - mu_cal) / se_null
        z_welch = (stat - mu_cal) / se

        rng = np.random.default_rng(self.seed + 17)
        scale = float(np.sqrt(1.0 + n_test / n_cal))
        boot_means = np.empty(self.n_null, dtype=float)
        for k in range(self.n_null):
            idx = rng.choice(n_cal, size=n_test, replace=True)
            boot_means[k] = mu_cal + scale * (float(np.mean(id_s[idx])) - mu_cal)
        if two_sided:
            thr_boot = float(
                mu_cal + np.quantile(np.abs(boot_means - mu_cal), 1.0 - self.fpr)
            )
            p_val = float(2.0 * (1.0 - norm.cdf(abs(z))))
        else:
            thr_boot = float(np.quantile(boot_means, 1.0 - self.fpr))
            p_val = float(1.0 - norm.cdf(z))
        return {
            "mu_cal": mu_cal,
            "sd_cal": sd_cal,
            "thr_boot": thr_boot,
            "z_score": float(z),
            "z_welch": float(z_welch),
            "p_value": p_val,
        }

    def evaluate(
        self,
        test_feat,
        test_resid=None,
        *,
        b: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Evaluate an unlabeled deployment batch for observable exchangeability shift.

        Returns per-family statistics (``'maha'`` and ``'resid'``), Hotelling ``T^2``,
        noise-standardized residual z-scores, single-voxel ID-vs-test ROC-AUC, and
        the combined boolean alert flag ``'fires'``.
        """
        if test_resid is None:
            b_use = getattr(self, "b_", DEFAULT_B_VALUES) if b is None else b
            _, test_feat, test_resid = extract_observable_features(test_feat, b=b_use)

        test_feat = np.atleast_2d(np.asarray(test_feat, dtype=float))
        test_maha = self._maha(test_feat)
        test_resid = np.asarray(test_resid, dtype=float).ravel()
        test_std_resid = self._std_resid(test_feat, test_resid)

        out: Dict[str, Any] = {}
        for fam, idsc, tsc, fixed_thr in (
            ("maha", self.cal_maha_, test_maha, self.thr_maha_),
            ("resid", self.cal_resid_score_, test_resid, self.thr_resid_),
        ):
            stat = float(np.mean(tsc))
            bstats = self._two_sample_batch_stats(idsc, tsc, two_sided=False)
            thr = bstats["thr_boot"] if self.null_subsample is None else fixed_thr
            out[fam] = {
                "stat": stat,
                "threshold": thr,
                "fires": bool(stat > thr),
                "auc": _safe_auc(idsc, tsc),
                "score_test": tsc,
                "thr_boot": bstats["thr_boot"],
                "z_score": bstats["z_score"],
                "p_value": bstats["p_value"],
            }

        n_cal, n_test = self.cal_feat_.shape[0], test_feat.shape[0]
        p_dim = self.cal_feat_.shape[1]
        Z_test = (test_feat - self.mu_) / self.sd_
        z_diff = Z_test.mean(0) - self.Z_cal_.mean(0)
        t2_stat = float(
            (n_cal * n_test / (n_cal + n_test)) * (z_diff @ self.inv_cov_ @ z_diff)
        )
        t2_thr = float(chi2.ppf(1.0 - self.fpr, p_dim))
        t2_p = float(1.0 - chi2.cdf(t2_stat, p_dim))
        out["maha"]["hotelling_t2"] = t2_stat
        out["maha"]["hotelling_thr"] = t2_thr
        out["maha"]["hotelling_p"] = t2_p

        sres_bstats = self._two_sample_batch_stats(
            self.cal_std_resid_, test_std_resid, two_sided=True
        )
        out["resid"]["std_stat"] = float(np.mean(test_std_resid))
        out["resid"]["std_mu_cal"] = sres_bstats["mu_cal"]
        out["resid"]["std_thr_boot"] = sres_bstats["thr_boot"]
        out["resid"]["std_z_score"] = sres_bstats["z_score"]
        out["resid"]["std_p_value"] = sres_bstats["p_value"]
        out["resid"]["std_auc"] = _safe_auc(self.cal_std_resid_, test_std_resid)

        out["fires"] = bool(out["maha"]["fires"] or out["resid"]["fires"])
        out["auc"] = float(max(out["maha"]["auc"], out["resid"]["auc"]))
        return out

    def evaluate_signals(
        self, test_signals, b: Optional[Sequence[float]] = None
    ) -> Dict[str, Any]:
        """Evaluate directly on raw unlabeled deployment DWI signals ``(N_test, n_b)``."""
        return self.evaluate(test_signals, test_resid=None, b=b)

    def power_by_sample_size(
        self,
        test_feat,
        test_resid=None,
        sample_sizes: Sequence[int] = (100, 500, 1000, 2500, 3000),
        is_control: bool = False,
        *,
        b: Optional[Sequence[float]] = None,
    ) -> Dict[int, Dict[str, float]]:
        """Two-sample batch detection power at ``alpha = self.fpr`` across batch sizes ``N``."""
        if test_resid is None:
            b_use = getattr(self, "b_", DEFAULT_B_VALUES) if b is None else b
            _, test_feat, test_resid = extract_observable_features(test_feat, b=b_use)

        test_feat = np.atleast_2d(np.asarray(test_feat, dtype=float))
        test_resid = np.asarray(test_resid, dtype=float).ravel()
        test_maha = self._maha(test_feat)
        test_sres = self._std_resid(test_feat, test_resid)

        n_cal = self.cal_feat_.shape[0]
        n_pool = test_feat.shape[0]
        p_dim = self.cal_feat_.shape[1]

        mu_maha, sd_maha = (
            float(self.cal_maha_.mean()),
            float(self.cal_maha_.std()) + 1e-12,
        )
        mu_res, sd_res = (
            float(self.cal_resid_score_.mean()),
            float(self.cal_resid_score_.std()) + 1e-12,
        )
        mu_sres, sd_sres = (
            float(self.cal_std_resid_.mean()),
            float(self.cal_std_resid_.std()) + 1e-12,
        )

        if is_control:
            delta_maha = 0.0
            mahal_sq_mean = 0.0
            delta_res = 0.0
            delta_sres = 0.0
        else:
            delta_maha = max(0.0, float(test_maha.mean()) - mu_maha)
            Z_test = (test_feat - self.mu_) / self.sd_
            zm = Z_test.mean(0) - self.Z_cal_.mean(0)
            mahal_sq_mean = max(
                0.0, float(zm @ self.inv_cov_ @ zm) - float(p_dim) / n_pool
            )
            delta_res = max(0.0, float(test_resid.mean()) - mu_res)
            delta_sres = abs(float(test_sres.mean()) - mu_sres)

        z_crit_1 = float(norm.ppf(1.0 - self.fpr))
        z_crit_2 = float(norm.ppf(1.0 - self.fpr / 2.0))
        t2_crit = float(chi2.ppf(1.0 - self.fpr, p_dim))

        out: Dict[int, Dict[str, float]] = {}
        for N in sample_sizes:
            N_int = int(N)
            se_factor = float(np.sqrt(1.0 / N_int + 1.0 / n_cal))
            z_m = delta_maha / (sd_maha * se_factor)
            p_maha = float(1.0 - norm.cdf(z_crit_1 - z_m))
            nc = (n_cal * N_int / (n_cal + N_int)) * mahal_sq_mean
            p_t2 = float(1.0 - ncx2.cdf(t2_crit, p_dim, nc))
            p_fam1 = max(p_maha, p_t2)

            z_r = delta_res / (sd_res * se_factor)
            p_res_raw = float(1.0 - norm.cdf(z_crit_1 - z_r))
            z_sr = delta_sres / (sd_sres * se_factor)
            p_res_std = float(
                1.0 - norm.cdf(z_crit_2 - z_sr) + norm.cdf(-z_crit_2 - z_sr)
            )
            p_fam2 = max(p_res_raw, p_res_std)

            p_comb = float(1.0 - (1.0 - p_fam1) * (1.0 - p_fam2))
            out[N_int] = {
                "power_maha": p_maha,
                "power_t2": p_t2,
                "power_fam1": p_fam1,
                "power_resid_raw": p_res_raw,
                "power_resid_std": p_res_std,
                "power_fam2": p_fam2,
                "power_combined": p_comb,
            }
        return out


# Backwards-compatible alias matching Gauge's internal class name
DeploymentMonitor = DeploymentValidityMonitor


class VoxelOODDetector:
    """Per-voxel conformal out-of-distribution (OOD) gate combining leave-one-out
    observable Mahalanobis distances and noise-standardized NLLS residual norms.

    Under calibration/test exchangeability, conformal p-values satisfy
    ``P(p_j <= alpha) <= alpha``, controlling the per-voxel false-positive rate
    while flagging localized artifact, partial-volume, or non-bi-exponential voxels.
    """

    def __init__(self, alpha: float = 0.05):
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must lie in (0, 1); got {alpha}.")
        self.alpha = float(alpha)

    def fit(
        self,
        cal_feat,
        cal_resid=None,
        *,
        b: Sequence[float] = DEFAULT_B_VALUES,
    ) -> "VoxelOODDetector":
        """Fit calibration reference score distribution from features or raw DWI signals."""
        self._mon = DeploymentValidityMonitor(fpr=self.alpha, n_null=64, seed=0)
        self._mon.fit(cal_feat, cal_resid=cal_resid, b=b)
        self.b_ = self._mon.b_

        maha_cal = self._mon.cal_maha_
        sres_cal = self._mon.cal_std_resid_
        self.mu_maha_ = float(np.mean(maha_cal))
        self.sd_maha_ = float(np.std(maha_cal)) + 1e-12
        self.mu_sres_ = float(np.mean(sres_cal))
        self.sd_sres_ = float(np.std(sres_cal)) + 1e-12

        z_maha = (maha_cal - self.mu_maha_) / self.sd_maha_
        z_sres = np.abs(sres_cal - self.mu_sres_) / self.sd_sres_
        self.cal_scores_ = np.sort(np.maximum(z_maha, z_sres))
        n = self.cal_scores_.size
        k = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        self.threshold_ = (
            float(self.cal_scores_[min(k - 1, n - 1)]) if k <= n else float("inf")
        )
        return self

    def score_voxels(
        self,
        test_feat,
        test_resid=None,
        *,
        b: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Compute joint Mahalanobis / standardized-residual OOD nonconformity scores."""
        if test_resid is None:
            b_use = getattr(self, "b_", DEFAULT_B_VALUES) if b is None else b
            _, test_feat, test_resid = extract_observable_features(test_feat, b=b_use)
        test_feat = np.atleast_2d(np.asarray(test_feat, dtype=float))
        test_resid = np.asarray(test_resid, dtype=float).ravel()
        maha_te = self._mon._maha(test_feat)
        sres_te = self._mon._std_resid(test_feat, test_resid)
        z_maha = (maha_te - self.mu_maha_) / self.sd_maha_
        z_sres = np.abs(sres_te - self.mu_sres_) / self.sd_sres_
        return np.maximum(z_maha, z_sres)

    def predict_ood(
        self,
        test_feat,
        test_resid=None,
        *,
        b: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Compute conformal OOD p-values, boolean mask, and flagged voxel fraction."""
        scores = self.score_voxels(test_feat, test_resid=test_resid, b=b)
        n_cal = self.cal_scores_.size
        n_ge = n_cal - np.searchsorted(self.cal_scores_, scores, side="left")
        p_vals = (1.0 + n_ge.astype(float)) / float(n_cal + 1)
        ood_mask = p_vals <= self.alpha
        return {
            "ood_mask": ood_mask,
            "p_values": p_vals,
            "scores": scores,
            "threshold": self.threshold_,
            "flagged_fraction": float(np.mean(ood_mask)) if ood_mask.size else 0.0,
        }


def _safe_auc(id_scores, ood_scores) -> float:
    """ROC-AUC separating calibration (ID=0) from deployment (test=1) scores."""
    id_scores = np.asarray(id_scores, dtype=float)
    ood_scores = np.asarray(ood_scores, dtype=float)
    y = np.concatenate([np.zeros(id_scores.size), np.ones(ood_scores.size)])
    s = np.concatenate([id_scores, ood_scores])
    if not np.isfinite(s).all() or np.unique(s).size < 2:
        return 0.5
    return float(roc_auc_score(y, s))
