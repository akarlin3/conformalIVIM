"""Base point and conditional-quantile estimators for IVIM parameter recovery.

Provides two built-in base families feeding ``ConformalIVIM``:

* ``fit_nlls`` / ``fit_nlls_batch`` / ``NLLSEstimator`` -- bounded Trust-Region
  Reflective non-linear least-squares (NLLS) fitting of the bi-exponential IVIM
  model with two-step segmented initialization (plus ``fit_varpro_2d_batch`` for
  2D multi-echo ``S(b, TE)`` acquisitions).
* ``IVIMQuantileRegressor`` -- gradient-boosted conditional quantile regression
  (``HistGradientBoostingRegressor`` with pinball loss) mapping noisy DWI decay
  curves to per-parameter conditional quantiles for Conformalized Quantile
  Regression (CQR).

Distribution-free conformal coverage holds for ANY base estimator -- base
estimator quality affects interval sharpness (width), not marginal validity.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from sklearn.ensemble import HistGradientBoostingRegressor

from conformal_ivim.forward import (
    DEFAULT_B_VALUES,
    DEFAULT_T2_BLOOD,
    DEFAULT_T2_TISSUE,
    ivim_reference_signal_2d,
    ivim_signal,
)

PARAM_NAMES: Tuple[str, str, str] = ("D", "Dstar", "f")

# Fit bounds in scaled units [D_e3, Dstar_e3, f]; D_e3 = D / 1e-3.
_LB = np.array([0.1, 3.0, 0.0], dtype=float)
_UB = np.array([4.0, 300.0, 0.7], dtype=float)


def fit_nlls(signal, b=DEFAULT_B_VALUES, return_s0: bool = False) -> Dict[str, float]:
    """Fit (D, Dstar, f) [+ S0] to a single 1D IVIM decay curve via bounded NLLS.

    Uses a segmented log-linear initialization on the high-b tail (b >= 200 s/mm^2)
    to seed D and f prior to 4-parameter Trust-Region Reflective optimization.
    """
    signal = np.asarray(signal, dtype=float)
    b = np.asarray(b, dtype=float)
    s0_obs = float(signal[np.argmin(b)])
    s0_obs = s0_obs if s0_obs > 0 else float(np.max(signal))

    tail = b >= 200.0
    if tail.sum() >= 2 and np.all(signal[tail] > 0):
        slope, intercept = np.polyfit(b[tail], np.log(signal[tail]), 1)
        D_init = float(np.clip(-slope, 0.2e-3, 3.5e-3))
        A = np.exp(intercept)
        f_init = float(np.clip(1.0 - A / s0_obs, 0.02, 0.5))
    else:
        D_init, f_init = 1.2e-3, 0.15
    p0 = np.array([D_init * 1e3, 20.0, f_init], dtype=float)
    p0 = np.clip(p0, _LB + 1e-6, _UB - 1e-6)

    s0_lb, s0_ub = 0.3 * s0_obs, 3.0 * s0_obs
    lb = np.array([_LB[0], _LB[1], _LB[2], s0_lb], dtype=float)
    ub = np.array([_UB[0], _UB[1], _UB[2], s0_ub], dtype=float)
    p0 = np.append(p0, np.clip(s0_obs, s0_lb + 1e-9, s0_ub - 1e-9))

    def residuals(p):
        D, Dstar, f, S0 = p[0] * 1e-3, p[1] * 1e-3, p[2], p[3]
        return ivim_signal(b, D, Dstar, f, S0=S0) - signal

    sol = least_squares(
        residuals,
        p0,
        bounds=(lb, ub),
        method="trf",
        x_scale=[1.0, 30.0, 0.2, 1.0],
        max_nfev=2000,
    )
    D, Dstar, f, S0 = sol.x[0] * 1e-3, sol.x[1] * 1e-3, sol.x[2], sol.x[3]
    out = {"D": float(D), "Dstar": float(Dstar), "f": float(f)}
    if return_s0:
        out["S0"] = float(S0)
    return out


def fit_nlls_batch(signals, b=DEFAULT_B_VALUES) -> np.ndarray:
    """Fit a batch of 1D signals. Returns an ``(N, 3)`` array of ``(D, Dstar, f)``."""
    signals = np.atleast_2d(np.asarray(signals, dtype=float))
    out = np.empty((signals.shape[0], 3), dtype=float)
    for i, s in enumerate(signals):
        est = fit_nlls(s, b)
        out[i] = (est["D"], est["Dstar"], est["f"])
    return out


def nlls_fit_and_noise(
    signals, b=DEFAULT_B_VALUES
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-voxel NLLS point estimate ``(D, Dstar, f)``, ``S0``, and residual SD ``sigma``."""
    signals = np.atleast_2d(np.asarray(signals, dtype=float))
    b = np.asarray(b, dtype=float)
    N = signals.shape[0]
    theta = np.empty((N, 3), dtype=float)
    s0 = np.empty(N, dtype=float)
    sigma = np.empty(N, dtype=float)
    for i, s in enumerate(signals):
        e = fit_nlls(s, b, return_s0=True)
        theta[i] = (e["D"], e["Dstar"], e["f"])
        s0[i] = e["S0"]
        model = ivim_signal(b, e["D"], e["Dstar"], e["f"], S0=e["S0"])
        sigma[i] = max(float(np.std(s - model)), 1e-3)
    return theta, s0, sigma


def fit_varpro_2d_batch(
    signals,
    b_flat,
    te_flat,
    T2_tissue: float = DEFAULT_T2_TISSUE,
    T2_blood: float = DEFAULT_T2_BLOOD,
    n_iter: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Global Variable-Projection + damped Gauss-Newton 2D NLLS fitter for S(b, TE).

    Exploits separable linear amplitudes over a 45-point D* grid prior to
    4-parameter damped Gauss-Newton refinement. Returns
    ``(theta_hat, s0_hat, sigma_hat, snr_hat)`` with ``theta_hat`` columns
    ordered ``(D, Dstar, f)``.
    """
    signals = np.atleast_2d(np.asarray(signals, dtype=float))
    b_flat = np.asarray(b_flat, dtype=float)
    te_flat = np.asarray(te_flat, dtype=float)
    N, M = signals.shape
    w_b = np.exp(-te_flat / float(T2_blood))
    w_t = np.exp(-te_flat / float(T2_tissue))
    te_min = np.min(te_flat)

    hi = (b_flat >= 200.0) & (te_flat == te_min)
    b_hi = b_flat[hi]
    wt_hi = w_t[hi]
    y_hi = np.log(np.clip(signals[:, hi] / wt_hi[None, :], 1e-3, None))
    bm = b_hi.mean()
    D0 = -np.sum(
        (b_hi[None, :] - bm) * (y_hi - y_hi.mean(axis=1, keepdims=True)),
        axis=1,
    ) / (np.sum((b_hi - bm) ** 2) + 1e-12)
    D0 = np.clip(D0, 0.4e-3, 2.6e-3)

    phi_t = w_t[None, :] * np.exp(-b_flat[None, :] * D0[:, None])
    dstar_grid = np.linspace(5.0e-3, 105.0e-3, 45)
    best_rss = np.full(N, np.inf)
    best_ds = np.full(N, 30.0e-3)
    best_Ab = np.full(N, 0.15)
    best_At = np.full(N, 0.85)

    Stt = np.sum(phi_t * phi_t, axis=1)
    Sty = np.sum(phi_t * signals, axis=1)
    for ds in dstar_grid:
        phi_b = w_b * np.exp(-b_flat * ds)
        Sbb = float(np.sum(phi_b * phi_b))
        Sbt = np.sum(phi_t * phi_b[None, :], axis=1)
        Sby = signals @ phi_b
        det = np.clip(Sbb * Stt - Sbt * Sbt, 1e-10, None)
        Ab = np.clip((Stt * Sby - Sbt * Sty) / det, 0.02, 0.55)
        At = np.clip((Sty - Sbt * Ab) / np.clip(Stt, 1e-10, None), 0.45, 1.40)
        pred = Ab[:, None] * phi_b[None, :] + At[:, None] * phi_t
        rss = np.sum((signals - pred) ** 2, axis=1)
        mask = rss < best_rss
        best_rss = np.where(mask, rss, best_rss)
        best_ds = np.where(mask, ds, best_ds)
        best_Ab = np.where(mask, Ab, best_Ab)
        best_At = np.where(mask, At, best_At)

    S0_0 = np.clip(best_Ab + best_At, 0.5, 1.6)
    f0 = np.clip(best_Ab / S0_0, 0.02, 0.45)
    theta = np.column_stack([D0, best_ds, f0, S0_0])

    lo_b = np.array([0.2e-3, 4.0e-3, 0.01, 0.4])
    hi_b = np.array([3.5e-3, 120.0e-3, 0.50, 1.8])
    bb = b_flat[None, :]
    wb = w_b[None, :]
    wt = w_t[None, :]
    diag_lam = np.diag([1e-7, 5e-5, 1e-3, 1e-3])[None, :, :]

    for _ in range(n_iter):
        D, Ds, f, S0 = theta[:, 0:1], theta[:, 1:2], theta[:, 2:3], theta[:, 3:4]
        ed = wt * np.exp(-bb * D)
        eds = wb * np.exp(-bb * Ds)
        pred = S0 * (f * eds + (1.0 - f) * ed)
        res = signals - pred
        J = np.stack(
            [
                S0 * (1.0 - f) * (-bb) * ed,
                S0 * f * (-bb) * eds,
                S0 * (eds - ed),
                f * eds + (1.0 - f) * ed,
            ],
            axis=2,
        )
        JTJ = np.einsum("nmi,nmj->nij", J, J)
        JTr = np.einsum("nmi,nm->ni", J, res)
        H = JTJ + diag_lam
        try:
            step = np.linalg.solve(H, JTr[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        theta_new = np.clip(theta + 0.6 * step, lo_b, hi_b)
        theta_new[:, 1] = np.maximum(theta_new[:, 1], theta_new[:, 0] * 1.5)
        D2, Ds2, f2, S02 = (
            theta_new[:, 0:1],
            theta_new[:, 1:2],
            theta_new[:, 2:3],
            theta_new[:, 3:4],
        )
        pred2 = S02 * (
            f2 * wb * np.exp(-bb * Ds2) + (1.0 - f2) * wt * np.exp(-bb * D2)
        )
        improve = np.sum((signals - pred2) ** 2, axis=1) <= np.sum(res ** 2, axis=1)
        theta = np.where(improve[:, None], theta_new, theta)

    D, Ds, f, S0 = theta[:, 0:1], theta[:, 1:2], theta[:, 2:3], theta[:, 3:4]
    pred = S0 * (f * wb * np.exp(-bb * Ds) + (1.0 - f) * wt * np.exp(-bb * D))
    dof = max(M - 4, 1)
    sigma_hat = np.sqrt(np.sum((signals - pred) ** 2, axis=1) / dof)
    s_ref_hat = ivim_reference_signal_2d(
        te_min, theta[:, 2], S0=theta[:, 3], T2_tissue=T2_tissue, T2_blood=T2_blood
    )
    snr_hat = s_ref_hat / np.clip(sigma_hat, 1e-6, None)
    return theta[:, :3], theta[:, 3], sigma_hat, snr_hat


class NLLSEstimator:
    """Scikit-learn-compatible bounded NLLS point estimator for ``(D, Dstar, f)``.

    Can be passed directly to ``ConformalIVIM(estimator=NLLSEstimator(b=...))``
    without requiring supervised training data ``(X_train, Y_train)``.
    """

    def __init__(
        self,
        b: Sequence[float] = DEFAULT_B_VALUES,
        te: Optional[Sequence[float]] = None,
        T2_tissue: float = DEFAULT_T2_TISSUE,
        T2_blood: float = DEFAULT_T2_BLOOD,
    ):
        self.b = np.asarray(b, dtype=float)
        self.te = None if te is None else np.asarray(te, dtype=float)
        self.T2_tissue = float(T2_tissue)
        self.T2_blood = float(T2_blood)
        self.is_fitted_ = True

    def fit(self, X=None, Y=None) -> "NLLSEstimator":
        self.is_fitted_ = True
        return self

    def predict(self, X) -> np.ndarray:
        signals = np.atleast_2d(np.asarray(X, dtype=float))
        if self.te is not None:
            theta, _, _, _ = fit_varpro_2d_batch(
                signals,
                self.b,
                self.te,
                T2_tissue=self.T2_tissue,
                T2_blood=self.T2_blood,
            )
            return theta
        return fit_nlls_batch(signals, self.b)


class IVIMQuantileRegressor:
    """Gradient-boosted conditional quantile regression base estimator for CQR.

    Fits one ``HistGradientBoostingRegressor`` per ``(parameter, quantile_level)``
    pair. Targets are the three IVIM parameters ``(D, Dstar, f)``.
    """

    def __init__(
        self,
        quantile_levels: Iterable[float] = (0.05, 0.95),
        random_state: int = 0,
        max_iter: int = 200,
        learning_rate: float = 0.08,
        max_leaf_nodes: int = 31,
    ):
        self.quantile_levels = sorted({round(float(q), 6) for q in quantile_levels})
        self.random_state = random_state
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.models_: Dict[Tuple[int, float], HistGradientBoostingRegressor] = {}

    def fit(self, X, Y) -> "IVIMQuantileRegressor":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        self.n_outputs_ = Y.shape[1]
        for j in range(self.n_outputs_):
            for q in self.quantile_levels:
                model = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=q,
                    learning_rate=self.learning_rate,
                    max_iter=self.max_iter,
                    max_leaf_nodes=self.max_leaf_nodes,
                    random_state=self.random_state,
                )
                model.fit(X, Y[:, j])
                self.models_[(j, round(float(q), 6))] = model
        return self

    def predict_quantile(self, X, param_idx: int, q: float) -> np.ndarray:
        key = (int(param_idx), round(float(q), 6))
        if key not in self.models_:
            raise KeyError(
                f"Quantile {round(float(q), 6)} for parameter index {param_idx} was not "
                f"fitted. Available levels: {self.quantile_levels}"
            )
        return self.models_[key].predict(np.atleast_2d(np.asarray(X, dtype=float)))

    def predict_interval(self, X, alpha: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
        """Predict uncalibrated ``(alpha/2, 1 - alpha/2)`` quantile bands ``(lo, hi)``."""
        X_arr = np.atleast_2d(np.asarray(X, dtype=float))
        n_out = getattr(self, "n_outputs_", len(PARAM_NAMES))
        q_lo = round(float(alpha) / 2.0, 6)
        q_hi = round(1.0 - float(alpha) / 2.0, 6)
        lo = np.column_stack(
            [self.predict_quantile(X_arr, j, q_lo) for j in range(n_out)]
        )
        hi = np.column_stack(
            [self.predict_quantile(X_arr, j, q_hi) for j in range(n_out)]
        )
        return lo, hi
