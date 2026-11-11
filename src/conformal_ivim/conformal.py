"""Split-conformal, Conformalized Quantile Regression (CQR), covariate-shift /
feature-localized conformal methods, and the high-level ``ConformalIVIM`` wrapper.

Under exchangeability of calibration and test observations, marginal coverage
satisfies the finite-sample, distribution-free bound:

    1 - alpha <= P(Y in C(X)) <= 1 - alpha + 1 / (n_cal + 1)

for ANY base point predictor or conditional quantile estimator.

References
----------
- Vovk, Gammerman & Shafer (2005), *Algorithmic Learning in a Random World*.
- Lei et al. (2018), Distribution-free predictive inference for regression, *JASA*.
- Romano, Patterson & Candès (2019), Conformalized Quantile Regression, *NeurIPS*.
- Tibshirani et al. (2019), Conformal prediction under covariate shift, *NeurIPS*.
- Guan (2023), Localized conformal prediction, *Biometrika*.
- Gibbs, Cherian & Candès (2023), Conformal prediction with conditional guarantees.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.linear_model import QuantileRegressor

from conformal_ivim.estimators import (
    IVIMQuantileRegressor,
    NLLSEstimator,
    PARAM_NAMES,
)
from conformal_ivim.forward import DEFAULT_B_VALUES
from conformal_ivim.metrics import (
    empirical_coverage,
    evaluate_intervals,
    interval_score,
    interval_width,
)


def conformal_quantile(scores, alpha: float) -> float:
    """Finite-sample conformal quantile of nonconformity ``scores`` at level ``alpha``.

    Returns the k-th smallest score with ``k = ceil((n + 1) * (1 - alpha))``.
    When ``k > n`` (calibration set too small for ``alpha``), returns ``np.inf``.
    """
    scores_arr = np.asarray(scores, dtype=float).ravel()
    n = scores_arr.size
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1.0 - float(alpha))))
    if k <= 0:
        return float(np.min(scores_arr))
    if k > n:
        return np.inf
    return float(np.sort(scores_arr)[k - 1])


def split_conformal(
    cal_pred, cal_true, test_pred, alpha: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Symmetric absolute-residual split conformal prediction.

    Calibration score: ``s_i = |y_i - yhat_i|``.
    Test prediction interval: ``[test_pred - q, test_pred + q]``.
    """
    cal_pred = np.asarray(cal_pred, dtype=float)
    cal_true = np.asarray(cal_true, dtype=float)
    test_pred = np.asarray(test_pred, dtype=float)
    scores = np.abs(cal_true - cal_pred)
    q = conformal_quantile(scores, alpha)
    return test_pred - q, test_pred + q, q


def cqr(
    cal_lo, cal_hi, cal_true, test_lo, test_hi, alpha: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Conformalized Quantile Regression (Romano, Patterson & Candès, 2019).

    Calibration signed conformity score:
        ``E_i = max(q_lo(X_i) - Y_i, Y_i - q_hi(X_i))``.
    Calibrated prediction band:
        ``[test_lo - q, test_hi + q]``.
    """
    cal_lo = np.asarray(cal_lo, dtype=float)
    cal_hi = np.asarray(cal_hi, dtype=float)
    cal_true = np.asarray(cal_true, dtype=float)
    test_lo = np.asarray(test_lo, dtype=float)
    test_hi = np.asarray(test_hi, dtype=float)
    scores = np.maximum(cal_lo - cal_true, cal_true - cal_hi)
    q = conformal_quantile(scores, alpha)
    return test_lo - q, test_hi + q, q


def mondrian_split(
    cal_pred, cal_true, cal_strata, test_pred, test_strata, alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Group-conditional (Mondrian) split conformal calibrated per stratum."""
    cal_pred = np.asarray(cal_pred, dtype=float)
    cal_true = np.asarray(cal_true, dtype=float)
    test_pred = np.asarray(test_pred, dtype=float)
    cal_strata = np.asarray(cal_strata)
    test_strata = np.asarray(test_strata)
    lo = np.empty_like(test_pred)
    hi = np.empty_like(test_pred)
    for s in np.unique(test_strata):
        cm, tm = cal_strata == s, test_strata == s
        q = conformal_quantile(np.abs(cal_true[cm] - cal_pred[cm]), alpha)
        lo[tm] = test_pred[tm] - q
        hi[tm] = test_pred[tm] + q
    return lo, hi


def mondrian_cqr(
    cal_lo,
    cal_hi,
    cal_true,
    cal_strata,
    test_lo,
    test_hi,
    test_strata,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Group-conditional (Mondrian) CQR calibrated per observable stratum (e.g. SNR)."""
    cal_lo = np.asarray(cal_lo, dtype=float)
    cal_hi = np.asarray(cal_hi, dtype=float)
    cal_true = np.asarray(cal_true, dtype=float)
    test_lo = np.asarray(test_lo, dtype=float)
    test_hi = np.asarray(test_hi, dtype=float)
    cal_strata = np.asarray(cal_strata)
    test_strata = np.asarray(test_strata)
    lo = np.empty_like(test_lo)
    hi = np.empty_like(test_hi)
    for s in np.unique(test_strata):
        cm, tm = cal_strata == s, test_strata == s
        scores = np.maximum(cal_lo[cm] - cal_true[cm], cal_true[cm] - cal_hi[cm])
        q = conformal_quantile(scores, alpha)
        lo[tm] = test_lo[tm] - q
        hi[tm] = test_hi[tm] + q
    return lo, hi


def weighted_conformal_quantile(
    scores, weights, alpha: float, self_weight: Optional[float] = None
) -> float:
    """Test-point-augmented weighted conformal quantile (Tibshirani et al., 2019)."""
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if self_weight is None:
        self_weight = float(weights.max()) if weights.size else 1.0
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    total = float(weights.sum()) + float(self_weight)
    if total <= 0:
        return np.inf
    cum = np.cumsum(weights[order]) / total
    idx = int(np.searchsorted(cum, 1.0 - float(alpha), side="left"))
    if idx >= s_sorted.size:
        return np.inf
    return float(s_sorted[idx])


def weighted_split_conformal(
    cal_scores, w_cal, w_test, alpha: float
) -> np.ndarray:
    """Vectorized covariate-shift weighted split conformal radii (Tibshirani et al., 2019)."""
    cal_scores = np.asarray(cal_scores, dtype=float)
    w_cal = np.asarray(w_cal, dtype=float)
    w_test = np.asarray(w_test, dtype=float)
    order = np.argsort(cal_scores, kind="mergesort")
    s_sorted = cal_scores[order]
    cumw = np.cumsum(w_cal[order])
    W = float(w_cal.sum())
    targets = (1.0 - float(alpha)) * (W + w_test)
    idx = np.searchsorted(cumw, targets, side="left")
    out = np.full(w_test.shape[0], np.inf, dtype=float)
    ok = idx < s_sorted.size
    out[ok] = s_sorted[idx[ok]]
    return out


def localized_conformal(
    cal_scores,
    cal_feat,
    test_feat,
    alpha: float,
    bandwidth: float,
    kernel: str = "gaussian",
) -> np.ndarray:
    """Localized conformal prediction thresholds (Guan, 2023)."""
    cal_scores = np.asarray(cal_scores, dtype=float)
    cal_feat = np.atleast_2d(np.asarray(cal_feat, dtype=float))
    test_feat = np.atleast_2d(np.asarray(test_feat, dtype=float))
    mu = cal_feat.mean(0)
    sd = cal_feat.std(0) + 1e-12
    C = (cal_feat - mu) / sd
    T = (test_feat - mu) / sd
    h = float(bandwidth)
    thr = np.empty(T.shape[0], dtype=float)
    for i in range(T.shape[0]):
        d2 = np.sum((C - T[i]) ** 2, axis=1)
        if kernel == "gaussian":
            w = np.exp(-0.5 * d2 / (h * h))
        elif kernel == "boxcar":
            w = (np.sqrt(d2) <= h).astype(float)
        else:
            raise ValueError(f"unknown kernel {kernel!r}")
        thr[i] = weighted_conformal_quantile(cal_scores, w, alpha, self_weight=1.0)
    return thr


def conditional_conformal(
    cal_scores, cal_Phi, test_Phi, alpha: float, ridge: float = 0.0
) -> np.ndarray:
    """Conditional conformal thresholds via pinball quantile regression on features
    (Gibbs, Cherian & Candès, 2023).
    """
    cal_scores = np.asarray(cal_scores, dtype=float)
    cal_Phi = np.atleast_2d(np.asarray(cal_Phi, dtype=float))
    test_Phi = np.atleast_2d(np.asarray(test_Phi, dtype=float))
    qr = QuantileRegressor(
        quantile=1.0 - float(alpha), alpha=float(ridge), solver="highs"
    )
    qr.fit(cal_Phi, cal_scores)
    return qr.predict(test_Phi)


# --------------------------------------------------------------------------- #
# High-Level Public Wrapper: ConformalIVIM
# --------------------------------------------------------------------------- #
class ConformalIVIM:
    """Distribution-free conformal prediction wrapper for IVIM parameter estimation.

    Wraps any point regressor (for split-conformal intervals) or conditional
    quantile estimator (for Conformalized Quantile Regression, CQR) to produce
    finite-sample calibrated ``(1 - alpha)`` prediction intervals for ``(D, D*, f)``.

    Parameters
    ----------
    estimator : object, tuple, callable, or None, default=None
        Base estimator:
        - ``None``: defaults to ``NLLSEstimator(b=b)`` when ``method in ('auto', 'split')``
          or ``IVIMQuantileRegressor((alpha/2, 1 - alpha/2))`` when ``method == 'cqr'``.
        - Any point regressor exposing ``.predict(X) -> (N, 3)`` (or callable ``fn(X)``)
          for split-conformal calibration.
        - Any quantile estimator exposing ``.predict_interval(X, alpha) -> (lo, hi)``,
          ``.predict_quantile(X, param_idx, q)``, or a ``(lower_est, upper_est)`` pair
          of regressors for CQR calibration.
    alpha : float, default=0.10
        Target miscoverage rate in ``(0, 1)`` (nominal coverage ``1 - alpha``).
    method : {'auto', 'split', 'cqr'}, default='auto'
        Conformalization strategy. ``'auto'`` selects ``'cqr'`` for quantile
        estimators and ``'split'`` for point estimators.
    b : array-like, default=DEFAULT_B_VALUES
        Diffusion-weighting b-values in s/mm^2 used when constructing the default
        ``NLLSEstimator``.
    clip_bounds : sequence of (low, high) pairs or None, default=None
        Optional post-calibration physiological clipping bounds per parameter.
    random_state : int, default=0
        Random seed used when instantiating a default ``IVIMQuantileRegressor``.
    """

    def __init__(
        self,
        estimator: Any = None,
        alpha: float = 0.10,
        method: str = "auto",
        b: Sequence[float] = DEFAULT_B_VALUES,
        clip_bounds: Optional[Sequence[Tuple[float, float]]] = None,
        random_state: int = 0,
    ):
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha!r}")
        if method not in ("auto", "split", "cqr"):
            raise ValueError(
                f"method must be 'auto', 'split', or 'cqr', got {method!r}"
            )
        self.estimator = estimator
        self.alpha = float(alpha)
        self.method = method
        self.b = np.asarray(b, dtype=float)
        self.clip_bounds = clip_bounds
        self.random_state = int(random_state)

    def _ensure_estimator(self) -> Tuple[Any, str]:
        est = self.estimator
        method = self.method
        if est is None:
            if method == "cqr":
                q_lo = round(self.alpha / 2.0, 6)
                q_hi = round(1.0 - self.alpha / 2.0, 6)
                est = IVIMQuantileRegressor(
                    quantile_levels=(q_lo, q_hi),
                    random_state=self.random_state,
                )
                return est, "cqr"
            est = NLLSEstimator(b=self.b)
            return est, "split"

        if method == "auto":
            if (
                hasattr(est, "predict_interval")
                or hasattr(est, "predict_quantile")
                or isinstance(est, (tuple, list))
            ):
                resolved = "cqr"
            else:
                resolved = "split"
        else:
            resolved = method
        return est, resolved

    def fit(self, X_train, Y_train=None) -> "ConformalIVIM":
        """Fit the underlying base point or quantile estimator on ``(X_train, Y_train)``."""
        est, resolved = self._ensure_estimator()
        self.estimator_ = est
        self.method_ = resolved

        if isinstance(est, (tuple, list)):
            if len(est) != 2:
                raise ValueError(
                    "Tuple/list quantile estimator must contain (lower_est, upper_est)."
                )
            est[0].fit(X_train, Y_train)
            est[1].fit(X_train, Y_train)
        elif hasattr(est, "fit"):
            if isinstance(est, NLLSEstimator):
                est.fit(X_train, Y_train)
            else:
                if Y_train is None:
                    raise ValueError(
                        "Supervised base estimator requires Y_train in fit(X_train, Y_train)."
                    )
                est.fit(X_train, Y_train)
        return self

    def _predict_base_point(self, X) -> np.ndarray:
        est = getattr(self, "estimator_", None)
        if est is None:
            est, self.method_ = self._ensure_estimator()
            self.estimator_ = est
        X_arr = np.atleast_2d(np.asarray(X, dtype=float))
        pred = est.predict(X_arr) if hasattr(est, "predict") else est(X_arr)
        pred = np.asarray(pred, dtype=float)
        return pred[:, None] if pred.ndim == 1 else pred

    def _predict_base_quantiles(
        self, X, alpha: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        est = getattr(self, "estimator_", None)
        if est is None:
            est, self.method_ = self._ensure_estimator()
            self.estimator_ = est
        X_arr = np.atleast_2d(np.asarray(X, dtype=float))
        if hasattr(est, "predict_interval"):
            lo, hi = est.predict_interval(X_arr, alpha=alpha)
        elif hasattr(est, "predict_quantile"):
            n_out = getattr(self, "n_outputs_", len(PARAM_NAMES))
            q_lo = round(float(alpha) / 2.0, 6)
            q_hi = round(1.0 - float(alpha) / 2.0, 6)
            lo = np.column_stack(
                [est.predict_quantile(X_arr, j, q_lo) for j in range(n_out)]
            )
            hi = np.column_stack(
                [est.predict_quantile(X_arr, j, q_hi) for j in range(n_out)]
            )
        elif isinstance(est, (tuple, list)) and len(est) == 2:
            lo = est[0].predict(X_arr)
            hi = est[1].predict(X_arr)
        elif callable(est):
            lo, hi = est(X_arr, alpha)
        else:
            raise TypeError(
                "CQR estimator must implement predict_interval(X, alpha), "
                "predict_quantile(X, j, q), be a (lo_est, hi_est) pair, or be callable."
            )
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        if lo.ndim == 1:
            lo = lo[:, None]
            hi = hi[:, None]
        return lo, hi

    def calibrate(
        self, X_cal, Y_cal, strata: Optional[Sequence[Any]] = None
    ) -> "ConformalIVIM":
        """Compute finite-sample conformal nonconformity scores on held-out ``(X_cal, Y_cal)``."""
        if not hasattr(self, "estimator_"):
            self.estimator_, self.method_ = self._ensure_estimator()

        Y_arr = np.asarray(Y_cal, dtype=float)
        self._squeeze_1d_ = Y_arr.ndim == 1
        if Y_arr.ndim == 1:
            Y_arr = Y_arr[:, None]
        self.n_outputs_ = Y_arr.shape[1]

        if self.method_ == "split":
            cal_pred = self._predict_base_point(X_cal)
            if cal_pred.shape != Y_arr.shape:
                raise ValueError(
                    f"Base predictions shape {cal_pred.shape} != Y_cal shape {Y_arr.shape}"
                )
            self.cal_pred_ = cal_pred
            self.scores_ = np.abs(Y_arr - cal_pred)
        else:
            cal_lo, cal_hi = self._predict_base_quantiles(X_cal, self.alpha)
            if cal_lo.shape != Y_arr.shape or cal_hi.shape != Y_arr.shape:
                raise ValueError(
                    f"Base quantile shapes {(cal_lo.shape, cal_hi.shape)} != Y_cal shape {Y_arr.shape}"
                )
            self.cal_lo_ = cal_lo
            self.cal_hi_ = cal_hi
            self.scores_ = np.maximum(cal_lo - Y_arr, Y_arr - cal_hi)

        self.cal_true_ = Y_arr
        self.cal_strata_ = None if strata is None else np.asarray(strata)
        self.quantiles_ = np.array(
            [
                conformal_quantile(self.scores_[:, j], self.alpha)
                for j in range(self.n_outputs_)
            ],
            dtype=float,
        )
        return self

    def fit_calibrate(
        self,
        X_train,
        Y_train,
        X_cal,
        Y_cal,
        strata: Optional[Sequence[Any]] = None,
    ) -> "ConformalIVIM":
        """Fit the base estimator on ``(X_train, Y_train)`` and calibrate on ``(X_cal, Y_cal)``."""
        return self.fit(X_train, Y_train).calibrate(X_cal, Y_cal, strata=strata)

    def predict(self, X_test) -> np.ndarray:
        """Predict point estimates (or interval midpoints for CQR) on ``X_test``."""
        if not hasattr(self, "estimator_"):
            self.estimator_, self.method_ = self._ensure_estimator()
        if self.method_ == "split":
            out = self._predict_base_point(X_test)
        else:
            lo, hi = self._predict_base_quantiles(X_test, self.alpha)
            out = 0.5 * (lo + hi)
        return out[:, 0] if getattr(self, "_squeeze_1d_", False) else out

    def predict_interval(
        self,
        X_test,
        alpha: Optional[float] = None,
        weights_cal: Optional[np.ndarray] = None,
        weights_test: Optional[np.ndarray] = None,
        strata: Optional[Sequence[Any]] = None,
        return_dict: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
        """Output calibrated ``(1 - alpha)`` prediction intervals ``(lower, upper)``.

        Parameters
        ----------
        X_test : array-like of shape (N_test, n_features)
            Test DWI signals or feature matrix.
        alpha : float or None, default=None
            Optional override for miscoverage rate (defaults to ``self.alpha``).
        weights_cal, weights_test : 1D array-like or None, default=None
            Optional likelihood-ratio importance weights ``dP_test / dP_cal`` for
            covariate-shift-aware weighted conformal intervals.
        strata : sequence or None, default=None
            Optional test-set stratum labels for Mondrian group-conditional
            intervals (requires ``strata`` passed during ``calibrate``).
        return_dict : bool, default=False
            If ``True``, return a rich dictionary keyed by ``'lower'``, ``'upper'``,
            ``'width'``, ``'quantiles'``, and parameter names ``('D', 'Dstar', 'f')``.
        """
        if not hasattr(self, "scores_"):
            raise RuntimeError(
                "ConformalIVIM must be calibrated via .calibrate(X_cal, Y_cal) "
                "before calling .predict_interval(X_test)."
            )
        a = self.alpha if alpha is None else float(alpha)
        P = self.n_outputs_

        if strata is not None:
            if getattr(self, "cal_strata_", None) is None:
                raise ValueError(
                    "Mondrian prediction requires calibration strata in .calibrate(..., strata=...)."
                )
            test_strata = np.asarray(strata)
            if self.method_ == "split":
                test_pred = self._predict_base_point(X_test)
                lower = np.empty_like(test_pred)
                upper = np.empty_like(test_pred)
                for j in range(P):
                    lower[:, j], upper[:, j] = mondrian_split(
                        self.cal_pred_[:, j],
                        self.cal_true_[:, j],
                        self.cal_strata_,
                        test_pred[:, j],
                        test_strata,
                        a,
                    )
            else:
                test_lo, test_hi = self._predict_base_quantiles(X_test, a)
                lower = np.empty_like(test_lo)
                upper = np.empty_like(test_hi)
                for j in range(P):
                    lower[:, j], upper[:, j] = mondrian_cqr(
                        self.cal_lo_[:, j],
                        self.cal_hi_[:, j],
                        self.cal_true_[:, j],
                        self.cal_strata_,
                        test_lo[:, j],
                        test_hi[:, j],
                        test_strata,
                        a,
                    )
            q_used = self.quantiles_
        elif weights_cal is not None and weights_test is not None:
            radii = np.column_stack(
                [
                    weighted_split_conformal(
                        self.scores_[:, j], weights_cal, weights_test, a
                    )
                    for j in range(P)
                ]
            )
            if self.method_ == "split":
                test_pred = self._predict_base_point(X_test)
                lower = test_pred - radii
                upper = test_pred + radii
            else:
                test_lo, test_hi = self._predict_base_quantiles(X_test, a)
                lower = test_lo - radii
                upper = test_hi + radii
            q_used = radii
        else:
            q_used = (
                self.quantiles_
                if abs(a - self.alpha) < 1e-12
                else np.array(
                    [conformal_quantile(self.scores_[:, j], a) for j in range(P)],
                    dtype=float,
                )
            )
            if self.method_ == "split":
                test_pred = self._predict_base_point(X_test)
                lower = test_pred - q_used[None, :]
                upper = test_pred + q_used[None, :]
            else:
                test_lo, test_hi = self._predict_base_quantiles(X_test, a)
                lower = test_lo - q_used[None, :]
                upper = test_hi + q_used[None, :]

        if self.clip_bounds is not None:
            lower = lower.copy()
            upper = upper.copy()
            for j, (lb, ub) in enumerate(self.clip_bounds[:P]):
                lower[:, j] = np.clip(lower[:, j], lb, ub)
                upper[:, j] = np.clip(upper[:, j], lb, ub)

        if self._squeeze_1d_:
            lower_out, upper_out = lower[:, 0], upper[:, 0]
        else:
            lower_out, upper_out = lower, upper

        if not return_dict:
            return lower_out, upper_out

        result: Dict[str, Any] = {
            "lower": lower_out,
            "upper": upper_out,
            "width": upper_out - lower_out,
            "quantiles": q_used,
            "alpha": a,
        }
        if not self._squeeze_1d_:
            for j, nm in enumerate(PARAM_NAMES[:P]):
                result[nm] = (lower[:, j], upper[:, j])
        return result

    def evaluate(
        self,
        X_test,
        Y_test,
        alpha: Optional[float] = None,
        snr: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Predict intervals on ``X_test`` and evaluate coverage/sharpness against ``Y_test``."""
        a = self.alpha if alpha is None else float(alpha)
        lo, hi = self.predict_interval(X_test, alpha=a)
        return evaluate_intervals(lo, hi, Y_test, alpha=a, snr=snr)
