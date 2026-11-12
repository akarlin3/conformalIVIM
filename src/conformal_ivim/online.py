"""Sequential multi-fraction online conformal calibration for longitudinal IVIM MRI.

Implements online adaptive conformal prediction across fractionated radiotherapy
or longitudinal monitoring sessions (``t = 1, ..., K``) under progressive tissue,
perfusion, and scanner/SNR drift:

  * **Gibbs & Candès Adaptive Conformal Inference (``method="aci"``)**:
    Updates the per-parameter effective miscoverage target via online subgradient
    descent on the pinball loss:

        alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

    where ``err_t`` is the realized fraction-``t`` miscoverage rate.
  * **Conformal PID / Quantile Tracker (``method="conformal_pid"``)**:
    Combines proportional quantile tracking, nonlinear saturating integral
    coverage-deficit feedback, and smoothed score-trend forecasting to stabilize
    rolling coverage and interval width across monotone physiological drift.
  * **Label-Free Surrogate Shift Adaptation (``update_from_shift_score``)**:
    Couples directly to :class:`~conformal_ivim.monitor.DeploymentValidityMonitor`
    so prediction intervals widen conservatively when observable Mahalanobis or
    bi-exponential residual drift increases between fractions—before ground-truth
    reference labels arrive.

References
----------
- Gibbs & Candès (2021), Adaptive conformal inference under distribution shift, *NeurIPS*.
- Angelopoulos, Candès & Tibshirani (2023), Conformal PID control for time series
  prediction, *NeurIPS*.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from conformal_ivim.conformal import ConformalIVIM, conformal_quantile
from conformal_ivim.estimators import PARAM_NAMES
from conformal_ivim.forward import DEFAULT_B_VALUES
from conformal_ivim.metrics import empirical_coverage, interval_width
from conformal_ivim.monitor import DeploymentValidityMonitor


def _tan_integrator(x: np.ndarray, clip_frac: float = 0.92) -> np.ndarray:
    """Nonlinear tangent saturation function (Angelopoulos et al., 2023)."""
    limit = 0.5 * np.pi * float(clip_frac)
    u = np.clip(np.asarray(x, dtype=float), -limit, limit)
    return np.tan(u)


class AdaptiveConformalIVIM:
    """Sequential multi-fraction online conformal wrapper for longitudinal IVIM sessions.

    Supports ``method="aci"`` (Gibbs & Candès Adaptive Conformal Inference) and
    ``method="conformal_pid"`` (Conformal PID / quantile tracker), plus a label-free
    surrogate drift-adaptation mode (:meth:`update_from_shift_score`) coupled to
    :class:`~conformal_ivim.monitor.DeploymentValidityMonitor`.

    Parameters
    ----------
    base_conformal : ConformalIVIM or None, default=None
        Optional pre-configured or pre-calibrated :class:`~conformal_ivim.conformal.ConformalIVIM`
        instance. If ``None``, a new ``ConformalIVIM`` is created using ``estimator``,
        ``base_method``, ``b``, ``clip_bounds``, and ``random_state``.
    alpha : float, default=0.10
        Target long-run miscoverage rate in ``(0, 1)`` (nominal coverage ``1 - alpha``).
    method : {'aci', 'conformal_pid'}, default='aci'
        Online adaptation rule across longitudinal fractions ``t = 1, ..., K``.
    gamma : float, default=0.05
        Step size ``gamma > 0`` governing online responsiveness (alpha learning rate
        in ACI; proportional quantile learning rate in Conformal PID).
    ki : float, default=0.25
        Integral coverage-deficit gain used when ``method='conformal_pid'``.
    kd : float, default=0.35
        Score-trend (derivative / scorecaster) gain used when ``method='conformal_pid'``.
    trend_ema : float, default=0.50
        Exponential smoothing weight in ``(0, 1]`` for nonconformity score trends in
        ``method='conformal_pid'``.
    alpha_bounds : tuple of (float, float), default=(-0.25, 0.99)
        Allowable range for the effective ACI miscoverage target ``alpha_t``.
        Values ``<= 0`` trigger conservative finite-sample boundary extrapolation.
    shift_gain : float, default=0.04
        Sensitivity of :meth:`update_from_shift_score` to observable Mahalanobis /
        residual z-score drift when labels are unavailable.
    shift_deadzone : float, default=0.50
        Dimensionless z-score deadzone below which benign within-distribution batch
        fluctuations do not widen intervals in :meth:`update_from_shift_score`.
    estimator : Any, default=None
        Base point or quantile estimator passed to ``ConformalIVIM`` if ``base_conformal``
        is ``None``.
    base_method : {'auto', 'split', 'cqr'}, default='auto'
        Static calibration score type used by the underlying ``ConformalIVIM``.
    b : sequence of float, default=DEFAULT_B_VALUES
        Diffusion-weighting b-values in ``s/mm^2``.
    clip_bounds : sequence of (low, high) pairs or None, default=None
        Optional physiological parameter clipping bounds per output parameter.
    monitor : DeploymentValidityMonitor or None, default=None
        Optional pre-fitted deployment validity monitor for label-free shift updates.
    random_state : int, default=0
        Deterministic ``PCG64`` seed.
    """

    def __init__(
        self,
        base_conformal: Optional[ConformalIVIM] = None,
        *,
        alpha: float = 0.10,
        method: str = "aci",
        gamma: float = 0.05,
        ki: float = 0.25,
        kd: float = 0.85,
        trend_ema: float = 0.50,
        alpha_bounds: Tuple[float, float] = (-0.25, 0.99),
        window_size: int = 180,
        shift_gain: float = 0.04,
        shift_deadzone: float = 0.50,
        estimator: Any = None,
        base_method: str = "auto",
        b: Sequence[float] = DEFAULT_B_VALUES,
        clip_bounds: Optional[Sequence[Tuple[float, float]]] = None,
        monitor: Optional[DeploymentValidityMonitor] = None,
        random_state: int = 0,
    ) -> None:
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha!r}")
        if method not in ("aci", "conformal_pid"):
            raise ValueError(
                f"method must be 'aci' or 'conformal_pid', got {method!r}"
            )
        if float(gamma) <= 0.0:
            raise ValueError(f"gamma must be strictly positive, got {gamma!r}")
        if not (0.0 < float(trend_ema) <= 1.0):
            raise ValueError(f"trend_ema must lie in (0, 1], got {trend_ema!r}")
        if float(alpha_bounds[0]) >= float(alpha_bounds[1]):
            raise ValueError(f"Invalid alpha_bounds {alpha_bounds!r}")
        if int(window_size) < 10:
            raise ValueError(f"window_size must be >= 10, got {window_size!r}")

        self.base_conformal = base_conformal
        self.alpha = float(alpha)
        self.method = method
        self.gamma = float(gamma)
        self.ki = float(ki)
        self.kd = float(kd)
        self.trend_ema = float(trend_ema)
        self.alpha_bounds = (float(alpha_bounds[0]), float(alpha_bounds[1]))
        self.window_size = int(window_size)
        self.shift_gain = float(shift_gain)
        self.shift_deadzone = float(shift_deadzone)
        self.estimator = estimator
        self.base_method = base_method
        self.b = np.asarray(b, dtype=float)
        self.clip_bounds = clip_bounds
        self.monitor = monitor
        self.random_state = int(random_state)

        if base_conformal is not None and hasattr(base_conformal, "scores_"):
            self.conformal_ = base_conformal
            self._init_online_state()

    def _rng(self) -> np.random.Generator:
        return np.random.Generator(np.random.PCG64(self.random_state))

    def _ensure_conformal(self) -> ConformalIVIM:
        if hasattr(self, "conformal_"):
            return self.conformal_
        if self.base_conformal is not None:
            self.conformal_ = self.base_conformal
        else:
            self.conformal_ = ConformalIVIM(
                estimator=self.estimator,
                alpha=self.alpha,
                method=self.base_method,
                b=self.b,
                clip_bounds=self.clip_bounds,
                random_state=self.random_state,
            )
        return self.conformal_

    def _refresh_score_scales(self, scores_pool: np.ndarray) -> None:
        P = self.n_outputs_
        scales = np.empty(P, dtype=float)
        max_scores = np.empty(P, dtype=float)
        for j in range(P):
            sc = np.asarray(scores_pool[:, j], dtype=float)
            q75, q25 = np.percentile(sc, [75.0, 25.0])
            iqr = float(q75 - q25)
            sd = float(np.std(sc))
            scales[j] = max(iqr, sd, abs(float(self.base_quantiles_[j])) * 0.25, 1e-8)
            max_scores[j] = float(np.max(sc)) if sc.size > 0 else 1.0
        self.score_scales_ = scales
        self.max_cal_scores_ = max_scores

    def _init_online_state(self) -> None:
        conf = self.conformal_
        P = int(conf.n_outputs_)
        self.n_outputs_ = P
        self._squeeze_1d_ = bool(getattr(conf, "_squeeze_1d_", False))
        self.fraction_ = 0
        self.alpha_t_ = np.full(P, self.alpha, dtype=float)
        self.base_quantiles_ = np.asarray(conf.quantiles_, dtype=float).copy()
        self.quantiles_t_ = self.base_quantiles_.copy()

        init_scores = np.asarray(conf.scores_, dtype=float).copy()
        if init_scores.shape[0] > self.window_size:
            init_scores = init_scores[-self.window_size :]
        self.rolling_scores_ = init_scores
        self._refresh_score_scales(self.rolling_scores_)

        # PID state variables (Angelopoulos, Candes & Tibshirani, 2023)
        self.cum_error_ = np.zeros(P, dtype=float)
        self.n_error_steps_ = 0
        self.ema_score_q_ = self.base_quantiles_.copy()
        self.prev_score_q_ = self.base_quantiles_.copy()
        self.ema_score_trend_ = np.zeros(P, dtype=float)
        self.prop_offset_ = np.zeros(P, dtype=float)
        self.shift_offset_ = np.zeros(P, dtype=float)
        self.history_: List[Dict[str, Any]] = []

    def fit(self, X_train, Y_train=None) -> "AdaptiveConformalIVIM":
        """Fit the underlying base point or quantile estimator on ``(X_train, Y_train)``."""
        conf = self._ensure_conformal()
        conf.fit(X_train, Y_train)
        return self

    def calibrate(
        self,
        X_cal,
        Y_cal,
        *,
        fit_monitor: bool = True,
        monitor_fpr: float = 0.05,
    ) -> "AdaptiveConformalIVIM":
        """Calibrate initial nonconformity scores on baseline ``(X_cal, Y_cal)``."""
        conf = self._ensure_conformal()
        conf.calibrate(X_cal, Y_cal)
        self._init_online_state()

        X_arr = np.atleast_2d(np.asarray(X_cal, dtype=float))
        if self.monitor is not None:
            self.monitor_ = self.monitor
        elif fit_monitor and X_arr.shape[1] == self.b.size:
            self.monitor_ = DeploymentValidityMonitor(
                fpr=monitor_fpr,
                seed=self.random_state,
            ).fit_signals(X_arr, b=self.b)
        else:
            self.monitor_ = None
        return self

    def fit_calibrate(
        self,
        X_train,
        Y_train,
        X_cal,
        Y_cal,
        *,
        fit_monitor: bool = True,
        monitor_fpr: float = 0.05,
    ) -> "AdaptiveConformalIVIM":
        """Fit on ``(X_train, Y_train)`` and initialize online calibration on ``(X_cal, Y_cal)``."""
        return self.fit(X_train, Y_train).calibrate(
            X_cal, Y_cal, fit_monitor=fit_monitor, monitor_fpr=monitor_fpr
        )

    def _aci_quantiles(self) -> np.ndarray:
        """Map per-parameter effective ``alpha_t_`` to rolling nonconformity quantiles."""
        P = self.n_outputs_
        q = np.empty(P, dtype=float)
        pool = self.rolling_scores_
        n_pool = pool.shape[0]
        min_resolvable_alpha = 1.0 / float(n_pool + 1)

        for j in range(P):
            a_j = float(self.alpha_t_[j])
            if a_j >= 1.0:
                q[j] = 0.0
            elif a_j > min_resolvable_alpha:
                q[j] = conformal_quantile(pool[:, j], a_j)
            else:
                # Smooth conservative extrapolation beyond empirical max rolling score
                excess = (min_resolvable_alpha - a_j) / max(self.alpha, 1e-3)
                q[j] = float(self.max_cal_scores_[j]) + float(self.score_scales_[j]) * (
                    0.5 + 1.5 * max(0.0, excess)
                )
        return q

    def current_quantiles(self) -> np.ndarray:
        """Return active per-parameter conformal score thresholds at current fraction."""
        if not hasattr(self, "alpha_t_"):
            raise RuntimeError(
                "AdaptiveConformalIVIM must be calibrated before querying quantiles."
            )
        if self.method == "aci":
            q = self._aci_quantiles() + self.shift_offset_
        else:
            q = self.quantiles_t_.copy()
        return q

    def predict(self, X_test) -> np.ndarray:
        """Predict point estimates (or CQR midpoints) on ``X_test``."""
        if not hasattr(self, "conformal_"):
            raise RuntimeError("AdaptiveConformalIVIM is not fitted/calibrated.")
        return self.conformal_.predict(X_test)

    def _compute_scores(self, X_batch, Y_batch) -> np.ndarray:
        """Compute nonconformity scores on a labeled fraction ``(X_batch, Y_batch)``."""
        conf = self.conformal_
        Y_arr = np.asarray(Y_batch, dtype=float)
        if Y_arr.ndim == 1:
            Y_arr = Y_arr[:, None] if self._squeeze_1d_ else Y_arr[None, :]
        if conf.method_ == "split":
            pred = conf._predict_base_point(X_batch)
            return np.abs(Y_arr - pred)
        lo, hi = conf._predict_base_quantiles(X_batch, self.alpha)
        return np.maximum(lo - Y_arr, Y_arr - hi)

    def predict_interval(
        self,
        X_test,
        *,
        return_dict: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
        """Predict adaptive ``(1 - alpha)`` intervals using current fraction state."""
        if not hasattr(self, "alpha_t_"):
            raise RuntimeError(
                "AdaptiveConformalIVIM must be calibrated before calling predict_interval."
            )
        conf = self.conformal_
        P = self.n_outputs_
        q_used = self.current_quantiles()

        if conf.method_ == "split":
            test_pred = conf._predict_base_point(X_test)
            lower = test_pred - q_used[None, :]
            upper = test_pred + q_used[None, :]
        else:
            test_lo, test_hi = conf._predict_base_quantiles(X_test, self.alpha)
            if conf.method_ == "cqr":
                lower = test_lo - q_used[None, :]
                upper = test_hi + q_used[None, :]
            else:
                lower = test_lo - q_used[None, :]
                upper = test_hi + q_used[None, :]

        # Ensure non-crossing intervals if quantile tracker becomes negative in CQR
        mid = 0.5 * (lower + upper)
        crossed = lower > upper
        if np.any(crossed):
            lower = np.where(crossed, mid, lower)
            upper = np.where(crossed, mid, upper)

        clip_b = self.clip_bounds if self.clip_bounds is not None else conf.clip_bounds
        if clip_b is not None:
            lower = lower.copy()
            upper = upper.copy()
            for j, (lb, ub) in enumerate(clip_b[:P]):
                lower[:, j] = np.clip(lower[:, j], lb, ub)
                upper[:, j] = np.clip(upper[:, j], lb, ub)

        if self._squeeze_1d_:
            lo_out, hi_out = lower[:, 0], upper[:, 0]
        else:
            lo_out, hi_out = lower, upper

        if not return_dict:
            return lo_out, hi_out

        out: Dict[str, Any] = {
            "lower": lo_out,
            "upper": hi_out,
            "width": hi_out - lo_out,
            "quantiles": q_used.copy(),
            "alpha_t": self.alpha_t_.copy(),
            "nominal_alpha": self.alpha,
            "fraction": self.fraction_,
            "method": self.method,
        }
        if not self._squeeze_1d_:
            for j, nm in enumerate(PARAM_NAMES[:P]):
                out[nm] = (lower[:, j], upper[:, j])
        return out

    def update(
        self,
        X_t,
        Y_t,
        *,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Update online conformal state after observing fraction-``t`` labels ``Y_t``.

        Parameters
        ----------
        X_t : array-like of shape (N_t, n_features)
            Fraction-``t`` DWI signals or input features.
        Y_t : array-like of shape (N_t, n_outputs) or (N_t,)
            Observed fraction-``t`` ground-truth / reference IVIM parameters.
        lower, upper : np.ndarray or None, optional
            Pre-computed intervals predicted prior to observing ``Y_t``. If ``None``,
            intervals are evaluated at the current pre-update online state.
        """
        if not hasattr(self, "alpha_t_"):
            raise RuntimeError(
                "AdaptiveConformalIVIM must be calibrated before calling update."
            )
        P = self.n_outputs_
        X_arr = np.atleast_2d(np.asarray(X_t, dtype=float))
        Y_arr = np.asarray(Y_t, dtype=float)
        if self._squeeze_1d_:
            Y_mat = Y_arr.reshape(-1, 1)
        else:
            Y_mat = Y_arr[None, :] if Y_arr.ndim == 1 else Y_arr

        if lower is None or upper is None:
            lo_pred, hi_pred = self.predict_interval(X_arr, return_dict=False)
        else:
            lo_pred = np.asarray(lower, dtype=float)
            hi_pred = np.asarray(upper, dtype=float)

        if self._squeeze_1d_:
            lo_mat = lo_pred.reshape(-1, 1)
            hi_mat = hi_pred.reshape(-1, 1)
        else:
            lo_mat = lo_pred[None, :] if lo_pred.ndim == 1 else lo_pred
            hi_mat = hi_pred[None, :] if hi_pred.ndim == 1 else hi_pred

        alpha_prev = self.alpha_t_.copy()
        q_prev = self.current_quantiles().copy()

        missed = (Y_mat < lo_mat) | (Y_mat > hi_mat)
        err_t = np.mean(missed.astype(float), axis=0)  # shape (P,)
        cov_t = 1.0 - err_t
        mean_w = np.mean(hi_mat - lo_mat, axis=0)

        scores_t = self._compute_scores(X_arr, Y_mat)
        emp_score_q = np.array(
            [
                conformal_quantile(scores_t[:, j], self.alpha)
                for j in range(P)
            ],
            dtype=float,
        )

        # Slide rolling nonconformity score pool with newest observed fraction scores
        combined_scores = np.vstack([self.rolling_scores_, scores_t])
        if combined_scores.shape[0] > self.window_size:
            combined_scores = combined_scores[-self.window_size :]
        self.rolling_scores_ = combined_scores
        self._refresh_score_scales(self.rolling_scores_)

        self.fraction_ += 1
        self.n_error_steps_ += 1
        self.cum_error_ = self.cum_error_ + (err_t - self.alpha)

        if self.method == "aci":
            # Gibbs & Candès (2021): alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
            alpha_next = self.alpha_t_ + self.gamma * (self.alpha - err_t)
            self.alpha_t_ = np.clip(alpha_next, self.alpha_bounds[0], self.alpha_bounds[1])
            self.quantiles_t_ = self._aci_quantiles() + self.shift_offset_
        else:
            # Conformal PID (Angelopoulos, Candès & Tibshirani, 2023):
            # 1) Proportional quantile-tracking offset update
            self.prop_offset_ = self.prop_offset_ + (
                self.gamma * self.score_scales_ * (err_t - self.alpha)
            )
            # 2) Nonlinear saturating integral coverage-deficit term
            norm_int = self.cum_error_ / np.sqrt(float(self.n_error_steps_) + 1.0)
            i_term = self.ki * self.score_scales_ * _tan_integrator(norm_int)
            # 3) Scorecaster (EWMA level + smoothed trend forecast of score quantiles)
            raw_trend = emp_score_q - self.prev_score_q_
            self.ema_score_q_ = (
                self.trend_ema * emp_score_q
                + (1.0 - self.trend_ema) * self.ema_score_q_
            )
            self.ema_score_trend_ = (
                self.trend_ema * raw_trend
                + (1.0 - self.trend_ema) * self.ema_score_trend_
            )
            scorecaster_q = self.ema_score_q_ + self.kd * self.ema_score_trend_
            self.prev_score_q_ = emp_score_q.copy()
            # Decay surrogate shift offset once true ground-truth labels arrive
            self.shift_offset_ = 0.5 * self.shift_offset_

            min_q = -0.5 * self.base_quantiles_ if self.conformal_.method_ == "cqr" else np.zeros(P)
            self.quantiles_t_ = np.maximum(
                scorecaster_q + self.prop_offset_ + i_term + self.shift_offset_, min_q
            )
            # Mirror an equivalent effective alpha_t diagnostic
            eff_alpha = self.alpha_t_ + self.gamma * (self.alpha - err_t)
            self.alpha_t_ = np.clip(eff_alpha, self.alpha_bounds[0], self.alpha_bounds[1])

        record: Dict[str, Any] = {
            "fraction": self.fraction_,
            "mode": "labeled",
            "method": self.method,
            "err_t": err_t.copy(),
            "coverage_t": cov_t.copy(),
            "mean_width_t": mean_w.copy(),
            "alpha_prev": alpha_prev,
            "alpha_t": self.alpha_t_.copy(),
            "quantiles_prev": q_prev,
            "quantiles_t": self.quantiles_t_.copy(),
            "score_quantile_t": emp_score_q,
        }
        self.history_.append(record)
        return record

    def update_from_shift_score(
        self,
        shift_or_signals: Union[Dict[str, Any], np.ndarray, float],
        *,
        monitor: Optional[DeploymentValidityMonitor] = None,
        b: Optional[Sequence[float]] = None,
        relax_when_in_dist: bool = True,
    ) -> Dict[str, Any]:
        """Conservatively adapt online intervals from label-free observable drift.

        Accepts:
          1. A ``dict`` returned by :meth:`DeploymentValidityMonitor.evaluate`,
          2. An unlabeled signal array ``X_t`` of shape ``(N_t, n_b)`` (evaluated
             using ``monitor`` or the monitor fitted during :meth:`calibrate`), or
          3. A scalar / per-parameter non-negative drift z-score.

        When observable Mahalanobis or bi-exponential residual drift increases
        between fractions, ``alpha_t`` decreases (ACI) and ``quantiles_t`` widen
        conservatively before ground-truth labels arrive.
        """
        if not hasattr(self, "alpha_t_"):
            raise RuntimeError(
                "AdaptiveConformalIVIM must be calibrated before calling update_from_shift_score."
            )
        P = self.n_outputs_
        alpha_prev = self.alpha_t_.copy()
        q_prev = self.current_quantiles().copy()

        report: Optional[Dict[str, Any]] = None
        fires = False
        maha_excess = 0.0
        resid_excess = 0.0

        if isinstance(shift_or_signals, dict):
            report = shift_or_signals
        elif np.ndim(shift_or_signals) >= 2:
            mon = monitor if monitor is not None else getattr(self, "monitor_", None)
            if mon is None:
                raise ValueError(
                    "Pass a fitted `monitor=DeploymentValidityMonitor(...)` or calibrate with "
                    "`fit_monitor=True` when passing raw signals to update_from_shift_score."
                )
            b_use = self.b if b is None else np.asarray(b, dtype=float)
            report = mon.evaluate_signals(shift_or_signals, b=b_use)

        if report is not None:
            fires = bool(report.get("fires", False))
            maha_info = report.get("maha", {})
            resid_info = report.get("resid", {})

            # Compute batch effect-size drift relative to calibration distribution
            if "score_test" in maha_info and hasattr(self, "monitor_") and self.monitor_ is not None:
                cal_m = self.monitor_.cal_maha_
                maha_excess = max(
                    0.0,
                    (float(maha_info["stat"]) - float(np.mean(cal_m)))
                    / (float(np.std(cal_m)) + 1e-12),
                )
            else:
                # Normalize z_score conservatively by 1/sqrt(n) scale or threshold ratio
                thr_m = max(abs(float(maha_info.get("threshold", 1.0))), 1e-8)
                stat_m = float(maha_info.get("stat", 0.0))
                z_m = max(0.0, float(maha_info.get("z_score", 0.0)))
                maha_excess = max(0.0, (stat_m - thr_m) / thr_m * 3.0 + 0.25 * min(z_m, 8.0))

            thr_r = max(abs(float(resid_info.get("threshold", 1.0))), 1e-8)
            stat_r = float(resid_info.get("stat", 0.0))
            z_r = max(0.0, abs(float(resid_info.get("std_z_score", resid_info.get("z_score", 0.0)))))
            resid_excess = max(0.0, (stat_r - thr_r) / thr_r * 3.0 + 0.25 * min(z_r, 8.0))

            raw_shift = max(maha_excess, resid_excess)
            if fires:
                raw_shift = max(raw_shift, self.shift_deadzone + 0.75)
            eff_shift = np.full(P, max(0.0, raw_shift - self.shift_deadzone), dtype=float)
        else:
            s_arr = np.asarray(shift_or_signals, dtype=float)
            if s_arr.ndim == 0:
                raw_val = max(0.0, float(s_arr))
                fires = raw_val > self.shift_deadzone
                eff_shift = np.full(P, max(0.0, raw_val - self.shift_deadzone), dtype=float)
            else:
                s_vec = np.broadcast_to(np.maximum(s_arr.ravel(), 0.0), (P,))
                fires = bool(np.any(s_vec > self.shift_deadzone))
                eff_shift = np.maximum(0.0, s_vec - self.shift_deadzone)

        self.fraction_ += 1
        capped_shift = np.minimum(eff_shift, 6.0)

        if np.any(capped_shift > 0.0):
            # Conservative widening proportional to observable drift severity
            delta_alpha = self.shift_gain * capped_shift
            self.alpha_t_ = np.clip(
                self.alpha_t_ - delta_alpha,
                self.alpha_bounds[0],
                self.alpha_bounds[1],
            )
            widen = (self.shift_gain * 4.0) * self.score_scales_ * capped_shift
            self.shift_offset_ = self.shift_offset_ + widen
            if self.method == "aci":
                self.quantiles_t_ = self._aci_quantiles() + self.shift_offset_
            else:
                self.quantiles_t_ = self.quantiles_t_ + widen
        elif relax_when_in_dist and not fires:
            # Gentle relaxation back toward nominal calibration when batch is in-distribution
            self.alpha_t_ = self.alpha_t_ + 0.20 * self.gamma * (self.alpha - self.alpha_t_)
            self.alpha_t_ = np.clip(self.alpha_t_, self.alpha_bounds[0], self.alpha_bounds[1])
            self.shift_offset_ = 0.80 * self.shift_offset_
            if self.method == "aci":
                self.quantiles_t_ = self._aci_quantiles() + self.shift_offset_
            else:
                self.quantiles_t_ = self.quantiles_t_ + 0.20 * (
                    self.base_quantiles_ - self.quantiles_t_
                )

        record = {
            "fraction": self.fraction_,
            "mode": "label_free_shift",
            "method": self.method,
            "fires": fires,
            "shift_score": capped_shift.copy(),
            "maha_excess": float(maha_excess),
            "resid_excess": float(resid_excess),
            "alpha_prev": alpha_prev,
            "alpha_t": self.alpha_t_.copy(),
            "quantiles_prev": q_prev,
            "quantiles_t": self.quantiles_t_.copy(),
        }
        self.history_.append(record)
        return record

    def run_fractions(
        self,
        fractions_X: Sequence[np.ndarray],
        fractions_Y: Optional[Sequence[Optional[np.ndarray]]] = None,
        *,
        monitor: Optional[DeploymentValidityMonitor] = None,
    ) -> Dict[str, Any]:
        """Execute a sequential multi-fraction session ``t = 1, ..., K``.

        At each fraction ``t``:
          1. Predicts ``(lower_t, upper_t)`` using the pre-fraction online state,
          2. Evaluates empirical coverage & width if ``fractions_Y[t]`` is available,
          3. Updates online parameters via :meth:`update` (when ``fractions_Y[t]`` is
             given) or :meth:`update_from_shift_score` (when unlabeled).
        """
        K = len(fractions_X)
        if K == 0:
            raise ValueError("fractions_X must contain at least one fraction.")
        if fractions_Y is not None and len(fractions_Y) != K:
            raise ValueError(
                f"Length of fractions_Y ({len(fractions_Y)}) != len(fractions_X) ({K})."
            )

        intervals: List[Tuple[np.ndarray, np.ndarray]] = []
        coverages: List[Optional[np.ndarray]] = []
        widths: List[np.ndarray] = []
        alpha_traj: List[np.ndarray] = []
        quantile_traj: List[np.ndarray] = []

        for t_idx in range(K):
            X_t = np.atleast_2d(np.asarray(fractions_X[t_idx], dtype=float))
            lo_t, hi_t = self.predict_interval(X_t, return_dict=False)
            intervals.append((lo_t, hi_t))

            lo_2d = lo_t.reshape(-1, 1) if self._squeeze_1d_ else lo_t
            hi_2d = hi_t.reshape(-1, 1) if self._squeeze_1d_ else hi_t
            widths.append(np.mean(interval_width(lo_2d, hi_2d), axis=0))
            alpha_traj.append(self.alpha_t_.copy())
            quantile_traj.append(self.current_quantiles().copy())

            Y_t = None if fractions_Y is None else fractions_Y[t_idx]
            if Y_t is not None:
                Y_arr = np.asarray(Y_t, dtype=float)
                Y_2d = Y_arr.reshape(-1, 1) if self._squeeze_1d_ else Y_arr
                cov_vec = np.array(
                    [
                        empirical_coverage(lo_2d[:, j], hi_2d[:, j], Y_2d[:, j])
                        for j in range(self.n_outputs_)
                    ],
                    dtype=float,
                )
                coverages.append(cov_vec)
                self.update(X_t, Y_t, lower=lo_t, upper=hi_t)
            else:
                coverages.append(None)
                self.update_from_shift_score(X_t, monitor=monitor)

        labeled_covs = [c for c in coverages if c is not None]
        mean_session_cov = (
            np.mean(np.vstack(labeled_covs), axis=0) if labeled_covs else None
        )
        return {
            "n_fractions": K,
            "method": self.method,
            "nominal_coverage": 1.0 - self.alpha,
            "intervals": intervals,
            "coverage_per_fraction": coverages,
            "mean_coverage": mean_session_cov,
            "width_per_fraction": np.vstack(widths),
            "alpha_trajectory": np.vstack(alpha_traj),
            "quantile_trajectory": np.vstack(quantile_traj),
        }
