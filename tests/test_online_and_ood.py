"""Deterministic unit tests for online multi-fraction ACI/PID calibration and
single-voxel OOD detection (conformalIVIM v1.2.0).
"""
from __future__ import annotations

import numpy as np
import pytest

import conformal_ivim
from conformal_ivim import (
    DEFAULT_B_VALUES,
    AdaptiveConformalIVIM,
    ConformalIVIM,
    DeploymentValidityMonitor,
    VoxelOODDetector,
    empirical_coverage,
    simulate_ivim_signal,
)


def _pcg_rng(seed: int = 2026) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def _sample_fraction(
    rng: np.random.Generator,
    n: int,
    *,
    snr: float = 35.0,
    d_shift: float = 0.0,
    f_scale: float = 1.0,
    b: np.ndarray = DEFAULT_B_VALUES,
):
    D = np.clip(rng.uniform(0.6e-3, 2.2e-3, size=n) + d_shift, 0.2e-3, 3.8e-3)
    Dstar = rng.uniform(12.0e-3, 85.0e-3, size=n)
    f = np.clip(rng.uniform(0.06, 0.32, size=n) * f_scale, 0.01, 0.65)
    Y = np.column_stack([D, Dstar, f])
    X = simulate_ivim_signal(b, D=D, Dstar=Dstar, f=f, snr=snr, noise="rician", rng=rng)
    return X, Y


class _FastBiasedPointEstimator:
    """Fast deterministic linear-log point estimator for multi-fraction unit tests."""

    def __init__(self, b: np.ndarray = DEFAULT_B_VALUES):
        self.b = np.asarray(b, dtype=float)

    def fit(self, X, Y):
        X_arr = np.asarray(X, dtype=float)
        Phi = np.column_stack([np.ones(X_arr.shape[0]), X_arr, np.log(np.clip(X_arr, 1e-4, None))])
        self.W_, *_ = np.linalg.lstsq(Phi, np.asarray(Y, dtype=float), rcond=None)
        return self

    def predict(self, X) -> np.ndarray:
        X_arr = np.atleast_2d(np.asarray(X, dtype=float))
        Phi = np.column_stack([np.ones(X_arr.shape[0]), X_arr, np.log(np.clip(X_arr, 1e-4, None))])
        return Phi @ self.W_


def test_version_and_top_level_exports():
    assert conformal_ivim.__version__ == "1.3.0"
    assert hasattr(conformal_ivim, "AdaptiveConformalIVIM")
    assert hasattr(conformal_ivim, "VoxelOODDetector")


def test_aci_exact_update_formula_and_drift_recovery():
    rng = _pcg_rng(101)
    X_train, Y_train = _sample_fraction(rng, 250, snr=40.0)
    X_cal, Y_cal = _sample_fraction(rng, 220, snr=40.0)

    est = _FastBiasedPointEstimator()
    static_cp = ConformalIVIM(estimator=est, method="split", alpha=0.10, random_state=0)
    static_cp.fit_calibrate(X_train, Y_train, X_cal, Y_cal)

    aci = AdaptiveConformalIVIM(
        estimator=_FastBiasedPointEstimator(),
        base_method="split",
        alpha=0.10,
        method="aci",
        gamma=0.08,
        random_state=0,
    ).fit_calibrate(X_train, Y_train, X_cal, Y_cal, fit_monitor=False)

    # Simulate K = 12 progressive radiotherapy fractions with severe SNR + diffusivity drift
    fractions_X, fractions_Y = [], []
    for t in range(12):
        snr_t = max(7.5, 38.0 - 2.8 * t)
        d_shift_t = 0.08e-3 * t
        Xt, Yt = _sample_fraction(rng, 90, snr=snr_t, d_shift=d_shift_t)
        fractions_X.append(Xt)
        fractions_Y.append(Yt)

    # Verify exact Gibbs & Candes step on fraction 1 before running remaining fractions
    lo_0, hi_0 = aci.predict_interval(fractions_X[0])
    rec0 = aci.update(fractions_X[0], fractions_Y[0], lower=lo_0, upper=hi_0)
    expected_alpha_1 = np.clip(0.10 + 0.08 * (0.10 - rec0["err_t"]), *aci.alpha_bounds)
    np.testing.assert_allclose(rec0["alpha_t"], expected_alpha_1, rtol=1e-12, atol=1e-12)

    # Re-run full 12-fraction session from fresh initialization and compare against static CP
    aci_full = AdaptiveConformalIVIM(
        estimator=_FastBiasedPointEstimator(),
        base_method="split",
        alpha=0.10,
        method="aci",
        gamma=0.08,
        random_state=0,
    ).fit_calibrate(X_train, Y_train, X_cal, Y_cal, fit_monitor=False)

    session = aci_full.run_fractions(fractions_X, fractions_Y)
    static_cov_late = []
    for Xt, Yt in zip(fractions_X[4:], fractions_Y[4:]):
        lo_s, hi_s = static_cp.predict_interval(Xt)
        static_cov_late.append(
            [empirical_coverage(lo_s[:, j], hi_s[:, j], Yt[:, j]) for j in range(3)]
        )
    static_late_mean = np.mean(np.asarray(static_cov_late), axis=0)
    aci_late_mean = np.mean(np.vstack(session["coverage_per_fraction"][4:]), axis=0)

    # Static split conformal collapses under progressive drift; ACI recovers coverage
    assert np.all(aci_late_mean > static_late_mean + 0.05)
    assert np.all(aci_late_mean >= 0.82)
    assert session["width_per_fraction"][-1, 0] > session["width_per_fraction"][0, 0]


def test_conformal_pid_tracks_drift_and_stabilizes_rolling_coverage():
    rng = _pcg_rng(202)
    X_train, Y_train = _sample_fraction(rng, 250, snr=40.0)
    X_cal, Y_cal = _sample_fraction(rng, 220, snr=40.0)

    pid = AdaptiveConformalIVIM(
        estimator=_FastBiasedPointEstimator(),
        base_method="split",
        alpha=0.10,
        method="conformal_pid",
        gamma=0.15,
        ki=0.35,
        kd=0.40,
        random_state=7,
    ).fit_calibrate(X_train, Y_train, X_cal, Y_cal, fit_monitor=False)

    fractions_X, fractions_Y = [], []
    for t in range(14):
        snr_t = max(8.0, 38.0 - 2.4 * t)
        Xt, Yt = _sample_fraction(rng, 100, snr=snr_t, d_shift=0.07e-3 * t)
        fractions_X.append(Xt)
        fractions_Y.append(Yt)

    res = pid.run_fractions(fractions_X, fractions_Y)
    assert res["n_fractions"] == 14
    assert res["method"] == "conformal_pid"
    assert res["mean_coverage"].shape == (3,)
    # Rolling session coverage stays close to nominal 0.90 across all 3 parameters
    assert np.all(res["mean_coverage"] >= 0.84)
    assert np.all(res["quantile_trajectory"][-1] > res["quantile_trajectory"][0])

    d_info = pid.predict_interval(fractions_X[-1], return_dict=True)
    assert set(("lower", "upper", "width", "D", "Dstar", "f")).issubset(d_info.keys())
    assert np.all(d_info["upper"] >= d_info["lower"])


def test_label_free_shift_score_widens_intervals_with_deployment_monitor():
    rng = _pcg_rng(303)
    X_train, Y_train = _sample_fraction(rng, 120, snr=35.0)
    X_cal, Y_cal = _sample_fraction(rng, 110, snr=35.0)
    X_in, _ = _sample_fraction(rng, 60, snr=35.0)
    X_drift, _ = _sample_fraction(rng, 60, snr=7.0, d_shift=0.6e-3)

    monitor = DeploymentValidityMonitor(fpr=0.05, n_null=80, seed=11).fit_signals(X_cal)

    for method in ("aci", "conformal_pid"):
        online_cp = AdaptiveConformalIVIM(
            estimator=_FastBiasedPointEstimator(),
            base_method="split",
            alpha=0.10,
            method=method,
            gamma=0.06,
            shift_gain=0.05,
            monitor=monitor,
            random_state=11,
        ).fit_calibrate(X_train, Y_train, X_cal, Y_cal, fit_monitor=False)

        lo_init, hi_init = online_cp.predict_interval(X_drift)
        w_init = np.mean(hi_init - lo_init, axis=0)

        # In-distribution batch does not spuriously blow up interval widths
        rec_in = online_cp.update_from_shift_score(X_in)
        assert rec_in["mode"] == "label_free_shift"
        lo_in, hi_in = online_cp.predict_interval(X_drift)
        w_after_in = np.mean(hi_in - lo_in, axis=0)
        assert np.all(w_after_in <= w_init * 1.15)

        # Shifted unlabeled fraction fires monitor and widens intervals before labels arrive
        report_drift = monitor.evaluate_signals(X_drift)
        assert report_drift["fires"] is True
        rec_drift = online_cp.update_from_shift_score(report_drift)
        assert rec_drift["fires"] is True
        assert np.all(rec_drift["alpha_t"] < rec_drift["alpha_prev"])
        lo_widened, hi_widened = online_cp.predict_interval(X_drift)
        w_widened = np.mean(hi_widened - lo_widened, axis=0)
        assert np.all(w_widened > w_after_in)


def test_voxel_ood_detector_finite_sample_fpr_and_corruption_power():
    rng = _pcg_rng(404)
    X_train, Y_train = _sample_fraction(rng, 160, snr=32.0)
    X_cal, Y_cal = _sample_fraction(rng, 180, snr=32.0)
    X_clean, Y_clean = _sample_fraction(rng, 150, snr=32.0)

    detector = VoxelOODDetector(alpha_ood=0.05, n_neighbors=8, seed=0).fit(X_cal)
    clean_res = detector.detect(X_clean)

    # Finite-sample conformal p-value properties on exchangeable test voxels
    p_clean = clean_res["p_values"]
    assert p_clean.shape == (150,)
    assert np.all((p_clean >= 1.0 / (180 + 1) - 1e-12) & (p_clean <= 1.0))
    # Exchangeable false-alarm rate controlled near alpha_ood = 0.05
    assert clean_res["ood_fraction"] <= 0.11
    assert clean_res["repeat_scan"] is False

    # Construct corrupted voxels: necrotic cavity, RF/motion spike, susceptibility dropout
    n_corrupt = 45
    X_corrupt, _ = _sample_fraction(rng, n_corrupt, snr=32.0)
    # 1) Necrotic free-fluid cavity (D = 3.15e-3, f ~ 0)
    X_corrupt[:15] = simulate_ivim_signal(
        DEFAULT_B_VALUES,
        D=np.full(15, 3.15e-3),
        Dstar=np.full(15, 25.0e-3),
        f=np.full(15, 0.002),
        snr=35.0,
        noise="rician",
        rng=rng,
    )
    # 2) Motion / RF rebound spikes at mid/high b-values
    X_corrupt[15:30, 8:14] = 1.65
    # 3) Susceptibility / slice signal dropout to zero across low/mid b-values
    X_corrupt[30:45, 2:12] = 0.0

    X_mixed = np.vstack([X_clean[:60], X_corrupt])
    Y_mixed = np.vstack([
        Y_clean[:60],
        np.column_stack([
            np.full(n_corrupt, 2.9e-3),
            np.full(n_corrupt, 40.0e-3),
            np.full(n_corrupt, 0.02),
        ]),
    ])

    cp = ConformalIVIM(
        estimator=_FastBiasedPointEstimator(),
        method="split",
        alpha=0.10,
        random_state=0,
    ).fit_calibrate(X_train, Y_train, X_cal, Y_cal)

    mixed_res = detector.detect(X_mixed, conformal_model=cp)
    corrupt_tpr = float(np.mean(mixed_res["ood_mask"][60:]))
    inlier_fpr = float(np.mean(mixed_res["ood_mask"][:60]))

    assert corrupt_tpr >= 0.90
    assert inlier_fpr <= 0.12
    assert mixed_res["repeat_scan"] is True  # >20% corrupted voxels triggers repeat scan

    # Verify adjusted interval guarantees preserve marginal coverage across mixed cohort
    adj = mixed_res["adjusted_intervals"]
    assert adj["adjusted_marginal_coverage_bound"] == pytest.approx(0.90)
    assert adj["selective_inlier_coverage_bound"] == pytest.approx(0.85)
    for j in range(3):
        cov_j = empirical_coverage(adj["lower"][:, j], adj["upper"][:, j], Y_mixed[:, j])
        assert cov_j >= 0.85


def test_voxel_ood_fdr_and_determinism():
    rng1 = _pcg_rng(505)
    rng2 = _pcg_rng(505)
    X_cal1, _ = _sample_fraction(rng1, 220, snr=30.0)
    X_cal2, _ = _sample_fraction(rng2, 220, snr=30.0)
    np.testing.assert_array_equal(X_cal1, X_cal2)

    det_fdr = VoxelOODDetector(
        alpha_ood=0.05, ood_action="inflate", inflate_factor=3.0, fdr_control=True
    ).fit(X_cal1)
    X_spike = X_cal1[:20].copy()
    X_spike[:4, 5:10] = 2.1
    lo_dummy = np.zeros((20, 3))
    hi_dummy = np.ones((20, 3)) * 0.1

    res = det_fdr.detect(X_spike, lower=lo_dummy, upper=hi_dummy, nominal_alpha=0.10)
    assert np.all(res["ood_mask"][:4])
    np.testing.assert_allclose(res["adjusted_intervals"]["width"][:4], 0.3)
