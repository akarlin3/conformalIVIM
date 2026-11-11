"""Tests for 1D/2D NLLS and gradient-boosted quantile IVIM base estimators."""
import numpy as np
import pytest

from conformal_ivim.estimators import (
    IVIMQuantileRegressor,
    NLLSEstimator,
    fit_nlls,
    fit_nlls_batch,
    fit_varpro_2d_batch,
)
from conformal_ivim.forward import (
    CLINICAL_B,
    DEFAULT_B_VALUES,
    generate_cohort,
    ivim_signal,
    ivim_signal_2d,
)

CLEAN_CASES = [
    (0.8e-3, 15e-3, 0.10),
    (1.5e-3, 50e-3, 0.20),
    (1.2e-3, 30e-3, 0.30),
    (2.5e-3, 90e-3, 0.35),
    (1.0e-3, 10e-3, 0.05),
]


@pytest.mark.parametrize("D,Dstar,f", CLEAN_CASES)
def test_clean_recovery_returns_truth(D, Dstar, f):
    s = ivim_signal(DEFAULT_B_VALUES, D, Dstar, f, S0=1.0)
    est = fit_nlls(s, DEFAULT_B_VALUES)
    assert est["D"] == pytest.approx(D, rel=1e-2)
    assert est["Dstar"] == pytest.approx(Dstar, rel=1e-2)
    assert est["f"] == pytest.approx(f, rel=1e-2)


def test_clean_recovery_grid_max_error_small():
    rng = np.random.default_rng(0)
    max_rel = 0.0
    for _ in range(60):
        D = rng.uniform(0.5e-3, 3.0e-3)
        Dstar = rng.uniform(10e-3, 100e-3)
        f = rng.uniform(0.05, 0.40)
        s = ivim_signal(DEFAULT_B_VALUES, D, Dstar, f, S0=1.0)
        est = fit_nlls(s, DEFAULT_B_VALUES)
        for truth, key in [(D, "D"), (Dstar, "Dstar"), (f, "f")]:
            max_rel = max(max_rel, abs(est[key] - truth) / truth)
    assert max_rel < 1e-2


def test_batch_fit_shape_and_order():
    truths = np.array([[1.0e-3, 20e-3, 0.1], [1.8e-3, 60e-3, 0.25]])
    signals = np.stack([ivim_signal(DEFAULT_B_VALUES, *t) for t in truths])
    out = fit_nlls_batch(signals, DEFAULT_B_VALUES)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, truths, rtol=1e-2)

    est_cls = NLLSEstimator(b=DEFAULT_B_VALUES).fit()
    np.testing.assert_allclose(est_cls.predict(signals), out)


def test_fit_varpro_2d_batch_recovers_high_dstar_parameters():
    b_flat = np.tile(CLINICAL_B, 2)
    te_flat = np.repeat(np.array([45.0, 90.0]), len(CLINICAL_B))
    D_true = np.array([1.0e-3, 1.4e-3])
    Ds_true = np.array([25.0e-3, 85.0e-3])
    f_true = np.array([0.18, 0.24])
    sig = ivim_signal_2d(
        b_flat[None, :],
        te_flat[None, :],
        D_true[:, None],
        Ds_true[:, None],
        f_true[:, None],
        S0=1.0,
        T2_tissue=50.0,
        T2_blood=150.0,
    )
    theta_hat, s0_hat, _, _ = fit_varpro_2d_batch(sig, b_flat, te_flat)
    assert np.allclose(theta_hat[:, 0], D_true, atol=1e-4)
    assert np.allclose(theta_hat[:, 1], Ds_true, rtol=0.05)
    assert np.allclose(theta_hat[:, 2], f_true, atol=0.02)
    assert np.allclose(s0_hat, 1.0, atol=0.03)


def test_quantile_regressor_fit_and_predict_interval():
    cohort = generate_cohort(250, 50, 40, snr_grid=(20, 50), seed=11)
    qreg = IVIMQuantileRegressor(
        quantile_levels=(0.05, 0.95), max_iter=60, random_state=0
    )
    qreg.fit(cohort.signals["train"], cohort.params["train"])
    lo, hi = qreg.predict_interval(cohort.signals["test"], alpha=0.10)
    assert lo.shape == (40, 3)
    assert hi.shape == (40, 3)
    assert np.mean(hi >= lo) > 0.95
