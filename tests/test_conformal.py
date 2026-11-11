"""Correctness tests for split-conformal, CQR, weighted/localized/conditional
conformal prediction, and the high-level ``ConformalIVIM`` wrapper.
"""
import numpy as np
import pytest
from scipy.stats import beta as beta_dist
from sklearn.ensemble import RandomForestRegressor

from conformal_ivim import ConformalIVIM
from conformal_ivim.conformal import (
    conditional_conformal,
    conformal_quantile,
    cqr,
    empirical_coverage,
    interval_width,
    localized_conformal,
    split_conformal,
    weighted_conformal_quantile,
    weighted_split_conformal,
)
from conformal_ivim.forward import generate_cohort


# --------------------------------------------------------------------------- #
# Core finite-sample split-conformal & CQR guarantee tests (Gauge 01)
# --------------------------------------------------------------------------- #
def test_conformal_quantile_is_kth_order_statistic():
    scores = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5], float)
    assert conformal_quantile(scores, 0.1) == 9.0
    assert conformal_quantile(scores, 0.5) == np.sort(scores)[4]


def test_conformal_quantile_infinite_when_calibration_too_small():
    scores = np.arange(5, dtype=float)
    assert conformal_quantile(scores, 0.01) == np.inf


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20, 0.30])
def test_split_conformal_tracks_nominal_on_gaussian(alpha):
    rng = np.random.default_rng(0)
    mu, sigma = 5.0, 2.0
    n_cal, n_test = 4000, 20000
    cal_true = rng.normal(mu, sigma, n_cal)
    test_true = rng.normal(mu, sigma, n_test)
    cal_pred = np.full(n_cal, mu)
    test_pred = np.full(n_test, mu)
    lo, hi, _ = split_conformal(cal_pred, cal_true, test_pred, alpha)
    cov = empirical_coverage(lo, hi, test_true)
    assert cov == pytest.approx(1 - alpha, abs=0.02)


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_split_conformal_covers_despite_biased_base(alpha):
    rng = np.random.default_rng(1)
    mu, sigma, bias = 0.0, 1.0, 4.0
    n_cal, n_test = 4000, 20000
    cal_true = rng.normal(mu, sigma, n_cal)
    test_true = rng.normal(mu, sigma, n_test)
    cal_pred = np.full(n_cal, mu + bias)
    test_pred = np.full(n_test, mu + bias)
    lo, hi, _ = split_conformal(cal_pred, cal_true, test_pred, alpha)
    cov = empirical_coverage(lo, hi, test_true)
    assert cov == pytest.approx(1 - alpha, abs=0.02)


def test_split_conformal_marginal_guarantee_in_expectation():
    alpha = 0.10
    mu, sigma = 1.0, 0.5
    covs = []
    for s in range(40):
        rng = np.random.default_rng(100 + s)
        cal_true = rng.normal(mu, sigma, 1000)
        test_true = rng.normal(mu, sigma, 4000)
        lo, hi, _ = split_conformal(
            np.full(1000, mu), cal_true, np.full(4000, mu), alpha
        )
        covs.append(empirical_coverage(lo, hi, test_true))
    assert np.mean(covs) >= 1 - alpha - 0.005


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_cqr_covers_with_miscalibrated_heteroscedastic_band(alpha):
    rng = np.random.default_rng(2)
    n_cal, n_test = 6000, 20000
    x_cal = rng.uniform(0, 1, n_cal)
    x_test = rng.uniform(0, 1, n_test)
    sig = lambda x: 0.5 + 2.0 * x
    cal_true = rng.normal(0.0, sig(x_cal))
    test_true = rng.normal(0.0, sig(x_test))
    band = lambda x: sig(x)
    lo, hi, _ = cqr(
        -band(x_cal), band(x_cal), cal_true, -band(x_test), band(x_test), alpha
    )
    cov = empirical_coverage(lo, hi, test_true)
    assert cov == pytest.approx(1 - alpha, abs=0.02)


def test_cqr_intervals_widen_where_noise_grows():
    rng = np.random.default_rng(3)
    x_cal = rng.uniform(0, 1, 6000)
    x_test = np.array([0.05, 0.95])
    sig = lambda x: 0.5 + 2.0 * x
    cal_true = rng.normal(0.0, sig(x_cal))
    lo, hi, _ = cqr(
        -sig(x_cal), sig(x_cal), cal_true, -sig(x_test), sig(x_test), 0.1
    )
    w = interval_width(lo, hi)
    assert w[1] > w[0]


def test_empirical_coverage_and_width_basics():
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([1.0, 1.0, 1.0])
    true = np.array([0.5, 2.0, -1.0])
    assert empirical_coverage(lo, hi, true) == pytest.approx(1 / 3)
    np.testing.assert_array_equal(interval_width(lo, hi), np.ones(3))


# --------------------------------------------------------------------------- #
# Weighted, localized, and conditional conformal extensions (Gauge 03 / 04)
# --------------------------------------------------------------------------- #
def test_weighted_quantile_and_split_reduce_to_unweighted():
    rng = np.random.default_rng(0)
    scores = rng.gamma(2.0, 1.0, size=200)
    for alpha in (0.05, 0.1, 0.2, 0.5):
        w = np.ones_like(scores)
        got = weighted_conformal_quantile(scores, w, alpha, self_weight=1.0)
        assert got == pytest.approx(conformal_quantile(scores, alpha))

    q_w = weighted_split_conformal(scores, np.ones(200), np.ones(10), 0.1)
    assert np.allclose(q_w, conformal_quantile(scores, 0.1))
    assert (
        weighted_conformal_quantile(
            np.arange(5.0), np.ones(5), 0.01, self_weight=100.0
        )
        == np.inf
    )


def test_weighted_conformal_restores_coverage_under_covariate_shift():
    rng = np.random.default_rng(0)
    alpha = 0.1
    n = 6000
    xc = rng.uniform(0, 1, n)
    sc = np.abs(rng.normal(0, 1 + 3 * xc))
    xt = rng.beta(5, 2, n)
    yt = rng.normal(0, 1 + 3 * xt)
    q = conformal_quantile(sc, alpha)
    cov_plain = float(np.mean(np.abs(yt) <= q))
    w_cal = beta_dist.pdf(xc, 5, 2)
    w_test = beta_dist.pdf(xt, 5, 2)
    qj = weighted_split_conformal(sc, w_cal, w_test, alpha)
    cov_w = float(np.mean(np.abs(yt) <= qj))
    assert cov_plain < 0.87
    assert abs(cov_w - (1 - alpha)) < 0.03


def test_localized_and_conditional_conformal_improve_conditional_coverage():
    rng = np.random.default_rng(2)
    n_cal, n_test = 4000, 8000
    x_cal = rng.uniform(0, 1, (n_cal, 1))
    x_test = rng.uniform(0, 1, (n_test, 1))
    sig = lambda x: 0.2 + 3.0 * x[:, 0]
    y_cal = rng.normal(0.0, sig(x_cal))
    y_test = rng.normal(0.0, sig(x_test))
    scores = np.abs(y_cal)

    lo_g, hi_g, _ = split_conformal(np.zeros(n_cal), y_cal, np.zeros(n_test), 0.1)
    t_l = localized_conformal(scores, x_cal, x_test, 0.1, bandwidth=0.2)
    t_c = np.clip(conditional_conformal(scores, x_cal, x_test, 0.1), 0.0, None)

    edges = np.quantile(x_test[:, 0], [1 / 3, 2 / 3])
    reg = np.digitize(x_test[:, 0], edges)

    def worst_gap(lo, hi):
        return max(
            abs(0.9 - empirical_coverage(lo[reg == r], hi[reg == r], y_test[reg == r]))
            for r in range(3)
        )

    assert worst_gap(-t_l, t_l) < worst_gap(lo_g, hi_g)
    assert worst_gap(-t_l, t_l) < 0.05
    assert worst_gap(-t_c, t_c) < 0.05


# --------------------------------------------------------------------------- #
# ConformalIVIM high-level public API tests
# --------------------------------------------------------------------------- #
def test_conformal_ivim_split_with_any_sklearn_regressor():
    cohort = generate_cohort(400, 600, 1000, snr_grid=(20, 50), seed=21)
    rf = RandomForestRegressor(n_estimators=40, max_depth=8, random_state=0)
    wrapper = ConformalIVIM(estimator=rf, alpha=0.10, method="split")
    wrapper.fit_calibrate(
        cohort.signals["train"],
        cohort.params["train"],
        cohort.signals["cal"],
        cohort.params["cal"],
    )
    lo, hi = wrapper.predict_interval(cohort.signals["test"])
    assert lo.shape == (1000, 3) and hi.shape == (1000, 3)
    for j in range(3):
        cov = empirical_coverage(lo[:, j], hi[:, j], cohort.params["test"][:, j])
        assert cov == pytest.approx(0.90, abs=0.04)


def test_conformal_ivim_cqr_default_and_dict_output():
    cohort = generate_cohort(500, 600, 1000, snr_grid=(15, 40, 80), seed=33)
    cqr_model = ConformalIVIM(alpha=0.10, method="cqr", random_state=0)
    cqr_model.fit_calibrate(
        cohort.signals["train"],
        cohort.params["train"],
        cohort.signals["cal"],
        cohort.params["cal"],
    )
    out = cqr_model.predict_interval(cohort.signals["test"], return_dict=True)
    assert {"lower", "upper", "width", "D", "Dstar", "f"} <= set(out.keys())
    summary = cqr_model.evaluate(
        cohort.signals["test"], cohort.params["test"], snr=cohort.snr["test"]
    )
    for name in ("D", "Dstar", "f"):
        assert summary["parameters"][name]["abs_coverage_gap"] < 0.045
