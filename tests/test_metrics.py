"""Tests for interval scoring, sharpness summaries, and marginal/conditional
coverage diagnostics in ``conformal_ivim.metrics``.
"""
import numpy as np
import pytest

from conformal_ivim.metrics import (
    conditional_coverage_grid,
    coverage_by_stratum,
    directional_miss_rates,
    evaluate_intervals,
    interval_score,
    regime_terciles,
    sharpness_summary,
)


def test_interval_score_inside_is_just_width():
    assert interval_score(0.0, 2.0, 1.0, alpha=0.1) == pytest.approx(2.0)


def test_interval_score_penalizes_miss_above():
    assert interval_score(0.0, 2.0, 3.0, alpha=0.1) == pytest.approx(2.0 + 20 * 1.0)


def test_interval_score_penalizes_miss_below_symmetrically():
    assert interval_score(0.0, 2.0, -1.0, alpha=0.1) == pytest.approx(2.0 + 20 * 1.0)


def test_interval_score_vectorized():
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([2.0, 2.0, 2.0])
    y = np.array([1.0, 3.0, -1.0])
    got = interval_score(lo, hi, y, alpha=0.1)
    np.testing.assert_allclose(got, [2.0, 22.0, 22.0])


def test_interval_score_smaller_alpha_penalizes_misses_more():
    s_small = interval_score(0.0, 1.0, 2.0, alpha=0.05)
    s_big = interval_score(0.0, 1.0, 2.0, alpha=0.20)
    assert s_small > s_big


def test_sharpness_directional_and_conditional_diagnostics():
    lo = np.array([0.0, 0.0, 0.0, 0.0])
    hi = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([0.5, 2.5, 1.5, -0.5])

    sharp = sharpness_summary(lo, hi)
    assert sharp["median_width"] == pytest.approx(2.5)
    assert sharp["mean_width"] == pytest.approx(2.5)

    miss = directional_miss_rates(lo, hi, y)
    assert miss["coverage"] == pytest.approx(0.5)
    assert miss["lower_miss"] == pytest.approx(0.25)
    assert miss["upper_miss"] == pytest.approx(0.25)
    assert miss["upper_share_of_misses"] == pytest.approx(0.5)

    strata = np.array([10, 10, 50, 50])
    by_s = coverage_by_stratum(lo, hi, y, strata)
    assert by_s[10]["count"] == 2 and by_s[50]["count"] == 2

    dstar = np.linspace(10e-3, 100e-3, 60)
    reg, edges = regime_terciles(dstar, n=3)
    assert len(edges) == 2
    snr = np.tile([10, 30, 50], 20)
    grid = conditional_coverage_grid(
        dstar - 5e-3, dstar + 5e-3, dstar, reg, snr, snr_levels=[10, 30, 50]
    )
    assert grid["hi_marg"] == pytest.approx(1.0)
    assert grid["grid"].shape == (3, 3)

    eval_res = evaluate_intervals(
        np.column_stack([lo, lo, lo]),
        np.column_stack([hi, hi, hi]),
        np.column_stack([y, y, y]),
        alpha=0.10,
        snr=strata,
    )
    assert set(eval_res["parameters"].keys()) == {"D", "Dstar", "f"}
