"""Tests for the label-free ``DeploymentValidityMonitor`` (Family-1 Mahalanobis /
Hotelling T^2 and Family-2 NLLS residual-conformal two-sample shift tests).
"""
import numpy as np
import pytest

from conformal_ivim import DeploymentValidityMonitor, simulate_ivim_signal
from conformal_ivim.forward import DEFAULT_B_VALUES, generate_cohort
from conformal_ivim.monitor import estimate_importance_weights


def test_monitor_silent_on_indistribution_fires_on_shift():
    rng = np.random.default_rng(0)
    cal_feat = rng.normal(0, 1, (1500, 6))
    cal_resid = rng.exponential(1.0, 1500)
    mon = DeploymentValidityMonitor(seed=0).fit(cal_feat, cal_resid)

    id_feat = rng.normal(0, 1, (1500, 6))
    id_resid = rng.exponential(1.0, 1500)
    r_id = mon.evaluate(id_feat, id_resid)
    assert not r_id["fires"]
    assert r_id["auc"] < 0.6

    sh_feat = rng.normal(1.5, 1.4, (1500, 6))
    sh_resid = rng.exponential(2.5, 1500)
    r_sh = mon.evaluate(sh_feat, sh_resid)
    assert r_sh["fires"]
    assert r_sh["auc"] > 0.8


def test_monitor_batch_mechanics_and_power_curve():
    rng = np.random.default_rng(42)
    cal_feat = rng.normal(0.0, 1.0, (2000, 9))
    cal_feat[:, -1] = np.log(0.2)
    cal_resid = rng.normal(1.0, 0.2, 2000)
    mon = DeploymentValidityMonitor(fpr=0.05, seed=42).fit(cal_feat, cal_resid)

    id_feat = rng.normal(0.0, 1.0, (2000, 9))
    id_feat[:, -1] = np.log(0.2)
    id_resid = rng.normal(1.0, 0.2, 2000)
    r_id = mon.evaluate(id_feat, id_resid)
    assert r_id["maha"]["hotelling_p"] > 0.05
    p_id = mon.power_by_sample_size(
        id_feat, id_resid, sample_sizes=(100, 500, 2000), is_control=True
    )
    for n in (100, 500, 2000):
        assert p_id[n]["power_fam1"] == pytest.approx(0.05, abs=0.01)
        assert p_id[n]["power_fam2"] == pytest.approx(0.05, abs=0.01)

    sh_feat = rng.normal(0.0, 1.0, (2000, 9))
    sh_feat[:, 0] += 0.35
    sh_feat[:, -1] = np.log(0.2)
    sh_resid = rng.normal(1.03, 0.2, 2000)
    r_sh = mon.evaluate(sh_feat, sh_resid)
    assert r_sh["maha"]["auc"] < 0.60
    assert r_sh["maha"]["hotelling_t2"] > r_sh["maha"]["hotelling_thr"]
    assert r_sh["resid"]["std_z_score"] > 2.0
    p_sh = mon.power_by_sample_size(
        sh_feat, sh_resid, sample_sizes=(100, 500, 2000), is_control=False
    )
    assert p_sh[2000]["power_combined"] > p_sh[100]["power_combined"]


def test_importance_weights_separate_shifted_populations():
    rng = np.random.default_rng(0)
    cal = rng.normal(0, 1, (1200, 4))
    test = rng.normal(2.0, 1, (1200, 4))
    w_cal, w_test, _ = estimate_importance_weights(cal, test, seed=0)
    assert w_cal.shape == (1200,)
    assert w_test.shape == (1200,)
    assert np.median(w_test) > np.median(w_cal)


def test_monitor_direct_raw_dwi_signals_mode():
    cohort = generate_cohort(10, 120, 100, snr_grid=(25, 50), seed=77)
    mon = DeploymentValidityMonitor(fpr=0.05, seed=77).fit_signals(
        cohort.signals["cal"], b=DEFAULT_B_VALUES
    )
    rep_id = mon.evaluate_signals(cohort.signals["test"])
    assert not rep_id["fires"]

    rng = np.random.default_rng(88)
    shifted_signals = simulate_ivim_signal(
        DEFAULT_B_VALUES,
        D=rng.uniform(0.3e-3, 1.0e-3, 100),
        Dstar=rng.uniform(70e-3, 140e-3, 100),
        f=rng.uniform(0.02, 0.10, 100),
        snr=7.0,
        seed=88,
    )
    rep_shift = mon.evaluate_signals(shifted_signals)
    assert rep_shift["fires"]
    assert rep_shift["auc"] > 0.80
