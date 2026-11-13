"""Deterministic v1.3.0 tests for Lethe scan-rescan repeatability, Reverb counterexample,
AuditAttestation cryptographic provenance, and optional IVIMBench canonical DRO bridge.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from conformal_ivim import (
    AuditAttestation,
    LetheRepeatabilityReport,
    evaluate_echo_repeatability_scale,
    generate_cohort,
    resolve_clinical_b,
    run_audit,
    run_reverb_precision_coverage_counterexample,
    simulate_ivim_signal,
)
from conformal_ivim.cli import main


def test_v130_mit_license_isolation() -> None:
    pyproject_text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "1.3.0"' in pyproject_text
    assert "sextant-core" not in pyproject_text
    assert "sextant-ivim" not in pyproject_text
    assert "sextant_core" not in pyproject_text


def test_audit_attestation_cryptographic_provenance() -> None:
    rep1 = run_audit(seed=42)
    rep2 = run_audit(seed=42)
    rep_diff = run_audit(seed=43)

    assert isinstance(rep1.attestation, AuditAttestation)
    assert rep1.attestation == rep2.attestation
    assert len(rep1.attestation.input_sha256) == 64
    assert len(rep1.attestation.report_sha256) == 64
    assert rep1.attestation.input_sha256 != rep_diff.attestation.input_sha256
    assert rep1.attestation.report_sha256 != rep_diff.attestation.report_sha256
    assert set(rep1.attestation.platform_fingerprint.keys()) == {"python", "numpy", "scipy"}


@pytest.mark.parametrize("schedule_name", ["osipi", "tcga_lihc", "acrin_6698"])
@pytest.mark.parametrize(
    "family",
    ["biexp", "dispersion_gamma", "dispersion_lognormal", "stretched", "triexp"],
)
def test_canonical_dro_bridge_and_standalone_fallback(
    monkeypatch: pytest.MonkeyPatch,
    schedule_name: str,
    family: str,
) -> None:
    d_arr = np.array([0.9e-3, 1.4e-3, 1.8e-3])
    ds_arr = np.array([18.0e-3, 35.0e-3, 60.0e-3])
    f_arr = np.array([0.10, 0.22, 0.32])

    b_canon = resolve_clinical_b(schedule_name, use_canonical_dro=True)
    sig_canon = simulate_ivim_signal(
        schedule_name,
        D=d_arr,
        Dstar=ds_arr,
        f=f_arr,
        snr=None,
        use_canonical_dro=True,
        family=family,
    )
    assert sig_canon.shape == (3, b_canon.size)
    assert np.all(np.isfinite(sig_canon))

    # Force standalone fallback as if ivimbench were uninstalled
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    b_fb = resolve_clinical_b(schedule_name, use_canonical_dro=True)
    sig_fb = simulate_ivim_signal(
        schedule_name,
        D=d_arr,
        Dstar=ds_arr,
        f=f_arr,
        snr=None,
        use_canonical_dro=True,
        family=family,
    )
    assert np.allclose(b_canon, b_fb)
    assert np.allclose(sig_canon, sig_fb, rtol=1e-5, atol=1e-6)


def test_generate_cohort_canonical_dro_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    c1 = generate_cohort(
        n_train=24,
        n_cal=16,
        n_test=16,
        b="osipi",
        seed=42,
        use_canonical_dro=True,
        family="dispersion_gamma",
        organ="liver",
    )
    assert c1.signals["train"].shape == (24, 12)
    assert c1.params["cal"].shape == (16, 3)

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    c2 = generate_cohort(
        n_train=24,
        n_cal=16,
        n_test=16,
        b="osipi",
        seed=42,
        use_canonical_dro=True,
        family="dispersion_gamma",
        organ="liver",
    )
    assert c2.signals["train"].shape == (24, 12)
    assert c2.params["test"].shape == (16, 3)


def test_lethe_echo_repeatability_scale_and_bca_ci() -> None:
    rng = np.random.Generator(np.random.PCG64(42))
    n = 320
    theta_true = rng.uniform(0.7e-3, 2.0e-3, size=(n, 3))
    local_scale = np.linspace(0.5, 2.2, n)[:, None] * np.array([[8e-5, 4e-3, 0.025]])
    scan1 = theta_true + rng.normal(0.0, local_scale)
    scan2 = theta_true + rng.normal(0.0, local_scale)
    lo = scan1 - 1.645 * local_scale
    hi = scan1 + 1.645 * local_scale

    rep1 = evaluate_echo_repeatability_scale((lo, hi), scan1, scan2, seed=42, n_boot=250)
    rep2 = evaluate_echo_repeatability_scale({"lower": lo, "upper": hi}, scan1, scan2, seed=42, n_boot=250)

    assert isinstance(rep1, LetheRepeatabilityReport)
    assert rep1.to_dict() == rep2.to_dict()
    assert 0.65 <= rep1.repeatability_coverage <= 0.85
    ci_lo, ci_hi = rep1.repeatability_coverage_bca_ci
    assert ci_lo <= rep1.repeatability_coverage <= ci_hi
    assert rep1.sigma_repeat_over_width > 0.0
    assert rep1.spearman_r > 0.25
    assert set(rep1.per_parameter.keys()) == {"D", "Dstar", "f"}


def test_reverb_precision_coverage_counterexample_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    res = run_reverb_precision_coverage_counterexample(n_voxels=1000, seed=42)
    assert res["counterexample_verified"] is True
    assert res["empirical_truth_coverage"] >= 0.86
    assert res["repeatability_coverage_deficit"] >= 0.08
    assert res["empirical_scan_rescan_coverage"] < res["empirical_truth_coverage"]

    rc = main(["repeatability-check", "--n-voxels", "600", "--seed", "42", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["counterexample_verified"] is True
    assert "lethe_report" in payload
