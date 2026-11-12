"""Tests for standardized AveryCorp AuditReport contract in conformalIVIM."""
from __future__ import annotations

import json

from conformal_ivim import AuditReport, VoxelOODDetector, run_audit
from conformal_ivim.cli import main
from conformal_ivim.forward import DEFAULT_B_VALUES, generate_cohort, simulate_ivim_signal

REQUIRED_AUDIT_KEYS = {
    "package",
    "version",
    "suite",
    "stage_role",
    "status",
    "verdict",
    "metrics",
    "thresholds",
    "seed",
}


def test_audit_report_contract_and_determinism() -> None:
    rep1 = run_audit(seed=42)
    rep2 = run_audit(seed=42)
    assert isinstance(rep1, AuditReport)
    assert rep1 == rep2
    assert rep1.package == "conformal-ivim"
    assert rep1.suite == "Suite I: Clinical Quantitative-MRI Trust Spine"
    assert rep1.status in ("PASS", "WARN", "HALT")
    assert rep1.status == "PASS"
    assert "EXCHANGEABLE_AND_CALIBRATED" in rep1.verdict

    parsed = json.loads(rep1.to_json())
    assert set(parsed.keys()) == REQUIRED_AUDIT_KEYS
    assert parsed["seed"] == 42
    assert 0.80 <= parsed["metrics"]["marginal_coverage_mondrian_cqr"] <= 1.0
    assert parsed["metrics"]["hotelling_t2_p_value"] >= 0.05
    assert parsed["package"] in rep1.summary_line()


def test_voxel_ood_detector_flags_corrupted_voxels() -> None:
    import numpy as np

    cohort = generate_cohort(n_train=40, n_cal=150, n_test=120, snr_grid=(30,), seed=7)
    det = VoxelOODDetector(alpha=0.05).fit(cohort.signals["cal"], b=DEFAULT_B_VALUES)
    id_res = det.predict_ood(cohort.signals["test"])
    assert id_res["flagged_fraction"] <= 0.15

    ood_sig = simulate_ivim_signal(
        DEFAULT_B_VALUES,
        D=np.full(60, 2.5e-3),
        Dstar=np.full(60, 140.0e-3),
        f=np.full(60, 0.45),
        snr=5.0,
        seed=99,
    )
    ood_res = det.predict_ood(ood_sig)
    assert ood_res["flagged_fraction"] > 0.50


def test_cli_audit_and_demo_json(capsys) -> None:
    rc_audit = main(["audit", "--seed", "42", "--json"])
    assert rc_audit == 0
    out_audit = json.loads(capsys.readouterr().out)
    assert set(out_audit.keys()) == REQUIRED_AUDIT_KEYS
    assert out_audit["package"] == "conformal-ivim"

    rc_demo = main(["demo", "--seed", "42", "--n-train", "180", "--n-cal", "120", "--n-test", "120", "--json"])
    assert rc_demo == 0
    out_demo = json.loads(capsys.readouterr().out)
    assert set(out_demo.keys()) == REQUIRED_AUDIT_KEYS
