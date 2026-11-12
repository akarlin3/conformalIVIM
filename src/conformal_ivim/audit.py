"""Standardized AveryCorp Suite I (Stage 2) conformal calibration & shift/OOD audit."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from conformal_ivim.conformal import aci_conformal, cqr, mondrian_cqr, split_conformal
from conformal_ivim.estimators import IVIMQuantileRegressor
from conformal_ivim.forward import DEFAULT_B_VALUES, generate_cohort
from conformal_ivim.metrics import coverage_by_stratum, empirical_coverage, regime_terciles
from conformal_ivim.monitor import (
    DeploymentValidityMonitor,
    VoxelOODDetector,
    extract_observable_features,
)

SUITE_I_NAME: Literal["Suite I: Clinical Quantitative-MRI Trust Spine"] = (
    "Suite I: Clinical Quantitative-MRI Trust Spine"
)
SUITE_II_NAME: Literal["Suite II: Scientific Identifiability & Null-Audit Suite"] = (
    "Suite II: Scientific Identifiability & Null-Audit Suite"
)
SuiteLiteral = Literal[
    "Suite I: Clinical Quantitative-MRI Trust Spine",
    "Suite II: Scientific Identifiability & Null-Audit Suite",
]
StatusLiteral = Literal["PASS", "WARN", "HALT"]


def _to_builtin(val: Any) -> float | int | str | bool:
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return float(val)
    return str(val)


@dataclass(frozen=True)
class AuditReport:
    """Standardized AveryCorp scientific identifiability and diagnostic audit report."""

    package: str
    version: str
    suite: SuiteLiteral
    stage_role: str
    status: StatusLiteral
    verdict: str
    metrics: dict[str, float | int | str | bool]
    thresholds: dict[str, float | int | str]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize the audit report to a JSON-compatible Python dictionary."""
        return {
            "package": str(self.package),
            "version": str(self.version),
            "suite": str(self.suite),
            "stage_role": str(self.stage_role),
            "status": str(self.status),
            "verdict": str(self.verdict),
            "metrics": {str(k): _to_builtin(v) for k, v in self.metrics.items()},
            "thresholds": {str(k): _to_builtin(v) for k, v in self.thresholds.items()},
            "seed": int(self.seed),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the audit report to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary_line(self) -> str:
        """Return a concise one-line human-readable status summary."""
        return (
            f"[{self.status}] {self.package} v{self.version} "
            f"({self.suite} | {self.stage_role}) -> {self.verdict}"
        )


def run_audit(
    seed: int = 42,
    *,
    n_train: int = 350,
    n_cal: int = 200,
    n_test: int = 200,
    alpha: float = 0.10,
    ood_alpha: float = 0.05,
    min_marginal_coverage: float = 0.84,
    min_high_dstar_coverage: float = 0.78,
    min_hotelling_p: float = 0.05,
    max_ood_flagged_fraction: float = 0.12,
) -> AuditReport:
    """Run the Stage 2 Suite I conformal calibration, cohort shift, and voxel OOD audit."""
    from conformal_ivim import __version__

    # Seeded PCG64 cohort (numpy.random.default_rng uses PCG64 bit generator)
    _ = np.random.Generator(np.random.PCG64(int(seed)))
    cohort = generate_cohort(
        n_train=n_train,
        n_cal=n_cal,
        n_test=n_test,
        snr_grid=(15, 30, 60),
        seed=int(seed),
    )
    X_tr, Y_tr = cohort.signals["train"], cohort.params["train"]
    X_cal, Y_cal = cohort.signals["cal"], cohort.params["cal"]
    X_te, Y_te = cohort.signals["test"], cohort.params["test"]
    snr_cal, snr_te = cohort.snr["cal"], cohort.snr["test"]

    cal_nlls, cal_feat, cal_resid = extract_observable_features(X_cal, b=DEFAULT_B_VALUES)
    te_nlls, te_feat, te_resid = extract_observable_features(X_te, b=DEFAULT_B_VALUES)

    qr = IVIMQuantileRegressor(
        quantile_levels=(alpha / 2.0, 1.0 - alpha / 2.0),
        random_state=int(seed),
        max_iter=70,
    ).fit(X_tr, Y_tr)
    cal_qlo, cal_qhi = qr.predict_interval(X_cal, alpha=alpha)
    te_qlo, te_qhi = qr.predict_interval(X_te, alpha=alpha)

    # Evaluate D* (index 1) across Split, CQR, Mondrian-CQR, and online ACI
    j_ds = 1
    lo_sp, hi_sp, _ = split_conformal(cal_nlls[:, j_ds], Y_cal[:, j_ds], te_nlls[:, j_ds], alpha)
    lo_cqr, hi_cqr, _ = cqr(
        cal_qlo[:, j_ds], cal_qhi[:, j_ds], Y_cal[:, j_ds],
        te_qlo[:, j_ds], te_qhi[:, j_ds], alpha,
    )
    lo_mc, hi_mc = mondrian_cqr(
        cal_qlo[:, j_ds], cal_qhi[:, j_ds], Y_cal[:, j_ds], snr_cal,
        te_qlo[:, j_ds], te_qhi[:, j_ds], snr_te, alpha,
    )
    lo_aci, hi_aci, _ = aci_conformal(
        cal_nlls[:, j_ds], Y_cal[:, j_ds], te_nlls[:, j_ds], Y_te[:, j_ds], alpha=alpha
    )

    cov_split = float(empirical_coverage(lo_sp, hi_sp, Y_te[:, j_ds]))
    cov_cqr = float(empirical_coverage(lo_cqr, hi_cqr, Y_te[:, j_ds]))
    cov_mondrian_cqr = float(empirical_coverage(lo_mc, hi_mc, Y_te[:, j_ds]))
    cov_aci = float(empirical_coverage(lo_aci, hi_aci, Y_te[:, j_ds]))

    # High-D* tercile conditional coverage (tercile index 2)
    ds_terciles, _ = regime_terciles(Y_te[:, j_ds])
    strata_cov = coverage_by_stratum(lo_mc, hi_mc, Y_te[:, j_ds], ds_terciles)
    high_dstar_cov = float(strata_cov.get(2, {}).get("coverage", cov_mondrian_cqr))

    # Cohort shift monitor (Hotelling T^2 + residual conformal)
    mon = DeploymentValidityMonitor(fpr=0.05, n_null=200, seed=int(seed)).fit(cal_feat, cal_resid)
    rep_id = mon.evaluate(te_feat, te_resid)
    hotelling_p = float(rep_id["maha"]["hotelling_p"])
    monitor_fires = bool(rep_id["fires"])

    # Per-voxel conformal OOD detector
    ood_det = VoxelOODDetector(alpha=ood_alpha).fit(cal_feat, cal_resid)
    ood_rep = ood_det.predict_ood(te_feat, te_resid)
    flagged_fraction = float(ood_rep["flagged_fraction"])

    calibrated_ok = (
        cov_mondrian_cqr >= min_marginal_coverage
        and high_dstar_cov >= min_high_dstar_coverage
    )
    shift_ok = (not monitor_fires) and (hotelling_p >= min_hotelling_p) and (
        flagged_fraction <= max_ood_flagged_fraction
    )

    if calibrated_ok and shift_ok:
        status: StatusLiteral = "PASS"
        verdict = (
            f"EXCHANGEABLE_AND_CALIBRATED: Mondrian-CQR coverage = {cov_mondrian_cqr:.3f} "
            f"(high-D* tercile = {high_dstar_cov:.3f}), Hotelling p = {hotelling_p:.3f}, "
            f"OOD flagged = {flagged_fraction:.3f}."
        )
    elif calibrated_ok:
        status = "WARN"
        verdict = (
            f"SHIFT_OR_OOD_DETECTED: Cohort shift or elevated voxel OOD fraction "
            f"(Hotelling p = {hotelling_p:.3f}, flagged = {flagged_fraction:.3f}) despite "
            f"marginal coverage = {cov_mondrian_cqr:.3f}."
        )
    else:
        status = "HALT"
        verdict = (
            f"SHIFT_OR_OOD_DETECTED: Conformal coverage below target "
            f"(marginal = {cov_mondrian_cqr:.3f}, high-D* = {high_dstar_cov:.3f})."
        )

    return AuditReport(
        package="conformal-ivim",
        version=__version__,
        suite=SUITE_I_NAME,
        stage_role="Stage 2: Split/CQR/Mondrian/ACI Conformal Calibration, Cohort Shift & Voxel OOD Gate",
        status=status,
        verdict=verdict,
        metrics={
            "marginal_coverage_split": round(cov_split, 6),
            "marginal_coverage_cqr": round(cov_cqr, 6),
            "marginal_coverage_mondrian_cqr": round(cov_mondrian_cqr, 6),
            "marginal_coverage_aci": round(cov_aci, 6),
            "high_dstar_tercile_coverage": round(high_dstar_cov, 6),
            "hotelling_t2_p_value": round(hotelling_p, 6),
            "cohort_shift_fires": monitor_fires,
            "voxel_ood_flagged_fraction": round(flagged_fraction, 6),
        },
        thresholds={
            "nominal_coverage": round(1.0 - float(alpha), 4),
            "min_marginal_coverage": float(min_marginal_coverage),
            "min_high_dstar_coverage": float(min_high_dstar_coverage),
            "min_hotelling_p": float(min_hotelling_p),
            "max_ood_flagged_fraction": float(max_ood_flagged_fraction),
        },
        seed=int(seed),
    )
