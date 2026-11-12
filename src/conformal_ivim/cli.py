"""Command-line smoke-test and synthetic verification interface for ``conformal-ivim``."""
from __future__ import annotations

import argparse
import sys
import time
from typing import Sequence

import numpy as np

from conformal_ivim import __version__
from conformal_ivim.conformal import mondrian_cqr, mondrian_split, split_conformal
from conformal_ivim.estimators import IVIMQuantileRegressor, PARAM_NAMES
from conformal_ivim.forward import DEFAULT_B_VALUES, generate_cohort, simulate_ivim_signal
from conformal_ivim.metrics import empirical_coverage, interval_width
from conformal_ivim.monitor import DeploymentValidityMonitor, extract_observable_features


def run_demo(
    n_train: int = 350,
    n_cal: int = 200,
    n_test: int = 200,
    alpha: float = 0.10,
    seed: int = 42,
) -> str:
    """Run a fast synthetic verification of Split-Conformal, CQR, Mondrian, and Shift Monitor."""
    t0 = time.perf_counter()
    cohort = generate_cohort(
        n_train=n_train,
        n_cal=n_cal,
        n_test=n_test,
        snr_grid=(15, 30, 60),
        seed=seed,
    )

    X_tr, Y_tr = cohort.signals["train"], cohort.params["train"]
    X_cal, Y_cal = cohort.signals["cal"], cohort.params["cal"]
    X_te, Y_te = cohort.signals["test"], cohort.params["test"]
    snr_cal, snr_te = cohort.snr["cal"], cohort.snr["test"]

    # Extract NLLS point estimates + observable features once for calibration & test
    cal_nlls, cal_feat, cal_resid = extract_observable_features(X_cal, b=DEFAULT_B_VALUES)
    te_nlls, te_feat, te_resid = extract_observable_features(X_te, b=DEFAULT_B_VALUES)

    # Fit conditional quantile regressor once for CQR and Mondrian-CQR
    qr = IVIMQuantileRegressor(
        quantile_levels=(alpha / 2.0, 1.0 - alpha / 2.0),
        random_state=seed,
        max_iter=80,
    ).fit(X_tr, Y_tr)
    cal_qlo, cal_qhi = qr.predict_interval(X_cal, alpha=alpha)
    te_qlo, te_qhi = qr.predict_interval(X_te, alpha=alpha)

    lines = [
        f"conformal-ivim v{__version__} | Synthetic Verification (nominal 1-alpha = {1.0 - alpha:.2f})",
        "=" * 78,
        f"{'Method':<18} {'Param':<7} {'Coverage':>10} {'Median Width':>14} {'Status':>10}",
        "-" * 78,
    ]

    for j, pname in enumerate(PARAM_NAMES):
        # 1. Split-Conformal (NLLS base)
        lo_sp, hi_sp, _ = split_conformal(cal_nlls[:, j], Y_cal[:, j], te_nlls[:, j], alpha)
        # 2. CQR (GBR quantile base)
        scores_cqr = np.maximum(cal_qlo[:, j] - Y_cal[:, j], Y_cal[:, j] - cal_qhi[:, j])
        k = int(np.ceil((len(scores_cqr) + 1) * (1.0 - alpha)))
        q_cqr = float(np.sort(scores_cqr)[min(k - 1, len(scores_cqr) - 1)])
        lo_cqr, hi_cqr = te_qlo[:, j] - q_cqr, te_qhi[:, j] + q_cqr
        # 3. Mondrian Split-Conformal (SNR-stratified)
        lo_ms, hi_ms = mondrian_split(
            cal_nlls[:, j], Y_cal[:, j], snr_cal, te_nlls[:, j], snr_te, alpha
        )
        # 4. Mondrian CQR (SNR-stratified)
        lo_mc, hi_mc = mondrian_cqr(
            cal_qlo[:, j], cal_qhi[:, j], Y_cal[:, j], snr_cal,
            te_qlo[:, j], te_qhi[:, j], snr_te, alpha,
        )

        for label, lo, hi in (
            ("Split-Conformal", lo_sp, hi_sp),
            ("CQR", lo_cqr, hi_cqr),
            ("Mondrian-Split", lo_ms, hi_ms),
            ("Mondrian-CQR", lo_mc, hi_mc),
        ):
            cov = empirical_coverage(lo, hi, Y_te[:, j])
            med_w = float(np.median(interval_width(lo, hi)))
            ok = "PASS" if cov >= (1.0 - alpha - 0.08) else "WARN"
            lines.append(f"{label:<18} {pname:<7} {cov:>10.3f} {med_w:>14.6f} {ok:>10}")
        lines.append("-" * 78)

    # Two-sample label-free exchangeability monitor
    mon = DeploymentValidityMonitor(fpr=0.05, n_null=250, seed=seed).fit(cal_feat, cal_resid)
    rep_id = mon.evaluate(te_feat, te_resid)

    rng = np.random.default_rng(seed + 99)
    X_shift = simulate_ivim_signal(
        DEFAULT_B_VALUES,
        D=rng.uniform(0.3e-3, 1.0e-3, 120),
        Dstar=rng.uniform(70.0e-3, 140.0e-3, 120),
        f=rng.uniform(0.02, 0.10, 120),
        snr=7.5,
        seed=seed + 99,
    )
    _, sh_feat, sh_resid = extract_observable_features(X_shift, b=DEFAULT_B_VALUES)
    rep_sh = mon.evaluate(sh_feat, sh_resid)

    lines.extend(
        [
            "Label-Free Two-Sample Shift Monitor (FPR alpha_mon = 0.05):",
            f"  In-Distribution Test Batch : fires={str(rep_id['fires']):<5} "
            f"| AUC={rep_id['auc']:.3f} | Hotelling p={rep_id['maha']['hotelling_p']:.3f}",
            f"  Severe SNR/Domain Shift    : fires={str(rep_sh['fires']):<5} "
            f"| AUC={rep_sh['auc']:.3f} | Hotelling p={rep_sh['maha']['hotelling_p']:.3e}",
            "=" * 78,
            f"Completed in {time.perf_counter() - t0:.2f}s.",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conformal-ivim",
        description=(
            "Distribution-free conformal prediction intervals and label-free "
            "deployment monitoring for IVIM diffusion MRI."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    sub = parser.add_subparsers(dest="command")
    demo_p = sub.add_parser(
        "demo",
        help="Run fast (<5s) synthetic Split-Conformal, CQR, Mondrian & Shift-Monitor verification.",
    )
    demo_p.add_argument("--alpha", type=float, default=0.10, help="Miscoverage level (default: 0.10).")
    demo_p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    demo_p.add_argument("--n-train", type=int, default=350, help="Synthetic training size.")
    demo_p.add_argument("--n-cal", type=int, default=200, help="Synthetic calibration size.")
    demo_p.add_argument("--n-test", type=int, default=200, help="Synthetic test size.")
    demo_p.add_argument(
        "--json",
        action="store_true",
        help="Emit standardized AveryCorp AuditReport JSON diagnostic.",
    )

    audit_p = sub.add_parser(
        "audit",
        help="Run Stage 2 Suite I conformal calibration, cohort shift & voxel OOD diagnostic audit.",
    )
    audit_p.add_argument("--seed", type=int, default=42, help="Deterministic PCG64 seed (default: 42).")
    audit_p.add_argument("--alpha", type=float, default=0.10, help="Miscoverage level (default: 0.10).")
    audit_p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable AuditReport JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from conformal_ivim.audit import run_audit

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo":
        if getattr(args, "json", False):
            report = run_audit(
                seed=args.seed,
                n_train=args.n_train,
                n_cal=args.n_cal,
                n_test=args.n_test,
                alpha=args.alpha,
            )
            print(report.to_json())
            return 0
        print(
            run_demo(
                n_train=args.n_train,
                n_cal=args.n_cal,
                n_test=args.n_test,
                alpha=args.alpha,
                seed=args.seed,
            )
        )
        return 0
    if args.command == "audit":
        report = run_audit(seed=args.seed, alpha=args.alpha)
        if getattr(args, "json", False):
            print(report.to_json())
        else:
            print(report.summary_line())
            for k, v in report.metrics.items():
                print(f"  {k}: {v}")
        return 0
    parser.print_help(sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
