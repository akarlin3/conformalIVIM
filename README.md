# `conformalIVIM` (`conformal-ivim`)

[![CI](https://github.com/akarlin3/conformalIVIM/actions/workflows/ci.yml/badge.svg)](https://github.com/akarlin3/conformalIVIM/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20694662.svg)](https://doi.org/10.5281/zenodo.20694662)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Typed: PEP 561](https://img.shields.io/badge/typing-PEP%20561-informational.svg)](src/conformal_ivim/py.typed)

**Distribution-free conformal prediction intervals and label-free deployment monitoring for IVIM diffusion MRI.**

`conformal-ivim` provides finite-sample, distribution-free per-voxel uncertainty intervals for intravoxel incoherent motion (IVIM) diffusion-weighted MRI parameters $(D, D^{*}, f)$ alongside a label-free two-sample deployment validity monitor and per-voxel conformal OOD gate that flag observable exchangeability breaks on unlabeled clinical or phantom test batches.

---

## AveryCorp Scientific Suites Architecture

`conformalIVIM` is **Stage 2** of the **AveryCorp Open Scientific Software Ecosystem**, sharing a standardized `AuditReport` diagnostic schema (`conformal-ivim audit --json` / `run_audit()`) across two non-overlapping suites:

- **Suite I: Clinical Quantitative-MRI Trust Spine (`Diagnose -> Calibrate -> Benchmark`)**
  - [`sextantIVIM`](https://github.com/akarlin3/sextantIVIM) (Stage 1): Forward NLLS fitting, Fisher CRLB & assumption-free $D^*$ boundary-railing gate ->
  - **[`conformalIVIM`](https://github.com/akarlin3/conformalIVIM) (Stage 2 — *This Package*)**: Split/CQR/Mondrian/ACI finite-sample conformal calibration, cohort shift & voxel OOD gate ->
  - [`IVIMBench`](https://github.com/akarlin3/IVIMBench) (Stage 3): Canonical DRO phantoms, clean-room Gnomon/Caliper rulers & closed-loop adaptive RT benchmark.
- **Suite II: Scientific Identifiability & Null-Audit Suite**
  - [`weakSINDY`](https://github.com/akarlin3/weakSINDY) (Dynamical & Fractional PDEs) · [`chimeraAnneal`](https://github.com/akarlin3/chimeraAnneal) (Metastable Oscillator Survival) · [`proteusMine`](https://github.com/akarlin3/proteusMine) (Structural Bioinformatics TOST Auditing).

---

## Installation

```bash
pip install git+https://github.com/akarlin3/conformalIVIM.git
# Or via PyPI once registered:
pip install conformal-ivim
```

For an editable install from source with optional plotting and development extras, plus a `<5s` synthetic CLI verification table covering Split-Conformal, CQR, Mondrian, and two-sample shift monitoring:

```bash
git clone https://github.com/akarlin3/conformalIVIM.git
cd conformalIVIM
pip install -e ".[plotting,dev]"
conformal-ivim demo
```

## 3-Line Quickstart

```bash
pip install git+https://github.com/akarlin3/conformalIVIM.git
python -c "from conformal_ivim import ConformalIVIM, generate_cohort; c = generate_cohort(400, 400, 200); m = ConformalIVIM(method='cqr', alpha=0.10).fit_calibrate(c.signals['train'], c.params['train'], c.signals['cal'], c.params['cal']); lo, hi = m.predict_interval(c.signals['test']); print('90% interval shapes:', lo.shape, hi.shape)"
```

---

## Mathematical Formulation

### 1. Bi-Exponential and 2D Multi-Echo IVIM Forward Models

In 1D multi-$b$ diffusion MRI, the normalized signal decay across sensitizing $b$-values ($b \ge 0\text{ s/mm}^2$) separates slow parenchymal tissue diffusion ($D$) from fast microvascular pseudo-diffusion ($D^{*} \gg D$):

$$
\frac{S(b)}{S_0} = f \exp(-b D^{*}) + (1 - f) \exp(-b D)
$$

For 2D diffusion-relaxometry acquisitions $S(b, \mathrm{TE})$ with compartment-specific transverse relaxation constants $(T_{2,\text{blood}}, T_{2,\text{tissue}})$:

$$
S(b, \mathrm{TE}) = S_0 \left[ f \exp\!\left(-\frac{\mathrm{TE}}{T_{2,\text{blood}}}\right) \exp(-b D^{*}) + (1 - f) \exp\!\left(-\frac{\mathrm{TE}}{T_{2,\text{tissue}}}\right) \exp(-b D) \right]
$$

Magnitude MRI measurements follow a Rician noise law $y(b) = \sqrt{(S(b) + \varepsilon_{\text{re}})^2 + \varepsilon_{\text{im}}^2}$ with $\varepsilon_{\text{re}}, \varepsilon_{\text{im}} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$ and $\sigma = S_0 / \mathrm{SNR}$. Local unbiased parameter identifiability is quantified via the analytical Jacobian $J(\theta)$ and Fisher information matrix $\mathcal{I}(\theta) = \sigma^{-2} J(\theta)^\top J(\theta)$, yielding per-parameter Cramér–Rao lower bounds $\mathrm{CRLB}(\theta_k) = \sqrt{[\mathcal{I}(\theta)^{-1}]_{kk}}$ (`compute_crlb`).

### 2. Split-Conformal Prediction Intervals

Given any point estimator $\hat{\theta}(X)$ (such as segmented-initialized bounded NLLS or a neural/ensemble regressor) and a held-out calibration split $\{(X_i, \theta_i)\}_{i=1}^{n_{\text{cal}}}$, symmetric split-conformal prediction computes per-parameter absolute residual nonconformity scores:

$$
s_{i,k} = \left| \theta_{i,k} - \hat{\theta}_k(X_i) \right|, \qquad k \in \{D, D^{*}, f\}
$$

Let $\hat{q}_{1-\alpha, k}$ denote the $m$-th smallest calibration score where $m = \lceil (n_{\text{cal}} + 1)(1 - \alpha) \rceil$. The calibrated test prediction interval is:

$$
C_{\text{split}, k}(X_{\text{test}}) = \left[ \hat{\theta}_k(X_{\text{test}}) - \hat{q}_{1-\alpha, k}, \; \hat{\theta}_k(X_{\text{test}}) + \hat{q}_{1-\alpha, k} \right]
$$

### 3. Conformalized Quantile Regression (CQR)

Because IVIM parameter uncertainty varies strongly with voxelwise SNR and signal curvature, a global split-conformal radius under-covers noisy voxels and over-covers high-SNR voxels. **Conformalized Quantile Regression** (Romano, Patterson & Candès, 2019) fits conditional lower and upper quantile regressors $\hat{q}_{\alpha/2, k}(X)$ and $\hat{q}_{1-\alpha/2, k}(X)$ on the training split and evaluates signed conformity scores on the calibration split:

$$
E_{i,k} = \max\!\left\{ \hat{q}_{\alpha/2, k}(X_i) - \theta_{i,k}, \; \theta_{i,k} - \hat{q}_{1-\alpha/2, k}(X_i) \right\}
$$

Taking the finite-sample conformal quantile $\hat{Q}_{1-\alpha, k}$ at rank $\lceil (n_{\text{cal}} + 1)(1 - \alpha) \rceil$ produces input-adaptive intervals:

$$
C_{\text{CQR}, k}(X_{\text{test}}) = \left[ \hat{q}_{\alpha/2, k}(X_{\text{test}}) - \hat{Q}_{1-\alpha, k}, \; \hat{q}_{1-\alpha/2, k}(X_{\text{test}}) + \hat{Q}_{1-\alpha, k} \right]
$$

Under exchangeability between calibration and test draws, **both** split-conformal and CQR satisfy the distribution-free finite-sample marginal guarantee for any base predictor:

$$
1 - \alpha \le \mathbb{P}\!\left( \theta_{\text{test}, k} \in C_k(X_{\text{test}}) \right) \le 1 - \alpha + \frac{1}{n_{\text{cal}} + 1}
$$

---

## Minimal Runnable Example on Synthetic Arrays

```python
import numpy as np
from conformal_ivim import (
    DEFAULT_B_VALUES,
    ConformalIVIM,
    DeploymentValidityMonitor,
    compute_crlb,
    empirical_coverage,
    simulate_ivim_signal,
)

rng = np.random.default_rng(42)
b = DEFAULT_B_VALUES  # 22 b-values from 0 to 800 s/mm^2

def sample_synthetic_batch(n: int, snr: float = 30.0):
    D = rng.uniform(0.5e-3, 3.0e-3, size=n)
    Dstar = rng.uniform(10.0e-3, 100.0e-3, size=n)
    f = rng.uniform(0.05, 0.40, size=n)
    Y = np.column_stack([D, Dstar, f])
    X = simulate_ivim_signal(b, D=D, Dstar=Dstar, f=f, snr=snr, noise="rician", rng=rng)
    return X, Y

X_train, Y_train = sample_synthetic_batch(800, snr=30.0)
X_cal, Y_cal = sample_synthetic_batch(600, snr=30.0)
X_test, Y_test = sample_synthetic_batch(500, snr=30.0)

# 1. Fit & calibrate Conformalized Quantile Regression (90% nominal coverage)
cqr_ivim = ConformalIVIM(method="cqr", alpha=0.10, random_state=0)
cqr_ivim.fit_calibrate(X_train, Y_train, X_cal, Y_cal)
lower, upper = cqr_ivim.predict_interval(X_test)

for idx, name in enumerate(("D", "Dstar", "f")):
    cov = empirical_coverage(lower[:, idx], upper[:, idx], Y_test[:, idx])
    med_w = float(np.median(upper[:, idx] - lower[:, idx]))
    print(f"{name:>5s} | realized coverage = {cov:.3f} (nominal 0.900) | median width = {med_w:.5f}")

# 2. Compute analytical Cramér-Rao lower bounds at a representative high-D* voxel
bounds = compute_crlb(b, D=1.2e-3, Dstar=85.0e-3, f=0.20, snr=30.0)
print("CRLB stds:", {k: round(v, 6) for k, v in bounds.items()})

# 3. Monitor unlabeled deployment batches for exchangeability breaks
monitor = DeploymentValidityMonitor(fpr=0.05, seed=0).fit_signals(X_cal, b=b)
X_shifted, _ = sample_synthetic_batch(300, snr=8.0)  # severe SNR domain shift

print("In-distribution alert:", monitor.evaluate_signals(X_test)["fires"])
print("SNR-shifted alert:    ", monitor.evaluate_signals(X_shifted)["fires"])
```

---

## Observable Deployment Monitoring Under Domain Shift

Conformal coverage guarantees rely on **exchangeability** between the labeled synthetic (or phantom) calibration cohort and the target deployment cohort. In clinical DWI, scanner changes, coil configurations, severe SNR degradation, partial-volume spill-in, or multi-compartment tissue physics can break exchangeability without any ground-truth labels available to measure miscoverage directly.

`DeploymentValidityMonitor` implements a **label-free, two-sample aggregate validity monitor** controlled at false-positive rate $\alpha_{\text{monitor}}$ (`fpr=0.05` by default) combining two complementary observable test families:

1. **Family 1 — Summary-Space Mahalanobis & Hotelling $T^2$ (`report["maha"]`)**:
   Extracts 9 observable per-voxel features from unlabeled decay curves—4 model-free shape descriptors (low-$b$ log-slope, high-$b$ log-slope, bi-exponential curvature, and $b \approx 30\text{ s/mm}^2$ early drop) plus plug-in NLLS estimates $(\hat{D}, \hat{D}^{*}, \hat{f}, \widehat{\mathrm{SNR}}, \log\hat{\sigma})$. Evaluates per-voxel Mahalanobis distances against the standardized calibration covariance alongside a multivariate two-sample **Hotelling $T^2$** batch mean test (`hotelling_t2`, `hotelling_p`) to capture subtle distributional shifts whose single-voxel ROC-AUC is near chance ($\sim 0.52$) yet accumulate high statistical power across a cohort or ROI batch.
2. **Family 2 — Signal-Space Residual Conformal (`report["resid"]`)**:
   Evaluates bi-exponential NLLS fit-residual norms $\|S - \hat{S}\|_2$ and noise-standardized residuals $\|S - \hat{S}\|_2 / \hat{\sigma}$ against calibration subsampling/bootstrap null thresholds (`std_z_score`, `std_p_value`) to detect forward-model misspecification (e.g., tri-exponential fast flow or non-Rician noise floors).

**Observable vs. Latent Regimes:** Whenever either family fires (`report["fires"] == True`), the calibration split is no longer exchangeable with the deployment batch; users should refrain from claiming nominal $(1 - \alpha)$ coverage or reweight calibration scores via `estimate_importance_weights` + `weighted_split_conformal` under pure covariate shift. Conversely, because the monitor inspects only observable signal statistics $P(X)$, it is intentionally blind ($\text{AUC} \approx 0.50$) to purely within-distribution latent subgroup conditioning (such as the irreducible 1D high-$D^{*}$ conditional coverage deficit when $P(X)$ is unchanged).

---

## Package Architecture

```text
conformalIVIM/
├── .github/workflows/        # Multi-Python CI (ci.yml) & PyPI Trusted Publishing (publish.yml)
├── pyproject.toml
├── README.md
├── LICENSE
├── CITATION.cff
├── src/
│   └── conformal_ivim/
│       ├── __init__.py       # Clean top-level API (ConformalIVIM, DeploymentValidityMonitor, ...)
│       ├── cli.py            # `conformal-ivim demo` fast synthetic verification CLI
│       ├── forward.py        # 1D bi-exp & 2D (b, TE) IVIM physics, Rician/nc-chi noise, CRLB
│       ├── estimators.py     # Segmented-init bounded NLLS, 2D VarPro NLLS, GBR quantile regressor
│       ├── conformal.py      # Split-conformal, CQR, Mondrian, weighted/localized/conditional CP
│       ├── monitor.py        # Family-1 Mahalanobis/Hotelling T^2 & Family-2 residual shift tests
│       ├── metrics.py        # Empirical coverage, sharpness, interval score, conditional grids
│       └── py.typed          # PEP 561 inline typing marker
└── tests/                    # Property, unit, and CLI smoke-test suite
```

---

## Citation

If you use `conformalIVIM` (`conformal-ivim`) in your research, please cite the companion *Magnetic Resonance in Medicine* manuscript and Zenodo software archive (see [`CITATION.cff`](CITATION.cff)):

- **Manuscript:** Karlin A. *Distribution-Free Conformal Uncertainty Calibration for IVIM Parameter Maps: Characterizing High-$D^{*}$ Conditional Coverage Limits in Single- and Multi-Echo Acquisitions.* Magnetic Resonance in Medicine (2026).
- **Zenodo Archive:** [https://doi.org/10.5281/zenodo.20694662](https://doi.org/10.5281/zenodo.20694662) (`10.5281/zenodo.20694662`).

## License

Released under the [MIT License](LICENSE).
