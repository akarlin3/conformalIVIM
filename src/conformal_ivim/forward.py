"""Self-contained bi-exponential and multi-echo IVIM forward models, noise physics,
and Cramér-Rao lower bound (CRLB) identifiability analysis.

The canonical 1D single-echo Le Bihan bi-exponential signal model is:

    S(b) / S0 = f * exp(-b * Dstar) + (1 - f) * exp(-b * D)

with b in s/mm^2, D and Dstar in mm^2/s, and f in [0, 1] the dimensionless
perfusion fraction. Also includes the 2D multi-echo diffusion-relaxometry model
S(b, TE), Rician / Gaussian / non-central chi magnitude noise, off-model
velocity-dispersion and tri-exponential decay extensions, synthetic cohort
generation, and analytical Fisher information / CRLB calculators.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

# Canonical 22-b IVIM scheme (s/mm^2): dense at low b (perfusion-sensitive)
# and sparser at high b (parenchymal tissue diffusion D).
DEFAULT_B_VALUES = np.array(
    [
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
        120, 140, 160, 180, 200, 300, 400, 500, 600, 700, 800,
    ],
    dtype=float,
)

# Standard 11-b clinical IVIM acquisition scheme (s/mm^2).
CLINICAL_B = np.array(
    [0, 10, 20, 40, 60, 80, 100, 200, 400, 600, 800],
    dtype=float,
)

_FALLBACK_CLINICAL_SCHEDULES: Dict[str, Tuple[float, ...]] = {
    "osipi": (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0, 250.0, 400.0, 600.0),
    "tcga_lihc": (0.0, 50.0, 500.0, 800.0),
    "acrin_6698": (0.0, 100.0, 600.0, 800.0),
    "lattice_22b": (
        0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0,
        120.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 1000.0,
    ),
}


def has_ivimbench_phantoms() -> bool:
    """Return ``True`` when ``ivimbench`` is available in the Python environment."""
    return importlib.util.find_spec("ivimbench") is not None


def resolve_clinical_b(
    b: Union[Sequence[float], np.ndarray, str],
    *,
    use_canonical_dro: bool = False,
) -> np.ndarray:
    """Resolve an array of ``b``-values or a named clinical schedule (``osipi``, ``tcga_lihc``, ``acrin_6698``)."""
    if isinstance(b, str):
        key = b.strip().lower()
        if use_canonical_dro and has_ivimbench_phantoms():
            import importlib as _importlib

            proto_mod = _importlib.import_module("ivimbench.protocols")
            res = proto_mod.resolve_bvalues(key)
            arr = res[0] if isinstance(res, tuple) else res
            return np.asarray(arr, dtype=float)
        if key in _FALLBACK_CLINICAL_SCHEDULES:
            return np.asarray(_FALLBACK_CLINICAL_SCHEDULES[key], dtype=float)
        raise ValueError(
            f"Unknown clinical b-schedule {b!r}; supported fallback keys: "
            f"{sorted(_FALLBACK_CLINICAL_SCHEDULES)}."
        )
    return np.asarray(b, dtype=float)

# Published abdominal/hepatic IVIM physiological parameter ranges (mm^2/s for D, D*).
D_RANGE: Tuple[float, float] = (0.5e-3, 3.0e-3)
DSTAR_RANGE: Tuple[float, float] = (10.0e-3, 100.0e-3)
F_RANGE: Tuple[float, float] = (0.05, 0.40)

DEFAULT_SNR_GRID: Tuple[int, ...] = (10, 20, 30, 50, 100)
DEFAULT_SEED: int = 20260613

# Free-fluid (CSF / edema-like) diffusivity at body temperature (mm^2/s).
D_FREE_FLUID: float = 3.0e-3

# Standard abdominal 1.5T/3T transverse relaxation constants (ms) for 2D S(b, TE).
DEFAULT_T2_TISSUE: float = 50.0
DEFAULT_T2_BLOOD: float = 150.0
CLOSER_T2_BLOOD: float = 100.0
DEFAULT_TE_GRID_2 = np.array([45.0, 90.0], dtype=float)
DEFAULT_TE_GRID_3 = np.array([45.0, 75.0, 110.0], dtype=float)

_PARAM_ORDER = ("D", "Dstar", "f", "S0")


def ivim_signal(b, D, Dstar, f, S0=1.0):
    """Bi-exponential IVIM signal S(b) = S0 * [f*exp(-b*Dstar) + (1-f)*exp(-b*D)].

    Parameters
    ----------
    b : float or array-like
        Diffusion-weighting b-values in s/mm^2.
    D : float or array-like
        Parenchymal tissue diffusion coefficient in mm^2/s.
    Dstar : float or array-like
        Microvascular pseudo-diffusion coefficient in mm^2/s (physically >> D).
    f : float or array-like
        Perfusion fraction in [0, 1].
    S0 : float or array-like, default=1.0
        Baseline unweighted signal intensity at b=0.
    """
    b = np.asarray(b, dtype=float)
    return S0 * (f * np.exp(-b * Dstar) + (1.0 - f) * np.exp(-b * D))


def add_rician_noise(signal, snr, rng, S0=1.0):
    """Add Rician-distributed magnitude noise at the given SNR (defined at b=0).

    Measured magnitude MRI signal: sqrt((S + n_re)^2 + n_im^2) with
    n_re, n_im ~ N(0, sigma^2) i.i.d. and sigma = S0 / snr.
    """
    sigma = S0 / snr
    shape = np.shape(signal)
    n_re = rng.normal(0.0, sigma, size=shape)
    n_im = rng.normal(0.0, sigma, size=shape)
    return np.sqrt((np.asarray(signal, dtype=float) + n_re) ** 2 + n_im ** 2)


def add_gaussian_noise(signal, snr, rng, S0=1.0):
    """Add additive Gaussian noise S + N(0, sigma^2) with sigma = S0 / snr."""
    sigma = S0 / snr
    shape = np.shape(signal)
    return np.asarray(signal, dtype=float) + rng.normal(0.0, sigma, size=shape)


def add_ncchi_noise(signal, snr, rng, L=1, S0=1.0):
    """Add non-central chi noise (L-coil sum-of-squares magnitude reconstruction).

    At L = 1, reduces byte-identically to :func:`add_rician_noise`.
    """
    L = int(L)
    sigma = S0 / snr
    shape = np.shape(signal)
    n_re = rng.normal(0.0, sigma, size=shape)
    n_im = rng.normal(0.0, sigma, size=shape)
    acc = (np.asarray(signal, dtype=float) + n_re) ** 2 + n_im ** 2
    for _ in range(L - 1):
        acc = (
            acc
            + rng.normal(0.0, sigma, size=shape) ** 2
            + rng.normal(0.0, sigma, size=shape) ** 2
        )
    return np.sqrt(acc)


def partial_volume_mix(signal, b, frac, D_free=D_FREE_FLUID):
    """Blend a free-fluid mono-exponential compartment into a clean IVIM signal."""
    b = np.asarray(b, dtype=float)
    s_free = np.exp(-b[None, :] * D_free)
    return (1.0 - frac) * np.asarray(signal, dtype=float) + frac * s_free


def apply_motion_phase(signal, phase):
    """Return complex signal S * exp(i * phase) modeling bulk-motion phase."""
    return np.asarray(signal, dtype=float) * np.exp(
        1j * np.asarray(phase, dtype=float)
    )


def _gamma_perfusion_kernel(b, mu, k):
    """E[exp(-b D*)] for D* ~ Gamma(shape=k, mean=mu): (1 + b mu / k) ** (-k)."""
    b = np.asarray(b, dtype=float)
    mu = np.asarray(mu, dtype=float)
    k = np.asarray(k, dtype=float)
    finite = np.isfinite(k)
    k_safe = np.where(finite, k, 1.0)
    ratio = b * mu / k_safe
    log_perf_finite = -k_safe * np.log1p(ratio)
    log_perf_limit = -b * mu
    log_perf = np.where(finite, log_perf_finite, log_perf_limit)
    return np.exp(log_perf)


def ivim_signal_dispersion(b, D, mu, k, f, S0=1.0):
    """Velocity-dispersion IVIM signal with Gamma-distributed pseudo-diffusivity."""
    b = np.asarray(b, dtype=float)
    perf = _gamma_perfusion_kernel(b, mu, k)
    return S0 * ((1.0 - f) * np.exp(-b * D) + f * perf)


def dispersion_dstar_eff(mu, k=None):
    """Effective pseudo-diffusion D*eff (= distribution mean mu) of dispersion model."""
    return np.asarray(mu, dtype=float)


def _lognormal_perfusion_kernel(b, mu, cv, n_quad=64):
    """E[exp(-b D*)] for D* ~ LogNormal(mean=mu, CV=cv) via Gauss-Hermite quadrature."""
    cv = float(cv)
    b = np.asarray(b, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if cv == 0.0:
        return np.exp(-b * mu)
    bA, muA = np.broadcast_arrays(b, mu)
    sig2 = np.log1p(cv * cv)
    sig = np.sqrt(sig2)
    mu_ln = np.log(muA) - 0.5 * sig2
    nodes, weights = np.polynomial.hermite.hermgauss(int(n_quad))
    dstar_q = np.exp(mu_ln[..., None] + sig * np.sqrt(2.0) * nodes)
    return (weights * np.exp(-bA[..., None] * dstar_q)).sum(-1) / np.sqrt(np.pi)


def ivim_signal_lognormal_dispersion(b, D, mu, cv, f, S0=1.0, n_quad=64):
    """Log-normal velocity-dispersion IVIM signal."""
    b = np.asarray(b, dtype=float)
    perf = _lognormal_perfusion_kernel(b, mu, cv, n_quad=n_quad)
    return S0 * ((1.0 - f) * np.exp(-b * D) + f * perf)


def ivim_signal_stretched(b, D, Dstar, f, beta, S0=1.0):
    """Stretched-exponential (anomalous) perfusion IVIM signal."""
    b = np.asarray(b, dtype=float)
    perf = np.exp(-np.power(np.clip(b * Dstar, 0.0, None), beta))
    return S0 * ((1.0 - f) * np.exp(-b * D) + f * perf)


def ivim_signal_triexp(b, D, Dstar, f, Dstar2, g, S0=1.0):
    """Tri-exponential IVIM signal: bi-exponential plus a faster vascular pool."""
    b = np.asarray(b, dtype=float)
    return S0 * (
        (1.0 - f) * np.exp(-b * D)
        + f * (1.0 - g) * np.exp(-b * Dstar)
        + f * g * np.exp(-b * Dstar2)
    )


# --------------------------------------------------------------------------- #
# 2D Multi-Echo Diffusion-Relaxometry IVIM Forward Model S(b, TE)
# --------------------------------------------------------------------------- #
def ivim_signal_2d(
    b,
    te,
    D,
    Dstar,
    f,
    S0=1.0,
    T2_tissue=DEFAULT_T2_TISSUE,
    T2_blood=DEFAULT_T2_BLOOD,
):
    """Compartment-specific T2-weighted 2D multi-echo IVIM signal S(b, TE).

    Model:
        S(b, TE) = S0 * [ f * exp(-TE / T2_blood) * exp(-b * Dstar)
                        + (1 - f) * exp(-TE / T2_tissue) * exp(-b * D) ]
    """
    b = np.asarray(b, dtype=float)
    te = np.asarray(te, dtype=float)
    w_b = np.exp(-te / float(T2_blood))
    w_t = np.exp(-te / float(T2_tissue))
    return S0 * (f * w_b * np.exp(-b * Dstar) + (1.0 - f) * w_t * np.exp(-b * D))


def ivim_reference_signal_2d(
    te_min,
    f,
    S0=1.0,
    T2_tissue=DEFAULT_T2_TISSUE,
    T2_blood=DEFAULT_T2_BLOOD,
):
    """Reference signal S(b=0, TE=TE_min) defining sigma = S(0, TE_min) / SNR."""
    f = np.asarray(f, dtype=float)
    w_b0 = np.exp(-float(te_min) / float(T2_blood))
    w_t0 = np.exp(-float(te_min) / float(T2_tissue))
    return S0 * (f * w_b0 + (1.0 - f) * w_t0)


# --------------------------------------------------------------------------- #
# Analytical Jacobians & Cramér-Rao Lower Bounds (1D & 2D)
# --------------------------------------------------------------------------- #
def ivim_jacobian(b, D, Dstar, f, S0=1.0):
    """Jacobian of 1D IVIM signal wrt (D, Dstar, f, S0) for one voxel -> (n_b, 4)."""
    b = np.asarray(b, dtype=float)
    ed = np.exp(-b * D)
    eds = np.exp(-b * Dstar)
    dD = S0 * (1.0 - f) * (-b) * ed
    dDstar = S0 * f * (-b) * eds
    df = S0 * (eds - ed)
    dS0 = f * eds + (1.0 - f) * ed
    return np.stack([dD, dDstar, df, dS0], axis=1)


def ivim_jacobian_2d(
    b,
    te,
    D,
    Dstar,
    f,
    S0=1.0,
    T2_tissue=DEFAULT_T2_TISSUE,
    T2_blood=DEFAULT_T2_BLOOD,
):
    """Jacobian J(b, TE) of 2D multi-echo IVIM signal wrt (D, Dstar, f, S0)."""
    b = np.asarray(b, dtype=float)
    te = np.asarray(te, dtype=float)
    w_b = np.exp(-te / float(T2_blood))
    w_t = np.exp(-te / float(T2_tissue))
    ed = w_t * np.exp(-b * D)
    eds = w_b * np.exp(-b * Dstar)
    dD = S0 * (1.0 - f) * (-b) * ed
    dDstar = S0 * f * (-b) * eds
    df = S0 * (eds - ed)
    dS0 = f * eds + (1.0 - f) * ed
    return np.stack([dD, dDstar, df, dS0], axis=1)


def crlb(b, D, Dstar, f, S0=1.0, snr=None, sigma=None, fix_s0=False):
    """Cramér-Rao lower-bound standard deviations for a single 1D IVIM voxel."""
    if sigma is None:
        if snr is None:
            raise ValueError("provide snr or sigma")
        sigma = S0 / float(snr)
    J = ivim_jacobian(b, D, Dstar, f, S0=S0)
    if fix_s0:
        J = J[:, :3]
        names = _PARAM_ORDER[:3]
    else:
        names = _PARAM_ORDER
    info = J.T @ J / (sigma * sigma)
    try:
        cov = np.linalg.inv(info)
        var = np.diag(cov)
    except np.linalg.LinAlgError:
        var = np.full(len(names), np.inf)
    out = {}
    for k, nm in enumerate(names):
        v = var[k]
        out[nm] = float(np.sqrt(v)) if np.isfinite(v) and v > 0 else np.inf
    return out


def crlb_2d(
    b,
    te,
    D,
    Dstar,
    f,
    S0=1.0,
    snr=None,
    sigma=None,
    T2_tissue=DEFAULT_T2_TISSUE,
    T2_blood=DEFAULT_T2_BLOOD,
    fix_s0=False,
):
    """Cramér-Rao lower-bound standard deviations for one 2D (b, TE) IVIM voxel."""
    b = np.asarray(b, dtype=float)
    te = np.asarray(te, dtype=float)
    if sigma is None:
        if snr is None:
            raise ValueError("provide snr or sigma")
        s_ref = float(
            ivim_reference_signal_2d(
                np.min(te), f, S0=S0, T2_tissue=T2_tissue, T2_blood=T2_blood
            )
        )
        sigma = s_ref / float(snr)
    J = ivim_jacobian_2d(
        b, te, D, Dstar, f, S0=S0, T2_tissue=T2_tissue, T2_blood=T2_blood
    )
    if fix_s0:
        J = J[:, :3]
        names = _PARAM_ORDER[:3]
    else:
        names = _PARAM_ORDER
    info = J.T @ J / (sigma * sigma)
    try:
        cov = np.linalg.inv(info)
        var = np.diag(cov)
    except np.linalg.LinAlgError:
        var = np.full(len(names), np.inf)
    out = {}
    for k, nm in enumerate(names):
        v = var[k]
        out[nm] = float(np.sqrt(v)) if np.isfinite(v) and v > 0 else np.inf
    return out


def crlb_dstar_batch(b, D, Dstar, f, snr, S0=1.0, fix_s0=False):
    """Vectorized CRLB standard deviation for D* over N voxels in 1D multi-b IVIM."""
    b = np.asarray(b, dtype=float)
    D = np.asarray(D, dtype=float)
    Dstar = np.asarray(Dstar, dtype=float)
    f = np.asarray(f, dtype=float)
    snr = np.asarray(snr, dtype=float)
    N = D.shape[0]
    S0_arr = np.broadcast_to(np.asarray(S0, dtype=float), (N,))
    sigma = S0_arr / snr
    ed = np.exp(-b[None, :] * D[:, None])
    eds = np.exp(-b[None, :] * Dstar[:, None])
    bb = b[None, :]
    cols = [
        S0_arr[:, None] * (1.0 - f[:, None]) * (-bb) * ed,
        S0_arr[:, None] * f[:, None] * (-bb) * eds,
        S0_arr[:, None] * (eds - ed),
        f[:, None] * eds + (1.0 - f[:, None]) * ed,
    ]
    if fix_s0:
        cols = cols[:3]
    J = np.stack(cols, axis=2)
    info = np.einsum("nbi,nbj->nij", J, J) / (sigma * sigma)[:, None, None]
    out = np.full(N, np.inf)
    try:
        cov = np.linalg.inv(info)
        var = cov[:, 1, 1]
        good = np.isfinite(var) & (var > 0)
        out[good] = np.sqrt(var[good])
    except np.linalg.LinAlgError:
        for i in range(N):
            try:
                v = np.linalg.inv(info[i])[1, 1]
                out[i] = np.sqrt(v) if v > 0 else np.inf
            except np.linalg.LinAlgError:
                out[i] = np.inf
    return out


def crlb_dstar_2d_batch(
    b,
    te,
    D,
    Dstar,
    f,
    snr,
    S0=1.0,
    T2_tissue=DEFAULT_T2_TISSUE,
    T2_blood=DEFAULT_T2_BLOOD,
    fix_s0=False,
):
    """Vectorized CRLB standard deviation for D* over N voxels under 2D (b, TE)."""
    b = np.asarray(b, dtype=float)
    te = np.asarray(te, dtype=float)
    D = np.asarray(D, dtype=float)
    Dstar = np.asarray(Dstar, dtype=float)
    f = np.asarray(f, dtype=float)
    snr = np.asarray(snr, dtype=float)
    N = D.shape[0]
    S0_arr = np.broadcast_to(np.asarray(S0, dtype=float), (N,))
    s_ref = ivim_reference_signal_2d(
        np.min(te), f, S0=S0_arr, T2_tissue=T2_tissue, T2_blood=T2_blood
    )
    sigma = s_ref / snr

    bb = b[None, :]
    w_b = np.exp(-te[None, :] / float(T2_blood))
    w_t = np.exp(-te[None, :] / float(T2_tissue))
    ed = w_t * np.exp(-bb * D[:, None])
    eds = w_b * np.exp(-bb * Dstar[:, None])
    cols = [
        S0_arr[:, None] * (1.0 - f[:, None]) * (-bb) * ed,
        S0_arr[:, None] * f[:, None] * (-bb) * eds,
        S0_arr[:, None] * (eds - ed),
        f[:, None] * eds + (1.0 - f[:, None]) * ed,
    ]
    if fix_s0:
        cols = cols[:3]
    J = np.stack(cols, axis=2)
    info = np.einsum("nbi,nbj->nij", J, J) / (sigma * sigma)[:, None, None]
    out = np.full(N, np.inf)
    try:
        cov = np.linalg.inv(info)
        var = cov[:, 1, 1]
        good = np.isfinite(var) & (var > 0)
        out[good] = np.sqrt(var[good])
    except np.linalg.LinAlgError:
        for i in range(N):
            try:
                v = np.linalg.inv(info[i])[1, 1]
                out[i] = np.sqrt(v) if v > 0 else np.inf
            except np.linalg.LinAlgError:
                out[i] = np.inf
    return out


def dispersion_crlb_mu_batch(b, D, mu, k, f, snr, S0=1.0, fix_k=True):
    """Vectorized CRLB std for D*eff = mu under the Gamma velocity-dispersion model."""
    b = np.asarray(b, dtype=float)
    D = np.asarray(D, dtype=float)
    mu = np.asarray(mu, dtype=float)
    k = np.asarray(k, dtype=float)
    f = np.asarray(f, dtype=float)
    snr = np.asarray(snr, dtype=float)
    N = D.shape[0]
    S0_arr = np.broadcast_to(np.asarray(S0, dtype=float), (N,))
    sigma = S0_arr / snr
    bb = b[None, :]
    ed = np.exp(-bb * D[:, None])
    kk = k[:, None]
    u = 1.0 + bb * mu[:, None] / kk
    perf = np.power(u, -kk)
    dD = S0_arr[:, None] * (1.0 - f[:, None]) * (-bb) * ed
    dmu = -S0_arr[:, None] * f[:, None] * bb * np.power(u, -(kk + 1.0))
    df = S0_arr[:, None] * (perf - ed)
    dS0 = (1.0 - f[:, None]) * ed + f[:, None] * perf
    cols = [dD, dmu, df, dS0]
    if not fix_k:
        dk = S0_arr[:, None] * f[:, None] * perf * (1.0 - 1.0 / u - np.log(u))
        cols = [dD, dmu, dk, df, dS0]
    J = np.stack(cols, axis=2)
    info = np.einsum("nbi,nbj->nij", J, J) / (sigma * sigma)[:, None, None]
    out = np.full(N, np.inf)
    for i in range(N):
        try:
            v = np.linalg.inv(info[i])[1, 1]
            out[i] = np.sqrt(v) if np.isfinite(v) and v > 0 else np.inf
        except np.linalg.LinAlgError:
            out[i] = np.inf
    return out


def design_crlb_dstar(b, D, Dstar, f, snr, S0=1.0, fix_s0=False, hi_mask=None):
    """Acquisition-design objective: mean CRLB(D*) over a representative voxel set."""
    Dstar = np.asarray(Dstar, dtype=float)
    sd = crlb_dstar_batch(b, D, Dstar, f, snr, S0=S0, fix_s0=fix_s0)
    if hi_mask is None:
        hi_mask = Dstar >= np.quantile(Dstar, 2.0 / 3.0)
    finite = np.isfinite(sd)
    mean_all = float(np.mean(sd[finite])) if finite.any() else np.inf
    hi_finite = finite & hi_mask
    mean_hi = float(np.mean(sd[hi_finite])) if hi_finite.any() else np.inf
    frac_inf = float(np.mean(~finite))
    return mean_all, mean_hi, frac_inf


def crlb_optimal_scheme(n_b=11, seed=DEFAULT_SEED):
    """Greedy forward-selection b-value scheme minimizing high-D* CRLB(D*)."""
    rng = np.random.default_rng(int(seed) + 7)
    D = rng.uniform(*D_RANGE, size=300)
    Dstar = rng.uniform(*DSTAR_RANGE, size=300)
    f = rng.uniform(*F_RANGE, size=300)
    snr = rng.choice(np.asarray(DEFAULT_SNR_GRID, dtype=float), size=300)
    hi = Dstar >= np.quantile(Dstar, 2.0 / 3.0)
    cand = np.unique(
        np.r_[np.arange(0, 101, 5), np.arange(120, 801, 20)]
    ).astype(float)
    chosen = [0.0]
    cand = cand[cand != 0.0]

    def score(bvals):
        bv = np.sort(np.asarray(bvals, dtype=float))
        _, mean_hi, _ = design_crlb_dstar(bv, D, Dstar, f, snr, hi_mask=hi)
        return mean_hi

    while len(chosen) < n_b and cand.size:
        best, best_b = np.inf, None
        for c in cand:
            sc = score(chosen + [c])
            if sc < best:
                best, best_b = sc, c
        chosen.append(float(best_b))
        cand = cand[cand != best_b]
    return np.sort(np.array(chosen, dtype=float))


# --------------------------------------------------------------------------- #
# Clean High-Level Public API Functions: simulate_ivim_signal & compute_crlb
# --------------------------------------------------------------------------- #
def _evaluate_family_clean_1d(
    b_arr: np.ndarray,
    d_vec: np.ndarray,
    ds_vec: np.ndarray,
    f_vec: np.ndarray,
    s0_vec: np.ndarray,
    *,
    family: str = "biexp",
    use_canonical_dro: bool = False,
) -> np.ndarray:
    fam = str(family).strip().lower()
    if use_canonical_dro and has_ivimbench_phantoms():
        import importlib as _importlib

        ph_mod = _importlib.import_module("ivimbench.phantoms")
        d1 = np.asarray(d_vec, dtype=float).ravel()
        ds1 = np.asarray(ds_vec, dtype=float).ravel()
        f1 = np.asarray(f_vec, dtype=float).ravel()
        s01 = np.asarray(s0_vec, dtype=float).ravel()
        if fam == "biexp":
            return np.asarray(ph_mod.ivim_biexp(b_arr, d1, ds1, f1, s0=s01), dtype=float)
        if fam == "dispersion_gamma":
            return np.asarray(ph_mod.ivim_dispersion_gamma(b_arr, d1, ds1, 4.0, f1, s0=s01), dtype=float)
        if fam == "dispersion_lognormal":
            return np.asarray(ph_mod.ivim_dispersion_lognormal(b_arr, d1, ds1, 0.5, f1, s0=s01), dtype=float)
        if fam == "stretched":
            return np.asarray(ph_mod.ivim_stretched(b_arr, d1, ds1, f1, 0.7, s0=s01), dtype=float)
        if fam == "triexp":
            return np.asarray(ph_mod.ivim_triexp(b_arr, d1, ds1, f1, 4.0 * ds1, 0.3, s0=s01), dtype=float)
        raise ValueError(f"Unknown IVIM family {family!r}.")

    d_b = np.asarray(d_vec, dtype=float).reshape(-1, 1)
    ds_b = np.asarray(ds_vec, dtype=float).reshape(-1, 1)
    f_b = np.asarray(f_vec, dtype=float).reshape(-1, 1)
    s0_b = np.asarray(s0_vec, dtype=float).reshape(-1, 1)
    if fam == "biexp":
        return ivim_signal(b_arr[None, :], d_b, ds_b, f_b, S0=s0_b)
    if fam == "dispersion_gamma":
        return ivim_signal_dispersion(b_arr[None, :], d_b, ds_b, 4.0, f_b, S0=s0_b)
    if fam == "dispersion_lognormal":
        return ivim_signal_lognormal_dispersion(
            b_arr[None, :], d_b, ds_b, 0.5, f_b, S0=s0_b
        )
    if fam == "stretched":
        return ivim_signal_stretched(b_arr[None, :], d_b, ds_b, f_b, 0.7, S0=s0_b)
    if fam == "triexp":
        return ivim_signal_triexp(b_arr[None, :], d_b, ds_b, f_b, 4.0 * ds_b, 0.3, S0=s0_b)
    raise ValueError(
        f"Unknown IVIM family {family!r}; expected one of "
        "('biexp', 'dispersion_gamma', 'dispersion_lognormal', 'stretched', 'triexp')."
    )


def simulate_ivim_signal(
    b=DEFAULT_B_VALUES,
    D=1.2e-3,
    Dstar=30.0e-3,
    f=0.20,
    S0=1.0,
    snr=None,
    noise="rician",
    te=None,
    T2_tissue=DEFAULT_T2_TISSUE,
    T2_blood=DEFAULT_T2_BLOOD,
    ncchi_coils=1,
    seed=None,
    rng=None,
    *,
    use_canonical_dro: bool = False,
    family: str = "biexp",
):
    """Simulate clean or noisy 1D multi-b or 2D (b, TE) IVIM signals with optional canonical DRO delegation."""
    b_arr = resolve_clinical_b(b, use_canonical_dro=use_canonical_dro)
    D_arr = np.asarray(D, dtype=float)
    Ds_arr = np.asarray(Dstar, dtype=float)
    f_arr = np.asarray(f, dtype=float)
    S0_arr = np.asarray(S0, dtype=float)

    is_batch = any(
        arr.ndim >= 1 for arr in (D_arr, Ds_arr, f_arr, S0_arr)
    ) or (snr is not None and np.asarray(snr).ndim >= 1)

    if is_batch:
        N = max(
            [
                arr.shape[0]
                for arr in (D_arr, Ds_arr, f_arr, S0_arr)
                if arr.ndim >= 1
            ]
            + ([np.asarray(snr).shape[0]] if snr is not None and np.asarray(snr).ndim >= 1 else [1])
        )
        D_b = np.broadcast_to(D_arr, (N,))[:, None]
        Ds_b = np.broadcast_to(Ds_arr, (N,))[:, None]
        f_b = np.broadcast_to(f_arr, (N,))[:, None]
        S0_b = np.broadcast_to(S0_arr, (N,))[:, None]
        if te is None:
            if family == "biexp" and not use_canonical_dro:
                clean = ivim_signal(b_arr[None, :], D_b, Ds_b, f_b, S0=S0_b)
            else:
                clean = _evaluate_family_clean_1d(
                    b_arr, D_b, Ds_b, f_b, S0_b, family=family, use_canonical_dro=use_canonical_dro
                )
            s_ref = S0_b
        else:
            te_arr = np.asarray(te, dtype=float)
            clean = ivim_signal_2d(
                b_arr[None, :],
                te_arr[None, :],
                D_b,
                Ds_b,
                f_b,
                S0=S0_b,
                T2_tissue=T2_tissue,
                T2_blood=T2_blood,
            )
            s_ref = ivim_reference_signal_2d(
                np.min(te_arr),
                f_b,
                S0=S0_b,
                T2_tissue=T2_tissue,
                T2_blood=T2_blood,
            )
    else:
        if te is None:
            if family == "biexp" and not use_canonical_dro:
                clean = ivim_signal(
                    b_arr, float(D_arr), float(Ds_arr), float(f_arr), S0=float(S0_arr)
                )
            else:
                clean = _evaluate_family_clean_1d(
                    b_arr,
                    np.array([float(D_arr)]),
                    np.array([float(Ds_arr)]),
                    np.array([float(f_arr)]),
                    np.array([float(S0_arr)]),
                    family=family,
                    use_canonical_dro=use_canonical_dro,
                )[0]
            s_ref = float(S0_arr)
        else:
            te_arr = np.asarray(te, dtype=float)
            clean = ivim_signal_2d(
                b_arr,
                te_arr,
                float(D_arr),
                float(Ds_arr),
                float(f_arr),
                S0=float(S0_arr),
                T2_tissue=T2_tissue,
                T2_blood=T2_blood,
            )
            s_ref = float(
                ivim_reference_signal_2d(
                    np.min(te_arr),
                    float(f_arr),
                    S0=float(S0_arr),
                    T2_tissue=T2_tissue,
                    T2_blood=T2_blood,
                )
            )

    if snr is None:
        return clean

    gen = rng if rng is not None else np.random.default_rng(seed)
    snr_val = np.asarray(snr, dtype=float)
    if is_batch:
        snr_val = np.broadcast_to(snr_val, (clean.shape[0],))[:, None]
    else:
        snr_val = float(snr_val)

    noise_key = str(noise).lower()
    if noise_key == "rician":
        return add_rician_noise(clean, snr_val, gen, S0=s_ref)
    if noise_key == "gaussian":
        return add_gaussian_noise(clean, snr_val, gen, S0=s_ref)
    if noise_key in ("ncchi", "nc-chi"):
        return add_ncchi_noise(clean, snr_val, gen, L=ncchi_coils, S0=s_ref)
    raise ValueError(f"Unknown noise model {noise!r}; expected 'rician', 'gaussian', or 'ncchi'.")


def compute_crlb(
    b,
    D,
    Dstar,
    f,
    S0=1.0,
    snr=None,
    sigma=None,
    te=None,
    T2_tissue=DEFAULT_T2_TISSUE,
    T2_blood=DEFAULT_T2_BLOOD,
    fix_s0=False,
) -> Dict[str, Union[float, np.ndarray]]:
    """Compute Cramér-Rao lower-bound standard deviations for (D, Dstar, f[, S0]).

    Supports both single-voxel scalar inputs (returning floats) and batch 1D
    arrays of length ``N`` (returning ``(N,)`` arrays), for either 1D multi-b
    (``te=None``) or 2D multi-echo ``(b, TE)`` acquisitions.
    """
    b_arr = np.asarray(b, dtype=float)
    D_arr = np.asarray(D, dtype=float)
    Ds_arr = np.asarray(Dstar, dtype=float)
    f_arr = np.asarray(f, dtype=float)
    S0_arr = np.asarray(S0, dtype=float)

    extra_arrays = []
    if snr is not None:
        extra_arrays.append(np.asarray(snr, dtype=float))
    if sigma is not None:
        extra_arrays.append(np.asarray(sigma, dtype=float))

    is_batch = any(
        arr.ndim >= 1 for arr in (D_arr, Ds_arr, f_arr, S0_arr, *extra_arrays)
    )
    if not is_batch:
        if te is None:
            return crlb(
                b_arr,
                float(D_arr),
                float(Ds_arr),
                float(f_arr),
                S0=float(S0_arr),
                snr=snr,
                sigma=sigma,
                fix_s0=fix_s0,
            )
        return crlb_2d(
            b_arr,
            np.asarray(te, dtype=float),
            float(D_arr),
            float(Ds_arr),
            float(f_arr),
            S0=float(S0_arr),
            snr=snr,
            sigma=sigma,
            T2_tissue=T2_tissue,
            T2_blood=T2_blood,
            fix_s0=fix_s0,
        )

    N = max(
        arr.shape[0]
        for arr in (D_arr, Ds_arr, f_arr, S0_arr, *extra_arrays)
        if arr.ndim >= 1
    )
    D_v = np.broadcast_to(D_arr, (N,))
    Ds_v = np.broadcast_to(Ds_arr, (N,))
    f_v = np.broadcast_to(f_arr, (N,))
    S0_v = np.broadcast_to(S0_arr, (N,))

    if sigma is None:
        if snr is None:
            raise ValueError("provide snr or sigma")
        snr_v = np.broadcast_to(np.asarray(snr, dtype=float), (N,))
        if te is None:
            sigma_v = S0_v / snr_v
        else:
            te_arr = np.asarray(te, dtype=float)
            s_ref = ivim_reference_signal_2d(
                np.min(te_arr),
                f_v,
                S0=S0_v,
                T2_tissue=T2_tissue,
                T2_blood=T2_blood,
            )
            sigma_v = s_ref / snr_v
    else:
        sigma_v = np.broadcast_to(np.asarray(sigma, dtype=float), (N,))

    bb = b_arr[None, :]
    if te is None:
        ed = np.exp(-bb * D_v[:, None])
        eds = np.exp(-bb * Ds_v[:, None])
    else:
        te_arr = np.asarray(te, dtype=float)
        w_b = np.exp(-te_arr[None, :] / float(T2_blood))
        w_t = np.exp(-te_arr[None, :] / float(T2_tissue))
        ed = w_t * np.exp(-bb * D_v[:, None])
        eds = w_b * np.exp(-bb * Ds_v[:, None])

    cols = [
        S0_v[:, None] * (1.0 - f_v[:, None]) * (-bb) * ed,
        S0_v[:, None] * f_v[:, None] * (-bb) * eds,
        S0_v[:, None] * (eds - ed),
        f_v[:, None] * eds + (1.0 - f_v[:, None]) * ed,
    ]
    names = _PARAM_ORDER[:3] if fix_s0 else _PARAM_ORDER
    if fix_s0:
        cols = cols[:3]
    J = np.stack(cols, axis=2)
    info = np.einsum("nbi,nbj->nij", J, J) / (sigma_v * sigma_v)[:, None, None]

    P = len(names)
    sds = np.full((N, P), np.inf, dtype=float)
    try:
        cov = np.linalg.inv(info)
        diag_var = np.diagonal(cov, axis1=1, axis2=2)
        good = np.isfinite(diag_var) & (diag_var > 0)
        sds[good] = np.sqrt(diag_var[good])
    except np.linalg.LinAlgError:
        for i in range(N):
            try:
                cov_i = np.linalg.inv(info[i])
                v_i = np.diag(cov_i)
                good_i = np.isfinite(v_i) & (v_i > 0)
                sds[i, good_i] = np.sqrt(v_i[good_i])
            except np.linalg.LinAlgError:
                pass
    return {nm: sds[:, k] for k, nm in enumerate(names)}


# --------------------------------------------------------------------------- #
# Labeled Synthetic IVIM Cohort Generator (train / cal / test splits)
# --------------------------------------------------------------------------- #
@dataclass
class Cohort:
    """A seeded synthetic IVIM cohort with exchangeable train/cal/test splits."""

    b: np.ndarray
    snr_grid: Tuple[float, ...]
    seed: int
    signals: Dict[str, np.ndarray]
    params: Dict[str, np.ndarray]
    snr: Dict[str, np.ndarray]

    @property
    def sizes(self) -> Dict[str, int]:
        return {k: v.shape[0] for k, v in self.signals.items()}


def _draw_split(
    n: int,
    snr_grid: Sequence[float],
    b: np.ndarray,
    rng: np.random.Generator,
    *,
    use_canonical_dro: bool = False,
    family: str = "biexp",
    organ: Optional[str] = None,
):
    if use_canonical_dro and organ is not None and has_ivimbench_phantoms():
        import importlib as _importlib

        ph_mod = _importlib.import_module("ivimbench.phantoms")
        params = np.asarray(
            ph_mod.sample_organ_params(organ=organ, n_voxels=n, seed=rng),
            dtype=float,
        )
        D, Dstar, f = params[:, 0], params[:, 1], params[:, 2]
    else:
        D = rng.uniform(*D_RANGE, size=n)
        Dstar = rng.uniform(*DSTAR_RANGE, size=n)
        f = rng.uniform(*F_RANGE, size=n)
        params = np.stack([D, Dstar, f], axis=1)
    snr = rng.choice(np.asarray(snr_grid, dtype=float), size=n)
    if family == "biexp" and not use_canonical_dro:
        clean = ivim_signal(b[None, :], D[:, None], Dstar[:, None], f[:, None], S0=1.0)
    else:
        clean = _evaluate_family_clean_1d(
            b,
            D[:, None],
            Dstar[:, None],
            f[:, None],
            np.ones((n, 1), dtype=float),
            family=family,
            use_canonical_dro=use_canonical_dro,
        )
    noisy = add_rician_noise(clean, snr[:, None], rng, S0=1.0)
    return noisy, params, snr


def generate_cohort(
    n_train: int,
    n_cal: int,
    n_test: int,
    snr_grid: Sequence[float] = DEFAULT_SNR_GRID,
    b: Union[Sequence[float], str] = DEFAULT_B_VALUES,
    seed: int = DEFAULT_SEED,
    *,
    use_canonical_dro: bool = False,
    family: str = "biexp",
    organ: Optional[str] = None,
) -> Cohort:
    """Generate a reproducible train/cal/test synthetic IVIM cohort."""
    b_arr = resolve_clinical_b(b, use_canonical_dro=use_canonical_dro)
    rng = np.random.default_rng(seed)
    signals, params, snr = {}, {}, {}
    for name, n in (("train", n_train), ("cal", n_cal), ("test", n_test)):
        s, p, r = _draw_split(
            n,
            snr_grid,
            b_arr,
            rng,
            use_canonical_dro=use_canonical_dro,
            family=family,
            organ=organ,
        )
        signals[name], params[name], snr[name] = s, p, r
    return Cohort(
        b=b_arr,
        snr_grid=tuple(snr_grid),
        seed=seed,
        signals=signals,
        params=params,
        snr=snr,
    )

