"""Property and unit tests for 1D/2D IVIM forward models, noise physics,
Cramér-Rao lower bounds (CRLB), and synthetic cohort generation.
"""
import numpy as np
import pytest

from conformal_ivim import compute_crlb, simulate_ivim_signal
from conformal_ivim.forward import (
    CLINICAL_B,
    DEFAULT_B_VALUES,
    DSTAR_RANGE,
    D_RANGE,
    F_RANGE,
    add_gaussian_noise,
    add_ncchi_noise,
    add_rician_noise,
    apply_motion_phase,
    crlb,
    crlb_2d,
    crlb_dstar_2d_batch,
    crlb_dstar_batch,
    crlb_optimal_scheme,
    design_crlb_dstar,
    dispersion_crlb_mu_batch,
    dispersion_dstar_eff,
    generate_cohort,
    ivim_jacobian,
    ivim_jacobian_2d,
    ivim_signal,
    ivim_signal_2d,
    ivim_signal_dispersion,
    ivim_signal_lognormal_dispersion,
    ivim_signal_stretched,
    ivim_signal_triexp,
    partial_volume_mix,
)

# Physical-unit reference parameters (mm^2/s), liver-like.
D_REF, DSTAR_REF, F_REF = 1.5e-3, 50e-3, 0.20


# --------------------------------------------------------------------------- #
# Core bi-exponential forward model & Rician noise property tests (Gauge 01)
# --------------------------------------------------------------------------- #
def test_signal_at_b0_equals_s0():
    for S0 in (1.0, 0.5, 1234.0):
        for D, Dstar, f in [(1.0e-3, 20e-3, 0.1), (2.5e-3, 90e-3, 0.35)]:
            assert ivim_signal(0.0, D, Dstar, f, S0=S0) == pytest.approx(S0)


def test_f_zero_is_monoexponential():
    b = DEFAULT_B_VALUES
    got = ivim_signal(b, D_REF, DSTAR_REF, 0.0, S0=1.0)
    expected = np.exp(-b * D_REF)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)


def test_dstar_equals_d_is_monoexponential_for_any_f():
    b = DEFAULT_B_VALUES
    for f in (0.0, 0.2, 0.5, 0.9):
        got = ivim_signal(b, D_REF, D_REF, f, S0=1.0)
        expected = np.exp(-b * D_REF)
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)


def test_signal_strictly_decreasing_in_b():
    b = np.linspace(0, 1000, 200)
    s = ivim_signal(b, D_REF, DSTAR_REF, F_REF, S0=1.0)
    assert np.all(np.diff(s) < 0)


def test_high_b_logslope_recovers_D():
    b = np.array([300, 400, 500, 600, 700, 800], float)
    S0 = 1.0
    s = ivim_signal(b, D_REF, DSTAR_REF, F_REF, S0=S0)
    slope, intercept = np.polyfit(b, np.log(s), 1)
    assert -slope == pytest.approx(D_REF, rel=1e-3)
    assert np.exp(intercept) == pytest.approx(S0 * (1 - F_REF), rel=1e-3)


def test_rician_noise_is_deterministic_with_seed():
    clean = ivim_signal(DEFAULT_B_VALUES, D_REF, DSTAR_REF, F_REF)
    a = add_rician_noise(clean, snr=25.0, rng=np.random.default_rng(7))
    b = add_rician_noise(clean, snr=25.0, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_rician_noise_floor_on_zero_signal():
    sigma = 0.04
    snr = 1.0 / sigma
    rng = np.random.default_rng(0)
    zeros = np.zeros(200_000)
    noisy = add_rician_noise(zeros, snr=snr, rng=rng, S0=1.0)
    assert noisy.mean() == pytest.approx(sigma * np.sqrt(np.pi / 2), rel=2e-2)


def test_high_snr_approximates_clean_signal():
    clean = ivim_signal(DEFAULT_B_VALUES, D_REF, DSTAR_REF, F_REF)
    noisy = add_rician_noise(clean, snr=5000.0, rng=np.random.default_rng(1))
    np.testing.assert_allclose(noisy, clean, atol=2e-3)


def test_noise_std_scales_inversely_with_snr():
    clean = ivim_signal(np.full(50_000, 0.5), D_REF, DSTAR_REF, F_REF)
    rng = np.random.default_rng(3)
    std_lo = add_rician_noise(clean, snr=10.0, rng=rng).std()
    std_hi = add_rician_noise(clean, snr=40.0, rng=rng).std()
    assert std_lo / std_hi == pytest.approx(4.0, rel=0.15)


# --------------------------------------------------------------------------- #
# Synthetic cohort generator tests (Gauge 01)
# --------------------------------------------------------------------------- #
def test_cohort_is_deterministic_under_seed():
    a = generate_cohort(200, 100, 100, snr_grid=(20, 50), seed=42)
    b = generate_cohort(200, 100, 100, snr_grid=(20, 50), seed=42)
    np.testing.assert_array_equal(a.signals["train"], b.signals["train"])
    np.testing.assert_array_equal(a.params["test"], b.params["test"])
    np.testing.assert_array_equal(a.snr["cal"], b.snr["cal"])


def test_cohort_shapes_and_split_sizes():
    c = generate_cohort(300, 120, 150, snr_grid=(10, 30, 60), seed=1)
    nb = DEFAULT_B_VALUES.size
    assert c.signals["train"].shape == (300, nb)
    assert c.signals["cal"].shape == (120, nb)
    assert c.signals["test"].shape == (150, nb)
    assert c.params["train"].shape == (300, 3)
    assert c.snr["test"].shape == (150,)
    assert c.sizes == {"train": 300, "cal": 120, "test": 150}


def test_params_within_published_ranges():
    c = generate_cohort(2000, 10, 10, snr_grid=(20,), seed=7)
    P = c.params["train"]
    assert P[:, 0].min() >= D_RANGE[0] and P[:, 0].max() <= D_RANGE[1]
    assert P[:, 1].min() >= DSTAR_RANGE[0] and P[:, 1].max() <= DSTAR_RANGE[1]
    assert P[:, 2].min() >= F_RANGE[0] and P[:, 2].max() <= F_RANGE[1]


def test_snr_drawn_from_grid():
    grid = (10, 25, 80)
    c = generate_cohort(1000, 10, 10, snr_grid=grid, seed=3)
    assert set(np.unique(c.snr["train"]).astype(int)).issubset(set(grid))


def test_b0_signal_near_one_at_high_snr():
    c = generate_cohort(500, 10, 10, snr_grid=(100,), seed=5)
    b0 = c.signals["train"][:, 0]
    assert 0.97 < b0.mean() < 1.03


# --------------------------------------------------------------------------- #
# Noise, nuisance, dispersion, and 2D (b, TE) forward-model extensions
# --------------------------------------------------------------------------- #
def test_triexp_reduces_to_biexp_at_g_zero():
    b = DEFAULT_B_VALUES
    bi = ivim_signal(b, 1.5e-3, 40e-3, 0.2)
    tri = ivim_signal_triexp(b, 1.5e-3, 40e-3, 0.2, 200e-3, 0.0)
    assert np.allclose(bi, tri)


def test_triexp_perturbs_signal_when_g_positive():
    b = DEFAULT_B_VALUES
    bi = ivim_signal(b, 1.5e-3, 40e-3, 0.2)
    tri = ivim_signal_triexp(b, 1.5e-3, 40e-3, 0.2, 200e-3, 0.25)
    assert tri[0] == pytest.approx(bi[0])
    assert np.any(np.abs(tri - bi) > 1e-3)


def test_gaussian_noise_is_unbiased_and_deterministic():
    rng = np.random.default_rng(0)
    clean = np.full(5000, 0.5)
    noisy = add_gaussian_noise(clean, snr=20.0, rng=rng, S0=1.0)
    assert abs(noisy.mean() - 0.5) < 0.01
    assert np.std(noisy) == pytest.approx(0.05, rel=0.1)
    a = add_gaussian_noise(clean, 20.0, np.random.default_rng(1))
    b = add_gaussian_noise(clean, 20.0, np.random.default_rng(1))
    assert np.array_equal(a, b)


def test_rician_floor_lifts_low_signal_above_gaussian():
    rng_r = np.random.default_rng(2)
    rng_g = np.random.default_rng(2)
    clean = np.full(20000, 0.05)
    r = add_rician_noise(clean, 10.0, rng_r).mean()
    g = add_gaussian_noise(clean, 10.0, rng_g).mean()
    assert r > g + 0.01


def test_ncchi_and_nuisance_continuity():
    S = ivim_signal(
        DEFAULT_B_VALUES[None, :],
        np.array([[1.2e-3]]),
        np.array([[30e-3]]),
        np.array([[0.2]]),
    )
    snr = np.array([[30.0]])
    assert np.array_equal(
        add_rician_noise(S, snr, np.random.default_rng(5)),
        add_ncchi_noise(S, snr, np.random.default_rng(5), L=1),
    )
    assert np.array_equal(partial_volume_mix(S, DEFAULT_B_VALUES, 0.0), S)
    assert np.allclose(apply_motion_phase(S[0], 0.0).real, S[0])


def test_dispersion_and_stretched_biexp_limits():
    rng = np.random.default_rng(0)
    D = rng.uniform(0.5e-3, 3e-3, 50)
    mu = rng.uniform(10e-3, 100e-3, 50)
    f = rng.uniform(0.05, 0.40, 50)
    biexp = ivim_signal(DEFAULT_B_VALUES[None, :], D[:, None], mu[:, None], f[:, None])
    disp_gam = ivim_signal_dispersion(
        DEFAULT_B_VALUES[None, :], D[:, None], mu[:, None], np.inf, f[:, None]
    )
    disp_ln = ivim_signal_lognormal_dispersion(
        DEFAULT_B_VALUES[None, :], D[:, None], mu[:, None], 0.0, f[:, None]
    )
    stretch = ivim_signal_stretched(
        DEFAULT_B_VALUES[None, :], D[:, None], mu[:, None], f[:, None], 1.0
    )
    assert np.max(np.abs(disp_gam - biexp)) == 0.0
    assert np.max(np.abs(disp_ln - biexp)) == 0.0
    assert np.max(np.abs(stretch - biexp)) == 0.0
    assert np.allclose(dispersion_dstar_eff(mu), mu)
    sd_disp = dispersion_crlb_mu_batch(
        DEFAULT_B_VALUES, D, mu, np.full(50, 4.0), f, np.full(50, 30.0)
    )
    assert np.all(np.isfinite(sd_disp)) and np.all(sd_disp > 0)


# --------------------------------------------------------------------------- #
# Fisher Information / CRLB & 2D (b, TE) Diffusion-Relaxometry
# --------------------------------------------------------------------------- #
def test_jacobian_matches_finite_difference():
    b = DEFAULT_B_VALUES
    D, Dstar, f, S0 = 1.5e-3, 30e-3, 0.2, 1.0
    J = ivim_jacobian(b, D, Dstar, f, S0)
    base = ivim_signal(b, D, Dstar, f, S0=S0)
    steps = {0: 1e-7, 1: 1e-7, 2: 1e-5, 3: 1e-5}
    args = [D, Dstar, f, S0]
    for k, h in steps.items():
        up = list(args)
        up[k] += h
        fd = (ivim_signal(b, up[0], up[1], up[2], S0=up[3]) - base) / h
        np.testing.assert_allclose(J[:, k], fd, rtol=1e-3, atol=1e-6)


def test_crlb_dstar_grows_with_dstar_and_shrinks_with_snr():
    b = DEFAULT_B_VALUES
    lo = crlb(b, 1.5e-3, 20e-3, 0.2, snr=30)["Dstar"]
    hi = crlb(b, 1.5e-3, 95e-3, 0.2, snr=30)["Dstar"]
    assert hi > lo
    noisy = crlb(b, 1.5e-3, 90e-3, 0.2, snr=10)["Dstar"]
    clean = crlb(b, 1.5e-3, 90e-3, 0.2, snr=100)["Dstar"]
    assert clean < noisy


def test_crlb_batch_and_compute_crlb_api():
    b = DEFAULT_B_VALUES
    D = np.array([1.0e-3, 1.5e-3, 2.0e-3])
    Dstar = np.array([20e-3, 50e-3, 90e-3])
    f = np.array([0.1, 0.2, 0.3])
    snr = np.array([10.0, 30.0, 100.0])
    batch = crlb_dstar_batch(b, D, Dstar, f, snr)
    full_batch = compute_crlb(b, D, Dstar, f, snr=snr)
    np.testing.assert_allclose(full_batch["Dstar"], batch, rtol=1e-6)
    for i in range(3):
        scalar = compute_crlb(b, D[i], Dstar[i], f[i], snr=snr[i])
        assert batch[i] == pytest.approx(scalar["Dstar"], rel=1e-6)
        assert full_batch["D"][i] == pytest.approx(scalar["D"], rel=1e-6)
        assert full_batch["f"][i] == pytest.approx(scalar["f"], rel=1e-6)


def test_design_crlb_and_optimal_scheme():
    rng = np.random.default_rng(0)
    n = 200
    D = rng.uniform(0.8e-3, 2.0e-3, n)
    Dstar = rng.uniform(10e-3, 100e-3, n)
    f = rng.uniform(0.1, 0.3, n)
    snr = np.full(n, 30.0)
    lowb = np.array([0, 5, 10, 20, 40, 60, 80, 100, 200, 800], float)
    highb = np.array([0, 100, 200, 300, 400, 500, 600, 700, 750, 800], float)
    _, hi_low, _ = design_crlb_dstar(lowb, D, Dstar, f, snr)
    _, hi_high, _ = design_crlb_dstar(highb, D, Dstar, f, snr)
    assert hi_low < hi_high

    b_opt = crlb_optimal_scheme(n_b=11, seed=0)
    assert len(b_opt) == 11 and b_opt[0] == 0.0 and np.all(np.diff(b_opt) > 0)
    assert (b_opt < 100).sum() >= 6


def test_ivim_2d_reduces_to_1d_and_reduces_high_dstar_crlb():
    b = CLINICAL_B
    te_zero = np.zeros_like(b)
    s1 = ivim_signal(b, 1.2e-3, 80e-3, 0.2, S0=1.0)
    s2 = ivim_signal_2d(b, te_zero, 1.2e-3, 80e-3, 0.2, S0=1.0)
    assert np.allclose(s1, s2)
    j1 = ivim_jacobian(b, 1.2e-3, 80e-3, 0.2, S0=1.0)
    j2 = ivim_jacobian_2d(b, te_zero, 1.2e-3, 80e-3, 0.2, S0=1.0)
    assert np.allclose(j1, j2)

    te = np.full_like(b, 60.0)
    J_ana = ivim_jacobian_2d(
        b, te, 1.2e-3, 80e-3, 0.2, S0=1.1, T2_tissue=50.0, T2_blood=150.0
    )
    eps = 1e-7
    p0 = [1.2e-3, 80e-3, 0.2, 1.1]
    for col in range(4):
        pp, pm = list(p0), list(p0)
        pp[col] += eps
        pm[col] -= eps
        fd = (
            ivim_signal_2d(b, te, *pp, T2_tissue=50.0, T2_blood=150.0)
            - ivim_signal_2d(b, te, *pm, T2_tissue=50.0, T2_blood=150.0)
        ) / (2 * eps)
        assert np.allclose(J_ana[:, col], fd, rtol=1e-4, atol=1e-6)

    c_single = crlb_2d(b, te, 1.2e-3, 80e-3, 0.2, S0=1.0, snr=30.0)
    c_batch = crlb_dstar_2d_batch(
        b,
        te,
        np.array([1.2e-3]),
        np.array([80e-3]),
        np.array([0.2]),
        np.array([30.0]),
    )
    assert c_batch[0] == pytest.approx(c_single["Dstar"], rel=1e-6)

    b22 = np.tile(CLINICAL_B, 2)
    te22 = np.repeat(np.array([45.0, 90.0]), len(CLINICAL_B))
    sd_1d = crlb_dstar_batch(
        DEFAULT_B_VALUES,
        np.array([1.2e-3]),
        np.array([85e-3]),
        np.array([0.2]),
        np.array([30.0]),
    )[0]
    sd_2d = crlb_dstar_2d_batch(
        b22,
        te22,
        np.array([1.2e-3]),
        np.array([85e-3]),
        np.array([0.2]),
        np.array([30.0]),
    )[0]
    assert sd_2d < 0.70 * sd_1d


def test_simulate_ivim_signal_scalar_and_batch():
    clean = simulate_ivim_signal(DEFAULT_B_VALUES, D=1.2e-3, Dstar=35e-3, f=0.2)
    assert clean.shape == (DEFAULT_B_VALUES.size,)
    assert clean[0] == pytest.approx(1.0)

    noisy_batch = simulate_ivim_signal(
        DEFAULT_B_VALUES,
        D=np.full(25, 1.2e-3),
        Dstar=np.full(25, 35e-3),
        f=np.full(25, 0.2),
        snr=40.0,
        seed=123,
    )
    assert noisy_batch.shape == (25, DEFAULT_B_VALUES.size)
    assert np.all(noisy_batch >= 0.0)
