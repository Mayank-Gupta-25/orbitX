import numpy as np
import pytest

from orbitx.collision import log10_pc, pc_exact, pc_fast, pc_from_geometry
from orbitx.constants import PC_FLOOR


def _cov(s1, s2, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return R @ np.diag([s1 ** 2, s2 ** 2]) @ R.T


def test_centered_isotropic_matches_analytic():
    sigma, hbr = 100.0, 20.0
    expected = 1.0 - np.exp(-hbr ** 2 / (2 * sigma ** 2))        # exact closed form for m = 0
    C = np.diag([sigma ** 2, sigma ** 2])
    assert pc_exact([0, 0], C, hbr) == pytest.approx(expected, rel=1e-6)
    assert pc_fast([0, 0], C, hbr) == pytest.approx(expected, rel=0.02)


def test_small_hbr_limit():
    sigma, hbr = 500.0, 5.0
    C = np.diag([sigma ** 2, sigma ** 2])
    assert pc_fast([0, 0], C, hbr) == pytest.approx(hbr ** 2 / (2 * sigma ** 2), rel=0.01)


def test_fast_vs_exact_random_cases():
    """pc_fast must agree with the exact integral to <0.03 decades when hbr <= 0.5*sigma_min."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(400):
        s1 = 10 ** rng.uniform(1.5, 3.5)                        # 30 m .. 3 km
        s2 = s1 * rng.uniform(0.2, 1.0)
        C = _cov(s1, s2, rng.uniform(0, np.pi))
        hbr = rng.uniform(0.02, 0.5) * s2
        phi, k = rng.uniform(0, 2 * np.pi), rng.uniform(0, 5)
        m = k * s2 * np.array([np.cos(phi), np.sin(phi)])
        worst = max(worst, abs(log10_pc(pc_fast(m, C, hbr)) - log10_pc(pc_exact(m, C, hbr))))
    assert worst < 0.03


def test_rotation_invariance():
    rng = np.random.default_rng(3)
    C = _cov(300.0, 80.0, 0.7)
    m = np.array([120.0, -60.0])
    base = pc_fast(m, C, 10.0)
    for _ in range(10):
        a = rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        assert pc_fast(R @ m, R @ C @ R.T, 10.0) == pytest.approx(base, rel=1e-9)


def test_pc_decreases_with_miss_distance():
    C = np.diag([200.0 ** 2, 200.0 ** 2])
    pcs = [pc_fast([d, 0.0], C, 10.0) for d in (0, 100, 200, 400, 800, 1600)]
    assert all(a > b for a, b in zip(pcs, pcs[1:]))


def test_extreme_tail_is_floored_not_nan():
    C = np.diag([10.0 ** 2, 10.0 ** 2])
    pc = pc_fast([5000.0, 0.0], C, 10.0)
    assert np.isfinite(pc) and pc == PC_FLOOR
    assert log10_pc(pc) == pytest.approx(-30.0)


def test_degenerate_covariance_does_not_produce_nan():
    pc = pc_fast([0.0, 0.0], np.zeros((2, 2)), 10.0)
    assert np.isfinite(pc) and 0.0 <= pc <= 1.0


def test_pc_from_geometry_consistent_with_burn():
    miss = [30.0, 0.0, 0.0]
    vrel = [0.0, 0.0, 10e3]
    sigma = [100.0, 100.0, 100.0]
    base = pc_from_geometry(miss, vrel, sigma, 10.0)
    after = pc_from_geometry(miss, vrel, sigma, 10.0, delta_r_sat=[-200.0, 0.0, 0.0])
    assert after < base
