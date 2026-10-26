"""Collision probability (Pc) for a short-term 2-D encounter (Foster-style).

Pc = P( relative position in the B-plane falls inside the hard-body disc of radius hbr )
with the relative position ~ N(miss, C2).

* pc_fast  : equal-area-square approximation, closed form (erfc, numerically safe for tiny Pc).
* pc_exact : accurate reference (1-D adaptive quadrature of the exact disc integral), for tests.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import erfc

from .constants import PC_FLOOR
from .encounter import bplane_basis, project_covariance, project_vector, rtn_covariance

_SQRT2 = float(np.sqrt(2.0))
_MIN_SIGMA = 1e-6        # [m] guards against degenerate (zero) covariances


def _principal_axes(miss2, cov2):
    """Rotate into the covariance principal axes. Returns (m1, m2, sigma1, sigma2)."""
    miss2 = np.asarray(miss2, dtype=float)
    cov2 = np.asarray(cov2, dtype=float)
    a, b, d = cov2[0, 0], 0.5 * (cov2[0, 1] + cov2[1, 0]), cov2[1, 1]
    half_sum = 0.5 * (a + d)
    radius = np.hypot(0.5 * (a - d), b)
    lam1 = max(half_sum + radius, _MIN_SIGMA ** 2)
    lam2 = max(half_sum - radius, _MIN_SIGMA ** 2)
    theta = 0.5 * np.arctan2(2.0 * b, a - d)
    c, s = np.cos(theta), np.sin(theta)
    m1 = c * miss2[0] + s * miss2[1]
    m2 = -s * miss2[0] + c * miss2[1]
    return float(m1), float(m2), float(np.sqrt(lam1)), float(np.sqrt(lam2))


def _strip_prob(m: float, half_width: float, sigma: float) -> float:
    """P(|X| <= half_width) for X ~ N(m, sigma^2), computed with erfc (no cancellation in tails)."""
    m = abs(m)
    k = 1.0 / (_SQRT2 * sigma)
    return 0.5 * (erfc((m - half_width) * k) - erfc((m + half_width) * k))


def pc_fast(miss2, cov2, hbr: float) -> float:
    """Equal-area-square Pc. miss2: (2,) B-plane miss [m]; cov2: (2,2) [m^2]; hbr: radius [m]."""
    m1, m2, s1, s2 = _principal_axes(miss2, cov2)
    half = 0.5 * np.sqrt(np.pi) * hbr             # square with the same area as the hbr-disc
    pc = _strip_prob(m1, half, s1) * _strip_prob(m2, half, s2)
    return float(max(pc, PC_FLOOR))


def pc_exact(miss2, cov2, hbr: float) -> float:
    """Reference Pc: exact integral of the Gaussian over the hbr-disc (1-D adaptive quadrature)."""
    m1, m2, s1, s2 = _principal_axes(miss2, cov2)
    norm = 1.0 / (s1 * np.sqrt(2.0 * np.pi))

    def integrand(x: float) -> float:
        h = np.sqrt(max(hbr * hbr - x * x, 0.0))
        return norm * np.exp(-0.5 * ((x - m1) / s1) ** 2) * _strip_prob(m2, h, s2)

    pts = [m1] if abs(m1) < hbr else None
    val, _ = quad(integrand, -hbr, hbr, points=pts, epsabs=0.0, epsrel=1e-10, limit=400)
    return float(max(val, PC_FLOOR))


def log10_pc(pc: float) -> float:
    """log10 of a (floored) probability."""
    return float(np.log10(max(pc, PC_FLOOR)))


def pc_from_geometry(miss_rtn, vrel_rtn, sigma_rtn, hbr: float, delta_r_sat=None) -> float:
    """Convenience: Pc from 3-D RTN geometry (miss, relative velocity, per-axis sigma)."""
    basis = bplane_basis(vrel_rtn)
    m = np.asarray(miss_rtn, dtype=float)
    if delta_r_sat is not None:
        m = m - np.asarray(delta_r_sat, dtype=float)
    return pc_fast(project_vector(m, basis), project_covariance(rtn_covariance(sigma_rtn), basis), hbr)
