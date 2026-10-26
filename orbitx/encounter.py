"""Encounter-plane (B-plane) geometry for short, high-speed conjunctions.

Sign convention (same as the ESA Kelvins dataset):
    miss vector  m = (debris position) - (satellite position)  at the time of closest approach (TCA),
    expressed in the satellite's RTN frame.
If a burn displaces the satellite by dr, the new miss vector is  m - dr.

Because the relative speed is ~10 km/s, the encounter is a straight-line fly-by: only the miss
component perpendicular to the relative velocity matters. That is the B-plane.
"""
from __future__ import annotations

import numpy as np


def bplane_basis(v_rel) -> np.ndarray:
    """3x2 matrix E whose orthonormal columns span the plane perpendicular to v_rel."""
    v = np.asarray(v_rel, dtype=float)
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("relative velocity must be a finite non-zero vector")
    vh = v / norm
    ref = np.eye(3)[int(np.argmin(np.abs(vh)))]      # axis least aligned with v_rel
    e1 = np.cross(vh, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(vh, e1)
    return np.stack([e1, e2], axis=1)


def project_vector(vec3, basis: np.ndarray) -> np.ndarray:
    """Project a 3-vector onto the B-plane -> 2-vector."""
    return basis.T @ np.asarray(vec3, dtype=float)


def project_covariance(cov3, basis: np.ndarray) -> np.ndarray:
    """Project a 3x3 position covariance onto the B-plane -> 2x2."""
    return basis.T @ np.asarray(cov3, dtype=float) @ basis


def rtn_covariance(sigma_rtn) -> np.ndarray:
    """Diagonal 3x3 covariance from per-axis 1-sigma values [m] (cross-correlations ignored)."""
    s = np.asarray(sigma_rtn, dtype=float)
    return np.diag(s * s)


def bplane_miss(miss_rtn, basis: np.ndarray, delta_r_sat=None) -> np.ndarray:
    """2-D B-plane miss after the satellite is displaced by delta_r_sat (RTN, metres)."""
    m = np.asarray(miss_rtn, dtype=float)
    if delta_r_sat is not None:
        m = m - np.asarray(delta_r_sat, dtype=float)
    return basis.T @ m
