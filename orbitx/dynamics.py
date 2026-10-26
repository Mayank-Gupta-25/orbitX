"""Clohessy-Wiltshire (Hill) dynamics for the satellite's deviation from its nominal orbital slot.

Frame: RTN / LVLH of the reference circular orbit
    x = Radial (outward), y = Transverse (along-track), z = Normal (cross-track)
State: [x, y, z, vx, vy, vz]  (metres, metres/second).

IMPORTANT: CW is a linearisation valid only for SMALL deviations (metres .. a few km) from the
reference orbit. It is used here ONLY for the satellite's own deviation. Debris is never
propagated with CW (its geometry comes from real data, see encounter.py / generator.py).
"""
from __future__ import annotations

import numpy as np

from .constants import MU_EARTH, R_EARTH


def mean_motion(altitude_m: float) -> float:
    """Mean motion n = sqrt(mu / a^3) [rad/s] of a circular orbit at the given altitude."""
    a = R_EARTH + float(altitude_m)
    return float(np.sqrt(MU_EARTH / a ** 3))


def cw_stm(tau: float, n: float) -> np.ndarray:
    """6x6 closed-form CW state-transition matrix Phi(tau): state(t+tau) = Phi @ state(t)."""
    nt = n * tau
    s, c = np.sin(nt), np.cos(nt)
    phi = np.zeros((6, 6))
    phi[0] = [4.0 - 3.0 * c, 0.0, 0.0, s / n, 2.0 * (1.0 - c) / n, 0.0]
    phi[1] = [6.0 * (s - nt), 1.0, 0.0, -2.0 * (1.0 - c) / n, (4.0 * s - 3.0 * nt) / n, 0.0]
    phi[2] = [0.0, 0.0, c, 0.0, 0.0, s / n]
    phi[3] = [3.0 * n * s, 0.0, 0.0, c, 2.0 * s, 0.0]
    phi[4] = [6.0 * n * (c - 1.0), 0.0, 0.0, -2.0 * s, 4.0 * c - 3.0, 0.0]
    phi[5] = [0.0, 0.0, -n * s, 0.0, 0.0, c]
    return phi


def cw_rv(tau: float, n: float) -> np.ndarray:
    """3x3 block Phi_rv(tau): position change after tau seconds caused by a unit velocity change."""
    nt = n * tau
    s, c = np.sin(nt), np.cos(nt)
    return np.array(
        [
            [s / n, 2.0 * (1.0 - c) / n, 0.0],
            [-2.0 * (1.0 - c) / n, (4.0 * s - 3.0 * nt) / n, 0.0],
            [0.0, 0.0, s / n],
        ]
    )


def propagate(state, tau: float, n: float) -> np.ndarray:
    """Propagate a 6-vector deviation state forward by tau seconds."""
    return cw_stm(tau, n) @ np.asarray(state, dtype=float)


def apply_impulse(state, dv) -> np.ndarray:
    """Instantaneous velocity change dv (3-vector, m/s). Returns a new state."""
    out = np.array(state, dtype=float, copy=True)
    out[3:] += np.asarray(dv, dtype=float)
    return out


def burn_displacement(dv, tau: float, n: float) -> np.ndarray:
    """Position displacement [m] at time tau after an impulsive burn dv [m/s] applied now."""
    return cw_rv(tau, n) @ np.asarray(dv, dtype=float)


def cw_ode(_t: float, s: np.ndarray, n: float) -> np.ndarray:
    """CW ordinary differential equations (reference implementation used by the tests)."""
    x, y, z, vx, vy, vz = s
    return np.array([vx, vy, vz, 3.0 * n * n * x + 2.0 * n * vy, -2.0 * n * vx, -n * n * z])
