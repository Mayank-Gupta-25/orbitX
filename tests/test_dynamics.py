import numpy as np
import pytest
from scipy.integrate import solve_ivp

from orbitx.constants import MU_EARTH, R_EARTH
from orbitx.dynamics import (
    apply_impulse,
    burn_displacement,
    cw_ode,
    cw_rv,
    cw_stm,
    mean_motion,
    propagate,
)

N = mean_motion(550e3)


def test_mean_motion_period():
    period = 2 * np.pi / N
    assert 5700.0 < period < 5800.0          # ~95.7 min at 550 km


def test_stm_identity_at_zero():
    assert np.allclose(cw_stm(0.0, N), np.eye(6), atol=1e-12)


def test_stm_composition():
    a, b = 1234.5, 4321.0
    assert np.allclose(cw_stm(a + b, N), cw_stm(b, N) @ cw_stm(a, N), atol=1e-9)


def test_rv_block_matches_stm():
    assert np.allclose(cw_rv(777.0, N), cw_stm(777.0, N)[:3, 3:], atol=1e-12)


def test_stm_matches_ode_integration():
    rng = np.random.default_rng(1)
    s0 = np.concatenate([rng.normal(0, 500, 3), rng.normal(0, 0.5, 3)])
    tau = 1.5 * 2 * np.pi / N
    sol = solve_ivp(lambda t, s: cw_ode(t, s, N), (0, tau), s0, method="DOP853", rtol=1e-12, atol=1e-12)
    assert np.allclose(propagate(s0, tau, N), sol.y[:, -1], rtol=1e-7, atol=1e-6)


def test_along_track_burn_secular_drift():
    """0.01 m/s along-track burn one day earlier shifts the satellite ~ -3*dv*tau along-track."""
    dv, tau = 0.01, 86400.0
    dr = burn_displacement([0.0, dv, 0.0], tau, N)
    assert dr[1] == pytest.approx(-3.0 * dv * tau, rel=0.03)      # ~ -2.6 km
    one_hour = burn_displacement([0.0, dv, 0.0], 3600.0, N)
    assert abs(one_hour[1]) < 0.06 * abs(dr[1])                    # same burn, 1 h before: ~0.1 km


def test_apply_impulse_only_changes_velocity():
    s = apply_impulse(np.zeros(6), [0.0, 0.01, 0.0])
    assert np.allclose(s, [0, 0, 0, 0, 0.01, 0])


def _two_body(_t, y):
    r = y[:3]
    return np.concatenate([y[3:], -MU_EARTH * r / np.linalg.norm(r) ** 3])


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_cw_agrees_with_full_two_body(axis):
    """CW displacement vs full two-body truth for a 1 cm/s burn over one day (<1% error)."""
    a = R_EARTH + 550e3
    v_circ = np.sqrt(MU_EARTH / a)
    y_ref = np.array([a, 0.0, 0.0, 0.0, v_circ, 0.0])        # RTN axes == inertial axes at t=0
    dv = np.zeros(3)
    dv[axis] = 0.01
    y_pert = y_ref.copy()
    y_pert[3:] += dv
    tau = 86400.0
    kw = dict(method="DOP853", rtol=1e-12, atol=1e-9)
    ref = solve_ivp(_two_body, (0, tau), y_ref, **kw).y[:, -1]
    pert = solve_ivp(_two_body, (0, tau), y_pert, **kw).y[:, -1]
    # the reference orbit is an equatorial circle -> RTN at time tau:
    r_hat = ref[:3] / np.linalg.norm(ref[:3])
    n_hat = np.cross(ref[:3], ref[3:])
    n_hat /= np.linalg.norm(n_hat)
    t_hat = np.cross(n_hat, r_hat)
    rel = pert[:3] - ref[:3]
    truth = np.array([rel @ r_hat, rel @ t_hat, rel @ n_hat])
    pred = burn_displacement(dv, tau, mean_motion(550e3))
    assert np.linalg.norm(pred - truth) / np.linalg.norm(truth) < 0.01
