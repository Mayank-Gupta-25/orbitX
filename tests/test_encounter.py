import numpy as np
import pytest

from orbitx.encounter import (
    bplane_basis,
    bplane_miss,
    project_covariance,
    project_vector,
    rtn_covariance,
)


@pytest.mark.parametrize("v", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [3e3, -7e3, 5e3], [1e-3, 1e-3, 9e3]])
def test_basis_orthonormal_and_perpendicular(v):
    E = bplane_basis(v)
    assert E.shape == (3, 2)
    assert np.allclose(E.T @ E, np.eye(2), atol=1e-12)
    assert np.allclose(E.T @ (np.asarray(v, float) / np.linalg.norm(v)), 0.0, atol=1e-12)


def test_basis_rejects_zero_velocity():
    with pytest.raises(ValueError):
        bplane_basis([0, 0, 0])


def test_projection_removes_velocity_component():
    v = np.array([0.0, 0.0, 10e3])
    E = bplane_basis(v)
    m2 = project_vector([30.0, 40.0, 5000.0], E)
    assert np.linalg.norm(m2) == pytest.approx(50.0)          # along-v part (5000 m) is discarded


def test_covariance_projection_is_symmetric_psd():
    E = bplane_basis([4e3, 5e3, -9e3])
    C2 = project_covariance(rtn_covariance([100.0, 800.0, 50.0]), E)
    assert np.allclose(C2, C2.T)
    assert np.all(np.linalg.eigvalsh(C2) > 0)


def test_burn_sign_convention():
    """Moving the satellite directly away from the debris increases the miss distance."""
    v = np.array([0.0, 0.0, 10e3])
    E = bplane_basis(v)
    miss = np.array([30.0, 40.0, 0.0])                          # debris minus satellite
    dr_away = -0.5 * miss                                       # satellite moves opposite to the debris
    assert np.linalg.norm(bplane_miss(miss, E, dr_away)) == pytest.approx(75.0)
    dr_toward = +0.5 * miss
    assert np.linalg.norm(bplane_miss(miss, E, dr_toward)) == pytest.approx(25.0)
