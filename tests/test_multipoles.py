"""Tests of multipole expansions and control voltages (tiopt.multipoles)."""

import numpy as np
import pytest

import tiopt
from tiopt import multipoles


@pytest.fixture(scope="module")
def setup():
    ion = tiopt.Ion("40Ca+")
    trap = tiopt.Trap.from_preset(7, 50.0)
    trap.solve("analytical")
    return trap, ion, trap.find_rf_null((0.0, 0.0, None))


def test_expansions_reproduce_the_potential(setup):
    trap, ion, null = setup
    control = multipoles.control_matrix(trap, null, order=4)
    assert control.names[:9] == list(multipoles.BASIC) and len(control.names) == 9 + 7 + 9
    e = 6
    voltages = trap.zero_voltages()
    voltages[e] = 1.0
    coefficients = control.matrix[:, e]
    for u in ([2.0, -1.0, 1.5], [-3.0, 2.0, -2.0]):
        x, y, z = u
        basic = (coefficients[0] - coefficients[1] * x - coefficients[2] * y - coefficients[3] * z + coefficients[4] * (x * x - y * y)
                 + coefficients[5] * (2 * z * z - x * x - y * y) + coefficients[6] * x * y + coefficients[7] * y * z + coefficients[8] * x * z)
        higher = multipoles.real_solid_harmonics(4, np.array([u])) @ coefficients[9:]
        assert basic + higher[0] == pytest.approx(trap.potential(null + np.array(u), voltages), rel=1e-6)
    H = multipoles.hessian_from_quadrupoles(**multipoles.quadrupoles_from_hessian(np.array([[1.0, 0.2, 0.1], [0.2, -0.4, 0.3], [0.1, 0.3, -0.6]])))
    assert H == pytest.approx(np.array([[1.0, 0.2, 0.1], [0.2, -0.4, 0.3], [0.1, 0.3, -0.6]]))


def test_shims_and_control_voltages(setup, tmp_path):
    trap, ion, null = setup
    control = multipoles.control_matrix(trap, null)
    shims = multipoles.shim_matrix(control)
    assert shims[: trap.n_rf] == pytest.approx(0.0)
    assert control.rows(multipoles.SHIMS) @ shims == pytest.approx(np.eye(8), abs=1e-9)
    targets = multipoles.axial_quadrupoles(ion, 2 * np.pi * 1e6, "x")
    targets.update(Ex=0.0, Ey=0.0, Ez=0.0)
    voltages, achieved = multipoles.solve_voltages(control, targets, max_voltage=10.0)
    assert np.abs(voltages).max() <= 10.0 + 1e-9
    omega, _ = trap.normal_modes(ion, null, voltages)
    assert omega[0] == pytest.approx(2 * np.pi * 1e6, rel=1e-3)
    path = tmp_path / "shims.txt"
    multipoles.save_shim_matrix(path, shims)
    loaded, names = multipoles.load_shim_matrix(path)
    assert names == list(multipoles.SHIMS)
    assert loaded == pytest.approx(shims, rel=1e-8)
