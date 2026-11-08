"""Tests of Floquet stability, trap depth, anharmonicity, and micromotion (tiopt.stability)."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import tiopt
from tiopt import stability, voltages

OMEGA_RF = 2 * np.pi * 40e6


@pytest.fixture(scope="module")
def linear():
    ion = tiopt.Ion("40Ca+")
    trap = tiopt.Trap.from_preset(7, 50.0)
    trap.solve("analytical")
    null = trap.find_rf_null((0.0, 0.0, None))
    waveform = tiopt.Waveform.transport(trap, ion, null, null, 0.0, 1)
    return trap, ion, null, waveform.voltages[0]


def test_floquet_matches_an_independent_integration():
    q = 0.3

    def mathieu(t, y):
        return [y[1], 2 * q * np.cos(2 * t) * y[0]]

    M = np.column_stack([solve_ivp(mathieu, (0, np.pi), y0, method="DOP853", rtol=1e-12, atol=1e-14).y[:, -1] for y0 in ([1, 0], [0, 1])])
    beta = np.arccos(np.trace(M) / 2) / np.pi
    result = stability.floquet(np.zeros((3, 3)), np.diag([q, -q, 0.0]), OMEGA_RF)
    assert result.stable
    assert result.frequencies.max() == pytest.approx(beta * OMEGA_RF / 2, rel=1e-8)
    # The first stability region ends at q = 0.908
    assert stability.floquet(np.zeros((3, 3)), np.diag([0.9, -0.9, 0.0]), OMEGA_RF).stable
    assert not stability.floquet(np.zeros((3, 3)), np.diag([0.95, -0.95, 0.0]), OMEGA_RF).stable


def test_floquet_frequencies_in_a_trap(linear):
    trap, ion, null, dc = linear
    result = stability.floquet_at(trap, ion, null, dc)
    pseudo, _ = trap.normal_modes(ion, null, dc)
    assert result.stable
    # The axial (DC) mode is unchanged, and the radial modes are a little stiffer than the pseudopotential says
    assert np.sort(result.frequencies)[0] == pytest.approx(pseudo[0], rel=1e-3)
    assert 1.0 < np.sort(result.frequencies)[1] / np.sort(pseudo)[1] < 1.05
    diagram = stability.stability_diagram(trap, ion, null, dc, n_points=21, n_steps=200)
    assert diagram.stable[np.argmin(np.abs(diagram.dc_scales - 1.0)), np.argmin(np.abs(diagram.V_RF - trap.V_RF))]
    assert not diagram.stable.all()


def test_escape_points(linear):
    trap, ion, null, dc = linear
    # Without DC voltages the ion escapes upwards through the saddle above the RF null, as compute_trap_depth finds
    pseudo = stability.escape_point(trap, ion, null)
    assert pseudo.refined and not pseudo.lower_bound
    assert pseudo.depth == pytest.approx(trap.trap_depth(ion, null), rel=1e-3)
    assert pseudo.saddle[2] > 1.5 * null[2]
    # The DC voltages of the well move the saddle off to the side
    well = stability.escape_point(trap, ion, null, dc)
    assert well.refined
    assert abs(well.saddle[1]) > 10
    assert well.depth > pseudo.depth


def test_anharmonicity_and_micromotion(linear):
    trap, ion, null, dc = linear
    anharmonic = stability.anharmonicity(trap, ion, null, dc)
    assert anharmonic.frequency == pytest.approx(2 * np.pi * 1e6, rel=1e-3)
    assert anharmonic.quartic_length > 50
    amplitude, index = stability.micromotion(trap, ion, null, [0.0, 0.0, 1e7])
    assert np.linalg.norm(amplitude) < 1e-12 and index < 1e-4
    amplitude, index = stability.micromotion(trap, ion, null + [0.0, 0.0, 1.0], [0.0, 0.0, 1e7])
    assert np.linalg.norm(amplitude) > 1e-9 and index > 0.01


def test_tilt_and_report(linear):
    trap, ion, null, dc = linear
    tilted, info = voltages.static_well(trap, ion, null, 2 * np.pi * 1e6, tilt=15.0)
    assert info.converged
    _, _, tilt = stability.principal_axes(trap, ion, null, tilted)
    assert tilt == pytest.approx(15.0, abs=1e-3)
    lines = stability.report(trap, ion, null, dc, n_ions=2)
    assert any(line.startswith("Floquet frequencies") for line in lines)
    assert any(line.startswith("Depth") for line in lines)
