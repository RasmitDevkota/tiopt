"""Tests of total confinement and CTC paths (tiopt.confinement)."""

import numpy as np
import pytest

import tiopt
from tiopt import confinement


@pytest.fixture(scope="module")
def x_junction():
    trap = tiopt.Trap.junction((0, 90, 180, 270), 2, 40.0)
    trap.V_RF = 40.0
    trap.Omega_RF = 2 * np.pi * 40e6
    return trap


def test_total_confinement_matches_the_normal_modes(solved_trap, ion):
    solved_trap.V_RF = 40.0
    solved_trap.Omega_RF = 2 * np.pi * 40e6
    null = solved_trap.find_rf_null((0.0, 0.0, None))
    omega, _ = solved_trap.normal_modes(ion, null)
    field = confinement.RFField(solved_trap)
    assert field.total_confinement(ion, null) == pytest.approx(np.sum(omega ** 2), rel=1e-3)
    assert field.null_across([[10.0, 3.0, 40.0]], [[1.0, 0.0, 0.0]])[0] == pytest.approx(solved_trap.find_rf_null((10.0, 0.0, None)), abs=1e-3)


def test_arms_of_traps(x_junction):
    linear = tiopt.Trap.from_preset(7, 50.0)
    linear.V_RF, linear.Omega_RF = 40.0, 2 * np.pi * 40e6
    assert [round(angle) % 360 for angle, _ in confinement.arm_directions(linear)] == [0, 180]
    assert sorted(round(angle) % 360 for angle, _ in confinement.arm_directions(x_junction)) == [0, 90, 180, 270]
    y_junction = tiopt.Trap.junction((0, 120, 240), 2, 40.0)
    y_junction.V_RF, y_junction.Omega_RF = 40.0, 2 * np.pi * 40e6
    assert sorted(round(angle) % 360 for angle, _ in confinement.arm_directions(y_junction)) == [0, 120, 240]


def test_ctc_path_through_a_junction(x_junction, ion):
    path = confinement.confinement_path(x_junction, ion, confinement.default_waypoints(x_junction), 81)
    # The RF null rises at the center, where the total confinement drops to about a third (an eighth of the sum of the
    # squared secular frequencies), with a pseudopotential barrier in each arm
    assert path.null_positions[:, 2].max() > 80.0
    assert 0.25 < path.confinement_dip < 0.45
    assert 5.0 < path.barrier * 1e3 < 10.0
    # The CTC path runs beneath the null wherever it must and keeps the confinement of the arms
    assert np.all(path.ctc_positions[:, 2] <= path.null_positions[:, 2] + 1e-9)
    assert 20.0 < path.depth < 60.0
    assert path.ctc_confinement.min() == pytest.approx(path.target, rel=1e-3)
    assert np.allclose(path.ctc_positions[:, :2], path.null_positions[:, :2])


def test_linear_trap_needs_no_ctc_path(solved_trap, ion):
    path = confinement.confinement_path(solved_trap, ion, [[-60.0, 0.0], [60.0, 0.0]], 31)
    assert path.barrier < 1e-6
    assert path.depth < 1.0


def test_paths_follow_curved_sections(repository, ion):
    from tiopt import builder

    trap = builder.build_trap(builder.load_design(repository / "examples" / "designs" / "racetrack.json"))
    path = confinement.confinement_path(trap, ion, confinement.default_waypoints(trap), 61)
    # The path goes around the turns along the RF null rather than cutting across them
    assert path.s[-1] > 900.0
    assert np.ptp(path.null_positions[:, 2]) < 3.0
    assert path.barrier < 1e-4
    assert path.depth < 0.01
