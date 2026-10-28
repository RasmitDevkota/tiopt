"""Tests of the co-design flow and of acceleration through the Python bindings."""

import numpy as np
import pytest

import tiopt
from conftest import REPOSITORY


@pytest.fixture(scope="module")
def codesign():
    codesign = tiopt.Codesign(REPOSITORY / "examples" / "circuits" / "bell.qasm")
    yield codesign
    codesign.close()


def test_basic_design(codesign):
    layout = codesign.layout()
    assert (layout["n_zones"], layout["n_layers"], layout["n_ions"]) == (5, 3, 2)
    # The gate and measurement zones are RF nulls along the trap axis
    assert layout["gate_position"][2] == pytest.approx(tiopt.ION_HEIGHT, abs=1.5)
    assert layout["measure_position"][0] > layout["gate_position"][0]
    metrics = codesign.metrics
    assert 0.5 < metrics["circuit_fidelity"] < 1
    assert metrics["objective"] == pytest.approx(-np.log(metrics["circuit_fidelity"]))
    records = codesign.records()
    assert records[0].stage == "basic design"


def test_parameters_round_trip(codesign):
    parameters = codesign.parameters
    assert parameters == tiopt.default_codesign_parameters()
    with pytest.raises(KeyError):
        codesign.parameters = {"not_a_parameter": 1.0}


def test_environment_changes_metrics(codesign):
    environment = codesign.environment
    noisy = codesign.evaluate()["heating_rate"]
    codesign.environment = {"field_noise": environment["field_noise"] / 100}
    quiet = codesign.evaluate()["heating_rate"]
    codesign.environment = environment
    assert quiet < noisy / 10


def test_qc_aware_stage_does_not_worsen_design(codesign):
    before = codesign.evaluate()["objective"]
    objective, n_evaluations = codesign.optimize_stage("transport", qc_aware=True, max_evaluations=4)
    assert n_evaluations <= 4
    assert codesign.metrics["objective"] <= before + 1e-12


def test_codesign_trap_is_usable(codesign, ion):
    trap = codesign.trap()
    assert len(trap.grating_couplers()) == 4 and len(trap.detectors()) == 1
    null = trap.find_rf_null((codesign.layout()["gate_position"][0], 0.0, None))
    assert trap.trap_depth(ion, null) > 0


def test_threads():
    n_threads = tiopt.get_num_threads()
    tiopt.set_num_threads(2)
    assert tiopt.get_num_threads() == 2
    tiopt.set_num_threads(n_threads)
    assert tiopt.get_num_threads() == n_threads


def test_device_selection():
    assert tiopt.get_device() == "cpu"
    with pytest.raises(ValueError):
        tiopt.set_device("tpu")
    if not tiopt.opencl_available():
        with pytest.raises(tiopt.TiOptError):
            tiopt.set_device("opencl")
        assert tiopt.get_device() == "cpu"


def test_ensemble_matches_trajectories(solved_trap, ion):
    waveform = tiopt.Waveform.transport(solved_trap, ion, (0.0, 0.0, tiopt.ION_HEIGHT), (0.0, 0.0, tiopt.ION_HEIGHT), 0.0, 1)
    rng = np.random.default_rng(1)
    positions = np.array([0.0, 0.0, tiopt.ION_HEIGHT]) + 0.2 * rng.standard_normal((4, 3))
    velocities = 1e4 * rng.standard_normal((4, 3))
    n_steps, dt = 300, 5e-9
    final_positions, final_velocities = tiopt.simulate_ensemble(solved_trap, ion, waveform, positions, velocities, n_steps, dt)
    for p in range(4):
        trajectory = tiopt.simulate_trajectory(solved_trap, ion, waveform, n_steps * dt, dt, positions[p], velocities[p])
        assert final_positions[p] == pytest.approx(trajectory.positions[-1], abs=1e-12)
        assert final_velocities[p] == pytest.approx(trajectory.velocities[-1], rel=1e-12, abs=1e-9)


@pytest.mark.skipif(not tiopt.opencl_available(), reason="no OpenCL device with double precision")
def test_opencl_potential_map_matches_cpu(solved_trap):
    positions = np.column_stack([np.linspace(-50, 50, 200), np.zeros(200), np.full(200, tiopt.ION_HEIGHT)])
    voltages = np.linspace(-1, 1, solved_trap.n_electrodes)
    cpu = solved_trap.potential(positions, voltages)
    tiopt.set_device("opencl")
    try:
        gpu = solved_trap.potential(positions, voltages)
    finally:
        tiopt.set_device("cpu")
    assert gpu == pytest.approx(cpu, rel=1e-10, abs=1e-14)


def test_spectral_solver_matches_relaxation_problem(repository, ion):
    # The two square electrodes of trap.data, which lie in the trap surface
    trap = tiopt.Trap.from_file(repository / "trap.data")
    trap.solve("spectral")
    analytical = tiopt.Trap.from_file(repository / "trap.data")
    analytical.solve("analytical")
    # Close to an electrode, the grounded walls of the box are far away compared with the electrode, so the solutions
    # of the box problem and of the gapless plane agree to within the discretization of the electrode
    positions = np.array([[25.0, 0.0, 22.0], [50.0, 0.0, 22.0], [25.0, 5.0, 25.0]])
    voltages = np.array([1.0, 0.0])
    assert trap.potential(positions, voltages) == pytest.approx(analytical.potential(positions, voltages), rel=0.08)
