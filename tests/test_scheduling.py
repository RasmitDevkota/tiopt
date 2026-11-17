"""Tests of QASM parsing, circuit layering and transport scheduling through the Python bindings."""

import numpy as np
import pytest

import tiopt

ZONES = ["load", "empty", "storage", "empty", "gate", "empty", "measure"]
CAPACITIES = [5, 3, 10, 3, 2, 3, 4]


@pytest.fixture
def circuit(repository):
    return tiopt.Circuit(repository / "tests" / "quantumcircuits" / "test_0000.qasm")


def test_circuit_layers(circuit):
    instructions = circuit.instructions
    assert circuit.n_ions == 4
    assert [i.layer for i in instructions] == [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5]
    assert [i.targets for i in instructions[4:10]] == [[1, 2], [3, 4], [1, 3], [2, 4], [1, 4], [2, 3]]
    # The measurement of the whole register follows the last two gates
    assert instructions[-1].operation == "detect"
    assert instructions[-1].targets == [1, 2, 3, 4]
    assert sorted(instructions[-1].previous) == [9, 10]


def test_missing_qasm_file(tmp_path):
    with pytest.raises(tiopt.TiOptError):
        tiopt.Circuit(tmp_path / "missing.qasm")


def test_transport_schedule(circuit):
    trap_graph = tiopt.TrapGraph(ZONES, CAPACITIES, -150.0, 150.0)
    np.testing.assert_allclose(trap_graph.positions[:, 0], np.linspace(-150, 150, 7))

    program = tiopt.TransportProgram(trap_graph, circuit)
    assert (program.n_ions, program.n_transports) == (4, 11)
    np.testing.assert_allclose(program.initial_positions[:, 0], -150.0)
    # Gates happen in the gate zone and the measurement in the measure zone
    np.testing.assert_allclose(program.destinations[:10, 0], 50.0)
    assert program.destinations[10, 0] == 150.0
    # Each layer arrives in rounds of operations which fit in the gate zone (2 ions) at once, each after the previous
    # round, so the 4 single-qubit gates take 2 rounds and every layer of two-qubit gates takes 2 rounds
    layers = np.array([i.layer for i in circuit.instructions])
    n_targets = np.array([len(i.targets) for i in circuit.instructions])
    times = program.arrival_times
    rounds = np.unique(times, return_inverse=True)[1]
    assert rounds.max() + 1 == 9
    for r in range(rounds.max() + 1):
        assert len(np.unique(layers[rounds == r])) == 1
        if layers[rounds == r][0] < 5:
            assert n_targets[rounds == r].sum() <= 2
    assert np.all(np.diff(times) >= 0)
    assert times[0] == pytest.approx(200.0 / tiopt.TRANSPORT_SPEED_LIMIT)


def test_schedule_without_measure_zone_fails(circuit):
    trap_graph = tiopt.TrapGraph(["load", "gate", "storage"], [5, 2, 10], -50.0, 50.0)
    with pytest.raises(tiopt.TiOptError):
        tiopt.TransportProgram(trap_graph, circuit)


def test_waveform_for_schedule(solved_trap, ion, circuit):
    null = solved_trap.find_rf_null((0.0, 0.0, None))
    trap_graph = tiopt.TrapGraph(ZONES, CAPACITIES, -150.0, 150.0, null[2])
    program = tiopt.TransportProgram(trap_graph, circuit)
    waveform = tiopt.Waveform.from_program(solved_trap, ion, program, 21)
    assert waveform.n_wells == 4
    # 21 steps over every move which takes time, where moves which happen at once share their steps
    moving = [m for m in program.moves if m.arrive > m.depart]
    assert 21 <= waveform.n_steps <= 21 * len(moving)
    assert np.all(np.diff(waveform.times) > 0)
    # Every move, including the moves back into storage, brings its ion to the center of its zone when it arrives
    for move in moving:
        step = int(np.argmin(np.abs(waveform.times - move.arrive)))
        assert waveform.times[step] == pytest.approx(move.arrive)
        assert waveform.well_positions[step, move.ion - 1, 0] == pytest.approx(trap_graph.positions[move.to_zone, 0])
    # Every ion is in the measure zone for the measurement, and back in storage at the end
    measurement = int(np.argmin(np.abs(waveform.times - program.arrival_times[-1])))
    np.testing.assert_allclose(waveform.well_positions[measurement, :, 0], 150.0)
    np.testing.assert_allclose(waveform.well_positions[-1, :, 0], -50.0)


def test_program_times_every_move(circuit):
    trap_graph = tiopt.TrapGraph(ZONES, CAPACITIES, -150.0, 150.0)
    program = tiopt.TransportProgram(trap_graph, circuit, "rounds")
    moves = program.moves
    assert program.n_rounds == 9
    np.testing.assert_array_equal(np.unique(program.rounds), np.arange(1, 10))
    # The ions go back into storage between operations, and those moves take time too
    releases = [m for m in moves if m.transport < 0]
    assert releases and all(m.arrive > m.depart for m in releases)
    for move in moves:
        distance = abs(trap_graph.positions[move.to_zone, 0] - trap_graph.positions[move.from_zone, 0])
        assert move.arrive - move.depart == pytest.approx(distance / tiopt.TRANSPORT_SPEED_LIMIT)
        if move.transport >= 0:
            assert move.arrive <= program.arrival_times[move.transport] + 1e-12
    # Each ion makes one move at a time
    for ion in range(1, program.n_ions + 1):
        own = [m for m in moves if m.ion == ion]
        assert all(b.depart >= a.arrive - 1e-12 for a, b in zip(own, own[1:]))
    with pytest.raises(ValueError):
        tiopt.TransportProgram(trap_graph, circuit, "unknown")
