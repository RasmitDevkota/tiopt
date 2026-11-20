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
    # The first two ions leave the load zone one 50 µm segment apart, so the second arrives 250 µm of transport in
    assert times[0] == pytest.approx(250.0 / tiopt.TRANSPORT_SPEED_LIMIT)


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
        # A move can wait for segments and zones which other ions use, so it takes at least its distance
        distance = abs(trap_graph.positions[move.to_zone, 0] - trap_graph.positions[move.from_zone, 0])
        assert move.arrive - move.depart >= distance / tiopt.TRANSPORT_SPEED_LIMIT - 1e-12
        if move.transport >= 0:
            assert move.arrive <= program.arrival_times[move.transport] + 1e-12
    # Every step of a move lies within it, and moving along a 50 µm segment takes 50 µm at the speed limit
    for step in program.steps:
        if step.type == "operation":
            assert step.start == pytest.approx(program.arrival_times[step.index])
            continue
        move = moves[step.index]
        assert step.ion == move.ion and move.depart - 1e-12 <= step.start <= step.end <= move.arrive + 1e-12
        if step.type == "move":
            assert abs(step.to_zone - step.zone) == 1
            assert step.end - step.start == pytest.approx(50.0 / tiopt.TRANSPORT_SPEED_LIMIT)
    assert program.initial_zones == [0] * program.n_ions
    # Each ion makes one move at a time
    for ion in range(1, program.n_ions + 1):
        own = [m for m in moves if m.ion == ion]
        assert all(b.depart >= a.arrive - 1e-12 for a, b in zip(own, own[1:]))
    with pytest.raises(ValueError):
        tiopt.TransportProgram(trap_graph, circuit, "unknown")


def _replay_is_valid(program, trap_graph, circuit):
    """Whether every operation finds its ions in its zone once its round has arrived, with no zone over capacity."""
    capacities = trap_graph.capacities
    zones = {ion: 0 for ion in range(1, program.n_ions + 1)}
    rounds = program.rounds
    moves = program.moves
    for k, move in enumerate(moves):
        zones[move.ion] = move.to_zone
        if move.transport < 0:
            continue
        later = moves[k + 1] if k + 1 < len(moves) else None
        if later is not None and later.transport >= 0 and rounds[later.transport] == rounds[move.transport]:
            continue
        occupancy = np.bincount(list(zones.values()), minlength=len(capacities))
        if np.any(occupancy > capacities):
            return False
        for t in np.flatnonzero(rounds == rounds[move.transport]):
            for ion in circuit.instructions[t].targets:
                if trap_graph.positions[zones[ion], 0] != pytest.approx(program.destinations[t, 0]):
                    return False
    return True


@pytest.mark.parametrize("strategy", ["lookahead", "layer-assign"])
def test_lookahead_layers(repository, strategy):
    circuit = tiopt.Circuit(repository / "tests" / "quantumcircuits" / "lookahead_gap.qasm")
    def program(**kwargs):
        graph = tiopt.TrapGraph(["load+storage", "gate", "gate", "measure"], [4, 4, 4, 4], 0.0, 300.0)
        return graph, tiopt.TransportProgram(graph, circuit, strategy, **kwargs)

    default = program()[1]
    assert default.lookahead_layers == tiopt.DEFAULT_LOOKAHEAD_LAYERS == 1
    for kwargs in ({"lookahead_layers": 1}, {"lookahead_layers": 0}):
        same = program(**kwargs)[1]
        assert same.lookahead_layers == 1
        assert same.moves == default.moves
        np.testing.assert_array_equal(same.arrival_times, default.arrival_times)

    # Looking further ahead, qubit 1 waits in the gate zone through the layer it is not used in, instead of going back
    # into storage, and the schedule stays valid
    for depth, goes_back in ((1, True), (3, False)):
        graph, scheduled = program(lookahead_layers=depth)
        assert scheduled.lookahead_layers == depth
        assert _replay_is_valid(scheduled, graph, circuit)
        back = [m for m in scheduled.moves if m.ion == 1 and m.transport < 0 and m.arrive <= scheduled.arrival_times[2]]
        assert bool(back) == goes_back


def test_commute_moves_diagonal_gates_into_fewer_rounds(repository):
    circuit = tiopt.Circuit(repository / "tests" / "quantumcircuits" / "benchmarks" / "qaoa_06.qasm")
    layers = [i.layer for i in circuit.instructions]
    assert sum(i.diagonal for i in circuit.instructions) == 18
    rounds = {}
    for commute in (False, True):
        graph = tiopt.TrapGraph.from_layout("load, storage x2, gate x5, measure", 100.0, circuit.n_ions)
        program = tiopt.TransportProgram(graph, circuit, "layer-assign", commute=commute)
        assert program.commute == commute
        assert _replay_is_valid(program, graph, circuit)
        rounds[commute] = program.n_rounds
        # Operations on the same ion run in different rounds, in the order of the circuit unless both commute
        instructions = circuit.instructions
        for i in range(len(instructions)):
            for j in range(i + 1, len(instructions)):
                if set(instructions[i].targets) & set(instructions[j].targets):
                    assert program.rounds[i] != program.rounds[j]
                    if program.rounds[i] > program.rounds[j]:
                        assert commute and instructions[i].diagonal and instructions[j].diagonal
        # The layers of the circuit itself are unchanged
        assert [i.layer for i in circuit.instructions] == layers
    assert rounds[True] < rounds[False]


def test_trap_graph_from_segments_routes():
    # A T junction with a longer third arm
    graph = tiopt.TrapGraph.from_segments(["load+storage", "junction", "gate", "measure"], [4, 0, 2, 4],
                                          [(-200, 0), (0, 0), (200, 0), (0, 200)], [(0, 1), (1, 2), (1, 3)], [0, 0, 300])
    assert graph.distance(0, 2) == pytest.approx(400.0)
    assert graph.distance(2, 3) == pytest.approx(500.0)
    assert graph.route(3, 0) == [3, 1, 0]
    np.testing.assert_allclose(graph.positions[:, 2], tiopt.ION_HEIGHT)
    with pytest.raises(ValueError):
        tiopt.TrapGraph.from_segments(["load", "gate"], [2, 2], [(0, 0), (100, 0)], [(0, 2)])


def test_schedule_on_qccd_architecture(tmp_path):
    from tiopt import qccd

    # Storage on the left and two gate traps on the right of a Y junction, as a QCCD architecture
    architecture = qccd.Architecture()
    storage = architecture.add_trap("S", 4, (-200.0, 0.0), gate_zone=False)
    junction = architecture.add_junction("J", (0.0, 0.0))
    upper = architecture.add_trap("G1", 2, (150.0, 100.0))
    lower = architecture.add_trap("G2", 2, (150.0, -100.0))
    architecture.connect(storage, junction, 200.0)
    architecture.connect(junction, upper, 180.0)
    architecture.connect(junction, lower, 180.0)
    graph = tiopt.TrapGraph.from_architecture(architecture)
    assert graph.zone_types == ["load+storage", "junction", "gate+measure", "gate+measure"]
    assert graph.distance(storage, upper) == pytest.approx(380.0)
    assert graph.route(upper, lower) == [upper, junction, lower]

    qasm = tmp_path / "pairs.qasm"
    measurements = "".join(f"measure qreg[{q}] -> creg[{q}];\n" for q in range(4))
    qasm.write_text("qubit[4] qreg;\nbit[4] creg;\ncx qreg[0], qreg[1];\ncx qreg[2], qreg[3];\n" + measurements)
    program = tiopt.TransportProgram(graph, tiopt.Circuit(str(qasm)), "layer-assign")
    # Both gates run at once, one in each gate trap, and every ion crosses the junction on the way
    assert program.rounds[0] == program.rounds[1]
    assert {program.moves[k].to_zone for k in range(4)} == {upper, lower}
    steps = program.steps
    assert all(graph.distance(s.zone, s.to_zone) > 0 for s in steps if s.type == "move")
    assert not any(s.type == "merge" and s.zone == junction for s in steps)
    assert sum(s.type == "junction" for s in steps) == sum(1 for m in program.moves if m.from_zone != m.to_zone)

    # Every trap of an architecture made only of gate traps would have to load and store ions
    with pytest.raises(ValueError):
        tiopt.TrapGraph.from_architecture(qccd.Architecture.linear(3, 4))


def test_check_finds_no_violations_in_valid_schedules(circuit):
    # Schedules of every strategy which splits layers into rounds keep the rules of shuttling, while the original
    # strategies move ions away before their operations when a layer does not fit in the zones
    for strategy in ("rounds", "lookahead", "layer-assign"):
        program = tiopt.TransportProgram(tiopt.TrapGraph(ZONES, CAPACITIES, -150.0, 150.0), circuit, strategy, lookahead_layers=2)
        assert program.check() == []
    baseline = tiopt.TransportProgram(tiopt.TrapGraph(ZONES, CAPACITIES, -150.0, 150.0), circuit, "baseline")
    violations = baseline.check()
    assert violations and any("is not in the chain there" in v for v in violations)
