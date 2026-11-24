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
    assert [i.layer for i in instructions] == [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5]
    assert [i.targets for i in instructions[4:10]] == [[1, 2], [3, 4], [1, 3], [2, 4], [1, 4], [2, 3]]
    # The measurement of the whole register measures each qubit on its own, after the last gate on it
    assert [i.operation for i in instructions[10:]] == ["detect"] * 4
    assert [i.targets for i in instructions[10:]] == [[1], [2], [3], [4]]
    assert [i.previous for i in instructions[10:]] == [[9], [10], [10], [9]]


def test_missing_qasm_file(tmp_path):
    with pytest.raises(tiopt.TiOptError):
        tiopt.Circuit(tmp_path / "missing.qasm")


def test_transport_schedule(circuit):
    trap_graph = tiopt.TrapGraph(ZONES, CAPACITIES, -150.0, 150.0)
    np.testing.assert_allclose(trap_graph.positions[:, 0], np.linspace(-150, 150, 7))

    # Times only count distance here
    program = tiopt.TransportProgram(trap_graph, circuit, cost_model=tiopt.distance_cost_model())
    assert (program.n_ions, program.n_transports) == (4, 14)
    np.testing.assert_allclose(program.initial_positions[:, 0], -150.0)
    # Gates happen in the gate zone and the measurements in the measure zone
    np.testing.assert_allclose(program.destinations[:10, 0], 50.0)
    np.testing.assert_allclose(program.destinations[10:, 0], 150.0)
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


@pytest.mark.parametrize("swap_type", ["gate", "ion"])
def test_waveform_for_schedule(solved_trap, ion, circuit, swap_type):
    from tiopt import qccd

    null = solved_trap.find_rf_null((0.0, 0.0, None))
    trap_graph = tiopt.TrapGraph(["load", "storage", "gate", "gate", "measure"], [4, 4, 2, 2, 4], -150.0, 150.0, null[2])
    # Ions kept in the first gate zone make others pass through them on the way to the second, reordering with SWAP
    # gates where gates can be done (the default), or by swapping ions physically everywhere
    cost_model = tiopt.transport_cost_model(qccd.TimingModel(swap_type=swap_type))
    program = tiopt.TransportProgram(trap_graph, circuit, "lookahead", cost_model=cost_model)
    waveform = tiopt.Waveform.from_program(solved_trap, ion, program, 21)
    assert waveform.n_wells == 4
    assert np.all(np.diff(waveform.times) > 0)
    x = trap_graph.positions[:, 0]
    steps = program.steps
    # Each well follows the steps of its ion: it moves only along the segments while its ion moves, and holds still at
    # the center of its zone through splits, merges, swaps, operations, and waits
    expected = np.repeat(program.initial_positions[None, :, 0], waveform.n_steps, axis=0)
    for step in sorted((st for st in steps if st.type == "move"), key=lambda st: st.start):
        u = np.clip((waveform.times - step.start) / (step.end - step.start), 0.0, 1.0)
        during = waveform.times > step.start
        position = x[step.zone] + u**3 * (10 - 15 * u + 6 * u * u) * (x[step.to_zone] - x[step.zone])
        expected[during, step.ion - 1] = position[during]
    np.testing.assert_allclose(waveform.well_positions[:, :, 0], expected, atol=1e-9)
    # The waveform holds through every other step, with a step at its start and end, until the last operation is done
    for step in steps:
        if step.type in ("split", "merge", "ion swap", "gate swap"):
            for t in (step.start, step.end):
                k = int(np.argmin(np.abs(waveform.times - t)))
                assert waveform.times[k] == pytest.approx(t)
                for ion_ in (step.ion, step.other_ion):
                    if ion_:
                        assert waveform.well_positions[k, ion_ - 1, 0] == pytest.approx(x[step.zone])
    assert waveform.times[-1] == pytest.approx(program.duration)
    # A physical swap of the two ions of a chain turns their well by pi, and nothing else turns a well
    turns = np.zeros(4)
    for step in steps:
        k0 = int(np.argmin(np.abs(waveform.times - step.start)))
        k1 = int(np.argmin(np.abs(waveform.times - step.end)))
        here = np.abs(waveform.well_positions[k0, :, 0] - x[step.zone]) < 1e-6
        if step.type == "ion swap" and here.sum() == 2:
            for ion_ in (step.ion, step.other_ion):
                turns[ion_ - 1] += np.pi
                assert waveform.well_angles[k1, ion_ - 1] - waveform.well_angles[k0, ion_ - 1] == pytest.approx(np.pi)
    np.testing.assert_allclose(waveform.well_angles[-1], turns)
    assert (turns.sum() > 0) == (swap_type == "ion")
    # Every ion is in the measure zone for the measurement
    measurement = int(np.argmin(np.abs(waveform.times - program.arrival_times[-1])))
    np.testing.assert_allclose(waveform.well_positions[measurement, :, 0], 150.0)


def test_program_times_every_move(circuit):
    trap_graph = tiopt.TrapGraph(ZONES, CAPACITIES, -150.0, 150.0)
    # Times only count distance here
    program = tiopt.TransportProgram(trap_graph, circuit, "rounds", cost_model=tiopt.distance_cost_model())
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


def test_default_cost_model_is_qccd():
    from tiopt import gates, qccd

    # The scheduler's default cost model has the durations, heating and errors of tiopt.qccd, in s rather than µs
    default = tiopt.lib._lib.default_transport_cost_model()
    from_qccd = tiopt.transport_cost_model(qccd.TimingModel(), qccd.NoiseModel())
    for name, _ in tiopt.lib.TransportCostModelStruct._fields_:
        assert getattr(default, name) == pytest.approx(getattr(from_qccd, name), rel=1e-12), name
    assert default.speed == pytest.approx(20e6) and default.split == pytest.approx(80e-6)
    # Its gate model gives the durations of tiopt.gates
    for n_ions, separation in ((2, 1), (8, 3), (16, 7)):
        assert tiopt.lib._lib.two_qubit_gate_time(default, n_ions, separation) == pytest.approx(
            1e-6 * gates.gate_time_model("FM", n_ions, separation))


def test_move_through_junction_takes_qccd_time(tmp_path):
    from tiopt import qccd

    # Two ions in storage go through a Y junction to a gate trap: the first splits off, moves along both segments, and
    # crosses the junction, as QCCDSim's travel time without the merge (its destination is empty)
    architecture = qccd.Architecture()
    storage = architecture.add_trap("S", 4, (-200.0, 0.0), gate_zone=False)
    junction = architecture.add_junction("J", (0.0, 0.0))
    upper = architecture.add_trap("G1", 2, (150.0, 100.0))
    lower = architecture.add_trap("G2", 2, (150.0, -100.0))
    architecture.connect(storage, junction, 200.0)
    architecture.connect(junction, upper, 180.0)
    architecture.connect(junction, lower, 180.0)
    timing = qccd.TimingModel()

    qasm = tmp_path / "pair.qasm"
    qasm.write_text("qubit[2] qreg;\ncx qreg[0], qreg[1];\n")
    program = tiopt.TransportProgram(tiopt.TrapGraph.from_architecture(architecture), tiopt.Circuit(str(qasm)))
    first = program.moves[0]
    expected = timing.split + timing.move_time(200.0) + timing.junction_y + timing.move_time(180.0)
    assert (first.arrive - first.depart) * 1e6 == pytest.approx(expected)
    # The second merges into the chain there, which QCCDSim also counts
    compiler_time = qccd._Compiler(architecture, timing, qccd.NoiseModel(), 2)
    compiler_time.location = {0: storage, 1: storage}
    assert compiler_time.travel_time(0, upper) == pytest.approx(expected + timing.merge)
    assert program.check() == []


def test_fidelity_follows_qccd_noise_model(tmp_path):
    from tiopt import gates, qccd

    # Two cx gates on the same pair: with lookahead the ions stay in the gate zone, so the only operations are the two
    # gates, each with the fidelity 1 - gamma tau - A (2 n_bar + 1) of tiopt.qccd at the mean phonon number of the chain
    qasm = tmp_path / "twice.qasm"
    qasm.write_text("qubit[2] qreg;\ncx qreg[0], qreg[1];\ncx qreg[0], qreg[1];\n")
    graph = tiopt.TrapGraph.from_layout("load+storage, empty, gate, measure", 100.0, 2)
    program = tiopt.TransportProgram(graph, tiopt.Circuit(str(qasm)), "lookahead")
    noise = qccd.NoiseModel()
    tau = gates.gate_time_model("FM", 2, 1)
    n_bars = program.operation_n_bars
    assert n_bars[1] == pytest.approx(n_bars[0] + noise.background_heating * tau)
    a = noise.motional_error * 2 / np.log(2)
    expected = np.prod([1 - noise.gate_background_error * tau - a * (2 * n + 1) for n in n_bars])
    assert program.fidelity == pytest.approx(expected, rel=1e-12)
    assert program.max_n_bar == pytest.approx(n_bars.max())
    assert not any(step.type == "gate swap" for step in program.steps)
    assert program.check() == []


def test_cheaper_swaps(repository):
    from tiopt import qccd

    # Gate traps of 6 ions beyond a Y junction, where ions kept in a trap by lookahead make others reorder to reach them
    architecture = qccd.Architecture()
    storage = architecture.add_trap("S", 6, (-200.0, 0.0), gate_zone=False)
    junction = architecture.add_junction("J", (0.0, 0.0))
    upper = architecture.add_trap("G1", 6, (150.0, 100.0))
    lower = architecture.add_trap("G2", 6, (150.0, -100.0))
    architecture.connect(storage, junction, 200.0)
    architecture.connect(junction, upper, 180.0)
    architecture.connect(junction, lower, 180.0)
    circuit = tiopt.Circuit(repository / "tests" / "quantumcircuits" / "benchmarks" / "qft_06.qasm")
    programs = {}
    for swap in qccd.SWAP_TYPES:
        cost_model = tiopt.transport_cost_model(qccd.TimingModel(swap_type=swap))
        programs[swap] = tiopt.TransportProgram(tiopt.TrapGraph.from_architecture(architecture), circuit, "lookahead", cost_model=cost_model)
        assert programs[swap].check() == []
    # The default takes a SWAP gate or physical swaps, whichever costs less fidelity, which beats either on its own here
    kinds = {step.type for step in programs["cheaper"].steps}
    assert {"gate swap", "ion swap"} <= kinds
    assert programs["cheaper"].fidelity > max(programs["gate"].fidelity, programs["ion"].fidelity)
    assert tiopt.lib._lib.default_transport_cost_model().swap_gates == tiopt.lib.TRANSPORT_SWAP_METHODS["cheaper"]
    with pytest.raises(ValueError):
        tiopt.transport_cost_model(qccd.TimingModel(swap_type="teleport"))


@pytest.mark.parametrize("name", ["qft_06", "heisenberg_06", "ghz_12", "qft_12"])
def test_gates_take_chain_in_turn_on_qccd_architecture(repository, name):
    from tiopt import qccd

    # The architecture of the comparison with tiopt.qccd: gate traps of 6 hold several gates of a round at once
    architecture = qccd.Architecture()
    storage = architecture.add_trap("S", 16, (-200.0, 0.0), gate_zone=False)
    junction = architecture.add_junction("J", (0.0, 0.0))
    upper = architecture.add_trap("G1", 6, (150.0, 100.0))
    lower = architecture.add_trap("G2", 6, (150.0, -100.0))
    architecture.connect(storage, junction, 200.0)
    architecture.connect(junction, upper, 180.0)
    architecture.connect(junction, lower, 180.0)

    # The 12-qubit circuits fit only because each qubit is measured on its own
    circuit = tiopt.Circuit(repository / "tests" / "quantumcircuits" / "benchmarks" / f"{name}.qasm")
    for strategy in ["rounds", "lookahead", "layer-assign"]:
        program = tiopt.TransportProgram(tiopt.TrapGraph.from_architecture(architecture), circuit, strategy)
        assert program.check() == []

        # An operation on several ions never overlaps another operation in its zone
        targets = program.targets
        operations = [s for s in program.steps if s.type == "operation"]
        for a in operations:
            for b in operations:
                if a is b or a.zone != b.zone or max(len(targets[a.index]), len(targets[b.index])) < 2:
                    continue
                assert a.end <= b.start + 1e-12 or b.end <= a.start + 1e-12


@pytest.mark.parametrize("name", ["ghz_06", "qft_06", "qaoa_12"])
def test_stay_leaves_ions_in_place(repository, name):
    from tiopt import qccd

    architecture = qccd.Architecture()
    storage = architecture.add_trap("S", 16, (-200.0, 0.0), gate_zone=False)
    junction = architecture.add_junction("J", (0.0, 0.0))
    upper = architecture.add_trap("G1", 6, (150.0, 100.0))
    lower = architecture.add_trap("G2", 6, (150.0, -100.0))
    architecture.connect(storage, junction, 200.0)
    architecture.connect(junction, upper, 180.0)
    architecture.connect(junction, lower, 180.0)

    circuit = tiopt.Circuit(repository / "tests" / "quantumcircuits" / "benchmarks" / f"{name}.qasm")
    rounds = tiopt.TransportProgram(tiopt.TrapGraph.from_architecture(architecture), circuit, "rounds")
    stay = tiopt.TransportProgram(tiopt.TrapGraph.from_architecture(architecture), circuit, "stay")
    assert stay.check() == []

    # Ions go back into storage only to make room, so there are fewer moves and the program is shorter
    def moved(program):
        return [m for m in program.moves if m.from_zone != m.to_zone]

    assert len(moved(stay)) < len(moved(rounds))
    assert sum(m.transport < 0 for m in moved(stay)) < sum(m.transport < 0 for m in moved(rounds))
    assert stay.duration < rounds.duration
