"""Tests of the timing of transports in each priority mode (tiopt.scheduling)."""

import numpy as np
import pytest

import tiopt
from tiopt import circuits, scheduling


@pytest.fixture(scope="module")
def program(tmp_path_factory):
    path = tmp_path_factory.mktemp("circuits") / "ghz3.qasm"
    circuits.write_scheduler_qasm(circuits.parse_qasm(circuits.PRESETS["GHZ (3 qubits)"]), path)
    circuit = tiopt.Circuit(path)
    graph = tiopt.TrapGraph.from_layout(tiopt.lib.DEFAULT_LAYOUT, 100.0, circuit.n_ions)
    return circuit, graph, tiopt.TransportProgram(graph, circuit)


def test_excitation_of_minimum_jerk_transports():
    # Slower transports excite the motion far less, and the excitation grows as the square of the distance
    fast = scheduling.transport_excitation(100.0, 5e-6)
    slow = scheduling.transport_excitation(100.0, 20e-6)
    assert fast > 100 * slow > 0
    assert scheduling.transport_excitation(200.0, 5e-6) == pytest.approx(4 * fast, rel=1e-9)


def test_modes_keep_the_operations_and_trade_the_motion(program):
    circuit, graph, transports = program
    timing = scheduling.Timing(max_speed=3e7, min_speed=1e6, heating_rate=200.0, cooling_time=50e-6)
    plans = scheduling.compare(transports, circuit, graph.positions, graph.zone_types, timing)
    starts = {mode: [op.start for op in plan.operations] for mode, plan in plans.items()}
    assert starts["eager"] == pytest.approx(starts["just in time"])
    assert starts["eager"] == pytest.approx(starts["as soon as possible"])
    # Eager transports take longer and excite less than fast ones
    assert plans["eager"].transport_time > plans["as soon as possible"].transport_time
    assert plans["eager"].total_excitation < plans["just in time"].total_excitation
    # Waiting in storage, where ions are cooled again, beats waiting at the gate zone
    assert plans["just in time"].two_qubit_n_bar < plans["as soon as possible"].two_qubit_n_bar
    assert any(kind == "cooling" for _, _, _, kind in plans["just in time"].segments)


def test_waveform_from_a_plan(program):
    circuit, graph, transports = program
    trap = tiopt.Trap.from_layout(tiopt.lib.DEFAULT_LAYOUT, 100.0, 50.0, "40Ca+", False)
    trap.solve("analytical")
    plan = scheduling.plan(transports, circuit, graph.positions, graph.zone_types, mode="eager")
    waveform = scheduling.waveform_from_plan(trap, tiopt.Ion("40Ca+"), plan, transports, n_steps=11)
    assert waveform.n_wells == 3
    assert waveform.times[0] == pytest.approx(min(t.depart for t in plan.transports))
    assert waveform.times[-1] == pytest.approx(max(t.arrive for t in plan.transports))
    assert np.all(np.diff(waveform.times) > 0)


def test_moves_back_into_storage_are_planned(program):
    circuit, graph, transports = program
    returns = [m for m in transports.moves if m.transport < 0]
    assert returns
    timing = scheduling.Timing(max_speed=3e7, min_speed=1e6, heating_rate=200.0, cooling_time=50e-6)
    plans = scheduling.compare(transports, circuit, graph.positions, graph.zone_types, timing)
    for mode, plan in plans.items():
        back = [t for t in plan.transports if t.instruction < 0]
        # Every move back into storage before an operation is planned, and every ion makes one move at a time
        assert back, mode
        for ion in range(transports.n_ions):
            own = sorted((t for t in plan.transports if t.ion == ion), key=lambda t: t.depart)
            assert all(b.depart >= a.arrive - 1e-15 for a, b in zip(own, own[1:])), mode
        # Ions reach the zones of their operations by the time they start
        starts = {op.instruction: op.start for op in plan.operations}
        assert all(t.arrive <= starts[t.instruction] + 1e-15 for t in plan.transports if t.instruction >= 0), mode
    # Just in time, ions go back into storage at once and are cooled there if they have time before their next
    # operation, and otherwise leave again at once
    plan = plans["just in time"]
    after = [[kind for ion, start, _, kind in plan.segments if ion == t.ion and abs(start - t.arrive) < 1e-15]
             for t in plan.transports if t.instruction < 0]
    assert all(kinds and kinds[0] in ("cooling", "transport") for kinds in after)
    assert any(kinds[0] == "cooling" for kinds in after)


def test_programs_without_moves_are_planned_as_before(program):
    circuit, graph, transports = program

    class Program:
        n_ions = transports.n_ions
        targets = transports.targets
        destinations = transports.destinations
        initial_positions = transports.initial_positions

    plan = scheduling.plan(Program(), circuit, graph.positions, graph.zone_types, mode="eager")
    assert all(t.instruction >= 0 for t in plan.transports)
    assert len(plan.transports) < len(scheduling.plan(transports, circuit, graph.positions, graph.zone_types, mode="eager").transports)


def test_plans_with_commuting_operations_moved(repository):
    circuit = tiopt.Circuit(repository / "tests" / "quantumcircuits" / "benchmarks" / "qaoa_06.qasm")
    graph = tiopt.TrapGraph.from_layout("load, storage x2, gate x5, measure", 100.0, circuit.n_ions)
    transports = tiopt.TransportProgram(graph, circuit, "layer-assign", commute=True)
    plans = scheduling.compare(transports, circuit, graph.positions, graph.zone_types)
    starts = {mode: {op.instruction: op.start for op in plan.operations} for mode, plan in plans.items()}
    assert starts["eager"] == pytest.approx(starts["just in time"])
    assert starts["eager"] == pytest.approx(starts["as soon as possible"])
    # Each ion's operations start in the order of their rounds, which is not the order of the circuit
    for ion in range(1, circuit.n_ions + 1):
        own = [i for i, instruction in enumerate(circuit.instructions) if ion in instruction.targets]
        by_round = sorted(own, key=lambda i: transports.rounds[i])
        times = [starts["eager"][i] for i in by_round]
        assert all(b > a for a, b in zip(times, times[1:]))
    assert any(transports.rounds[i] > transports.rounds[j] for i in range(len(circuit.instructions))
               for j in range(i + 1, len(circuit.instructions)))
