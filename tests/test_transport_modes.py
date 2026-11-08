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
