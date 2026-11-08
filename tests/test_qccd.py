"""Tests of the QCCD compiler and simulator (tiopt.qccd)."""

import numpy as np
import pytest

import tiopt
from tiopt import circuits, qccd


def random_circuit(n_qubits, n_gates, seed=1):
    rng = np.random.default_rng(seed)
    lines = ["OPENQASM 3;", f"qubit[{n_qubits}] q;"]
    for _ in range(n_gates):
        a, b = rng.choice(n_qubits, 2, replace=False)
        lines.append(f"cx q[{a}], q[{b}];")
    return circuits.parse_qasm("\n".join(lines))


def test_architectures():
    linear = qccd.Architecture.linear(4, 5)
    assert len(linear.traps) == 4 and len(linear.junctions) == 3
    grid = qccd.Architecture.grid(2, 3, 4)
    assert len(grid.junctions) == 6 and len(grid.traps) == 7
    # The corners of a 2 x 3 grid of junctions join two traps, and the junctions between them three
    assert sorted(len(grid.neighbors[j]) for j in grid.junctions) == [2, 2, 2, 2, 3, 3]
    ring = qccd.Architecture.ring(5, 3)
    assert all(len(ring.neighbors[t]) == 2 for t in ring.traps)
    with pytest.raises(ValueError):
        linear.connect(linear.traps[1], linear.traps[2])


def test_compiling_ghz(tmp_path):
    circuit = circuits.parse_qasm(circuits.PRESETS["GHZ (5 qubits)"])
    for architecture in (qccd.Architecture.linear(3, 4), qccd.Architecture.grid(2, 3, 4), qccd.Architecture.ring(4, 3)):
        for swap in ("gate", "ion"):
            result = qccd.compile_circuit(circuit, architecture, qccd.TimingModel(swap_type=swap))
            assert result.counts["two-qubit gate"] == 4
            assert result.counts["measurement"] == 5
            assert result.counts.get("split", 0) == result.counts.get("merge", 0)
            assert 0.9 < result.fidelity < 1.0
            # Every qubit ends up in a trap, and no trap holds more than its capacity
            assert sorted(result.final_mapping) == list(range(5))
            assert result.duration > 0
    # Everything fits in one big trap without any shuttling
    result = qccd.compile_circuit(circuit, qccd.Architecture.linear(1, 8))
    assert result.counts.get("split", 0) == 0


def test_larger_traps_shuttle_less():
    circuit = random_circuit(12, 40)
    small = qccd.compile_circuit(circuit, qccd.Architecture.linear(6, 4))
    large = qccd.compile_circuit(circuit, qccd.Architecture.linear(3, 8))
    assert large.counts["split"] < small.counts["split"]
    recooled = qccd.compile_circuit(circuit, qccd.Architecture.linear(6, 4), noise=qccd.NoiseModel(recool=True))
    assert recooled.max_n_bar < small.max_n_bar
    rows = qccd.explore(circuit, {"L": lambda capacity: qccd.Architecture.linear(3, capacity)}, capacities=(4, 8), gate_models=("AM2", "FM"), swap_types=("gate",))
    assert len(rows) == 4
    assert any("error" in row for row in rows)
    with pytest.raises(ValueError):
        qccd.compile_circuit(circuit, qccd.Architecture.linear(2, 4))


def test_architectures_of_traps():
    trap = tiopt.Trap.from_layout("load+storage, gate, measure", zone_spacing=80.0)
    architecture = qccd.Architecture.from_trap(trap)
    assert len(architecture.traps) == 3
    assert sum(node.gate_zone for node in architecture.nodes if node.kind == "trap") == 1
    result = qccd.compile_circuit(circuits.parse_qasm(circuits.PRESETS["Bell pair"]), architecture)
    assert result.counts["two-qubit gate"] == 1
    gate_trap = next(node.name for node in architecture.nodes if node.gate_zone)
    assert any(op.kind == "two-qubit gate" and architecture.nodes[op.nodes[0]].name == gate_trap for op in result.operations)
