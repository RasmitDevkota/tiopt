"""Tests of circuits written in OpenQASM (tiopt.circuits)."""

import numpy as np
import pytest

import tiopt
from tiopt import circuits
from tiopt.circuits import GateErrors, QasmError, parse_qasm, simulate


def outcomes(result):
    return {outcome: probability for outcome, probability in zip(result.outcomes, result.ideal) if probability > 1e-9}


def test_presets_give_their_outcomes():
    assert outcomes(simulate(parse_qasm(circuits.PRESETS["Bernstein-Vazirani (101)"]))) == pytest.approx({"101": 1.0})
    assert outcomes(simulate(parse_qasm(circuits.PRESETS["Grover (2 qubits)"]))) == pytest.approx({"11": 1.0})
    assert outcomes(simulate(parse_qasm(circuits.PRESETS["Toffoli"]))) == pytest.approx({"111": 1.0})
    assert outcomes(simulate(parse_qasm(circuits.PRESETS["GHZ (5 qubits)"]))) == pytest.approx({"00000": 0.5, "11111": 0.5})
    assert len(outcomes(simulate(parse_qasm(circuits.PRESETS["QFT (3 qubits)"])))) == 8


def test_openqasm_2_and_3():
    two = parse_qasm('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nu3(pi/2, 0, pi) q[0];\nCX q[0],q[1];\nmeasure q -> c;')
    three = parse_qasm("OPENQASM 3;\nqubit[2] q;\nbit[2] c;\nh q[0];\ncx q[0], q[1];\nc = measure q;")
    assert outcomes(simulate(two)) == pytest.approx({"00": 0.5, "11": 0.5})
    assert outcomes(simulate(three)) == pytest.approx({"00": 0.5, "11": 0.5})
    # Bits are written with the first one on the right, and only measured qubits count
    result = simulate(parse_qasm("qubit[3] q; bit[2] c; x q[2]; measure q[2] -> c[0]; measure q[0] -> c[1];"))
    assert outcomes(result) == pytest.approx({"01": 1.0})


@pytest.mark.parametrize("text, message", [
    ("qubit[2] q; h q[2];", "out of range"),
    ("qubit[2] q; foo q[0];", "unknown gate"),
    ("qubit[2] q; bit[2] c; measure q -> c; h q[0];", "mid-circuit"),
    ("qubit[2] q; rx q[0];", "parameter"),
    ("qubit[2] q; cx q[0], q[0];", "same qubit"),
])
def test_errors_name_the_problem(text, message):
    with pytest.raises(QasmError, match=message):
        parse_qasm(text)


def test_gate_errors_lower_the_fidelity():
    circuit = parse_qasm(circuits.PRESETS["GHZ (3 qubits)"])
    ideal = simulate(circuit)
    noisy = simulate(circuit, GateErrors(two_qubit=0.01))
    worse = simulate(circuit, GateErrors(two_qubit=0.05))
    assert ideal.state_fidelity == 1.0
    assert 1 > noisy.state_fidelity > worse.state_fidelity
    # Each CNOT depolarizes its qubits with probability 5 e / 4 (for an average gate infidelity e), which leaves a
    # little of the state in place, so two CNOTs lower the fidelity of the state by a little under 2 (5 / 4) e
    assert 0.015 < 1 - noisy.state_fidelity < 2 * 1.25 * 0.01
    misread = simulate(circuit, GateErrors(measurement=0.1))
    assert misread.noisy[misread.outcomes.index("001")] > 0


def test_density_matrix_and_trajectories_agree():
    circuit = parse_qasm("qubit[3] q; bit[3] c; h q[0]; cx q[0], q[1]; cx q[1], q[2]; c = measure q;")
    errors = GateErrors(single_qubit=0.01, two_qubit=0.03)
    exact = simulate(circuit, errors)
    # Simulate the same circuit with state vectors and random errors, as larger circuits are
    saved = circuits.MAX_DENSITY_QUBITS
    circuits.MAX_DENSITY_QUBITS = 0
    try:
        sampled = simulate(circuit, errors, n_trajectories=3000, seed=1)
    finally:
        circuits.MAX_DENSITY_QUBITS = saved
    assert sampled.method.endswith("trajectories")
    assert sampled.state_fidelity == pytest.approx(exact.state_fidelity, abs=0.01)
    assert np.abs(sampled.noisy - exact.noisy).max() < 0.02


def test_native_gates_and_the_scheduler(tmp_path):
    circuit = parse_qasm(circuits.PRESETS["Toffoli"])
    counts = circuits.native_counts(circuit)
    assert counts.ms_gates == 6 and counts.measurements == 3
    path = tmp_path / "toffoli.qasm"
    circuits.write_scheduler_qasm(circuit, path)
    scheduled = tiopt.Circuit(path)
    operations = [instruction.operation for instruction in scheduled.instructions]
    assert operations.count("qubit") == 2 + 6 and operations.count("detect") == 3
