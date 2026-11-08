"""Quantum circuits written in OpenQASM, which can be edited, simulated with or without gate errors, compiled into the
native gates of trapped ions, and handed to the transport scheduler.

The parser reads OpenQASM 2 and 3 (registers declared with qreg/creg or qubit/bit, the standard gates of qelib1.inc,
parameters written with pi and arithmetic, measurements with -> or =, and barriers), and the simulator follows the state
vector of the circuit, or its density matrix when gates have errors. Every gate can be followed by depolarizing errors,
idle qubits can dephase in every layer, and measurements can be misread. Outcomes are written with the first classical
bit on the right, as Qiskit does.

The C scheduler (see transport.c) only reads a few operations on a register named qreg, so to_scheduler_qasm writes a
copy of a circuit with every physical single-qubit gate as x, every two-qubit gate as cx, and three-qubit gates broken
into those.
"""

import ast
import math
import operator
import re
from dataclasses import dataclass, field

import numpy as np

# Largest numbers of qubits which are simulated with a density matrix (for gate errors) and with a state vector
MAX_DENSITY_QUBITS = 8
MAX_STATE_QUBITS = 20

# Native gates of trapped ions per gate of the circuit: two-qubit gates are made of Mølmer-Sørensen gates and the
# single-qubit rotations around them (4 per CNOT), and Z rotations are virtual (made by shifting the phases of later
# pulses)
ROTATIONS_PER_MS = 4
MS_PER_GATE = {"swap": 3, "ccx": 6, "cswap": 8, "iswap": 2, "cu3": 2, "crx": 2, "cry": 2, "crz": 2, "cp": 2, "ch": 1}
VIRTUAL_GATES = {"id", "z", "s", "sdg", "t", "tdg", "rz", "p", "u1"}

PRESETS = {
    "Bell pair": """// Bell pair: a Hadamard and a CNOT entangle two qubits, which are then measured
OPENQASM 3;
qubit[2] qreg;
bit[2] creg;

h qreg[0];
cx qreg[0], qreg[1];
measure qreg -> creg;
""",
    "GHZ (3 qubits)": """// Three-qubit GHZ state
OPENQASM 3;
qubit[3] qreg;
bit[3] creg;

h qreg[0];
cx qreg[0], qreg[1];
cx qreg[1], qreg[2];
measure qreg -> creg;
""",
    "GHZ (5 qubits)": """// Five-qubit GHZ state, which needs more shuttling
OPENQASM 3;
qubit[5] qreg;
bit[5] creg;

h qreg[0];
cx qreg[0], qreg[1];
cx qreg[1], qreg[2];
cx qreg[2], qreg[3];
cx qreg[3], qreg[4];
measure qreg -> creg;
""",
    "Grover (2 qubits)": """// Grover search for |11> on two qubits, which finds it with one iteration
OPENQASM 3;
qubit[2] qreg;
bit[2] creg;

h qreg;
// Oracle marking |11>
cz qreg[0], qreg[1];
// Diffusion
h qreg;
x qreg;
cz qreg[0], qreg[1];
x qreg;
h qreg;
measure qreg -> creg;
""",
    "Bernstein-Vazirani (101)": """// Bernstein-Vazirani algorithm for the secret string 101, with qreg[3] as the ancilla
OPENQASM 3;
qubit[4] qreg;
bit[3] creg;

x qreg[3];
h qreg;
cx qreg[0], qreg[3];
cx qreg[2], qreg[3];
h qreg[0];
h qreg[1];
h qreg[2];
measure qreg[0] -> creg[0];
measure qreg[1] -> creg[1];
measure qreg[2] -> creg[2];
""",
    "QFT (3 qubits)": """// Quantum Fourier transform of |1> on three qubits, which gives every outcome equally
OPENQASM 3;
qubit[3] qreg;
bit[3] creg;

x qreg[0];
h qreg[2];
cp(pi/2) qreg[1], qreg[2];
cp(pi/4) qreg[0], qreg[2];
h qreg[1];
cp(pi/2) qreg[0], qreg[1];
h qreg[0];
swap qreg[0], qreg[2];
measure qreg -> creg;
""",
    "Toffoli": """// Toffoli gate on |110>, which flips the target to give |111>
OPENQASM 3;
qubit[3] qreg;
bit[3] creg;

x qreg[0];
x qreg[1];
ccx qreg[0], qreg[1], qreg[2];
measure qreg -> creg;
""",
    "Four-qubit mix": """// Single-qubit gates and controlled Pauli gates on four qubits (the circuit of the test suite)
OPENQASM 3;
qubit[4] qreg;
bit[4] creg;

h qreg[0];
x qreg[1];
y qreg[2];
z qreg[3];

cx qreg[0], qreg[1];
cx qreg[2], qreg[3];

cy qreg[0], qreg[2];
cz qreg[1], qreg[3];

cz qreg[0], qreg[3];
cz qreg[1], qreg[2];

measure qreg -> creg;
""",
}


class QasmError(ValueError):
    """Error in an OpenQASM program, with the line it is on."""


@dataclass
class Gate:
    """Gate, measurement ("measure", whose params hold the classical bit), or barrier of a circuit."""

    name: str
    qubits: tuple
    params: tuple = ()

    def __str__(self):
        params = "(" + ", ".join(f"{p:.4g}" for p in self.params) + ")" if self.params and self.name != "measure" else ""
        qubits = ", ".join(f"q{q}" for q in self.qubits)
        if self.name == "measure":
            return f"measure q{self.qubits[0]} -> c{self.params[0]}"
        return f"{self.name}{params} {qubits}"


@dataclass
class QuantumCircuit:
    """Circuit on n_qubits qubits and n_clbits classical bits, with its gates in order."""

    n_qubits: int
    n_clbits: int = 0
    gates: list = field(default_factory=list)
    qubit_names: list = field(default_factory=list)

    @property
    def operations(self):
        """Gates of the circuit without barriers and measurements."""
        return [gate for gate in self.gates if gate.name not in ("barrier", "measure")]

    @property
    def measurements(self):
        """Classical bit of every measured qubit, as {qubit: clbit}."""
        return {gate.qubits[0]: gate.params[0] for gate in self.gates if gate.name == "measure"}

    def layers(self):
        """Gates grouped into layers which run at the same time, where each gate goes in the layer after the last gate on
        any of its qubits.
        """
        depth = [0] * self.n_qubits
        layers = []
        for gate in self.gates:
            if gate.name == "barrier":
                level = max(depth[q] for q in gate.qubits) if gate.qubits else 0
                for q in gate.qubits:
                    depth[q] = level
                continue
            level = max(depth[q] for q in gate.qubits)
            if level == len(layers):
                layers.append([])
            layers[level].append(gate)
            for q in gate.qubits:
                depth[q] = level + 1
        return layers

    @property
    def depth(self):
        return len(self.layers())

    def to_qasm(self):
        """Write the circuit as OpenQASM 3 on registers named qreg and creg."""
        lines = ["OPENQASM 3;", f"qubit[{self.n_qubits}] qreg;"]
        if self.n_clbits:
            lines.append(f"bit[{self.n_clbits}] creg;")
        for gate in self.gates:
            qubits = ", ".join(f"qreg[{q}]" for q in gate.qubits)
            if gate.name == "measure":
                lines.append(f"measure qreg[{gate.qubits[0]}] -> creg[{gate.params[0]}];")
            elif gate.params:
                lines.append(f"{gate.name}({', '.join(repr(float(p)) for p in gate.params)}) {qubits};")
            else:
                lines.append(f"{gate.name} {qubits};")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------------------------------

_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCTIONS = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp, "ln": math.log, "sqrt": math.sqrt, "arcsin": math.asin, "arccos": math.acos, "arctan": math.atan}
_CONSTANTS = {"pi": math.pi, "π": math.pi, "tau": 2 * math.pi, "e": math.e}


def _evaluate(expression):
    """Evaluate a parameter such as pi/4 or -3*pi/2 without running any code."""

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](visit(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS and len(node.args) == 1:
            return _FUNCTIONS[node.func.id](visit(node.args[0]))
        raise ValueError(f"can't evaluate {expression!r}")

    return visit(ast.parse(expression.replace("^", "**").replace("π", "pi"), mode="eval"))


def _split_arguments(text):
    """Split a comma-separated list of arguments, keeping commas inside brackets and parentheses."""
    parts, depth, current = [], 0, ""
    for char in text:
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current.strip())
    return parts


# Number of qubits and parameters of each gate, with aliases mapped onto one name
GATES = {
    "id": (1, 0), "x": (1, 0), "y": (1, 0), "z": (1, 0), "h": (1, 0), "s": (1, 0), "sdg": (1, 0), "t": (1, 0), "tdg": (1, 0),
    "sx": (1, 0), "sxdg": (1, 0), "rx": (1, 1), "ry": (1, 1), "rz": (1, 1), "p": (1, 1), "u1": (1, 1), "u2": (1, 2), "u3": (1, 3),
    "r": (1, 2),
    "cx": (2, 0), "cy": (2, 0), "cz": (2, 0), "ch": (2, 0), "swap": (2, 0), "iswap": (2, 0), "crx": (2, 1), "cry": (2, 1), "crz": (2, 1),
    "cp": (2, 1), "cu3": (2, 3), "rxx": (2, 1), "ryy": (2, 1), "rzz": (2, 1), "ms": (2, 1),
    "ccx": (3, 0), "cswap": (3, 0),
}
ALIASES = {"cnot": "cx", "CX": "cx", "u": "u3", "U": "u3", "phase": "p", "cphase": "cp", "cu1": "cp", "toffoli": "ccx", "fredkin": "cswap", "i": "id"}


def parse_qasm(text):
    """Parse an OpenQASM 2 or 3 program into a QuantumCircuit.

    Raises:
        QasmError: if a statement can't be read, with the line it is on
    """
    # Remove comments, keeping line breaks so that errors can give their line
    text = re.sub(r"/\*.*?\*/", lambda match: "\n" * match.group(0).count("\n"), text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)

    qregs, cregs = {}, {}
    n_qubits = n_clbits = 0
    gates = []
    measured = set()
    statement_line = 1
    line = 1
    statements = []
    current = ""
    for char in text:
        if char == ";":
            statements.append((statement_line, current.strip()))
            current = ""
            statement_line = line
            continue
        if char == "\n":
            line += 1
            if not current.strip():
                statement_line = line
        current += char
    if current.strip():
        statements.append((statement_line, current.strip()))

    def operands(argument, registers, line, what):
        argument = argument.strip()
        match = re.fullmatch(r"([A-Za-z_][\w]*)\s*(?:\[\s*(\d+)\s*\])?", argument)
        if not match or match.group(1) not in registers:
            raise QasmError(f"line {line}: unknown {what} {argument!r}")
        offset, size = registers[match.group(1)]
        if match.group(2) is None:
            return [offset + i for i in range(size)]
        index = int(match.group(2))
        if index >= size:
            raise QasmError(f"line {line}: {argument} is out of range (the register has {size})")
        return [offset + index]

    for line, statement in statements:
        statement = " ".join(statement.split())
        if not statement:
            continue
        keyword = statement.split(" ")[0]
        if keyword in ("OPENQASM", "include"):
            continue
        if keyword in ("gate", "def", "if", "while", "for", "opaque"):
            raise QasmError(f"line {line}: {keyword} statements aren't supported, so write the circuit with standard gates")
        if keyword == "reset":
            raise QasmError(f"line {line}: resets aren't supported (every qubit starts in |0>)")

        # Declarations
        match = re.fullmatch(r"(qreg|creg)\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]", statement)
        match3 = re.fullmatch(r"(qubit|bit)\s*(?:\[\s*(\d+)\s*\])?\s+([A-Za-z_]\w*)", statement)
        if match or match3:
            kind, name, size = (match.group(1), match.group(2), int(match.group(3))) if match else ("qreg" if match3.group(1) == "qubit" else "creg", match3.group(3), int(match3.group(2) or 1))
            if kind == "qreg":
                qregs[name] = (n_qubits, size)
                n_qubits += size
            else:
                cregs[name] = (n_clbits, size)
                n_clbits += size
            continue

        # Measurements, as "measure q -> c" or "c = measure q"
        match = re.fullmatch(r"measure\s+(.+?)\s*->\s*(.+)", statement) or re.fullmatch(r"(?P<c>.+?)\s*=\s*measure\s+(?P<q>.+)", statement)
        if match:
            if "q" in match.groupdict() and match.groupdict()["q"] is not None:
                qubits, clbits = operands(match.group("q"), qregs, line, "qubit"), operands(match.group("c"), cregs, line, "bit")
            else:
                qubits, clbits = operands(match.group(1), qregs, line, "qubit"), operands(match.group(2), cregs, line, "bit")
            if len(qubits) != len(clbits):
                raise QasmError(f"line {line}: measuring {len(qubits)} qubits into {len(clbits)} bits")
            for q, c in zip(qubits, clbits):
                gates.append(Gate("measure", (q,), (c,)))
                measured.add(q)
            continue

        if keyword == "barrier":
            qubits = []
            for argument in _split_arguments(statement[len("barrier"):]):
                qubits += operands(argument, qregs, line, "qubit")
            gates.append(Gate("barrier", tuple(qubits or range(n_qubits))))
            continue

        # Gates, e.g. "cp(pi/2) q[0], q[1]" or "h q"
        match = re.fullmatch(r"([A-Za-z_]\w*)\s*(?:\((.*?)\))?\s+(.+)", statement)
        if not match:
            raise QasmError(f"line {line}: can't read {statement!r}")
        name = ALIASES.get(match.group(1), match.group(1))
        if name not in GATES:
            raise QasmError(f"line {line}: unknown gate {match.group(1)!r}")
        n_gate_qubits, n_params = GATES[name]
        try:
            params = tuple(_evaluate(p) for p in _split_arguments(match.group(2) or ""))
        except (ValueError, SyntaxError, ZeroDivisionError) as error:
            raise QasmError(f"line {line}: {error}") from None
        if len(params) != n_params:
            raise QasmError(f"line {line}: {name} takes {n_params} parameter{'s' if n_params != 1 else ''}, not {len(params)}")
        arguments = [operands(a, qregs, line, "qubit") for a in _split_arguments(match.group(3))]
        if len(arguments) != n_gate_qubits:
            raise QasmError(f"line {line}: {name} acts on {n_gate_qubits} qubit{'s' if n_gate_qubits != 1 else ''}, not {len(arguments)}")
        # Registers without an index broadcast the gate over the register
        width = max(len(a) for a in arguments)
        if any(len(a) not in (1, width) for a in arguments):
            raise QasmError(f"line {line}: registers of different sizes")
        for i in range(width):
            qubits = tuple(a[i] if len(a) > 1 else a[0] for a in arguments)
            if len(set(qubits)) != len(qubits):
                raise QasmError(f"line {line}: {name} acts on the same qubit twice")
            if measured.intersection(qubits):
                raise QasmError(f"line {line}: mid-circuit measurements aren't supported, so measure every qubit at the end")
            gates.append(Gate(name, qubits, params))

    if n_qubits == 0:
        raise QasmError("the program declares no qubits")
    names = [f"{name}[{i}]" for name, (offset, size) in sorted(qregs.items(), key=lambda item: item[1][0]) for i in range(size)]
    return QuantumCircuit(n_qubits, n_clbits, gates, names)


# ---------------------------------------------------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------------------------------------------------

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _u3(theta, phi, lam):
    return np.array([[np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
                     [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)]])


def _rotation(axis, theta):
    return np.cos(theta / 2) * _I - 1j * np.sin(theta / 2) * axis


def _controlled(u):
    """Controlled gate with the control as the first qubit (the most significant index of the matrix)."""
    matrix = np.eye(2 * len(u), dtype=complex)
    matrix[len(u):, len(u):] = u
    return matrix


def gate_matrix(gate):
    """Unitary of a gate, where the first qubit of the gate is the most significant index of the matrix."""
    name, p = gate.name, gate.params
    single = {
        "id": _I, "x": _X, "y": _Y, "z": _Z, "h": _H, "s": np.diag([1, 1j]), "sdg": np.diag([1, -1j]),
        "t": np.diag([1, np.exp(1j * np.pi / 4)]), "tdg": np.diag([1, np.exp(-1j * np.pi / 4)]),
        "sx": 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]), "sxdg": 0.5 * np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]),
    }
    if name in single:
        return np.asarray(single[name], dtype=complex)
    if name == "rx":
        return _rotation(_X, p[0])
    if name == "ry":
        return _rotation(_Y, p[0])
    if name == "rz":
        return _rotation(_Z, p[0])
    if name in ("p", "u1"):
        return np.diag([1, np.exp(1j * p[0])])
    if name == "u2":
        return _u3(np.pi / 2, p[0], p[1])
    if name == "u3":
        return _u3(*p)
    if name == "r":
        # Rotation by theta about the axis at angle phi in the xy plane, as made by a laser pulse with phase phi
        return _rotation(np.cos(p[1]) * _X + np.sin(p[1]) * _Y, p[0])
    if name == "cx":
        return _controlled(_X)
    if name == "cy":
        return _controlled(_Y)
    if name == "cz":
        return _controlled(_Z)
    if name == "ch":
        return _controlled(_H)
    if name == "crx":
        return _controlled(_rotation(_X, p[0]))
    if name == "cry":
        return _controlled(_rotation(_Y, p[0]))
    if name == "crz":
        return _controlled(_rotation(_Z, p[0]))
    if name == "cp":
        return np.diag([1, 1, 1, np.exp(1j * p[0])])
    if name == "cu3":
        return _controlled(_u3(*p))
    if name == "swap":
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    if name == "iswap":
        return np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex)
    if name in ("rxx", "ms"):
        # The Mølmer-Sørensen gate is an XX rotation, which entangles at theta = pi / 2
        return np.cos(p[0] / 2) * np.eye(4) - 1j * np.sin(p[0] / 2) * np.kron(_X, _X)
    if name == "ryy":
        return np.cos(p[0] / 2) * np.eye(4) - 1j * np.sin(p[0] / 2) * np.kron(_Y, _Y)
    if name == "rzz":
        return np.cos(p[0] / 2) * np.eye(4) - 1j * np.sin(p[0] / 2) * np.kron(_Z, _Z)
    if name == "ccx":
        return _controlled(_controlled(_X))
    if name == "cswap":
        return _controlled(gate_matrix(Gate("swap", (0, 1))))
    raise ValueError(f"no matrix for {name}")


# ---------------------------------------------------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class GateErrors:
    """Errors of the operations of a circuit, as infidelities (1 - average gate fidelity) of single-qubit and two-qubit
    gates (three-qubit gates err like the two-qubit gates they are made of), the probability that a measurement is
    misread, and the probability that an idle qubit dephases in each layer of the circuit.
    """

    single_qubit: float = 0.0
    two_qubit: float = 0.0
    measurement: float = 0.0
    idle: float = 0.0

    @property
    def noiseless(self):
        return self.single_qubit <= 0 and self.two_qubit <= 0 and self.measurement <= 0 and self.idle <= 0


@dataclass
class SimulationResult:
    """Distributions of the outcomes of a circuit without and with errors (outcomes are strings with the first
    classical bit on the right), the fidelity of the final state with and without errors (before measurement), and the
    classical (Hellinger) fidelity of the two distributions.
    """

    outcomes: list
    ideal: np.ndarray
    noisy: np.ndarray
    state_fidelity: float
    distribution_fidelity: float
    n_qubits: int
    method: str

    def top(self, n=8):
        """Most likely outcomes as (outcome, ideal probability, noisy probability)."""
        order = np.argsort(-(self.ideal + self.noisy))[:n]
        return [(self.outcomes[i], self.ideal[i], self.noisy[i]) for i in order]


def _apply_state(state, matrix, qubits, n):
    """Apply a k-qubit unitary to a state vector held as a tensor with one axis per qubit (axis q is qubit q)."""
    k = len(qubits)
    tensor = matrix.reshape((2,) * (2 * k))
    state = np.tensordot(tensor, state, axes=(list(range(k, 2 * k)), list(qubits)))
    return np.moveaxis(state, list(range(k)), list(qubits))


def _apply_density(rho, matrix, qubits, n):
    """Apply a unitary to a density matrix held as a tensor with axes (rows of qubits 0..n-1, columns of qubits 0..n-1)."""
    rho = _apply_state(rho, matrix, qubits, 2 * n)
    return _apply_state(rho, matrix.conj(), [q + n for q in qubits], 2 * n)


def _depolarize(rho, qubits, p, n):
    """Depolarize qubits of a density matrix, rho -> (1 - p) rho + p (their partial trace) x I / 2^k, where the
    completely depolarized state is made one qubit at a time by averaging over the Pauli matrices.
    """
    if p <= 0:
        return rho
    mixed = rho
    for q in qubits:
        mixed = 0.25 * sum(_apply_density(mixed, pauli, [q], n) for pauli in (_I, _X, _Y, _Z))
    return (1 - p) * rho + p * mixed


def _dephase(rho, qubit, p, n):
    """Flip the phase of a qubit with probability p."""
    if p <= 0:
        return rho
    return (1 - p) * rho + p * _apply_density(rho, _Z, [qubit], n)


def _depolarizing_probability(infidelity, k):
    """Strength of a depolarizing channel on k qubits with the given average gate infidelity."""
    d = 2 ** k
    return min(infidelity * (d + 1) / d, 1.0)


def _outcome_distribution(probabilities, n, measurements, readout_error):
    """Distribution of the measured classical bits from the probabilities of the computational basis states (with qubit
    q as bit q of the index), with each measurement misread with probability readout_error.
    """
    if measurements:
        qubits = sorted(measurements, key=lambda q: measurements[q])
    else:
        qubits = list(range(n))
    m = len(qubits)
    # Bit q of the index is qubit q, so the reshaped axes run from qubit n - 1 down to qubit 0
    tensor = np.transpose(probabilities.reshape((2,) * n), list(range(n))[::-1])
    unmeasured = tuple(q for q in range(n) if q not in qubits)
    marginal = tensor.sum(axis=unmeasured) if unmeasured else tensor
    kept = [q for q in range(n) if q in qubits]
    marginal = np.transpose(marginal, [kept.index(q) for q in qubits])
    if readout_error > 0:
        flip = np.array([[1 - readout_error, readout_error], [readout_error, 1 - readout_error]])
        for axis in range(m):
            marginal = np.moveaxis(np.tensordot(flip, marginal, axes=([1], [axis])), 0, axis)
    # Outcome strings have the first measured bit on the right
    flat = np.transpose(marginal, list(range(m))[::-1]).reshape(-1)
    outcomes = [format(i, f"0{m}b") for i in range(2 ** m)]
    return outcomes, flat


def simulate(circuit, errors=None, n_trajectories=400, seed=0):
    """Simulate a circuit and the distribution of its measured outcomes, with gate errors if any are given.

    Circuits of up to MAX_DENSITY_QUBITS qubits with errors are simulated with their density matrix, and larger ones
    with n_trajectories state vectors which each get random Pauli errors.

    Returns:
        SimulationResult
    """
    errors = errors or GateErrors()
    n = circuit.n_qubits
    if n > MAX_STATE_QUBITS:
        raise ValueError(f"circuits of up to {MAX_STATE_QUBITS} qubits can be simulated, not {n}")
    measurements = circuit.measurements

    # Ideal state vector
    state = np.zeros((2,) * n, dtype=complex)
    state[(0,) * n] = 1.0
    for gate in circuit.operations:
        state = _apply_state(state, gate_matrix(gate), gate.qubits, n)
    # Index the probabilities with qubit q as bit q
    ideal_probabilities = np.abs(np.transpose(state, list(range(n))[::-1]).reshape(-1)) ** 2
    outcomes, ideal = _outcome_distribution(ideal_probabilities, n, measurements, 0.0)

    if errors.noiseless:
        return SimulationResult(outcomes, ideal, ideal.copy(), 1.0, 1.0, n, "state vector")

    layers = circuit.layers()

    def gate_error(gate):
        k = len(gate.qubits)
        if k == 1:
            return _depolarizing_probability(errors.single_qubit, 1)
        # Three-qubit gates are made of several two-qubit gates
        n_two = MS_PER_GATE.get(gate.name, 1)
        return _depolarizing_probability(1 - (1 - errors.two_qubit) ** n_two, k) if k > 2 else _depolarizing_probability(1 - (1 - errors.two_qubit) ** n_two, 2)

    if n <= MAX_DENSITY_QUBITS:
        rho = np.zeros((2,) * (2 * n), dtype=complex)
        rho[(0,) * (2 * n)] = 1.0
        for layer in layers:
            busy = set()
            for gate in layer:
                if gate.name == "measure":
                    continue
                rho = _apply_density(rho, gate_matrix(gate), gate.qubits, n)
                rho = _depolarize(rho, list(gate.qubits), gate_error(gate), n)
                busy.update(gate.qubits)
            for q in range(n):
                if q not in busy:
                    rho = _dephase(rho, q, errors.idle, n)
        matrix = np.transpose(rho, list(range(n))[::-1] + list(range(2 * n - 1, n - 1, -1))).reshape(2 ** n, 2 ** n)
        probabilities = np.real(np.diag(matrix))
        psi = np.transpose(state, list(range(n))[::-1]).reshape(-1)
        state_fidelity = float(np.real(np.conj(psi) @ matrix @ psi))
        method = "density matrix"
    else:
        rng = np.random.default_rng(seed)
        paulis = [_I, _X, _Y, _Z]
        probabilities = np.zeros(2 ** n)
        psi = np.transpose(state, list(range(n))[::-1]).reshape(-1)
        state_fidelity = 0.0
        for _ in range(n_trajectories):
            trajectory = np.zeros((2,) * n, dtype=complex)
            trajectory[(0,) * n] = 1.0
            for layer in layers:
                busy = set()
                for gate in layer:
                    if gate.name == "measure":
                        continue
                    trajectory = _apply_state(trajectory, gate_matrix(gate), gate.qubits, n)
                    k = len(gate.qubits)
                    if rng.random() < gate_error(gate) * (4 ** k - 1) / 4 ** k:
                        choice = rng.integers(1, 4 ** k)
                        for i, q in enumerate(gate.qubits):
                            trajectory = _apply_state(trajectory, paulis[(choice >> (2 * i)) & 3], [q], n)
                    busy.update(gate.qubits)
                for q in range(n):
                    if q not in busy and rng.random() < errors.idle:
                        trajectory = _apply_state(trajectory, _Z, [q], n)
            amplitudes = np.transpose(trajectory, list(range(n))[::-1]).reshape(-1)
            probabilities += np.abs(amplitudes) ** 2 / n_trajectories
            state_fidelity += abs(np.vdot(psi, amplitudes)) ** 2 / n_trajectories
        method = f"{n_trajectories} trajectories"

    _, noisy = _outcome_distribution(probabilities, n, measurements, errors.measurement)
    distribution_fidelity = float(np.sum(np.sqrt(np.clip(ideal, 0, None) * np.clip(noisy, 0, None))) ** 2)
    return SimulationResult(outcomes, ideal, noisy, state_fidelity, distribution_fidelity, n, method)


# ---------------------------------------------------------------------------------------------------------------------
# Native gates and scheduling
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class NativeCounts:
    """Native gates of trapped ions which a circuit compiles into, i.e. Mølmer-Sørensen gates, single-qubit rotations
    made by laser pulses (with consecutive gates on a qubit merged into one), virtual Z rotations, and measurements.
    """

    ms_gates: int
    rotations: int
    virtual: int
    measurements: int
    n_qubits: int

    @property
    def two_qubit_gates(self):
        return self.ms_gates


def native_counts(circuit):
    """Count the native gates a circuit compiles into (see NativeCounts)."""
    ms = rotations = virtual = 0
    pending = [False] * circuit.n_qubits
    for gate in circuit.gates:
        if gate.name in ("barrier", "measure"):
            continue
        if len(gate.qubits) == 1:
            if gate.name in VIRTUAL_GATES:
                virtual += 1
            else:
                # Single-qubit gates in a row merge into one rotation (and virtual Z rotations)
                pending[gate.qubits[0]] = True
            continue
        for q in gate.qubits:
            if pending[q]:
                rotations += 1
                pending[q] = False
        n_ms = MS_PER_GATE.get(gate.name, 1)
        ms += n_ms
        if gate.name not in ("rxx", "ms"):
            rotations += ROTATIONS_PER_MS * n_ms
    rotations += sum(pending)
    measured = len(circuit.measurements) or circuit.n_qubits
    return NativeCounts(ms, rotations, virtual, measured, circuit.n_qubits)


def native_gates(circuit):
    """List the native gates a circuit compiles into, one line per gate, e.g. "MS q0, q1" or "R q2"."""
    lines = []
    pending = {}
    for gate in circuit.gates:
        if gate.name == "barrier":
            continue
        if gate.name == "measure":
            q = gate.qubits[0]
            if q in pending:
                lines.append(f"R q{q}  ({pending.pop(q)})")
            lines.append(f"Measure q{q}")
            continue
        if len(gate.qubits) == 1:
            if gate.name in VIRTUAL_GATES:
                lines.append(f"Z q{gate.qubits[0]}  ({gate.name}, virtual)")
            else:
                q = gate.qubits[0]
                pending[q] = f"{pending[q]} + {gate.name}" if q in pending else gate.name
            continue
        for q in gate.qubits:
            if q in pending:
                lines.append(f"R q{q}  ({pending.pop(q)})")
        n_ms = MS_PER_GATE.get(gate.name, 1)
        qubits = ", ".join(f"q{q}" for q in gate.qubits)
        if gate.name in ("rxx", "ms"):
            lines.append(f"MS q{gate.qubits[0]}, q{gate.qubits[1]}  ({gate.name})")
        else:
            lines.append(f"{n_ms} x (MS + {ROTATIONS_PER_MS} R) on {qubits}  ({gate.name})")
    for q, names in pending.items():
        lines.append(f"R q{q}  ({names})")
    return lines


def to_scheduler_qasm(circuit):
    """Write a circuit for the C scheduler, with every physical single-qubit rotation as x, every two-qubit gate as cx,
    three-qubit gates as the two-qubit gates they are made of, and every qubit measured at the end if the circuit has no
    measurements.
    """
    lines = [f"qubit[{circuit.n_qubits}] qreg;", f"bit[{circuit.n_qubits}] creg;", ""]
    pending = set()
    for gate in circuit.gates:
        if gate.name == "barrier":
            continue
        if gate.name == "measure":
            q = gate.qubits[0]
            if q in pending:
                lines.append(f"x qreg[{q}];")
                pending.discard(q)
            lines.append(f"measure qreg[{q}] -> creg[{q}];")
            continue
        if len(gate.qubits) == 1:
            if gate.name not in VIRTUAL_GATES:
                pending.add(gate.qubits[0])
            continue
        for q in gate.qubits:
            if q in pending:
                lines.append(f"x qreg[{q}];")
                pending.discard(q)
        if len(gate.qubits) == 2:
            for _ in range(MS_PER_GATE.get(gate.name, 1) if gate.name in ("swap", "iswap") else 1):
                lines.append(f"cx qreg[{gate.qubits[0]}], qreg[{gate.qubits[1]}];")
        else:
            # Three-qubit gates as CNOTs between each pair, as in their usual decompositions
            a, b, c = gate.qubits
            for x, y in ((b, c), (a, c), (b, c), (a, c), (a, b), (a, b)) if gate.name == "ccx" else ((c, b), (b, c), (a, b), (a, c), (b, c), (a, c), (a, b), (c, b)):
                lines.append(f"cx qreg[{x}], qreg[{y}];")
    for q in sorted(pending):
        lines.append(f"x qreg[{q}];")
    if not circuit.measurements:
        lines.append("measure qreg -> creg;")
    return "\n".join(lines) + "\n"


def write_scheduler_qasm(circuit, path):
    """Write to_scheduler_qasm(circuit) to a file, which lib.Circuit can read."""
    with open(path, "w") as file:
        file.write(to_scheduler_qasm(circuit))


def load(path):
    """Parse an OpenQASM file into a QuantumCircuit."""
    with open(path) as file:
        return parse_qasm(file.read())
