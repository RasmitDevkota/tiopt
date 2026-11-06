"""Generate the circuits which schedule_benchmark compares the transport schedulers on, into benchmarks/ (run from this
directory). Each family is made at several sizes, so that results can be checked for consistency across circuits:

- GHZ and Bernstein-Vazirani: sanity checks
- QFT: dense, with every pair of qubits interacting
- QAOA MaxCut on random 3-regular graphs: structure set by the graph
- Trotterized transverse-field Ising and Heisenberg chains: regular layers
- Cuccaro ripple-carry adder: deep and serial (Toffolis decomposed into CNOTs and T gates)

Only which qubits each gate acts on matters to the scheduler, so angles are placeholders.
"""

import math
import os
import random

SIZES = [6, 12, 16]
OUTPUT_DIR = "benchmarks"


class Circuit:
    def __init__(self, n_qubits, description):
        self.n_qubits = n_qubits
        self.lines = [f"// {description}", f"qubit[{n_qubits}] qreg;", f"bit[{n_qubits}] creg;", ""]

    def gate(self, name, *qubits):
        self.lines.append(f"{name} " + ", ".join(f"qreg[{q}]" for q in qubits) + ";")

    def ccx(self, a, b, c):
        # Standard decomposition into 6 CNOTs
        for name, qubits in [("h", (c,)), ("cx", (b, c)), ("tdg", (c,)), ("cx", (a, c)), ("t", (c,)), ("cx", (b, c)),
                             ("tdg", (c,)), ("cx", (a, c)), ("t", (b,)), ("t", (c,)), ("h", (c,)), ("cx", (a, b)),
                             ("t", (a,)), ("tdg", (b,)), ("cx", (a, b))]:
            self.gate(name, *qubits)

    def write(self, name):
        self.lines += ["", "measure qreg -> creg;", ""]
        with open(os.path.join(OUTPUT_DIR, f"{name}.qasm"), "w") as f:
            f.write("\n".join(self.lines))


def ghz(n):
    c = Circuit(n, f"GHZ state on {n} qubits")
    c.gate("h", 0)
    for q in range(n - 1):
        c.gate("cx", q, q + 1)
    return c


def bernstein_vazirani(n, rng):
    secret = [rng.randint(0, 1) for _ in range(n - 1)]
    secret[0] = 1
    c = Circuit(n, f"Bernstein-Vazirani with secret {''.join(map(str, secret))} and an ancilla, on {n} qubits")
    ancilla = n - 1
    c.gate("x", ancilla)
    for q in range(n):
        c.gate("h", q)
    for q, bit in enumerate(secret):
        if bit:
            c.gate("cx", q, ancilla)
    for q in range(n - 1):
        c.gate("h", q)
    return c


def qft(n):
    c = Circuit(n, f"Quantum Fourier transform on {n} qubits")
    for j in range(n):
        c.gate("h", j)
        for k in range(j + 1, n):
            c.gate(f"cp({math.pi / 2 ** (k - j):.6f})", k, j)
    for q in range(n // 2):
        c.gate("swap", q, n - 1 - q)
    return c


def random_regular_graph(n, degree, rng):
    while True:
        stubs = [q for q in range(n) for _ in range(degree)]
        rng.shuffle(stubs)
        edges = {tuple(sorted(stubs[i:i + 2])) for i in range(0, len(stubs), 2)}
        if len(edges) == len(stubs) // 2 and all(a != b for a, b in edges):
            return sorted(edges)


def qaoa_maxcut(n, rng, p=2):
    edges = random_regular_graph(n, 3, rng)
    c = Circuit(n, f"QAOA MaxCut with p = {p} on a random 3-regular graph of {n} vertices")
    for q in range(n):
        c.gate("h", q)
    for _ in range(p):
        for a, b in edges:
            c.gate("rzz(0.5)", a, b)
        for q in range(n):
            c.gate("rx(0.3)", q)
    return c


def ising(n, steps=4):
    c = Circuit(n, f"Trotterized transverse-field Ising chain of {n} spins, {steps} steps")
    for _ in range(steps):
        for start in (0, 1):
            for q in range(start, n - 1, 2):
                c.gate("rzz(0.2)", q, q + 1)
        for q in range(n):
            c.gate("rx(0.1)", q)
    return c


def heisenberg(n, steps=2):
    c = Circuit(n, f"Trotterized Heisenberg chain of {n} spins, {steps} steps")
    for _ in range(steps):
        for start in (0, 1):
            for q in range(start, n - 1, 2):
                for name in ("rxx(0.2)", "ryy(0.2)", "rzz(0.2)"):
                    c.gate(name, q, q + 1)
    return c


def cuccaro_adder(n_bits, rng):
    # Qubits: carry in, then b_i and a_i for each bit, then the carry out
    n = 2 * n_bits + 2
    c = Circuit(n, f"Cuccaro ripple-carry adder of two {n_bits}-bit numbers, on {n} qubits")
    carry, out = 0, n - 1
    b = [1 + 2 * i for i in range(n_bits)]
    a = [2 + 2 * i for i in range(n_bits)]

    for q in a + b:
        if rng.random() < 0.5:
            c.gate("x", q)

    def maj(x, y, z):
        c.gate("cx", z, y)
        c.gate("cx", z, x)
        c.ccx(x, y, z)

    def uma(x, y, z):
        c.ccx(x, y, z)
        c.gate("cx", z, x)
        c.gate("cx", x, y)

    maj(carry, b[0], a[0])
    for i in range(1, n_bits):
        maj(a[i - 1], b[i], a[i])
    c.gate("cx", a[-1], out)
    for i in reversed(range(1, n_bits)):
        uma(a[i - 1], b[i], a[i])
    uma(carry, b[0], a[0])
    return c


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = random.Random(2026)

    for n in SIZES:
        ghz(n).write(f"ghz_{n:02d}")
        bernstein_vazirani(n, rng).write(f"bv_{n:02d}")
        qft(n).write(f"qft_{n:02d}")
        qaoa_maxcut(n, rng).write(f"qaoa_{n:02d}")
        ising(n).write(f"ising_{n:02d}")
        heisenberg(n).write(f"heisenberg_{n:02d}")
        cuccaro_adder((n - 2) // 2, rng).write(f"cuccaro_{n:02d}")
