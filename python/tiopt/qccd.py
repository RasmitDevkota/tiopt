"""Compilation and simulation of circuits on quantum charge-coupled device (QCCD) architectures, i.e. traps which hold
chains of ions, connected by shuttling segments and junctions, like QCCDSim (P. Murali et al., ISCA 2020) and the MQT
IonShuttler.

An architecture is a graph of traps (each holding a chain of up to its capacity of ions, with two ends) and junctions,
joined by segments. The compiler places the qubits greedily in order of their first use (leaving room in every trap for
incoming ions), and then schedules every gate of the circuit as soon as its ions and the resources it needs are free:
when the two ions of a gate are in different traps, one of them is brought to the trap of the other along the shortest
path, which takes reordering it to the end of its chain (with SWAP gates, made of three Mølmer-Sørensen gates, or by
physically swapping ions, which splits, rotates, and merges them), splitting it off, moving it through every segment
and junction on the way, and merging it into the other chain. Every segment and junction holds one ion at a time, so
transports which share them wait for each other.

Every chain carries motional energy, which splits share in proportion to their ions, every operation adds to (a split,
a merge, a junction crossing, and every segment), and background heating raises over time. Each two-qubit gate takes
the time of its gate model (amplitude-, phase-, or frequency-modulated gates, whose durations depend on the length of
the chain and on how far apart the ions are) and has the fidelity 1 - Gamma tau - A (2 n_bar + 1), where A grows as
N / ln N with the number of ions N in the chain (the model of Murali et al.), so the fidelity of the whole program is
the product of the fidelities of its gates.
"""

import heapq
from dataclasses import dataclass, field

import numpy as np

from .gates import GATE_MODELS, gate_time_model


@dataclass
class Node:
    """Trap or junction of an architecture, with its capacity (for traps), position (µm, for drawing), and whether
    two-qubit gates can be done in it.
    """
    name: str
    kind: str
    capacity: int = 0
    position: tuple = (0.0, 0.0)
    gate_zone: bool = True


class Architecture:
    """Graph of traps and junctions joined by shuttling segments."""

    def __init__(self):
        self.nodes = []
        self.neighbors = {}
        self.lengths = {}

    def add_trap(self, name, capacity, position=(0.0, 0.0), gate_zone=True):
        self.nodes.append(Node(name, "trap", int(capacity), tuple(position), gate_zone))
        self.neighbors[len(self.nodes) - 1] = []
        return len(self.nodes) - 1

    def add_junction(self, name, position=(0.0, 0.0)):
        self.nodes.append(Node(name, "junction", 0, tuple(position), False))
        self.neighbors[len(self.nodes) - 1] = []
        return len(self.nodes) - 1

    def connect(self, a, b, length=100.0):
        """Join two nodes with a segment of a length (µm)."""
        for node in (a, b):
            if self.nodes[node].kind == "trap" and len(self.neighbors[node]) >= 2:
                raise ValueError(f"trap {self.nodes[node].name} already has two ends")
        self.neighbors[a].append(b)
        self.neighbors[b].append(a)
        self.lengths[(min(a, b), max(a, b))] = float(length)

    @property
    def traps(self):
        return [i for i, node in enumerate(self.nodes) if node.kind == "trap"]

    @property
    def junctions(self):
        return [i for i, node in enumerate(self.nodes) if node.kind == "junction"]

    def length(self, a, b):
        return self.lengths[(min(a, b), max(a, b))]

    def gate_traps(self):
        traps = [t for t in self.traps if self.nodes[t].gate_zone]
        return traps or self.traps

    def path(self, start, end, timing):
        """Fastest path between two nodes (as a list of nodes) for an ion, where passing through another trap costs a
        merge into its chain, a reordering to its far end, and a split.
        """
        best = {start: 0.0}
        previous = {}
        queue = [(0.0, start)]
        while queue:
            cost, node = heapq.heappop(queue)
            if node == end:
                break
            if cost > best.get(node, np.inf):
                continue
            for neighbor in self.neighbors[node]:
                step = timing.move_time(self.length(node, neighbor))
                if self.nodes[neighbor].kind == "junction":
                    step += timing.junction_time(len(self.neighbors[neighbor]))
                elif neighbor != end:
                    step += timing.merge + timing.split + timing.two_qubit_time(max(self.nodes[neighbor].capacity, 2), 1) * 3
                if cost + step < best.get(neighbor, np.inf):
                    best[neighbor] = cost + step
                    previous[neighbor] = node
                    heapq.heappush(queue, (cost + step, neighbor))
        if end not in best:
            return None
        nodes = [end]
        while nodes[-1] != start:
            nodes.append(previous[nodes[-1]])
        return nodes[::-1]

    # Topologies ----------------------------------------------------------------------------------------------------

    @classmethod
    def linear(cls, n_traps, capacity, spacing=100.0, junctions=True):
        """Traps in a line (L-n of Murali et al.), joined by segments through 2-way junctions (or directly)."""
        architecture = cls()
        previous = None
        for k in range(n_traps):
            trap = architecture.add_trap(f"T{k}", capacity, (k * 2 * spacing, 0.0))
            if previous is not None:
                if junctions:
                    junction = architecture.add_junction(f"J{k - 1}", ((k - 0.5) * 2 * spacing, 0.0))
                    architecture.connect(previous, junction, spacing / 2)
                    architecture.connect(junction, trap, spacing / 2)
                else:
                    architecture.connect(previous, trap, spacing)
            previous = trap
        return architecture

    @classmethod
    def grid(cls, rows, cols, capacity, spacing=200.0):
        """Junctions on a grid of rows x cols points, with a trap on every edge between neighboring junctions (so that
        every junction in the interior is an X junction and every one on the edges a T junction).
        """
        architecture = cls()
        junction = {}
        for r in range(rows):
            for c in range(cols):
                junction[r, c] = architecture.add_junction(f"J{r}{c}", (c * spacing, -r * spacing))
        for r in range(rows):
            for c in range(cols):
                for dr, dc in ((0, 1), (1, 0)):
                    if r + dr < rows and c + dc < cols:
                        a, b = junction[r, c], junction[r + dr, c + dc]
                        position = ((c + dc / 2) * spacing, -(r + dr / 2) * spacing)
                        trap = architecture.add_trap(f"T{r}{c}{'h' if dc else 'v'}", capacity, position)
                        architecture.connect(a, trap, spacing / 2)
                        architecture.connect(trap, b, spacing / 2)
        return architecture

    @classmethod
    def ring(cls, n_traps, capacity, spacing=100.0):
        """Traps around a closed loop (like a racetrack), joined through 2-way junctions."""
        architecture = cls()
        traps = []
        for k in range(n_traps):
            angle = 2 * np.pi * k / n_traps
            traps.append(architecture.add_trap(f"T{k}", capacity, (n_traps * spacing / np.pi * np.cos(angle), n_traps * spacing / np.pi * np.sin(angle))))
        for k in range(n_traps):
            a, b = traps[k], traps[(k + 1) % n_traps]
            mid = 0.5 * (np.array(architecture.nodes[a].position) + np.array(architecture.nodes[b].position))
            junction = architecture.add_junction(f"J{k}", tuple(mid))
            architecture.connect(a, junction, spacing / 2)
            architecture.connect(junction, b, spacing / 2)
        return architecture

    @classmethod
    def from_zones(cls, zones, default_capacity=4):
        """Architecture of the zones of a trap (tiopt Zone objects): every zone is a trap (where the zones with the
        gate type can do two-qubit gates) except junction zones, which are junctions, and the zones are joined along
        the minimum spanning tree of their positions.
        """
        from scipy.sparse.csgraph import minimum_spanning_tree

        architecture = cls()
        indices = []
        for k, zone in enumerate(zones):
            types = zone.types
            position = (float(zone.position[0]), float(zone.position[1]))
            if "junction" in types:
                indices.append(architecture.add_junction(f"J{k}", position))
            else:
                capacity = zone.capacity if zone.capacity and zone.capacity < 1000 else default_capacity
                indices.append(architecture.add_trap(f"Z{k} ({types})", capacity, position, "gate" in types))
        if len(zones) > 1:
            points = np.array([zone.position[:2] for zone in zones], dtype=float)
            distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1) + 1e-9
            tree = minimum_spanning_tree(distances).tocoo()
            # Traps have two ends, so each needs a junction between it and any third neighbor
            for a, b, d in sorted(zip(tree.row, tree.col, tree.data), key=lambda e: e[2]):
                u, v = indices[a], indices[b]
                try:
                    architecture.connect(u, v, d)
                except ValueError:
                    junction = architecture.add_junction(f"J{len(architecture.nodes)}", tuple(0.5 * (points[a] + points[b])))
                    full = u if len(architecture.neighbors[u]) >= 2 else v
                    other = v if full == u else u
                    # Reroute one existing neighbor of the full trap through the new junction
                    moved = architecture.neighbors[full][-1]
                    length = architecture.length(full, moved)
                    architecture.neighbors[full].remove(moved)
                    architecture.neighbors[moved].remove(full)
                    del architecture.lengths[(min(full, moved), max(full, moved))]
                    architecture.connect(full, junction, 1.0)
                    architecture.connect(junction, moved, length)
                    architecture.connect(junction, other, d)
        return architecture

    @classmethod
    def from_trap(cls, trap, default_capacity=4):
        """Architecture of the zones of a tiopt Trap (see from_zones)."""
        return cls.from_zones(trap.zones(), default_capacity)


@dataclass
class TimingModel:
    """Durations (µs) of the operations of a QCCD architecture, with the defaults of Murali et al.: a split or merge
    takes 80 µs, crossing a Y or X junction 100 or 120 µs, and moving at speed (µm/µs, 20 by default) takes 5 µs per
    100 µm segment. Two-qubit gates follow a gate model of tiopt.gates.gate_time_model, and reordering uses SWAP gates
    ("gate") or physical ion swaps ("ion", a split, a rotation, and a merge per position).
    """
    speed: float = 20.0
    min_move: float = 1.0
    split: float = 80.0
    merge: float = 80.0
    junction_y: float = 100.0
    junction_x: float = 120.0
    rotation: float = 42.0
    single_qubit: float = 5.0
    measurement: float = 100.0
    gate_model: str = "FM"
    swap_type: str = "gate"

    def move_time(self, length):
        return max(length / self.speed, self.min_move)

    def junction_time(self, degree):
        if degree <= 2:
            return 0.0
        return self.junction_y if degree == 3 else self.junction_x

    def two_qubit_time(self, n_ions, separation):
        return gate_time_model(self.gate_model, n_ions, separation)


@dataclass
class NoiseModel:
    """Heating and errors of a QCCD architecture: the motional quanta added to a chain by a split or merge, by every
    junction crossing, and by every 100 µm moved, background heating (quanta/µs), the background error of two-qubit
    gates (per µs), the coefficient A0 of their thermal error A = A0 N / ln N, the errors of single-qubit gates and
    measurements, and optional recooling of every chain after a merge (with its duration in µs and the mean phonon
    number it reaches).
    """
    split_heating: float = 0.1
    merge_heating: float = 0.1
    junction_heating: float = 0.1
    move_heating: float = 0.01
    background_heating: float = 1e-4
    gate_background_error: float = 1e-6
    motional_error: float = 1e-4
    single_qubit_error: float = 1e-4
    measurement_error: float = 1e-3
    recool: bool = False
    cooling_time: float = 200.0
    cooled_n_bar: float = 0.05


@dataclass
class Operation:
    """Operation of a compiled program: its kind, start and end (µs), the nodes it occupies, the qubits it acts on, and
    its fidelity.
    """
    kind: str
    start: float
    end: float
    nodes: tuple
    qubits: tuple
    fidelity: float = 1.0
    label: str = ""


@dataclass
class CompileResult:
    """Result of compiling a circuit on an architecture: every operation in order, the duration of the program (µs),
    its estimated fidelity, the largest mean phonon number of any chain at a gate, how many operations of each kind it
    took, and where every qubit started and ended.
    """
    operations: list
    duration: float
    fidelity: float
    max_n_bar: float
    counts: dict
    initial_mapping: dict
    final_mapping: dict
    architecture: Architecture = field(repr=False, default=None)

    def summary(self):
        """Lines of text which describe the result."""
        counts = ", ".join(f"{n} {kind}" for kind, n in sorted(self.counts.items()) if n)
        return [
            f"Duration {self.duration / 1e3:.3f} ms, estimated fidelity {self.fidelity:.4f}",
            f"Largest mean phonon number at a gate: {self.max_n_bar:.3f}",
            f"Operations: {counts}",
        ]


def _pairs(gate, n_ms):
    """Pairs of qubits of the Mølmer-Sørensen gates which a gate on two or three qubits takes."""
    q = gate.qubits
    if len(q) == 2:
        return [tuple(q)] * n_ms
    # Three-qubit gates are made of gates on the pairs among their qubits
    pairs = [(q[0], q[2]), (q[1], q[2]), (q[0], q[1])]
    return [pairs[k % 3] for k in range(n_ms)]


class _Compiler:
    def __init__(self, architecture, timing, noise, n_qubits):
        self.arch = architecture
        self.timing = timing
        self.noise = noise
        self.chains = {t: [] for t in architecture.traps}
        self.location = {}
        self.energy = {t: 0.0 for t in architecture.traps}
        self.energy_time = {t: 0.0 for t in architecture.traps}
        self.free = {node: 0.0 for node in range(len(architecture.nodes))}
        self.edge_free = {edge: 0.0 for edge in architecture.lengths}
        self.qubit_free = {q: 0.0 for q in range(n_qubits)}
        self.incoming = {t: 0 for t in architecture.traps}
        self.operations = []
        self.counts = {}
        self.fidelity = 1.0
        self.max_n_bar = 0.0

    def count(self, kind, n=1):
        self.counts[kind] = self.counts.get(kind, 0) + n

    def heat(self, trap, t):
        """Background heating of the chain in a trap up to time t."""
        self.energy[trap] += self.noise.background_heating * max(t - self.energy_time[trap], 0.0)
        self.energy_time[trap] = max(t, self.energy_time[trap])

    def record(self, kind, start, duration, nodes, qubits, fidelity=1.0, label=""):
        self.operations.append(Operation(kind, start, start + duration, tuple(nodes), tuple(qubits), fidelity, label))
        self.fidelity *= fidelity
        self.count(kind)
        return start + duration

    def place(self, circuit):
        """Place the qubits greedily in order of first use, filling traps in order of a walk through the architecture
        and leaving room for two incoming ions in each.
        """
        order = []
        for gate in circuit.gates:
            for q in gate.qubits:
                if q not in order:
                    order.append(q)
        order += [q for q in range(circuit.n_qubits) if q not in order]
        # Visit the traps breadth first from the first gate trap
        start = self.arch.gate_traps()[0]
        seen, queue, walk = {start}, [start], []
        while queue:
            node = queue.pop(0)
            if self.arch.nodes[node].kind == "trap":
                walk.append(node)
            for neighbor in self.arch.neighbors[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        walk += [t for t in self.arch.traps if t not in walk]
        room = sum(max(self.arch.nodes[t].capacity - 2, 1) for t in walk)
        if room < len(order):
            if sum(self.arch.nodes[t].capacity for t in walk) < len(order):
                raise ValueError(f"the architecture holds {sum(self.arch.nodes[t].capacity for t in walk)} ions, fewer than the {len(order)} qubits")
        k = 0
        for trap in walk:
            limit = max(self.arch.nodes[trap].capacity - 2, 1) if room >= len(order) else self.arch.nodes[trap].capacity
            while k < len(order) and len(self.chains[trap]) < limit:
                self.chains[trap].append(order[k])
                self.location[order[k]] = trap
                k += 1

    def room(self, trap):
        return self.arch.nodes[trap].capacity - len(self.chains[trap]) - self.incoming[trap]

    def gate_time(self, trap, a, b):
        chain = self.chains[trap]
        return self.timing.two_qubit_time(len(chain), abs(chain.index(a) - chain.index(b)))

    def two_qubit_fidelity(self, trap, duration):
        n = max(len(self.chains[trap]), 2)
        A = self.noise.motional_error * n / np.log(n)
        n_bar = self.energy[trap]
        self.max_n_bar = max(self.max_n_bar, n_bar)
        return max(1 - self.noise.gate_background_error * duration - A * (2 * n_bar + 1), 0.0)

    def gate(self, a, b, label):
        trap = self.location[a]
        start = max(self.free[trap], self.qubit_free[a], self.qubit_free[b])
        self.heat(trap, start)
        duration = self.gate_time(trap, a, b)
        end = self.record("two-qubit gate", start, duration, [trap], [a, b], self.two_qubit_fidelity(trap, duration), label)
        self.free[trap] = self.qubit_free[a] = self.qubit_free[b] = end

    def single(self, q, label, kind="single-qubit gate", duration=None, error=None):
        trap = self.location[q]
        start = max(self.free[trap], self.qubit_free[q])
        duration = self.timing.single_qubit if duration is None else duration
        error = self.noise.single_qubit_error if error is None else error
        end = self.record(kind, start, duration, [trap], [q], 1 - error, label)
        self.free[trap] = self.qubit_free[q] = end

    def to_end(self, q, end_index):
        """Bring qubit q to an end of its chain (0 for the front and -1 for the back)."""
        trap = self.location[q]
        chain = self.chains[trap]
        position = chain.index(q)
        target = 0 if end_index == 0 else len(chain) - 1
        if position == target:
            return
        start = max(self.free[trap], self.qubit_free[q])
        self.heat(trap, start)
        if self.timing.swap_type == "gate":
            # A SWAP gate (three MS gates) exchanges the states of q and the ion at the end
            other = chain[target]
            duration = 3 * self.timing.two_qubit_time(len(chain), abs(target - position))
            fidelity = self.two_qubit_fidelity(trap, duration / 3) ** 3
            end = self.record("swap gate", start, duration, [trap], [q, other], fidelity, f"swap q{q} q{other}")
            chain[position], chain[target] = chain[target], chain[position]
            self.qubit_free[other] = end
        else:
            # Physical swaps with each neighbor on the way (a split, a rotation, and a merge each)
            steps = abs(target - position)
            duration = steps * (self.timing.split + self.timing.rotation + self.timing.merge)
            end = self.record("ion swap", start, duration, [trap], [q], 1.0, f"{steps} ion swaps of q{q}")
            self.count("ion swap", steps - 1)
            self.energy[trap] += steps * (self.noise.split_heating + self.noise.merge_heating)
            chain.remove(q)
            chain.insert(target, q)
        self.free[trap] = self.qubit_free[q] = end

    def end_facing(self, trap, neighbor):
        """End of the chain of a trap (0 for the front and -1 for the back) which faces one of its neighbors."""
        return 0 if self.arch.neighbors[trap].index(neighbor) == 0 else -1

    def split_off(self, q, toward):
        """Bring q to the end of its chain which faces a neighbor and split it off, returning the time it is free and
        the motional energy it carries.
        """
        trap = self.location[q]
        self.to_end(q, self.end_facing(trap, toward))
        chain = self.chains[trap]
        t = max(self.free[trap], self.qubit_free[q])
        self.heat(trap, t)
        share = self.energy[trap] / max(len(chain), 1)
        t = self.record("split", t, self.timing.split, [trap], [q], 1.0, f"split q{q} from {self.arch.nodes[trap].name}")
        chain.remove(q)
        self.energy[trap] = self.energy[trap] - share + self.noise.split_heating
        self.free[trap] = t
        return t, share + self.noise.split_heating

    def merge_into(self, q, trap, coming_from, t, energy):
        """Merge q into the end of the chain of a trap which faces where it came from."""
        t = max(t, self.free[trap])
        self.heat(trap, t)
        t = self.record("merge", t, self.timing.merge, [trap], [q], 1.0, f"merge q{q} into {self.arch.nodes[trap].name}")
        if self.end_facing(trap, coming_from) == 0:
            self.chains[trap].insert(0, q)
        else:
            self.chains[trap].append(q)
        self.energy[trap] += energy + self.noise.merge_heating
        self.location[q] = trap
        self.free[trap] = self.qubit_free[q] = t
        return t

    def move(self, q, destination):
        """Shuttle qubit q to a trap along the fastest path, merging into and splitting out of any trap on the way."""
        source = self.location[q]
        path = self.arch.path(source, destination, self.timing)
        if path is None:
            raise ValueError(f"no path from {self.arch.nodes[source].name} to {self.arch.nodes[destination].name}")
        self.incoming[destination] += 1
        t, energy = self.split_off(q, path[1])
        for k in range(1, len(path)):
            a, b = path[k - 1], path[k]
            edge = (min(a, b), max(a, b))
            t = max(t, self.edge_free[edge])
            length = self.arch.length(a, b)
            t = self.record("move", t, self.timing.move_time(length), [a, b], [q], 1.0, f"q{q} {self.arch.nodes[a].name}->{self.arch.nodes[b].name}")
            self.edge_free[edge] = t
            energy += self.noise.move_heating * length / 100.0 + self.noise.background_heating * self.timing.move_time(length)
            if self.arch.nodes[b].kind == "junction":
                if len(self.arch.neighbors[b]) > 2:
                    t = max(t, self.free[b])
                    t = self.record("junction crossing", t, self.timing.junction_time(len(self.arch.neighbors[b])), [b], [q], 1.0, f"q{q} through {self.arch.nodes[b].name}")
                    self.free[b] = t
                    energy += self.noise.junction_heating
                continue
            t = self.merge_into(q, b, a, t, energy)
            if b != destination:
                # Pass through the chain of this trap to its far end
                self.qubit_free[q] = t
                t, energy = self.split_off(q, path[k + 1])
        self.incoming[destination] -= 1
        if self.noise.recool:
            t = self.record("cooling", t, self.noise.cooling_time, [destination], [], 1.0, "recool")
            self.energy[destination] = self.noise.cooled_n_bar
            self.energy_time[destination] = t
        self.free[destination] = self.qubit_free[q] = t

    def travel_time(self, q, destination):
        path = self.arch.path(self.location[q], destination, self.timing)
        if path is None:
            return np.inf
        total = self.timing.split + self.timing.merge
        for a, b in zip(path[:-1], path[1:]):
            total += self.timing.move_time(self.arch.length(a, b))
            if self.arch.nodes[b].kind == "junction":
                total += self.timing.junction_time(len(self.arch.neighbors[b]))
        return total

    def bring_together(self, a, b):
        """Move a or b (or both) so that they share a trap where gates can be done."""
        ta, tb = self.location[a], self.location[b]
        if ta == tb and self.arch.nodes[ta].gate_zone:
            return
        gate_traps = set(self.arch.gate_traps())
        options = []
        for mover, stay in ((a, b), (b, a)):
            trap = self.location[stay]
            if trap in gate_traps and self.room(trap) >= 1:
                options.append((self.travel_time(mover, trap), mover, trap))
        if options:
            _, mover, trap = min(options)
            self.move(mover, trap)
            return
        # Otherwise both go to the nearest gate trap with room for them
        candidates = [(self.travel_time(a, t) + self.travel_time(b, t), t) for t in gate_traps if self.room(t) >= 2 - (self.location[a] == t) - (self.location[b] == t)]
        if not candidates:
            raise ValueError("no gate trap has room for the ions of a gate; use larger capacities or more traps")
        _, trap = min(candidates)
        for q in (a, b):
            if self.location[q] != trap:
                self.move(q, trap)


def compile_circuit(circuit, architecture, timing=None, noise=None):
    """Compile and simulate a circuit on a QCCD architecture.

    Args:
        circuit: tiopt.circuits.QuantumCircuit.
        architecture: Architecture.
        timing: TimingModel.
        noise: NoiseModel.

    Returns:
        CompileResult
    """
    from .circuits import MS_PER_GATE, VIRTUAL_GATES

    timing = TimingModel() if timing is None else timing
    noise = NoiseModel() if noise is None else noise
    if timing.gate_model not in GATE_MODELS:
        raise ValueError(f"unknown gate model {timing.gate_model!r}; choose from {', '.join(GATE_MODELS)}")
    compiler = _Compiler(architecture, timing, noise, circuit.n_qubits)
    compiler.place(circuit)
    initial = {q: architecture.nodes[t].name for q, t in compiler.location.items()}
    for gate in circuit.gates:
        if gate.name == "barrier":
            continue
        if gate.name == "measure":
            compiler.single(gate.qubits[0], str(gate), "measurement", timing.measurement, noise.measurement_error)
            continue
        if len(gate.qubits) == 1:
            if gate.name not in VIRTUAL_GATES:
                compiler.single(gate.qubits[0], str(gate))
            continue
        for a, b in _pairs(gate, MS_PER_GATE.get(gate.name, 1)):
            compiler.bring_together(a, b)
            compiler.gate(a, b, str(gate))
    duration = max((op.end for op in compiler.operations), default=0.0)
    final = {q: architecture.nodes[t].name for q, t in compiler.location.items()}
    return CompileResult(compiler.operations, duration, compiler.fidelity, compiler.max_n_bar, compiler.counts, initial, final, architecture)


def explore(circuit, architectures, capacities=(4, 8, 16), gate_models=GATE_MODELS, swap_types=("gate", "ion"), noise=None):
    """Compile a circuit on every combination of architectures (a dict from a name to a function of the capacity which
    builds the architecture), trap capacities, gate models, and reordering methods, like the design space exploration of
    Murali et al.

    Returns:
        List of dicts with the parameters and the duration (µs), fidelity, and largest mean phonon number of each, or the
        error if the circuit does not fit
    """
    rows = []
    for name, build in architectures.items():
        for capacity in capacities:
            for model in gate_models:
                for swap in swap_types:
                    row = dict(architecture=name, capacity=capacity, gate_model=model, swap_type=swap)
                    try:
                        result = compile_circuit(circuit, build(capacity), TimingModel(gate_model=model, swap_type=swap), noise)
                        row.update(duration=result.duration, fidelity=result.fidelity, max_n_bar=result.max_n_bar,
                                   shuttles=result.counts.get("split", 0))
                    except ValueError as error:
                        row.update(error=str(error))
                    rows.append(row)
    return rows
