"""Photonic circuits which carry light from the edge of the chip to the grating couplers, like the circuit simulations
and routing of gdsfactory and SAX: scattering (S) matrices of components (straight waveguides, bends, directional
couplers, Y branches, multimode interference splitters, crossings, rings, phase shifters, and gratings), netlists of
them which are solved for the S matrix of the whole circuit, trees of splitters which share one input among several
gratings (e.g. one per gate zone) with any split of the power, and the routes of waveguides from the edge of the chip
to every grating of a trap.

Directional couplers are found with the mode solver of tiopt.components: the two cores, with the effective index method,
have an even and an odd supermode, and light crosses from one core to the other over the coupling length
lambda / (2 (n_even - n_odd)), so that a coupler of length L sends sin^2(pi L / (2 L_c)) of the power across.
"""

from dataclasses import dataclass

import numpy as np

from . import components

DB = 10 / np.log(10)


# Directional couplers ------------------------------------------------------------------------------------------------


@dataclass
class CouplerResult:
    """Supermodes of two identical waveguides side by side: the gap (µm), the effective indices of the even and odd
    supermodes, the length over which all of the power crosses over (µm), and the coupling coefficient (1/µm).
    """
    gap: float
    n_even: float
    n_odd: float
    coupling_length: float
    kappa: float

    def cross_fraction(self, length):
        """Fraction of the power which crosses over in a coupler of a length (µm)."""
        return float(np.sin(self.kappa * length) ** 2)

    def length_for(self, fraction):
        """Shortest length (µm) which sends a fraction of the power across."""
        return float(np.arcsin(np.sqrt(np.clip(fraction, 0.0, 1.0))) / self.kappa)


def analyze_coupler(stack, width, thickness, gap):
    """Find the supermodes of two waveguides of a width and thickness (µm) with a gap (µm) between them, by the effective
    index method.

    Returns:
        CouplerResult
    """
    vertical = components._vertical_mode(stack, thickness)
    if vertical is None:
        raise ValueError("the waveguides do not guide light")
    n_vertical, n_lossless = vertical[0], vertical[1]
    lateral_polarization = "TM" if stack.polarization == "TE" else "TE"
    h = min(stack.wavelength / (n_lossless * 30), width / 25, gap / 6)
    centers = (-(gap + width) / 2, (gap + width) / 2)
    layers = [(c - width / 2, c + width / 2, n_lossless) for c in centers]
    extent = (gap + 2 * width) / 2 + 2.5
    x, eps = components._grid(-extent, extent, h, layers, stack.n_cladding)
    n_pml = int(round(0.8 / h))
    A = components._operator(eps, h, stack.k_0, lateral_polarization, (n_pml, n_pml))
    from scipy.sparse.linalg import eigs
    values, vectors = eigs(A, k=6, sigma=(stack.k_0 * n_lossless) ** 2)
    core = np.real(eps) > stack.n_cladding ** 2 + 1e-9
    candidates = []
    for j in range(len(values)):
        power = np.abs(vectors[:, j]) ** 2
        confined = np.sum(power[core]) / np.sum(power)
        if confined > 0.3:
            n = np.sqrt(values[j] + 0j) / stack.k_0
            mode = vectors[:, j]
            left = np.sum(mode[x < 0] * np.abs(mode[x < 0]))
            right = np.sum(mode[x > 0] * np.abs(mode[x > 0]))
            even = np.real(left * np.conj(right)) > 0
            candidates.append((abs(n.real), even))
    evens = [n for n, even in candidates if even]
    odds = [n for n, even in candidates if not even]
    if not evens or not odds:
        raise ValueError("the supermodes of the coupler were not found; the gap may be too large")
    n_even, n_odd = max(evens), max(odds)
    delta = max(n_even - n_odd, 1e-12)
    coupling_length = stack.wavelength / (2 * delta)
    return CouplerResult(gap, n_even, n_odd, coupling_length, np.pi / (2 * coupling_length))


# S matrices ------------------------------------------------------------------------------------------------------------


@dataclass
class Component:
    """Component of a photonic circuit with named ports and an S matrix (S[i, j] is the amplitude out of port i for
    light into port j).
    """
    ports: list
    S: np.ndarray
    name: str = ""

    def port(self, name):
        return self.ports.index(name)


def _amplitude(loss_db):
    return 10 ** (-loss_db / 20)


def straight(length, n_eff, wavelength, loss_db_per_cm=0.0):
    """Straight waveguide of a length (µm) with ports "in" and "out"."""
    t = _amplitude(loss_db_per_cm * length / 1e4) * np.exp(-2j * np.pi * n_eff * length / wavelength)
    return Component(["in", "out"], np.array([[0, t], [t, 0]]), "straight")


def bend(loss_db, length=0.0, n_eff=1.5, wavelength=1.0):
    """Waveguide bend with a loss (dB) and a path length (µm), with ports "in" and "out"."""
    t = _amplitude(loss_db) * np.exp(-2j * np.pi * n_eff * length / wavelength)
    return Component(["in", "out"], np.array([[0, t], [t, 0]]), "bend")


def directional_coupler(cross_fraction, loss_db=0.0):
    """Directional coupler which sends a fraction of the power across, with inputs "in1" and "in2" and outputs "out1"
    (through from in1) and "out2" (across from in1).
    """
    a = _amplitude(loss_db)
    t = a * np.sqrt(1 - cross_fraction)
    k = -1j * a * np.sqrt(cross_fraction)
    S = np.zeros((4, 4), dtype=complex)
    # Ports in1, in2, out1, out2
    S[2, 0] = S[0, 2] = t
    S[3, 1] = S[1, 3] = t
    S[3, 0] = S[0, 3] = k
    S[2, 1] = S[1, 2] = k
    return Component(["in1", "in2", "out1", "out2"], S, "coupler")


def splitter(ratios=(0.5, 0.5), excess_loss_db=0.2):
    """Y branch or multimode interference splitter from port "in" to outputs "out1", "out2", ... with the given power
    ratios (normalized to sum to one).
    """
    ratios = np.asarray(ratios, dtype=float)
    ratios = ratios / ratios.sum()
    n = len(ratios)
    S = np.zeros((n + 1, n + 1), dtype=complex)
    amplitudes = _amplitude(excess_loss_db) * np.sqrt(ratios)
    S[1:, 0] = amplitudes
    S[0, 1:] = amplitudes
    return Component(["in"] + [f"out{k + 1}" for k in range(n)], S, "splitter")


def crossing(loss_db=0.2, crosstalk_db=-40.0):
    """Waveguide crossing with ports "west", "east", "south", and "north"."""
    t = _amplitude(loss_db)
    x = 10 ** (crosstalk_db / 20)
    S = np.zeros((4, 4), dtype=complex)
    for a, b in ((0, 1), (2, 3)):
        S[a, b] = S[b, a] = t
    for a in (0, 1):
        for b in (2, 3):
            S[a, b] = S[b, a] = x
    return Component(["west", "east", "south", "north"], S, "crossing")


def ring(radius, n_eff, wavelength, self_coupling=0.95, loss_db_per_cm=1.0):
    """All-pass ring resonator of a radius (µm) next to a bus waveguide with ports "in" and "out"."""
    circumference = 2 * np.pi * radius
    a = _amplitude(loss_db_per_cm * circumference / 1e4)
    phi = 2 * np.pi * n_eff * circumference / wavelength
    r = self_coupling
    t = (r - a * np.exp(-1j * phi)) / (1 - r * a * np.exp(-1j * phi))
    return Component(["in", "out"], np.array([[0, t], [t, 0]]), "ring")


def phase_shifter(phase, loss_db=0.0):
    """Phase shifter (e.g. a heater) with ports "in" and "out"."""
    t = _amplitude(loss_db) * np.exp(1j * phase)
    return Component(["in", "out"], np.array([[0, t], [t, 0]]), "phase shifter")


def grating(efficiency):
    """Grating coupler which emits a fraction of the power from port "in" to port "beam" (towards the ion)."""
    t = np.sqrt(efficiency)
    return Component(["in", "beam"], np.array([[0, t], [t, 0]]), "grating")


class Circuit:
    """Netlist of components, whose connections join pairs of ports and whose external ports are named."""

    def __init__(self):
        self.instances = {}
        self.connections = []
        self.external = {}

    def add(self, name, component):
        self.instances[name] = component
        return self

    def connect(self, a, port_a, b, port_b):
        self.connections.append(((a, port_a), (b, port_b)))
        return self

    def expose(self, name, instance, port):
        self.external[name] = (instance, port)
        return self

    def s_matrix(self):
        """S matrix between the external ports, as a dict from (output, input) to the amplitude."""
        offsets = {}
        n = 0
        for name, component in self.instances.items():
            offsets[name] = n
            n += len(component.ports)
        S = np.zeros((n, n), dtype=complex)
        for name, component in self.instances.items():
            o = offsets[name]
            S[o:o + len(component.ports), o:o + len(component.ports)] = component.S

        def index(instance, port):
            return offsets[instance] + self.instances[instance].port(port)

        # Light leaving one port of a connection enters the other
        C = np.zeros((n, n))
        for (a, pa), (b, pb) in self.connections:
            i, j = index(a, pa), index(b, pb)
            C[i, j] = C[j, i] = 1.0
        names = list(self.external)
        external = [index(*self.external[name]) for name in names]
        E = np.zeros((n, len(names)))
        for column, i in enumerate(external):
            E[i, column] = 1.0
        # b = S (C b + E)  =>  (I - S C) b = S E
        b = np.linalg.solve(np.eye(n) - S @ C, S @ E)
        out = {}
        for column, name_in in enumerate(names):
            for row, name_out in enumerate(names):
                out[(name_out, name_in)] = b[external[row], column]
        return out

    def transmission(self, output, input_):
        """Power transmission from an external port to another."""
        return float(abs(self.s_matrix()[(output, input_)]) ** 2)


# Splitter trees --------------------------------------------------------------------------------------------------------


@dataclass
class SplitterTree:
    """Tree of splitters which shares the power of one input among several outputs: the circuit (with external ports
    "input" and "output1", ...), the fraction sent across by each splitter, the power reaching each output (as a fraction
    of the input), and the target weights.
    """
    circuit: Circuit
    splits: list
    outputs: np.ndarray
    weights: np.ndarray


def splitter_tree(weights, kind="coupler", excess_loss_db=0.1, coupler=None):
    """Design a binary tree of splitters which shares one input among outputs in proportion to weights (e.g. equal
    weights for equal power at every gate zone, or weights which make the Rabi frequencies of zones equal).

    Args:
        weights: Relative power of every output.
        kind: "coupler" for directional couplers (whose lengths come from coupler, a CouplerResult, if given) or
            "y" for splitters with any ratio.
        excess_loss_db: Excess loss of every splitter (dB).
        coupler: CouplerResult which gives the length of every coupler.

    Returns:
        SplitterTree, with a list of (name, fraction across, length in µm or None) for every splitter
    """
    weights = np.asarray(weights, dtype=float)
    circuit = Circuit()
    splits = []
    counter = [0]

    def build(indices, source):
        """Split the light at (instance, port) source among the outputs with the given indices."""
        if len(indices) == 1:
            circuit.expose(f"output{indices[0] + 1}", *source) if source[0] is not None else None
            return
        half = len(indices) // 2
        left, right = indices[:half], indices[half:]
        fraction = weights[right].sum() / weights[indices].sum()
        name = f"s{counter[0]}"
        counter[0] += 1
        if kind == "coupler":
            component = directional_coupler(fraction, excess_loss_db)
            length = coupler.length_for(fraction) if coupler is not None else None
            circuit.add(name, component)
            inputs, outs = "in1", ("out1", "out2")
        else:
            component = splitter((1 - fraction, fraction), excess_loss_db)
            length = None
            circuit.add(name, component)
            inputs, outs = "in", ("out1", "out2")
        splits.append((name, fraction, length))
        if source[0] is None:
            circuit.expose("input", name, inputs)
        else:
            circuit.connect(source[0], source[1], name, inputs)
        for branch, port in zip((left, right), outs):
            if len(branch) == 1:
                circuit.expose(f"output{branch[0] + 1}", name, port)
            else:
                build(branch, (name, port))

    indices = list(range(len(weights)))
    if len(indices) == 1:
        circuit.add("w", straight(0.0, 1.0, 1.0))
        circuit.expose("input", "w", "in").expose("output1", "w", "out")
    else:
        build(indices, (None, None))
    S = circuit.s_matrix()
    outputs = np.array([abs(S[(f"output{k + 1}", "input")]) ** 2 for k in range(len(weights))])
    return SplitterTree(circuit, splits, outputs, weights / weights.sum())


# Routing ---------------------------------------------------------------------------------------------------------------


@dataclass
class Route:
    """Route of a waveguide from the edge of the chip to a grating: its points (µm), length (µm), and bend angles
    (degrees), and the grating it feeds.
    """
    points: np.ndarray
    length: float
    bends: list
    grating: int

    def loss(self, stack, width, thickness, radius, euler_fraction=0.0):
        """Loss of the route (dB) from propagation and bends, using tiopt.components."""
        waveguide = components.analyze_waveguide(stack, width, thickness)
        if waveguide is None:
            return np.inf
        total = waveguide.loss * self.length / 1e4
        for angle in self.bends:
            result = components.analyze_bend(stack, width, thickness, radius, angle=abs(angle), euler_fraction=euler_fraction)
            total += result.radiation_loss + result.transition_loss
        return total


def _fillet(points, radius, n_arc=12):
    """Round the corners of a polyline with arcs of a radius, returning the new points, the length, and the angle of
    every bend (degrees).
    """
    points = [np.asarray(p, dtype=float) for p in points]
    out = [points[0]]
    angles = []
    length = 0.0
    previous = points[0]
    for k in range(1, len(points) - 1):
        a, b, c = points[k - 1], points[k], points[k + 1]
        u = (b - a) / np.linalg.norm(b - a)
        v = (c - b) / np.linalg.norm(c - b)
        turn = np.arctan2(u[0] * v[1] - u[1] * v[0], u @ v)
        if abs(turn) < 1e-9:
            continue
        tangent = radius * np.tan(abs(turn) / 2)
        start = b - u * tangent
        end = b + v * tangent
        length += np.linalg.norm(start - previous) + radius * abs(turn)
        normal = np.array([-u[1], u[0]]) * np.sign(turn)
        center = start + normal * radius
        phi0 = np.arctan2(start[1] - center[1], start[0] - center[0])
        for t in np.linspace(0, 1, n_arc):
            phi = phi0 + turn * t
            out.append(center + radius * np.array([np.cos(phi), np.sin(phi)]))
        angles.append(np.degrees(turn))
        previous = end
    length += np.linalg.norm(points[-1] - previous)
    out.append(points[-1])
    return np.array(out), length, angles


def route_grating(grating, bounds, radius=20.0, lead=None):
    """Route a waveguide from the nearest edge of the chip (bounds = (x_min, x_max, y_min, y_max) in µm) to the input of
    a grating, along a straight line from the edge facing its direction and one bend onto the grating axis.

    Args:
        grating: tiopt Grating.
        bounds: Edges of the chip (µm).
        radius: Radius of the bend (µm).
        lead: Length of the straight lead into the grating (µm), at least what the bend needs by default.

    Returns:
        Route
    """
    azimuth = grating.azimuth
    direction = np.array([np.cos(azimuth), np.sin(azimuth)])
    center = np.asarray(grating.position[:2], dtype=float)
    entry = center - 0.5 * grating.length * direction
    edges = {
        (1.0, 0.0): lambda p: np.array([bounds[0], p[1]]),
        (-1.0, 0.0): lambda p: np.array([bounds[1], p[1]]),
        (0.0, 1.0): lambda p: np.array([p[0], bounds[2]]),
        (0.0, -1.0): lambda p: np.array([p[0], bounds[3]]),
    }
    # Leave the edge whose inward direction is closest to the direction of the grating
    heading = max(edges, key=lambda d: np.dot(d, direction))
    turn = np.arccos(np.clip(np.dot(heading, direction), -1, 1))
    lead = max(lead or 0.0, radius * np.tan(turn / 2) + 1.0)
    corner = entry - lead * direction
    start = edges[heading](corner)
    points, length, bends = _fillet([start, corner, entry], radius)
    return Route(points, length, bends, -1)


def route_trap(trap, margin=200.0, radius=20.0, beams=None):
    """Route a waveguide to every grating of a trap (or those of the given beams) from the edges of a chip which extends
    margin (µm) beyond the electrodes, and count where routes cross.

    Returns:
        List of Route, and the number of crossings between them
    """
    polygons = trap.electrode_polygons()
    points = np.vstack(polygons) if polygons else np.zeros((1, 2))
    bounds = (points[:, 0].min() - margin, points[:, 0].max() + margin, points[:, 1].min() - margin, points[:, 1].max() + margin)
    routes = []
    for g, grating in enumerate(trap.grating_couplers()):
        if beams is not None and grating.beam not in beams:
            continue
        route = route_grating(grating, bounds, radius)
        route.grating = g
        routes.append(route)
    crossings = 0
    for a in range(len(routes)):
        for b in range(a + 1, len(routes)):
            crossings += _count_crossings(routes[a].points, routes[b].points)
    return routes, crossings


def _cross(u, v):
    return u[0] * v[1] - u[1] * v[0]


def _count_crossings(p, q):
    count = 0
    for a0, a1 in zip(p[:-1], p[1:]):
        for b0, b1 in zip(q[:-1], q[1:]):
            d1 = _cross(a1 - a0, b0 - a0)
            d2 = _cross(a1 - a0, b1 - a0)
            d3 = _cross(b1 - b0, a0 - b0)
            d4 = _cross(b1 - b0, a1 - b0)
            if d1 * d2 < 0 and d3 * d4 < 0:
                count += 1
    return count
