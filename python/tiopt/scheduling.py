"""Timing of the transports which run a circuit, with different priorities for when and how fast ions are moved.

The transport program of a circuit (see lib.TransportProgram) says which ions each operation needs in which zone, and
this module decides when every transport leaves and how long it takes:
- "eager": transports leave as soon as their ions are idle and take as long as they can until the operation needs them
  (up to the slowest speed), so that they excite the motion as little as possible
- "just in time": transports leave as late as they can and move at full speed, so that ions wait where they were (e.g.
  in storage, where they can be cooled again) rather than in the zone of their next operation
- "as soon as possible": transports leave as soon as their ions are idle and move at full speed, so that ions arrive
  early and wait in the zone of their next operation

Every plan follows the mean phonon number of each ion, which grows at the heating rate, jumps by the coherent excitation
of every transport (for a well moved along a minimum-jerk profile), and returns to that after cooling if the ion waits
long enough in a zone where it can be cooled. The operations themselves start at the same times in every mode, since
they are set by the fastest transports, so the modes trade the motion of the ions at the gates against time spent
moving.
"""

from dataclasses import dataclass, field

import numpy as np

from .lib import AXIAL_FREQUENCY, HBAR, Waveform

MODES = {
    "eager": "As soon as idle, slowly (least excitation)",
    "just in time": "Just in time, quickly (wait where cooled)",
    "as soon as possible": "As soon as possible, quickly (wait at the destination)",
}


@dataclass
class Timing:
    """Durations of operations (s), the speeds of transports (µm/s), and the motion of the ions.

    Attributes:
        single_qubit_time: Duration of a single-qubit gate.
        two_qubit_time: Duration of a two-qubit gate, including the single-qubit rotations around it.
        detection_time: Duration of a measurement.
        max_speed: Fastest speed of a transport (20 m/s by default, i.e. 100 µm in 5 µs).
        min_speed: Slowest speed of a transport, which limits how slowly eager transports move (1 m/s by default).
        min_duration: Shortest duration of a transport (e.g. a few axial periods).
        omega_axial: Axial secular frequency of the wells (rad/s).
        mass: Mass of the ion (kg).
        heating_rate: Heating rate of the axial mode (quanta/s).
        n_bar_cooled: Mean phonon number after cooling.
        cooling_time: Time an ion must wait in a zone where it can be cooled to be cooled again (s).
        recool: Specifies whether ions waiting in load, storage, and cooling zones are cooled again.
    """

    single_qubit_time: float = 10e-6
    two_qubit_time: float = 200e-6
    detection_time: float = 200e-6
    max_speed: float = 2e7
    min_speed: float = 1e6
    min_duration: float = 5e-6
    omega_axial: float = AXIAL_FREQUENCY
    mass: float = 40 * 1.66053906660e-27
    heating_rate: float = 100.0
    n_bar_cooled: float = 0.05
    cooling_time: float = 100e-6
    recool: bool = True


@dataclass
class Transport:
    """Transport of one ion for an operation, from a position to a destination (µm)."""

    instruction: int
    ion: int
    start: np.ndarray
    destination: np.ndarray
    depart: float
    arrive: float
    excitation: float

    @property
    def duration(self):
        return self.arrive - self.depart

    @property
    def distance(self):
        return float(np.linalg.norm(self.destination - self.start))


@dataclass
class Operation:
    """Operation of the circuit on its ions in a zone, with the mean phonon number of its ions when it starts."""

    instruction: int
    kind: str
    ions: list
    zone: int
    start: float
    end: float
    n_bar: float


@dataclass
class Plan:
    """Timing of every transport and operation of a circuit."""

    mode: str
    transports: list = field(default_factory=list)
    operations: list = field(default_factory=list)
    segments: list = field(default_factory=list)

    @property
    def duration(self):
        """Duration of the circuit (s)."""
        return max([op.end for op in self.operations] + [0.0])

    @property
    def two_qubit_n_bar(self):
        """Mean phonon number at the two-qubit gates."""
        values = [op.n_bar for op in self.operations if op.kind == "two-qubit gate"]
        return float(np.mean(values)) if values else 0.0

    @property
    def max_n_bar(self):
        values = [op.n_bar for op in self.operations if op.kind != "measurement"]
        return float(max(values)) if values else 0.0

    @property
    def total_excitation(self):
        """Quanta excited by all transports."""
        return float(sum(t.excitation for t in self.transports))

    @property
    def transport_time(self):
        """Time spent moving, summed over the ions (s)."""
        return float(sum(t.duration for t in self.transports))

    def summary(self):
        """One line describing the plan."""
        return (f"{MODES[self.mode]}: {self.duration * 1e6:.1f} µs, {len(self.transports)} transports taking "
                f"{self.transport_time * 1e6:.1f} µs in all, {self.total_excitation:.4f} quanta excited, n̄ = "
                f"{self.two_qubit_n_bar:.3f} at the two-qubit gates (largest {self.max_n_bar:.3f})")


def transport_profile(u):
    """Fraction of the distance covered by a minimum-jerk transport after a fraction u of its duration."""
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u ** 2)


def transport_excitation(distance, duration, omega_axial=AXIAL_FREQUENCY, mass=40 * 1.66053906660e-27, n_samples=2001):
    """Quanta excited by moving a harmonic well of angular frequency omega_axial (rad/s) a distance (µm) over a duration
    (s) along a minimum-jerk profile, from the energy |integral of a(t) e^(i omega t)|^2 m / 2 which the acceleration of
    the well leaves the ion with.
    """
    if distance <= 0 or duration <= 0:
        return 0.0
    t = np.linspace(0.0, duration, n_samples)
    u = t / duration
    # Second derivative of the minimum-jerk profile
    acceleration = distance * 1e-6 * (60 * u - 180 * u ** 2 + 120 * u ** 3) / duration ** 2
    amplitude = np.trapezoid(acceleration * np.exp(1j * omega_axial * t), t)
    return float(0.5 * mass * abs(amplitude) ** 2 / (HBAR * omega_axial))


def _kind(instruction, n_targets):
    if instruction.operation == "detect":
        return "measurement"
    return "two-qubit gate" if n_targets > 1 else "single-qubit gate"


def plan(program, circuit, zone_positions, zone_types, timing=None, mode="eager"):
    """Time the transports and operations of a transport program.

    Args:
        program: lib.TransportProgram of the circuit.
        circuit: lib.Circuit the program runs (for the kind of each operation).
        zone_positions: Positions of the zones (µm) with shape (n_zones, 3).
        zone_types: Types of each zone (e.g. "load+storage").
        timing: Timing, or None for the defaults.
        mode: Priority of the transports (see MODES).

    Returns:
        Plan
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
    timing = timing or Timing()
    zone_positions = np.asarray(zone_positions, dtype=float).reshape(-1, 3)
    instructions = circuit.instructions
    n_ions = program.n_ions
    positions = program.initial_positions.copy()
    free = np.zeros(n_ions)
    n_bar = np.full(n_ions, timing.n_bar_cooled)
    zone_free = np.zeros(len(zone_positions))
    cooling = [any(t in str(types) for t in ("cool", "storage", "load")) for types in zone_types]
    result = Plan(mode)

    def zone_of(position):
        return int(np.argmin(np.linalg.norm(zone_positions - position, axis=1))) if len(zone_positions) else -1

    def wait(ion, start, end, zone):
        """Heat (or cool) an ion waiting from start to end in a zone."""
        if end <= start:
            return
        if timing.recool and zone >= 0 and cooling[zone] and end - start >= timing.cooling_time:
            n_bar[ion] = timing.n_bar_cooled
            result.segments.append((ion, start, end, "cooling"))
        else:
            n_bar[ion] += timing.heating_rate * (end - start)
            result.segments.append((ion, start, end, "waiting"))

    for i, (targets, destination) in enumerate(zip(program.targets, program.destinations)):
        ions = [t - 1 for t in targets]
        kind = _kind(instructions[i], len(ions))
        duration = {"measurement": timing.detection_time, "two-qubit gate": timing.two_qubit_time}.get(kind, timing.single_qubit_time)
        zone = zone_of(destination)
        distances = {j: float(np.linalg.norm(destination - positions[j])) for j in ions}
        fast = {j: max(d / timing.max_speed, timing.min_duration) if d > 1e-9 else 0.0 for j, d in distances.items()}
        start = max([zone_free[zone] if zone >= 0 else 0.0] + [free[j] + fast[j] for j in ions])

        for j in ions:
            if distances[j] <= 1e-9:
                wait(j, free[j], start, zone)
                continue
            if mode == "just in time":
                depart, arrive = start - fast[j], start
            elif mode == "eager":
                slowest = max(distances[j] / timing.min_speed, fast[j])
                depart = free[j]
                arrive = min(start, depart + slowest)
            else:
                depart, arrive = free[j], free[j] + fast[j]
            wait(j, free[j], depart, zone_of(positions[j]))
            excitation = transport_excitation(distances[j], arrive - depart, timing.omega_axial, timing.mass)
            n_bar[j] += timing.heating_rate * (arrive - depart) + excitation
            result.transports.append(Transport(i, j, positions[j].copy(), np.array(destination, dtype=float), depart, arrive, excitation))
            result.segments.append((j, depart, arrive, "transport"))
            wait(j, arrive, start, -1)

        result.operations.append(Operation(i, kind, ions, zone, start, start + duration, float(np.mean(n_bar[ions]))))
        for j in ions:
            n_bar[j] += timing.heating_rate * duration
            result.segments.append((j, start, start + duration, kind))
            free[j] = start + duration
            positions[j] = destination
        if zone >= 0:
            zone_free[zone] = start + duration

    return result


def compare(program, circuit, zone_positions, zone_types, timing=None):
    """Plan a transport program in every mode, as {mode: Plan}."""
    return {mode: plan(program, circuit, zone_positions, zone_types, timing, mode) for mode in MODES}


def waveform_from_plan(trap, ion, plan, program, n_steps=51, omega_axial=AXIAL_FREQUENCY):
    """Solve for a waveform which holds a well for every ion and moves the wells as a plan does, so that transports
    which overlap in time move together.

    Args:
        trap: Trap whose electrodynamics have been solved.
        ion: Ion species.
        plan: Plan of the transports.
        program: lib.TransportProgram the plan was made for (for the starting positions).
        n_steps: Steps of the waveform per transport.
        omega_axial: Axial secular frequency of the wells (rad/s).

    Returns:
        lib.Waveform
    """
    if not plan.transports:
        raise ValueError("the plan has no transports")
    times = np.unique(np.concatenate([np.linspace(t.depart, t.arrive, n_steps) for t in plan.transports]))
    wells = np.repeat(program.initial_positions[None].copy(), len(times), axis=0)
    for transport in sorted(plan.transports, key=lambda t: t.depart):
        u = transport_profile((times - transport.depart) / max(transport.duration, 1e-12))
        moving = times >= transport.depart
        wells[moving, transport.ion] = transport.start + u[moving, None] * (transport.destination - transport.start)
    return Waveform.from_wells(trap, ion, times, wells, omega_axial)
