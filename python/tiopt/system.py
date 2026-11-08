"""Full co-design of a trap, its integrated photonics, gate pulses, transport schedule, and wells for a circuit, from the
inputs of every tab of the GUI at once.

Unlike the co-design flow of the C library (codesign.c), which designs a linear trap of a zone layout through a few
parameters in stages, this starts from any trap (a preset, a trap from the Trap builder, or a file) with its zones and
photonics, and optimizes every enabled group of variables together:
- trap: amplitude and frequency of the RF drive
- shapes: free-form deformation of the electrodes around a center, e.g. a junction (see shapes.py)
- waveguides: width and thickness of the waveguides which carry the qubit light (which set their loss and the grating)
- bends: radius, Euler fraction, and taper length of the bends on the way to the grating (see components.py)
- gratings: fill factor, number of teeth, angular width, period (unless chirped), and apodization of the qubit grating
- pulses: fraction of the available Rabi frequency used for gates, and the ramp time of the pulse shape (see pulses.py)
- transport: fastest speed of a transport, in the priority mode chosen (see scheduling.py)
- waveforms: axial frequency of the wells
- detection: detection time and detector size

Every evaluation follows the light from the chip edge through the waveguides and bends to the grating and the ion, the
gates from the light, the schedule of the circuit on the zones of the trap with the motion of every ion, and the errors
of every native gate of the circuit, whose product is the estimated circuit fidelity. The objective is -log of that
fidelity, plus penalties for designs which cannot run (unstable or shallow traps, voltages beyond the limit, or
deformations which fold electrodes over), plus a small weight on the pseudopotential barrier and confinement dip along a
path through the shaped region (which make transport through junctions harder in ways the schedule does not model).
"""

import os
import tempfile
import threading
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import optimize

from . import circuits, components, pulses, scheduling, shapes
from .builder import zone_nulls
from .confinement import RFField, confinement_path, default_waypoints
from .lib import (C_LIGHT, E_CHARGE, HBAR, Circuit, TiOptError, TransportProgram, TrapGraph, Waveform, detection_fidelity, expansion_region,
                  rabi_frequency_from_intensity, saturation_intensity, scattering_rate)

GROUPS = {
    "trap": "RF drive",
    "shapes": "Electrode shapes",
    "waveguides": "Waveguide cross-section",
    "bends": "Waveguide bends",
    "gratings": "Qubit grating",
    "pulses": "Gate pulses",
    "transport": "Transport",
    "waveforms": "Wells",
    "detection": "Detection",
}


@dataclass(frozen=True)
class Variable:
    """Variable of the co-design, with its bounds in the units of SystemInputs, optimized in its logarithm if log."""

    group: str
    name: str
    label: str
    lower: float
    upper: float
    log: bool = False

    def to_unit(self, value):
        value = float(np.clip(value, self.lower, self.upper))
        if self.log:
            return np.log(value / self.lower) / np.log(self.upper / self.lower)
        return (value - self.lower) / (self.upper - self.lower)

    def from_unit(self, u):
        u = float(np.clip(u, 0.0, 1.0))
        if self.log:
            return self.lower * (self.upper / self.lower) ** u
        return self.lower + u * (self.upper - self.lower)


# In the order the optimizer first searches along them, with the variables which matter most for the circuit first
VARIABLES = (
    Variable("pulses", "rabi_fraction", "Fraction of the Rabi frequency used", 0.01, 1.0, True),
    Variable("pulses", "ramp", "Pulse ramp time (s)", 0.0, 20e-6),
    Variable("pulses", "gaussian_width", "Gaussian width / duration", 0.05, 0.5),
    Variable("waveforms", "omega_axial", "Axial frequency (rad/s)", 2 * np.pi * 0.3e6, 2 * np.pi * 3e6, True),
    Variable("trap", "V_RF", "RF amplitude (V)", 20.0, 400.0, True),
    Variable("trap", "Omega_RF", "RF frequency (rad/s)", 2 * np.pi * 15e6, 2 * np.pi * 100e6, True),
    Variable("gratings", "grating_fill", "Grating fill factor", 0.2, 0.8),
    Variable("gratings", "grating_teeth", "Grating teeth", 8.0, 80.0),
    Variable("gratings", "angular_width", "Grating angular width (°)", 4.0, 40.0),
    Variable("gratings", "grating_period_scale", "Grating period / phase matched", 0.9, 1.1),
    Variable("gratings", "fill_start", "Fill factor of the first tooth", 0.05, 0.95),
    Variable("waveguides", "waveguide_width", "Waveguide width (µm)", 0.3, 1.5),
    Variable("waveguides", "waveguide_thickness", "Waveguide thickness (µm)", 0.08, 0.3),
    Variable("bends", "bend_radius", "Bend radius (µm)", 5.0, 100.0, True),
    Variable("bends", "euler_fraction", "Euler fraction", 0.0, 1.0),
    Variable("bends", "taper_length", "Bend taper length (µm)", 0.0, 30.0),
    Variable("transport", "max_speed", "Fastest transport (µm/s)", 1e6, 1e8, True),
    Variable("detection", "detection_time", "Detection time (s)", 2e-6, 1e-3, True),
    Variable("detection", "detector_size", "Detector size (µm)", 10.0, 150.0),
)


@dataclass
class SystemInputs:
    """Everything the co-design starts from, as the tabs of the GUI set it.

    Trap (Trap and Trap builder tabs): the trap (solved, with its zones, including a gate zone, and its photonics), the
    ion species, and the RF drive.
    Photonics (Photonics tab): the chip stack, the waveguide cross-section, the route from the edge of the chip to the
    qubit grating (its length, number of 90° bends, and the bends), the coupling loss at the edge, and the qubit grating.
    Circuit and pulses (Quantum tab): the circuit, the pulse shape and ramp of the gates, the fraction of the available
    Rabi frequency used, the mean phonon number after cooling, and the dephasing rate.
    Transport (Scheduling tab): the priority mode (see scheduling.MODES, or "best" for whichever is best), the fastest
    and slowest transports, and whether ions are cooled again while they wait.
    Wells (Waveforms tab): the axial frequency and the voltage limit.
    Noise (Dynamics tab): the voltage noise of the DC electrodes and the electric field noise at the ion.
    Detection and limits (Co-design tab): the light and detector used for detection, and the limits of the trap.
    Electrode shapes (Trap tab): the design region and largest displacement of the electrodes.
    """

    trap: object
    ion: object
    circuit: object
    V_RF: float = 40.0
    Omega_RF: float = 2 * np.pi * 40e6
    stack: components.Stack = field(default_factory=components.Stack)
    waveguide_width: float = 0.6
    waveguide_thickness: float = 0.15
    route_length: float = 5000.0
    n_bends: int = 2
    bend_radius: float = 20.0
    euler_fraction: float = 0.5
    bend_width: float = None
    taper_length: float = 0.0
    taper_profile: str = "linear"
    coupling_loss: float = 3.0
    grating: components.GratingDesign = field(default_factory=components.GratingDesign)
    qubit_power: float = 1e-3
    pulse_shape: str = "square"
    ramp: float = 0.0  # ramp time (s) of trapezoidal and cosine pulses, or the width of Gaussian pulses / their duration
    rabi_fraction: float = 0.2
    n_bar_cooled: float = 0.05
    dephasing_rate: float = 100.0
    transport_mode: str = "eager"
    max_speed: float = 2e7
    min_speed: float = 1e6
    recool: bool = True
    cooling_time: float = 100e-6
    omega_axial: float = 2 * np.pi * 1e6
    max_voltage: float = 20.0
    voltage_noise: float = 1e-19
    field_noise: float = 1e-12
    field_noise_exponent: float = 1.0
    detection_power: float = 1e-5
    detection_time: float = 50e-6
    detector_size: float = 50.0
    detector_efficiency: float = 0.8
    dark_count_density: float = 1e-3
    position_drift: float = 0.1
    max_mathieu_q: float = 0.4
    min_trap_depth: float = 0.02
    shape_center: tuple = None
    shape_radius: float = None
    max_shift: float = 8.0
    shape_symmetry: str = "none"
    path_weight: float = 0.01

    @property
    def gaussian_width(self):
        return self.ramp if self.pulse_shape == "gaussian" else 0.2

    @property
    def grating_period_scale(self):
        return 1.0

    @property
    def grating_fill(self):
        return self.grating.fill_factor

    @property
    def grating_teeth(self):
        return float(self.grating.n_teeth)

    @property
    def angular_width(self):
        return self.grating.angular_width

    @property
    def fill_start(self):
        return self.grating.fill_start if self.grating.fill_start is not None else self.grating.fill_factor


@dataclass
class Evaluation:
    """Metrics of one design, the objective, the contribution of each kind of error to -log(circuit fidelity), and the
    parameters it was evaluated with.
    """

    index: int
    objective: float
    metrics: dict
    budget: dict
    parameters: dict
    feasible: bool = True
    notes: list = field(default_factory=list)


class Stop(Exception):
    """Raised inside the optimizer when the co-design is stopped."""


class SystemCodesign:
    """Co-design of a whole system from SystemInputs (see the module docstring)."""

    def __init__(self, inputs, groups=("trap", "waveguides", "bends", "gratings", "pulses", "transport", "waveforms", "detection"), simulate_gates=False):
        """Set up the co-design: schedule the circuit on the zones of the trap and prepare the variables of the groups.

        Args:
            inputs: SystemInputs.
            groups: Groups of variables to optimize (see GROUPS).
            simulate_gates: Specifies whether to simulate the two-qubit gates with the master equation at every
                evaluation (otherwise only for the first and best designs, and estimated in between).
        """
        unknown = set(groups) - set(GROUPS)
        if unknown:
            raise ValueError(f"unknown groups {', '.join(sorted(unknown))}; choose from {', '.join(GROUPS)}")
        self.inputs = inputs
        self.groups = tuple(groups)
        self.simulate_gates = simulate_gates
        trap = inputs.trap
        if trap is None or not trap.solved:
            raise ValueError("Generate, build, or load a trap first")
        if trap.n_rf == 0:
            raise ValueError("The trap needs RF electrodes")
        zones = trap.zones()
        self.zone_types = [zone.types for zone in zones]
        gates = [z for z, types in enumerate(self.zone_types) if "gate" in types]
        if not gates:
            raise ValueError("The trap needs a gate zone (add one in the Trap builder or use a trap from a zone layout)")
        self.gate_zone = gates[0]
        measures = [z for z, types in enumerate(self.zone_types) if "measure" in types]
        self.measure_zone = measures[0] if measures else None
        if not any("load" in types for types in self.zone_types):
            raise ValueError("The trap needs a loading zone, where the ions of the circuit start")

        # The circuit, and the transports which run it on the zones of the trap
        self.circuit = inputs.circuit
        self.counts = circuits.native_counts(self.circuit)
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "circuit.qasm")
        try:
            circuits.write_scheduler_qasm(self.circuit, path)
            self.scheduler_circuit = Circuit(path)
        finally:
            if os.path.exists(path):
                os.remove(path)
            os.rmdir(directory)
        self.trap_graph = TrapGraph.from_trap(trap)
        try:
            self.program = TransportProgram(self.trap_graph, self.scheduler_circuit)
        except TiOptError:
            raise ValueError("The circuit can't be scheduled on the zones of the trap (the loading and storage zones must hold every qubit, "
                             "and the gate zones the ions of every gate)") from None
        self.zone_positions = self.trap_graph.positions

        # Electrode shapes
        self.waypoints = None
        self.deformation = None
        if "shapes" in groups or inputs.path_weight > 0:
            self.waypoints = default_waypoints(trap)
        if "shapes" in groups:
            center = inputs.shape_center if inputs.shape_center is not None else (self.waypoints[len(self.waypoints) // 2] if len(self.waypoints) > 2 else trap.zones()[self.gate_zone].position[:2])
            radius = inputs.shape_radius or 2.5 * 55.0
            self.deformation = shapes.Deformation(center, radius, radius / 2, inputs.max_shift, inputs.shape_symmetry)
            originals = [polygon[:, :2].copy() for polygon in trap.electrode_polygons()]
            self.refined = [shapes.refine_polygon(polygon, self.deformation.center, radius + 8.0, 4.0) for polygon in originals]

        self.variables = [v for v in VARIABLES if v.group in groups and self._applies(v)]
        self.n_shape = self.deformation.n_variables if self.deformation is not None else 0
        self.history = []
        self.best = None
        self.initial = None
        self._stop = threading.Event()
        self._caches = {}
        self._lock = threading.Lock()

    def _applies(self, variable):
        grating = self.inputs.grating
        if variable.name == "grating_period_scale":
            return not grating.chirp
        if variable.name == "fill_start":
            return grating.fill_start is not None
        if variable.name == "ramp":
            return self.inputs.pulse_shape in ("trapezoid", "cosine")
        if variable.name == "gaussian_width":
            return self.inputs.pulse_shape == "gaussian"
        return True

    # Parameters --------------------------------------------------------------------------------------------------

    def initial_parameters(self):
        """Parameters of the design the inputs describe."""
        inputs = self.inputs
        parameters = {v.name: float(getattr(inputs, v.name)) for v in VARIABLES}
        parameters["shape"] = np.zeros(self.n_shape)
        return parameters

    def to_unit(self, parameters):
        """Normalized variables of the optimizer for parameters."""
        u = [v.to_unit(parameters[v.name]) for v in self.variables]
        return np.concatenate([u, 0.5 * (np.asarray(parameters["shape"]) + 1)]) if self.n_shape else np.array(u)

    def from_unit(self, u):
        parameters = self.initial_parameters()
        for v, value in zip(self.variables, u):
            parameters[v.name] = v.from_unit(value)
        if self.n_shape:
            parameters["shape"] = 2 * np.clip(np.asarray(u[len(self.variables):]), 0, 1) - 1
        return parameters

    # Caches ------------------------------------------------------------------------------------------------------

    def _cached(self, name, key, compute):
        cache = self._caches.setdefault(name, {})
        if key not in cache:
            if len(cache) > 200:
                cache.clear()
            cache[key] = compute()
        return cache[key]

    def _geometry(self, shape):
        """Trap with the electrode shapes (solved), its zone nulls, and the path metrics of its RF null."""
        key = tuple(np.round(shape, 6)) if self.n_shape else ()

        def compute():
            jacobian = 1.0
            if self.n_shape:
                jacobian = self.deformation.min_jacobian(shape, 21)
                polygons = shapes.deformed_polygons(self.refined, self.deformation, shape)
            else:
                polygons = [polygon[:, :2] for polygon in self.inputs.trap.electrode_polygons()]
            # A copy, so that the RF drive of the trap of the inputs is left alone
            trap = shapes.trap_from_polygons(self.inputs.trap, polygons)
            trap.V_RF = self.inputs.V_RF
            trap.Omega_RF = self.inputs.Omega_RF
            nulls = zone_nulls(trap, self.inputs.ion)
            path = None
            if self.waypoints is not None:
                try:
                    path = confinement_path(trap, self.inputs.ion, self.waypoints, 41)
                except (ValueError, np.linalg.LinAlgError):
                    path = None
            return trap, nulls, path, jacobian

        return self._cached("geometry", key, compute)

    # Evaluation --------------------------------------------------------------------------------------------------

    def evaluate(self, parameters, simulate_gates=None):
        """Evaluate a design (see Evaluation)."""
        inputs = self.inputs
        ion = self.inputs.ion
        simulate_gates = self.simulate_gates if simulate_gates is None else simulate_gates
        p = parameters
        metrics = {}
        notes = []
        penalty = 0.0

        # Trap -------------------------------------------------------------------------------------------------------
        trap, nulls, path, jacobian = self._geometry(p["shape"])
        trap.V_RF = p["V_RF"]
        trap.Omega_RF = p["Omega_RF"]
        if nulls[self.gate_zone] is None:
            return self._failed(p, "no RF null at the gate zone")
        gate = nulls[self.gate_zone][0]
        field_ = RFField(trap)
        gradient, hessian = field_.derivatives(gate)
        scale = field_._scale(ion)
        # At the null the pseudopotential curves as 2 scale H H (with H in 1/m^2)
        hessian_U = 2 * scale * (hessian @ hessian) * 1e24
        values, vectors = np.linalg.eigh(hessian_U)
        omegas = np.sqrt(np.maximum(values, 0.0) / ion.mass)
        axis = vectors[:, 0].copy()
        metrics["radial_frequency"] = float(omegas[1])
        metrics["mathieu_q"] = float(2 * np.sqrt(2) * omegas[2] / p["Omega_RF"])
        heights = np.linspace(gate[2], gate[2] + 4 * 55.0, 160)
        column = np.repeat(gate[None], len(heights), axis=0)
        column[:, 2] = heights
        energy = field_.pseudopotential(ion, column)
        metrics["trap_depth"] = float((energy.max() - energy[0]) / E_CHARGE)
        S_E = inputs.field_noise * (2 * np.pi * 1e6 / p["omega_axial"]) ** inputs.field_noise_exponent
        heating = ion.charge ** 2 * S_E / (4 * ion.mass * HBAR * p["omega_axial"])
        x_min, x_max, y_min, y_max = expansion_region()
        inside = [n is not None and x_min <= n[0][0] <= x_max and y_min <= n[0][1] <= y_max for n in nulls]
        if inside[self.gate_zone]:
            heating += trap.voltage_noise_heating_rate(ion, gate, p["omega_axial"], axis, inputs.voltage_noise)
        else:
            notes.append("the gate zone is beyond the spherical harmonic expansions, so voltage noise and voltages are left out")
        metrics["heating_rate"] = float(heating)

        # Voltages which hold every zone the circuit uses at once, and which move an ion into the gate zone
        used = sorted({int(np.argmin(np.linalg.norm(self.zone_positions - q, axis=1))) for q in np.vstack([self.program.initial_positions, self.program.destinations])})
        wells = [nulls[z][0] for z in used if nulls[z] is not None and inside[z]]
        key = (tuple(np.round(p["shape"], 6)) if self.n_shape else (), round(p["V_RF"], 6), round(p["Omega_RF"], 3), round(p["omega_axial"], 3))

        def voltages():
            static = peak = np.nan
            try:
                if wells:
                    waveform = Waveform.from_wells(trap, ion, [0.0], np.array(wells)[None], p["omega_axial"])
                    static = float(np.abs(waveform.voltages[0, trap.n_rf:]).max())
                    waveform.close()
                neighbors = [z for z in range(len(nulls)) if z != self.gate_zone and nulls[z] is not None and inside[z]]
                if neighbors and inside[self.gate_zone]:
                    start = nulls[min(neighbors, key=lambda z: np.linalg.norm(nulls[z][0] - gate))][0]
                    u = scheduling.transport_profile(np.linspace(0, 1, 21))
                    positions = start[None] + u[:, None] * (gate - start)[None]
                    waveform = Waveform.from_wells(trap, ion, np.linspace(0, 1e-5, 21), positions[:, None], p["omega_axial"])
                    peak = float(np.abs(waveform.voltages[:, trap.n_rf:]).max())
                    waveform.close()
            except TiOptError:
                pass
            return static, peak

        metrics["static_voltage"], metrics["peak_voltage"] = self._cached("voltages", key, voltages)

        # Light from the edge of the chip to the ion ---------------------------------------------------------------
        stack = inputs.stack
        width, thickness = p["waveguide_width"], p["waveguide_thickness"]
        straight = components.analyze_waveguide(stack, width, thickness)
        if straight is None:
            return self._failed(p, "the waveguide guides no light")
        metrics["waveguide_loss"] = float(straight.loss)
        metrics["waveguide_modes"] = int(straight.n_modes)
        bend_width = inputs.bend_width if inputs.bend_width is not None else width
        bend_key = (round(width, 5), round(thickness, 5), round(p["bend_radius"], 4), round(p["euler_fraction"], 4), round(p["taper_length"], 4), round(bend_width, 5))
        bend = self._cached("bends", bend_key, lambda: components.analyze_bend(stack, width, thickness, p["bend_radius"], 90.0, bend_width, p["euler_fraction"],
                                                                               taper_length=p["taper_length"], taper_profile=inputs.taper_profile, n_taper_steps=10))
        bend_loss = bend.loss if bend is not None else 10.0
        metrics["bend_loss"] = float(bend_loss)
        route = inputs.coupling_loss + straight.loss * inputs.route_length / components.CM + inputs.n_bends * bend_loss
        metrics["route_loss"] = float(route)
        power = inputs.qubit_power * 10 ** (-route / 10)
        metrics["grating_power"] = float(power)

        # Qubit grating, aimed at the gate zone from where the trap's qubit grating is
        frame_origin, direction, distance = self._qubit_frame(trap, gate)
        design = replace(inputs.grating, fill_factor=p["grating_fill"], n_teeth=int(round(p["grating_teeth"])), angular_width=p["angular_width"],
                         thickness=thickness, etch_depth=min(inputs.grating.etch_depth, thickness), target_height=float(gate[2]), target_distance=distance, power=power)
        if design.fill_start is not None:
            design = replace(design, fill_start=p["fill_start"])
        if not design.chirp:
            design = replace(design, period=self._cached("periods", (round(design.fill_factor, 5), round(thickness, 5), round(design.etch_depth, 5), round(distance, 3), round(design.target_height, 3)),
                                                         lambda: components.phase_matched_period(stack, replace(design, period=0.6))) * p["grating_period_scale"])
        metrics["grating_period"] = float(design.period)
        grating_key = tuple(np.round([design.period, design.fill_factor, design.n_teeth, design.angular_width, thickness, design.etch_depth, distance, design.target_height,
                                      design.fill_start if design.fill_start is not None else -1], 6))
        unit = replace(design, power=1.0)

        def grating():
            result = components.analyze_grating(stack, unit)
            if result is None:
                return None
            layout = components.tooth_layout(stack, unit)
            others = []
            for z, null in enumerate(nulls):
                if z == self.gate_zone or null is None:
                    continue
                offset = null[0][:2] - gate[:2]
                others.append([offset @ direction, offset @ np.array([-direction[1], direction[0]]), null[0][2]])
            drift = inputs.position_drift
            displaced = [[drift, 0, gate[2]], [-drift, 0, gate[2]], [0, drift, gate[2]], [0, 0, gate[2] + drift], [0, 0, gate[2] - drift]]
            points = np.array(others + displaced, dtype=float).reshape(-1, 3)
            values = np.abs(components.grating_field(stack, unit, points, layout)) ** 2 * 1e12 if len(points) else np.zeros(0)
            return result, values[:len(others)], values[len(others):]

        cached = self._cached("gratings", grating_key, grating)
        if cached is None:
            return self._failed(p, "the grating guides no light")
        grating_result, other_intensities, drift_intensities = cached
        intensity = grating_result.intensity * power
        metrics["intensity"] = float(intensity)
        metrics["crosstalk"] = float(other_intensities.max() / grating_result.intensity) if len(other_intensities) else 0.0
        metrics["rabi_sensitivity"] = float(np.max(0.5 * np.abs(np.log(np.maximum(drift_intensities, 1e-300) / grating_result.intensity)))) if len(drift_intensities) else 0.0
        available = rabi_frequency_from_intensity(ion.qubit_linewidth, ion.qubit_wavelength, intensity) if intensity > 0 else 0.0
        rabi = p["rabi_fraction"] * available
        metrics["rabi_frequency"] = float(rabi)
        theta = np.radians(grating_result.emission_angle) if np.isfinite(grating_result.emission_angle) else 0.0
        k = 2 * np.pi / (stack.wavelength * 1e-6)
        k_vector = k * np.array([np.sin(theta) * direction[0], np.sin(theta) * direction[1], np.cos(theta)])
        eta = abs(k_vector @ axis) * np.sqrt(HBAR / (2 * ion.mass * p["omega_axial"]))
        metrics["lamb_dicke_parameter"] = float(eta)
        if rabi <= 0 or eta <= 0:
            return self._failed(p, "no light reaches the ion")

        # Detection -----------------------------------------------------------------------------------------------
        metrics["detection_fidelity"] = self._detection(trap, nulls, p)

        # Gates -------------------------------------------------------------------------------------------------
        shape = inputs.pulse_shape
        ramp = p["gaussian_width"] if shape == "gaussian" else p["ramp"]
        nu = p["omega_axial"]
        t_1 = pulses.duration_for_area(shape, ramp, 0.5 * np.pi, rabi)
        tone = rabi / np.sqrt(2)
        ms_key = (shape, round(ramp, 12), round(eta, 8), round(tone, 3))
        try:
            t_ms, delta, shape_parameters = self._cached("ms", ms_key, lambda: pulses.ms_gate(shape, ramp, eta, tone))
        except ValueError:
            return self._failed(p, "the pulse shape can't make a two-qubit gate")
        t_2 = t_ms + circuits.ROTATIONS_PER_MS * t_1
        metrics["single_qubit_time"] = float(t_1)
        metrics["two_qubit_time"] = float(t_ms)
        carrier = 2 * (tone / nu) ** 2 * (pulses.carrier_suppression(shape, shape_parameters, t_ms, nu) if shape != "square" else 1.0)
        metrics["carrier_error"] = float(carrier)

        # Schedule ----------------------------------------------------------------------------------------------
        timing = scheduling.Timing(single_qubit_time=t_1, two_qubit_time=t_2, detection_time=p["detection_time"], max_speed=p["max_speed"],
                                   min_speed=min(inputs.min_speed, p["max_speed"]), omega_axial=nu, mass=ion.mass, heating_rate=metrics["heating_rate"],
                                   n_bar_cooled=inputs.n_bar_cooled, cooling_time=inputs.cooling_time, recool=inputs.recool)
        modes = list(scheduling.MODES) if inputs.transport_mode == "best" else [inputs.transport_mode]
        best = None
        for mode in modes:
            plan = scheduling.plan(self.program, self.scheduler_circuit, self.zone_positions, self.zone_types, timing, mode)
            result = self._circuit_errors(plan, metrics, eta, rabi, tone, t_1, t_ms, carrier, p, simulate_gates)
            if best is None or result[0] < best[0]:
                best = (result[0], result[1], plan, mode, result[2])
        log_fidelity_loss, budget, plan, mode, gate_metrics = best
        metrics.update(gate_metrics)
        metrics["transport_mode"] = mode
        metrics["circuit_duration"] = float(plan.duration)
        metrics["transport_excitation"] = float(plan.total_excitation)
        metrics["gate_n_bar"] = float(plan.two_qubit_n_bar)
        metrics["circuit_fidelity"] = float(np.exp(-log_fidelity_loss))

        # Penalties ---------------------------------------------------------------------------------------------
        if not metrics["radial_frequency"] > 0:
            penalty += 1e2
        if metrics["mathieu_q"] > inputs.max_mathieu_q:
            penalty += 1e2 * (metrics["mathieu_q"] / inputs.max_mathieu_q - 1) ** 2
        if metrics["trap_depth"] < inputs.min_trap_depth:
            penalty += 1e2 * (1 - metrics["trap_depth"] / inputs.min_trap_depth) ** 2
        for key in ("static_voltage", "peak_voltage"):
            if np.isfinite(metrics[key]) and metrics[key] > inputs.max_voltage:
                penalty += 1e2 * (metrics[key] / inputs.max_voltage - 1) ** 2
        if jacobian < 0.3:
            penalty += 1e2 * (0.3 - jacobian) ** 2
        path_term = 0.0
        if path is not None:
            metrics["barrier"] = float(path.barrier * 1e3)
            metrics["confinement_dip"] = float(path.confinement_dip)
            path_term = inputs.path_weight * (metrics["barrier"] + 10 * (1 - metrics["confinement_dip"]))
        metrics["penalty"] = float(penalty)
        objective = log_fidelity_loss + penalty + path_term
        metrics["objective"] = float(objective)
        return Evaluation(len(self.history), float(objective), metrics, budget, self._public(p), penalty == 0, notes)

    def _public(self, p):
        values = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
        return values

    def _failed(self, p, reason):
        return Evaluation(len(self.history), 1e3, {"objective": 1e3, "circuit_fidelity": 0.0}, {}, self._public(p), False, [reason])

    def _qubit_frame(self, trap, gate):
        """Position of the qubit grating for the gate zone, the direction from it to the zone, and its distance."""
        for grating in trap.grating_couplers():
            if grating.beam == "qubit" and grating.zone == self.gate_zone:
                offset = gate[:2] - grating.position[:2]
                distance = float(np.linalg.norm(offset))
                if distance > 1.0:
                    return grating.position[:2], offset / distance, distance
        return gate[:2] - [41.25, 0.0], np.array([1.0, 0.0]), 41.25

    def _detection(self, trap, nulls, p):
        """Fidelity of detection in the measurement zone with its detection grating and detector."""
        if self.measure_zone is None or nulls[self.measure_zone] is None or self.counts.measurements == 0:
            return 1.0
        ion = self.inputs.ion
        position = nulls[self.measure_zone][0]
        light = [g for g in trap.grating_couplers() if g.beam == "detection" and g.zone == self.measure_zone]
        detectors = [d for d in trap.detectors() if d.zone == self.measure_zone]
        if not light or not detectors:
            return 1.0
        grating = light[0]
        saved = grating.power
        grating.power = self.inputs.detection_power
        try:
            intensity = float(grating.intensity(position))
        finally:
            grating.power = saved
        gamma = ion.struct.gamma_SP
        wavelength = 2 * np.pi * C_LIGHT / ion.struct.omega_SP
        s = intensity / saturation_intensity(gamma, wavelength)
        detector = detectors[0]
        saved = (detector.width, detector.length, detector.quantum_efficiency, detector.dark_count_rate)
        detector.width = detector.length = p["detector_size"]
        detector.quantum_efficiency = self.inputs.detector_efficiency
        try:
            collection = detector.collection_efficiency(position)
        finally:
            detector.width, detector.length, detector.quantum_efficiency, detector.dark_count_rate = saved
        rate = scattering_rate(gamma, s, -0.5 * gamma) * collection
        dark = self.inputs.dark_count_density * p["detector_size"] ** 2
        fidelity, _ = detection_fidelity(rate, dark, ion.struct.gamma_SD, p["detection_time"])
        return float(fidelity)

    def _circuit_errors(self, plan, metrics, eta, rabi, tone, t_1, t_ms, carrier, p, simulate_gates):
        """-log(circuit fidelity) of a plan, its contributions, and the gate metrics."""
        inputs = self.inputs
        gamma = inputs.dephasing_rate
        heating = metrics["heating_rate"]
        spectator = min(self.counts.n_qubits - 1, 2) * np.sin(0.25 * np.pi * np.sqrt(metrics["crosstalk"])) ** 2
        drift = metrics["rabi_sensitivity"]
        ops = plan.operations
        single_n = [op.n_bar for op in ops if op.kind == "single-qubit gate"] or [inputs.n_bar_cooled]
        two_n = [op.n_bar for op in ops if op.kind == "two-qubit gate"]
        n_mean = float(np.mean(two_n)) if two_n else float(np.mean(single_n))

        def single_error(n):
            spread = eta ** 2 * np.sqrt(n * (n + 1))
            return (0.25 * np.pi * spread) ** 2 + (0.25 * np.pi * drift) ** 2 + 0.5 * gamma * t_1 + spectator

        def ms_error(n):
            return 0.5 * heating * t_ms + 0.5 * gamma * t_ms + 0.25 * np.pi ** 2 * eta ** 4 * (n * n + n)

        if simulate_gates and two_n:
            key = (inputs.pulse_shape, round(p["gaussian_width"] if inputs.pulse_shape == "gaussian" else p["ramp"], 12), round(eta, 8), round(tone, 3), round(n_mean, 4), round(heating, 2), round(gamma, 3), round(p["omega_axial"], 2))
            simulated = self._cached("ms_fidelity", key, lambda: self._simulate_ms(eta, tone, n_mean, heating, p))
            # The simulation gives the gate at the mean phonon number, and the estimate how it changes with it
            offset = (1 - simulated) - ms_error(n_mean)
        else:
            offset = 0.0
        drift_error = (0.5 * np.pi * drift) ** 2

        def two_fidelity(n):
            return max(1 - ms_error(n) - offset, 1e-12) * (1 - carrier) * (1 - drift_error) * (1 - spectator)

        # Every physical rotation of the circuit, and every MS gate with the phonon number of its ions
        n_rotations = self.counts.rotations
        n_ms = self.counts.ms_gates
        single = float(np.mean([1 - single_error(n) for n in single_n]))
        if two_n:
            per_gate = [two_fidelity(n) for n in two_n]
            two = float(np.mean(per_gate))
        else:
            two = two_fidelity(n_mean)
        measurement = metrics["detection_fidelity"]
        idle = 1 - 0.5 * (1 - np.exp(-gamma * plan.duration))
        budget = {
            "single-qubit gates": -n_rotations * np.log(max(single, 1e-300)),
            "two-qubit gates": -n_ms * np.log(max(two, 1e-300)),
            "measurements": -self.counts.measurements * np.log(max(measurement, 1e-300)),
            "dephasing while waiting": -self.counts.n_qubits * np.log(max(idle, 1e-300)),
        }
        gate_metrics = {"single_qubit_fidelity": single, "two_qubit_fidelity": two}
        return float(sum(budget.values())), {k: float(v) for k, v in budget.items()}, gate_metrics

    def _simulate_ms(self, eta, tone, n_bar, heating, p):
        """Fidelity of a Mølmer-Sørensen gate with the shaped pulses, simulated with the master equation."""
        from .lib import simulate_pulses

        ramp = p["gaussian_width"] if self.inputs.pulse_shape == "gaussian" else p["ramp"]
        gates = pulses.ms_pulses(self.inputs.pulse_shape, ramp, eta, tone, p["omega_axial"])
        n_final = n_bar + heating * gates[0].duration
        n_max = int(np.clip(np.ceil(np.log(1e-4) / np.log(max(n_final, 1e-6) / (n_final + 1))) + 6, 8, 40))
        result = simulate_pulses(self.inputs.ion, gates, n_ions=2, mode_frequency=p["omega_axial"], n_max=n_max, n_bar=n_bar, heating_rate=heating,
                                 dephasing_rate=self.inputs.dephasing_rate, decay=True, n_samples=2, density_matrix=True)
        rho = result.density_matrix
        n_motion = n_max + 1
        populations = [np.real(sum(rho[c * n_motion + k, c * n_motion + k] for k in range(n_motion))) for c in (0, 3)]
        coherence = sum(rho[k, 3 * n_motion + k] for k in range(n_motion))
        return float(0.5 * sum(populations) + abs(coherence))

    # Optimization ------------------------------------------------------------------------------------------------

    def stop(self):
        """Stop a running optimization after the current evaluation."""
        self._stop.set()

    def _record(self, evaluation):
        with self._lock:
            self.history.append(evaluation)
            if self.best is None or evaluation.objective < self.best.objective:
                self.best = evaluation

    def run(self, max_evaluations=100, method="Powell", callback=None):
        """Optimize every enabled group of variables together.

        Args:
            max_evaluations: Largest number of evaluations of the design.
            method: "Powell" or "Nelder-Mead" (see scipy.optimize.minimize).
            callback: Function called with every Evaluation.

        Returns:
            Best Evaluation, which is simulated with the master equation if the evaluations were not
        """
        self._stop.clear()
        start = self.initial_parameters()
        if self.initial is None:
            self.initial = self.evaluate(start, simulate_gates=True)
            self._record(self.initial)
            if callback is not None:
                callback(self.initial)
        u0 = self.to_unit(self.from_unit(self.to_unit(start)))

        def objective(u):
            if self._stop.is_set() or len(self.history) >= max_evaluations + 1:
                raise Stop
            evaluation = self.evaluate(self.from_unit(u))
            self._record(evaluation)
            if callback is not None:
                callback(evaluation)
            return evaluation.objective

        if len(u0):
            bounds = [(0.0, 1.0)] * len(u0)
            options = dict(maxfev=max_evaluations, xtol=1e-3, ftol=1e-5) if method == "Powell" else dict(maxfev=max_evaluations, xatol=1e-3, fatol=1e-5)
            try:
                optimize.minimize(objective, u0, method=method, bounds=bounds, options=options)
            except Stop:
                pass
        # The best design, with its two-qubit gates simulated
        best = self.evaluate(self._parameters(self.best), simulate_gates=True)
        best.index = self.best.index
        self.best = best
        return best

    def _parameters(self, evaluation):
        parameters = dict(evaluation.parameters)
        parameters["shape"] = np.asarray(parameters.get("shape", []), dtype=float)
        return parameters

    def best_trap(self):
        """Copy of the trap of the best design (solved), with its RF drive and electrode shapes."""
        evaluation = self.best or self.initial
        parameters = self._parameters(evaluation)
        trap = self.inputs.trap
        if self.n_shape:
            trap = shapes.trap_from_polygons(trap, shapes.deformed_polygons(self.refined, self.deformation, parameters["shape"]))
        else:
            trap = shapes.trap_from_polygons(trap, [polygon[:, :2] for polygon in trap.electrode_polygons()])
        trap.V_RF = parameters["V_RF"]
        trap.Omega_RF = parameters["Omega_RF"]
        return trap

    def close(self):
        for trap, *_ in self._caches.get("geometry", {}).values():
            trap.close()
        self._caches.clear()
        self.trap_graph.close()
        self.program.close()
        self.scheduler_circuit.close()


def format_evaluation(evaluation, initial=None):
    """Describe an evaluation (and how it compares with the initial design) in lines of text."""
    m = evaluation.metrics
    if "radial_frequency" not in m:
        return [f"Evaluation {evaluation.index}: infeasible ({'; '.join(evaluation.notes)})"]
    two_pi = 2 * np.pi

    def change(key, scale=1.0, fmt="{:.4g}"):
        if initial is None or key not in initial.metrics:
            return ""
        return f"  (was {fmt.format(initial.metrics[key] * scale)})"

    lines = [
        f"Evaluation {evaluation.index}, objective {evaluation.objective:.5g}",
        f"Circuit fidelity      {m['circuit_fidelity']:.6f}{change('circuit_fidelity', 1, '{:.6f}')}",
        f"Circuit duration      {m['circuit_duration'] * 1e6:.1f} µs{change('circuit_duration', 1e6, '{:.1f}')}",
        f"Transport             {scheduling.MODES[m['transport_mode']]}, {m['transport_excitation']:.4f} quanta excited",
        f"Two-qubit gate        {m['two_qubit_time'] * 1e6:.1f} µs, F = {m['two_qubit_fidelity']:.6f}, n̄ = {m['gate_n_bar']:.3f}",
        f"Single-qubit gate     {m['single_qubit_time'] * 1e6:.2f} µs, F = {m['single_qubit_fidelity']:.6f}",
        f"Detection             F = {m['detection_fidelity']:.6f}",
        f"Radial frequency      {m['radial_frequency'] / two_pi / 1e6:.3f} MHz (q = {m['mathieu_q']:.3f}), depth {m['trap_depth'] * 1e3:.1f} meV",
        f"Heating rate          {m['heating_rate']:.1f} quanta/s",
        f"Voltages              {m['static_voltage']:.2f} V static, {m['peak_voltage']:.2f} V moving",
        f"Light at the grating  {m['grating_power'] * 1e3:.3f} mW after {m['route_loss']:.2f} dB ({m['waveguide_loss']:.2f} dB/cm, {m['bend_loss']:.3f} dB per bend)",
        f"Rabi frequency        {m['rabi_frequency'] / two_pi / 1e3:.1f} kHz, η = {m['lamb_dicke_parameter']:.4f}, crosstalk {m['crosstalk']:.2e}",
        f"Carrier error (MS)    {m['carrier_error']:.2e}",
    ]
    if "barrier" in m:
        lines.append(f"Path through shapes   barrier {m['barrier']:.2f} meV, confinement dip {m['confinement_dip']:.3f}")
    if m.get("penalty", 0) > 0:
        lines.append(f"Penalty               {m['penalty']:.3g} (a limit is exceeded)")
    lines += [f"Note: {note}" for note in evaluation.notes]
    return lines
