"""Molecular dynamics of many ions in a trap, like pylion (which wraps LAMMPS), for forming, cooling, heating, melting,
and moving Coulomb crystals.

Every ion feels the trap (the pseudopotential, or the full RF drive with micromotion, plus the DC voltages of a waveform
at each moment), the Coulomb force of every other ion, Doppler cooling by laser beams (the scattering force of a
two-level atom with saturation and the Doppler shift, along with the random recoil of every absorbed and emitted
photon), and electric field noise (a Langevin bath with the heating rate of tiopt's field noise spectra). The ions are
integrated with the velocity Verlet method, as in verlet.c, with the stochastic kicks applied at the end of every step.
"""

from dataclasses import dataclass, field

import numpy as np

from .crystals import COULOMB_CONSTANT, HarmonicPotential, TrapPotential, _species, coulomb
from .lib import C_LIGHT, HBAR, ION_HEIGHT

K_B = 1.380649e-23


@dataclass
class CoolingBeam:
    """Doppler cooling beam along a direction, with a detuning from the S-P transition (rad/s, negative for red) and a
    saturation parameter, which cools only the ions of one species if element is given (e.g. for sympathetic cooling).
    The wavelength (m) and linewidth (1/s) are those of the S-P transition of each species unless they are given.
    """
    direction: tuple = (1.0, 1.0, 1.0)
    detuning: float = None
    saturation: float = 1.0
    element: str = None
    wavelength: float = None
    linewidth: float = None

    def parameters(self, ion):
        """Wavevector (1/m), linewidth (1/s), and detuning (rad/s) of the beam for an ion, or None if the beam does not
        address the species.
        """
        if self.element is not None and self.element != ion.element:
            return None
        wavelength = self.wavelength if self.wavelength is not None else 2 * np.pi * C_LIGHT / ion.struct.omega_SP
        gamma = self.linewidth if self.linewidth is not None else ion.struct.gamma_SP
        detuning = self.detuning if self.detuning is not None else -gamma / 2
        direction = np.asarray(self.direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        return 2 * np.pi / wavelength * direction, gamma, detuning


def doppler_limit(ion):
    """Doppler cooling limit (K) of the S-P transition of a species."""
    return HBAR * ion.struct.gamma_SP / (2 * K_B)


@dataclass
class MDResult:
    """Result of a molecular dynamics simulation: the sample times (s), positions (µm) and velocities (m/s) of every ion
    at each sample, the kinetic temperature of each ion averaged over the samples (K) along x, y, and z, whether each
    ion was lost (left the region where the trap can be evaluated), and the species of each ion.
    """
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    lost: np.ndarray
    ions: list
    scattered_photons: np.ndarray = field(default=None)

    def temperatures(self, start=0.5):
        """Kinetic temperature (K) of each ion along x, y, and z from the samples after a fraction of the run."""
        first = int(start * len(self.times))
        masses = np.array([ion.mass for ion in self.ions])
        v2 = np.mean(self.velocities[first:] ** 2, axis=0)
        return masses[:, None] * v2 / K_B

    def kinetic_energy(self):
        """Total kinetic energy (J) of the ions at each sample."""
        masses = np.array([ion.mass for ion in self.ions])
        return 0.5 * np.sum(masses[None, :, None] * self.velocities ** 2, axis=(1, 2))


def thermal_velocities(ions, temperature, rng):
    """Velocities (m/s) drawn from the Maxwell-Boltzmann distribution of each ion at a temperature (K)."""
    masses = np.array([ion.mass for ion in ions])
    return rng.standard_normal((len(ions), 3)) * np.sqrt(K_B * temperature / masses)[:, None]


def ion_cloud(n_ions, center=(0.0, 0.0, ION_HEIGHT), size=10.0, min_distance=2.0, seed=0):
    """Random positions (µm) of n_ions in a Gaussian cloud of a size (µm) around a center, at least min_distance (µm)
    apart (since ions which start very close to each other would fly apart with a great deal of energy).
    """
    rng = np.random.default_rng(seed)
    center = np.asarray(center, dtype=float)
    positions = []
    for _ in range(1000 * n_ions):
        if len(positions) == n_ions:
            break
        candidate = center + size * rng.standard_normal(3)
        if all(np.linalg.norm(candidate - other) >= min_distance for other in positions):
            positions.append(candidate)
    if len(positions) < n_ions:
        raise ValueError("the cloud is too small for its ions to be min_distance apart")
    return np.array(positions)


class _Waveform:
    """Linear interpolation of the voltages of a waveform in time (holding its ends)."""

    def __init__(self, waveform=None, voltages=None, n_electrodes=0):
        if waveform is not None:
            self.times = waveform.times
            self.voltages = waveform.voltages
        else:
            self.times = np.zeros(1)
            self.voltages = (np.zeros(n_electrodes) if voltages is None else np.asarray(voltages, dtype=float))[None, :]

    def __call__(self, t):
        if len(self.times) == 1 or t <= self.times[0]:
            return self.voltages[0]
        if t >= self.times[-1]:
            return self.voltages[-1]
        k = int(np.searchsorted(self.times, t)) - 1
        u = (t - self.times[k]) / (self.times[k + 1] - self.times[k])
        return (1 - u) * self.voltages[k] + u * self.voltages[k + 1]


def simulate(potential, ions, positions, velocities=None, duration=10e-6, dt=None, waveform=None, voltages=None,
             full_rf=False, rf_phase=0.0, beams=(), field_noise=0.0, temperature=None, n_samples=200, seed=0,
             coulomb_interaction=True):
    """Simulate the motion of ions in a trap.

    Args:
        potential: Solved Trap or a HarmonicPotential (which only supports the pseudopotential).
        ions: Ion species of every ion (an Ion, an element, or a list of either).
        positions: Starting positions of the ions (µm) with shape (N, 3).
        velocities: Starting velocities (m/s), which are drawn at temperature if None (or zero).
        duration: Duration of the simulation (s).
        dt: Timestep (s), by default 1/40 of the RF period with full_rf and otherwise 5 ns.
        waveform: Waveform whose DC voltages the electrodes follow, if any.
        voltages: Constant DC voltages (V) if there is no waveform, where None means 0 V.
        full_rf: Specifies whether to apply the full RF drive (with micromotion) rather than the pseudopotential.
        rf_phase: Phase of the RF drive at the start (rad).
        beams: Doppler cooling beams (CoolingBeam).
        field_noise: Spectral density of the electric field noise at every ion (V^2/m^2/Hz), which heats the ions at
            q^2 S_E / (4 m hbar omega) quanta/s.
        temperature: Temperature (K) of the starting velocities if velocities is None.
        n_samples: Number of samples recorded.
        seed: Seed of the random numbers.
        coulomb_interaction: Specifies whether the ions repel each other.

    Returns:
        MDResult
    """
    if not isinstance(potential, (TrapPotential, HarmonicPotential)):
        potential = TrapPotential(potential)
    harmonic = isinstance(potential, HarmonicPotential)
    if harmonic and full_rf:
        raise ValueError("model potentials only support the pseudopotential")
    ions = _species(ions)
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    if len(ions) == 1 and len(positions) > 1:
        ions = ions * len(positions)
    n = len(ions)
    if len(positions) != n:
        raise ValueError(f"{n} ions need {n} positions, got {len(positions)}")
    rng = np.random.default_rng(seed)
    masses = np.array([ion.mass for ion in ions])
    charges = np.array([ion.charge for ion in ions])
    if velocities is None:
        velocities = thermal_velocities(ions, temperature, rng) if temperature else np.zeros((n, 3))
    velocities = np.asarray(velocities, dtype=float).reshape(n, 3).copy()

    trap = None if harmonic else potential.trap
    if dt is None:
        dt = (2 * np.pi / trap.Omega_RF / 40) if full_rf else 5e-9
    n_steps = max(int(np.ceil(duration / dt)), 1)
    schedule = _Waveform(waveform, voltages, 0 if harmonic else trap.n_electrodes)
    rf_weights = None if harmonic else trap.rf_weights()

    # Wavevector (1/m), linewidth (1/s), detuning (rad/s), and saturation of every beam for every ion (zero for ions
    # which a beam does not address)
    beam_arrays = []
    for beam in beams:
        K = np.zeros((n, 3))
        gamma = np.ones(n)
        detuning = np.zeros(n)
        saturation = np.zeros(n)
        for i, ion in enumerate(ions):
            parameters = beam.parameters(ion)
            if parameters is not None:
                K[i], gamma[i], detuning[i] = parameters
                saturation[i] = beam.saturation
        beam_arrays.append((K, gamma, detuning, saturation))

    def scattering_rates(v):
        for K, gamma, detuning, saturation in beam_arrays:
            delta = detuning - np.sum(K * v, axis=1)
            yield K, gamma / 2 * saturation / (1 + saturation + (2 * delta / gamma) ** 2)

    x = positions * 1e-6
    v = velocities
    lost = np.zeros(n, dtype=bool)
    region_low = np.array([-1e9, -1e9, 1.0]) if harmonic else None
    if not harmonic:
        from .lib import constant, expansion_region
        x_min, x_max, y_min, y_max = expansion_region()
        R = constant("SPH_R")
        z_top = constant("SPH_CENTER_Z_MIN") + constant("SPH_SPACING") * (constant("NSPH_Z") - 1) + R
        region_low = np.array([x_min - R, y_min - R, 1.0])
        region_high = np.array([x_max + R, y_max + R, z_top])
    else:
        region_high = np.array([1e9, 1e9, 1e9])

    def forces(x, v, t):
        F = np.zeros((n, 3))
        active = ~lost
        p = x[active] * 1e6
        if harmonic:
            _, grad, _, _ = potential.evaluate([ions[i] for i in np.flatnonzero(active)], p)
            F[active] = -grad * 1e6
        elif full_rf:
            total = schedule(t) + trap.V_RF * np.cos(trap.Omega_RF * t + rf_phase) * rf_weights
            _, grad = trap.potential(p, total, gradient=True)
            F[active] = -charges[active, None] * grad * 1e6
        else:
            _, grad, _, _ = potential.__class__(trap, schedule(t)).evaluate([ions[i] for i in np.flatnonzero(active)], p)
            F[active] = -grad * 1e6
        if coulomb_interaction and np.count_nonzero(active) > 1:
            _, grad_c = coulomb(p, charges[active])
            F[active] += -grad_c * 1e6
        # Mean scattering force of the cooling beams
        for K, rate in scattering_rates(v):
            F += HBAR * K * (rate * active)[:, None]
        return F

    sample_steps = np.unique(np.linspace(0, n_steps, n_samples).astype(int))
    times = np.zeros(len(sample_steps))
    out_x = np.zeros((len(sample_steps), n, 3))
    out_v = np.zeros((len(sample_steps), n, 3))
    scattered = np.zeros(n)
    sample = 0
    t = 0.0
    k_max = COULOMB_CONSTANT * charges.max() ** 2
    a = forces(x, v, t) / masses[:, None]
    for step in range(n_steps + 1):
        if sample < len(sample_steps) and step == sample_steps[sample]:
            times[sample] = t
            out_x[sample] = x * 1e6
            out_v[sample] = v
            sample += 1
        if step == n_steps:
            break
        # Close encounters between ions take the time sqrt(m d^3 / (k q^2)), which every step must resolve, so steps
        # are divided into substeps while ions are close
        n_sub = 1
        active = ~lost
        if coulomb_interaction and np.count_nonzero(active) > 1:
            p = x[active]
            d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            # Ions also close in on each other at their relative speed, which the substeps must follow too
            closing = 2 * np.linalg.norm(v[active], axis=1).max() + 1e-30
            encounter = min(np.sqrt(masses.min() * d.min() ** 3 / k_max), d.min() / closing)
            n_sub = int(min(max(np.ceil(dt / (0.03 * encounter)), 1), 2000))
        h = dt / n_sub
        for _ in range(n_sub):
            v_half = v + a * h / 2
            x = x + v_half * h
            t += h
            outside = np.any((x * 1e6 < region_low) | (x * 1e6 > region_high), axis=1) & ~lost
            if np.any(outside):
                lost |= outside
                v_half[outside] = 0.0
            a = forces(x, v_half, t) / masses[:, None]
            v = v_half + a * h / 2
            v[lost] = 0.0
        # Recoil of every scattered photon: one along the beam when absorbed and one in a random direction when emitted
        # (whose mean force is already in the forces above)
        for K, rate in scattering_rates(v):
            rate = rate * ~lost
            counts = rng.poisson(rate * dt)
            scattered += counts
            # Fluctuations of the number of absorbed photons about the mean, and the random emissions
            kick = (counts - rate * dt)[:, None] * HBAR * K
            total = int(counts.sum())
            if total:
                directions = rng.standard_normal((total, 3))
                directions /= np.linalg.norm(directions, axis=1)[:, None]
                owners = np.repeat(np.arange(n), counts)
                np.add.at(kick, owners, HBAR * np.linalg.norm(K[owners], axis=1)[:, None] * directions)
            v += kick / masses[:, None]
        if field_noise > 0:
            sigma = charges * np.sqrt(field_noise * dt / 2) / masses
            v[~lost] += sigma[~lost, None] * rng.standard_normal((np.count_nonzero(~lost), 3))
    return MDResult(times, out_x, out_v, lost, ions, scattered)


def rf_phase_scan(trap, ion, waveform, phases=None, dt=None, settle=1e-6, omega_axial=None, axis=(1.0, 0.0, 0.0)):
    """Simulate one ion through a waveform with the full RF drive at several RF phases, whose spread of final excitations
    shows how far the pseudopotential approximation holds for the waveform (as suggested by Conta et al.).

    Args:
        trap: Solved trap.
        ion: Ion species.
        waveform: Waveform.
        phases: RF phases at the start (rad), 8 evenly spaced phases by default.
        dt: Timestep (s), 1/40 of the RF period by default.
        settle: Time after the waveform which is simulated too (s).
        omega_axial: Frequency of the final well along axis (rad/s), from its curvature by default.
        axis: Axis of the motional excitation.

    Returns:
        Phases (rad) and the final motional excitation (quanta) at each
    """
    from .lib import motional_quanta

    phases = np.linspace(0, 2 * np.pi, 8, endpoint=False) if phases is None else np.asarray(phases, dtype=float)
    start = waveform.well_positions[0, 0]
    end = waveform.well_positions[-1, 0]
    if omega_axial is None:
        omega, axes = trap.normal_modes(ion, end, waveform.voltages[-1])
        omega_axial = abs(omega[int(np.argmax(np.abs(axes @ np.asarray(axis, dtype=float))))])
    excitations = []
    for phase in phases:
        result = simulate(trap, [ion], start[None, :], None, waveform.duration + settle, dt, waveform, full_rf=True, rf_phase=phase, n_samples=2)
        excitations.append(motional_quanta(ion, omega_axial, result.positions[-1, 0], result.velocities[-1, 0] * 1e6, end, axis))
    return phases, np.array(excitations)
