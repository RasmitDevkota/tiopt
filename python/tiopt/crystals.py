"""Coulomb crystals of trapped ions, i.e. the equilibrium positions of several ions (of one or more species) in the
potential of a trap or of a model trap, along with their normal modes of motion and Lamb-Dicke parameters.

The equilibrium minimizes the potential energy of every ion in the trap (the DC potential and the RF pseudopotential,
which depends on the species through q^2/m) plus the Coulomb energy between the ions, and the normal modes come from
the mass-weighted Hessian of the same energy. Mixed-species crystals, zigzags, and 2D and 3D crystals therefore all come
out of the same calculation. Modes with imaginary frequencies (which are returned as negative frequencies) mean that a
configuration is not stable, which find_crystal fixes by pushing the ions along the unstable mode and minimizing again.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from .lib import HBAR, ION_HEIGHT, Ion, TiOptError

COULOMB_CONSTANT = 8.9875517923e9  # 1/(4 pi epsilon_0) (N m^2/C^2)
ENERGY_SCALE = 1e-24  # energy unit of the minimization (J), about the energy of an ion 1 µm from the center of a 1 MHz well
AXES = ("x", "y", "z")


def _species(ions, n_ions=None):
    """Make a list of Ion objects from an Ion, an element, or a list of either (repeated n_ions times for one species)."""
    if isinstance(ions, (Ion, str)):
        ions = [ions] * (1 if n_ions is None else int(n_ions))
    species = {}
    out = []
    for ion in ions:
        element = ion if isinstance(ion, str) else ion.element
        if element not in species:
            species[element] = ion if isinstance(ion, Ion) else Ion(element)
        out.append(species[element])
    if not out:
        raise ValueError("a crystal needs at least one ion")
    return out


def coulomb(positions, charges, hessian=False):
    """Compute the Coulomb energy of ions, with its gradient and optionally its Hessian.

    Args:
        positions: Positions of the ions (µm) with shape (N, 3).
        charges: Charge of each ion (C).
        hessian: Specifies whether to also return the Hessian.

    Returns:
        Coulomb energy (J), its gradient (J/µm) with shape (N, 3), and its Hessian (J/µm^2) with shape (3N, 3N) if it
        is requested
    """
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    r = positions[:, None, :] - positions[None, :, :]
    d = np.linalg.norm(r, axis=-1)
    np.fill_diagonal(d, np.inf)
    # Coupling of each pair (J µm), i.e. U_ij = c_ij / d_ij with d_ij in µm
    c = COULOMB_CONSTANT * np.outer(charges, charges) * 1e6
    energy = 0.5 * np.sum(c / d)
    gradient = -np.sum((c / d ** 3)[:, :, None] * r, axis=1)
    if not hessian:
        return energy, gradient
    blocks = c[:, :, None, None] * (3 * r[:, :, :, None] * r[:, :, None, :] / d[:, :, None, None] ** 5 - np.eye(3) / d[:, :, None, None] ** 3)
    H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            H[3 * i:3 * i + 3, 3 * j:3 * j + 3] = -blocks[i, j]
            H[3 * i:3 * i + 3, 3 * i:3 * i + 3] += blocks[i, j]
    return energy, gradient, H


class TrapPotential:
    """Potential energy of ions in a trap (DC potential plus RF pseudopotential) for given electrode voltages."""

    def __init__(self, trap, voltages=None):
        trap._require_solved()
        self.trap = trap
        self.voltages = None if voltages is None else np.asarray(voltages, dtype=float)

    def evaluate(self, ions, positions, hessian=False):
        """Evaluate the potential energy (J) of each ion at its position (µm), its gradient (J/µm), its Hessian (J/µm^2,
        or None), and the number of ions outside of the spherical harmonic expansions.
        """
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        n = len(positions)
        U = np.zeros(n)
        grad = np.zeros((n, 3))
        H = np.zeros((n, 3, 3)) if hessian else None
        n_outside = 0
        groups = {}
        for i, ion in enumerate(ions):
            groups.setdefault(ion.element, (ion, []))[1].append(i)
        for ion, indices in groups.values():
            result = self.trap.energy_derivatives_batch(ion, positions[indices], self.voltages, hessian=hessian)
            U[indices] = result[0]
            grad[indices] = result[1]
            if hessian:
                H[indices] = result[2]
            n_outside += result[-1]
        return U, grad, H, n_outside


class HarmonicPotential:
    """Model of a linear Paul trap with given secular frequencies for a reference species, like the analytic traps of
    pytrans, for designing crystals without a trap.

    The first principal axis (the trap axis) is confined by the DC potential alone, and the two radial axes by the RF
    pseudopotential, minus the half of the axial DC curvature which Laplace's equation takes from each (plus any
    splitting between them). For other species, the DC curvature scales with the charge and the pseudopotential with
    q^2/m. Optional cubic and quartic terms along the trap axis (in J/µm^3 and J/µm^4 for the reference charge) make
    the axial potential anharmonic.
    """

    def __init__(self, ion, frequencies, center=(0.0, 0.0, ION_HEIGHT), axes=None, cubic=0.0, quartic=0.0):
        self.ion = _species(ion)[0]
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.center = np.asarray(center, dtype=float)
        self.axes = np.eye(3) if axes is None else np.asarray(axes, dtype=float)
        self.cubic = cubic
        self.quartic = quartic
        m0 = self.ion.mass
        k_total = m0 * self.frequencies ** 2 * 1e-12  # J/µm^2
        k_axial = k_total[0]
        # The radial DC curvatures share -k_axial between them, split so that the DC part has no extra anisotropy
        self.k_dc = np.array([k_axial, -k_axial / 2, -k_axial / 2])
        self.k_rf = k_total - self.k_dc
        if np.any(self.k_rf < -1e-30):
            raise ValueError("the radial frequencies must be large enough to hold the ion against the axial DC curvature")

    def evaluate(self, ions, positions, hessian=False):
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        u = (positions - self.center) @ self.axes.T
        q0, m0 = self.ion.charge, self.ion.mass
        n = len(positions)
        U = np.zeros(n)
        grad = np.zeros((n, 3))
        H = np.zeros((n, 3, 3)) if hessian else None
        for i, ion in enumerate(ions):
            k = self.k_dc * ion.charge / q0 + self.k_rf * (ion.charge / q0) ** 2 * m0 / ion.mass
            scale = ion.charge / q0
            x = u[i, 0]
            U[i] = 0.5 * np.sum(k * u[i] ** 2) + scale * (self.cubic * x ** 3 + self.quartic * x ** 4)
            g = k * u[i]
            g[0] += scale * (3 * self.cubic * x ** 2 + 4 * self.quartic * x ** 3)
            grad[i] = g @ self.axes
            if hessian:
                h = np.diag(k)
                h[0, 0] += scale * (6 * self.cubic * x + 12 * self.quartic * x ** 2)
                H[i] = self.axes.T @ h @ self.axes
        return U, grad, H, 0


def linear_chain(n_ions, omega_axial, ion="40Ca+"):
    """Compute the equilibrium positions of a chain of identical ions in a harmonic axial well (D. F. V. James, Appl.
    Phys. B 66, 181 (1998)).

    Args:
        n_ions: Number of ions.
        omega_axial: Axial secular frequency (rad/s).
        ion: Ion species.

    Returns:
        Axial positions of the ions (µm), in increasing order
    """
    ion = _species(ion)[0]
    length = (COULOMB_CONSTANT * ion.charge ** 2 / (ion.mass * omega_axial ** 2)) ** (1 / 3) * 1e6
    return length * _chain_positions(n_ions)


def _chain_positions(n_ions):
    """Dimensionless equilibrium positions of a chain in a harmonic well, by Newton's method."""
    if n_ions == 1:
        return np.zeros(1)
    u = np.linspace(-1, 1, n_ions) * 1.1 * n_ions ** 0.56
    for _ in range(100):
        d = u[:, None] - u[None, :]
        np.fill_diagonal(d, np.inf)
        force = u - np.sum(np.sign(d) / d ** 2, axis=1)
        J = np.diag(np.ones(n_ions) + np.sum(2 / np.abs(d) ** 3, axis=1)) - 2 / np.abs(d) ** 3 * (1 - np.eye(n_ions))
        step = np.linalg.solve(J, force)
        u -= step
        if np.max(np.abs(step)) < 1e-14:
            break
    return np.sort(u)


def chain_frequencies(n_ions, omega_axial, omega_radial):
    """Compute the axial and radial mode frequencies (rad/s) of a chain of identical ions (James 1998), with the radial
    modes in the order of the axial modes they come from (so the center-of-mass mode is the highest radial mode).
    """
    u = _chain_positions(n_ions)
    d = np.abs(u[:, None] - u[None, :])
    np.fill_diagonal(d, np.inf)
    A = np.diag(1 + np.sum(2 / d ** 3, axis=1)) - 2 / d ** 3 * (1 - np.eye(n_ions))
    mu = np.sort(np.linalg.eigvalsh(A))
    axial = omega_axial * np.sqrt(mu)
    radial = np.sqrt(np.maximum(omega_radial ** 2 - omega_axial ** 2 * (mu - 1) / 2, 0.0))
    return axial, radial


@dataclass
class Crystal:
    """Equilibrium of a Coulomb crystal and its normal modes.

    The modes are sorted by frequency, and vectors[m] holds the mass-weighted displacement of every ion in mode m
    (normalized over the whole crystal), from which the motion of ion i is vectors[m, i] / sqrt(mass_i) per unit of the
    normal coordinate.
    """
    ions: list
    positions: np.ndarray
    energy: float
    frequencies: np.ndarray
    vectors: np.ndarray
    n_outside: int = 0
    n_restarts: int = 0
    hessian: np.ndarray = field(default=None, repr=False)

    @property
    def n_ions(self):
        """Number of ions."""
        return len(self.ions)

    @property
    def stable(self):
        """Whether every mode has a real frequency, i.e. the crystal is at a minimum of its energy."""
        return bool(np.all(self.frequencies > 0))

    @property
    def masses(self):
        """Mass of each ion (kg)."""
        return np.array([ion.mass for ion in self.ions])

    def directions(self):
        """Fraction of the motion of each mode along x, y, and z, with shape (3N, 3)."""
        return np.sum(self.vectors ** 2, axis=1)

    def mode_axes(self):
        """Axis (x, y, or z) along which each mode mostly moves."""
        return [AXES[k] for k in np.argmax(self.directions(), axis=1)]

    def modes_along(self, axis):
        """Indices of the modes which mostly move along an axis ("x", "y", or "z")."""
        return [m for m, a in enumerate(self.mode_axes()) if a == axis]

    def participation(self):
        """Fraction of each mode carried by each ion, with shape (3N, N)."""
        return np.sum(self.vectors ** 2, axis=2)

    def lamb_dicke(self, k_vector, modes=None):
        """Compute the Lamb-Dicke parameter of every ion in every mode for light with wavevector k_vector (1/m), or
        the difference of the wavevectors of a pair of Raman beams.

        Args:
            k_vector: Wavevector (1/m).
            modes: Indices of the modes, or None for every mode.

        Returns:
            Lamb-Dicke parameters with shape (N, n_modes), whose signs give the phases of the ions in each mode
        """
        modes = range(len(self.frequencies)) if modes is None else modes
        k = np.asarray(k_vector, dtype=float)
        eta = np.zeros((self.n_ions, len(modes)))
        for column, m in enumerate(modes):
            omega = abs(self.frequencies[m])
            eta[:, column] = (self.vectors[m] @ k) * np.sqrt(HBAR / (2 * self.masses * omega))
        return eta

    def spacings(self):
        """Distances between neighboring ions along the softest axis of the crystal (µm)."""
        if self.n_ions < 2:
            return np.zeros(0)
        centered = self.positions - self.positions.mean(axis=0)
        axis = np.linalg.svd(centered)[2][0]
        projection = np.sort(centered @ axis)
        return np.diff(projection)

    def summary(self):
        """Lines of text which describe the crystal and its modes."""
        species = {}
        for ion in self.ions:
            species[ion.element] = species.get(ion.element, 0) + 1
        lines = [
            f"{self.n_ions} ions ({', '.join(f'{n} {element}' for element, n in species.items())}), "
            + ("stable" if self.stable else "NOT stable (imaginary mode frequencies)"),
        ]
        if self.n_ions > 1:
            spacings = self.spacings()
            lines.append(f"Spacing along the crystal: {spacings.min():.3f} to {spacings.max():.3f} µm")
        extent = np.ptp(self.positions, axis=0)
        lines.append("Extent: " + ", ".join(f"{a} {e:.3f} µm" for a, e in zip(AXES, extent)))
        for axis in AXES:
            modes = self.modes_along(axis)
            if modes:
                lines.append(f"{axis} modes (MHz): " + ", ".join(f"{self.frequencies[m] / (2 * np.pi * 1e6):.4f}" for m in modes))
        if self.n_outside:
            lines.append(f"Warning: {self.n_outside} ions are outside of the spherical harmonic expansions")
        return lines


def _energy(x, potential, ions, charges, hessian):
    positions = x.reshape(-1, 3)
    U, grad, H, n_outside = potential.evaluate(ions, positions, hessian)
    result = coulomb(positions, charges, hessian)
    energy = (U.sum() + result[0]) / ENERGY_SCALE
    gradient = (grad + result[1]).reshape(-1) / ENERGY_SCALE
    if not hessian:
        return energy, gradient, n_outside
    full = result[2].copy()
    for i in range(len(ions)):
        full[3 * i:3 * i + 3, 3 * i:3 * i + 3] += H[i]
    return energy, gradient, full / ENERGY_SCALE, n_outside


def normal_modes(hessian, masses):
    """Compute the frequencies (rad/s, negative for unstable modes) and mass-weighted vectors (with shape (3N, N, 3)) of
    the normal modes from the Hessian of the energy (J/µm^2) and the masses of the ions (kg).
    """
    masses = np.repeat(np.asarray(masses, dtype=float), 3)
    weighted = hessian * 1e12 / np.sqrt(np.outer(masses, masses))
    values, vectors = np.linalg.eigh(0.5 * (weighted + weighted.T))
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    frequencies = np.sign(values) * np.sqrt(np.abs(values))
    return frequencies, vectors.T.reshape(len(values), -1, 3)


def _initial_positions(potential, ions, position, rng):
    """Start with a linear chain along the softest axis of the potential of the first ion at position."""
    reference = ions[0]
    _, _, H, _ = potential.evaluate([reference], position[None, :], hessian=True)
    values, vectors = np.linalg.eigh(H[0])
    axis = vectors[:, 0]
    # A soft or unstable direction still spreads the chain over a reasonable length
    curvature = max(values[0], 1e-3 * max(values[-1], 1e-40), 1e-40)
    omega = np.sqrt(curvature * 1e12 / reference.mass)
    length = (COULOMB_CONSTANT * reference.charge ** 2 / (reference.mass * omega ** 2)) ** (1 / 3) * 1e6
    u = _chain_positions(len(ions)) * length
    positions = position + u[:, None] * axis[None, :]
    return positions + 1e-3 * max(length, 1.0) * rng.standard_normal(positions.shape)


def find_crystal(potential, ions, n_ions=None, voltages=None, position=None, initial_positions=None, seed=0,
                 max_restarts=5, tolerance=1e-10):
    """Find the equilibrium of a Coulomb crystal and its normal modes.

    Args:
        potential: Solved Trap (with the given voltages), TrapPotential, or HarmonicPotential.
        ions: Ion species of every ion (an Ion, an element such as "40Ca+", or a list of either, e.g. for a
            mixed-species crystal).
        n_ions: Number of ions if ions is a single species.
        voltages: Voltage of each electrode (V) if potential is a Trap, where None means 0 V on every electrode.
        position: Point near which the crystal forms (µm), which is the RF null at x = 0 (or the center of a model
            potential) by default.
        initial_positions: Starting positions of the ions (µm), instead of a linear chain along the softest axis.
        seed: Seed of the small random displacements which break the symmetry of the starting chain.
        max_restarts: Largest number of times to push the ions along an unstable mode and minimize again.
        tolerance: Tolerance on the largest force on any ion (in units of ENERGY_SCALE per µm).

    Returns:
        Crystal
    """
    if not isinstance(potential, (TrapPotential, HarmonicPotential)):
        potential = TrapPotential(potential, voltages)
    ions = _species(ions, n_ions)
    charges = np.array([ion.charge for ion in ions])
    masses = np.array([ion.mass for ion in ions])
    rng = np.random.default_rng(seed)

    if initial_positions is not None:
        x0 = np.asarray(initial_positions, dtype=float).reshape(len(ions), 3)
    else:
        if position is None:
            if isinstance(potential, HarmonicPotential):
                position = potential.center
            elif potential.trap.n_rf > 0:
                position = potential.trap.find_rf_null((0.0, 0.0, None))
            else:
                position = (0.0, 0.0, ION_HEIGHT)
        position = np.asarray(position, dtype=float)
        if isinstance(potential, TrapPotential):
            try:
                position = potential.trap.find_minimum(ions[0], position, potential.voltages)
            except TiOptError:
                pass
        x0 = _initial_positions(potential, ions, position, rng)

    def fun(x):
        energy, gradient, _ = _energy(x, potential, ions, charges, False)
        return energy, gradient

    def hess(x):
        return _energy(x, potential, ions, charges, True)[2]

    x = x0.reshape(-1)
    restarts = 0
    while True:
        result = optimize.minimize(fun, x, jac=True, hess=hess, method="trust-exact", options=dict(gtol=tolerance, maxiter=500))
        x = result.x
        energy, _, H, n_outside = _energy(x, potential, ions, charges, True)
        frequencies, vectors = normal_modes(H * ENERGY_SCALE, masses)
        if frequencies[0] > 0 or restarts >= max_restarts:
            break
        # Push the ions along the unstable mode (by about a tenth of their spacing) and minimize again
        restarts += 1
        displacement = vectors[0] / np.sqrt(masses)[:, None]
        displacement /= np.abs(displacement).max()
        spacing = Crystal(ions, x.reshape(-1, 3), 0.0, frequencies, vectors).spacings()
        scale = 0.1 * (spacing.min() if len(spacing) else 1.0)
        x = x + scale * displacement.reshape(-1) * (1 if rng.random() < 0.5 else -1)

    return Crystal(ions, x.reshape(-1, 3), energy * ENERGY_SCALE, frequencies, vectors, n_outside, restarts, H * ENERGY_SCALE)
