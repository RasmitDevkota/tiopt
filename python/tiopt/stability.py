"""Stability and depth of trapping sites beyond the pseudopotential approximation, i.e. the Floquet (Mathieu) analysis
of the full RF drive, stability diagrams, the escape point and true depth of a trap, anharmonicity, micromotion, and the
orientation of the principal axes.

Near a point r0, the equation of motion of an ion in the RF and DC fields is
    m x'' = -q (H_dc + V_RF cos(Omega t) H_rf) x,
where H_dc and H_rf are the Hessians of the DC potential and of the RF potential at 1 V. With tau = Omega t / 2 this is
the Mathieu equation x'' + (A - 2 Q cos(2 tau)) x = 0 with the matrices
    A = 4 q H_dc / (m Omega^2),    Q = -2 q V_RF H_rf / (m Omega^2),
which hold the familiar a and q parameters along their axes. When A and Q do not share their axes (e.g. in asymmetric
surface traps, where the DC electrodes rotate the radial axes), the three directions are coupled, so the motion is found
here from the monodromy matrix of the full 3D equation over one RF period: the motion is stable if all of its
eigenvalues lie on the unit circle, and their phases give the secular frequencies including the corrections beyond the
pseudopotential approximation.
"""

from dataclasses import dataclass

import numpy as np

from .lib import E_CHARGE, HBAR, LENGTH_SCALE

AXES = ("x", "y", "z")


def mathieu_matrices(trap, ion, position, voltages=None):
    """Compute the Mathieu matrices A and Q of an ion at a position.

    Args:
        trap: Solved trap.
        ion: Ion species.
        position: Position (µm).
        voltages: Voltage of each electrode (V), where None means 0 V on every electrode.

    Returns:
        A and Q, each with shape (3, 3)
    """
    _, _, H_dc = trap.evaluate(position, voltages)
    _, _, H_rf = trap.evaluate(position, trap.rf_weights())
    scale = 4 * ion.charge / (ion.mass * trap.Omega_RF ** 2) / LENGTH_SCALE ** 2
    return scale * H_dc, -0.5 * scale * trap.V_RF * H_rf


def _monodromy(A, Q, n_steps):
    """Monodromy matrices of x'' = -(A - 2 Q cos(2 tau)) x over one period (tau from 0 to pi) for a batch of A and Q
    with shape (..., 3, 3), by the classical Runge-Kutta method.
    """
    A = np.asarray(A, dtype=float)
    Q = np.asarray(Q, dtype=float)
    batch = np.broadcast_shapes(A.shape[:-2], Q.shape[:-2])
    A = np.broadcast_to(A, batch + (3, 3))
    Q = np.broadcast_to(Q, batch + (3, 3))
    Y = np.zeros(batch + (6, 6))
    Y[..., :, :] = np.eye(6)
    h = np.pi / n_steps

    def derivative(tau, Y):
        K = A - 2 * Q * np.cos(2 * tau)
        dY = np.empty_like(Y)
        dY[..., :3, :] = Y[..., 3:, :]
        dY[..., 3:, :] = -K @ Y[..., :3, :]
        return dY

    tau = 0.0
    for _ in range(n_steps):
        k1 = derivative(tau, Y)
        k2 = derivative(tau + h / 2, Y + h / 2 * k1)
        k3 = derivative(tau + h / 2, Y + h / 2 * k2)
        k4 = derivative(tau + h, Y + h * k3)
        Y = Y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        tau += h
    return Y


@dataclass
class FloquetResult:
    """Floquet analysis of an ion at a point: whether its motion is stable, the largest magnitude of the eigenvalues of
    the monodromy matrix (1 if stable), the secular frequencies (rad/s) and their directions (as rows), the Mathieu
    matrices, and the a and q parameters along the principal axes of Q.
    """
    stable: bool
    max_multiplier: float
    frequencies: np.ndarray
    directions: np.ndarray
    A: np.ndarray
    Q: np.ndarray
    a: np.ndarray
    q: np.ndarray


def floquet(A, Q, Omega_RF, n_steps=2000):
    """Find the stability and secular frequencies of the motion of x'' + (A - 2 Q cos(2 tau)) x = 0.

    Args:
        A: Mathieu matrix A with shape (3, 3).
        Q: Mathieu matrix Q with shape (3, 3).
        Omega_RF: Angular frequency of the RF drive (rad/s), which converts the characteristic exponents into
            frequencies.
        n_steps: Number of steps over one RF period.

    Returns:
        FloquetResult
    """
    A = np.asarray(A, dtype=float)
    Q = np.asarray(Q, dtype=float)
    M = _monodromy(A, Q, n_steps)
    values, vectors = np.linalg.eig(M)
    max_multiplier = float(np.max(np.abs(values)))
    stable = max_multiplier < 1 + 1e-6
    # Each conjugate pair of eigenvalues exp(+-i pi beta) gives one characteristic exponent beta in (0, 1) in the
    # lowest stability region, and the secular frequency beta Omega / 2
    beta = np.abs(np.angle(values)) / np.pi
    order = np.argsort(beta)
    frequencies = []
    directions = []
    for k in order[1::2] if len(order) == 6 else order:
        frequencies.append(beta[k] * Omega_RF / 2)
        direction = np.abs(vectors[:3, k])
        directions.append(direction / max(np.linalg.norm(direction), 1e-300))
    q_values, q_axes = np.linalg.eigh(0.5 * (Q + Q.T))
    a_values = np.einsum("ij,jk,ki->i", q_axes.T, A, q_axes)
    return FloquetResult(stable, max_multiplier, np.array(frequencies), np.array(directions), A, Q, a_values, q_values)


def floquet_at(trap, ion, position, voltages=None, n_steps=2000):
    """Run the Floquet analysis (see floquet) of an ion at a position in a trap."""
    A, Q = mathieu_matrices(trap, ion, position, voltages)
    return floquet(A, Q, trap.Omega_RF, n_steps)


@dataclass
class StabilityDiagram:
    """Stability of an ion over a grid of RF amplitudes and DC scales: the RF amplitudes (V), the factors which scale
    the DC voltages, the largest magnitude of the eigenvalues of the monodromy matrix at each point (with shape
    (len(dc_scales), len(V_RF))), and the a and q parameters of the operating point.
    """
    V_RF: np.ndarray
    dc_scales: np.ndarray
    max_multiplier: np.ndarray
    operating_V_RF: float
    q_max: float
    a_max: float

    @property
    def stable(self):
        """Whether each point of the grid is stable."""
        return self.max_multiplier < 1 + 1e-6


def stability_diagram(trap, ion, position, voltages=None, V_RF_max=None, dc_scale_max=20.0, n_points=61, n_steps=400):
    """Map where an ion at a position is stable as the RF amplitude and the DC voltages change, i.e. a stability diagram
    in the a-q plane for the actual (possibly coupled) axes of the trap.

    Args:
        trap: Solved trap.
        ion: Ion species.
        position: Position (µm).
        voltages: DC voltages of the operating point (V), which are scaled from -dc_scale_max to dc_scale_max times.
        V_RF_max: Largest RF amplitude (V), which is three times the operating amplitude by default.
        dc_scale_max: Largest factor by which the DC voltages are scaled.
        n_points: Number of points along each axis of the grid.
        n_steps: Number of steps over one RF period.

    Returns:
        StabilityDiagram
    """
    A0, Q0 = mathieu_matrices(trap, ion, position, voltages)
    Q_unit = Q0 / trap.V_RF if trap.V_RF != 0 else Q0
    V_RF_max = 3 * trap.V_RF if V_RF_max is None else V_RF_max
    V_RF = np.linspace(0.0, V_RF_max, n_points)
    dc_scales = np.linspace(-dc_scale_max, dc_scale_max, n_points)
    A = dc_scales[:, None, None, None] * A0[None, None, :, :]
    Q = V_RF[None, :, None, None] * Q_unit[None, None, :, :]
    M = _monodromy(A, Q, n_steps)
    max_multiplier = np.max(np.abs(np.linalg.eigvals(M)), axis=-1)
    q_max = float(np.max(np.abs(np.linalg.eigvalsh(0.5 * (Q0 + Q0.T)))))
    a_max = float(np.max(np.abs(np.linalg.eigvalsh(0.5 * (A0 + A0.T)))))
    return StabilityDiagram(V_RF, dc_scales, max_multiplier, trap.V_RF, q_max, a_max)


class AnalyticalEnergy:
    """Potential energy (J) of an ion from the gapless plane potentials of the electrodes, which can be evaluated
    anywhere above the trap (unlike the spherical harmonic expansions, which end about 70 µm above the surface).
    """

    STEP = 0.02  # finite difference step (µm)

    def __init__(self, trap, ion, voltages=None):
        self.trap = trap
        self.ion = ion
        voltages = trap.zero_voltages() if voltages is None else np.asarray(voltages, dtype=float)
        self.dc = [(e, v) for e, v in enumerate(voltages) if e >= trap.n_rf and v != 0.0]
        has_rf = trap.n_rf > 0 and trap.V_RF != 0.0 and trap.Omega_RF != 0.0
        self.prefactor = (ion.charge * trap.V_RF) ** 2 / (4 * ion.mass * (trap.Omega_RF * LENGTH_SCALE) ** 2) if has_rf else 0.0

    def __call__(self, points):
        points = np.asarray(points, dtype=float)
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        U = np.zeros(len(flat))
        for e, v in self.dc:
            U += self.ion.charge * v * self.trap.analytical_electrode_potential(e, flat)
        if self.prefactor:
            h = self.STEP / 2
            offsets = h * np.vstack([np.eye(3), -np.eye(3)])
            stencil = flat[:, None, :] + offsets[None, :, :]
            phi = sum(self.trap.analytical_electrode_potential(e, stencil) for e in range(self.trap.n_rf))
            gradient = (phi[:, :3] - phi[:, 3:]) / (2 * h)
            U += self.prefactor * np.sum(gradient ** 2, axis=1)
        return U.reshape(shape)

    def derivatives(self, point):
        """Gradient (J/µm) and Hessian (J/µm^2) at a point, by finite differences."""
        h = self.STEP
        point = np.asarray(point, dtype=float)
        offsets = [np.zeros(3)]
        for i in range(3):
            for sign in (1, -1):
                offsets.append(sign * h * np.eye(3)[i])
        for i in range(3):
            for j in range(i + 1, 3):
                for si in (1, -1):
                    for sj in (1, -1):
                        offsets.append(h * (si * np.eye(3)[i] + sj * np.eye(3)[j]))
        U = self(point[None, :] + np.array(offsets))
        gradient = np.zeros(3)
        H = np.zeros((3, 3))
        for i in range(3):
            plus, minus = U[1 + 2 * i], U[2 + 2 * i]
            gradient[i] = (plus - minus) / (2 * h)
            H[i, i] = (plus - 2 * U[0] + minus) / h ** 2
        k = 7
        for i in range(3):
            for j in range(i + 1, 3):
                H[i, j] = H[j, i] = (U[k] - U[k + 1] - U[k + 2] + U[k + 3]) / (4 * h ** 2)
                k += 4
        return gradient, H


@dataclass
class EscapeResult:
    """Escape point of a trapping site: the minimum (µm), the saddle point through which the ion escapes most easily
    (µm), the depth of the trap (eV), whether the saddle was refined to a stationary point with one unstable direction,
    the barrier along the vertical through the minimum (eV, or inf if the potential still rises at the top of the
    search), the path of least resistance from the minimum out of the searched region (µm), and whether the highest
    point of the path is on the edge of the searched region (so that the depth is only a lower bound).
    """
    minimum: np.ndarray
    saddle: np.ndarray
    depth: float
    refined: bool
    vertical_depth: float
    path: np.ndarray
    lower_bound: bool = False


def _newton(energy, point, basis=None, max_iterations=60, max_step=2.0):
    """Find the stationary point of the energy nearest to a point by Newton's method, limiting each step and moving only
    within the span of the columns of basis (if given).
    """
    point = np.asarray(point, dtype=float)
    basis = np.eye(3) if basis is None else basis
    for _ in range(max_iterations):
        gradient, H = energy.derivatives(point)
        try:
            step = -np.linalg.solve(basis.T @ H @ basis, basis.T @ gradient)
        except np.linalg.LinAlgError:
            return point, False
        length = np.linalg.norm(step)
        if length > max_step:
            step *= max_step / length
        point = point + basis @ step
        if point[2] < 1.0:
            return point, False
        if length < 1e-6:
            return point, True
    return point, False


def _minimax_path(U, boundary, start):
    """Path from a point of a grid of energies (given by its flat index) out through the boundary of the grid whose
    highest energy is lowest, which follows the minimum spanning tree of the grid (with each edge weighted by the higher
    energy of its ends).

    Returns:
        Flat indices of the points along the path, starting at start
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree

    shape = U.shape
    flat = U.reshape(-1)
    index = np.arange(flat.size).reshape(shape)
    # Weights are shifted to be positive, since zeros are not edges
    shift = -min(flat.min(), 0.0) + 1e-6 * max(np.abs(flat).max(), 1e-30) + 1e-300
    rows, cols, weights = [], [], []
    for axis in range(len(shape)):
        a = np.take(index, range(shape[axis] - 1), axis=axis).reshape(-1)
        b = np.take(index, range(1, shape[axis]), axis=axis).reshape(-1)
        rows.append(a)
        cols.append(b)
        weights.append(np.maximum(flat[a], flat[b]) + shift)
    outside = flat.size
    border = index[boundary]
    rows.append(border)
    cols.append(np.full(len(border), outside))
    weights.append(flat[border] + shift)
    graph = coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), shape=(outside + 1, outside + 1)).tocsr()
    tree = minimum_spanning_tree(graph)
    tree = tree + tree.T
    _, predecessors = breadth_first_order(tree, start, directed=False, return_predecessors=True)
    path = []
    node = predecessors[outside]
    while node >= 0 and node != start:
        path.append(node)
        node = predecessors[node]
    path.append(start)
    return np.array(path[::-1])


def _search(energy, minimum, U0, basis, across, extent, height, spacing):
    """Search a grid around a minimum (in the cross-section across the direction across, if given) for the path of least
    resistance out of it, and refine its highest point to a saddle point.

    Returns:
        Saddle point (µm), barrier (J), whether it was refined, whether the highest point is on the side of the grid
        (so that the barrier is only a lower bound), and the path (µm)
    """
    horizontal = np.arange(-extent, extent + spacing / 2, spacing)
    heights = np.arange(1.0, height + spacing / 2, spacing)
    if across is not None:
        A, Z = np.meshgrid(horizontal, heights, indexing="ij")
        points = minimum + A[..., None] * across + (Z - minimum[2])[..., None] * np.array([0.0, 0.0, 1.0])
    else:
        X, Y, Z = np.meshgrid(horizontal, horizontal, heights, indexing="ij")
        points = np.stack([minimum[0] + X, minimum[1] + Y, Z], axis=-1)
    U = energy(points.reshape(-1, 3)).reshape(points.shape[:-1]) - U0
    boundary = np.zeros(U.shape, dtype=bool)
    for axis in range(U.ndim):
        boundary[(slice(None),) * axis + (0,)] = True
        boundary[(slice(None),) * axis + (-1,)] = True

    flat_points = points.reshape(-1, 3)
    start = int(np.argmin(np.sum((flat_points - minimum) ** 2, axis=1)))
    nodes = _minimax_path(U, boundary, start)
    flat_U = U.reshape(-1)
    path = flat_points[nodes]
    peak = int(np.argmax(flat_U[nodes]))
    saddle = path[peak].copy()
    depth = float(flat_U[nodes][peak])
    # A path which leaves through the bottom (onto the electrodes) has really escaped
    lower_bound = bool(boundary.reshape(-1)[nodes[peak]]) and saddle[2] > 1.0 + spacing

    refined = False
    candidate, converged = _newton(energy, saddle, basis)
    if converged:
        _, H = energy.derivatives(candidate)
        U_candidate = float(energy(candidate)) - U0
        if np.sum(np.linalg.eigvalsh(basis.T @ H @ basis) < 0) == 1 and 0 < U_candidate <= depth + 0.1 * abs(depth):
            saddle, depth, refined = candidate, U_candidate, True
    return saddle, depth, refined, lower_bound and not refined, path


def escape_point(trap, ion, position, voltages=None, extent=None, height=None, spacing=None, cross_section=None):
    """Find the true depth of a trapping site from the lowest saddle point of the potential energy around it, rather
    than from the barrier along the vertical alone (like the saddle search of the electrode package).

    The potential energy is evaluated on a grid around the minimum, and the path out of the grid whose highest point is
    lowest (the minimax path, which follows the minimum spanning tree of the grid) gives the escape route, e.g. off to
    the side of a well or along the rising RF null into the center of a junction rather than straight up. Its highest
    point then starts Newton's method on the gradient of the potential energy, which converges to the saddle point. The
    potential energy comes from the gapless plane potentials of the electrodes (see AnalyticalEnergy), so the search
    reaches escape points high above the trap, and points within 1 µm of the surface count as escaped (since the ion
    would hit the electrodes).

    Without axial confinement (e.g. for the pseudopotential alone), an ion escapes along the RF null for free, so the
    search is then made in the cross-section of the trap through the minimum, which gives the usual depth of the
    pseudopotential.

    Args:
        trap: Solved trap.
        ion: Ion species.
        position: Point near the trapping site (µm).
        voltages: Voltage of each electrode (V), where None means 0 V on every electrode (i.e. the pseudopotential).
        extent: Half-width of the searched region around the minimum (µm), 2.5 times its height by default (doubled up
            to twice while the ion escapes through its side).
        height: Height of the top of the searched region (µm), 3 times the height of the minimum by default.
        spacing: Spacing of the grid (µm), chosen for about 70 points across in 3D (and 200 in a cross-section).
        cross_section: Specifies whether to search in the cross-section across the softest axis, which by default is
            done if that axis is much softer than the others.

    Returns:
        EscapeResult
    """
    voltages = trap.zero_voltages() if voltages is None else np.asarray(voltages, dtype=float)
    energy = AnalyticalEnergy(trap, ion, voltages)
    minimum = np.asarray(position, dtype=float)
    try:
        minimum = trap.find_minimum(ion, position, voltages)
    except Exception:
        pass

    _, H = energy.derivatives(minimum)
    values, vectors = np.linalg.eigh(H)
    if cross_section is None:
        cross_section = values[0] < 1e-2 * abs(values[-1])
    if cross_section:
        axis = vectors[:, 0]
        across = np.cross(axis, [0.0, 0.0, 1.0])
        across = across / np.linalg.norm(across) if np.linalg.norm(across) > 1e-9 else np.array([0.0, 1.0, 0.0])
        basis = np.column_stack([across, [0.0, 0.0, 1.0]])
    else:
        basis = np.eye(3)

    # Polish the minimum on the same potential energy as the search
    polished, converged = _newton(energy, minimum, basis, 30, 1.0)
    if converged:
        minimum = polished
    U0 = float(energy(minimum))
    automatic = extent is None and spacing is None
    extent = 2.5 * minimum[2] if extent is None else extent
    height = 3 * minimum[2] if height is None else height

    # The search grows (up to twice, unless its size was given) while the path of least resistance leaves through the
    # side of the searched region still rising, since the saddle point is then further away
    for _ in range(3 if automatic else 1):
        grid_spacing = max(2 * extent / (200 if cross_section else 70), 0.5) if spacing is None else spacing
        saddle, depth, refined, lower_bound, path = _search(energy, minimum, U0, basis, across if cross_section else None, extent, height, grid_spacing)
        if not lower_bound:
            break
        extent *= 2

    # Barrier along the vertical through the minimum
    line = minimum + np.linspace(0, height - minimum[2], 400)[:, None] * np.array([0.0, 0.0, 1.0])
    U_line = energy(line) - U0
    k = int(np.argmax(U_line))
    vertical = U_line[k] if k < len(U_line) - 1 else np.inf

    return EscapeResult(minimum, saddle, depth / E_CHARGE, refined, vertical / E_CHARGE, path, lower_bound)


@dataclass
class Anharmonicity:
    """Anharmonicity of the potential along an axis: the axis, the curvature (J/m^2), the cubic and quartic
    coefficients of the potential energy (J/m^3 and J/m^4), the lengths over which the cubic and quartic terms grow as
    large as the harmonic term (µm), the shift of the secular frequency per added phonon (rad/s), and the frequency of the
    axis (rad/s).
    """
    axis: np.ndarray
    curvature: float
    cubic: float
    quartic: float
    cubic_length: float
    quartic_length: float
    shift_per_phonon: float
    frequency: float


def anharmonicity(trap, ion, position, voltages=None, axis=None, extent=3.0, n_samples=41):
    """Fit the potential energy along an axis through a minimum with a polynomial, and find how anharmonic it is.

    The cubic and quartic terms (c3 x^3 and c4 x^4) shift the secular frequency by
        delta omega = (3 c4 hbar / (m^2 omega^2) - 15 c3^2 hbar / (2 m^3 omega^4)) (n + 1)
    between phonon numbers n and n + 1 (from first-order perturbation theory in c4 and second-order in c3), which
    broadens the motional sidebands of hot ions.

    Args:
        trap: Solved trap.
        ion: Ion species.
        position: Minimum of the potential energy (µm).
        voltages: Voltage of each electrode (V).
        axis: Direction of the axis, which is the softest principal axis by default.
        extent: Half-length of the fitted interval (µm).
        n_samples: Number of points fitted.

    Returns:
        Anharmonicity
    """
    position = np.asarray(position, dtype=float)
    if axis is None:
        _, _, H = trap.energy_derivatives(ion, position, voltages)
        axis = np.linalg.eigh(H)[1][:, 0]
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    u = np.linspace(-extent, extent, n_samples)
    U = trap.potential_energy(ion, position + u[:, None] * axis[None, :], voltages)
    coefficients = np.polynomial.polynomial.polyfit(u * LENGTH_SCALE, U - U[n_samples // 2], 6)
    k = 2 * coefficients[2]
    c3, c4 = coefficients[3], coefficients[4]
    m = ion.mass
    omega = np.sqrt(abs(k) / m)
    shift = 3 * c4 * HBAR / (m ** 2 * omega ** 2) - 7.5 * c3 ** 2 * HBAR / (m ** 3 * omega ** 4) if omega > 0 else 0.0
    cubic_length = abs(k / (2 * c3)) / LENGTH_SCALE if c3 != 0 else np.inf
    quartic_length = np.sqrt(abs(k / (2 * c4))) / LENGTH_SCALE if c4 != 0 else np.inf
    return Anharmonicity(axis, k, c3, c4, cubic_length, quartic_length, shift, omega)


def micromotion(trap, ion, position, k_vector=None):
    """Compute the excess micromotion of an ion held at a position off the RF null.

    Args:
        trap: Solved trap.
        ion: Ion species.
        position: Position of the ion (µm).
        k_vector: Wavevector of a beam (1/m), for the modulation index of its micromotion sidebands.

    Returns:
        Amplitude of the micromotion (m) as a vector, and the modulation index of the beam (or None)
    """
    _, grad, _ = trap.evaluate(position, trap.rf_weights())
    amplitude = ion.charge * trap.V_RF * grad / LENGTH_SCALE / (ion.mass * trap.Omega_RF ** 2)
    index = None if k_vector is None else float(abs(np.dot(np.asarray(k_vector, dtype=float), amplitude)))
    return amplitude, index


def principal_axes(trap, ion, position, voltages=None):
    """Compute the secular frequencies (rad/s) and principal axes (as rows) of the pseudopotential at a position, along
    with the angle (degrees) by which the radial axes are rotated about the trap axis (x) from y and z, which pytrans and
    Sion call the tilt or rotation angle of the radial modes.
    """
    omega, axes = trap.normal_modes(ion, position, voltages)
    # The two axes most perpendicular to x are the radial axes, and the lower one is measured from y
    radial = np.argsort(np.abs(axes[:, 0]))[:2]
    lower = radial[np.argmin(np.abs(omega[radial]))]
    v = axes[lower]
    tilt = np.degrees(np.arctan2(v[2], v[1]))
    tilt = (tilt + 90) % 180 - 90
    return omega, axes, tilt


def report(trap, ion, position, voltages=None, k_vector=None, n_ions=2):
    """Analyze a trapping site, like analyze_static of the electrode package: its minimum, micromotion, secular
    frequencies from the pseudopotential and the Floquet analysis, Mathieu parameters, principal axes, depth and escape
    point, anharmonicity, and the modes of a small crystal.

    Args:
        trap: Solved trap.
        ion: Ion species.
        position: Point near the trapping site (µm).
        voltages: Voltage of each electrode (V), where None means 0 V on every electrode.
        k_vector: Wavevector of a beam (1/m), for the modulation index of its micromotion sidebands.
        n_ions: Number of ions of the crystal whose modes are reported (0 to skip it).

    Returns:
        Lines of text
    """
    from . import crystals

    voltages = trap.zero_voltages() if voltages is None else np.asarray(voltages, dtype=float)
    lines = []
    minimum = trap.find_minimum(ion, position, voltages)
    lines.append("Minimum: (" + ", ".join(f"{v:.3f}" for v in minimum) + ") µm")
    if trap.n_rf > 0:
        try:
            null = trap.find_rf_null(minimum)
            lines.append(f"Distance from the RF null: {np.linalg.norm(minimum - null) * 1e3:.1f} nm")
        except Exception:
            pass
        amplitude, index = micromotion(trap, ion, minimum, k_vector)
        lines.append(f"Excess micromotion amplitude: {np.linalg.norm(amplitude) * 1e9:.2f} nm" + (f" (modulation index {index:.3g})" if index is not None else ""))
    omega, axes, tilt = principal_axes(trap, ion, minimum, voltages)
    lines.append("Pseudopotential frequencies: " + ", ".join(f"{w / (2 * np.pi * 1e6):.4f}" for w in omega) + " MHz")
    for w, axis in zip(omega, axes):
        angles = np.degrees(np.arccos(np.clip(np.abs(axis), 0, 1)))
        lines.append(f"  axis ({axis[0]:+.3f}, {axis[1]:+.3f}, {axis[2]:+.3f}), {angles[0]:.1f}° from x")
    lines.append(f"Tilt of the radial axes from y and z: {tilt:.2f}°")
    if trap.n_rf > 0 and trap.Omega_RF > 0:
        result = floquet_at(trap, ion, minimum, voltages)
        lines.append("Floquet frequencies: " + ", ".join(f"{w / (2 * np.pi * 1e6):.4f}" for w in np.sort(result.frequencies)) + " MHz" + ("" if result.stable else " (UNSTABLE)"))
        lines.append("Mathieu q along the axes of Q: " + ", ".join(f"{q:+.4f}" for q in result.q) + ", a: " + ", ".join(f"{a:+.2e}" for a in result.a))
    try:
        escape = escape_point(trap, ion, minimum, voltages)
        lines.append(f"Depth: {'at least ' if escape.lower_bound else ''}{escape.depth * 1e3:.2f} meV through " + ("the edge of the search" if escape.lower_bound else "the saddle") + " at (" + ", ".join(f"{v:.1f}" for v in escape.saddle) + ") µm"
                     + (f" (vertical barrier {escape.vertical_depth * 1e3:.2f} meV)" if np.isfinite(escape.vertical_depth) else ""))
    except Exception as error:
        lines.append(f"Depth: not found ({error})")
    anharmonic = anharmonicity(trap, ion, minimum, voltages)
    lines.append(f"Along the softest axis: cubic length {anharmonic.cubic_length:.3g} µm, quartic length {anharmonic.quartic_length:.3g} µm, "
                 f"shift {anharmonic.shift_per_phonon / (2 * np.pi):.3g} Hz per phonon")
    if n_ions > 1:
        try:
            crystal = crystals.find_crystal(trap, ion, n_ions=n_ions, voltages=voltages, position=minimum)
            lines.append(f"Crystal of {n_ions} ions:")
            lines.extend("  " + line for line in crystal.summary()[1:])
        except Exception as error:
            lines.append(f"Crystal of {n_ions} ions: not found ({error})")
    return lines
