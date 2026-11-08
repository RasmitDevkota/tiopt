"""Multipole expansions of the potentials of a trap at a point, and the control (shim) voltages which produce each
multipole on its own, like the multipole control used by many trapped-ion labs (e.g. to null stray fields with Ex, Ey,
and Ez, or to set the axial confinement and rotate the radial axes with the quadrupoles).

Near a point r0, the potential of any set of voltages is expanded in u = r - r0 (µm) as
    phi = C - Ex x - Ey y - Ez z + U1 (x^2 - y^2) + U2 (2 z^2 - x^2 - y^2) + U3 x y + U4 y z + U5 x z + ...,
so that Ex, Ey, and Ez are the components of the electric field (V/µm) and U1 to U5 are the five independent
quadrupoles (V/µm^2), which are related to the Hessian H of the potential by
    U1 = (H_xx - H_yy) / 4,  U2 = H_zz / 4,  U3 = H_xy,  U4 = H_yz,  U5 = H_xz.
Higher orders are expanded in real solid harmonics R_lm = r^l sqrt(4 pi / (2l + 1)) Y_lm (Racah normalization), whose
coefficients are found by fitting the potential on small spheres around r0, and are named e.g. "R3,-2".
"""

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special

from .lib import LENGTH_SCALE

FIELDS = ("Ex", "Ey", "Ez")
QUADRUPOLES = ("U1", "U2", "U3", "U4", "U5")
BASIC = ("C",) + FIELDS + QUADRUPOLES
SHIMS = FIELDS + QUADRUPOLES


def multipole_names(order=2):
    """Names of the multipoles up to an order (0 for C, 1 for the fields, 2 for the quadrupoles, and solid harmonics
    "Rl,m" beyond).
    """
    names = list(BASIC[: {0: 1, 1: 4}.get(order, 9)])
    for l in range(3, order + 1):
        names.extend(f"R{l},{m}" for m in range(-l, l + 1))
    return names


def unit(name):
    """Unit of a multipole coefficient."""
    if name == "C":
        return "V"
    if name in FIELDS:
        return "V/µm"
    if name in QUADRUPOLES:
        return "V/µm²"
    return f"V/µm^{name[1:].split(',')[0]}"


def real_solid_harmonics(l_max, points):
    """Evaluate the real solid harmonics R_lm (Racah normalization) with 3 <= l <= l_max at points (µm) with shape
    (n, 3), giving shape (n, n_harmonics) in the order of multipole_names.
    """
    points = np.asarray(points, dtype=float)
    r = np.linalg.norm(points, axis=1)
    polar = np.arccos(np.clip(points[:, 2] / np.maximum(r, 1e-300), -1, 1))
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    columns = []
    for l in range(3, l_max + 1):
        norm = np.sqrt(4 * np.pi / (2 * l + 1)) * r ** l
        for m in range(-l, l + 1):
            Y = special.sph_harm_y(l, abs(m), polar, azimuth)
            if m > 0:
                value = np.sqrt(2) * (-1) ** m * Y.real
            elif m < 0:
                value = np.sqrt(2) * (-1) ** m * Y.imag
            else:
                value = Y.real
            columns.append(norm * value)
    return np.column_stack(columns) if columns else np.zeros((len(points), 0))


def _basic_coefficients(phi, grad, H):
    """Coefficients of C, Ex, Ey, Ez, and U1 to U5 from the potential, gradient, and Hessian."""
    return np.array([
        phi, -grad[0], -grad[1], -grad[2],
        (H[0, 0] - H[1, 1]) / 4, H[2, 2] / 4, H[0, 1], H[1, 2], H[0, 2],
    ])


def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    polar = np.arccos(1 - 2 * i / n)
    azimuth = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)])


@dataclass
class ControlMatrix:
    """Multipoles which every electrode produces at 1 V at a point, i.e. matrix[k, e] is multipole names[k] of
    electrode e, and the DC electrodes which were included (which control voltages use by default, since RF electrodes
    carry no DC voltage).
    """
    position: np.ndarray
    names: list
    matrix: np.ndarray
    dc_electrodes: list

    def multipoles(self, voltages):
        """Multipoles (in the units of unit) produced by a set of voltages."""
        return dict(zip(self.names, self.matrix @ np.asarray(voltages, dtype=float)))

    def rows(self, names):
        return self.matrix[[self.names.index(name) for name in names]]


def control_matrix(trap, position, order=2, electrodes=None, radius=5.0, n_points=None):
    """Compute the multipoles which every electrode produces at a point.

    Orders up to 2 come from the exact potential, gradient, and Hessian of the spherical harmonic expansions of the
    trap, and higher orders from a least-squares fit of the potential on spheres of radius and radius / 2 around the
    point (with harmonics up to order + 2 in the fit, so that they do not alias into the ones which are kept).

    Args:
        trap: Solved trap.
        position: Point (µm).
        order: Highest order of the multipoles.
        electrodes: Electrodes to include (all of them by default).
        radius: Radius of the outer sphere of the fit (µm).
        n_points: Number of points on each sphere of the fit.

    Returns:
        ControlMatrix
    """
    position = np.asarray(position, dtype=float)
    electrodes = range(trap.n_electrodes) if electrodes is None else list(electrodes)
    names = multipole_names(order)
    matrix = np.zeros((len(names), trap.n_electrodes))
    fit_order = order + 2
    n_points = max(4 * (fit_order + 1) ** 2, 60) if n_points is None else n_points
    unit_points = _fibonacci_sphere(n_points)
    offsets = np.vstack([radius * unit_points, 0.5 * radius * unit_points])
    if order >= 3:
        # The basis of the fit holds every regular solid harmonic up to fit_order, i.e. 1, the linear and quadratic
        # terms, and the higher harmonics
        linear = offsets
        quadratic = np.column_stack([
            offsets[:, 0] ** 2 - offsets[:, 1] ** 2, 2 * offsets[:, 2] ** 2 - offsets[:, 0] ** 2 - offsets[:, 1] ** 2,
            offsets[:, 0] * offsets[:, 1], offsets[:, 1] * offsets[:, 2], offsets[:, 0] * offsets[:, 2],
        ])
        basis = np.column_stack([np.ones(len(offsets)), -linear, quadratic, real_solid_harmonics(fit_order, offsets)])
        n_higher = len(names) - 9
    for e in electrodes:
        voltages = trap.zero_voltages()
        voltages[e] = 1.0
        phi, grad, H = trap.evaluate(position, voltages)
        matrix[: min(len(names), 9), e] = _basic_coefficients(phi, grad, H)[: min(len(names), 9)]
        if order >= 3:
            samples = trap.potential(position + offsets, voltages)
            coefficients = np.linalg.lstsq(basis, samples, rcond=None)[0]
            matrix[9:, e] = coefficients[9:9 + n_higher]
    return ControlMatrix(position, names, matrix, [e for e in electrodes if e >= trap.n_rf])


def decompose(trap, position, voltages, order=2):
    """Find the multipoles which a set of voltages produces at a point, as a dict from name to coefficient."""
    return control_matrix(trap, position, order).multipoles(voltages)


def solve_voltages(control, targets, electrodes=None, max_voltage=None, regularization=1e-6):
    """Find the voltages which produce target multipoles, with the smallest voltages among the solutions.

    Args:
        control: ControlMatrix.
        targets: Dict from the name of a multipole to its target (in the units of unit); multipoles which are not given
            are left free.
        electrodes: Electrodes which may be used (the DC electrodes of the control matrix by default).
        max_voltage: Largest magnitude of any voltage (V), or None for no limit.
        regularization: Weight on the voltages relative to the multipoles, which picks the smallest voltages among
            solutions.

    Returns:
        Voltage of every electrode (V), and the multipoles which they actually produce
    """
    names = list(targets)
    M = control.rows(names)
    n_electrodes = control.matrix.shape[1]
    if electrodes is None:
        electrodes = [e for e in control.dc_electrodes if np.any(control.matrix[:, e] != 0)]
    electrodes = list(electrodes)
    t = np.array([targets[name] for name in names], dtype=float)
    A = M[:, electrodes]
    # Scale each row so that every multipole counts the same
    scale = np.maximum(np.abs(A).max(axis=1), 1e-300)
    A_scaled = A / scale[:, None]
    t_scaled = t / scale
    weight = np.sqrt(regularization)
    system = np.vstack([A_scaled, weight * np.eye(len(electrodes))])
    rhs = np.concatenate([t_scaled, np.zeros(len(electrodes))])
    if max_voltage is None:
        solution = np.linalg.lstsq(system, rhs, rcond=None)[0]
    else:
        solution = optimize.lsq_linear(system, rhs, bounds=(-max_voltage, max_voltage)).x
    voltages = np.zeros(n_electrodes)
    voltages[electrodes] = solution
    return voltages, control.multipoles(voltages)


def shim_matrix(control, names=SHIMS, electrodes=None):
    """Find the voltages which produce one unit of each multipole and none of the others, i.e. the matrix of control
    voltages (a "C-file") which turns multipoles into voltages.

    Args:
        control: ControlMatrix.
        names: Multipoles which are controlled.
        electrodes: Electrodes which may be used (the DC electrodes of the control matrix with any effect by default).

    Returns:
        Voltages with shape (n_electrodes, len(names)), so that voltages @ targets gives the voltages which produce
        the targets
    """
    names = list(names)
    M = control.rows(names)
    n_electrodes = control.matrix.shape[1]
    if electrodes is None:
        electrodes = [e for e in control.dc_electrodes if np.any(M[:, e] != 0)]
    electrodes = list(electrodes)
    if len(electrodes) < len(names):
        raise ValueError(f"{len(names)} multipoles need at least as many electrodes, but only {len(electrodes)} are available")
    A = M[:, electrodes]
    shims = np.zeros((n_electrodes, len(names)))
    shims[electrodes] = np.linalg.pinv(A)
    achieved = M @ shims
    if not np.allclose(achieved, np.eye(len(names)), atol=1e-6 * max(np.abs(achieved).max(), 1.0)):
        raise ValueError("the electrodes cannot produce every multipole independently at this point")
    return shims


def save_shim_matrix(path, shims, names=SHIMS):
    """Write a matrix of control voltages to a text file with a header of multipole names and one row per electrode."""
    header = "electrode " + " ".join(f"{name}[{unit(name)}]" for name in names)
    rows = [f"{e} " + " ".join(f"{v:.9e}" for v in row) for e, row in enumerate(shims)]
    with open(path, "w") as f:
        f.write("# Control voltages (V) per unit of each multipole, from tiopt.multipoles\n# " + header + "\n" + "\n".join(rows) + "\n")


def load_shim_matrix(path):
    """Read a matrix of control voltages written by save_shim_matrix, returning the voltages and the multipole names."""
    with open(path) as f:
        lines = f.read().splitlines()
    header = [line for line in lines if line.startswith("# electrode")][0]
    names = [item.split("[")[0] for item in header[2:].split()[1:]]
    rows = [line.split() for line in lines if line and not line.startswith("#")]
    return np.array([[float(v) for v in row[1:]] for row in rows]), names


def quadrupoles_from_hessian(H):
    """Quadrupoles U1 to U5 (V/µm^2) of a Hessian of the potential (V/µm^2), as a dict."""
    H = np.asarray(H, dtype=float)
    return dict(zip(QUADRUPOLES, _basic_coefficients(0.0, np.zeros(3), H)[4:]))


def hessian_from_quadrupoles(U1=0.0, U2=0.0, U3=0.0, U4=0.0, U5=0.0):
    """Hessian of the potential (V/µm^2) made by the quadrupoles U1 to U5 (V/µm^2)."""
    return np.array([
        [2 * U1 - 2 * U2, U3, U5],
        [U3, -2 * U1 - 2 * U2, U4],
        [U5, U4, 4 * U2],
    ])


def axial_quadrupoles(ion, omega, axis="x"):
    """Quadrupoles (V/µm^2) which confine an ion along an axis with secular frequency omega (rad/s), taking half of
    that curvature from each perpendicular axis (as Laplace's equation requires), as a dict.
    """
    k = ion.mass * omega ** 2 / ion.charge * LENGTH_SCALE ** 2
    index = "xyz".index(axis)
    H = np.full(3, -k / 2)
    H[index] = k
    return quadrupoles_from_hessian(np.diag(H))
