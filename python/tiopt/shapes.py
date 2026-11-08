"""Free-form optimization of the shapes of the electrodes of a trap, e.g. the RF electrodes near the center of a
junction, where the shapes of the electrodes set the pseudopotential barrier and the dip in confinement an ion crosses.

Rather than changing a few dimensions of rectangles, the trap surface is deformed by a smooth displacement field inside
a circular design region: every vertex moves by the displacement at its position, and the edges inside the region are
first divided into short pieces so that straight boundaries can bend. Since every electrode moves with the same field,
the boundaries which electrodes share move together, so no gaps or overlaps open up (as long as the deformation stays
smooth, which a penalty on the Jacobian of the deformation enforces), and the electrodes keep their topology (holes and
islands are not created, unlike in density-based topology optimization). The field is a sum of Gaussian bumps on a grid
of control points with displacements of at most max_shift, which can be made symmetric (e.g. under the symmetries of an
X junction) and fades to zero at the edge of the region, where the rest of the trap is untouched.

The objective is evaluated from the gapless plane potentials of the RF electrodes along a path (see confinement.py): the
pseudopotential barrier on the RF null, the variation of the total confinement along it (which a CTC path would
otherwise have to make up for by leaving the null), and how far the RF null strays from its height at the start.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from .confinement import RFField, confinement_path, default_waypoints
from .lib import E_CHARGE, ION_HEIGHT, Trap

SYMMETRIES = ("none", "mirror x", "mirror y", "mirror x and y", "3-fold", "4-fold")
OBJECTIVES = {
    "barrier": "Pseudopotential barrier along the path",
    "confinement": "Constant total confinement along the path",
    "both": "Barrier and constant total confinement",
}


def symmetry_group(symmetry):
    """Orthogonal 2x2 matrices of a symmetry (see SYMMETRIES), where "mirror x" reflects across the x axis."""
    identity = np.eye(2)
    flip_y = np.diag([1.0, -1.0])
    flip_x = np.diag([-1.0, 1.0])

    def rotation(angle):
        return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    if symmetry == "none":
        return [identity]
    if symmetry == "mirror x":
        return [identity, flip_y]
    if symmetry == "mirror y":
        return [identity, flip_x]
    if symmetry == "mirror x and y":
        return [identity, flip_y, flip_x, -identity]
    if symmetry in ("3-fold", "4-fold"):
        n = 3 if symmetry == "3-fold" else 4
        rotations = [rotation(2 * np.pi * k / n) for k in range(n)]
        return rotations + [r @ flip_y for r in rotations]
    raise ValueError(f"unknown symmetry {symmetry!r}; choose from {', '.join(SYMMETRIES)}")


class Deformation:
    """Smooth displacement field of the trap surface inside a circle (see the module docstring)."""

    def __init__(self, center, radius, spacing, max_shift, symmetry="none"):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.max_shift = float(max_shift)
        self.group = symmetry_group(symmetry)
        # Control points on a square grid inside the circle
        n = int(np.floor(radius / spacing))
        grid = np.arange(-n, n + 1) * spacing
        X, Y = np.meshgrid(grid, grid)
        points = np.column_stack([X.ravel(), Y.ravel()])
        self.controls = points[np.hypot(points[:, 0], points[:, 1]) < radius - 0.25 * spacing]
        self.sigma = 0.6 * spacing

    @property
    def n_variables(self):
        return 2 * len(self.controls)

    def _raw(self, offsets, u):
        """Displacement before symmetrizing, at offsets from the center with shape (n, 2)."""
        coefficients = u.reshape(-1, 2)
        distance2 = np.sum((offsets[:, None, :] - self.controls[None, :, :]) ** 2, axis=-1)
        weights = np.exp(-0.5 * distance2 / self.sigma ** 2)
        # Normalizing the bumps keeps every displacement within max_shift
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-300)
        r = np.hypot(offsets[:, 0], offsets[:, 1]) / self.radius
        window = np.where(r < 1, (1 - r ** 2) ** 2, 0.0)
        return self.max_shift * window[:, None] * (weights @ coefficients)

    def displacement(self, points, u):
        """Displacement (µm) of points (µm) with shape (n, 2) for the variables u (each within [-1, 1])."""
        offsets = np.asarray(points, dtype=float) - self.center
        total = np.zeros_like(offsets)
        for g in self.group:
            total += self._raw(offsets @ g.T, u) @ g
        return total / len(self.group)

    def min_jacobian(self, u, n=41):
        """Smallest determinant of the Jacobian of the map from the surface to the deformed surface inside the circle,
        which must stay positive for the deformation to keep the electrodes from folding over each other.
        """
        grid = np.linspace(-self.radius, self.radius, n)
        X, Y = np.meshgrid(grid, grid)
        points = np.column_stack([X.ravel(), Y.ravel()]) + self.center
        h = 0.25
        dx = (self.displacement(points + [h, 0], u) - self.displacement(points - [h, 0], u)) / (2 * h)
        dy = (self.displacement(points + [0, h], u) - self.displacement(points - [0, h], u)) / (2 * h)
        determinant = (1 + dx[:, 0]) * (1 + dy[:, 1]) - dx[:, 1] * dy[:, 0]
        return float(determinant.min())


def refine_polygon(polygon, center, radius, max_length):
    """Divide the edges of a polygon which come near a circle into pieces no longer than max_length (µm), placing the
    new vertices by the endpoints of each edge alone, so that edges which two polygons share are divided alike.
    """
    polygon = np.asarray(polygon, dtype=float)[:, :2]
    out = []
    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i + 1) % len(polygon)]
        out.append(a)
        # Distance from the center to the edge
        edge = b - a
        t = np.clip(np.dot(center - a, edge) / max(np.dot(edge, edge), 1e-12), 0, 1)
        if np.linalg.norm(a + t * edge - center) > radius:
            continue
        n = int(np.ceil(np.linalg.norm(edge) / max_length))
        # Divide every shared edge from the same end, so that both polygons get the same points
        forward = tuple(np.round(a, 6)) < tuple(np.round(b, 6))
        first, last = (a, b) if forward else (b, a)
        points = first + (np.arange(1, n) / n)[:, None] * (last - first)
        out.extend(points if forward else points[::-1])
    return np.array(out)


def deformed_polygons(polygons, deformation, u):
    """Move every vertex of refined polygons by the displacement field."""
    return [polygon + deformation.displacement(polygon, u) for polygon in polygons]


def trap_from_polygons(trap, polygons, solve=True):
    """Copy of a trap with new electrode polygons (in the same order), keeping its RF drive, zones, and photonics."""
    new = Trap.from_polygons(polygons, trap.n_rf, trap.V_RF, trap.Omega_RF)
    zones = trap.zones()
    if zones:
        new.set_zones(zones)
    gratings, detectors = trap.grating_couplers(), trap.detectors()
    if gratings or detectors:
        new.set_photonics(gratings, detectors)
    if solve:
        new.solve("analytical")
    return new


@dataclass
class PathMetrics:
    """Figures of merit of the RF null along a path: the barrier (meV, from a smooth maximum), the spread of the total
    confinement (the standard deviation of its logarithm), how far the null strays from the target height (rms, µm),
    and the mean total confinement (as sqrt(sum of omega^2) / 2 pi, Hz).
    """

    barrier: float
    confinement_spread: float
    height_error: float
    mean_confinement: float


@dataclass
class ShapeResult:
    """Result of optimize_shapes: the trap with the new shapes (solved), the deformation and its variables, the
    figures of merit along the path before and after, the confinement paths before and after, and the history of the
    objective.
    """

    trap: Trap
    deformation: Deformation
    u: np.ndarray
    initial: PathMetrics
    final: PathMetrics
    initial_path: object
    final_path: object
    history: list = field(default_factory=list)
    polygons: list = field(default_factory=list, repr=False)
    original_polygons: list = field(default_factory=list, repr=False)

    @property
    def max_displacement(self):
        """Largest distance any vertex moved (µm)."""
        return float(max(np.max(np.linalg.norm(a - b, axis=1)) for a, b in zip(self.polygons, self.original_polygons)))


def optimize_shapes(trap, ion, waypoints=None, center=None, radius=None, max_shift=8.0, spacing=None, symmetry="none", objective="both",
                    max_evaluations=300, n_samples=41, refine=4.0, target_height=None, callback=None, stop=None):
    """Deform the electrodes of a trap near a center to improve the RF null along a path through it.

    Args:
        trap: Trap with an RF drive (it does not have to be solved, and it is not changed).
        ion: Ion species.
        waypoints: Points of the path in the trap surface (µm), or None for confinement.default_waypoints.
        center: Center of the design region (µm), or None for the middle waypoint (e.g. the center of a junction).
        radius: Radius of the design region (µm), or None for 2.5 ion heights.
        max_shift: Largest displacement of any point (µm).
        spacing: Spacing of the control points (µm), or None for a third of the radius.
        symmetry: Symmetry of the deformation (see SYMMETRIES), e.g. "4-fold" for an X junction.
        objective: What to improve (see OBJECTIVES).
        max_evaluations: Largest number of evaluations of the objective.
        n_samples: Number of points along the path.
        refine: Longest edge inside the design region after dividing edges (µm).
        target_height: Height the RF null should stay at (µm), or None for its mean height along the path at the start.
        callback: Function called with (evaluation, objective, PathMetrics) after every evaluation.
        stop: Function returning True to stop the optimization early.

    Returns:
        ShapeResult
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; choose from {', '.join(OBJECTIVES)}")
    if trap.n_rf == 0:
        raise ValueError("the trap has no RF electrodes to shape")
    waypoints = default_waypoints(trap) if waypoints is None else np.asarray(waypoints, dtype=float)
    if center is None:
        center = waypoints[len(waypoints) // 2][:2] if len(waypoints) > 2 else 0.5 * (waypoints[0][:2] + waypoints[-1][:2])
    radius = radius or 2.5 * ION_HEIGHT
    spacing = spacing or radius / 3
    deformation = Deformation(center, radius, spacing, max_shift, symmetry)
    original = [polygon[:, :2].copy() for polygon in trap.electrode_polygons()]
    refined = [refine_polygon(polygon, deformation.center, radius + 2 * refine, refine) for polygon in original]
    # The null at the start, from which every evaluation starts its search across the path
    path = confinement_path(trap, ion, waypoints, n_samples)
    start, directions = path.null_positions, path.directions
    height = target_height if target_height is not None else float(np.mean(start[:, 2]))
    weights = {"barrier": (1.0, 0.0), "confinement": (0.0, 1.0), "both": (1.0, 1.0)}[objective]

    def metrics(polygons):
        candidate = Trap.from_polygons(polygons, trap.n_rf, trap.V_RF, trap.Omega_RF)
        try:
            field = RFField(candidate)
            null = field.null_across(start, directions, max_iterations=12)
            energy = field.pseudopotential(ion, null) / E_CHARGE * 1e3
            squares = field.total_confinement(ion, null)
        finally:
            candidate.close()
        # A smooth maximum of the pseudopotential, above its lower end
        relative = np.maximum(energy - min(energy[0], energy[-1]), 0.0)
        barrier = float(np.sum(relative ** 8) ** (1 / 8))
        logs = np.log(np.maximum(squares, 1e-300))
        return PathMetrics(barrier, float(np.std(logs)), float(np.sqrt(np.mean((null[:, 2] - height) ** 2))), float(np.sqrt(np.mean(squares)) / (2 * np.pi)))

    initial = metrics(refined)
    history = []
    best = dict(value=np.inf, u=np.zeros(deformation.n_variables))

    class Stop(Exception):
        pass

    def evaluate(u):
        if stop is not None and stop():
            raise Stop
        polygons = deformed_polygons(refined, deformation, u)
        m = metrics(polygons)
        value = weights[0] * m.barrier / max(initial.barrier, 0.1) + weights[1] * m.confinement_spread / max(initial.confinement_spread, 1e-3)
        # Keep the null from straying further from its height and the confinement from falling overall, and keep the
        # deformation smooth
        value += (max(m.height_error - initial.height_error, 0.0) / 2.0) ** 2 + 4 * max(1 - m.mean_confinement / initial.mean_confinement, 0.0) ** 2
        value += 100 * max(0.3 - deformation.min_jacobian(u, 21), 0.0) ** 2
        history.append(value)
        if value < best["value"]:
            best.update(value=value, u=np.array(u, dtype=float))
        if callback is not None:
            callback(len(history), value, m)
        return value

    u0 = np.zeros(deformation.n_variables)
    try:
        optimize.minimize(evaluate, u0, method="L-BFGS-B", bounds=[(-1.0, 1.0)] * len(u0), options=dict(maxfun=max_evaluations, eps=1e-3, maxiter=max_evaluations))
    except Stop:
        pass
    u = best["u"]
    polygons = deformed_polygons(refined, deformation, u)
    new = trap_from_polygons(trap, polygons)
    initial_path = confinement_path(trap, ion, waypoints, max(n_samples, 61))
    final_path = confinement_path(new, ion, waypoints, max(n_samples, 61))
    return ShapeResult(new, deformation, u, initial, metrics(polygons), initial_path, final_path, history, polygons, refined)
