"""Confinement of the RF pseudopotential, and paths of constant total confinement (CTC) through junctions and bends.

The DC electrodes obey Laplace's equation, so they can only move confinement from one direction to another, and the sum
of the squares of the three secular frequencies (the total confinement) is set by the RF electrodes alone:
    omega_x^2 + omega_y^2 + omega_z^2 = q^2 V_RF^2 |H|^2 / (2 m^2 Omega_RF^2),
where H is the Hessian of the potential of the RF electrodes at 1 V. Along a linear section the total confinement is the
same everywhere on the RF null, but near the center of a junction (where the RF null rises and the pseudopotential has
a barrier) it drops, so ions there are held weakly. A CTC path runs directly beneath the RF null wherever the total
confinement on the null is lower than in the arms, at the height where it is the same as in the arms, so that an ion
moved along it is held as strongly everywhere (at the cost of some micromotion, since it is off the null).

Everything here uses the gapless plane potentials of the electrodes directly (Trap.analytical_electrode_potential), so
it works anywhere above the trap, beyond the region of the spherical harmonic expansions too.
"""

from dataclasses import dataclass

import numpy as np

from .lib import E_CHARGE, ION_HEIGHT

STEP = 0.05  # finite difference step (µm)


def _offsets():
    """Stencil of the finite differences: the center, the six points along the axes, and the twelve points along the
    diagonals of each pair of axes.
    """
    offsets = [np.zeros(3)]
    for i in range(3):
        for sign in (1, -1):
            offsets.append(sign * STEP * np.eye(3)[i])
    for i in range(3):
        for j in range(i + 1, 3):
            for si in (1, -1):
                for sj in (1, -1):
                    offsets.append(STEP * (si * np.eye(3)[i] + sj * np.eye(3)[j]))
    return np.array(offsets)


OFFSETS = _offsets()


class RFField:
    """Potential of the RF electrodes of a trap (at 1 V), with its derivatives, pseudopotential, and total confinement.
    """

    def __init__(self, trap):
        self.trap = trap

    def potential(self, points):
        """Potential of the RF electrodes at 1 V at points (µm) with shape (..., 3)."""
        points = np.asarray(points, dtype=float)
        total = np.zeros(points.shape[:-1])
        for e in range(self.trap.n_rf):
            total += self.trap.analytical_electrode_potential(e, points)
        return total

    def derivatives(self, points):
        """Gradient (V/µm) and Hessian (V/µm^2) of the potential of the RF electrodes at 1 V at points (µm) with shape
        (..., 3).
        """
        points = np.asarray(points, dtype=float)
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        phi = self.potential(flat[:, None, :] + OFFSETS[None, :, :])
        center = phi[:, 0]
        gradient = np.zeros((len(flat), 3))
        hessian = np.zeros((len(flat), 3, 3))
        for i in range(3):
            plus, minus = phi[:, 1 + 2 * i], phi[:, 2 + 2 * i]
            gradient[:, i] = (plus - minus) / (2 * STEP)
            hessian[:, i, i] = (plus - 2 * center + minus) / STEP ** 2
        k = 7
        for i in range(3):
            for j in range(i + 1, 3):
                pp, pm, mp, mm = phi[:, k], phi[:, k + 1], phi[:, k + 2], phi[:, k + 3]
                hessian[:, i, j] = hessian[:, j, i] = (pp - pm - mp + mm) / (4 * STEP ** 2)
                k += 4
        return gradient.reshape(shape + (3,)), hessian.reshape(shape + (3, 3))

    def _scale(self, ion):
        """q^2 V_RF^2 / (4 m Omega_RF^2) in SI units."""
        omega = self.trap.Omega_RF
        if not omega > 0:
            raise ValueError("the trap has no RF drive (set its RF frequency)")
        return (ion.charge * self.trap.V_RF) ** 2 / (4 * ion.mass * omega ** 2)

    def pseudopotential(self, ion, points):
        """Pseudopotential (J) at points (µm) with shape (..., 3)."""
        gradient, _ = self.derivatives(points)
        return self._scale(ion) * np.sum((gradient * 1e6) ** 2, axis=-1)

    def total_confinement(self, ion, points):
        """Sum of the squares of the secular frequencies (rad^2/s^2) of the pseudopotential at points (µm) with shape
        (..., 3), which no DC voltages can change.
        """
        _, hessian = self.derivatives(points)
        return 2 * self._scale(ion) * np.sum((hessian * 1e12) ** 2, axis=(-2, -1)) / ion.mass

    def null_across(self, points, directions, max_iterations=30, tolerance=1e-6):
        """Find the points of least pseudopotential in the planes across directions in the trap surface (e.g. the
        direction of a path), starting from points (µm) with shape (n, 3), by Gauss-Newton steps on |grad phi|^2.
        """
        points = np.array(points, dtype=float).reshape(-1, 3)
        directions = np.array(directions, dtype=float).reshape(-1, 3)
        directions[:, 2] = 0.0
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        normals = np.stack([-directions[:, 1], directions[:, 0], np.zeros(len(points))], axis=-1)
        basis = np.stack([normals, np.broadcast_to([0.0, 0.0, 1.0], normals.shape)], axis=-1)
        active = np.ones(len(points), dtype=bool)
        for _ in range(max_iterations):
            if not active.any():
                break
            gradient, hessian = self.derivatives(points[active])
            B = basis[active]
            HB = hessian @ B
            A = np.swapaxes(HB, 1, 2) @ HB
            b = np.einsum("nij,ni->nj", HB, gradient)
            step = -np.linalg.solve(A + 1e-30 * np.eye(2), b[..., None])[..., 0]
            # Keep steps short and the points above the surface
            length = np.linalg.norm(step, axis=1, keepdims=True)
            step *= np.minimum(1.0, 5.0 / np.maximum(length, 1e-300))
            moved = np.einsum("nij,nj->ni", B, step)
            new = points[active] + moved
            # Where there is no null (e.g. away from the electrodes), the search would run off upwards
            new[:, 2] = np.clip(new[:, 2], 2.0, 5 * ION_HEIGHT)
            points[active] = new
            done = np.linalg.norm(moved, axis=1) < tolerance
            indices = np.flatnonzero(active)
            active[indices[done]] = False
        return points


def path_points(waypoints, n_samples=101):
    """Points spaced evenly along a path through waypoints (µm, with shape (n, 2) or (n, 3)), with the direction of the
    path and the distance along it at each point.
    """
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.shape[1] == 2:
        waypoints = np.column_stack([waypoints, np.full(len(waypoints), ION_HEIGHT)])
    lengths = np.linalg.norm(np.diff(waypoints[:, :2], axis=0), axis=1)
    if len(waypoints) < 2 or lengths.sum() <= 0:
        raise ValueError("a path needs at least two different waypoints")
    s_waypoints = np.concatenate(([0.0], np.cumsum(lengths)))
    s = np.linspace(0.0, s_waypoints[-1], n_samples)
    points = np.column_stack([np.interp(s, s_waypoints, waypoints[:, k]) for k in range(3)])
    segment = np.clip(np.searchsorted(s_waypoints, s, side="right") - 1, 0, len(lengths) - 1)
    directions = np.column_stack([np.diff(waypoints[:, :2], axis=0)[segment] / lengths[segment, None], np.zeros(n_samples)])
    return points, directions, s


@dataclass
class ConfinementPath:
    """RF null along a path and the CTC path beneath it, with the distance along the path (µm), the positions (µm),
    pseudopotential (eV), and total confinement (as sqrt(sum of omega^2) / 2 pi, Hz) along each, the total confinement
    the CTC path keeps, the DC field (V/m) which must hold an ion on the CTC path against the pseudopotential, and the
    direction of the path at each point.
    """

    s: np.ndarray
    null_positions: np.ndarray
    null_energy: np.ndarray
    null_confinement: np.ndarray
    ctc_positions: np.ndarray
    ctc_energy: np.ndarray
    ctc_confinement: np.ndarray
    target: float
    holding_field: np.ndarray
    directions: np.ndarray = None

    @property
    def barrier(self):
        """Highest pseudopotential on the RF null above the lower of its ends (eV)."""
        return float(self.null_energy.max() - min(self.null_energy[0], self.null_energy[-1]))

    @property
    def confinement_dip(self):
        """Lowest total confinement on the RF null relative to the CTC target."""
        return float(self.null_confinement.min() / self.target) if self.target > 0 else 1.0

    @property
    def depth(self):
        """Largest drop of the CTC path below the RF null (µm)."""
        return float(np.max(self.null_positions[:, 2] - self.ctc_positions[:, 2]))


def soft_axis(field, point):
    """Direction in the trap surface along which the RF potential curves least at a point, i.e. along the RF null line,
    or None where it curves alike in every direction in the surface (e.g. at the center of a junction).
    """
    _, hessian = field.derivatives(point)
    values, vectors = np.linalg.eigh(hessian)
    order = np.argsort(np.abs(values))
    axis = vectors[:, order[0]].copy()
    axis[2] = 0.0
    if np.linalg.norm(axis) < 0.5 or abs(values[order[1]]) < 1e-6 or abs(values[order[0]]) > 0.3 * abs(values[order[1]]):
        return None
    return axis / np.linalg.norm(axis)


def trace_null(trap, waypoints, step):
    """Follow the RF null line of a trap from near the first waypoint past every other waypoint in turn, stepping along
    the line (so that the path follows curved sections) and heading for the next waypoint where the line splits or ends
    (e.g. at the center of a junction).

    Args:
        trap: Trap with RF electrodes.
        waypoints: Points the path goes past in the trap surface (µm), with shape (n, 2) or (n, 3).
        step: Distance between the points of the path (µm).

    Returns:
        Points of the RF null (µm) with shape (n_points, 3), and the direction of the path at each
    """
    field = RFField(trap)
    waypoints = np.asarray(waypoints, dtype=float)[:, :2]
    if len(waypoints) < 2:
        raise ValueError("a path needs at least two different waypoints")
    first = waypoints[1] - waypoints[0]
    if np.linalg.norm(first) <= 0:
        raise ValueError("a path needs at least two different waypoints")
    direction = np.append(first / np.linalg.norm(first), 0.0)
    current = field.null_across([[*waypoints[0], ION_HEIGHT]], [direction])[0]
    points, directions = [current], [direction]
    chord = np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))
    budget = int(4 * chord / step) + 10
    target = 1
    while target < len(waypoints) and budget > 0:
        budget -= 1
        heading = np.append(waypoints[target] - current[:2], 0.0)
        distance = np.linalg.norm(heading)
        if distance < 0.5 * step:
            target += 1
            continue
        heading /= distance
        axis = soft_axis(field, current)
        # Follow the null line where it heads roughly for the waypoint, and head straight for it elsewhere
        if axis is not None and abs(axis @ heading) > 0.5:
            direction = axis if axis @ heading > 0 else -axis
        else:
            direction = heading
        guess = current + step * direction
        candidate = field.null_across([guess], [direction])[0]
        # Once the path has passed the waypoint, head for the next one
        if np.linalg.norm(candidate[:2] - waypoints[target]) > distance and distance < 3 * step:
            target += 1
            if target >= len(waypoints):
                break
        points.append(candidate)
        directions.append(direction)
        current = candidate
    return np.array(points), np.array(directions)


def confinement_path(trap, ion, waypoints, n_samples=101, target=None, z_min=5.0):
    """Follow the RF null along a path past waypoints (see trace_null), and the CTC path beneath it.

    Args:
        trap: Trap (which does not have to be solved).
        ion: Ion species.
        waypoints: Points the path goes past in the trap surface (µm), with shape (n, 2) or (n, 3).
        n_samples: Number of points along the path if it were straight from waypoint to waypoint (a curved path gets
            more, at the same spacing).
        target: Total confinement the CTC path keeps (as sqrt(sum of omega^2) / 2 pi, Hz), or None for the lower of
            its values on the RF null at the two ends of the path.
        z_min: Lowest height the CTC path may go down to (µm).

    Returns:
        ConfinementPath
    """
    field = RFField(trap)
    waypoints = np.asarray(waypoints, dtype=float)
    chord = np.sum(np.linalg.norm(np.diff(waypoints[:, :2], axis=0), axis=1))
    null, directions = trace_null(trap, waypoints, chord / max(n_samples - 1, 1))
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(null[:, :2], axis=0), axis=1))))
    to_ev = 1 / E_CHARGE
    null_energy = field.pseudopotential(ion, null) * to_ev
    sum_squares = field.total_confinement(ion, null)
    if target is None:
        target_squares = min(sum_squares[0], sum_squares[-1])
    else:
        target_squares = (2 * np.pi * target) ** 2
    ctc = null.copy()
    for i in np.flatnonzero(sum_squares < target_squares * (1 - 1e-6)):
        # The total confinement grows below the null (until close to the electrodes), so scan down for the highest point
        # where it reaches the target, and then bisect
        heights = np.arange(null[i, 2], z_min - 1e-9, -0.5)
        column = np.repeat(null[i][None], len(heights), axis=0)
        column[:, 2] = heights
        values = field.total_confinement(ion, column)
        reached = np.flatnonzero(values >= target_squares)
        if len(reached) == 0:
            ctc[i, 2] = heights[np.argmax(values)]
            continue
        high, low = heights[max(reached[0] - 1, 0)], heights[reached[0]]
        point = null[i].copy()
        for _ in range(40):
            point[2] = 0.5 * (low + high)
            if field.total_confinement(ion, point) >= target_squares:
                low = point[2]
            else:
                high = point[2]
        ctc[i, 2] = low
    gradient, hessian = field.derivatives(ctc)
    # The pseudopotential pushes with -grad U, which the DC field must cancel: E = grad U / q
    scale = field._scale(ion)
    grad_U = 2 * scale * np.einsum("nij,nj->ni", hessian * 1e12, gradient * 1e6)
    holding = np.linalg.norm(grad_U, axis=1) / ion.charge
    return ConfinementPath(s, null, null_energy, np.sqrt(sum_squares) / (2 * np.pi), ctc, field.pseudopotential(ion, ctc) * to_ev,
                           np.sqrt(field.total_confinement(ion, ctc)) / (2 * np.pi), np.sqrt(target_squares) / (2 * np.pi), holding, directions)


def confinement_map(trap, ion, x, y, z):
    """Total confinement (as sqrt(sum of omega^2) / 2 pi, Hz) over a grid of x and y (µm) at height z (µm)."""
    X, Y = np.meshgrid(x, y)
    points = np.stack([X, Y, np.full_like(X, z)], axis=-1)
    return np.sqrt(RFField(trap).total_confinement(ion, points)) / (2 * np.pi)


def arm_directions(trap, center=(0.0, 0.0), radius=2.5 * ION_HEIGHT):
    """Directions (degrees) and lengths (µm) of the arms which run out from a center, e.g. 0 and 180 for a linear trap or
    0, 120, and 240 for a Y junction, found where the RF field on a circle around the center at the ion height is
    weakest (since the RF null runs along every arm).
    """
    if trap.n_rf == 0:
        return []
    field = RFField(trap)
    center = np.asarray(center, dtype=float)
    angles = np.arange(0.0, 360.0, 2.0)
    radians = np.radians(angles)
    points = np.column_stack([center[0] + radius * np.cos(radians), center[1] + radius * np.sin(radians), np.full(len(angles), ION_HEIGHT)])
    magnitude = np.linalg.norm(field.derivatives(points)[0], axis=1)
    rf = np.vstack([trap.electrode_polygons()[e][:, :2] for e in range(trap.n_rf)]) - center
    arms = []
    for i in range(len(angles)):
        before, after = magnitude[i - 1], magnitude[(i + 1) % len(angles)]
        if not (magnitude[i] <= before and magnitude[i] < after and magnitude[i] < 0.2 * np.median(magnitude)):
            continue
        # Refine the minimum with a parabola through its neighbors
        curvature = before - 2 * magnitude[i] + after
        shift = 0.5 * (before - after) / curvature if curvature > 0 else 0.0
        angle = (angles[i] + 2.0 * shift) % 360
        direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        along = rf @ direction
        across = np.abs(rf @ np.array([-direction[1], direction[0]]))
        near = along[(across < 3 * ION_HEIGHT) & (along > 0)]
        arms.append((float(angle), float(near.max()) if len(near) else radius))
    return arms


def default_waypoints(trap):
    """Waypoints of a path to look at confinement along: through the zones of the trap if it has several (from one end of
    the trap to the other), and otherwise from the arm nearest -x through the center into the next arm clockwise (along the trap for a linear trap,
    or around the corner of a junction), stopping short of the ends of the arms, where the rails end.
    """
    zones = trap.zones()
    if len(zones) >= 2:
        # Chain the zones from the one furthest out to the nearest zone each time, so that the path runs along the trap
        # whatever order the zones were made in
        points = np.array([zone.position[:2] for zone in zones])
        order = [int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1)))]
        remaining = set(range(len(points))) - set(order)
        while remaining:
            last = points[order[-1]]
            order.append(min(remaining, key=lambda z: np.linalg.norm(points[z] - last)))
            remaining.discard(order[-1])
        return points[order]
    vertices = np.vstack([polygon[:, :2] for polygon in trap.electrode_polygons()])
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    if len(zones) == 1:
        center = np.asarray(zones[0].position[:2])
    arms = arm_directions(trap, center)
    if len(arms) < 2:
        return np.array([center - [100.0, 0.0], center + [100.0, 0.0]])

    def separation(a, b):
        return abs((a - b + 180) % 360 - 180)

    start = min(arms, key=lambda arm: separation(arm[0], 180))
    others = [arm for arm in arms if arm is not start]
    if len(arms) == 2:
        end = others[0]
    else:
        end = min(others, key=lambda arm: separation(arm[0], start[0] - 90))

    def along(arm):
        angle = np.radians(arm[0])
        return center + 0.4 * arm[1] * np.array([np.cos(angle), np.sin(angle)])

    if len(arms) == 2:
        return np.array([along(start), along(end)])
    return np.array([along(start), center, along(end)])
