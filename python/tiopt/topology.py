"""Topology optimization of RF electrodes on a grid of pixels, like the pattern optimization of SurfacePattern
(R. Schmied et al., PRL 102, 233002 (2009)) and the electrode package, so that the RF electrodes of e.g. a junction can
take any shape and topology (with holes and islands), unlike the deformations of tiopt.shapes.

Every pixel of a square design region is either part of an RF electrode or grounded. In the gapless plane approximation
the RF potential is linear in the pixels, so RF nulls at given points (where the gradient of the RF potential vanishes)
are linear equations in them, and the confinement at each null (the curvature of the RF potential along the vertical,
which is the strength of the quadrupole there) is a linear function of them. The design which pins every null and makes
the weakest confinement as strong as possible is then a linear program over the fraction of each pixel which is RF,
whose solutions are mostly 0 or 1 (since linear programs end on vertices of their feasible set), and the few fractional
pixels are rounded. Symmetries (e.g. the 4-fold symmetry of an X junction) tie the pixels of each orbit together. The RF
pixels are then traced into polygons, whose holes are cut open to their outer boundaries (since the electrodes of tiopt
are simple polygons, and the two sides of each cut cancel in the gapless plane approximation).
"""

from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path as MplPath
from scipy import optimize

from .lib import ION_HEIGHT, Trap

LAYOUTS = ("X junction", "T junction", "Y junction", "Linear")
SYMMETRIES = ("none", "mirror x", "mirror xy", "4-fold")


def rectangle_potential(x0, x1, y0, y1, points):
    """Gapless plane potential (V/V) of rectangles [x0, x1] x [y0, y1] at 1 V at points (shape (n, 3)), with shape
    (n, n_rectangles) (J. H. Wesenberg, PRA 78, 063410 (2008); M. G. House, PRA 78, 033402 (2008)).
    """
    x = points[:, None, 0]
    y = points[:, None, 1]
    z = np.maximum(points[:, None, 2], 1e-9)
    total = 0.0
    for xi, sx in ((x0, -1.0), (x1, 1.0)):
        for yj, sy in ((y0, -1.0), (y1, 1.0)):
            dx = xi[None, :] - x
            dy = yj[None, :] - y
            R = np.sqrt(dx ** 2 + dy ** 2 + z ** 2)
            total = total + sx * sy * np.arctan(dx * dy / (z * R))
    return total / (2 * np.pi)


STEP = 0.05


def _stencil():
    offsets = [np.zeros(3)]
    for i in range(3):
        for sign in (1, -1):
            offsets.append(sign * STEP * np.eye(3)[i])
    offsets.append(np.zeros(3))
    return np.array(offsets)


def pixel_derivatives(x0, x1, y0, y1, points):
    """Gradient (shape (n_points, 3, n_pixels)) and vertical curvature (shape (n_points, n_pixels)) of the potential of
    every pixel at 1 V at points, by finite differences.
    """
    stencil = _stencil()
    gradients = np.zeros((len(points), 3, len(x0)))
    curvatures = np.zeros((len(points), len(x0)))
    for k, point in enumerate(points):
        phi = rectangle_potential(x0, x1, y0, y1, point[None, :] + stencil[:7])
        for a in range(3):
            gradients[k, a] = (phi[1 + 2 * a] - phi[2 + 2 * a]) / (2 * STEP)
        curvatures[k] = (phi[5] - 2 * phi[0] + phi[6]) / STEP ** 2
    return gradients, curvatures


def layout_nulls(layout, half_width, height=ION_HEIGHT, spacing=30.0, inner=40.0):
    """Points (µm) along the arms of a layout ("X junction", "T junction", "Y junction", or "Linear") where RF nulls are
    asked for, from inner from the center out to 3/4 of the half-width of the region.
    """
    angles = {"X junction": [0, 90, 180, 270], "T junction": [0, 90, 180], "Y junction": [90, 210, 330], "Linear": [0, 180]}[layout]
    distances = np.arange(inner, 0.75 * half_width + 1e-9, spacing)
    if layout == "Linear":
        distances = np.concatenate([[0.0], distances])
    points = []
    for angle in angles:
        direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        for d in distances:
            if layout == "Linear" and d == 0.0 and angle != 0:
                continue
            points.append([*(d * direction), height])
    return np.array(points)


def _orbits(centers, symmetry, tolerance=1e-6):
    """Group the pixels into orbits of a symmetry, returning the orbit of every pixel and the number of orbits."""
    transforms = {
        "none": [lambda p: p],
        "mirror x": [lambda p: p, lambda p: p * [1, -1]],
        "mirror xy": [lambda p: p, lambda p: p * [1, -1], lambda p: p * [-1, 1], lambda p: -p],
        "4-fold": [lambda p: p, lambda p: p * [1, -1], lambda p: p * [-1, 1], lambda p: -p,
                   lambda p: p[:, ::-1], lambda p: p[:, ::-1] * [1, -1], lambda p: p[:, ::-1] * [-1, 1], lambda p: -p[:, ::-1]],
    }[symmetry]
    keys = np.round(centers / tolerance ** 0.5).astype(np.int64)
    index = {tuple(k): i for i, k in enumerate(keys)}
    orbit = -np.ones(len(centers), dtype=int)
    n = 0
    for i in range(len(centers)):
        if orbit[i] >= 0:
            continue
        for transform in transforms:
            image = tuple(np.round(transform(centers[i:i + 1]) / tolerance ** 0.5).astype(np.int64)[0])
            j = index.get(image)
            if j is not None and orbit[j] < 0:
                orbit[j] = n
        n += 1
    return orbit, n


@dataclass
class TopologyResult:
    """Pixel design of RF electrodes: the pixel size (µm), the centers of the pixels (µm), the fraction of each pixel which
    is RF from the linear program and the rounded pattern, the RF null points, the confinement (vertical curvature of
    the RF potential, V/µm^2 per volt) at each null and the largest residual gradient (V/µm per volt) of the rounded
    pattern, and the polygons of its RF electrodes.
    """
    pixel: float
    centers: np.ndarray
    density: np.ndarray
    pattern: np.ndarray
    nulls: np.ndarray
    confinement: np.ndarray
    residual: float
    polygons: list = field(default_factory=list)
    shape: tuple = (0, 0)

    def to_trap(self, V_RF=40.0, Omega_RF=2 * np.pi * 40e6, solve=True):
        """Trap whose RF electrodes are the traced polygons (with the rest of the plane grounded)."""
        trap = Trap.from_polygons([np.column_stack([p, np.zeros(len(p))]) for p in self.polygons], len(self.polygons), V_RF, Omega_RF)
        if solve:
            trap.solve("analytical")
        return trap


def optimize_topology(half_width=200.0, pixel=10.0, layout="X junction", height=ION_HEIGHT, nulls=None, symmetry=None,
                      tolerance=1e-6, spacing=30.0, inner=40.0):
    """Design RF electrodes on a grid of pixels by linear programming (see the module docstring).

    Args:
        half_width: Half-width of the square design region around the origin (µm).
        pixel: Size of the pixels (µm).
        layout: Layout whose arms carry the RF nulls (see LAYOUTS), if nulls is not given.
        height: Height of the RF nulls (µm).
        nulls: Points (µm) where RF nulls are asked for, instead of those of the layout.
        symmetry: Symmetry of the pattern (see SYMMETRIES), which by default is that of the layout.
        tolerance: Largest gradient of the RF potential allowed at each null (V/µm per volt) in the linear program.
        spacing: Spacing of the nulls along the arms (µm).
        inner: Distance from the center of the first null on each arm (µm), since the center of a junction cannot
            hold a null at the height of the arms.

    Returns:
        TopologyResult
    """
    n = int(round(2 * half_width / pixel))
    edges = -half_width + pixel * np.arange(n + 1)
    cx, cy = np.meshgrid(0.5 * (edges[:-1] + edges[1:]), 0.5 * (edges[:-1] + edges[1:]), indexing="ij")
    centers = np.column_stack([cx.ravel(), cy.ravel()])
    x0, x1 = centers[:, 0] - pixel / 2, centers[:, 0] + pixel / 2
    y0, y1 = centers[:, 1] - pixel / 2, centers[:, 1] + pixel / 2
    points = layout_nulls(layout, half_width, height, spacing, inner) if nulls is None else np.asarray(nulls, dtype=float)
    if symmetry is None:
        symmetry = {"X junction": "4-fold", "T junction": "mirror x", "Y junction": "mirror x", "Linear": "mirror xy"}.get(layout, "none")
    gradients, curvatures = pixel_derivatives(x0, x1, y0, y1, points)
    if layout in ("T junction", "Y junction") and symmetry == "mirror x":
        # The T and Y junctions are symmetric about the y axis (the stem of the T), so x and y swap places
        orbit, n_orbits = _orbits(centers[:, ::-1], "mirror x")
    else:
        orbit, n_orbits = _orbits(centers, symmetry)
    # Sum the columns of each orbit
    T = np.zeros((len(centers), n_orbits))
    T[np.arange(len(centers)), orbit] = 1.0
    G = gradients @ T
    C = curvatures @ T

    best = None
    for sign in (1.0, -1.0):
        # Variables: the RF fraction of every orbit, and the weakest confinement t (maximized)
        c = np.zeros(n_orbits + 1)
        c[-1] = -1.0
        A_ub, b_ub = [], []
        for k in range(len(points)):
            for a in range(3):
                A_ub.append(np.concatenate([G[k, a], [0.0]]))
                b_ub.append(tolerance)
                A_ub.append(np.concatenate([-G[k, a], [0.0]]))
                b_ub.append(tolerance)
            # t <= sign * curvature
            A_ub.append(np.concatenate([-sign * C[k], [1.0]]))
            b_ub.append(0.0)
        bounds = [(0.0, 1.0)] * n_orbits + [(None, None)]
        result = optimize.linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs")
        if result.status == 0 and (best is None or -result.fun > -best[0].fun):
            best = (result, sign)
    if best is None:
        raise ValueError("no pixel pattern holds RF nulls at every point; try fewer nulls or a larger region")
    result, sign = best
    density = T @ result.x[:-1]
    pattern = density > 0.5
    residual = float(np.abs(np.einsum("kap,p->ka", gradients, pattern.astype(float))).max())
    confinement = np.abs(curvatures @ pattern.astype(float))
    polygons = trace_pixels(pattern.reshape(n, n), edges, edges)
    return TopologyResult(pixel, centers, density, pattern, points, confinement, residual, polygons, (n, n))


def trace_pixels(mask, x_edges, y_edges):
    """Trace the boundaries of the true pixels of a mask (indexed [x, y]) into counterclockwise polygons, with each hole
    cut open to the boundary around it.
    """
    nx, ny = mask.shape
    padded = np.zeros((nx + 2, ny + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask
    outgoing = {}
    for i in range(nx):
        for j in range(ny):
            if not mask[i, j]:
                continue
            I, J = i + 1, j + 1
            # Directed edges with the pixel on their left, between grid vertices (i, j)
            if not padded[I, J - 1]:
                outgoing.setdefault((i, j), []).append((i + 1, j))
            if not padded[I + 1, J]:
                outgoing.setdefault((i + 1, j), []).append((i + 1, j + 1))
            if not padded[I, J + 1]:
                outgoing.setdefault((i + 1, j + 1), []).append((i, j + 1))
            if not padded[I - 1, J]:
                outgoing.setdefault((i, j + 1), []).append((i, j))
    loops = []
    while outgoing:
        start = next(iter(outgoing))
        loop = [start]
        previous = None
        current = start
        while True:
            options = outgoing[current]
            if len(options) == 1 or previous is None:
                following = options[0]
            else:
                # Where pixels touch at a corner, turn right so that they stay separate
                incoming = np.subtract(current, previous)
                def turn(option):
                    out = np.subtract(option, current)
                    return incoming[0] * out[1] - incoming[1] * out[0]
                following = min(options, key=turn)
            options.remove(following)
            if not options:
                del outgoing[current]
            previous, current = current, following
            if current == start:
                break
            loop.append(current)
        points = np.array([[x_edges[i], y_edges[j]] for i, j in loop], dtype=float)
        loops.append(_simplify(points))
    outers = [loop for loop in loops if _signed_area(loop) > 0]
    holes = [loop for loop in loops if _signed_area(loop) < 0]
    for hole in holes:
        for k, outer in enumerate(outers):
            if MplPath(outer).contains_point(hole[0]):
                outers[k] = _cut_open(outer, hole)
                break
    return outers


def _signed_area(points):
    x, y = points[:, 0], points[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _simplify(points):
    """Remove vertices along straight lines."""
    keep = []
    n = len(points)
    for k in range(n):
        a, b, c = points[k - 1], points[k], points[(k + 1) % n]
        if abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])) > 1e-9:
            keep.append(b)
    return np.array(keep)


def _cut_open(outer, hole):
    """Join a hole to the polygon around it with a cut of zero width between their nearest vertices."""
    distances = np.linalg.norm(outer[:, None, :] - hole[None, :, :], axis=-1)
    i, j = np.unravel_index(np.argmin(distances), distances.shape)
    return np.vstack([outer[:i + 1], np.roll(hole, -j, axis=0), hole[j:j + 1], outer[i:]])
