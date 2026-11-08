"""Boundary element method (BEM) for the electrodes of a trap, which models what the gapless plane approximation leaves
out (like the bem package of the Haeffner lab and the solver of the shuttling toolchain of Conta et al.): real gaps
between the electrodes, a grounded plane beneath the electrodes (e.g. the substrate or the bottom of the trenches under
the gaps), and a grounded cover plane above the trap (e.g. a window coated with indium tin oxide).

The electrodes are thin conductors in the plane z = 0, which are meshed into panels by a quadtree (fine panels at the
edges, where the charge density diverges, and coarse panels inside), and the charge density on every panel is constant.
The potential of a uniformly charged polygon has a closed form, which gives the potential of every panel at the center
of every other (with a monopole and quadrupole expansion for distant panels), so that holding every panel of one
electrode at 1 V and every other panel at 0 V is a dense linear system for the charges. The grounded planes are included
by images of the panels. Optionally, a grounded frame around the trap stands in for the rest of the chip, which the
gapless plane approximation also holds at 0 V.

solve samples the potential of every electrode on the spheres of the spherical harmonic expansions of the trap (through
Trap.expand_samples), so every other part of tiopt then uses the BEM potentials.
"""

from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path as MplPath
from scipy import linalg

from .lib import sphere_sample_positions


def _openblas():
    """OpenBLAS (which tiopt links for its own linear algebra), whose LAPACK is much faster than a reference LAPACK, or
    None if it cannot be loaded.
    """
    import ctypes
    import ctypes.util

    for name in ("libopenblas.so.0", "libopenblas.so", ctypes.util.find_library("openblas")):
        if not name:
            continue
        try:
            library = ctypes.CDLL(name)
            library.LAPACKE_dgesv.restype = ctypes.c_int
            return library
        except (OSError, AttributeError):
            continue
    return None


_OPENBLAS = _openblas()


def solve_dense(A, B):
    """Solve A X = B for a dense square matrix A and right-hand sides B (shape (n, k)) by LU decomposition, with
    OpenBLAS if it is available.
    """
    import ctypes

    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64).reshape(len(A), -1)
    if _OPENBLAS is None:
        return linalg.lu_solve(linalg.lu_factor(A, overwrite_a=True, check_finite=False), B, check_finite=False)
    n, k = B.shape
    pivots = np.zeros(n, dtype=np.int32)
    double_p = ctypes.POINTER(ctypes.c_double)
    LAPACK_ROW_MAJOR = 101
    status = _OPENBLAS.LAPACKE_dgesv(
        ctypes.c_int(LAPACK_ROW_MAJOR), ctypes.c_int(n), ctypes.c_int(k), A.ctypes.data_as(double_p), ctypes.c_int(n),
        pivots.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), B.ctypes.data_as(double_p), ctypes.c_int(k)
    )
    if status != 0:
        raise np.linalg.LinAlgError(f"LAPACKE_dgesv failed with status {status}")
    return B


def _orient(polygon):
    polygon = np.asarray(polygon, dtype=float)[:, :2]
    x, y = polygon[:, 0], polygon[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return polygon if area >= 0 else polygon[::-1]


def _clip(subject, x0, y0, x1, y1):
    """Clip a polygon to a rectangle (Sutherland-Hodgman)."""
    def inside(p, edge):
        return [p[0] >= x0, p[0] <= x1, p[1] >= y0, p[1] <= y1][edge]

    def intersect(p, q, edge):
        if edge < 2:
            x = x0 if edge == 0 else x1
            t = (x - p[0]) / (q[0] - p[0])
        else:
            y = y0 if edge == 2 else y1
            t = (y - p[1]) / (q[1] - p[1])
        return p + t * (q - p)

    output = list(subject)
    for edge in range(4):
        if not output:
            break
        points, output = output, []
        for k in range(len(points)):
            p, q = points[k - 1], points[k]
            if inside(q, edge):
                if not inside(p, edge):
                    output.append(intersect(p, q, edge))
                output.append(q)
            elif inside(p, edge):
                output.append(intersect(p, q, edge))
    return np.array(output) if output else np.zeros((0, 2))


def _area_and_centroid(polygon):
    x, y = polygon[:, 0], polygon[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = 0.5 * cross.sum()
    if abs(area) < 1e-14:
        return 0.0, polygon.mean(axis=0)
    cx = np.sum((x + xn) * cross) / (6 * area)
    cy = np.sum((y + yn) * cross) / (6 * area)
    return area, np.array([cx, cy])


def _second_moments(polygon, centroid):
    """Second moments of area J_xx, J_yy, J_xy of a polygon about a point."""
    p = polygon - centroid
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    J_yy_x = np.sum((x ** 2 + x * xn + xn ** 2) * cross) / 12
    J_xx_y = np.sum((y ** 2 + y * yn + yn ** 2) * cross) / 12
    J_xy = np.sum((x * yn + 2 * x * y + 2 * xn * yn + xn * y) * cross) / 24
    return J_yy_x, J_xx_y, J_xy


def _segments_cross_box(a, b, x0, y0, x1, y1):
    """Whether any of the segments ab crosses a rectangle (Liang-Barsky)."""
    d = b - a
    t0 = np.zeros(len(a))
    t1 = np.ones(len(a))
    ok = np.ones(len(a), dtype=bool)
    for p, q in ((-d[:, 0], a[:, 0] - x0), (d[:, 0], x1 - a[:, 0]), (-d[:, 1], a[:, 1] - y0), (d[:, 1], y1 - a[:, 1])):
        parallel = np.abs(p) < 1e-15
        ok &= ~(parallel & (q < 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(parallel, 0.0, q / p)
        entering = (p < 0) & ~parallel
        leaving = (p > 0) & ~parallel
        t0 = np.where(entering, np.maximum(t0, r), t0)
        t1 = np.where(leaving, np.minimum(t1, r), t1)
    return bool(np.any(ok & (t0 <= t1)))


def _distance_to_boundary(point, a, b):
    ab = b - a
    t = np.clip(np.sum((point - a) * ab, axis=1) / np.maximum(np.sum(ab * ab, axis=1), 1e-300), 0, 1)
    return np.min(np.linalg.norm(point - (a + t[:, None] * ab), axis=1))


def mesh_polygon(polygon, h_min=4.0, h_max=60.0, grading=0.5, focus=None):
    """Mesh a polygon (µm) into panels by a quadtree, with squares of h_min at its edges (clipped to the polygon) which
    grow up to h_max away from them.

    Args:
        polygon: Vertices (µm).
        h_min: Size of the panels at the edges (µm).
        h_max: Largest size of a panel (µm).
        grading: How far from the edges (in units of their size) panels are divided.
        focus: Rectangle (x_min, y_min, x_max, y_max) whose nearby edges are resolved with h_min, while edges far from
            it are resolved with panels which grow with the distance (e.g. for a large grounded frame), or None.

    Returns:
        List of panels (each an array of counterclockwise vertices)
    """
    polygon = _orient(polygon)
    path = MplPath(polygon)
    a, b = polygon, np.roll(polygon, -1, axis=0)
    lo, hi = polygon.min(axis=0), polygon.max(axis=0)
    size = h_max
    # The cells sit on one grid through the origin for every polygon, so that symmetric designs get symmetric meshes
    start = np.floor(lo / size) * size
    nx = int(np.ceil((hi[0] - start[0]) / size)) or 1
    ny = int(np.ceil((hi[1] - start[1]) / size)) or 1
    stack = [(start[0] + i * size, start[1] + j * size, size) for i in range(nx) for j in range(ny)]
    panels = []
    while stack:
        x0, y0, s = stack.pop()
        x1, y1 = x0 + s, y0 + s
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        inside = path.contains_points(corners, radius=1e-9)
        vertex_inside = np.any((polygon[:, 0] > x0) & (polygon[:, 0] < x1) & (polygon[:, 1] > y0) & (polygon[:, 1] < y1))
        crossed = vertex_inside or (0 < inside.sum() < 4) or _segments_cross_box(a, b, x0 + 1e-9, y0 + 1e-9, x1 - 1e-9, y1 - 1e-9)
        limit = h_min
        if focus is not None:
            # Edges far from the focus only need panels which are a fraction of their distance from it
            dx = max(focus[0] - x1, x0 - focus[2], 0.0)
            dy = max(focus[1] - y1, y0 - focus[3], 0.0)
            limit = max(h_min, 0.15 * np.hypot(dx, dy))
        if not crossed:
            if not inside.all():
                continue
            # Cells next to an edge are divided until they reach the size there, which grades the mesh
            if s > limit * 1.0001 and _distance_to_boundary(np.array([x0 + s / 2, y0 + s / 2]), a, b) < (0.5 + grading) * s:
                h = s / 2
                stack += [(x0, y0, h), (x0 + h, y0, h), (x0, y0 + h, h), (x0 + h, y0 + h, h)]
            else:
                panels.append(corners)
            continue
        if s > limit * 1.0001:
            h = s / 2
            stack += [(x0, y0, h), (x0 + h, y0, h), (x0, y0 + h, h), (x0 + h, y0 + h, h)]
            continue
        clipped = _clip(polygon, x0, y0, x1, y1)
        if len(clipped) >= 3 and _area_and_centroid(clipped)[0] > 1e-6 * s * s:
            panels.append(clipped)
    return panels


# Potential of a uniformly charged polygon --------------------------------------------------------------------------------


def polygon_potential(vertices, points, z_plane=0.0):
    """Integral of 1/|r - r'| over polygons with unit charge density in the plane z = z_plane, at points.

    Args:
        vertices: Counterclockwise vertices with shape (n, k, 2) for n polygons of k vertices each (where repeated
            vertices close polygons with fewer).
        points: Points with shape (n, 3), one for each polygon.
        z_plane: Height of the plane of the polygons (µm).

    Returns:
        Integrals (µm) with shape (n,)
    """
    A = vertices
    B = np.roll(vertices, -1, axis=1)
    p = points[:, None, :2]
    h = np.abs(points[:, 2] - z_plane)[:, None]
    edge = B - A
    length = np.linalg.norm(edge, axis=-1)
    valid = length > 1e-12
    t = edge / np.where(valid, length, 1.0)[..., None]
    m = np.stack([t[..., 1], -t[..., 0]], axis=-1)
    d = np.sum((A - p) * m, axis=-1)
    s1 = np.sum((A - p) * t, axis=-1)
    s2 = np.sum((B - p) * t, axis=-1)
    R1 = np.sqrt(np.sum((A - p) ** 2, axis=-1) + h ** 2)
    R2 = np.sqrt(np.sum((B - p) ** 2, axis=-1) + h ** 2)
    R0 = d ** 2 + h ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where(np.abs(d) > 1e-12, d * np.log(np.maximum(s2 + R2, 1e-300) / np.maximum(s1 + R1, 1e-300)), 0.0)
        angle_term = np.arctan2(d * s2, R0 + h * R2) - np.arctan2(d * s1, R0 + h * R1)
    total = np.where(valid, log_term - h * angle_term, 0.0)
    return np.sum(total, axis=1)


@dataclass
class Mesh:
    """Panels of the electrodes: their vertices (padded to the same count by repeating the last), areas (µm^2),
    centroids (µm), second moments of area, sizes (µm), the electrode of each (-1 for the grounded frame), and the
    number of vertices of each.
    """
    vertices: np.ndarray
    areas: np.ndarray
    centroids: np.ndarray
    moments: np.ndarray
    sizes: np.ndarray
    electrodes: np.ndarray
    counts: np.ndarray = None

    @property
    def n_panels(self):
        return len(self.areas)


def mesh_trap(polygons, h_min=4.0, h_max=60.0, frame=None, frame_gap=5.0, grading=0.5):
    """Mesh the electrode polygons of a trap, with a grounded frame of a width (µm) around them (separated by frame_gap)
    if frame is given, whose panels grow with the distance from the trap.
    """
    panels = []
    owners = []
    for e, polygon in enumerate(polygons):
        for panel in mesh_polygon(polygon, h_min, h_max, grading):
            panels.append(panel)
            owners.append(e)
    if frame:
        points = np.vstack([np.asarray(p)[:, :2] for p in polygons])
        lo, hi = points.min(axis=0) - frame_gap, points.max(axis=0) + frame_gap
        outer_lo, outer_hi = lo - frame, hi + frame
        # The frame is four rectangles around the trap
        strips = [
            [[outer_lo[0], outer_lo[1]], [outer_hi[0], outer_lo[1]], [outer_hi[0], lo[1]], [outer_lo[0], lo[1]]],
            [[outer_lo[0], hi[1]], [outer_hi[0], hi[1]], [outer_hi[0], outer_hi[1]], [outer_lo[0], outer_hi[1]]],
            [[outer_lo[0], lo[1]], [lo[0], lo[1]], [lo[0], hi[1]], [outer_lo[0], hi[1]]],
            [[hi[0], lo[1]], [outer_hi[0], lo[1]], [outer_hi[0], hi[1]], [hi[0], hi[1]]],
        ]
        for strip in strips:
            # The frame only needs to hold the plane around the trap at 0 V, so its panels grow away from the trap
            for panel in mesh_polygon(np.array(strip), h_min, max(h_max, frame / 4), grading, focus=(lo[0], lo[1], hi[0], hi[1])):
                panels.append(panel)
                owners.append(-1)
    k = max(len(p) for p in panels)
    counts = np.array([len(p) for p in panels], dtype=np.int32)
    vertices = np.zeros((len(panels), k, 2))
    areas = np.zeros(len(panels))
    centroids = np.zeros((len(panels), 2))
    moments = np.zeros((len(panels), 3))
    sizes = np.zeros(len(panels))
    for i, panel in enumerate(panels):
        vertices[i, :len(panel)] = panel
        vertices[i, len(panel):] = panel[-1]
        area, centroid = _area_and_centroid(panel)
        areas[i] = area
        centroids[i] = centroid
        moments[i] = _second_moments(panel, centroid)
        sizes[i] = np.max(np.linalg.norm(panel - centroid, axis=1)) * 2
    return Mesh(vertices, areas, centroids, moments, sizes, np.array(owners), counts)


def _images(ground_depth, cover_height, n_terms):
    """Heights and signs of the images of a charge at z = 0 in grounded planes at z = -ground_depth and z = cover_height."""
    images = [(0.0, 1.0)]
    if ground_depth is not None and cover_height is None:
        images.append((-2 * ground_depth, -1.0))
    elif cover_height is not None and ground_depth is None:
        images.append((2 * cover_height, -1.0))
    elif ground_depth is not None and cover_height is not None:
        L = ground_depth + cover_height
        for n in range(-n_terms, n_terms + 1):
            if n != 0:
                images.append((2 * n * L, 1.0))
            images.append((-2 * ground_depth + 2 * n * L, -1.0))
    return images


def _kernel(mesh, points, images, near=4.0, charges=None):
    """Potential at points (µm) with shape (n, 3) of every panel with unit charge density (with shape (n, n_panels)),
    or of the given charges of the panels (with shape (n_panels, n_sets), giving shape (n, n_sets)), using the closed
    form for nearby panels and the quadrupole expansion for distant ones (in evaluate_panel_potentials of the C library).
    """
    from .lib import _int_p, _lib, _ptr

    points = np.ascontiguousarray(np.asarray(points, dtype=float).reshape(-1, 3))
    heights = np.ascontiguousarray([h for h, _ in images], dtype=float)
    signs = np.ascontiguousarray([s for _, s in images], dtype=float)
    vertices = np.ascontiguousarray(mesh.vertices, dtype=float)
    areas = np.ascontiguousarray(mesh.areas, dtype=float)
    centroids = np.ascontiguousarray(mesh.centroids, dtype=float)
    moments = np.ascontiguousarray(mesh.moments, dtype=float)
    sizes = np.ascontiguousarray(mesh.sizes, dtype=float)
    counts = np.ascontiguousarray(mesh.counts, dtype=np.int32)
    if charges is not None:
        charges = np.ascontiguousarray(charges, dtype=float).reshape(mesh.n_panels, -1)
        out = np.zeros((len(points), charges.shape[1]))
    else:
        out = np.zeros((len(points), mesh.n_panels))
    _lib.evaluate_panel_potentials(
        len(points), _ptr(points), mesh.n_panels, vertices.shape[1], counts.ctypes.data_as(_int_p), _ptr(vertices), _ptr(areas),
        _ptr(centroids), _ptr(moments), _ptr(sizes), len(images), _ptr(heights), _ptr(signs), float(near),
        0 if charges is None else charges.shape[1], _ptr(charges) if charges is not None else None, _ptr(out)
    )
    return out


@dataclass
class BEMSolution:
    """Charges of every panel (in units of 4 pi epsilon_0 V/µm) with every electrode at 1 V and the others grounded,
    along with the mesh and the grounded planes.
    """
    mesh: Mesh
    charges: np.ndarray
    n_electrodes: int
    images: list = field(default_factory=list)

    def potential(self, points):
        """Potential (V/V) of every electrode at points (µm) with shape (n, 3), with shape (n, n_electrodes)."""
        return _kernel(self.mesh, points, self.images, charges=self.charges)


MAX_PANELS = 24000


def solve_bem(polygons, h_min=4.0, h_max=60.0, ground_depth=None, cover_height=None, frame=3000.0, frame_gap=5.0,
              n_image_terms=4, grading=0.5, max_panels=MAX_PANELS):
    """Solve for the charges on the electrodes (each at 1 V with the others grounded) by the boundary element method.

    Args:
        polygons: Electrode polygons (µm), which must not overlap (gaps between them are left open).
        h_min: Size of the panels at the edges of the electrodes (µm).
        h_max: Largest size of a panel (µm).
        ground_depth: Depth of a grounded plane beneath the electrodes (µm), or None for none.
        cover_height: Height of a grounded cover plane above the trap (µm), or None for none.
        frame: Width of a grounded frame around the electrodes (µm), or None for none.
        frame_gap: Gap between the electrodes and the frame (µm).
        n_image_terms: Number of reflections kept when there are two grounded planes.
        grading: How quickly the panels grow away from the edges of the electrodes.
        max_panels: Largest number of panels, since the dense system takes 8 n^2 bytes.

    Returns:
        BEMSolution
    """
    mesh = mesh_trap(polygons, h_min, h_max, frame, frame_gap, grading)
    if mesh.n_panels > max_panels:
        raise ValueError(f"the mesh has {mesh.n_panels} panels (more than {max_panels}, which would take {8 * mesh.n_panels ** 2 / 1e9:.1f} GB); use larger panels")
    images = _images(ground_depth, cover_height, n_image_terms)
    collocation = np.column_stack([mesh.centroids, np.zeros(mesh.n_panels)])
    P = _kernel(mesh, collocation, images)
    n_electrodes = len(polygons)
    rhs = np.zeros((mesh.n_panels, n_electrodes))
    owned = mesh.electrodes >= 0
    rhs[np.flatnonzero(owned), mesh.electrodes[owned]] = 1.0
    charges = solve_dense(P, rhs)
    return BEMSolution(mesh, charges, n_electrodes, images)


def solve(trap, gap=0.0, **kwargs):
    """Solve the electrodynamics of a trap by the boundary element method and expand the potential of every electrode in
    spherical harmonics, so that every other part of tiopt uses it (like Trap.solve).

    Args:
        trap: Trap, whose electrodes are inset by half of gap (µm) first to open gaps between them.
        gap: Width of the gaps between electrodes (µm).
        kwargs: Arguments of solve_bem.

    Returns:
        BEMSolution
    """
    from .drc import inset_polygons

    polygons = [np.asarray(p, dtype=float)[:, :2] for p in trap.electrode_polygons()]
    if gap > 0:
        polygons = inset_polygons(polygons, gap)
    solution = solve_bem(polygons, **kwargs)
    samples = solution.potential(sphere_sample_positions())
    for e in range(trap.n_electrodes):
        trap.expand_samples(e, samples[:, e])
    trap.solved = True
    trap.solver = "bem"
    return solution
