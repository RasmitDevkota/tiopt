"""Design rule checks of trap layouts before fabrication (like the DRC decks of KLayout which gdsfactory runs), and the
insetting of electrodes to open the gaps between them which a fabricated trap needs (like the gapping of Sion).

The rules cover the narrowest part of every electrode, the narrowest notch within one, the gaps between electrodes
(where electrodes which touch, as in tiopt's gapless designs, are pointed out so that they can be gapped before they are
made), sharp corners (which concentrate the field and are hard to make), tiny electrodes, overlapping electrodes, the
smallest teeth and grooves of grating couplers, and the radii of waveguide bends.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Rules:
    """Design rules (µm and degrees): the smallest width of an electrode, notch within one, and gap between electrodes
    (gaps of zero are electrodes which touch), the sharpest corner, the smallest area (µm^2), the smallest tooth or
    groove of a grating, and the smallest bend radius of a waveguide.
    """
    min_width: float = 5.0
    min_notch: float = 3.0
    min_gap: float = 3.0
    min_angle: float = 30.0
    min_area: float = 25.0
    min_feature: float = 0.15
    min_bend_radius: float = 10.0


@dataclass
class Violation:
    """Broken rule, with a description, where it is (µm), and its severity ("error", "warning", or "info")."""
    rule: str
    message: str
    location: tuple
    severity: str = "error"


def _orient(polygon):
    """Polygon with counterclockwise vertices."""
    polygon = np.asarray(polygon, dtype=float)
    x, y = polygon[:, 0], polygon[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return polygon if area >= 0 else polygon[::-1]


def polygon_area(polygon):
    polygon = np.asarray(polygon, dtype=float)
    x, y = polygon[:, 0], polygon[:, 1]
    return abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _edges(polygon):
    a = np.asarray(polygon, dtype=float)
    return a, np.roll(a, -1, axis=0)


def _point_segment(p, a, b):
    """Distances from points p to segments ab (broadcasting), and the closest points."""
    ab = b - a
    t = np.clip(np.sum((p - a) * ab, axis=-1) / np.maximum(np.sum(ab * ab, axis=-1), 1e-300), 0, 1)
    closest = a + t[..., None] * ab
    return np.linalg.norm(p - closest, axis=-1), closest


def _segment_distances(a0, a1, b0, b1):
    """Distances between every segment a (shape (n, 2)) and every segment b (shape (m, 2)), with the midpoints between
    their closest points, with shapes (n, m) and (n, m, 2).
    """
    A0, A1 = a0[:, None, :], a1[:, None, :]
    B0, B1 = b0[None, :, :], b1[None, :, :]
    candidates = [
        _point_segment(A0, B0, B1), _point_segment(A1, B0, B1),
        _point_segment(B0, A0, A1), _point_segment(B1, A0, A1),
    ]
    sources = [A0, A1, B0, B1]
    distances = np.stack([c[0] for c in candidates])
    k = np.argmin(distances, axis=0)
    d = np.take_along_axis(distances, k[None], axis=0)[0]
    points = np.zeros(d.shape + (2,))
    for index, ((_, closest), source) in enumerate(zip(candidates, sources)):
        mask = k == index
        midpoint = 0.5 * (closest + np.broadcast_to(source, closest.shape))
        points[mask] = midpoint[mask]

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    d1 = cross(A1 - A0, B0 - A0)
    d2 = cross(A1 - A0, B1 - A0)
    d3 = cross(B1 - B0, A0 - B0)
    d4 = cross(B1 - B0, A1 - B0)
    crossing = (d1 * d2 < 0) & (d3 * d4 < 0)
    d = np.where(crossing, 0.0, d)
    return d, points


def _normals(a0, a1):
    """Outward normals of the edges of a counterclockwise polygon."""
    t = a1 - a0
    n = np.column_stack([t[:, 1], -t[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]


def _width_and_notch(polygon):
    """Narrowest width (between facing edges with the material between them) and narrowest notch (with the outside
    between them) of a polygon, with their locations.
    """
    polygon = _orient(polygon)
    a0, a1 = _edges(polygon)
    n = len(a0)
    d, points = _segment_distances(a0, a1, a0, a1)
    normals = _normals(a0, a1)
    facing = normals @ normals.T < -0.5
    index = np.arange(n)
    adjacent = (np.abs(index[:, None] - index[None, :]) <= 1) | (np.abs(index[:, None] - index[None, :]) == n - 1)
    mask = facing & ~adjacent
    centers = 0.5 * (a0 + a1)
    offset = centers[None, :, :] - centers[:, None, :]
    # The other edge lies behind an edge (inside the material) for a width, and in front of it for a notch
    behind = np.sum(offset * normals[:, None, :], axis=-1) < 0
    out = {}
    for name, selection in (("width", mask & behind), ("notch", mask & ~behind)):
        if np.any(selection):
            values = np.where(selection, d, np.inf)
            i, j = np.unravel_index(np.argmin(values), values.shape)
            out[name] = (values[i, j], tuple(points[i, j]))
        else:
            out[name] = (np.inf, None)
    return out


def _angles(polygon):
    """Interior angles (degrees) at the vertices of a polygon."""
    polygon = _orient(polygon)
    previous = np.roll(polygon, 1, axis=0) - polygon
    following = np.roll(polygon, -1, axis=0) - polygon
    cosine = np.sum(previous * following, axis=1) / np.maximum(np.linalg.norm(previous, axis=1) * np.linalg.norm(following, axis=1), 1e-300)
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    turn = previous[:, 0] * following[:, 1] - previous[:, 1] * following[:, 0]
    # Reflex vertices of a counterclockwise polygon turn clockwise
    return np.where(turn > 0, 360 - angle, angle)


def check_polygons(polygons, rules=None, names=None):
    """Check electrode polygons (µm) against design rules.

    Returns:
        List of Violation
    """
    rules = Rules() if rules is None else rules
    names = names or [f"electrode {e}" for e in range(len(polygons))]
    violations = []
    polygons = [np.asarray(p, dtype=float)[:, :2] for p in polygons]
    for e, polygon in enumerate(polygons):
        if len(polygon) < 3:
            violations.append(Violation("vertices", f"{names[e]} has fewer than three vertices", tuple(polygon.mean(axis=0)) if len(polygon) else (0, 0)))
            continue
        area = polygon_area(polygon)
        if area < rules.min_area:
            violations.append(Violation("area", f"{names[e]} covers {area:.1f} µm², below {rules.min_area:g} µm²", tuple(polygon.mean(axis=0))))
        measures = _width_and_notch(polygon)
        width, where = measures["width"]
        if width < rules.min_width:
            violations.append(Violation("width", f"{names[e]} narrows to {width:.2f} µm, below {rules.min_width:g} µm", where))
        notch, where = measures["notch"]
        if notch < rules.min_notch:
            violations.append(Violation("notch", f"{names[e]} has a notch {notch:.2f} µm wide, below {rules.min_notch:g} µm", where))
        angles = _angles(polygon)
        sharp = np.flatnonzero(angles < rules.min_angle - 1e-6)
        for v in sharp:
            violations.append(Violation("angle", f"{names[e]} has a {angles[v]:.1f}° corner, sharper than {rules.min_angle:g}°", tuple(polygon[v]), "warning"))
    # Gaps between electrodes, after skipping pairs whose bounding boxes are far apart
    boxes = [(p.min(axis=0), p.max(axis=0)) for p in polygons if len(p) >= 3]
    touching = []
    for a in range(len(polygons)):
        for b in range(a + 1, len(polygons)):
            if len(polygons[a]) < 3 or len(polygons[b]) < 3:
                continue
            (lo_a, hi_a), (lo_b, hi_b) = boxes[a], boxes[b]
            if np.any(lo_a - hi_b > rules.min_gap) or np.any(lo_b - hi_a > rules.min_gap):
                continue
            a0, a1 = _edges(polygons[a])
            b0, b1 = _edges(polygons[b])
            d, points = _segment_distances(a0, a1, b0, b1)
            i, j = np.unravel_index(np.argmin(d), d.shape)
            gap = d[i, j]
            if gap < 1e-6:
                touching.append((a, b, tuple(points[i, j])))
            elif gap < rules.min_gap:
                violations.append(Violation("gap", f"{names[a]} and {names[b]} are {gap:.2f} µm apart, below {rules.min_gap:g} µm", tuple(points[i, j])))
    if touching:
        violations.append(Violation("gap", f"{len(touching)} pairs of electrodes touch (a gapless design), which need gaps before fabrication (see inset_polygons)",
                                    touching[0][2], "info"))
    return violations


def check_trap(trap, rules=None, routes=None):
    """Check the electrodes, overlaps, grating couplers, and (optionally) waveguide routes of a trap against design
    rules.

    Args:
        trap: Trap.
        rules: Rules.
        routes: Routes of tiopt.photonic_circuits.route_trap, whose bends are checked too.

    Returns:
        List of Violation
    """
    from . import builder

    rules = Rules() if rules is None else rules
    polygons = [np.asarray(p, dtype=float)[:, :2] for p in trap.electrode_polygons()]
    names = [f"{'RF' if trap.is_rf(e) else 'DC'} electrode {e}" for e in range(len(polygons))]
    violations = check_polygons(polygons, rules, names)
    for a in range(len(polygons)):
        for b in range(a + 1, len(polygons)):
            if builder.polygons_overlap(polygons[a], polygons[b]):
                violations.append(Violation("overlap", f"{names[a]} and {names[b]} overlap", tuple(polygons[a].mean(axis=0))))
    for g, grating in enumerate(trap.grating_couplers()):
        tooth = grating.period * grating.fill_factor
        groove = grating.period * (1 - grating.fill_factor)
        if min(tooth, groove) < rules.min_feature:
            violations.append(Violation("grating", f"grating {g} ({grating.beam or 'beam'}) has {'teeth' if tooth < groove else 'grooves'} {min(tooth, groove) * 1e3:.0f} nm wide, below {rules.min_feature * 1e3:.0f} nm",
                                        tuple(grating.position[:2])))
    for route in routes or []:
        radius = getattr(route, "radius", None)
        if radius is not None and radius < rules.min_bend_radius:
            violations.append(Violation("bend", f"the route to grating {route.grating} bends with radius {radius:g} µm", tuple(route.points[len(route.points) // 2])))
    return violations


def summary(violations):
    """Lines of text which describe a list of violations."""
    if not violations:
        return ["No design rules are broken"]
    counts = {}
    for violation in violations:
        counts[violation.severity] = counts.get(violation.severity, 0) + 1
    lines = [", ".join(f"{n} {severity}{'s' if n > 1 else ''}" for severity, n in counts.items())]
    order = {"error": 0, "warning": 1, "info": 2}
    for violation in sorted(violations, key=lambda v: order[v.severity]):
        x, y = violation.location if violation.location is not None else (np.nan, np.nan)
        lines.append(f"  [{violation.severity}] {violation.message} at ({x:.1f}, {y:.1f})")
    return lines


def inset_polygon(polygon, distance, max_miter=4.0):
    """Move every edge of a polygon inwards by a distance (µm), joining neighboring edges where they meet (with miters
    no longer than max_miter times the distance at sharp corners).
    """
    polygon = _orient(np.asarray(polygon, dtype=float)[:, :2])
    a0, a1 = _edges(polygon)
    normals = _normals(a0, a1)
    n = len(polygon)
    out = np.zeros_like(polygon)
    for v in range(n):
        before, after = (v - 1) % n, v
        n1, n2 = normals[before], normals[after]
        # Point on both inset edges: p - d n1 and p - d n2 move along each edge
        bisector = n1 + n2
        norm = np.linalg.norm(bisector)
        if norm < 1e-9:
            out[v] = polygon[v] - distance * n2
            continue
        bisector /= norm
        cosine = max(bisector @ n1, 1e-9)
        length = min(distance / cosine, max_miter * distance)
        out[v] = polygon[v] - length * bisector
    return out


def inset_polygons(polygons, gap):
    """Open gaps of a width (µm) between electrodes which touch, by insetting every electrode by half of the gap."""
    return [inset_polygon(polygon, gap / 2) for polygon in polygons]


def gapped_trap(trap, gap, solve=False):
    """Copy of a trap whose electrodes are inset to open gaps of a width (µm) between them, e.g. to write a layout for
    fabrication (with Trap.save_gds) or to solve with the boundary element method of tiopt.bem.
    """
    from .shapes import trap_from_polygons

    return trap_from_polygons(trap, inset_polygons(trap.electrode_polygons(), gap), solve=solve)
