"""Trap builder, which assembles a trap from parts such as linear sections, junctions, single electrodes, zones, grating
couplers, and detectors placed anywhere in the trap surface (which is what the "Trap builder" tab of the GUI drags and
drops).

Each part has a kind (see CATALOG), a position (µm) and rotation (degrees) in the trap surface, and parameters of its
kind. The parts are turned into a Trap by build_trap, where every RF electrode comes before every DC electrode, every
grating coupler is designed to focus on the nearest zone, and every detector is assigned to the nearest zone. Designs
can be saved to and loaded from JSON files.
"""

import json
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from matplotlib.path import Path as MplPath

from .lib import E_CHARGE, ION_HEIGHT, PHOTONICS_BEAMS, Detector, Grating, TiOptError, Trap, parse_layout, photonics_beam

# Transverse dimensions of a five-wire section whose RF null is at ION_HEIGHT for long rails in the gapless plane
# approximation, where the null height is sqrt(y_center y_rail); they are those of the arms of the junctions (see
# generate_junction_trap in traps.c), so that sections can be joined onto junctions
Y_CENTER = 0.6 * ION_HEIGHT
Y_RAIL = ION_HEIGHT * ION_HEIGHT / Y_CENTER
CENTER_WIDTH = 2 * Y_CENTER
RAIL_WIDTH = Y_RAIL - Y_CENTER
SEGMENT_LENGTH = ION_HEIGHT


@dataclass
class Geometry:
    """Electrodes (polygons with shape (n_vertices, 2) and whether each is RF), zones (types, capacity, and position),
    grating couplers (beam, position, and the index of the zone they address, or -1 for the nearest), detectors
    (position, width along x, and length along y), and ports (position, and the direction in degrees in which the trap
    axis leaves the part there, where other sections can be joined on) of a part, in µm.
    """

    electrodes: list = field(default_factory=list)
    zones: list = field(default_factory=list)
    gratings: list = field(default_factory=list)
    detectors: list = field(default_factory=list)
    ports: list = field(default_factory=list)

    def extend(self, other):
        self.electrodes += other.electrodes
        self.zones += other.zones
        self.gratings += other.gratings
        self.detectors += other.detectors
        self.ports += other.ports
        return self

    def bounds(self):
        """Smallest and largest x and y of everything in the geometry, as (x_min, y_min, x_max, y_max)."""
        points = [polygon for polygon, _ in self.electrodes]
        points += [np.array([position]) for _, _, position in self.zones]
        points += [np.array([position]) for _, position, _ in self.gratings]
        points += [np.array([[x - 0.5 * w, y - 0.5 * l], [x + 0.5 * w, y + 0.5 * l]]) for (x, y), w, l in self.detectors]
        if not points:
            return np.zeros(4)
        points = np.vstack(points)
        return np.concatenate([points.min(axis=0), points.max(axis=0)])


def _rectangle(x_min, x_max, y_min, y_max):
    return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=float)


def _linear_section(n_segments, segment_width, center_width, rail_width, segment_length):
    half_length = 0.5 * n_segments * segment_width
    y_center = 0.5 * center_width
    y_rail = y_center + rail_width
    y_segment = y_rail + segment_length
    geometry = Geometry(ports=[((half_length, 0.0), 0.0), ((-half_length, 0.0), 180.0)])
    geometry.electrodes.append((_rectangle(-half_length, half_length, y_center, y_rail), True))
    geometry.electrodes.append((_rectangle(-half_length, half_length, -y_rail, -y_center), True))
    geometry.electrodes.append((_rectangle(-half_length, half_length, -y_center, y_center), False))
    for side in (1, -1):
        for s in range(n_segments):
            x_min = -half_length + s * segment_width
            y_min, y_max = sorted((side * y_rail, side * y_segment))
            geometry.electrodes.append((_rectangle(x_min, x_min + segment_width, y_min, y_max), False))
    return geometry


def _arc(radius, start, end, step=2.0):
    """Points of an arc of a circle about the origin from angle start to angle end (degrees)."""
    n = max(int(np.ceil(abs(end - start) / step)), 1)
    angles = np.radians(np.linspace(start, end, n + 1))
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=-1)


def _annular_sector(r_in, r_out, start, end, step=2.0):
    return np.concatenate([_arc(r_out, start, end, step), _arc(r_in, end, start, step)])


def _curved_section(radius, angle, n_segments, center_width, rail_width, segment_length):
    """Five-wire section bent around a circle about the origin, whose trap axis has the given radius and turns through
    the given angle (degrees), symmetric about the +y axis, with n_segments DC segments on each side.
    """
    y_center = 0.5 * center_width
    y_rail = y_center + rail_width
    y_segment = y_rail + segment_length
    if radius - y_segment <= 1.0:
        raise ValueError(f"the radius of a curved section must be larger than {y_segment:.0f} µm, so that its inner electrodes fit")
    if not 0 < angle <= 360:
        raise ValueError("a curved section turns through between 0 and 360 degrees")
    start, end = 90.0 - 0.5 * angle, 90.0 + 0.5 * angle
    geometry = Geometry()
    geometry.electrodes.append((_annular_sector(radius + y_center, radius + y_rail, start, end), True))
    geometry.electrodes.append((_annular_sector(radius - y_rail, radius - y_center, start, end), True))
    geometry.electrodes.append((_annular_sector(radius - y_center, radius + y_center, start, end), False))
    edges = np.linspace(start, end, n_segments + 1)
    for r_in, r_out in ((radius + y_rail, radius + y_segment), (radius - y_segment, radius - y_rail)):
        for a, b in zip(edges[:-1], edges[1:]):
            geometry.electrodes.append((_annular_sector(r_in, r_out, a, b), False))
    # The axis leaves the ends along the tangent of the circle
    for a, direction in ((start, start - 90.0), (end, end + 90.0)):
        geometry.ports.append(((radius * np.cos(np.radians(a)), radius * np.sin(np.radians(a))), direction % 360))
    return geometry


def curved_axis(radius, angle, n_points=50):
    """Points along the trap axis of a curved section (see _curved_section), e.g. for zones and paths along it."""
    return _arc(radius, 90.0 - 0.5 * angle, 90.0 + 0.5 * angle, angle / max(n_points - 1, 1))


def _junction_ports(geometry, arms):
    """Ports at the ends of the arms of a junction, as far out along each arm as its electrodes reach."""
    vertices = np.vstack([polygon for polygon, _ in geometry.electrodes])
    ports = []
    for arm in arms:
        direction = np.array([np.cos(np.radians(arm)), np.sin(np.radians(arm))])
        along = vertices @ direction
        across = np.abs(vertices @ np.array([-direction[1], direction[0]]))
        reach = along[across < Y_RAIL].max()
        ports.append((tuple(reach * direction), arm % 360))
    return ports


def _segments(n_segments, segment_width, length):
    half_length = 0.5 * n_segments * segment_width
    geometry = Geometry()
    for s in range(n_segments):
        x_min = -half_length + s * segment_width
        geometry.electrodes.append((_rectangle(x_min, x_min + segment_width, -0.5 * length, 0.5 * length), False))
    return geometry


@lru_cache(maxsize=32)
def _trap_geometry(kind, arguments):
    """Electrodes and zones of a junction or point trap generated by the C library."""
    if kind == "junction":
        arms, n_segments, segment_width = arguments
        trap = Trap.junction(arms, n_segments, segment_width)
    else:
        n_electrodes, ring_width = arguments
        trap = Trap.from_preset(n_electrodes, ring_width, preset="point")
    geometry = Geometry()
    for e, polygon in enumerate(trap.electrode_polygons()):
        geometry.electrodes.append((polygon[:, :2].copy(), trap.is_rf(e)))
    for zone in trap.zones():
        geometry.zones.append((zone.types, zone.capacity, tuple(zone.position[:2])))
    trap.close()
    if kind == "junction":
        geometry.ports = _junction_ports(geometry, arguments[0])
    return geometry


def _arms(text):
    try:
        arms = tuple(float(value) for value in str(text).replace(";", ",").split(",") if value.strip())
    except ValueError:
        raise ValueError(f"the arms of a junction are angles in degrees separated by commas, not {text!r}") from None
    if len(arms) < 2:
        raise ValueError("a junction needs at least two arms")
    return arms


def _copy(geometry):
    return Geometry([(polygon.copy(), rf) for polygon, rf in geometry.electrodes], list(geometry.zones), list(geometry.gratings), list(geometry.detectors), list(geometry.ports))


@dataclass(frozen=True)
class PartKind:
    """Kind of part, with its name, category, default parameters, choices for any parameters which take one of a few
    values, and a function which builds its geometry (centered on the origin) from its parameters.
    """

    name: str
    category: str
    defaults: dict
    build: object
    description: str = ""
    choices: dict = field(default_factory=dict)


CATALOG = {
    "linear_section": PartKind(
        "Linear section", "Sections",
        dict(n_segments=5, segment_width=50.0, center_width=CENTER_WIDTH, rail_width=RAIL_WIDTH, segment_length=SEGMENT_LENGTH),
        lambda p: _linear_section(int(p["n_segments"]), p["segment_width"], p["center_width"], p["rail_width"], p["segment_length"]),
        "five-wire section (RF rails around a central DC electrode, with a row of DC segments on each side) whose RF "
        "null is at the ion height, which can be joined end to end with other sections and onto the arms of junctions",
    ),
    "curved_section": PartKind(
        "Curved section", "Sections",
        dict(radius=200.0, angle=90.0, n_segments=3, center_width=CENTER_WIDTH, rail_width=RAIL_WIDTH, segment_length=SEGMENT_LENGTH),
        lambda p: _curved_section(p["radius"], p["angle"], int(p["n_segments"]), p["center_width"], p["rail_width"], p["segment_length"]),
        "five-wire section bent around a circle (like the turns of a racetrack trap), whose position is the center of the "
        "circle, with the radius of its axis and the angle it turns through (degrees)",
    ),
    "junction": PartKind(
        "Junction", "Sections",
        dict(arms="0, 90, 180, 270", n_segments=2, segment_width=40.0),
        lambda p: _copy(_trap_geometry("junction", (_arms(p["arms"]), int(p["n_segments"]), float(p["segment_width"])))),
        "junction of five-wire arms at the given angles (degrees), with a junction zone at its center",
    ),
    "point_trap": PartKind(
        "Point trap", "Sections",
        dict(n_electrodes=6, ring_width=50.0),
        lambda p: _copy(_trap_geometry("point", (int(p["n_electrodes"]), float(p["ring_width"])))),
        "ring of RF which holds one ion above its center, surrounded by compensation electrodes",
    ),
    "rf_rail": PartKind(
        "RF rail", "Electrodes",
        dict(length=250.0, width=RAIL_WIDTH),
        lambda p: Geometry([(_rectangle(-0.5 * p["length"], 0.5 * p["length"], -0.5 * p["width"], 0.5 * p["width"]), True)]),
        "rectangular RF electrode (length along x, width along y)",
    ),
    "dc_electrode": PartKind(
        "DC electrode", "Electrodes",
        dict(length=50.0, width=SEGMENT_LENGTH),
        lambda p: Geometry([(_rectangle(-0.5 * p["length"], 0.5 * p["length"], -0.5 * p["width"], 0.5 * p["width"]), False)]),
        "rectangular DC electrode (length along x, width along y)",
    ),
    "dc_segments": PartKind(
        "DC segments", "Electrodes",
        dict(n_segments=5, segment_width=50.0, length=SEGMENT_LENGTH),
        lambda p: _segments(int(p["n_segments"]), p["segment_width"], p["length"]),
        "row of DC segments along x",
    ),
    "polygon": PartKind(
        "Electrode", "Electrodes",
        dict(vertices=[[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]], rf=0),
        lambda p: Geometry([(np.array(p["vertices"], dtype=float), bool(p["rf"]))]),
        "polygonal electrode (vertices relative to its position), which is RF if rf is 1",
    ),
    "zone": PartKind(
        "Zone", "Zones",
        dict(types="gate", capacity=0),
        lambda p: Geometry(zones=[(str(p["types"]), int(p["capacity"]), (0.0, 0.0))]),
        "zone where ions are loaded, stored, gated, measured, etc. (types joined by +, and a capacity of 0 for the "
        "default)",
    ),
    "grating": PartKind(
        "Grating coupler", "Photonics",
        dict(beam="qubit", zone=-1),
        lambda p: Geometry(gratings=[(str(p["beam"]), (0.0, 0.0), int(p["zone"]))]),
        "grating coupler which focuses a beam onto a zone (the nearest one if zone is -1)",
        dict(beam=PHOTONICS_BEAMS),
    ),
    "detector": PartKind(
        "Detector", "Photonics",
        dict(width=50.0, length=50.0),
        lambda p: Geometry(detectors=[((0.0, 0.0), p["width"], p["length"])]),
        "single-photon detector beneath a zone (width along x and length along y)",
    ),
}


@dataclass
class Part:
    """Part of a trap design, placed at (x, y) (µm) and rotated counterclockwise by rotation (degrees)."""

    kind: str
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in CATALOG:
            raise ValueError(f"unknown part {self.kind!r}; choose from {', '.join(CATALOG)}")
        self.params = {**CATALOG[self.kind].defaults, **self.params}

    @property
    def name(self):
        """Name of the part, e.g. "Linear section" or "Gate zone"."""
        if self.kind == "zone":
            return f"{str(self.params['types']).capitalize()} zone"
        if self.kind == "grating":
            return f"{str(self.params['beam']).capitalize()} grating"
        if self.kind == "polygon":
            return "RF electrode" if self.params["rf"] else "DC electrode"
        return CATALOG[self.kind].name

    def local_geometry(self):
        """Geometry of the part before it is rotated and moved into place."""
        return CATALOG[self.kind].build(self.params)

    def geometry(self):
        """Geometry of the part in the trap surface."""
        angle = np.radians(self.rotation)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        offset = np.array([self.x, self.y])

        def place(points):
            return np.asarray(points, dtype=float) @ rotation.T + offset

        local = self.local_geometry()
        # Detectors stay aligned with the axes, so their sides swap when the part is turned by 90 degrees
        turned = int(round(self.rotation / 90.0)) % 2 == 1
        return Geometry(
            [(place(polygon), rf) for polygon, rf in local.electrodes],
            [(types, capacity, tuple(place(position))) for types, capacity, position in local.zones],
            [(beam, tuple(place(position)), zone) for beam, position, zone in local.gratings],
            [(tuple(place(position)), length if turned else width, width if turned else length) for position, width, length in local.detectors],
            [(tuple(place(position)), (direction + self.rotation) % 360) for position, direction in local.ports],
        )

    def explode(self):
        """Split the part into one part for each of its electrodes, zones, grating couplers, and detectors."""
        geometry = self.geometry()
        parts = []
        for polygon, rf in geometry.electrodes:
            center = polygon.mean(axis=0)
            parts.append(Part("polygon", center[0], center[1], 0.0, dict(vertices=(polygon - center).tolist(), rf=int(rf))))
        for types, capacity, (x, y) in geometry.zones:
            parts.append(Part("zone", x, y, 0.0, dict(types=types, capacity=capacity)))
        for beam, (x, y), zone in geometry.gratings:
            parts.append(Part("grating", x, y, 0.0, dict(beam=beam, zone=zone)))
        for (x, y), width, length in geometry.detectors:
            parts.append(Part("detector", x, y, 0.0, dict(width=width, length=length)))
        return parts

    def to_dict(self):
        return dict(kind=self.kind, x=self.x, y=self.y, rotation=self.rotation, params=self.params)

    @classmethod
    def from_dict(cls, values):
        return cls(values["kind"], float(values.get("x", 0.0)), float(values.get("y", 0.0)), float(values.get("rotation", 0.0)), dict(values.get("params", {})))


def combine(parts):
    """Combine the geometry of every part of a design."""
    geometry = Geometry()
    for part in parts:
        geometry.extend(part.geometry())
    return geometry


# ---------------------------------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------------------------------


def _signed_area(polygon):
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _clip(subject, clipper):
    """Clip a polygon by a convex counterclockwise polygon (Sutherland-Hodgman)."""
    output = list(subject)
    for i in range(len(clipper)):
        a = clipper[i]
        b = clipper[(i + 1) % len(clipper)]
        edge = b - a
        points = output
        output = []
        if not points:
            break
        for j in range(len(points)):
            p = points[j]
            q = points[(j + 1) % len(points)]
            side_p = edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])
            side_q = edge[0] * (q[1] - a[1]) - edge[1] * (q[0] - a[0])
            if side_p >= 0:
                output.append(p)
            if (side_p >= 0) != (side_q >= 0):
                output.append(p + (q - p) * side_p / (side_p - side_q))
    return np.array(output) if len(output) >= 3 else None


def overlap_area(a, b):
    """Area of the intersection of two simple polygons (µm^2).

    Each polygon is a signed sum of the triangles between a common origin and each of its edges, so the area of the
    intersection is the signed sum of the areas of the intersections of those triangles, which are convex.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    origin = 0.5 * (np.maximum(a.min(axis=0), b.min(axis=0)) + np.minimum(a.max(axis=0), b.max(axis=0)))

    def fan(polygon):
        triangles = []
        for i in range(len(polygon)):
            triangle = np.array([origin, polygon[i], polygon[(i + 1) % len(polygon)]])
            area = _signed_area(triangle)
            if abs(area) > 1e-12:
                triangles.append((np.sign(area), triangle if area > 0 else triangle[::-1]))
        return triangles

    total = 0.0
    for sign_a, triangle_a in fan(a):
        for sign_b, triangle_b in fan(b):
            clipped = _clip(triangle_a, triangle_b)
            if clipped is not None:
                total += sign_a * sign_b * abs(_signed_area(clipped))
    return abs(total)


def _inside_points(polygon, depth):
    """Points just inside a polygon, by depth (µm) from each vertex (along the bisector) and from the middle of each edge.
    """
    points = np.asarray(polygon, dtype=float)
    orientation = np.sign(_signed_area(points)) or 1.0
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.maximum(np.linalg.norm(edges, axis=1, keepdims=True), 1e-12)
    # Inward normals, for a counterclockwise polygon to the left of each edge
    normals = orientation * np.stack([-edges[:, 1], edges[:, 0]], axis=-1) / lengths
    middles = points + 0.5 * edges + depth * normals
    bisectors = normals + np.roll(normals, 1, axis=0)
    bisectors /= np.maximum(np.linalg.norm(bisectors, axis=1, keepdims=True), 1e-12)
    corners = points + depth * bisectors
    return np.vstack([middles, corners])


def _edges_cross(a, b, tolerance=1e-3):
    """Check whether any edge of polygon a properly crosses an edge of polygon b, i.e. whether the ends of each lie
    further than tolerance (µm) on either side of the other, so that edges which only touch or lie along each other do
    not.
    """
    p, r = a, np.roll(a, -1, axis=0) - a
    q, s = b, np.roll(b, -1, axis=0) - b
    r_hat = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-300)
    s_hat = s / np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-300)

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    d = q[None, :, :] - p[:, None, :]
    # Distances of the ends of each edge of b from the line of each edge of a, and the other way around
    q_0 = cross(r_hat[:, None, :], d)
    q_1 = cross(r_hat[:, None, :], d + s[None, :, :])
    p_0 = cross(s_hat[None, :, :], -d)
    p_1 = cross(s_hat[None, :, :], -d + r[:, None, :])

    def opposite(u, v):
        return ((u > tolerance) & (v < -tolerance)) | ((u < -tolerance) & (v > tolerance))

    return bool(np.any(opposite(q_0, q_1) & opposite(p_0, p_1)))


def polygons_overlap(a, b, tolerance=1e-3):
    """Check whether the interiors of two polygons overlap by more than about tolerance (µm), so that polygons which
    only share edges or corners do not: they overlap if their edges cross, or if points just inside one are inside the
    other.
    """
    a = np.asarray(a, dtype=float)[:, :2]
    b = np.asarray(b, dtype=float)[:, :2]
    if np.any(a.max(axis=0) <= b.min(axis=0) + 1e-9) or np.any(b.max(axis=0) <= a.min(axis=0) + 1e-9):
        return False
    if _edges_cross(a, b, tolerance):
        return True
    return bool(MplPath(b).contains_points(_inside_points(a, tolerance)).any() or MplPath(a).contains_points(_inside_points(b, tolerance)).any())


def _electrodes_overlap(electrodes):
    return any(polygons_overlap(electrodes[i], electrodes[j]) for i in range(len(electrodes)) for j in range(i + 1, len(electrodes)))


@lru_cache(maxsize=256)
def _overlaps_itself(kind, params):
    """Check whether electrodes of a part overlap each other, which does not depend on where the part is placed."""
    part = Part(kind, params=json.loads(params))
    return _electrodes_overlap([polygon for polygon, _ in part.local_geometry().electrodes])


def find_overlaps(parts, changed=None):
    """Find the parts of a design whose electrodes overlap, which the gapless plane approximation counts twice.

    Args:
        parts: Parts of the design.
        changed: Indices of the parts to check against every other part (every part by default), e.g. the ones which
            just moved.

    Returns:
        List of (part index, part index) for every pair of parts with overlapping electrodes, where the indices are the
        same for a part whose own electrodes overlap
    """
    electrodes = [[polygon for polygon, _ in part.geometry().electrodes] for part in parts]
    bounds = [np.concatenate([np.vstack(e).min(axis=0), np.vstack(e).max(axis=0)]) if e else None for e in electrodes]
    overlaps = []
    for p, part in enumerate(parts):
        if (changed is None or p in changed) and len(electrodes[p]) > 1:
            if _overlaps_itself(part.kind, json.dumps(part.params, sort_keys=True)):
                overlaps.append((p, p))
    for p in range(len(parts)):
        for q in range(p + 1, len(parts)):
            if changed is not None and p not in changed and q not in changed:
                continue
            if bounds[p] is None or bounds[q] is None:
                continue
            if np.any(bounds[p][2:] <= bounds[q][:2] + 1e-9) or np.any(bounds[q][2:] <= bounds[p][:2] + 1e-9):
                continue
            if any(polygons_overlap(a, b) for a in electrodes[p] for b in electrodes[q]):
                overlaps.append((p, q))
    return overlaps


# ---------------------------------------------------------------------------------------------------------------------
# Building traps
# ---------------------------------------------------------------------------------------------------------------------


def build_trap(parts, element="40Ca+", V_RF=40.0, Omega_RF=2 * np.pi * 40e6, check_overlaps=True):
    """Build a trap from the parts of a design.

    Args:
        parts: Parts of the design.
        element: Ion species, which sets the light of the grating couplers and detectors (see photonics_beam).
        V_RF: Amplitude of the RF voltage (V).
        Omega_RF: Angular frequency of the RF drive (rad/s).
        check_overlaps: Specifies whether to refuse designs with overlapping electrodes.

    Returns:
        New trap, which still has to be solved
    """
    if check_overlaps:
        overlaps = find_overlaps(parts)
        if overlaps:
            names = ", ".join(f"{parts[i].name} and {parts[j].name}" if i != j else f"{parts[i].name} with itself" for i, j in overlaps[:5])
            raise ValueError(f"Electrodes overlap ({names}), so move them apart or delete one")

    geometry = combine(parts)
    rf = [polygon for polygon, is_rf in geometry.electrodes if is_rf]
    dc = [polygon for polygon, is_rf in geometry.electrodes if not is_rf]
    if not rf and not dc:
        raise ValueError("The design has no electrodes")

    trap = Trap.from_polygons(rf + dc, len(rf), V_RF, Omega_RF)
    try:
        for types, capacity, _ in geometry.zones:
            parse_layout(types)
        if geometry.zones:
            trap.set_zones([(types, capacity, (x, y, None)) for types, capacity, (x, y) in geometry.zones])
        zone_positions = np.array([[x, y] for _, _, (x, y) in geometry.zones]).reshape(-1, 2)

        def nearest_zone(position, zone=-1):
            if 0 <= zone < len(zone_positions):
                return zone
            if len(zone_positions) == 0:
                return -1
            return int(np.argmin(np.linalg.norm(zone_positions - np.asarray(position), axis=1)))

        gratings = []
        for beam, position, zone in geometry.gratings:
            if beam not in PHOTONICS_BEAMS:
                raise ValueError(f"unknown beam {beam!r}; choose from {', '.join(PHOTONICS_BEAMS)}")
            light = photonics_beam(element, beam)
            if light is None:
                raise ValueError(f"{element} does not use a {beam} beam")
            z = nearest_zone(position, zone)
            if z < 0:
                raise ValueError("Grating couplers focus on zones, so add a zone for every grating")
            waveguide, power = light
            grating = Grating(waveguide=waveguide, position=(position[0], position[1], 0.0), beam=beam, zone=z, etch_depth=waveguide.thickness, power=power)
            target = (zone_positions[z][0], zone_positions[z][1], ION_HEIGHT)
            grating.design(target)
            gratings.append(grating)

        detection = photonics_beam(element, "detection")
        detectors = []
        for position, width, length in geometry.detectors:
            detector = Detector(position=(position[0], position[1], 0.0), width=width, length=length, zone=nearest_zone(position))
            if detection is not None:
                detector.wavelength = detection[0].wavelength
            detectors.append(detector)

        if gratings or detectors:
            trap.set_photonics(gratings, detectors)
    except (ValueError, TiOptError):
        trap.close()
        raise
    return trap


def zone_nulls(trap, ion):
    """Find the RF null nearest each zone of a trap, across the local axis of the trap there (the direction along
    which the RF potential curves least, so that zones on curved sections and junction arms find their null too), from
    the gapless plane potentials of the RF electrodes (so zones beyond the spherical harmonic expansions are found too).

    Returns:
        (point (µm), pseudopotential (meV)) for each zone, or None for a zone without an RF null nearby
    """
    from .confinement import RFField

    field = RFField(trap)
    nulls = []
    for zone in trap.zones():
        x, y = zone.position[:2]
        point = np.array([x, y, ION_HEIGHT])
        direction = np.array([1.0, 0.0, 0.0])
        try:
            for _ in range(3):
                _, hessian = field.derivatives(point)
                values, vectors = np.linalg.eigh(hessian)
                axis = vectors[:, np.argmin(np.abs(values))].copy()
                axis[2] = 0.0
                # Where the potential curves alike in every direction in the surface (e.g. at the center of a junction),
                # any direction across will do
                if np.linalg.norm(axis) > 0.3:
                    direction = axis / np.linalg.norm(axis)
                point = field.null_across([[x, y, point[2]]], [direction])[0]
            energy = float(field.pseudopotential(ion, point)) / E_CHARGE * 1e3
        except (np.linalg.LinAlgError, ValueError):
            nulls.append(None)
            continue
        if not (np.all(np.isfinite(point)) and 1.0 < point[2] < 10 * ION_HEIGHT and np.hypot(point[0] - x, point[1] - y) < 2 * ION_HEIGHT):
            nulls.append(None)
            continue
        nulls.append((point, energy))
    return nulls


def parts_from_trap(trap):
    """Turn the electrodes, zones, grating couplers, and detectors of a trap into parts, e.g. to edit a preset."""
    parts = []
    for e, polygon in enumerate(trap.electrode_polygons()):
        center = polygon[:, :2].mean(axis=0)
        parts.append(Part("polygon", center[0], center[1], 0.0, dict(vertices=(polygon[:, :2] - center).tolist(), rf=int(trap.is_rf(e)))))
    for zone in trap.zones():
        parts.append(Part("zone", zone.position[0], zone.position[1], 0.0, dict(types=zone.types, capacity=zone.capacity)))
    for grating in trap.grating_couplers():
        if grating.beam is not None:
            parts.append(Part("grating", grating.position[0], grating.position[1], 0.0, dict(beam=grating.beam, zone=-1 if grating.zone is None else grating.zone)))
    for detector in trap.detectors():
        parts.append(Part("detector", detector.position[0], detector.position[1], 0.0, dict(width=detector.width, length=detector.length)))
    return parts


def save_design(parts, path):
    """Write the parts of a design to a JSON file."""
    with open(path, "w") as file:
        json.dump(dict(parts=[part.to_dict() for part in parts]), file, indent=1)


def load_design(path):
    """Read the parts of a design from a JSON file."""
    with open(path) as file:
        values = json.load(file)
    return [Part.from_dict(part) for part in values.get("parts", [])]
