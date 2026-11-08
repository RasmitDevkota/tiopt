"""Export trap designs to GDSII and SVG files, import electrodes and zones from GDSII files, and export potentials to
VTK files.

GDSII files hold the electrodes, integrated photonics, and zones of a trap for layout tools such as KLayout, with one
layer for each kind of shape (see LAYERS and GRATING_LAYERS). Coordinates are in µm, with a database unit of 1 nm.
SVG files draw the same shapes for figures and quick looks at a design. VTK files hold the potentials of a trap on a
grid of points (like the output of the bem package), which ParaView and VisIt can show in 3D.
"""

import math
import struct
import warnings
from pathlib import Path

import numpy as np

# Layers of the shapes in GDSII files: RF and DC electrodes, detectors, and zones (as text labels with their types)
LAYERS = {"rf": 1, "dc": 2, "detector": 20, "zone": 30}

# Grating couplers are drawn as rectangles on a layer for each beam (layer 19 for gratings which deliver other light)
GRATING_BEAMS = ("qubit", "detection", "repump", "quench", "photoionization", "ionization")
GRATING_LAYERS = {beam: 10 + b for b, beam in enumerate(GRATING_BEAMS)}
OTHER_GRATING_LAYER = 19

# Database unit in µm (1 nm) and in meters
_DATABASE_UNIT = 1e-3
_DATABASE_UNIT_METERS = 1e-9

# GDSII record types and data types
_HEADER, _BGNLIB, _LIBNAME, _UNITS, _ENDLIB, _BGNSTR, _STRNAME, _ENDSTR = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07
_BOUNDARY, _PATH, _SREF, _AREF, _TEXT, _LAYER, _DATATYPE, _XY, _ENDEL = 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x10, 0x11
_TEXTTYPE, _STRING, _BOX, _BOXTYPE = 0x16, 0x19, 0x2D, 0x2E
_NO_DATA, _INT16, _INT32, _REAL8, _ASCII = 0, 2, 3, 5, 6

# Colors of the shapes in SVG files
_RF_COLOR = "#e8833a"
_DC_COLOR = "#3c78b4"
_DETECTOR_COLOR = "#7d4fb8"
BEAM_COLORS = {"qubit": "#d62728", "detection": "#7b3fbf", "repump": "#8c564b", "quench": "#e377c2", "photoionization": "#1f77b4", "ionization": "#9467bd"}


def _record(record_type, data_type, payload=b""):
    """Pack one GDSII record."""
    return struct.pack(">HBB", 4 + len(payload), record_type, data_type) + payload


def _ascii(text):
    """Pack a string for a GDSII record, padded to an even length."""
    data = text.encode("ascii", "replace")
    return data + b"\0" * (len(data) % 2)


def _real8(value):
    """Pack a GDSII 8-byte real, which has a sign bit, a base-16 exponent in excess 64, and a 56-bit mantissa."""
    if value == 0:
        return b"\0" * 8
    sign = 0x80 if value < 0 else 0x00
    value = abs(value)
    exponent = 64
    while value >= 1:
        value /= 16
        exponent += 1
    while value < 1 / 16:
        value *= 16
        exponent -= 1
    mantissa = int(round(value * 2**56))
    if mantissa >= 2**56:
        mantissa //= 16
        exponent += 1
    return bytes([sign | exponent]) + mantissa.to_bytes(7, "big")


def _from_real8(data):
    """Unpack a GDSII 8-byte real."""
    sign = -1 if data[0] & 0x80 else 1
    exponent = (data[0] & 0x7F) - 64
    mantissa = int.from_bytes(data[1:8], "big")
    return sign * mantissa / 2**56 * 16.0**exponent


def _timestamp():
    return struct.pack(">12h", *([2026, 1, 1, 0, 0, 0] * 2))


def _boundary(layer, points):
    """Pack a closed polygon on a layer, with points in µm."""
    points = np.asarray(points, dtype=float)[:, :2]
    coordinates = np.round(points / _DATABASE_UNIT).astype(np.int64)
    coordinates = np.vstack([coordinates, coordinates[:1]])
    xy = struct.pack(f">{2 * len(coordinates)}i", *coordinates.ravel())
    return (
        _record(_BOUNDARY, _NO_DATA)
        + _record(_LAYER, _INT16, struct.pack(">h", layer))
        + _record(_DATATYPE, _INT16, struct.pack(">h", 0))
        + _record(_XY, _INT32, xy)
        + _record(_ENDEL, _NO_DATA)
    )


def _text(layer, position, string):
    """Pack a text label on a layer, at a position in µm."""
    x, y = (int(round(c / _DATABASE_UNIT)) for c in position[:2])
    return (
        _record(_TEXT, _NO_DATA)
        + _record(_LAYER, _INT16, struct.pack(">h", layer))
        + _record(_TEXTTYPE, _INT16, struct.pack(">h", 0))
        + _record(_XY, _INT32, struct.pack(">2i", x, y))
        + _record(_STRING, _ASCII, _ascii(string))
        + _record(_ENDEL, _NO_DATA)
    )


def grating_outline(grating):
    """Get the corners of the rectangle a grating coupler covers in the trap surface (µm), with its length along the
    direction of the guided light."""
    cos_a, sin_a = math.cos(grating.azimuth), math.sin(grating.azimuth)
    along = np.array([cos_a, sin_a]) * 0.5 * grating.length
    across = np.array([-sin_a, cos_a]) * 0.5 * grating.width
    center = grating.position[:2]
    return np.array([center - along - across, center + along - across, center + along + across, center - along + across])


def detector_outline(detector):
    """Get the corners of the active area of a detector (µm)."""
    x, y = detector.position[:2]
    w, l = 0.5 * detector.width, 0.5 * detector.length
    return np.array([[x - w, y - l], [x + w, y - l], [x + w, y + l], [x - w, y + l]])


def write_gds(trap, path, cell="TIOPT"):
    """Write the electrodes, grating couplers, detectors, and zones of a trap to a GDSII file.

    Args:
        trap: Trap to write.
        path: GDSII file to write.
        cell: Name of the cell which holds the shapes.
    """
    data = bytearray()
    data += _record(_HEADER, _INT16, struct.pack(">h", 600))
    data += _record(_BGNLIB, _INT16, _timestamp())
    data += _record(_LIBNAME, _ASCII, _ascii("TIOPT.DB"))
    data += _record(_UNITS, _REAL8, _real8(_DATABASE_UNIT) + _real8(_DATABASE_UNIT_METERS))
    data += _record(_BGNSTR, _INT16, _timestamp())
    data += _record(_STRNAME, _ASCII, _ascii(cell))

    for e, polygon in enumerate(trap.electrode_polygons()):
        data += _boundary(LAYERS["rf"] if trap.is_rf(e) else LAYERS["dc"], polygon)
    for grating in trap.grating_couplers():
        data += _boundary(GRATING_LAYERS.get(grating.beam, OTHER_GRATING_LAYER), grating_outline(grating))
    for detector in trap.detectors():
        data += _boundary(LAYERS["detector"], detector_outline(detector))
    for zone in trap.zones():
        data += _text(LAYERS["zone"], zone.position, zone.types)

    data += _record(_ENDSTR, _NO_DATA)
    data += _record(_ENDLIB, _NO_DATA)
    Path(path).write_bytes(bytes(data))


def read_gds(path):
    """Read the polygons and text labels of a GDSII file, with coordinates in µm.

    Polygons (boundaries and boxes) and texts are read from every cell; paths and references to other cells are skipped
    with a warning, so hierarchical layouts should be flattened (e.g. in KLayout) first.

    Args:
        path: GDSII file to read.

    Returns:
        Dictionary from each layer to its polygons, as (n_vertices, 2) arrays without the repeated closing vertex, and
        a list of (layer, string, position) for every text label
    """
    data = Path(path).read_bytes()
    unit = _DATABASE_UNIT
    polygons = {}
    texts = []
    skipped = set()

    element = None
    layer = 0
    xy = None
    string = ""
    offset = 0
    while offset + 4 <= len(data):
        length, record_type, _ = struct.unpack(">HBB", data[offset:offset + 4])
        if length < 4:
            break
        payload = data[offset + 4:offset + length]
        offset += length

        if record_type == _UNITS:
            # The second number is the database unit in meters
            unit = _from_real8(payload[8:16]) / 1e-6
        elif record_type in (_BOUNDARY, _BOX, _TEXT, _PATH, _SREF, _AREF):
            element = record_type
            xy = None
            string = ""
        elif record_type == _LAYER:
            layer = struct.unpack(">h", payload[:2])[0]
        elif record_type == _XY:
            xy = np.array(struct.unpack(f">{len(payload) // 4}i", payload), dtype=float).reshape(-1, 2) * unit
        elif record_type == _STRING:
            string = payload.rstrip(b"\0").decode("ascii", "replace")
        elif record_type == _ENDEL:
            if element in (_BOUNDARY, _BOX) and xy is not None:
                points = xy[:-1] if len(xy) > 1 and np.allclose(xy[0], xy[-1]) else xy
                polygons.setdefault(layer, []).append(points)
            elif element == _TEXT and xy is not None:
                texts.append((layer, string, xy[0]))
            elif element in (_PATH, _SREF, _AREF):
                skipped.add({_PATH: "paths", _SREF: "cell references", _AREF: "arrays of cell references"}[element])
            element = None
        elif record_type == _ENDLIB:
            break

    if skipped:
        warnings.warn(f"skipped {', '.join(sorted(skipped))} in {path}; flatten the layout and convert paths to polygons to import them")

    return polygons, texts


def write_svg(trap, path, scale=2.0, margin=20.0):
    """Draw the electrodes, grating couplers, detectors, and zones of a trap in an SVG file.

    Args:
        trap: Trap to draw.
        path: SVG file to write.
        scale: Pixels per µm.
        margin: Space around the trap (µm).
    """
    electrodes = [polygon[:, :2] for polygon in trap.electrode_polygons()]
    gratings = [(grating, grating_outline(grating)) for grating in trap.grating_couplers()]
    detectors = [detector_outline(detector) for detector in trap.detectors()]
    zones = trap.zones()

    points = np.vstack(electrodes + [outline for _, outline in gratings] + detectors + [np.zeros((1, 2))])
    x_min, y_min = points.min(axis=0) - margin
    x_max, y_max = points.max(axis=0) + margin
    width, height = (x_max - x_min) * scale, (y_max - y_min) * scale

    def svg_points(polygon):
        # SVG coordinates grow downwards, so y is flipped
        return " ".join(f"{(x - x_min) * scale:.2f},{(y_max - y) * scale:.2f}" for x, y in polygon)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.2f} {height:.2f}">',
        f'<rect width="100%" height="100%" fill="white"/>',
    ]
    for e, polygon in enumerate(electrodes):
        color = _RF_COLOR if trap.is_rf(e) else _DC_COLOR
        lines.append(f'<polygon points="{svg_points(polygon)}" fill="{color}" fill-opacity="0.35" stroke="{color}" stroke-width="1"/>')
    for detector in detectors:
        lines.append(f'<polygon points="{svg_points(detector)}" fill="none" stroke="{_DETECTOR_COLOR}" stroke-width="1.5" stroke-dasharray="4 3"/>')
    for grating, outline in gratings:
        color = BEAM_COLORS.get(grating.beam, "#555555")
        lines.append(f'<polygon points="{svg_points(outline)}" fill="{color}" fill-opacity="0.8" stroke="{color}" stroke-width="0.5"/>')
    for zone in zones:
        x, y = (zone.position[0] - x_min) * scale, (y_max - zone.position[1]) * scale
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="black"/>')
        lines.append(f'<text x="{x:.2f}" y="{y - 8:.2f}" font-family="sans-serif" font-size="12" text-anchor="middle">{zone.types}</text>')
    lines.append("</svg>")
    Path(path).write_text("\n".join(lines) + "\n")


def write_vtk(trap, path, ion=None, voltages=None, bounds=None, spacing=2.0, electrodes=False):
    """Write potentials of a trap on a grid of points to a legacy VTK file (structured points), which ParaView and
    VisIt can read.

    Args:
        trap: Solved trap.
        path: VTK file to write.
        ion: Ion species, for the pseudopotential and the potential energy (in meV).
        voltages: DC voltages (V) of the potential energy, if any.
        bounds: (x_min, x_max, y_min, y_max, z_min, z_max) of the grid (µm), which by default spans the region of the
            spherical harmonic expansions around the electrodes.
        spacing: Spacing of the grid (µm).
        electrodes: Specifies whether to also write the potential of every electrode at 1 V.
    """
    from .lib import E_CHARGE, constant, expansion_region

    if bounds is None:
        x_min, x_max, y_min, y_max = expansion_region()
        z_top = constant("SPH_CENTER_Z_MIN") + constant("SPH_SPACING") * (constant("NSPH_Z") - 1) + constant("SPH_R")
        bounds = (x_min, x_max, y_min, y_max, 5.0, z_top)
    x = np.arange(bounds[0], bounds[1] + spacing / 2, spacing)
    y = np.arange(bounds[2], bounds[3] + spacing / 2, spacing)
    z = np.arange(bounds[4], bounds[5] + spacing / 2, spacing)
    # VTK orders the points with x varying fastest
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    fields = []
    if ion is not None and trap.n_rf > 0:
        fields.append(("pseudopotential_meV", trap.pseudopotential(ion, points) / E_CHARGE * 1e3))
        if voltages is not None:
            U = trap.potential_energy(ion, points, voltages)
            fields.append(("potential_energy_meV", (U - U.min()) / E_CHARGE * 1e3))
    if voltages is not None:
        fields.append(("dc_potential_V", trap.potential(points, voltages)))
    if trap.n_rf > 0:
        fields.append(("rf_potential_per_volt", trap.potential(points, trap.rf_weights())))
    if electrodes:
        for e in range(trap.n_electrodes):
            weights = trap.zero_voltages()
            weights[e] = 1.0
            fields.append((f"electrode_{e}", trap.potential(points, weights)))
    lines = [
        "# vtk DataFile Version 3.0",
        "tiopt potentials (lengths in um)",
        "ASCII",
        "DATASET STRUCTURED_POINTS",
        f"DIMENSIONS {len(x)} {len(y)} {len(z)}",
        f"ORIGIN {x[0]:.6g} {y[0]:.6g} {z[0]:.6g}",
        f"SPACING {spacing:.6g} {spacing:.6g} {spacing:.6g}",
        f"POINT_DATA {len(points)}",
    ]
    for name, values in fields:
        lines.append(f"SCALARS {name} double 1")
        lines.append("LOOKUP_TABLE default")
        lines.extend(" ".join(f"{v:.9g}" for v in values[k:k + 8]) for k in range(0, len(values), 8))
    Path(path).write_text("\n".join(lines) + "\n")
