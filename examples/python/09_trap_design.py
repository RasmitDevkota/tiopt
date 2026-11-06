"""Trap design example, which generates a linear trap for a zone layout with integrated photonics for every zone,
evaluates every zone of the design, lists the light the photonics deliver for each species, and exports the design to
trap, GDSII and SVG files. A layout, zone spacing and species can be passed to try other designs, e.g.
python3 examples/python/09_trap_design.py --layout "load+storage, gate x3, measure" --zone-spacing 70 --species 88Sr+
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from _common import OUTPUT, REPOSITORY, save

import tiopt
from tiopt import export

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--layout", default="load, storage, gate x2, measure", help="zone layout, e.g. 'load, storage x2, gate x5, measure'")
parser.add_argument("--zone-spacing", type=float, default=80.0, help="distance between neighboring zones (µm)")
parser.add_argument("--segment-width", type=float, default=50.0, help="width of the DC segments (µm)")
parser.add_argument("--species", default="40Ca+", choices=tiopt.Ion.PRESETS)
args = parser.parse_args()

ion = tiopt.Ion(args.species)
trap = tiopt.Trap.from_layout(args.layout, args.zone_spacing, args.segment_width, element=args.species)
trap.solve("analytical")

print(f"{args.layout!r} with zones {args.zone_spacing:g} µm apart: {trap.n_rf} RF and {trap.n_dc} DC electrodes, {len(trap.grating_couplers())} grating couplers and {len(trap.detectors())} detector{'s' if len(trap.detectors()) != 1 else ''}")
for z, zone in enumerate(trap.zones()):
    beams = [grating.beam for grating in trap.grating_couplers() if grating.zone == z]
    print(f"  zone {z} ({zone.types}, {zone.capacity} ions) at x = {zone.position[0]:.0f} µm: {', '.join(beams) or 'no beams'}")

# Every zone of the design: its confinement, the light it receives, and detection
print()
print(tiopt.format_design_report(trap.design_report(ion)))

# The light which the photonics deliver for each species
print()
for element in tiopt.Ion.PRESETS:
    beams = []
    for beam in tiopt.PHOTONICS_BEAMS:
        light = tiopt.photonics_beam(element, beam)
        if light is not None:
            beams.append(f"{beam} {light[0].wavelength * 1e3:.0f} nm")
    print(f"{element:>7}: {', '.join(beams)}")

# Files for the rest of the team: a trap file for tiopt, a GDSII layout (RF electrodes on layer 1, DC electrodes on layer
# 2, gratings on layers 10-15, detectors on layer 20 and zone labels on layer 30) and an SVG drawing
OUTPUT.mkdir(exist_ok=True)
for path, write in ((OUTPUT / "trap_design.trap", trap.save), (OUTPUT / "trap_design.gds", trap.save_gds), (OUTPUT / "trap_design.svg", trap.save_svg)):
    write(path)
    print(f"Saved {path.relative_to(REPOSITORY)}")

# The design from above, and the RF null and pseudopotential along the trap axis
figure, (ax_top, ax_axis) = plt.subplots(2, 1, figsize=(9, 7), height_ratios=(2, 1))
for e, polygon in enumerate(trap.electrode_polygons()):
    color = "tab:orange" if trap.is_rf(e) else "tab:blue"
    ax_top.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.3, edgecolor=color)
for grating in trap.grating_couplers():
    corners = export.grating_outline(grating)
    ax_top.fill(corners[:, 0], corners[:, 1], color=export.BEAM_COLORS.get(grating.beam, "gray"), label=grating.beam)
for zone in trap.zones():
    ax_top.plot(*zone.position[:2], "ko", markersize=3)
    ax_top.annotate(zone.types, zone.position[:2], textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
handles, labels = ax_top.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax_top.legend(unique.values(), unique.keys(), fontsize=7, ncols=3, loc="lower right")
ax_top.set_aspect("equal")
ax_top.set_xlabel("x (µm)")
ax_top.set_ylabel("y (µm)")
ax_top.set_title(f"{args.layout} ({args.species})")

x_min, x_max, _, _ = tiopt.expansion_region()
reach = min(0.5 * (len(trap.zones()) - 1) * args.zone_spacing + 20.0, -x_min, x_max)
positions, energies, _ = trap.rf_null_path(ion, [(-reach, 0.0, tiopt.ION_HEIGHT), (reach, 0.0, tiopt.ION_HEIGHT)], 81)
ax_axis.plot(positions[:, 0], positions[:, 2], color="tab:blue")
ax_axis.set_xlabel("x (µm)")
ax_axis.set_ylabel("RF null height (µm)", color="tab:blue")
ax_energy = ax_axis.twinx()
ax_energy.plot(positions[:, 0], energies * 1e3, color="tab:red")
ax_energy.set_ylabel("Pseudopotential (meV)", color="tab:red")
ax_axis.set_title("Along the trap axis")
figure.tight_layout()
save(figure, "09_trap_design.png")
