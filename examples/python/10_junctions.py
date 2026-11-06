"""Junction example, which generates the X, Y and T junctions of the preset library and a five-way junction of our own,
and follows the RF null from one arm through each junction into the next to find the pseudopotential barrier an ion
has to cross, which is the main figure of merit of a junction.
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import save

import tiopt

ion = tiopt.Ion("40Ca+")

junctions = {
    "X junction": (tiopt.Trap.from_preset(2, 40.0, preset="x_junction"), (0, 90)),
    "Y junction": (tiopt.Trap.from_preset(2, 40.0, preset="y_junction"), (0, 120)),
    "T junction": (tiopt.Trap.from_preset(2, 40.0, preset="t_junction"), (0, 90)),
    "Five-way junction": (tiopt.Trap.junction((0, 72, 144, 216, 288), n_segments=2, segment_width=40.0), (0, 72)),
}

# Turning from one arm into the next, 150 µm from the center (within the spherical harmonics expansions)
figure, axes = plt.subplots(2, 4, figsize=(15, 7), height_ratios=(3, 2))
for (name, (trap, (first, second))), ax_top, ax_path in zip(junctions.items(), axes[0], axes[1]):
    trap.solve("analytical")

    first_arm = np.array([np.cos(np.radians(first)), np.sin(np.radians(first)), 0.0])
    second_arm = np.array([np.cos(np.radians(second)), np.sin(np.radians(second)), 0.0])
    waypoints = [150 * first_arm + [0, 0, tiopt.ION_HEIGHT], (0.0, 0.0, tiopt.ION_HEIGHT), 150 * second_arm + [0, 0, tiopt.ION_HEIGHT]]
    positions, energies, barrier = trap.rf_null_path(ion, waypoints, 121)
    highest = positions[np.nanargmax(energies)]
    print(f"{name}: {trap.n_rf} RF and {trap.n_dc} DC electrodes, barrier of {barrier * 1e3:.2f} meV at ({highest[0]:.0f}, {highest[1]:.0f}, {highest[2]:.0f}) µm")

    for e, polygon in enumerate(trap.electrode_polygons()):
        color = "tab:orange" if trap.is_rf(e) else "tab:blue"
        ax_top.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.3, edgecolor=color, linewidth=0.5)
    ax_top.plot(positions[:, 0], positions[:, 1], color="black", linewidth=1)
    ax_top.plot(*highest[:2], "r*", markersize=10)
    ax_top.set_xlim(-250, 250)
    ax_top.set_ylim(-250, 250)
    ax_top.set_aspect("equal")
    ax_top.set_title(name)
    ax_top.set_xlabel("x (µm)")

    # Points where no minimum was found across the path are NaN, and show up as gaps
    distance = np.linspace(0.0, 300.0, len(energies))
    ax_path.plot(distance, energies * 1e3, color="tab:red")
    ax_path.set_xlabel("Distance along the path (µm)")
    ax_path.set_title(f"Barrier {barrier * 1e3:.2f} meV", fontsize=10)

axes[0, 0].set_ylabel("y (µm)")
axes[1, 0].set_ylabel("Pseudopotential (meV)")
figure.tight_layout()
save(figure, "10_junctions.png")
