"""Electrostatics example, which compares the analytical (gapless plane) and spectral (grounded box) solvers for the
two-electrode example and maps the pseudopotential of the linear trap.
"""

import time

import numpy as np
from _common import EXAMPLES, save
from matplotlib.figure import Figure

import tiopt

ion = tiopt.Ion("40Ca+")

# The two solvers for a small trap
squares = {}
for solver in ("analytical", "spectral"):
    trap = tiopt.Trap.from_file(EXAMPLES / "traps" / "two_squares.trap")
    start = time.perf_counter()
    trap.solve(solver)
    print(f"{solver} solve took {time.perf_counter() - start:.2f} s")
    squares[solver] = trap

z = np.linspace(5, 70, 66)
positions = np.column_stack([np.full_like(z, 25.0), np.zeros_like(z), z])
figure = Figure(figsize=(10, 4), layout="constrained")
ax = figure.add_subplot(1, 2, 1)
for solver, trap in squares.items():
    ax.plot(z, trap.potential(positions, [1.0, 0.0]), label=solver)
ax.set_xlabel("Height above electrode 0 (µm)")
ax.set_ylabel("Potential of electrode 0 at 1 V (V)")
ax.set_title("The spectral solver's box is grounded at its top (z = 74 µm)")
ax.legend()

# Pseudopotential of the linear trap in the plane across the trap axis
trap = tiopt.Trap.from_file(EXAMPLES / "traps" / "linear_trap.trap")
trap.solve("analytical")
y, z = np.meshgrid(np.linspace(-40, 40, 121), np.linspace(35, 75, 81))
grid = np.stack([np.zeros_like(y), y, z], axis=-1)
U = trap.pseudopotential(ion, grid) / tiopt.lib.E_CHARGE * 1e3
ax = figure.add_subplot(1, 2, 2)
image = ax.contourf(y, z, np.minimum(U, 100), levels=30)
figure.colorbar(image, ax=ax, label="Pseudopotential (meV)")
ax.set_xlabel("y (µm)")
ax.set_ylabel("z (µm)")
ax.set_title("Pseudopotential across the trap axis")
save(figure, "02_electrostatics.png")
