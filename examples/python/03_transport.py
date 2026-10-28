"""Transport example, which shuttles an ion between two zones with an optimized waveform, follows it with velocity
Verlet, and runs a thermal ensemble through the same shuttle to see how much it heats up.
"""

import numpy as np
from _common import EXAMPLES, save
from matplotlib.figure import Figure

import tiopt

ion = tiopt.Ion("40Ca+")
trap = tiopt.Trap.from_file(EXAMPLES / "traps" / "linear_trap.trap")
trap.solve("analytical")

omega_axial = 2 * np.pi * 1e6
start = trap.find_rf_null((-50.0, 0.0, None))
end = trap.find_rf_null((50.0, 0.0, None))

for duration in (5e-6, 20e-6):
    waveform = tiopt.Waveform.transport(trap, ion, start, end, duration, 401, omega_axial)
    max_force, max_frequency_error = waveform.errors(trap, ion, omega_axial)
    trajectory = tiopt.simulate_trajectory(trap, ion, waveform, duration + 5e-6, 5e-9, start)
    quanta = tiopt.motional_quanta(ion, omega_axial, trajectory.positions[-1], trajectory.velocities[-1], end)
    print(f"{duration * 1e6:.0f} µs shuttle: peak voltage {np.abs(waveform.voltages).max():.2f} V, {quanta:.3f} axial quanta excited")

figure = Figure(figsize=(10, 4), layout="constrained")
ax = figure.add_subplot(1, 2, 1)
for e in range(trap.n_rf, trap.n_electrodes):
    ax.plot(waveform.times * 1e6, waveform.voltages[:, e], linewidth=0.8)
ax.set_xlabel("Time (µs)")
ax.set_ylabel("DC voltage (V)")
ax.set_title("Waveform of the 20 µs shuttle")

# A thermal ensemble with 5 quanta on average along each axis, run through the shuttle in parallel
rng = np.random.default_rng(0)
n_ions = 256
n_bar = 5.0
sigma_x = np.sqrt(tiopt.lib.HBAR * (2 * n_bar + 1) / (2 * ion.mass * omega_axial)) / tiopt.LENGTH_SCALE
sigma_v = sigma_x * omega_axial
positions = start + sigma_x * rng.standard_normal((n_ions, 3)) * [1, 0.25, 0.25]
velocities = sigma_v * rng.standard_normal((n_ions, 3)) * [1, 0, 0]
dt = 5e-9
final_positions, final_velocities = tiopt.simulate_ensemble(trap, ion, waveform, positions, velocities, int(25e-6 / dt), dt)
before = [tiopt.motional_quanta(ion, omega_axial, p, v, start) for p, v in zip(positions, velocities)]
after = [tiopt.motional_quanta(ion, omega_axial, p, v, end) for p, v in zip(final_positions, final_velocities)]
print(f"Ensemble of {n_ions} ions: {np.mean(before):.2f} axial quanta before the shuttle, {np.mean(after):.2f} after")

ax = figure.add_subplot(1, 2, 2)
ax.hist(before, bins=30, alpha=0.6, label="Before")
ax.hist(after, bins=30, alpha=0.6, label="After")
ax.set_xlabel("Axial quanta")
ax.set_ylabel("Ions")
ax.set_title("Thermal ensemble through the shuttle")
ax.legend()
save(figure, "03_transport.png")
