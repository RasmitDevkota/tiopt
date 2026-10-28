"""Photonics example, which looks at the beams of the grating couplers of the example trap and at how the detector size
and detection time trade collection against dark counts and decay of the dark state.
"""

import numpy as np
from _common import EXAMPLES, save
from matplotlib.figure import Figure

import tiopt

ion = tiopt.Ion("40Ca+")
trap = tiopt.Trap.from_file(EXAMPLES / "traps" / "linear_trap_photonics.trap")
gratings = trap.grating_couplers()

figure = Figure(figsize=(12, 4), layout="constrained")

# The qubit beam in the plane of its emission
qubit = gratings[0]
theta, k_hat = qubit.emission()
print(f"Qubit grating: period {qubit.period:.4f} µm, emitted {np.degrees(theta):.1f}° from the normal, focused {qubit.focal_distance:.1f} µm away")
s, z = np.meshgrid(np.linspace(-60, 20, 161), np.linspace(1, 80, 160))
direction = np.array([k_hat[0], k_hat[1], 0.0]) / np.hypot(k_hat[0], k_hat[1])
points = np.stack([s * direction[0], s * direction[1], z], axis=-1)
ax = figure.add_subplot(1, 3, 1)
image = ax.pcolormesh(s, z, qubit.intensity(points), shading="auto")
figure.colorbar(image, ax=ax, label="Intensity (W/m²)")
ax.set_xlabel("Distance along the beam's azimuth (µm)")
ax.set_ylabel("z (µm)")
ax.set_title("729 nm beam focused on the gate zone")

# Collection and dark counts against detector size
measure = np.array([100.0, 0.0, tiopt.ION_HEIGHT])
sizes = np.linspace(10, 200, 40)
dark_count_density = 1e-3
detection_intensity, _ = gratings[1].beam_at(measure)
gamma = ion.struct.gamma_SP
s_397 = detection_intensity / tiopt.saturation_intensity(gamma, 2 * np.pi * tiopt.lib.C_LIGHT / ion.struct.omega_SP)
scattering = tiopt.scattering_rate(gamma, s_397, -0.5 * gamma)
fidelities = []
for size in sizes:
    detector = tiopt.Detector(position=(100.0, 0.0, 0.0), width=size, length=size, dark_count_rate=dark_count_density * size ** 2)
    fidelities.append(tiopt.detection_fidelity(scattering * detector.collection_efficiency(measure), detector.dark_count_rate, ion.qubit_linewidth, 5e-6)[0])
ax = figure.add_subplot(1, 3, 2)
ax.semilogy(sizes, 1 - np.array(fidelities))
ax.set_xlabel("Detector size (µm)")
ax.set_ylabel("Detection infidelity in 5 µs")
ax.set_title("Bigger detectors collect more light")

# Detection time
detector = trap.detectors()[0]
photon_rate = scattering * detector.collection_efficiency(measure)
times = np.logspace(-6, -3, 40)
ax = figure.add_subplot(1, 3, 3)
ax.loglog(times * 1e6, [1 - tiopt.detection_fidelity(photon_rate, detector.dark_count_rate, ion.qubit_linewidth, t)[0] for t in times])
ax.set_xlabel("Detection time (µs)")
ax.set_ylabel("Detection infidelity")
ax.set_title("Decay limits long detection")
save(figure, "04_photonics.png")
