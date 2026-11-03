"""Quickstart example, which loads the example trap with integrated photonics, solves its electrodynamics, and looks at
what an ion sees at the gate and measurement zones.
"""

import numpy as np
from _common import EXAMPLES

import tiopt

ion = tiopt.Ion("40Ca+")
trap = tiopt.Trap.from_file(EXAMPLES / "traps" / "linear_trap_photonics.trap")
trap.solve("analytical")

# RF null above the trap center (the gate zone) and its confinement
null = trap.find_rf_null((0.0, 0.0, None))
omega, axes = trap.normal_modes(ion, null)
print(f"RF null at (x, y, z) = ({null[0]:.2f}, {null[1]:.2f}, {null[2]:.2f}) µm")
print("Pseudopotential secular frequencies:", ", ".join(f"{w / (2 * np.pi * 1e6):.3f}" for w in omega), "MHz")
print(f"Trap depth {trap.trap_depth(ion, null) * 1e3:.1f} meV")

# A static well with a 1 MHz axial frequency at the gate zone
well = tiopt.Waveform.transport(trap, ion, null, null, 0.0, 1, omega_axial=2 * np.pi * 1e6)
voltages = well.voltages[0]
omega, axes = trap.normal_modes(ion, null, voltages)
print("Secular frequencies in the well:", ", ".join(f"{w / (2 * np.pi * 1e6):.3f}" for w in omega), "MHz")
print(f"Largest DC voltage {np.abs(voltages).max():.2f} V")

# Light from every grating coupler at the zone it addresses (the gate zone at the center and the measurement zone at
# x = 100 µm)
zones = trap.zones()
measure = trap.find_rf_null((zones[1].position[0], 0.0, None))
for grating in trap.grating_couplers():
    target = null if zones[grating.zone].types == "gate" else measure
    intensity, k_vector = grating.beam_at(target)
    theta, k_hat = grating.emission()
    print(f"{grating.beam:>9} beam ({grating.struct.waveguide.wavelength * 1e3:.0f} nm) for the {zones[grating.zone].types} zone: {intensity:.3e} W/m^2, emitted {np.degrees(theta):.1f}° from the normal")

# Qubit Rabi frequency and Lamb-Dicke parameter of the axial mode
intensity, k_vector = trap.grating("qubit").beam_at(null)
rabi = tiopt.rabi_frequency_from_intensity(ion.qubit_linewidth, ion.qubit_wavelength, intensity)
eta = tiopt.lamb_dicke_parameter(k_vector, (1.0, 0.0, 0.0), ion, 2 * np.pi * 1e6)
print(f"Qubit Rabi frequency 2π × {rabi / (2 * np.pi * 1e3):.1f} kHz, axial Lamb-Dicke parameter {eta:.4f}")

# State detection with the integrated detector beneath the measurement zone
detector = trap.detectors()[0]
detection_intensity, _ = trap.grating("detection").beam_at(measure)
gamma_SP = ion.struct.gamma_SP
wavelength_SP = 2 * np.pi * tiopt.lib.C_LIGHT / ion.struct.omega_SP
s = detection_intensity / tiopt.saturation_intensity(gamma_SP, wavelength_SP)
photon_rate = tiopt.scattering_rate(gamma_SP, s, -0.5 * gamma_SP) * detector.collection_efficiency(measure)
fidelity, threshold = tiopt.detection_fidelity(photon_rate, detector.dark_count_rate, ion.qubit_linewidth, 20e-6)
print(f"Detector collects {detector.collection_efficiency(measure) * 100:.2f}% of the light: {photon_rate:.3e} counts/s")
print(f"Detection in 20 µs: fidelity {fidelity:.6f} with a threshold of {threshold} counts")
