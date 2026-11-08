"""Crystal and gate example, which holds a mixed-species crystal in a well of the linear trap, looks at its normal modes
and the depth and stability of the well beyond the pseudopotential, finds the control voltages of the multipoles, and
designs an amplitude-modulated Mølmer-Sørensen gate which closes every axial mode of a two-ion chain and stays good when
the modes drift.
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import save

import tiopt
from tiopt import crystals, gates, multimode, multipoles, stability, voltages

ion = tiopt.Ion("40Ca+")
trap = tiopt.Trap.from_preset(7, 50.0)
trap.solve("analytical")
null = trap.find_rf_null((0.0, 0.0, None))

# A 1 MHz well whose radial axes are tilted by 10 degrees
well, info = voltages.static_well(trap, ion, null, 2 * np.pi * 1e6, tilt=10.0)
print("\n".join(info.summary()))
print("\n".join(stability.report(trap, ion, null, well, n_ions=0)))

# A crystal of three 40Ca+ ions and one 9Be+ ion in the well
crystal = crystals.find_crystal(trap, ["40Ca+", "40Ca+", "9Be+", "40Ca+"], voltages=well, position=null)
print("\n".join(crystal.summary()))

# The control voltages which make one unit of each multipole on its own
control = multipoles.control_matrix(trap, null)
shims = multipoles.shim_matrix(control)
# The multipoles are in V/µm and V/µm^2, so a field of 100 V/m takes 1e-4 of the first column
print("Voltages for Ex = 100 V/m (mV):", np.round(shims[:, 0] * 1e-4 * 1e3, 2))

# A gate on both axial modes of two 40Ca+ ions, which closes every mode and their derivatives by the mode frequencies
pair = crystals.find_crystal(trap, ion, n_ions=2, voltages=well, position=null)
modes = multimode.Modes.from_crystal(pair, [2 * np.pi / ion.qubit_wavelength, 0.0, 0.0], pair.modes_along("x"), cutoffs=5)
duration = 100e-6
gate = gates.design_am_gate(modes.frequencies, modes.lamb_dicke, duration, modes.frequencies.max() + 2 * np.pi * 20e3)
print("\n".join(gate.summary()))
drifts = np.linspace(-2, 2, 5) * 2 * np.pi * 1e3
fidelities = []
for drift in drifts:
    drifted = multimode.Modes(modes.frequencies + drift, modes.lamb_dicke, modes.cutoffs, modes.n_bar)
    fidelities.append(multimode.simulate(2, drifted, gate.tones(), duration, n_samples=3, rwa_cutoff=2 * np.pi * 0.6e6).bell_fidelity())
    print(f"Drift of {drift / (2 * np.pi * 1e3):+.1f} kHz: Bell state fidelity {fidelities[-1]:.5f}")

figure, (ax_crystal, ax_gate, ax_drift) = plt.subplots(1, 3, figsize=(15, 4))
for element, marker in (("40Ca+", "o"), ("9Be+", "s")):
    members = [i for i, member in enumerate(crystal.ions) if member.element == element]
    ax_crystal.plot(crystal.positions[members, 0], crystal.positions[members, 2], marker, label=element)
ax_crystal.set_xlabel("x (µm)")
ax_crystal.set_ylabel("z (µm)")
ax_crystal.set_title("Mixed-species crystal")
ax_crystal.legend()
ax_gate.stairs(gate.amplitudes / (2 * np.pi * 1e3), gate.edges * 1e6)
ax_gate.set_xlabel("Time (µs)")
ax_gate.set_ylabel("Rabi frequency (kHz)")
ax_gate.set_title("Robust amplitude-modulated gate")
ax_drift.plot(drifts / (2 * np.pi * 1e3), fidelities, marker="o")
ax_drift.set_xlabel("Drift of the mode frequencies (kHz)")
ax_drift.set_ylabel("Bell state fidelity")
save(figure, "11_crystals_and_gates.png")
