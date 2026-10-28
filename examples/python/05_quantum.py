"""Quantum dynamics example, which simulates Rabi oscillations of one ion on the carrier and the blue sideband, and a
Molmer-Sorensen gate on two ions with motional heating, using the master equation.
"""

import time

import numpy as np
from _common import save
from matplotlib.figure import Figure

import tiopt

ion = tiopt.Ion("40Ca+")
nu = 2 * np.pi * 1e6
figure = Figure(figsize=(10, 4), layout="constrained")

ax = figure.add_subplot(1, 2, 1)
for name, detuning in (("Carrier", 0.0), ("Blue sideband", nu)):
    pulse = tiopt.Pulse(start=0.0, duration=40e-6, detuning=detuning, rabi_frequency=2 * np.pi * 50e3, targets=[1], lamb_dicke=0.1)
    result = tiopt.simulate_pulses(ion, [pulse], n_ions=1, mode_frequency=nu, n_max=15, n_bar=0.5, n_samples=201)
    ax.plot(result.times * 1e6, result.excited_populations[:, 0], label=name)
ax.set_xlabel("Time (µs)")
ax.set_ylabel("D state population")
ax.set_title("Rabi oscillations of a thermal ion (n̄ = 0.5)")
ax.legend()

# Molmer-Sorensen gate: bichromatic light detuned by delta from both sidebands, with eta Omega = delta / 2
delta = 2 * np.pi * 20e3
eta = 0.05
ax = figure.add_subplot(1, 2, 2)
for heating_rate in (0.0, 1e3):
    pulses = [
        tiopt.Pulse(start=0.0, duration=2 * np.pi / delta, detuning=sign * (nu + delta), rabi_frequency=delta / (2 * eta), targets=[1, 2], lamb_dicke=eta)
        for sign in (1, -1)
    ]
    start = time.perf_counter()
    result = tiopt.simulate_pulses(ion, pulses, n_ions=2, mode_frequency=nu, n_max=20, heating_rate=heating_rate, n_samples=101)
    print(f"Molmer-Sorensen gate with heating {heating_rate:g} quanta/s simulated in {time.perf_counter() - start:.3f} s")
    ax.plot(result.times * 1e6, result.excited_populations[:, 0], label=f"P_D of ion 1, heating {heating_rate:g}/s")
    ax.plot(result.times * 1e6, result.mean_phonon_numbers, linestyle="--", label=f"n̄, heating {heating_rate:g}/s")
ax.set_xlabel("Time (µs)")
ax.set_title("Molmer-Sorensen gate")
ax.legend(fontsize=7)
save(figure, "05_quantum.png")
