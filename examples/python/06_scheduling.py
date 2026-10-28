"""Scheduling example, which turns a circuit into a transport program on a line of zones and finds the waveform of each
transport of the program, which shuttles the ions it moves (as one crystal in one well) to the zone of their next
operation.

Note that Waveform.from_program instead holds every ion in its own harmonic well throughout the program, which needs
very large voltages whenever an ion is brought into a zone where another ion is waiting, since the two wells have to
stay separate until they coincide. Merging ions properly needs a merge primitive which lowers the axial frequency as the
wells combine.
"""

import numpy as np
from _common import EXAMPLES, save
from matplotlib.figure import Figure

import tiopt

ion = tiopt.Ion("40Ca+")
trap = tiopt.Trap.from_file(EXAMPLES / "traps" / "long_linear_trap_photonics.trap")
trap.solve("analytical")

circuit = tiopt.Circuit(EXAMPLES / "circuits" / "ghz3.qasm")
print(f"{len(circuit.instructions)} instructions on {circuit.n_ions} qubits")

# A zone to load and store ions, a gate zone and a measurement zone, separated by empty zones, 100 µm apart (wells
# closer than several ion heights need large voltages to be held at once)
types = ["load+storage", "empty", "gate", "empty", "measure"]
capacities = [circuit.n_ions, 1, 2, 1, circuit.n_ions]
height = trap.find_rf_null((0.0, 0.0, None))[2]
graph = tiopt.TrapGraph(types, capacities, -200.0, 200.0, height)
program = tiopt.TransportProgram(graph, circuit)

# Shuttle the ions of every transport from where they are to their destination, at 5 µm per µs
omega_axial = 2 * np.pi * 1e6
speed = 5.0
positions = program.initial_positions.copy()
history_times = [0.0]
history = [positions[:, 0].copy()]
shuttles = []
for targets, destination, arrival in zip(program.targets, program.destinations, program.arrival_times):
    start = positions[targets[-1] - 1]
    zone = types[int(np.argmin(np.abs(graph.positions[:, 0] - destination[0])))]
    distance = np.linalg.norm(destination - start)
    if distance > 1e-9:
        duration = distance / speed * 1e-6
        start_null = trap.find_rf_null((start[0], 0.0, None))
        end_null = trap.find_rf_null((destination[0], 0.0, None))
        shuttle = tiopt.Waveform.transport(trap, ion, start_null, end_null, duration, 201, omega_axial)
        max_force, max_frequency_error = shuttle.errors(trap, ion, omega_axial)
        shuttles.append(shuttle)
        print(f"Ions {targets} to the {zone} zone (x = {destination[0]:.0f} µm) by {arrival * 1e6:.1f} µs: {duration * 1e6:.0f} µs shuttle, peak voltage {np.abs(shuttle.voltages).max():.2f} V")
    else:
        print(f"Ions {targets} are already in the {zone} zone at {arrival * 1e6:.1f} µs")
    for target in targets:
        positions[target - 1] = destination
    history_times.append(arrival * 1e6)
    history.append(positions[:, 0].copy())

figure = Figure(figsize=(10, 4), layout="constrained")
ax = figure.add_subplot(1, 2, 1)
history = np.array(history)
for i in range(circuit.n_ions):
    ax.step(history_times, history[:, i] + 3.0 * i, where="post", label=f"Ion {i + 1}")
for position, zone in zip(graph.positions[:, 0], types):
    ax.axhline(position, color="gray", linewidth=0.5, linestyle=":")
    ax.text(history_times[-1], position, f" {zone}", va="center", fontsize=7)
ax.set_xlabel("Arrival time (µs)")
ax.set_ylabel("Zone position (µm, offset per ion)")
ax.set_title("Transport program")
ax.legend(fontsize=7)
ax = figure.add_subplot(1, 2, 2)
shuttle = shuttles[0]
ax.plot(shuttle.times * 1e6, shuttle.voltages[:, trap.n_rf:], linewidth=0.6)
ax.set_xlabel("Time (µs)")
ax.set_ylabel("DC voltage (V)")
ax.set_title("Waveform of the first shuttle")
save(figure, "06_scheduling.png")
