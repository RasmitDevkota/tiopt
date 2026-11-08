"""Fabrication and architecture example, which checks the design rules of a trap for a zone layout, opens gaps between
its electrodes and solves it with the boundary element method, routes waveguides to its gratings, designs a tree of
couplers which shares one beam among its gate zones, and compiles a circuit onto QCCD architectures of several trap
capacities.
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import save

import tiopt
from tiopt import bem, circuits, components, drc, photonic_circuits, qccd

ion = tiopt.Ion("40Ca+")
trap = tiopt.Trap.from_layout("load+storage, gate x2, measure", zone_spacing=80.0)
print("\n".join(drc.summary(drc.check_trap(trap))))

# Gaps of 5 µm, solved with the boundary element method and compared to the gapless plane approximation
gapless = tiopt.Trap.from_layout("load+storage, gate x2, measure", zone_spacing=80.0)
gapless.solve("analytical")
gapped = drc.gapped_trap(trap, 5.0)
print("\n".join(drc.summary(drc.check_trap(gapped))))
solution = bem.solve(gapped)
for name, solved in (("Gapless plane", gapless), ("Boundary elements with gaps", gapped)):
    null = solved.find_rf_null((0.0, 0.0, None))
    omega, _ = solved.normal_modes(ion, null)
    print(f"{name}: RF null at {null[2]:.2f} µm, radial frequencies " + ", ".join(f"{w / (2 * np.pi * 1e6):.3f}" for w in omega[1:]) + " MHz")
print(f"({solution.mesh.n_panels} panels)")

# Waveguides from the edge of the chip to every grating, and a tree of couplers for the two gate zones
stack = components.Stack()
routes, crossings = photonic_circuits.route_trap(trap)
for route in routes:
    print(f"Grating {route.grating} ({trap.grating_couplers()[route.grating].beam}): {route.length:.0f} µm, {route.loss(stack, 0.6, 0.15, 20.0):.2f} dB")
coupler = photonic_circuits.analyze_coupler(stack, 0.6, 0.15, 0.3)
tree = photonic_circuits.splitter_tree([1, 1], coupler=coupler)
print(f"Coupler length for an even split: {tree.splits[0][2]:.1f} µm, outputs " + ", ".join(f"{p:.3f}" for p in tree.outputs))

# A five-qubit GHZ circuit on linear QCCD architectures with traps of every capacity
circuit = circuits.parse_qasm(circuits.PRESETS["GHZ (5 qubits)"])
rows = qccd.explore(circuit, {"L4": lambda capacity: qccd.Architecture.linear(4, capacity)}, capacities=(2, 3, 4, 6), swap_types=("gate",))

figure, (ax_trap, ax_qccd) = plt.subplots(1, 2, figsize=(13, 5))
for e, polygon in enumerate(gapped.electrode_polygons()):
    color = "tab:orange" if gapped.is_rf(e) else "tab:blue"
    ax_trap.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.3, edgecolor=color, linewidth=0.5)
for route in routes:
    ax_trap.plot(route.points[:, 0], route.points[:, 1], color="black", linewidth=0.8)
ax_trap.set_aspect("equal")
ax_trap.set_title(f"Gapped trap with {len(routes)} waveguide routes ({crossings} crossings)")
for model in qccd.GATE_MODELS:
    subset = [row for row in rows if row["gate_model"] == model and "fidelity" in row]
    ax_qccd.plot([row["capacity"] for row in subset], [row["fidelity"] for row in subset], marker="o", label=model)
ax_qccd.set_xlabel("Ions per trap")
ax_qccd.set_ylabel("Estimated fidelity of the GHZ circuit")
ax_qccd.legend()
save(figure, "12_fabrication_and_architectures.png")
