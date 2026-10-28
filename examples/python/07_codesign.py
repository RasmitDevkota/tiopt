"""Co-design example, which starts from a basic design of the trap, photonics, transport program, and waveforms for a
circuit, optimizes each of them a little on its own, computes the quantum computing metrics of the whole design, and
then repeats with every stage optimizing those metrics, so that the trap is good for actually running circuits rather
than just for its own figures of merit. A QASM file and the numbers of passes and evaluations per stage can be passed to
change the defaults.
"""

import argparse
import threading
import time

import numpy as np
from _common import EXAMPLES, OUTPUT, save
from matplotlib.figure import Figure

import tiopt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("qasm", nargs="?", default=EXAMPLES / "circuits" / "ghz3.qasm")
parser.add_argument("--passes", type=int, default=3)
parser.add_argument("--evaluations", type=int, default=10)
parser.add_argument("--field-noise", type=float, default=1e-12, help="electric field noise at 1 MHz (V^2/m^2/Hz)")
arguments = parser.parse_args()

codesign = tiopt.Codesign(arguments.qasm, environment={"field_noise": arguments.field_noise})
print(f"Basic design: circuit fidelity {codesign.metrics['circuit_fidelity']:.6f}")

# Run the flow on another thread and report every stage as it finishes
start = time.perf_counter()
worker = threading.Thread(target=codesign.run, args=(arguments.passes, arguments.evaluations))
worker.start()
seen = codesign.n_records
while worker.is_alive() or codesign.n_records > seen:
    for record in codesign.records(seen):
        print(f">>> pass {record.pass_index}, {record.stage}: circuit fidelity {record.metrics['circuit_fidelity']:.6f}, duration {record.metrics['circuit_duration'] * 1e3:.3f} ms")
        seen += 1
    time.sleep(0.2)
worker.join()
print(f"Finished in {time.perf_counter() - start:.1f} s")

records = codesign.records()
metrics = codesign.metrics
parameters = codesign.parameters
print("Final design:")
print(f"  RF {parameters['V_RF']:.1f} V at {parameters['Omega_RF'] / (2 * np.pi * 1e6):.1f} MHz, segments {parameters['segment_width']:.1f} µm wide")
print(f"  Axial frequency {parameters['omega_axial'] / (2 * np.pi * 1e6):.3f} MHz, zones {parameters['zone_spacing']:.1f} µm apart, shuttles {parameters['shuttle_duration'] * 1e6:.2f} µs")
print(f"  Two-qubit gates {metrics['two_qubit_time'] * 1e6:.1f} µs with fidelity {metrics['two_qubit_fidelity']:.5f}; detection fidelity {metrics['detection_fidelity']:.6f}")
print(f"  Circuit fidelity {metrics['circuit_fidelity']:.6f} over {metrics['circuit_duration'] * 1e3:.3f} ms")

OUTPUT.mkdir(exist_ok=True)
codesign.save_trap(OUTPUT / "codesigned.trap")
print(f"Saved the co-designed trap to {(OUTPUT / 'codesigned.trap').relative_to(EXAMPLES.parent)}")

figure = Figure(figsize=(10, 4), layout="constrained")
ax = figure.add_subplot(1, 2, 1)
steps = np.arange(len(records))
ax.semilogy(steps, [1 - r.metrics["circuit_fidelity"] for r in records], marker="o")
for r, record in enumerate(records):
    if record.pass_index > 0:
        ax.axvspan(r - 0.5, r + 0.5, color="tab:green", alpha=0.08, linewidth=0)
ax.set_xticks(steps)
ax.set_xticklabels([f"{r.pass_index}:{r.stage}" for r in records], rotation=60, ha="right", fontsize=7)
ax.set_ylabel("Circuit infidelity")
ax.set_title("Co-design history (QC-aware passes shaded)")
ax = figure.add_subplot(1, 2, 2)
for key, label in (("two_qubit_fidelity", "Two-qubit gate"), ("single_qubit_fidelity", "Single-qubit gate"), ("detection_fidelity", "Detection")):
    ax.semilogy(steps, [1 - r.metrics[key] for r in records], marker="o", label=label)
ax.set_xticks(steps)
ax.set_xticklabels([f"{r.pass_index}:{r.stage}" for r in records], rotation=60, ha="right", fontsize=7)
ax.set_ylabel("Infidelity")
ax.legend()
save(figure, "07_codesign.png")
