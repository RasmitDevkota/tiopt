"""Acceleration example, which times potential maps and ensembles with different numbers of threads and on an OpenCL
device if one with double precision is available (TIOPT_OPENCL_DEVICE can be set to choose among several, or
TIOPT_OPENCL_ANY_DEVICE=1 to allow CPU devices).
"""

import time

import numpy as np
from _common import EXAMPLES

import tiopt

ion = tiopt.Ion("40Ca+")
trap = tiopt.Trap.from_file(EXAMPLES / "traps" / "linear_trap.trap")
trap.solve("analytical")

x, z = np.meshgrid(np.linspace(-150, 150, 300), np.linspace(40, 70, 200))
grid = np.stack([x, np.zeros_like(x), z], axis=-1)
null = trap.find_rf_null((0.0, 0.0, None))
rng = np.random.default_rng(0)
positions = null + 0.3 * rng.standard_normal((2048, 3))
velocities = 1e4 * rng.standard_normal((2048, 3))


def benchmark():
    start = time.perf_counter()
    trap.potential_energy(ion, grid)
    map_time = time.perf_counter() - start
    start = time.perf_counter()
    tiopt.simulate_ensemble(trap, ion, None, positions, velocities, 500, 5e-9)
    ensemble_time = time.perf_counter() - start
    return map_time, ensemble_time


n_threads = tiopt.get_num_threads()
print(f"{'Configuration':<30} {'60000-point map':>16} {'2048-ion ensemble':>18}")
for threads in sorted({1, max(n_threads // 2, 1), n_threads}):
    tiopt.set_num_threads(threads)
    map_time, ensemble_time = benchmark()
    print(f"{f'CPU, {threads} threads':<30} {map_time:>15.3f}s {ensemble_time:>17.3f}s")

if tiopt.opencl_available():
    tiopt.set_device("opencl")
    map_time, ensemble_time = benchmark()
    print(f"{'OpenCL: ' + tiopt.opencl_device_name()[:22]:<30} {map_time:>15.3f}s {ensemble_time:>17.3f}s")
    tiopt.set_device("cpu")
else:
    print("No OpenCL device with double precision was found (see the README for setting up GPUs)")
