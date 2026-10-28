# tiopt

tiopt is a C/Fortran-based program which simulates ion traps and ion transport, with the goal of optimizing trap design, transport waveforms, and ion control. It aims to enable all of the following in one tool:
- Construct a seed trap design from presets, consisting of electrodes and (optionally) integrated photonics and electronics
- Simulate trap components (e.g. electrodes, waveguide unit cells)
- Represent electric potentials of electrodes using spherical harmonic expansions
- Obtain curve fits of parameters of photonic components
- Simulate ion transport via robust symplectic integration
- Simulate ion-photon interactions during laser pulses
- Analyze various aspects of ion control based on experiment data
- Optimize all aspects of trap geometry, transport waveforms, and ion addressing based on analysis
- Co-design the trap, its photonics, transport programs, and waveforms based on how well the trap runs quantum circuits

tiopt also comes with a Python interface and a GUI which cover every part of the library, and it can run in parallel on every core of your CPU or move its heaviest computations onto a GPU. The `examples` directory has example traps (including a trap with all of the main photonic components), circuits, Python scripts, and a C program which can be used to get started with tiopt - see the [examples README](examples/README.md) for more.

# Documentation

Documentation is hosted online on ReadtheDocs at [https://tiopt.readthedocs.io](https://tiopt.readthedocs.io). It is generated from the code with Doxygen, and it can be built locally by running the command
```
doxygen Doxyfile
```
which writes the documentation to `docs/html`.

# Installation

## Unix-based systems

tiopt depends on the following packages, which can be installed with the package manager of your distribution (the names below are for Ubuntu):
- git
- gcc and gfortran
- make
- libgsl-dev
- libfftw3-dev
- libnlopt-dev
- libblas-dev, liblapack-dev, and libopenblas-dev

The Python interface and GUI also need Python 3 with numpy, matplotlib, and tkinter (and scipy, pytest, and Xvfb to run all of the tests). GPUs only need an OpenCL driver at runtime (see [Performance and acceleration](#performance-and-acceleration)), so nothing extra is needed to build tiopt.

tiopt uses SHTOOLS for its spherical harmonic expansions through a small wrapper, so the first step is to make a new folder and clone the following repositories into it:
```
git clone https://github.com/SHTOOLS/SHTOOLS.git
git clone https://github.com/RasmitDevkota/shtools_wrapper.git
git clone https://github.com/RasmitDevkota/tiopt.git
```

SHTOOLS can then be built and installed by running the following commands inside the `SHTOOLS` directory:
```
make fortran
sudo make install
```

Before building the wrapper, the following lines in the makefile of `shtools_wrapper`
```
SHTOOLS_SRC = /home/rasmitdevkota/projects/SHTOOLS/src
SHTOOLS_LIB = /home/rasmitdevkota/projects/SHTOOLS
```
need to be replaced with the directories which SHTOOLS was installed to:
```
SHTOOLS_SRC = /usr/local/include
SHTOOLS_LIB = /usr/local/lib
```

The wrapper can then be built by running `sudo make` inside the `shtools_wrapper` directory. tiopt works whether the wrapper calls `SHExpandDH` with the 4π (`norm = 1`) or orthonormal (`norm = 4`) normalization, since it checks which one the installed library uses.

Finally, tiopt can be built by running the following commands inside the `tiopt` directory:
```
make clean
make
```
which produce the `tiopt` executable and the `libtiopt.so` shared library (which the Python interface uses).

# Usage

tiopt can be started by running
```
./tiopt
```
which offers a menu of demos: transport schedules, the electrodynamics of `trap.data`, transport waveforms and dynamics, trap optimization, grating coupler design, quantum gates, and the co-design flow.

## GUI

The GUI can be started by running `make gui` (or `PYTHONPATH=python python3 -m tiopt`). It has a tab for each part of tiopt, and computations run in the background while the output of the C library shows up in the log at the bottom. The toolbar at the top sets the number of threads and the device (CPU or OpenCL) which tiopt runs on.
- Trap - generate the linear surface trap preset (optionally with a grating coupler, or with the full 40Ca+ photonics of four grating couplers and a detector), load or save trap files, set the RF drive, and choose the ion species
- Electrostatics - solve the electrodynamics with the analytical, spectral, or relaxation solver, and map the pseudopotential, the potential of any electrode, or the potential energy of a static well in any plane
- Waveforms - solve for a static well or a transport, and plot the electrode voltages and the moving potential wells
- Dynamics - simulate the ion through the current waveform (with the pseudopotential or the full RF drive), and compute its final motional excitation and heating rates
- Photonics - solve waveguide modes, design a grating coupler which focuses on the ion, and compute the beam, Rabi frequency, Lamb-Dicke parameter, and directionality
- Quantum - edit a sequence of laser pulses (or load one of the presets for Rabi flopping, sidebands, Ramsey, and a Mølmer-Sørensen gate) and simulate the qubits and motional mode
- Scheduling - parse an OpenQASM circuit, schedule the transports of its ions between trap zones, and build the waveform which carries them out
- Co-design - run the [co-design flow](#co-design) for a circuit, watch the fidelity of the circuit and its operations improve stage by stage, and load the co-designed trap into the other tabs

## Python

tiopt can also be used from Python through the `tiopt` package in the `python` directory, which wraps the C library with ctypes. For example:
```python
import sys; sys.path.insert(0, "python")
import tiopt

ion = tiopt.Ion("40Ca+")
trap = tiopt.Trap.from_preset(n_segments=7, segment_width=50.0, gratings=True)
trap.solve("analytical")

start = trap.find_rf_null((-50.0, 0.0, None))
end = trap.find_rf_null((50.0, 0.0, None))
waveform = tiopt.Waveform.transport(trap, ion, start, end, duration=50e-6, n_steps=501)
trajectory = tiopt.simulate_trajectory(trap, ion, waveform, 55e-6, 5e-9, start)
print(tiopt.motional_quanta(ion, tiopt.AXIAL_FREQUENCY, trajectory.positions[-1], trajectory.velocities[-1], end))

grating = trap.grating_couplers()[0]
intensity, k_vector = grating.beam_at(end)

result = tiopt.simulate_pulses(ion, [tiopt.Pulse(0.0, 20e-6, 0.0, 2 * 3.14159 * 100e3)], n_ions=1)

# Co-design a trap for a circuit, with a pass of local objectives and a QC-aware pass
codesign = tiopt.Codesign("examples/circuits/ghz3.qasm")
codesign.run(n_passes=2, max_evaluations=10)
print(codesign.metrics["circuit_fidelity"], codesign.parameters)

# Run on 8 threads, and on the GPU if there is one
tiopt.set_num_threads(8)
if tiopt.opencl_available():
    tiopt.set_device("opencl")
```

The structures in the bindings (`python/tiopt/lib.py`) are checked against the compiled library whenever the package is imported, so the two can't silently get out of sync.

# Conventions

- Lengths in the trap frame are in micrometers. The origin of the trap frame is the center of the electrode surface (which is also the center of the bottom face of the relaxation grid), the trap axis is x, and z is the height above the surface.
- Electrode vertices are relative to the position of the electrode. Electrodes `[0, n_electrodes_rf)` are RF electrodes, which are all driven in phase with amplitude `V_RF` at `Omega_RF`, and the rest are DC electrodes.
- Times are in seconds, frequencies are in rad/s, masses are in kg, charges are in C, and energies are in J. The exception is `struct IonParameters`, which gives masses in atomic mass units and charges in units of the elementary charge.
- The potential of each electrode is expanded in regular solid harmonics within each sphere of a lattice (see `defs.h`), from which potentials, gradients, and Hessians are evaluated exactly. The expansions are most accurate around the height of the ions (to about 1E-5 relative error for the default lattice) and lose accuracy below about 36 µm, where the lowest layer of spheres gets close to the electrodes.
- The relaxation and spectral solvers solve Laplace's equation on a grid inside a grounded box whose lid is `RELAXATION_NZ - 1` µm above the surface (74 µm by default). Because of this, their potentials fall below those of the analytical (gapless plane) solver towards the lid, by about half at the default ion height. The spectral solver solves the same discrete problem as relaxation, but exactly and with fast sine transforms, for electrodes in the trap surface.
- Waveforms hold the voltages of every electrode at each step (RF electrodes carry no DC voltage), and they are interpolated linearly in time. Steps should be much shorter than the axial period, since stepping the voltages near the axial period excites the ion.

# Trap files

Trap files (e.g. `trap.data`) contain one parameter per line, and lines starting with `#` are ignored:
```
trap n_electrodes 2
trap n_electrodes_rf 0
trap n_electrodes_dc 2
trap V_RF 40
trap Omega_RF 2.5132741228718345e8
trap electrode_positions 0 (25 0 0)
electrode n_vertices 0 4
electrode n_edges 0 4
electrode v 0 0 (10 10 0)
electrode e 0 0 (0 1)
trap n_grating_couplers 1
grating position 0 (-41.25 0 0)
grating period 0 0.75
```

Grating couplers also take `azimuth`, `fill_factor`, `etch_depth`, `length`, `width`, `focal_distance`, `power`, `efficiency`, `wavelength`, `waveguide_width`, `waveguide_thickness`, `n_core`, and `n_cladding`. Detectors are given by `trap n_detectors` followed by `detector position`, `width`, `length`, `wavelength`, `quantum_efficiency`, and `dark_count_rate`. `write_trap_to_file` writes every parameter, and complete examples can be found in `examples/traps`.

# Modules

- `electrodynamics.c` - relaxation, spectral, and analytical (gapless plane) solvers, Driscoll-Healy sampling, spherical harmonic expansions, potentials, the pseudopotential, normal modes, and the RF null
- `traps.c`, `io.c`, and `ions.c` - trap presets (with integrated photonics for 40Ca+), allocation of traps, trap files, and ion species
- `waveforms.c` - well constraints, the waveform solver, NLopt refinement, transport waveforms, and waveforms for transport programs
- `verlet.c` - velocity Verlet integration of ion trajectories and of ensembles of ions
- `analysis.c` and `ion_heating` - heating rates from RF and DC voltage noise and from field noise spectra, motional excitation, and trap depth
- `experiments.c` - idle and back-and-forth transport experiments, which the optimization loop in `main.c` uses
- `photonics.c` - slab and channel waveguide modes, grating couplers, Gaussian beams, multilayer stacks, integrated detectors (collection efficiency, scattering, and detection fidelity), and curve fits
- `quantum.c` - the Lindblad master equation for ions sharing a motional mode during laser pulses, integrated with an adaptive Dormand-Prince method
- `transport.c` and `circuits.c` - OpenQASM parsing, circuit layers, and transport scheduling
- `codesign.c` - the co-design flow, including its stages, objectives, and quantum computing metrics
- `acceleration.c` and `kernels.cl` - threads and the OpenCL backend (which is loaded at runtime) with its kernels
- `bindings.c` - helpers for the Python bindings

Note that the transport programs from `generate_transport_schedule` record each transport into the zone of an operation, but not the transports of the ions back into storage afterwards.

# Co-design

The co-design flow (`codesign.c`) designs a linear trap with integrated photonics for the circuit in a QASM file. It goes through the following stages:

0. Start with basic designs of everything, i.e. the trap, the photonics (grating couplers focused on the gate and measurement zones, and a detector), the transport program which runs the circuit on a line of zones, and the waveform of the shuttle primitive between neighboring zones
1. Optimize the trap a little (segment width and RF drive)
2. Optimize the photonics a little (size of the qubit grating, fraction of the available Rabi frequency used for gates, detector size, and detection time)
3. Optimize the transport program a little (zone spacing and shuttle duration)
4. Optimize the waveforms a little (axial frequency and roughness weight)
5. Compute all of the QD/QC metrics, i.e. secular frequencies, Mathieu q, trap depth, and heating rate; Rabi frequency, Lamb-Dicke parameter, crosstalk, and detection fidelity; shuttle excitation and voltages; gate times and fidelities (with the Mølmer-Sørensen gate simulated by the master equation, including heating and dephasing); and the duration and estimated success probability of the whole circuit

The first pass optimizes each stage based on its own figures of merit. Every later pass repeats stages 1-5 using QC-aware optimization, where every stage minimizes -log(circuit fidelity) (plus penalties for unstable or shallow traps and for voltages beyond the limit), so that the trap is good for actually running circuits rather than just for its substeps. Each stage only takes a few evaluations of NLopt's COBYLA, and stages only redo the work which their parameters affect (e.g. the electrodynamics are only solved again when the geometry of the trap changes), so two passes take a few seconds.

The hardware and noise which the design assumes (`struct CodesignEnvironment`, which holds the voltage limit, laser powers, field and voltage noise, dephasing, cooling, drift, and detector dark counts) can be changed from C, Python, or the GUI. More details can be found in the [examples README](examples/README.md), and `examples/python/07_codesign.py` is a good place to try it out.

# Performance and acceleration

Every expensive part of tiopt runs in parallel with OpenMP, including the solvers, the spherical harmonic expansions, potential maps, waveform constraints, ensembles of ions, and the master equation. The number of threads can be set with `OMP_NUM_THREADS`, `tiopt.set_num_threads`, or the toolbar of the GUI (by default, tiopt uses one thread per core). `make benchmark` times the main computations on one thread, on every thread, and on the OpenCL device, which gives the following on a 16-thread laptop CPU (i7-1360P):

| Computation | 1 thread | 16 threads |
| --- | --- | --- |
| Analytical solve (linear trap, 17 electrodes) | 0.54 s | 0.061 s |
| Grid solve of one electrode (spectral) | 2.2 s | 0.28 s |
| Potential energy map (40000 points) | 0.034 s | 0.0043 s |
| Transport waveform (501 steps) | 0.013 s | 0.0045 s |
| Trajectory (10000 steps) | 0.025 s | 0.025 s |
| Ensemble (1024 ions, 1000 steps) | 2.6 s | 0.33 s |
| Mølmer-Sørensen gate (2 ions, dimension 52) | 0.044 s | 0.052 s |
| Mølmer-Sørensen gate with heating (3 ions, dimension 208) | 0.89 s | 0.23 s |

Most of the speedup on one thread comes from the algorithms themselves - the spectral solver solves the grid problem exactly with fast sine transforms instead of iterating (relaxation takes about 25 s to converge for the same electrode), the solid harmonics are evaluated for every electrode at once, the Driscoll-Healy expansion matrix is cached, and the master equation is integrated with an adaptive Dormand-Prince method which only computes the nonzero part of the Hamiltonian.

## GPUs

Relaxation, potential maps (`evaluate_trap_potential_batch`, `evaluate_potential_energy_batch`, and `Trap.potential` and `Trap.potential_energy` in Python), and ensembles (`simulate_ensemble`) can run on a GPU through OpenCL in double precision. tiopt only loads `libOpenCL.so.1` once a GPU is requested, so it builds and runs without any GPU software, and everything falls back to the CPU if no device is found. The GPU can be selected with `set_compute_device(OPENCL_DEVICE)` in C, `tiopt.set_device("opencl")` in Python, the toolbar of the GUI, or `TIOPT_DEVICE=opencl ./tiopt`.

For NVIDIA GPUs, the proprietary driver includes an OpenCL platform (`nvidia-opencl-icd` or the equivalent for your distribution, along with `ocl-icd`). For AMD GPUs (e.g. the Radeon RX 5700 XT), an OpenCL runtime such as the one from ROCm needs to be installed, but the 5700 XT (gfx1010) is not officially supported by ROCm, so it's worth checking that the runtime finds it. Either way, `clinfo` should list the device with `cl_khr_fp64`. If there are several devices, `TIOPT_OPENCL_DEVICE=n` picks the n-th device with double precision, and `TIOPT_OPENCL_ANY_DEVICE=1` also allows CPU devices (such as PoCL, which the tests use to check the kernels without a GPU).

Consumer GPUs run double precision at 1/16 (AMD RDNA) to 1/64 (NVIDIA GeForce) of their single-precision rate, so they pay off for large maps and ensembles (thousands of points or ions) rather than for single trajectories. Data center GPUs (e.g. the NVIDIA A100 or H100) run double precision at half rate. The OpenCL results agree with the CPU up to rounding error.

# Tests

The test suite can be run with
```
make test
```
which builds and runs the C tests (`tests/tiopt-test`) and then the Python tests (`python3 -m pytest tests`). The tests check the library against analytical results and independent calculations. The GUI tests use a virtual display from Xvfb when it is available, and the OpenCL tests only run when an OpenCL device with double precision is available.
