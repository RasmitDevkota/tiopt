# tiopt examples

This directory has example traps, circuits, and scripts which can be used to try out tiopt. The library needs to be built first by running `make` in the root of the repository, and the Python scripts save their figures to `examples/output`.

# Traps

The `traps` directory has the following traps:
- `linear_trap_photonics.trap` - a five-wire linear surface trap for 40Ca+ (with seven 50 µm DC segments on each side and the RF null 55 µm above the surface) with integrated photonics, i.e. grating couplers which deliver 729 nm (qubit) and 854 nm (quench) light to the gate zone at the center of the trap and 397 nm (detection) and 866 nm (repump) light to the measurement zone at x = 100 µm, and a 50 µm detector beneath the measurement zone
- `linear_trap.trap` - the same trap without any photonics
- `long_linear_trap_photonics.trap` - a longer trap (with fifteen 40 µm segments on each side) for transporting ions between several zones, with the same photonics
- `qccd_trap.trap` - a linear trap for the zone layout `load, storage x2, gate x5, measure` with zones 100 µm apart, i.e. a loading zone with photoionization and cooling beams, two storage zones, five gate zones with qubit and quench beams, and a measurement zone with detection and repump beams and a detector (its zones span 800 µm, so wells can only be made in all of them with a build made by running `make clean && make DEFINES="-DNSPH_X=31"`)
- `x_junction.trap`, `y_junction.trap`, and `t_junction.trap` - the junction presets, with two 40 µm segments on each side of every arm and a junction zone at the center
- `point_trap.trap` - the point trap preset, which holds one ion above a ring of RF, with photonics for gates and detection of the ion
- `two_squares.trap` - two 20 µm square electrodes, which is the smallest example of the trap file format

Trap files can be loaded with `Trap.from_file` in Python, `generate_trap_from_file` in C, or the Trap tab of the GUI, and they can be written with `Trap.save` or `write_trap_to_file`. Lines starting with `#` are comments. Every trap can also be exported to a GDSII layout with `Trap.save_gds` (or the Trap tab of the GUI) for layout tools such as KLayout.

# Circuits

The `circuits` directory has the following OpenQASM circuits, which act on a register named `qreg` so that the scheduler and the co-design flow can read them:
- `bell.qasm` - a Bell pair, i.e. a Hadamard and a CNOT followed by measurements of both qubits
- `ghz3.qasm` - a three-qubit GHZ state
- `ghz5.qasm` - a five-qubit GHZ state, which needs more shuttling
- `four_qubit_mix.qasm` - single-qubit gates and controlled Pauli gates on four qubits (which is also the circuit used by the test suite)

# Python scripts

The scripts in the `python` directory can be run from any directory, e.g. `python3 examples/python/01_quickstart.py`.
- `01_quickstart.py` - load the trap with photonics and find its RF null, secular frequencies, and depth, a 1 MHz well, the light from every grating, the Rabi frequency and Lamb-Dicke parameter of the qubit, and the fidelity of detection with the integrated detector
- `02_electrostatics.py` - compare the analytical (gapless plane) and spectral (grounded box) solvers, and map the pseudopotential
- `03_transport.py` - shuttle an ion between zones, count the quanta which a fast and a slow shuttle excite, and run a thermal ensemble of 256 ions through the shuttle in parallel
- `04_photonics.py` - look at the focused 729 nm beam, and at how the detector size and detection time trade collection against dark counts and decay of the dark state
- `05_quantum.py` - simulate Rabi oscillations on the carrier and sideband, and a Mølmer-Sørensen gate with motional heating
- `06_scheduling.py` - schedule the transports of a GHZ circuit on the zones of the long trap, and find the waveform of every shuttle
- `07_codesign.py` - run the co-design flow (see below) for a circuit and report each stage as it finishes (the `--passes`, `--evaluations`, and `--field-noise` options are worth playing with)
- `08_acceleration.py` - time potential maps and ensembles with different numbers of threads and on an OpenCL device
- `09_trap_design.py` - design a linear trap for a zone layout with photonics for every zone, evaluate every zone of the design, list the beams of each species, and export the design to trap, GDSII, and SVG files (the `--layout`, `--zone-spacing`, and `--species` options can be used to try other designs)
- `10_junctions.py` - generate the X, Y, and T junction presets and a five-way junction, and find the pseudopotential barrier an ion crosses turning from one arm into another

# C

`basic_trap.c` in the `c` directory uses the library from C. It loads the trap with photonics, makes a well, and evaluates the qubit beam, and then it co-designs a trap for the Bell circuit. It can be built and run by running `make run` inside the `c` directory.

# The co-design flow

The co-design flow (`codesign.c`, `tiopt.Codesign`, the Co-design tab of the GUI, and option 8 of `./tiopt`) designs a linear trap with integrated photonics for a circuit on the zones of a zone layout through the following stages:

0. Basic design of everything, i.e. the trap, the photonics focused on every zone which needs them, the transport program which runs the circuit on the zones (by default `load+storage, empty, gate, empty, measure`, i.e. load/storage, gate, and measurement zones separated by empty zones, but any layout with a gate zone can be given), and the waveform of the shuttle primitive into the first gate zone
1. Trap - segment width and RF drive
2. Photonics - size of the qubit grating, fraction of the available Rabi frequency used for gates, detector size, and detection time
3. Transport - zone spacing and shuttle duration
4. Waveforms - axial frequency and the roughness weight of the waveform solutions
5. Metrics - secular frequencies, Mathieu q, trap depth, and heating rate; Rabi frequency, Lamb-Dicke parameter, crosstalk, and detection fidelity; shuttle excitation and voltages; gate times and fidelities (where the Mølmer-Sørensen gate is simulated with the master equation, including heating and dephasing); and the duration and estimated success probability of the whole circuit

The first pass optimizes each of stages 1-4 a little based on its own figures of merit (e.g. a target radial frequency for the trap, or the errors of single operations for the photonics). Every later pass repeats stages 1-5 with -log(circuit fidelity) as the objective, plus penalties for designs which can't run (unstable or shallow traps, or voltages beyond the limit), so that every part is tuned for running the circuit rather than for its own figures of merit. The first pass can actually make the circuit worse - for the Bell circuit, optimizing the photonics for single operations lowers the fidelity of the circuit - which the QC-aware passes then correct!

Wells which are closer than several ion heights need large voltages to be held at the same time, so the zones are kept apart, and the voltages which hold every zone that the circuit occupies at the same time are checked against the limit. The spherical harmonic expansions of the default build only reach about 200 µm on either side of the trap center, so the flow brings the zones of longer layouts closer together (with a warning), and layouts with more zones need a build with more spheres, which can be made by running the command
```
make clean && make DEFINES="-DNSPH_X=32"
```
in the root of the repository (the warning gives the number of spheres a layout needs).
