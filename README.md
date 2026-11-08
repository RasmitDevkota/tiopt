# tiopt

tiopt is a C/Fortran-based program which simulates ion traps and ion transport, with the goal of optimizing trap design, transport waveforms, and ion control. It aims to enable all of the following in one tool:
- Construct a seed trap design from presets or zone layouts, consisting of electrodes and (optionally) integrated photonics and electronics
- Simulate trap components (e.g. electrodes, waveguide unit cells)
- Represent electric potentials of electrodes using spherical harmonic expansions
- Obtain curve fits of parameters of photonic components
- Simulate ion transport via robust symplectic integration
- Simulate ion-photon interactions during laser pulses
- Analyze various aspects of ion control based on experiment data
- Optimize all aspects of trap geometry, transport waveforms, and ion addressing based on analysis
- Simulate crystals of many ions (of one or more species), their normal modes, gates on all of their modes, and their laser cooling
- Compile circuits onto architectures of many traps and estimate how well they run
- Co-design the trap, its photonics, transport programs, and waveforms based on how well the trap runs quantum circuits

tiopt also comes with a Python interface and a GUI which cover every part of the library, and it can run in parallel on every core of your CPU or move its heaviest computations onto a GPU. Trap designs can be exported to GDSII files for layout tools such as KLayout and imported back from them, so teams can go back and forth between tiopt and their usual layout workflow. The `examples` directory has example traps (including a trap with all of the main photonic components), circuits, Python scripts, and a C program which can be used to get started with tiopt - see the [examples README](examples/README.md) for more.

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
which produce the `build/tiopt` executable and the `build/libtiopt.so` shared library (which the Python interface uses). The repository is laid out as follows:

- `src` - the C sources of the library and executable, and the OpenCL kernels
- `include` - the C headers, which programs using the library should add to their include path
- `build` - object files, the executable, and the shared library (made by `make` and removed by `make clean`)
- `python` - the Python bindings and GUI
- `tests`, `examples`, `demos`, and `docs` - the test suites, examples, notebooks, and documentation

# Usage

tiopt can be started by running
```
./build/tiopt
```
which offers a menu of demos: transport schedules, the electrodynamics of `trap.data`, transport waveforms and dynamics, trap optimization, grating coupler design, quantum gates, the co-design flow, and the design of a trap for a zone layout and of a junction.

## GUI

The GUI can be started by running `make gui` (or `PYTHONPATH=python python3 -m tiopt`). It has a tab for each part of tiopt, and computations run in the background while the output of the C library shows up in the log at the bottom. The toolbar at the top sets the number of threads and the device (CPU or OpenCL) which tiopt runs on.
- Trap - generate a trap from the preset library (linear, junction, and point traps, or a linear trap for a zone layout) with integrated photonics for every zone, load or save trap files, import and export GDSII and SVG layouts, export potentials to VTK files, evaluate every zone of the design, set the RF drive, choose the ion species, show the total confinement and the [CTC path](#constant-total-confinement) along a path through the trap, [optimize the shapes of the electrodes](#shaping-electrodes) around e.g. a junction or [their topology](#topology-optimization-of-rf-electrodes) on a grid of pixels, and [check the design rules and open gaps](#design-rules-and-gaps) for fabrication
- Trap builder - build a trap by dragging parts (linear and curved sections, junctions, point traps, single electrodes, zones, grating couplers, and detectors) onto the trap surface, move, rotate, and edit them, and build them into the trap which every other tab uses (see [Building traps from parts](#building-traps-from-parts))
- Electrostatics - solve the electrodynamics with the analytical, spectral, relaxation, or [boundary element](#boundary-element-method) solver, map the pseudopotential, the potential of any electrode, or the potential energy of a static well in any plane, and [analyze a trapping site](#trapping-sites) (its Floquet stability and stability diagram, escape point and depth, anharmonicity, micromotion, multipoles, and the control voltages of every multipole)
- Waveforms - solve for a static well or a transport (with the C solver or from [objectives and constraints](#voltages-from-objectives) such as voltage and slew rate limits and a tilt of the radial axes), plot the electrode voltages and the moving potential wells, and send the waveform through a [DAC and filters](#dacs-and-filters) with pre-compensation
- Crystals - find the equilibrium of a [Coulomb crystal](#coulomb-crystals) of any ions in a well of the trap or in a model trap, with its normal modes, the participation of every ion, and the Lamb-Dicke parameters of a beam
- Dynamics - simulate the ion through the current waveform (with the pseudopotential or the full RF drive), compute its final motional excitation and heating rates, scan the phase of the RF drive, and run the [molecular dynamics](#molecular-dynamics) of a laser-cooled crystal
- Photonics - design a grating coupler which focuses on the ion and compute its beam, Rabi frequency, Lamb-Dicke parameter, and directionality, or pick a component (a straight waveguide, a waveguide bend, or a grating coupler) to simulate and optimize, i.e. the width and thickness of a waveguide for the lowest loss per length, the radius, width, Euler taper, straight tapers, and offset of a bend, or the period, tooth width, number of teeth, etch depth, angular width, and apodization of a uniform or non-uniform grating for the most light at the ion (see [Photonic components](#photonic-components)), as well as directional couplers, trees of splitters which share one input among the gratings of several zones, and the routes of waveguides from the edge of the chip to every grating (see [Photonic circuits](#photonic-circuits))
- Quantum - edit a sequence of laser pulses with [shaped envelopes](#pulse-shapes) (or load one of the presets for Rabi flopping, sidebands, Ramsey, and a Mølmer-Sørensen gate with any pulse shape) and simulate the qubits and motional mode, edit an OpenQASM [circuit](#circuits) and simulate it with and without gate errors, design and simulate [gates on several modes](#gates-on-several-modes), show the [Zeeman lines](#zeeman-structure) which a beam drives, and [fit measured flopping](#fits-to-measured-data)
- Scheduling - schedule the transports of the circuit in the Quantum tab (or in an OpenQASM file) between the zones of a zone layout (or of the trap) with a [transport priority](#transport-priority), compare the priorities, build the waveform which carries them out, or compile the circuit onto a [QCCD architecture](#qccd-architectures) of many traps and compare trap capacities and gate models
- Co-design - run the [full co-design](#full-co-design) of the trap, photonics, pulses, transport, and wells together from the inputs of every tab, or the [co-design flow](#co-design-flow-of-the-c-library) of the C library for a circuit on a zone layout, watch the fidelity of the circuit improve, and send the co-designed system back to the other tabs

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

# Design a trap with a loading zone, a storage zone and two gate zones, with photonics for every zone
design = tiopt.Trap.from_layout("load, storage, gate x2, measure", zone_spacing=80.0, element="40Ca+")
design.solve("analytical")
print(tiopt.format_design_report(design.design_report(ion)))
design.save_gds("design.gds")

# Co-design a trap for a circuit, with a pass of local objectives and a QC-aware pass
codesign = tiopt.Codesign("examples/circuits/ghz3.qasm", layout="load+storage, empty, gate, empty, measure")
codesign.run(n_passes=2, max_evaluations=10)
print(codesign.metrics["circuit_fidelity"], codesign.parameters)

# Run on 8 threads, and on the GPU if there is one
tiopt.set_num_threads(8)
if tiopt.opencl_available():
    tiopt.set_device("opencl")
```

The structures in the bindings (`python/tiopt/lib.py`) are checked against the compiled library whenever the package is imported, so the two can't silently get out of sync. The rest of the package is written in Python:
- `export.py` - GDSII and SVG files
- `builder.py` - the trap builder
- `components.py` - the simulation and optimization of straight waveguides, bends, and grating couplers
- `confinement.py` - total confinement and CTC paths
- `shapes.py` - the optimization of the shapes of electrodes
- `circuits.py` - OpenQASM circuits and their simulation
- `pulses.py` - shaped pulses and Mølmer-Sørensen gates made with them
- `scheduling.py` - the timing of transports with each transport priority
- `system.py` - the full co-design
- `crystals.py` - Coulomb crystals and their normal modes
- `stability.py` - Floquet stability, escape points and depths, anharmonicity, and micromotion
- `multipoles.py` - multipole expansions and control voltages
- `voltages.py` - voltages from objectives and constraints
- `hardware.py` - DACs, filters, and pre-compensation
- `dynamics.py` - molecular dynamics of crystals
- `levels.py` - Zeeman and hyperfine structure, and the couplings of beams to it
- `multimode.py` - laser-ion dynamics with several motional modes
- `gates.py` - Mølmer-Sørensen gates for several modes
- `analytic.py` - analytic models of laser-ion interactions and fits to measured data
- `qccd.py` - compilation of circuits onto QCCD architectures
- `photonic_circuits.py` - S matrices of photonic circuits, splitter trees, and waveguide routes
- `drc.py` - design rule checks and gaps
- `bem.py` - the boundary element method
- `topology.py` - topology optimization of RF electrodes

# Designing traps

Traps can be generated from the preset library, from a zone layout, or from any set of polygons (including GDSII layouts drawn in other tools).

## Presets

The preset library has the following traps, which can be generated with `Trap.from_preset` in Python, `generate_trap_from_preset` in C, or the Trap tab of the GUI:
- `linear` - a five-wire linear surface trap, with RF rails on either side of a central DC electrode and segmented DC electrodes outside of the rails (optionally with a single grating coupler, or with photonics for a gate zone and a measurement zone)
- `x_junction`, `y_junction`, and `t_junction` - junctions where four arms meet at right angles, where three arms meet 120 degrees apart, or where a third arm meets a straight trap at right angles
- `point` - a ring of RF which holds a single ion above its center, surrounded by compensation electrodes

The RF nulls of every preset are at the height of the ions (`ION_HEIGHT`, which is 55 µm by default). Junctions with any number of arms at any angles (as long as neighboring arms are at most 180 degrees apart) can be generated with `Trap.junction` or `generate_junction_trap`, e.g. `tiopt.Trap.junction([0, 72, 144, 216, 288])` for a five-way junction.

## Zone layouts

Zones are where ions are loaded, stored, gated, measured, etc., and every trap can have zones which tiopt then uses for its photonics, scheduling, design reports, and the co-design flow. Zones can be described by a zone layout, which lists entries separated by commas, each with zone types joined by "+", an optional capacity after ":", and an optional count after "x". For example,
```
load, storage x2, gate x5, measure
```
describes a loading zone, two storage zones, five gate zones, and a measurement zone. The zone types are `load`, `storage`, `gate`, `measure`, `cool`, `empty`, `junction`, and `pre_junction`, and zones without a capacity hold two ions if they are gate or cooling zones, one ion if they are empty, and every ion otherwise.

A linear trap for a zone layout can be generated by running
```python
trap = tiopt.Trap.from_layout("load, storage x2, gate x5, measure", zone_spacing=100.0, element="40Ca+")
```
(or with `generate_trap_from_layout` in C, or with the "Linear from zone layout" preset of the GUI), which spaces the zones evenly along the trap axis and gives the trap enough segments and long enough rails for every zone. The zones of any trap can also be set by hand with `Trap.set_zones` (e.g. to put zones along the arms of a junction), and they are saved in trap files.

The same layouts are used for scheduling (`TrapGraph.from_layout`, or `TrapGraph.from_trap` for the zones of a trap) and by the co-design flow (the `layout` argument of `Codesign`). The scheduler sends each operation to the nearest zone of its type with room for its ions, so the operations of a layer run in parallel when there are several gate zones, and the ions go back into storage after each layer.

## Integrated photonics

`Trap.add_zone_photonics` (or `add_zone_photonics` in C) gives every zone of a trap the photonics its types need:
- Gate zones get the qubit and quench beams
- Measurement zones get the detection and repump beams, as well as a detector beneath the zone
- Cooling zones get the detection and repump beams
- Loading zones get both photoionization beams, along with the detection and repump beams to cool the ions they load

Each beam comes from a focusing grating coupler which is designed for the position of its zone, with a waveguide of silicon nitride for visible and infrared light or of alumina for ultraviolet light. The wavelengths depend on the species (see `tiopt.photonics_beam`):
- 40Ca+ - 729 nm qubit, 397 nm detection, 866 nm repump, 854 nm quench, and 423 nm and 375 nm photoionization
- 88Sr+ - 674 nm qubit, 422 nm detection, 1092 nm repump, 1033 nm quench, and 461 nm and 405 nm photoionization
- 171Yb+ - 411 nm qubit, 370 nm detection, 935 nm repump, 976 nm quench, and 399 nm and 370 nm photoionization
- 9Be+ - 313 nm Raman beams for the hyperfine qubit, 313 nm detection and repump, and 235 nm photoionization (9Be+ has no D states to quench)
- 43Ca+ - the same light as 40Ca+
- 25Mg+ - 280 nm Raman beams for the 1.79 GHz hyperfine qubit, 279.6 nm detection, 280.4 nm repump, and 285 nm photoionization (25Mg+ has no D states to quench either)
- 133Ba+ - 532 nm Raman beams for the 9.93 GHz hyperfine qubit, 494 nm detection, 650 nm repump, 614 nm quench, and 554 nm and 405 nm photoionization
- 137Ba+ and 138Ba+ - a 1762 nm qubit, and otherwise the same light as 133Ba+

Every grating coupler records the beam it delivers and the zone it addresses (`Grating.beam` and `Grating.zone`), and e.g. `trap.grating("qubit", zone=3)` finds the qubit grating of zone 3.

## Photonic components

`tiopt.components` simulates the integrated photonic components between the edge of the chip and the ion with its own mode solvers (finite differences with perfectly matched layers and the effective index method):
- Straight waveguides - the effective index, number of guided modes, confinement, and loss per length (from leakage into the substrate, sidewall and surface roughness, and absorption) of a waveguide of a given width and thickness, which `optimize_waveguide` minimizes while keeping the waveguide single-mode
- Bends - the loss of a bend from radiation, the transitions between its sections, and propagation, with Euler sections (whose curvature rises gradually), a wider bend, an offset of the straight waveguides, and straight tapers (linear, exponential, or parabolic) which widen the waveguide before the bend and narrow it again after it, where the light a taper scatters falls the longer it is
- Grating couplers - the emission, efficiency, and intensity at the ion of a focusing grating, whose teeth can be uniform or non-uniform, i.e. apodized (the fill factor changes from the first tooth to the last, which spreads the light more evenly over the grating) and chirped (the period of every tooth points its light at the ion, which focuses the light along the grating as well as across it)

Non-uniform gratings help quite a bit: with the default stack at 729 nm, an optimized grating which is apodized and chirped puts about 18 times as much light on the ion as an optimized uniform grating (about 7e7 W/m² per mW rather than 4e6 W/m² per mW), since it focuses the light to a spot of a few µm.

## Photonic circuits

`tiopt.photonic_circuits` follows the light from the edge of the chip to the gratings, like the circuit simulations and routing of gdsfactory and SAX:
- Directional couplers - `analyze_coupler` finds the even and odd supermodes of two waveguides side by side with the mode solver of `tiopt.components`, which give the length over which the light crosses from one to the other (e.g. 24 µm for 0.2 µm gaps between the default waveguides at 729 nm)
- Circuits - the S matrices of straight waveguides, bends, directional couplers, splitters, crossings, rings, phase shifters, and gratings are joined into a `Circuit`, whose S matrix between its external ports is found by solving for the light at every port
- Splitter trees - `splitter_tree` shares one input among several outputs with any split of the power (e.g. equal power at every gate zone), and gives the length of every directional coupler
- Routes - `route_trap` routes a waveguide from the nearest edge of the chip to every grating of a trap, with the length, bends, and loss of each and the number of places where routes cross

## Constant total confinement

The DC electrodes obey Laplace's equation, so they can only move confinement from one direction to another, and the total confinement (the sum of the squares of the three secular frequencies, which is proportional to the Laplacian of the pseudopotential) is set by the RF electrodes alone. Along a linear section it is the same everywhere on the RF null, but near the center of a junction the RF null rises and the total confinement drops, e.g. to about a third (an eighth of the sum of the squares) at the center of the X junction preset. A constant total confinement (CTC) path runs directly beneath the RF null wherever the confinement there is lower than in the arms, at the height where it is the same as in the arms, so that an ion which is moved along it is held just as strongly everywhere (at the cost of some micromotion, since the DC electrodes must hold it off the RF null).

`tiopt.confinement.confinement_path` follows the RF null along a path through waypoints, along with the CTC path beneath it, and gives the height, pseudopotential, and total confinement along both, as well as the DC field which holds the ion on the CTC path. The "Show confinement" button of the Trap tab draws them, along with a map of the total confinement, for a path through the zones of the trap or around the corner of a junction (or any path given as x, y points). Everything here comes from the gapless plane potentials of the electrodes directly, so it works beyond the spherical harmonic expansions too, e.g. around a whole racetrack.

## Shaping electrodes

`tiopt.shapes.optimize_shapes` (and the "Electrode shape optimization" form of the Trap tab) optimizes the shapes of the electrodes around a center, e.g. the RF electrodes near the center of a junction, rather than a few dimensions of rectangles. The trap surface inside a circular design region is deformed by a smooth displacement field (a sum of Gaussian bumps on a grid of control points), after the edges inside the region are divided into short pieces so that straight boundaries can bend. Since every electrode moves with the same field, the boundaries which electrodes share move together, so no gaps or overlaps open up, and the deformation can be made symmetric (e.g. under the symmetries of an X junction) and limited to a largest displacement. The objective is the pseudopotential barrier along the RF null of a path, the variation of the total confinement along it (so that the RF null itself comes closer to a CTC path), or both. For example, letting the electrodes of the X junction preset move by up to 15 µm with its 4-fold symmetry lowers its barrier from 7.6 meV to 6.4 meV and raises the lowest total confinement on the null from 0.34 to 0.39 of that in the arms.

Note that this is shape optimization rather than density-based topology optimization, so electrodes keep their topology (holes and islands are not created), which [topology optimization](#topology-optimization-of-rf-electrodes) does instead.

## Topology optimization of RF electrodes

`tiopt.topology.optimize_topology` (and the "Pixel topology optimization" form of the Trap tab) designs RF electrodes from scratch on a grid of pixels, like the pattern optimization of SurfacePattern and the electrode package, so that they can take any shape and topology. Every pixel of a square region is either RF or grounded, and since the RF potential is linear in the pixels (in the gapless plane approximation), asking for RF nulls along the arms of a layout (an X, T, or Y junction or a linear trap) is a set of linear equations in them, and the confinement at every null is a linear function of them. The design which pins every null and makes the weakest confinement as strong as possible is then a linear program over the fraction of each pixel which is RF, whose solutions are mostly 0 or 1, and the few fractional pixels are rounded. The pixels are tied together by the symmetry of the layout, and the RF pixels are traced into polygons (holes included). For example, a 300 µm X junction of 5 µm pixels holds every RF null within 0.5 µm of its height after rounding, and an X junction designed this way from 10 µm pixels has a pseudopotential barrier of under 1 meV, compared to 7.6 meV for the X junction preset.

## Design reports

`Trap.design_report` evaluates every zone of a solved trap, i.e. the RF null nearest to it and the pseudopotential there (which means the ions see micromotion if it is not zero), the secular frequencies of the pseudopotential, the Mathieu q parameter, the trap depth, the light from every beam addressing the zone (with the Rabi frequency, Lamb-Dicke parameter, and crosstalk of the qubit beam), and the fidelity of detection. `tiopt.format_design_report` turns the report into text, and the "Evaluate zones" button of the GUI shows it for the current trap.

For junctions, `Trap.rf_null_path` (or `compute_rf_null_path` in C) follows the RF null from one arm into another and finds the pseudopotential barrier an ion has to cross, which is the main figure of merit of a junction. The X junction preset has a barrier of about 7 meV, the T junction about 6 meV, and the Y junction about 2 meV.

## Trapping sites

`tiopt.stability` looks at a trapping site beyond the pseudopotential approximation (like the analysis of the electrode and Sion packages):
- Floquet stability - near a point, the full RF drive gives the Mathieu equation x'' + (A - 2 Q cos 2τ) x = 0 with 3x3 matrices A and Q, whose axes differ in e.g. asymmetric surface traps, so `floquet_at` integrates the full 3D equation over one RF period and finds whether the motion is stable and its secular frequencies (which are a few percent higher than those of the pseudopotential at q = 0.3), and `stability_diagram` maps the stable region as the RF amplitude and the DC voltages change
- Escape points - `escape_point` finds the true depth of a site from the lowest saddle point of the potential energy around it, along the path of least resistance out of a grid of points (the minimax path, which follows the minimum spanning tree of the grid) and then Newton's method, so e.g. the DC voltages of a well move the escape point off to the side of the trap (sites with only the pseudopotential are searched in the cross-section of the trap, since they hold the ion along the RF null only weakly)
- Anharmonicity - `anharmonicity` fits the cubic and quartic terms of the potential along an axis, and gives the lengths over which they grow as large as the harmonic term and the shift of the secular frequency per phonon
- Micromotion - `micromotion` gives the amplitude of the excess micromotion of an ion held off the RF null, and the modulation index of a beam
- Principal axes - `principal_axes` gives the tilt of the radial axes from y and z

`stability.report` puts all of these together for a site, along with the modes of a small crystal there. `tiopt.multipoles` expands the potential of every electrode at a point in multipoles, i.e. the constant, the fields Ex, Ey, and Ez, the quadrupoles U1 = (H_xx - H_yy) / 4, U2 = H_zz / 4, U3 = H_xy, U4 = H_yz, and U5 = H_xz (from the Hessian H), and real solid harmonics beyond, and `shim_matrix` finds the voltages which make each multipole on its own (the control voltages, or "C-file", which many labs use to null stray fields and set the quadrupoles), which `save_shim_matrix` writes to a text file.

## Layout files

Traps can be exported to GDSII files for layout tools such as KLayout with `Trap.save_gds`, drawn in SVG files with `Trap.save_svg`, and their potentials can be written on a grid of points to a VTK file with `tiopt.export.write_vtk` (for ParaView or VisIt). The GDSII files put RF electrodes on layer 1, DC electrodes on layer 2, grating couplers on layers 10-15 (one for each beam, in the order of `PHOTONICS_BEAMS`), detectors on layer 20, and zone labels on layer 30. `Trap.from_gds` reads the electrodes and zones back from the same layers, so trap designs can be drawn or edited in KLayout and then simulated with tiopt (hierarchical layouts should be flattened first).

## Design rules and gaps

`tiopt.drc.check_trap` (and the "Design rules and gaps" form of the Trap tab) checks a trap against design rules before fabrication, i.e. the narrowest part of every electrode and notch within one, the gaps between electrodes, sharp corners, tiny electrodes, overlapping electrodes, and the smallest teeth and grooves of the grating couplers. tiopt's designs are gapless (the electrodes touch), which the check points out, and `drc.gapped_trap` insets every electrode by half of a gap to open the gaps a fabricated trap needs, e.g. before writing its layout or solving it with the [boundary element method](#boundary-element-method).

## Boundary element method

`tiopt.bem.solve` (and the "BEM (finite gaps)" solver of the Electrostatics tab) solves the electrodynamics of a trap with the boundary element method, like the bem package of the Haeffner lab, which models what the gapless plane approximation leaves out: real gaps between the electrodes, a grounded plane beneath them (e.g. the substrate under the gaps), and a grounded cover plane above the trap (e.g. a window coated with indium tin oxide). The electrodes are thin conductors meshed into panels by a quadtree (with small panels at their edges, where the charge diverges), the charge on every panel is uniform, and the potential of every panel at the center of every other has a closed form (with a multipole expansion for distant panels, in `evaluate_panel_potentials` of the C library), which gives a dense linear system for the charges of every electrode at 1 V. The grounded planes are included by images of the panels, and a grounded frame around the trap stands in for the rest of the chip. The potential of every electrode is then expanded in spherical harmonics like those of the other solvers (through `Trap.expand_samples`), so the rest of tiopt uses it as usual. For the linear trap preset with 5 µm gaps, the solve takes about 10 s with panels of 4 µm at the edges (halving them moves the RF null by less than 0.1 µm and the secular frequencies by less than 0.3%), and its RF null sits about 0.8 µm higher and its radial frequencies are about 4% lower than with the gapless plane approximation. Electrodes which touch (as in gapless designs) have much stronger charges at their shared edges, so their potentials converge more slowly with the size of the panels, which is why gaps should be opened (e.g. with `drc.gapped_trap` or the gap of `bem.solve`) before solving a design with the boundary element method.

## Building traps from parts

The Trap builder tab of the GUI puts traps together from parts which are dragged from a palette onto the trap surface:
- Sections - linear sections (five-wire sections with a row of DC segments on each side, which can be joined end to end and onto the arms of junctions), curved sections and U-turns (five-wire sections bent around a circle, like the turns of a racetrack trap), X, T, and Y junctions, and point traps
- Electrodes - RF rails, DC electrodes, and rows of DC segments
- Zones - load, storage, gate, measurement, cooling, and junction zones
- Photonics - a grating coupler for each beam, which focuses on the nearest zone, and detectors

Parts snap to a grid and to the edges and centers of the parts next to them (so sections line up end to end and zones sit on the trap axis), the ends of sections snap onto the ends of other sections which point the opposite way (so that e.g. a curved section joins onto a linear section at any angle), and they can be rotated with R, deleted with Delete, duplicated with Ctrl+D, and edited in the "Selected part" panel, where composite parts can also be split up into their electrodes. Electrodes which overlap are outlined in red, since the gapless plane approximation would count them twice. "Build trap" turns the design into a solved trap which every other tab then uses, and it marks the RF null above every zone along with its height, which it finds across the local axis of the trap (so that zones on curved sections find their RF null too). Zones near the ends of rails see micromotion, which the summary points out, while a closed racetrack has no ends at all (see `examples/designs/racetrack.json`). On curved sections, the RF null sits a little lower and closer to the center of the circle than on a straight section. "Start from current trap" splits the current trap (e.g. a preset) into parts so that it can be edited, and designs can be saved to and opened from JSON files.

The same can be done from Python with `tiopt.builder`, e.g.
```python
from tiopt import builder
from tiopt.builder import Part

parts = [
    Part("linear_section", -125, 0), Part("linear_section", 125, 0),
    Part("zone", -100, 0, params=dict(types="load+storage")), Part("zone", 50, 0, params=dict(types="gate")),
    Part("grating", 10, 30, params=dict(beam="qubit")),
]
trap = builder.build_trap(parts, element="40Ca+")
trap.solve("analytical")
```

## Long traps

Wells can only be made within the spherical harmonic expansions, which reach about 200 µm on either side of the trap center in the default build. Zones beyond them are flagged in design reports and the co-design flow brings zones closer together to fit, but longer traps need a build with more spheres along the trap axis. For example, the build made by running the command
```
make clean && make DEFINES="-DNSPH_X=32"
```
reaches about 420 µm on either side, which covers nine zones 100 µm apart. `tiopt.required_nsph_x` gives the number of spheres a trap needs, and the design reports and error messages also give the command to rebuild with.

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

Grating couplers also take `azimuth`, `fill_factor`, `etch_depth`, `length`, `width`, `focal_distance`, `power`, `efficiency`, `wavelength`, `waveguide_width`, `waveguide_thickness`, `n_core`, and `n_cladding`, as well as the beam they deliver (`grating beam 0 qubit`) and the zone they address (`grating zone 0 2`). Detectors are given by `trap n_detectors` followed by `detector position`, `width`, `length`, `wavelength`, `quantum_efficiency`, `dark_count_rate`, and `zone`. Zones are given by `trap n_zones` followed by `zone type` (e.g. `zone type 0 load+storage`), `capacity`, and `position`. `write_trap_to_file` writes every parameter, and complete examples can be found in `examples/traps`.

# Modules

The sources are in `src` and their headers in `include`.

- `electrodynamics.c` - relaxation, spectral, and analytical (gapless plane) solvers, Driscoll-Healy sampling, spherical harmonic expansions (also of potentials from other solvers, such as the boundary element method), potentials, the pseudopotential, normal modes, the RF null, and the potentials of the panels of the boundary element method
- `traps.c`, `io.c`, and `ions.c` - the preset library (linear, junction, and point traps, and linear traps for zone layouts), integrated photonics for every zone, allocation of traps, trap files, and ion species (40Ca+, 43Ca+, 88Sr+, 171Yb+, 9Be+, 25Mg+, 133Ba+, 137Ba+, and 138Ba+)
- `zones.c` - zone layouts, zone types, and the zones of traps
- `waveforms.c` - well constraints, the waveform solver, NLopt refinement, transport waveforms, and waveforms for transport programs
- `verlet.c` - velocity Verlet integration of ion trajectories and of ensembles of ions
- `analysis.c` and `ion_heating` - heating rates from RF and DC voltage noise and from field noise spectra, motional excitation, trap depth, and barriers along the RF null
- `experiments.c` - idle and back-and-forth transport experiments, which the optimization loop in `main.c` uses
- `photonics.c` - slab and channel waveguide modes, grating couplers, the beams of each species, Gaussian beams, multilayer stacks, integrated detectors (collection efficiency, scattering, and detection fidelity), and curve fits
- `quantum.c` - the Lindblad master equation for ions sharing a motional mode during laser pulses with shaped envelopes, integrated with an adaptive Dormand-Prince method
- `transport.c` and `circuits.c` - OpenQASM parsing, circuit layers, and transport scheduling onto the nearest zones
- `codesign.c` - the co-design flow, including its stages, objectives, and quantum computing metrics
- `acceleration.c` and `kernels.cl` - threads and the OpenCL backend (which is loaded at runtime) with its kernels
- `bindings.c` - helpers for the Python bindings

`generate_transport_schedule` splits each layer of a circuit into rounds of operations which fit in the zones at once, which run one after another (`SCHEDULE_BASELINE` in `generate_transport_schedule_strategy` runs every layer at once instead, which only gives a valid schedule if every layer fits). Note that the transport programs record each transport into the zone of an operation, but not the transports of the ions back into storage after each round; `generate_transport_schedule_strategy` records them in a `ScheduleTrace`, along with the round of each transport, and the co-design flow adds them to the next transport of each ion. Circuits with gates which the parser does not know are rejected, and the scheduler prints the occupancy of the zones as it goes only if `TIOPT_SCHEDULE_VERBOSE` is set.

# Circuits, pulses, and transport

## Circuits

The Quantum tab of the GUI has an editor for OpenQASM circuits, which the Scheduling and Co-design tabs also use. `tiopt.circuits` reads OpenQASM 2 and 3 (registers declared with `qreg`/`creg` or `qubit`/`bit`, the standard gates of `qelib1.inc`, parameters written with `pi`, measurements with `->` or `=`, and barriers), and it simulates circuits with a state vector, or with a density matrix (or random errors on state vectors for more than 8 qubits) when gates have errors:
- Single-qubit and two-qubit gates are followed by depolarizing errors with the given average gate infidelities (which the co-design can fill in from its best design)
- Idle qubits dephase in every layer of the circuit
- Measurements are misread with the given probability

The simulation gives the distribution of outcomes with and without errors, the fidelity of the final state, and the native gates which the circuit compiles into on trapped ions, i.e. Mølmer-Sørensen gates, single-qubit rotations, and virtual Z rotations. The C scheduler only reads a few operations on a register named `qreg`, so `circuits.write_scheduler_qasm` writes a copy of a circuit with every physical single-qubit rotation as `x` and every two-qubit gate as `cx` for it.

## Pulse shapes

Laser pulses can have shaped envelopes (`enum PulseShape` in C, and `shape` and `shape_parameters` of `tiopt.Pulse`), which the master equation follows as it integrates:
- Square - on and off at once
- Trapezoid - ramps on and off linearly over a rise time and a fall time (a triangle if they add up to the duration)
- Gaussian - a Gaussian of a given width, shifted down so that it starts and ends at zero
- Cosine - ramps on and off as sin² over a rise time and a fall time (a sin² pulse if they add up to the duration)
- Blackman - a Blackman window over the whole pulse

Shaped pulses are longer than square pulses of the same peak Rabi frequency and area, but they excite far less off-resonant light (e.g. the carrier during a Mølmer-Sørensen gate). `tiopt.pulses.ms_gate` finds the duration and detuning of a Mølmer-Sørensen gate made with any of these shapes, so that the motion returns to where it started and the ions end in a Bell state, and the presets of the Quantum tab use it (and keep the area of the other pulses) for whichever shape is chosen.

## Transport priority

`tiopt.scheduling` decides when each transport of a circuit leaves and how fast it moves, with one of three priorities:
- As soon as idle, slowly - transports leave as soon as their ions are idle and take as long as they can until the operation needs them (down to the slowest speed), which excites the motion the least
- Just in time, quickly - transports leave as late as they can and move at full speed, so that ions wait where they were (e.g. in storage, where they can be cooled again) rather than in the zone of their next operation
- As soon as possible, quickly - transports leave as soon as their ions are idle and move at full speed, and the ions wait in the zone of their next operation

Every plan follows the mean phonon number of each ion, which grows at the heating rate, jumps by the coherent excitation of every transport (for a well which moves along a minimum-jerk profile), and returns to that after cooling if the ion waits long enough in a zone where it can be cooled. The operations start at the same times with every priority (since the fastest transports set them), so the priorities trade the motion of the ions at the gates against the time spent moving. The Scheduling tab shows every transport, wait, and operation of a plan in a timeline, compares the priorities, and builds a waveform which moves several ions at once wherever their transports overlap.

# Waveforms and hardware

## Voltages from objectives

`tiopt.voltages` finds static potentials and whole waveforms from objectives and constraints, in the style of pytrans. A `Problem` holds the voltages of every electrode at every step, and each objective is a linear function of them (the potential, gradient, or Hessian of the potential energy at a point, including the pseudopotential, the tilt of the radial axes, a multipole, a voltage, the slew rate of every voltage, or a symmetry between electrodes) which is either minimized as a weighted penalty or enforced as a constraint, so that the whole problem is a convex quadratic program, which tiopt solves with the alternating direction method of multipliers of OSQP. `voltages.static_well` and `voltages.transport` set up the usual problems (e.g. a transport along a sine squared profile with every voltage within 5 V and slew rates below 2 V/µs), and the static wells match those of the C solver.

## DACs and filters

`tiopt.hardware` follows a waveform through the hardware between it and the electrodes, like the filter transforms of pytrans and the post-processing of the shuttling toolchain of Conta et al.: a DAC which updates at a sample rate with a resolution and range, and a low-pass filter on every electrode (an RC or Butterworth filter, or one whose step response was measured). Pre-compensation finds the DAC samples whose filtered output follows the waveform as closely as possible while staying smooth and within the range of the DAC, e.g. for a 5 µs transport through 1 MHz RC filters it lowers the largest error at the electrodes from 0.19 V to 0.2 mV, so that the ion is excited as much as by the waveform itself.

# Crystals, gates, and architectures

## Coulomb crystals

`tiopt.crystals.find_crystal` (and the Crystals tab) finds the equilibrium of a crystal of ions (of one or more species) in a well of a trap or in a model of a linear trap, by minimizing the potential energy of every ion (the DC potential and the pseudopotential, which depends on the species) plus their Coulomb energy, and its normal modes from the mass-weighted Hessian, so linear chains, zigzags, and 2D and 3D crystals come out of the same calculation. Unstable configurations are pushed along their unstable mode and minimized again. The modes give the participation of every ion and the Lamb-Dicke parameter of every ion in every mode for a beam (`Crystal.lamb_dicke`), and the chains of `linear_chain` and `chain_frequencies` follow D. F. V. James (1998), which the crystals of the model trap match exactly.

## Molecular dynamics

`tiopt.dynamics.simulate` (and the "Crystal molecular dynamics" form of the Dynamics tab) follows many ions through a trap like pylion, with the Coulomb force between them, the pseudopotential or the full RF drive, the voltages of a waveform, Doppler cooling (the scattering force of a two-level ion with the recoil of every photon), and electric field noise, which heats them at the rate of tiopt's field noise spectra. Steps are divided while ions are close to each other, so that energy is conserved through close encounters, and e.g. a crystal at 0.1 K cools to a few mK in 100 µs. `dynamics.rf_phase_scan` runs a transport with the full RF drive at several phases of the drive, whose spread of final excitations shows how far the pseudopotential approximation holds for the transport.

## Gates on several modes

`tiopt.multimode.simulate` follows several ions sharing several modes through tones of light, like the Hamiltonians of IonSim.jl, with the Lamb-Dicke parameter of every ion in every mode, terms kept up to a Lamb-Dicke order and a rotating wave cutoff, the displacements either truncated to the Fock space or analytic, thermal modes, and quasi-static noise of the qubit and mode frequencies. `tiopt.gates.design_am_gate` designs amplitude-modulated Mølmer-Sørensen gates which close every mode with the least power (and, to be robust, every derivative of the displacements by the mode frequencies), as the eigenvector of the entangling phase on the null space of the closure conditions, and `design_fm_gate` designs frequency-modulated gates. For two 40Ca+ ions, a robust 100 µs gate on both axial modes reaches a fidelity of 0.9997, and still 0.9986 when the modes drift by 1 kHz, where a gate which is not robust drops to about 0.9. `gate_time_model` gives the durations of the gate models of Murali et al., which `tiopt.qccd` uses.

## Zeeman structure

`tiopt.levels` gives the Zeeman lines of the qubit of every species in a magnetic field, and how strongly a beam drives each of them from its wavevector and polarization relative to the field (with the Clebsch-Gordan coefficients and the quadrupole tensors of the S-D transition, and the dipole couplings of the S-P transitions), along with the Rabi frequency of each line at an intensity. For the odd isotopes, it diagonalizes the hyperfine structure of the ground state in a field (the Breit-Rabi levels), which gives the clock transition of each hyperfine qubit and how sensitive it is to the field.

## Fits to measured data

`tiopt.analytic` has analytic models of Rabi flopping on the carrier and sidebands of thermal ions (to every order in the Lamb-Dicke parameter) and of Mølmer-Sørensen interactions with thermal motion, and `fit_rabi_flop` fits measured flopping to find the Rabi frequency and mean phonon number of an ion (or its Lamb-Dicke parameter too).

## QCCD architectures

`tiopt.qccd.compile_circuit` compiles a circuit onto an architecture of traps which hold chains of ions, joined by segments and junctions (a line, a grid, a ring, or the zones of a trap), like QCCDSim and the MQT IonShuttler. It places the qubits in order of their first use and schedules every gate as soon as its ions and the traps, segments, and junctions it needs are free, bringing ions together by reordering their chains (with SWAP gates or physical swaps), splitting them off, moving them through every segment and junction, and merging them into the other chain. Chains carry motional energy, which every split, merge, junction, and segment adds to, and every two-qubit gate takes the time of its gate model (AM, PM, or FM) with the fidelity 1 - Γτ - A(2n̄ + 1) of Murali et al., where A grows with the length of the chain. `qccd.explore` (and "Explore capacities" in the Scheduling tab) compares trap capacities and gate models for a circuit.

# Co-design

## Full co-design

The full co-design (`tiopt.system`, and the default flow of the Co-design tab) designs the whole system for the circuit in the Quantum tab, starting from any trap with its zones and photonics (a preset, a trap from the Trap builder, or a file) and the inputs of every other tab:
- Trap and Trap builder tabs - the trap, the ion species, the RF drive, and the design region of the electrode shapes
- Photonics tab - the chip stack, the waveguide, the bends, and the qubit grating
- Quantum tab - the circuit and the pulse shape of the gates
- Scheduling tab - the transport priority, the fastest and slowest transports, and the cooling of waiting ions
- Waveforms tab - the axial frequency and the voltage limit
- Dynamics tab - the voltage noise of the DC electrodes

The Co-design tab adds the rest of what a system needs, i.e. the route of the qubit light from the edge of the chip to the grating (its length, number of bends, and coupling loss), the laser powers, the detector, the field noise, the dephasing rate, the drift of the ions, and the limits of the trap. Every evaluation follows the light from the edge of the chip through the waveguides and bends to the grating and the ion, the gates from the light (with the pulse shape), the plan of the transports with the motion of every ion, and the errors of every native gate of the circuit, whose product is the estimated fidelity of the circuit. The objective is -log(circuit fidelity), plus penalties for designs which can't run (unstable or shallow traps, voltages beyond the limit, or deformations which fold electrodes over) and a small weight on the pseudopotential barrier and confinement dip along a path through the shaped electrodes.

Every enabled group of variables is optimized together, i.e. the RF drive, the shapes of the electrodes, the cross-section of the waveguides, the radius, Euler fraction, and taper length of the bends, the fill factor, teeth, angular width, period, and apodization of the qubit grating, the fraction of the available Rabi frequency used and the ramps of the pulses, the fastest transport, the axial frequency, and the detection time and detector size. Parts of the evaluation which only some variables affect are kept between evaluations (e.g. the trap is only solved again when the shapes of the electrodes change), so a few hundred evaluations take under a minute without the electrode shapes. For example, for the Bell circuit on the default zone layout, 300 evaluations raise the estimated fidelity of the circuit from 0.71 to 0.93 by using more of the light, a lower axial frequency (and so a larger Lamb-Dicke parameter), and a larger grating. "Send to tabs" puts the best design back into the tabs it came from.

The same can be done from Python, e.g.
```python
import tiopt
from tiopt import circuits, system

ion = tiopt.Ion("40Ca+")
trap = tiopt.Trap.from_layout("load+storage, empty, gate, empty, measure", element="40Ca+")
trap.solve("analytical")
inputs = system.SystemInputs(trap=trap, ion=ion, circuit=circuits.parse_qasm(circuits.PRESETS["Bell pair"]), pulse_shape="cosine", ramp=2e-6)
codesign = system.SystemCodesign(inputs, groups=("trap", "gratings", "pulses", "waveforms"))
best = codesign.run(max_evaluations=200)
print("\n".join(system.format_evaluation(best, codesign.initial)))
```

## Co-design flow of the C library

The co-design flow (`codesign.c`) designs a linear trap with integrated photonics for the circuit in a QASM file, on the zones of a [zone layout](#zone-layouts) (by default `load+storage, empty, gate, empty, measure`). It goes through the following stages:

0. Start with basic designs of everything, i.e. the trap, the photonics (grating couplers focused on every zone which needs them, and a detector), the transport program which runs the circuit on the zones, and the waveform of the shuttle primitive into the first gate zone
1. Optimize the trap a little (segment width and RF drive)
2. Optimize the photonics a little (size of the qubit grating, fraction of the available Rabi frequency used for gates, detector size, and detection time)
3. Optimize the transport program a little (zone spacing and shuttle duration)
4. Optimize the waveforms a little (axial frequency and roughness weight)
5. Compute all of the QD/QC metrics, i.e. secular frequencies, Mathieu q, trap depth, and heating rate; Rabi frequency, Lamb-Dicke parameter, crosstalk, and detection fidelity; shuttle excitation and voltages; gate times and fidelities (with the Mølmer-Sørensen gate simulated by the master equation, including heating and dephasing); and the duration and estimated success probability of the whole circuit

The first pass optimizes each stage based on its own figures of merit. Every later pass repeats stages 1-5 using QC-aware optimization, where every stage minimizes -log(circuit fidelity) (plus penalties for unstable or shallow traps and for voltages beyond the limit), so that the trap is good for actually running circuits rather than just for its substeps. Each stage only takes a few evaluations of NLopt's COBYLA, and stages only redo the work which their parameters affect (e.g. the electrodynamics are only solved again when the geometry of the trap changes), so two passes take a few seconds.

With several gate zones, the gates of a layer run in parallel and the qubit beams are evaluated at every gate zone (so the gates are as good as those of the worst zone). Wells which are closer than several ion heights need large voltages to be held at the same time, so layouts where the ions wait in many zones at once can be limited by the voltages, e.g. a five-qubit GHZ circuit on `load, storage x2, gate x5, measure` needs over 20 V to hold its wells 100 µm apart.

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

Relaxation, potential maps (`evaluate_trap_potential_batch`, `evaluate_potential_energy_batch`, and `Trap.potential` and `Trap.potential_energy` in Python), and ensembles (`simulate_ensemble`) can run on a GPU through OpenCL in double precision. tiopt only loads `libOpenCL.so.1` once a GPU is requested, so it builds and runs without any GPU software, and everything falls back to the CPU if no device is found. The GPU can be selected with `set_compute_device(OPENCL_DEVICE)` in C, `tiopt.set_device("opencl")` in Python, the toolbar of the GUI, or `TIOPT_DEVICE=opencl ./build/tiopt`.

For NVIDIA GPUs, the proprietary driver includes an OpenCL platform (`nvidia-opencl-icd` or the equivalent for your distribution, along with `ocl-icd`). For AMD GPUs (e.g. the Radeon RX 5700 XT), an OpenCL runtime such as the one from ROCm needs to be installed, but the 5700 XT (gfx1010) is not officially supported by ROCm, so it's worth checking that the runtime finds it. Either way, `clinfo` should list the device with `cl_khr_fp64`. If there are several devices, `TIOPT_OPENCL_DEVICE=n` picks the n-th device with double precision, and `TIOPT_OPENCL_ANY_DEVICE=1` also allows CPU devices (such as PoCL, which the tests use to check the kernels without a GPU).

Consumer GPUs run double precision at 1/16 (AMD RDNA) to 1/64 (NVIDIA GeForce) of their single-precision rate, so they pay off for large maps and ensembles (thousands of points or ions) rather than for single trajectories. Data center GPUs (e.g. the NVIDIA A100 or H100) run double precision at half rate. The OpenCL results agree with the CPU up to rounding error.

# Related tools

Several open-source tools for trapped ions do parts of what tiopt does, and tiopt has taken many of their ideas:
- [pytrans](https://github.com/tiqi-group/pytrans) - objectives and constraints for potentials and waveforms, multi-ion analysis, and filter transforms
- [electrode](https://github.com/nist-ionstorage/electrode) - surface electrode patterns, saddle points, Mathieu analysis, and shim voltages
- [Sion](https://github.com/underforestaaa/Sion) - normal modes of mixed-species crystals, stability of asymmetric traps, and gaps in layouts
- [IonSim.jl](https://github.com/HaeffnerLab/IonSim.jl) - Hamiltonians of ions and lasers with several modes and Zeeman sublevels
- [bem](https://github.com/HaeffnerLab/bem) - the boundary element method for traps
- [pylion](https://pylion.readthedocs.io) - molecular dynamics of ions with LAMMPS
- [QCCDSim](https://github.com/prakashmurali/QCCDSim) and the [MQT IonShuttler](https://github.com/munich-quantum-toolkit/ionshuttler) - compilation of circuits onto QCCD architectures
- [gdsfactory](https://gdsfactory.github.io/gdsfactory/) - photonic layouts, routing, design rules, and circuit simulation

# Tests

The test suite can be run with
```
make test
```
which builds and runs the C tests (`tests/tiopt-test`) and then the Python tests (`python3 -m pytest tests`). The tests check the library against analytical results and independent calculations. The GUI tests use a virtual display from Xvfb when it is available, and the OpenCL tests only run when an OpenCL device with double precision is available.
