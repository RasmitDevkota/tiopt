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
