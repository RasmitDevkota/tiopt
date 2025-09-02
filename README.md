# tiopt

tiopt is a C/Fortran-based program which simulates ion traps and ion transport, with the goal of optimizing trap design, transport waveforms, and ion control. It aims to enable all of the following in one tool:
- Construct a seed trap design, consisting of electrodes and (optionally) integrated photonics and electronics
- Simulate basic components once (e.g. simple electrodes, waveguide unit shapes)
- Represent electric potentials of electrodes in spherical harmonic expansions
- Obtain curve fits of parameters of photonic component
- Simulate ion transport via robust symplectic integration
- Simulate ion-photon interactions
- Optimize trap geometry, transport waveforms, and ion control based on experiments
