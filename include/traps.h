#pragma once

#include "data_structures.h"

enum TrapPreset
{
	LINEAR_SURFACE_TRAP,
	LINEAR_SURFACE_TRAP_WITH_GRATINGS,
	LINEAR_SURFACE_TRAP_WITH_PHOTONICS,
};

// Grating couplers added by add_trap_photonics, by the 40Ca+ transition they drive
enum PhotonicsBeam
{
	QUBIT_BEAM, // 729 nm S1/2-D5/2 qubit transition, at the gate zone
	DETECTION_BEAM, // 397 nm S1/2-P1/2 cooling and detection transition, at the measurement zone
	REPUMP_BEAM, // 866 nm D3/2-P1/2 repumper, at the measurement zone
	QUENCH_BEAM, // 854 nm D5/2-P3/2 quench for qubit reset, at the gate zone
	N_PHOTONICS_BEAMS,
};

void allocate_trap
(
	struct Trap *trap,
	int n_electrodes
);

void allocate_grating_couplers
(
	struct Trap *trap,
	int n_grating_couplers
);

void allocate_detectors
(
	struct Trap *trap,
	int n_detectors
);

int add_trap_photonics
(
	struct Trap *trap,
	double gate_position[3],
	double measure_position[3]
);

void set_polygon_electrode
(
	struct Trap *trap,
	int e,
	int n_vertices,
	double (*vertices)[3],
	double position[3]
);

void set_rectangular_electrode
(
	struct Trap *trap,
	int e,
	double x_min,
	double x_max,
	double y_min,
	double y_max
);

int generate_trap_from_preset
(
	enum TrapPreset trap_preset,
	int n_segments,
	double segment_width,
	struct Trap *trap
);

void free_trap
(
	struct Trap *trap
);
