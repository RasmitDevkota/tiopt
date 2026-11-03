#pragma once

#include "data_structures.h"

enum TrapPreset
{
	LINEAR_SURFACE_TRAP,
	LINEAR_SURFACE_TRAP_WITH_GRATINGS,
	LINEAR_SURFACE_TRAP_WITH_PHOTONICS,
	X_JUNCTION_TRAP, // four arms at right angles
	Y_JUNCTION_TRAP, // three arms 120 degrees apart
	T_JUNCTION_TRAP, // a straight trap with a third arm at right angles
	POINT_TRAP, // a ring of RF which holds one ion above its center
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

int add_zone_photonics
(
	struct Trap *trap,
	const char *element
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

int generate_junction_trap
(
	int n_arms,
	double *arm_angles,
	int n_segments,
	double segment_width,
	struct Trap *trap
);

int generate_trap_from_preset
(
	enum TrapPreset trap_preset,
	int n_segments,
	double segment_width,
	struct Trap *trap
);

int generate_trap_from_layout
(
	const char *layout,
	int n_ions,
	double zone_spacing,
	double segment_width,
	const char *element,
	int photonics,
	struct Trap *trap
);

void free_trap
(
	struct Trap *trap
);
