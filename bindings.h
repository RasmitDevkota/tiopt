#pragma once

#include "data_structures.h"
#include "codesign.h"

int tiopt_struct_size
(
	char *name
);

double tiopt_constant
(
	char *name
);

void evaluate_trap_potential_batch
(
	struct Trap *trap,
	double *voltages,
	int n_points,
	double (*positions)[3],
	double *phi,
	double (*grad_phi)[3]
);

void evaluate_potential_energy_batch
(
	struct Trap *trap,
	const double m,
	const double q,
	double *voltages,
	int n_points,
	double (*positions)[3],
	double *U
);

void evaluate_electrode_potential_analytical_batch
(
	struct Electrode *electrode,
	double electrode_position[3],
	int n_points,
	double (*positions)[3],
	double *phi
);

void evaluate_grating_beam_batch
(
	struct GratingCoupler *grating,
	int n_points,
	double (*positions)[3],
	double *intensity
);

int get_codesign_n_records
(
	struct Codesign *codesign
);

void get_codesign_record
(
	struct Codesign *codesign,
	int r,
	struct CodesignRecord *record
);

void get_codesign_state
(
	struct Codesign *codesign,
	struct CodesignParameters *parameters,
	struct CodesignEnvironment *environment,
	struct QCMetrics *metrics
);

void set_codesign_state
(
	struct Codesign *codesign,
	struct CodesignParameters *parameters,
	struct CodesignEnvironment *environment
);

struct Trap* get_codesign_trap
(
	struct Codesign *codesign
);

void get_codesign_layout
(
	struct Codesign *codesign,
	double gate_position[3],
	double measure_position[3],
	int *n_zones,
	int *n_layers,
	int *n_ions
);
