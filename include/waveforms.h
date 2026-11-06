#pragma once

#include "data_structures.h"

// Linear constraints which place a potential well at a position, with the trap axis (x) as the axial direction
enum WaveformConstraintType
{
	CONSTRAINT_EX_FIELD_X,
	CONSTRAINT_EX_FIELD_Y,
	CONSTRAINT_EX_FIELD_Z,
	CONSTRAINT_HESSIAN_XY,
	CONSTRAINT_HESSIAN_XZ,
	CONSTRAINT_AXIAL_CURVATURE,
	N_WELL_CONSTRAINTS,
};

struct WaveformConstraintData
{
	int n_voltages; // number of DC electrodes, i.e. optimization variables per step
	int voltage_offset; // index of the first optimization variable of this step
	int n_rows; // number of constraint rows (N_WELL_CONSTRAINTS per well)
	double *rows; // constraint coefficients; implicitly, length n_rows-by-n_voltages
	double *targets; // constraint targets; implicitly, length n_rows
	double max_voltage_individual;
	double max_voltage_total;
};

struct WaveformCostData
{
	int n_steps;
	int n_voltages;
	double voltage_weight;
	double roughness_weight;
};

struct TransportWaveform* allocate_transport_waveform
(
	int n_waveform_steps,
	int n_electrodes,
	int n_wells
);

void free_transport_waveform
(
	struct TransportWaveform *waveform
);

double waveform_cost_function(
	unsigned n,
	const double *x,
	double *grad,
	void *cost_data
);

void waveform_well_constraints
(
	unsigned m_constraints,
	double *result,
	unsigned n,
	const double *x,
	double *grad,
	void *constraint_data
);

double transport_waveform_constraint(
	unsigned n,
	const double *x,
	double *grad,
	void *constraint_data
);

void waveform_potential_derivatives
(
	struct Trap *trap,
	double *V,
	double position[3],
	double *grad_phi,
	double H[3][3]
);

void build_well_constraints
(
	struct Trap *trap,
	const double m,
	const double q,
	double position[3],
	const double omega_axial,
	double *rows,
	double *targets
);

int solve_waveform
(
	struct Trap *trap,
	const double m,
	const double q,
	int n_steps,
	int n_wells,
	double (*well_positions)[3],
	const double omega_axial,
	const double roughness_weight,
	double *voltages
);

int optimize_primitive
(
	struct Trap *trap,
	struct IonParameters *ion,
	struct TransportWaveform *waveform,
	const double omega_axial,
	const double max_voltage_individual,
	const double max_voltage_total,
	int max_iterations
);

struct TransportWaveform* generate_transport_waveform
(
	struct Trap *trap,
	struct IonParameters *ion,
	double start[3],
	double end[3],
	const double duration,
	const int n_steps,
	const double omega_axial
);

struct TransportWaveform* generate_transport_primitive
(
	struct Trap *trap,
	struct IonParameters *ion,
	double start[3],
	double end[3],
	const double duration,
	const int n_steps,
	const double omega_axial,
	const double roughness_weight
);

struct TransportWaveform* transport_program_to_waveform
(
	struct Trap *trap,
	struct IonParameters *ion,
	struct TransportProgram *transport_program,
	int *n_waveform_steps_list, // implicitly, length transport_program->n_transports
	const double omega_axial
);

void evaluate_waveform_errors
(
	struct Trap *trap,
	struct IonParameters *ion,
	struct TransportWaveform *waveform,
	const double omega_axial,
	double *max_force,
	double *max_frequency_error
);

void waveform_voltages_at_time
(
	struct TransportWaveform *waveform,
	const double t,
	double *voltages
);

void interpolate_waveform
(
	struct Trap *trap,
	struct TransportWaveform *waveform_coarse,
	struct TransportWaveform *waveform_fine,
	double dx
);

struct TransportWaveform* compose_waveforms
(
	int n_waveforms,
	struct TransportWaveform *waveforms,
	double *waveform_scales
);

void compute_secular_frequencies
(
	double q,
	double m,
	double V_DC,
	double V_RF,
	double Omega,
	double Z_0,
	double *omega
);

void compute_U_pond
(
	double m,
	double omega_rf,
	double V_rf,
	const int len,
	double *U_rf,
	double *U_pond,
	double dx
);
