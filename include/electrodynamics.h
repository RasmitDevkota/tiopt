#pragma once

#include "defs.h"
#include "data_structures.h"

enum ElectrodynamicsSolver
{
	// In-house solvers:
	RELAXATION,
	ANALYTICAL,
	SPECTRAL,
	// External solvers:
};

void solve_trap_electrodynamics
(
	struct Trap *trap,
	enum ElectrodynamicsSolver electrodynamics_solver,
	char *grid_filename
);

int solver_relaxation
(
	const int NX,
	const int NY,
	const int NZ,
	struct Electrode *electrode,
	double electrode_position[3],
	double (*V)[NX][NY][NZ],
	int dx,
	int dy,
	int dz,
	int max_iterations,
	int chebyshev_acceleration
);

int solver_spectral
(
	const int NX,
	const int NY,
	const int NZ,
	struct Electrode *electrode,
	double electrode_position[3],
	double (*V)[NX][NY][NZ]
);

void write_potential_grid
(
	const int NX,
	const int NY,
	const int NZ,
	double (*V)[NX][NY][NZ],
	char *grid_filename
);

void evaluate_electrode_potential_analytical
(
	struct Electrode *electrode,
	double electrode_position[3],
	double position[3],
	double *phi
);

void sample_dh1
(
	const int NX,
	const int NY,
	const int NZ,
	double (*V)[NX][NY][NZ],
	double x_c,
	double y_c,
	double z_c,
	double grid[NLAT][NLON],
	double dx,
	double dy,
	double dz
);

void sample_dh1_analytical
(
	struct Electrode *electrode,
	double electrode_position[3],
	double x_c,
	double y_c,
	double z_c,
	double grid[NLAT][NLON]
);

extern void compute_shcoeffs_cmplx(double *grid, int *nlat, int *nlon, int *lmax, double *alm);
extern void compute_shcoeffs_real(double *grid, int *nlat, int *nlon, int *lmax, double *alm);

void expand_spherical_harmonics
(
	double grid[NLAT][NLON],
	double *alm
);

void compute_sphere_center
(
	int sx,
	int sy,
	int sz,
	double (*center)[3]
);

int find_nearest_sphere
(
	double position[3],
	int *sx,
	int *sy,
	int *sz
);

int evaluate_expansion_sets
(
	int n_electrodes,
	struct Electrode (*electrodes)[],
	int n_sets,
	double *weights,
	double position[3],
	double *phi,
	double (*grad_phi)[3],
	double (*H)[3][3]
);

int evaluate_expansion
(
	int n_electrodes,
	struct Electrode (*electrodes)[],
	double *weights,
	double position[3],
	double *phi,
	double grad_phi[3],
	double H[3][3]
);

int evaluate_electrode_derivatives
(
	struct Electrode *electrode,
	double position[3],
	double *phi,
	double grad_phi[3],
	double H[3][3]
);

int evaluate_electrode_derivatives_all
(
	struct Trap *trap,
	double position[3],
	double *phi,
	double (*grad_phi)[3],
	double (*H)[3][3]
);

void pack_expansion_coefficients
(
	struct Trap *trap,
	double *coefficients
);

int evaluate_electrode_potential
(
	struct Electrode *electrode,
	double position[3],
	double *phi
);

int evaluate_electrode_gradient
(
	struct Electrode *electrode,
	double position[3],
	double grad_phi[3]
);

int evaluate_electrode_hessian_analytical
(
	struct Electrode *electrode,
	double position[3],
	double H[3][3]
);

void evaluate_electrode_hessian_finite_differences
(
	struct Electrode *electrode,
	double position[3],
	double H[3][3]
);

int evaluate_trap_potential
(
	struct Trap *trap,
	double *voltages,
	double position[3],
	double *phi,
	double grad_phi[3],
	double H[3][3]
);

int evaluate_pseudopotential
(
	struct Trap *trap,
	const double m,
	const double q,
	double position[3],
	double *U_pond,
	double grad_U_pond[3],
	double H_U_pond[3][3]
);

int evaluate_potential_energy
(
	struct Trap *trap,
	const double m,
	const double q,
	double *voltages,
	double position[3],
	double *U,
	double grad_U[3],
	double H_U[3][3]
);

int compute_normal_modes
(
	struct Trap *trap,
	const double m,
	const double q,
	double *voltages,
	double position[3],
	double (*omega)[3],
	double (*axes)[3][3]
);

int find_potential_minimum
(
	struct Trap *trap,
	const double m,
	const double q,
	double *voltages,
	double (*position)[3],
	int max_iterations
);

int find_rf_null
(
	struct Trap *trap,
	double (*position)[3],
	int max_iterations
);
