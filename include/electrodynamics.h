#include "defs.h"
#include "data_structures.h"

enum ElectrodynamicsSolver
{
	// In-house solvers:
	RELAXATION,
	// External solvers:
};

void solve_trap_electrodynamics
(
	struct Trap *trap,
	enum ElectrodynamicsSolver electrodynamics_solver
);

void solver_relaxation
(
	struct Electrode *electrode,
	double (*V)[RELAXATION_NX][RELAXATION_NY][RELAXATION_NZ],
	int dx,
	int dy,
	int dz,
	int max_iterations,
	int chebyshev_acceleration
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

extern void compute_shcoeffs_cmplx(double *grid, int *nlat, int *nlon, int *lmax, double *alm);
extern void compute_shcoeffs_real(double *grid, int *nlat, int *nlon, int *lmax, double *alm);

void expand_spherical_harmonics
(
	double grid[NLAT][NLON],
	double *alm
);

