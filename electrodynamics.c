#include "electrodynamics.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"
#include "ti_utils.h"

void solve_trap_electrodynamics
(
	struct Trap *trap,
	enum ElectrodynamicsSolver electrodynamics_solver
)
{
	// @TODO - implement parallelism
	for (int e = 0; e < trap->n_electrodes; e++)
	{
		printf("Solving electrodynamics for electrode %d/%d\n", e+1, trap->n_electrodes);

		struct Electrode electrode = (*trap->electrodes)[e];

		if (electrodynamics_solver == RELAXATION)
		{
			double (*V)[RELAXATION_NX][RELAXATION_NY][RELAXATION_NZ] = calloc(RELAXATION_NX * RELAXATION_NY * RELAXATION_NZ, sizeof(double));

			int dx = 1;
			int dy = 1;
			int dz = 1;

			solver_relaxation(
				&electrode,
				V,
				dx, dy, dz,
				(int) 1E3,
				0
			);

			printf("Writing electric potential to CSV format...\n");
			FILE *f = fopen("V.txt", "w");
			for (int x = 0; x < RELAXATION_NX; x++)
				for (int y = 0; y < RELAXATION_NY; y++)
					for (int z = 0; z < +RELAXATION_NZ; z++)
						if ((*V)[x][y][z] > 0.0 || (*V)[x][y][z] < 0.0)
							fprintf(f, "%d,%d,%d,%f\n", x, y, z, (*V)[x][y][z]);
			fclose(f);

			printf("Sampling grid and expanding potential in spherical harmonics basis...\n");
			for (int sx = 0; sx < NSPH_X; sx++)
				for (int sy = 0; sy < NSPH_Y; sy++)
					for (int sz = 0; sz < NSPH_Z; sz++)
					{
						double x_c = sx * SPH_SPACING + SPH_R;
						double y_c = sy * SPH_SPACING + SPH_R;
						double z_c = sz * SPH_SPACING + SPH_Z_MIN + 1; // # @TEST - check the +1
						printf("Sphere at index (%d, %d, %d), position (%f, %f, %f)\n", sx, sy, sz, x_c, y_c, z_c);

						// Sample grid based on method by Driscoll and Healy (1994)
						double grid[NLAT][NLON];
						sample_dh1(
							RELAXATION_NX, RELAXATION_NY, RELAXATION_NZ,
							V,
							x_c, y_c, z_c,
							grid,
							dx, dy, dz
						);

						// Compute the spherical harmonics expansion of the potential energy contribution
						expand_spherical_harmonics(
							grid,
							VLM_SLICE(electrode.Vlm, sx, sy, sz)
						);
					}

			free(V);
		}
	}
}

void solver_relaxation
(
	struct Electrode *electrode,
	double (*V)[RELAXATION_NX][RELAXATION_NY][RELAXATION_NZ],
	int dx,
	int dy,
	int dz,
	int max_iterations,
	int chebyshev_acceleration
)
{
	printf("Constructing relaxation problem...\n");

	for (int x = 0; x < RELAXATION_NX; x++)
		for (int y = 0; y < RELAXATION_NY; y++)
			for (int z = 0; z < RELAXATION_NZ; z++)
				if (
					point_in_polygon_zslice(x - RELAXATION_NX/2, y - RELAXATION_NY/2, z, electrode->n_vertices, electrode->vertices, 0)
				)
					(*V)[x][y][z] = 1.0;

	// Poisson equation-tuned constant vs. Chebyshev-accelerated estimate for omega
	double rho = 1.0;
	double omega = chebyshev_acceleration ? 1.0 : 1.98;

	// Residual tracking
	double xi_max = 0.0;
	double xi_it_max_prev = 0.0;

	printf("Solving relaxation problem...\n");

	for (int it = 0; it < max_iterations; it++)
	{
		double xi_it_max = 1E-13;

		for (int p = 0; p < 2; p++)
		{
			for (int x = 0; x < RELAXATION_NX-1; x++)
				for (int y = 0; y < RELAXATION_NY-1; y++)
					for (int z = 0; z < RELAXATION_NZ-1; z++)
					{
						if (
							!point_in_polygon_zslice(x - RELAXATION_NX/2, y - RELAXATION_NY/2, z, electrode->n_vertices, electrode->vertices, 1) &&
							(x != 0 && y != 0 && z != 0) &&
							((int)(x + y + z) % 2 == p)
						)
						{
							double xi = (
								(*V)[x - dx][y][z] + (*V)[x + dx][y][z] +
								(*V)[x][y - dy][z] + (*V)[x][y + dy][z] +
								(*V)[x][y][z - dz] + (*V)[x][y][z + dz]
							) - 6 * (*V)[x][y][z];

							if (fabs(xi) > fabs(xi_it_max))
								xi_it_max = xi;

							(*V)[x][y][z] += 1.0/6.0 * omega * xi;
						}
					}

			if (rho > 1E-9)
			{
				rho = 1.0 - 1E-6;
				omega = (it == 0 && p == 0) ? 1 / (1 - 0.5 * rho*rho) : 1 / (1 - 0.25 * rho*rho * omega);
			}
		}

		if (fabs(xi_it_max - xi_it_max_prev) < 1E-9)
		{
			printf("Stopping relaxation loop early at iteration %d due to small residuals\n", it);
			break;
		}

		xi_max = MAX(xi_it_max, xi_max);
		xi_it_max_prev = xi_it_max;
	}
}

// @TODO - consider generalizing syntax and moving to ti_utils.c for generality
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
)
{
	for (int i = 0; i < NLAT; i++) {
		double phi = (double)i/NLAT * PI;

		// Precompute trigonometric quantities to reduce cost
		double R_sin_phi = SPH_R * sin(phi);
		double R_cos_phi = SPH_R * cos(phi);

		for (int j = 0; j < NLON; j++) {
			double theta = (double)j/NLON * PI;

			double sample_coord[3] = { x_c + R_sin_phi * cos(theta), y_c + R_sin_phi * sin(theta), z_c + R_cos_phi };

			interpolate_3d(
				RELAXATION_NX, RELAXATION_NY, RELAXATION_NZ,
				V,
				sample_coord,
				&grid[i][j],
				dx, dy, dz
			);

			// @TEST - print non-zero values
			// if (fabs(grid[i][j]) > 1E-6)
			// 	printf("V(%f,%f,%f)=%.13f\n", SPH_R, phi, theta, grid[i][j]);
		}
	}
}

// @TODO - generalize syntax and move to ti_utils.c for generality
void expand_spherical_harmonics
(
	double grid[NLAT][NLON],
	double *alm
)
{
	// Define new variables so that Fortran recognizes the types
	int nlat = NLAT;
	int nlon = NLON;
	int lmax = LMAX;

	// Call the Fortran subroutine
	// @TODO - get the complex subroutine working
	// compute_shcoeffs_cmplx((double*) grid, &nlat, &nlon, &lmax, alm);
	compute_shcoeffs_real((double*) grid, &nlat, &nlon, &lmax, alm);

	// @TEST - print non-zero coefficients
	for (int l = 0; l <= LMAX; l++) {
		for (int m = -l; m <= l; m++) {
			double alm_real = 0.0;
			double alm_imag = 0.0;

			if (m < 0)
			{
				// Y_l,-m = (-1)^m (Y_l,m)*
				alm_real = pow(-1, m) * alm[2 * (l * (LMAX + 1) + abs(m))];
				alm_imag = pow(-1, m) * -1 * alm[2 * (l * (LMAX + 1) + abs(m)) + 1];
			}
			else
			{
				alm_real = alm[2 * (l * (LMAX + 1) + m)];
				alm_imag = alm[2 * (l * (LMAX + 1) + m) + 1];
			}

			if (fabs(alm_real) > 1E-6 || fabs(alm_imag) > 1E-6)
				printf("a_lm[%d,+%d] = %f + %fi\n", l, m, alm_real, alm_imag);
		}
	}
}

