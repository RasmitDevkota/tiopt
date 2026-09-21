#include "test_electrodynamics.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../electrodynamics.h"
#include "../defs.h"
#include "../data_structures.h"
#include "../ti_utils.h"

int test_all_electrodynamics
(
	double atol
)
{
    printf("- Testing electrodynamics module...\n");

	int result = 0;

	result += test_solver_relaxation(atol);
	result += test_sample_dh1(atol);
	result += test_expand_spherical_harmonics(atol);

	result = result > 0;

	return result;
}

int test_solver_relaxation
(
	double atol
)
{
	// 1. Analytical result
	double (*V_analytical)[RELAXATION_NX][RELAXATION_NY][RELAXATION_NZ] = calloc(RELAXATION_NX * RELAXATION_NY * RELAXATION_NZ, sizeof(double));
	// @TODO - fill in data based on equation

	// 2. tiopt result
	struct Electrode *electrode = malloc(sizeof(struct Electrode));

	electrode->n_vertices = 4;
	electrode->vertices[0][0] = 1.0;
	electrode->vertices[0][1] = 1.0;
	electrode->vertices[0][2] = 0.0;
	electrode->vertices[1][0] = 1.0;
	electrode->vertices[1][1] = -1.0;
	electrode->vertices[1][2] = 0.0;
	electrode->vertices[2][0] = -1.0;
	electrode->vertices[2][1] = -1.0;
	electrode->vertices[2][2] = 0.0;
	electrode->vertices[3][0] = -1.0;
	electrode->vertices[3][1] = 1.0;
	electrode->vertices[3][2] = 0.0;

	electrode->n_edges = 4;
	electrode->edges[0][0] = 0;
	electrode->edges[0][1] = 1;
	electrode->edges[1][0] = 1;
	electrode->edges[1][1] = 2;
	electrode->edges[2][0] = 2;
	electrode->edges[2][1] = 3;
	electrode->edges[3][0] = 3;
	electrode->edges[3][1] = 0;

	electrode->Vlm_len = NSPH_X*NSPH_Y*NSPH_Z*((LMAX+1)*(LMAX+1)*2);

	double (*V_solver_relaxation)[RELAXATION_NX][RELAXATION_NY][RELAXATION_NZ] = calloc(RELAXATION_NX * RELAXATION_NY * RELAXATION_NZ, sizeof(double));

	int dx = 1;
	int dy = 1;
	int dz = 1;

	solver_relaxation(
		electrode,
		V_solver_relaxation,
		dx, dy, dz,
		(int) 1E3,
		0
	);

	double error = 0.0;

	int result = (error < atol) ? 1 : 0;

	free(electrode);
	free(V_analytical);
	free(V_solver_relaxation);

	return result;
}

// @TODO - generalize beyond the relaxation solver parameters
int test_sample_dh1
(
	double atol
)
{
	double (*V)[RELAXATION_NX][RELAXATION_NY][RELAXATION_NZ] = calloc(RELAXATION_NX * RELAXATION_NY * RELAXATION_NZ, sizeof(double));

	double dx = 1.0;
	double dy = 1.0;
	double dz = 1.0;

	// Define V(x, y, z) = x + y + z
	// @TODO - either keep this or use a function with a simpler analytical form in spherical coordinates
	for (int x = 0; x < RELAXATION_NX; x++)
		for (int y = 0; y < RELAXATION_NY; y++)
			for (int z = 0; z < +RELAXATION_NZ; z++)
				(*V)[x][y][z] = x * dx + y * dy + z * dz;

	int x_c = RELAXATION_NX/2;
	int y_c = RELAXATION_NY/2;
	int z_c = RELAXATION_NZ/2;

	// 1. Analytical result
	double grid_analytical[NLAT][NLON];
	for (int i = 0; i < NLAT; i++)
	{
		double phi = (double)i/NLAT * PI;

		// Precompute trigonometric quantities to reduce cost
		double R_sin_phi = SPH_R * sin(phi);
		double R_cos_phi = SPH_R * cos(phi);

		for (int j = 0; j < NLON; j++)
		{
			double theta = (double)j/NLON * PI;

			double x = x_c + R_sin_phi * cos(theta);
			double y = y_c + R_sin_phi * sin(theta);
			double z = z_c + R_cos_phi;

			grid_analytical[i][j] = x + y + z;
		}
	}

	// 2. tiopt result
	double grid_sample_dh1[NLAT][NLON];

	sample_dh1(
		RELAXATION_NX, RELAXATION_NY, RELAXATION_NZ,
		V,
		x_c, y_c, z_c,
		grid_sample_dh1,
		dx, dy, dz
	);

	double error = 0.0;
	for (int i = 0; i < NLAT; i++)
		for (int j = 0; j < NLON; j++)
			error += fabs(grid_sample_dh1[i][j] - grid_analytical[i][j]);

	int result = (error < atol) ? 1 : 0;

	free(V);

	return result;
}

// @TODO - generalize beyond the relaxation solver parameters
int test_expand_spherical_harmonics
(
	double atol
)
{
	double grid[NLAT][NLON];

	// Define grid(theta, phi) = 2 * Y_0,0 + 4 * Y_1,-1 + 8 * Y_2,2
	// @TODO - either keep this or use a function with a simpler analytical form in spherical coordinates
	for (int i = 0; i < NLAT; i++)
	{
		double phi = (double)i/NLAT * PI;

		for (int j = 0; j < NLON; j++)
		{
			double theta = (double)j/NLON * PI;

			grid[i][j] = 2 * 1/2 * 1/sqrt(PI) + 4 * 1/2 * sqrt(3/(2*PI)) * sin(theta) * cos(phi) + 8 * 1/4 * sqrt(15/(2*PI)) * pow(sin(theta), 2) * cos(2 * phi);
		}
	}

	int x_c = RELAXATION_NX/2;
	int y_c = RELAXATION_NY/2;
	int z_c = RELAXATION_NZ/2;

	// 1. Analytical result
	// @TODO - check this
	double *alm_analytical;
	alm_analytical[2 * (0 * (LMAX + 1) + 0)] = 2.0; // Y_0,0
	alm_analytical[2 * (1 * (LMAX + 1) + 1)] = pow(-1, 1) * 4.0; // Y_1,-1 = (-1)^1 (Y_1,1)*
	alm_analytical[2 * (2 * (LMAX + 1) + 2)] = 8.0; // Y_2,2

	// 2. tiopt result
	double *alm_expand_spherical_harmonics;
	expand_spherical_harmonics(
		grid,
		alm_expand_spherical_harmonics
	);

	double error = 0.0;
	for (int i = 0; i < NLAT; i++)
		for (int j = 0; j < NLON; j++)
			error += fabs(alm_expand_spherical_harmonics - alm_analytical);

	int result = (error < atol) ? 1 : 0;

	free(alm_analytical);

	return result;
}

