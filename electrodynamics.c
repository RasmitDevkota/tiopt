#include "electrodynamics.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#include <fftw3.h>

#include "defs.h"
#include "data_structures.h"
#include "ti_utils.h"
#include "acceleration.h"

static double sh_wrapper_normalization
(
);

// Sample a solved potential grid on every sphere and expand it in spherical harmonics, in parallel over the spheres
static void expand_potential_grid
(
	const int NX,
	const int NY,
	const int NZ,
	double (*V)[NX][NY][NZ],
	struct Electrode *electrode
)
{
	int dx = 1;
	int dy = 1;
	int dz = 1;

	printf("Sampling grid and expanding potential in spherical harmonics basis...\n");

	#pragma omp parallel for collapse(3) schedule(dynamic)
	for (int sx = 0; sx < NSPH_X; sx++)
		for (int sy = 0; sy < NSPH_Y; sy++)
			for (int sz = 0; sz < NSPH_Z; sz++)
			{
				// The relaxation grid is indexed from its corner rather than from the trap frame origin
				double x_c = SPH_CENTER_X(sx) + NX/2;
				double y_c = SPH_CENTER_Y(sy) + NY/2;
				double z_c = SPH_CENTER_Z(sz);
				// @TEST - print sphere positions
				// printf("Sphere at index (%d, %d, %d), position (%f, %f, %f)\n", sx, sy, sz, x_c, y_c, z_c);

				// Sample grid based on method by Driscoll and Healy (1994)
				double grid[NLAT][NLON];
				sample_dh1(
					NX, NY, NZ,
					V,
					x_c, y_c, z_c,
					grid,
					dx, dy, dz
				);

				// Compute the spherical harmonics expansion of the potential energy contribution
				expand_spherical_harmonics(
					grid,
					VLM_SLICE(electrode->Vlm, sx, sy, sz)
				);
			}
}

// Solve for the potential of every electrode of a trap with the given solver and expand it in spherical
// harmonics within every sphere, optionally writing the grids of the grid solvers to grid_filename
void solve_trap_electrodynamics
(
	struct Trap *trap,
	enum ElectrodynamicsSolver electrodynamics_solver,
	char *grid_filename
)
{
	// FFTW plans may be made by several threads at once (by SHTOOLS and the spectral solver), which needs a thread-safe
	// FFTW planner
	make_fftw_thread_safe();

	for (int e = 0; e < trap->n_electrodes; e++)
	{
		printf("Solving electrodynamics for electrode %d/%d\n", e+1, trap->n_electrodes);

		struct Electrode electrode = (*trap->electrodes)[e];
		double *electrode_position = (*trap->electrode_positions)[e];

		if (electrodynamics_solver == RELAXATION || electrodynamics_solver == SPECTRAL)
		{
			double (*V)[RELAXATION_NX][RELAXATION_NY][RELAXATION_NZ] = calloc(RELAXATION_NX * RELAXATION_NY * RELAXATION_NZ, sizeof(double));

			int dx = 1;
			int dy = 1;
			int dz = 1;

			// The spectral solver only handles electrodes in the trap surface, so other electrodes fall back to relaxation
			int status = 1;
			if (electrodynamics_solver == SPECTRAL)
				status = solver_spectral(
					RELAXATION_NX, RELAXATION_NY, RELAXATION_NZ,
					&electrode,
					electrode_position,
					V
				);

			if (status != 0)
				solver_relaxation(
					RELAXATION_NX, RELAXATION_NY, RELAXATION_NZ,
					&electrode,
					electrode_position,
					V,
					dx, dy, dz,
					(int) 1E4,
					1
				);

			if (grid_filename != NULL)
			{
				// Each electrode gets its own file, with the electrode index inserted before the extension
				char electrode_grid_filename[256];
				const char *extension = strrchr(grid_filename, '.');
				int stem_len = extension ? (int)(extension - grid_filename) : (int) strlen(grid_filename);
				snprintf(electrode_grid_filename, sizeof(electrode_grid_filename), "%.*s_%d%s", stem_len, grid_filename, e, extension ? extension : "");

				printf("Writing electric potential to CSV format...\n");
				write_potential_grid(RELAXATION_NX, RELAXATION_NY, RELAXATION_NZ, V, electrode_grid_filename);
			}

			expand_potential_grid(RELAXATION_NX, RELAXATION_NY, RELAXATION_NZ, V, &electrode);

			free(V);
		}
		else if (electrodynamics_solver == ANALYTICAL)
		{
			printf("Sampling analytical potential and expanding in spherical harmonics basis...\n");

			#pragma omp parallel for collapse(3) schedule(dynamic)
			for (int sx = 0; sx < NSPH_X; sx++)
				for (int sy = 0; sy < NSPH_Y; sy++)
					for (int sz = 0; sz < NSPH_Z; sz++)
					{
						// Sample the potential directly on the sphere, so no volumetric grid is needed
						double grid[NLAT][NLON];
						sample_dh1_analytical(
							&electrode,
							electrode_position,
							SPH_CENTER_X(sx), SPH_CENTER_Y(sy), SPH_CENTER_Z(sz),
							grid
						);

						expand_spherical_harmonics(
							grid,
							VLM_SLICE(electrode.Vlm, sx, sy, sz)
						);
					}
		}
	}
}

// Set up the boundary value problem for an electrode held at 1 V in a grounded box: cells on the boundary of the box are
// held at 0 V, and cells covered by the electrode at the fraction of them it covers. The coverage of each cell is
// estimated from four samples a quarter of a cell from its center, so cells on an edge of the electrode are half covered
// rather than being assigned to one side of it. Returns the number of cells held fixed inside the box
static int rasterize_electrode
(
	const int NX,
	const int NY,
	const int NZ,
	struct Electrode *electrode,
	double electrode_position[3],
	double (*V)[NX][NY][NZ],
	unsigned char (*fixed)[NX][NY][NZ],
	int dx,
	int dy,
	int dz
)
{
	int n_vertices = electrode->n_vertices;
	double sample_offsets[4][2] = { { 0.25, 0.25 }, { -0.25, 0.25 }, { -0.25, -0.25 }, { 0.25, -0.25 } };

	// Coordinates of the electrode's vertices in the trap frame, shifted for each of the samples
	double (*vertices)[3] = malloc(4 * n_vertices * sizeof(double[3]));
	for (int s = 0; s < 4; s++)
		for (int v = 0; v < n_vertices; v++)
			for (int k = 0; k < 3; k++)
				vertices[s * n_vertices + v][k] = electrode_position[k] + electrode->vertices[v][k] + ((k < 2) ? sample_offsets[s][k] : 0.0);

	#pragma omp parallel for
	for (int x = 0; x < NX; x++)
		for (int y = 0; y < NY; y++)
			for (int z = 0; z < NZ; z++)
			{
				int boundary = x < dx || y < dy || z < dz || x >= NX-dx || y >= NY-dy || z >= NZ-dz;

				(*V)[x][y][z] = 0.0;
				(*fixed)[x][y][z] = boundary;
			}

	// Only cells within the bounding box of the electrode can be covered by it
	double min_corner[3] = { HUGE_VAL, HUGE_VAL, HUGE_VAL };
	double max_corner[3] = { -HUGE_VAL, -HUGE_VAL, -HUGE_VAL };
	for (int v = 0; v < 4 * n_vertices; v++)
		for (int k = 0; k < 3; k++)
		{
			min_corner[k] = MIN(min_corner[k], vertices[v][k]);
			max_corner[k] = MAX(max_corner[k], vertices[v][k]);
		}

	int x_min = MAX((int) floor(min_corner[0]) + NX/2, 0);
	int x_max = MIN((int) ceil(max_corner[0]) + NX/2, NX - 1);
	int y_min = MAX((int) floor(min_corner[1]) + NY/2, 0);
	int y_max = MIN((int) ceil(max_corner[1]) + NY/2, NY - 1);
	int z_min = MAX((int) floor(min_corner[2]), 0);
	int z_max = MIN((int) ceil(max_corner[2]), NZ - 1);

	int n_fixed_inside = 0;

	#pragma omp parallel for reduction(+:n_fixed_inside)
	for (int x = x_min; x <= x_max; x++)
		for (int y = y_min; y <= y_max; y++)
			for (int z = z_min; z <= z_max; z++)
			{
				int n_inside = 0;
				for (int s = 0; s < 4; s++)
					n_inside += point_in_polygon_zslice(x - NX/2, y - NY/2, z, n_vertices, &vertices[s * n_vertices], 1);

				if (n_inside > 0)
				{
					(*V)[x][y][z] = n_inside / 4.0;

					if (!(*fixed)[x][y][z])
						n_fixed_inside++;

					(*fixed)[x][y][z] = 1;
				}
			}

	free(vertices);

	return n_fixed_inside;
}

// Solve Laplace's equation for the potential of an electrode held at 1 V in a grounded box by successive
// overrelaxation with red-black ordering, on the OpenCL device if one is selected; returns the number of iterations
// performed
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
)
{
	printf("Constructing relaxation problem...\n");

	// Cells on the electrode and on the boundary of the domain are held at a fixed potential
	unsigned char (*fixed)[NX][NY][NZ] = calloc(NX * NY * NZ, sizeof(unsigned char));

	rasterize_electrode(NX, NY, NZ, electrode, electrode_position, V, fixed, dx, dy, dz);

	// Spectral radius of the Jacobi iteration on this grid, which sets the optimal overrelaxation parameter
	double rho = (cos(PI/NX) + cos(PI/NY) + cos(PI/NZ)) / 3.0;

	// Optimal constant omega vs. Chebyshev-accelerated sequence of omegas
	double omega = chebyshev_acceleration ? 1.0 : 2.0 / (1.0 + sqrt(1.0 - rho*rho));

	printf("Solving relaxation problem...\n");

	int it;
	if (
		dx == 1 && dy == 1 && dz == 1 &&
		opencl_relaxation(NX, NY, NZ, (unsigned char *) fixed, (double *) V, max_iterations, chebyshev_acceleration, rho, &it) == 0
	)
	{
		free(fixed);
		return it;
	}

	for (it = 0; it < max_iterations; it++)
	{
		// Residual tracking
		double xi_it_max = 0.0;

		// Red-black ordering, so that each half-sweep only reads cells of the other parity (which also makes the cells
		// of one parity independent of each other, so that they can be updated in parallel)
		for (int p = 0; p < 2; p++)
		{
			#pragma omp parallel for reduction(max:xi_it_max) schedule(static)
			for (int x = dx; x < NX-dx; x++)
				for (int y = dy; y < NY-dy; y++)
					for (int z = dz + (x + y + dz + p) % 2; z < NZ-dz; z += 2)
					{
						if ((*fixed)[x][y][z])
							continue;

						double xi = (
							(*V)[x - dx][y][z] + (*V)[x + dx][y][z] +
							(*V)[x][y - dy][z] + (*V)[x][y + dy][z] +
							(*V)[x][y][z - dz] + (*V)[x][y][z + dz]
						) - 6 * (*V)[x][y][z];

						if (fabs(xi) > xi_it_max)
							xi_it_max = fabs(xi);

						(*V)[x][y][z] += 1.0/6.0 * omega * xi;
					}

			// Chebyshev acceleration (Numerical Recipes, Section 19.5)
			if (chebyshev_acceleration)
				omega = (it == 0 && p == 0) ? 1 / (1 - 0.5 * rho*rho) : 1 / (1 - 0.25 * rho*rho * omega);
		}

		if (xi_it_max / 6.0 < RELAXATION_TOLERANCE)
		{
			printf("Stopping relaxation loop early at iteration %d due to small residuals\n", it);
			break;
		}

		if (it % 100 == 0)
			printf("Relaxation iteration %d, maximum residual %.3e\n", it, xi_it_max / 6.0);
	}

	free(fixed);

	return it;
}

// Solve the same boundary value problem as solver_relaxation exactly, for electrodes in the trap surface: the discrete
// Laplace equation in the grounded box separates in a basis of sines along x and y, in which the potential decays along
// z as sinh(mu (NZ - 1 - z)) / sinh(mu (NZ - 1)) with 2 cosh(mu) = 6 - 2 cos(k_x) - 2 cos(k_y). Returns 1 (leaving V
// unsolved) if the electrode extends into the box
int solver_spectral
(
	const int NX,
	const int NY,
	const int NZ,
	struct Electrode *electrode,
	double electrode_position[3],
	double (*V)[NX][NY][NZ]
)
{
	printf("Constructing spectral problem...\n");

	unsigned char (*fixed)[NX][NY][NZ] = calloc(NX * NY * NZ, sizeof(unsigned char));

	int n_fixed_inside = rasterize_electrode(NX, NY, NZ, electrode, electrode_position, V, fixed, 1, 1, 1);

	free(fixed);

	if (n_fixed_inside > 0)
	{
		printf("WARNING: The spectral solver only supports electrodes in the trap surface; falling back to relaxation\n");
		return 1;
	}

	printf("Solving spectral problem...\n");

	// Sine transforms (DST-I) of the interior of each plane
	int nx = NX - 2;
	int ny = NY - 2;
	int N = NZ - 1;

	double *boundary = fftw_malloc((size_t) nx * ny * sizeof(double));
	double *boundary_hat = fftw_malloc((size_t) nx * ny * sizeof(double));

	fftw_plan forward_plan;
	fftw_plan inverse_plan;

	#pragma omp critical (fftw_planner)
	{
		forward_plan = fftw_plan_r2r_2d(nx, ny, boundary, boundary_hat, FFTW_RODFT00, FFTW_RODFT00, FFTW_ESTIMATE);
		inverse_plan = fftw_plan_r2r_2d(nx, ny, boundary_hat, boundary, FFTW_RODFT00, FFTW_RODFT00, FFTW_ESTIMATE);
	}

	for (int x = 0; x < nx; x++)
		for (int y = 0; y < ny; y++)
			boundary[x * ny + y] = (*V)[x + 1][y + 1][0];

	fftw_execute(forward_plan);

	// Applying the transform twice multiplies by 4 (nx + 1) (ny + 1)
	double normalization = 1.0 / (4.0 * (nx + 1) * (ny + 1));

	// Decay rate of each mode along z
	double *mu = malloc((size_t) nx * ny * sizeof(double));
	for (int k = 0; k < nx; k++)
		for (int l = 0; l < ny; l++)
		{
			double lambda = 6 - 2 * cos(PI * (k + 1) / (nx + 1)) - 2 * cos(PI * (l + 1) / (ny + 1));
			mu[k * ny + l] = acosh(0.5 * lambda);
		}

	#pragma omp parallel
	{
		double *plane_hat = fftw_malloc((size_t) nx * ny * sizeof(double));
		double *plane = fftw_malloc((size_t) nx * ny * sizeof(double));

		#pragma omp for schedule(dynamic)
		for (int z = 1; z < NZ-1; z++)
		{
			// sinh(mu (N - z)) / sinh(mu N), written to avoid overflow
			for (int i = 0; i < nx * ny; i++)
				plane_hat[i] = normalization * boundary_hat[i] * exp(-mu[i] * z) * (1 - exp(-2 * mu[i] * (N - z))) / (1 - exp(-2 * mu[i] * N));

			fftw_execute_r2r(inverse_plan, plane_hat, plane);

			for (int x = 0; x < nx; x++)
				for (int y = 0; y < ny; y++)
					(*V)[x + 1][y + 1][z] = plane[x * ny + y];
		}

		fftw_free(plane);
		fftw_free(plane_hat);
	}

	free(mu);

	#pragma omp critical (fftw_planner)
	{
		fftw_destroy_plan(forward_plan);
		fftw_destroy_plan(inverse_plan);
	}

	fftw_free(boundary_hat);
	fftw_free(boundary);

	return 0;
}

// Write the nonzero values of a potential grid to a file, one x,y,z,V line per point
void write_potential_grid
(
	const int NX,
	const int NY,
	const int NZ,
	double (*V)[NX][NY][NZ],
	char *grid_filename
)
{
	FILE *f = fopen(grid_filename, "w");

	if (f == NULL)
	{
		printf("ERROR: Failed to open '%s' for writing\n", grid_filename);
		return;
	}

	for (int x = 0; x < NX; x++)
		for (int y = 0; y < NY; y++)
			for (int z = 0; z < NZ; z++)
				if ((*V)[x][y][z] > 0.0 || (*V)[x][y][z] < 0.0)
					fprintf(f, "%d,%d,%d,%f\n", x, y, z, (*V)[x][y][z]);

	fclose(f);
}

// Compute the potential of a planar polygonal electrode held at 1 V in an otherwise grounded plane (the gapless
// plane approximation), which is the solid angle subtended by the electrode divided by 2*PI (M. G. House,
// Phys. Rev. A 78, 033402 (2008); J. P. Oliveira and A. J. Miranda, Eur. J. Phys. 22, 31 (2001))
void evaluate_electrode_potential_analytical
(
	struct Electrode *electrode,
	double electrode_position[3],
	double position[3],
	double *phi
)
{
	double solid_angle = 0.0;

	// Triangulate the polygon as a fan from its first vertex; the signed solid angles of the triangles add up to
	// the solid angle of the polygon even if it is not convex
	double r_1[3];
	for (int k = 0; k < 3; k++)
		r_1[k] = electrode_position[k] + electrode->vertices[0][k] - position[k];

	double r_1_norm = sqrt(r_1[0]*r_1[0] + r_1[1]*r_1[1] + r_1[2]*r_1[2]);

	for (int v = 1; v < electrode->n_vertices - 1; v++)
	{
		double r_2[3];
		double r_3[3];
		for (int k = 0; k < 3; k++)
		{
			r_2[k] = electrode_position[k] + electrode->vertices[v][k] - position[k];
			r_3[k] = electrode_position[k] + electrode->vertices[v+1][k] - position[k];
		}

		double r_2_norm = sqrt(r_2[0]*r_2[0] + r_2[1]*r_2[1] + r_2[2]*r_2[2]);
		double r_3_norm = sqrt(r_3[0]*r_3[0] + r_3[1]*r_3[1] + r_3[2]*r_3[2]);

		double r_2_cross_r_3[3];
		cross(&r_2, &r_3, &r_2_cross_r_3);

		double r_1_dot_r_2;
		double r_1_dot_r_3;
		double r_2_dot_r_3;
		double triple_product;
		dot(3, &r_1, &r_2, &r_1_dot_r_2);
		dot(3, &r_1, &r_3, &r_1_dot_r_3);
		dot(3, &r_2, &r_3, &r_2_dot_r_3);
		dot(3, &r_1, &r_2_cross_r_3, &triple_product);

		// Solid angle of a triangle by A. Van Oosterom and J. Strackee, IEEE Trans. Biomed. Eng. 30, 125 (1983)
		double denominator = r_1_norm * r_2_norm * r_3_norm + r_1_dot_r_2 * r_3_norm + r_1_dot_r_3 * r_2_norm + r_2_dot_r_3 * r_1_norm;

		solid_angle += 2 * atan2(triple_product, denominator);
	}

	*phi = fabs(solid_angle) / (2 * PI);
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
		// Here, phi is the colatitude, sampled from the north pole down to (but excluding) the south pole
		double phi = (double)i/NLAT * PI;

		// Precompute trigonometric quantities to reduce cost
		double R_sin_phi = SPH_R * sin(phi);
		double R_cos_phi = SPH_R * cos(phi);

		for (int j = 0; j < NLON; j++) {
			// Here, theta is the longitude, which spans the full circle
			double theta = (double)j/NLON * 2 * PI;

			double sample_coord[3] = { x_c + R_sin_phi * cos(theta), y_c + R_sin_phi * sin(theta), z_c + R_cos_phi };

			grid[i][j] = 0.0;

			interpolate_3d(
				NX, NY, NZ,
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

// Sample the analytical potential of an electrode on the Driscoll-Healy grid of the sphere centered on
// (x_c, y_c, z_c)
void sample_dh1_analytical
(
	struct Electrode *electrode,
	double electrode_position[3],
	double x_c,
	double y_c,
	double z_c,
	double grid[NLAT][NLON]
)
{
	for (int i = 0; i < NLAT; i++)
	{
		// Here, phi is the colatitude and theta is the longitude, as in sample_dh1
		double phi = (double)i/NLAT * PI;

		double R_sin_phi = SPH_R * sin(phi);
		double R_cos_phi = SPH_R * cos(phi);

		for (int j = 0; j < NLON; j++)
		{
			double theta = (double)j/NLON * 2 * PI;

			double sample_coord[3] = { x_c + R_sin_phi * cos(theta), y_c + R_sin_phi * sin(theta), z_c + R_cos_phi };

			evaluate_electrode_potential_analytical(
				electrode,
				electrode_position,
				sample_coord,
				&grid[i][j]
			);
		}
	}
}

// Determine the factor which converts the coefficients produced by compute_shcoeffs_real into 4π-normalized
// coefficients, since the wrapper may call SHExpandDH with either the 4π (norm = 1) or orthonormal (norm = 4) convention
static double sh_wrapper_normalization
(
)
{
	static double normalization = 0.0;

	if (normalization == 0.0)
	{
		int nlat = NLAT;
		int nlon = NLON;
		int lmax = LMAX;

		// A constant function has C_00 = 1 with 4π-normalized harmonics and C_00 = sqrt(4π) with orthonormal harmonics
		double grid_fortran[NLON][NLAT];
		for (int j = 0; j < NLON; j++)
			for (int i = 0; i < NLAT; i++)
				grid_fortran[j][i] = 1.0;

		double alm[VLM_SPHERE_LEN];
		compute_shcoeffs_real((double*) grid_fortran, &nlat, &nlon, &lmax, alm);

		if (fabs(alm[VLM_INDEX(0, 0, 0)] - 1.0) < 1E-9)
			normalization = 1.0;
		else if (fabs(alm[VLM_INDEX(0, 0, 0)] - sqrt(4 * PI)) < 1E-9)
			normalization = 1.0 / sqrt(4 * PI);
		else
		{
			printf("ERROR: compute_shcoeffs_real uses an unsupported normalization (C_00 = %f for a constant function)\n", alm[VLM_INDEX(0, 0, 0)]);
			normalization = 1.0;
		}
	}

	return normalization;
}

// The expansion is linear in the sampled grid, so it is a matrix, which is built once from SHTOOLS (by expanding a grid
// with a single nonzero sample at a time) and then applied without calling SHTOOLS, so that expansions can run in
// parallel without SHTOOLS's per-call FFT planning. Only the rows of coefficients with m <= l are stored
static double *sh_expansion_matrix = NULL;
static int sh_expansion_rows[VLM_SPHERE_LEN];
static int n_sh_expansion_rows = 0;
static int sh_expansion_matrix_ready = 0;

static void build_sh_expansion_matrix
(
)
{
	int nlat = NLAT;
	int nlon = NLON;

	for (int l = 0; l <= LMAX; l++)
		for (int m = 0; m <= l; m++)
			for (int c = 0; c < 2; c++)
				if (c == 0 || m > 0)
					sh_expansion_rows[n_sh_expansion_rows++] = VLM_INDEX(c, l, m);

	sh_expansion_matrix = calloc((size_t) n_sh_expansion_rows * NLAT * NLON, sizeof(double));

	double normalization = sh_wrapper_normalization();

	double grid_fortran[NLON][NLAT];
	double alm[VLM_SPHERE_LEN];
	for (int i = 0; i < NLAT; i++)
		for (int j = 0; j < NLON; j++)
		{
			memset(grid_fortran, 0, sizeof(grid_fortran));
			grid_fortran[j][i] = 1.0;

			int lmax = LMAX;
			compute_shcoeffs_real((double*) grid_fortran, &nlat, &nlon, &lmax, alm);

			for (int r = 0; r < n_sh_expansion_rows; r++)
				sh_expansion_matrix[(size_t) r * NLAT * NLON + i * NLON + j] = normalization * alm[sh_expansion_rows[r]];
		}
}

// @TODO - generalize syntax and move to ti_utils.c for generality
void expand_spherical_harmonics
(
	double grid[NLAT][NLON],
	double *alm
)
{
	// The coefficients are computed by SHTOOLS's compute_shcoeffs_real (which produces real coefficients without the
	// Condon-Shortley phase, transposing the grid since Fortran arrays are column-major) through the expansion matrix
	// @TODO - get the complex subroutine working
	// compute_shcoeffs_cmplx((double*) grid, &nlat, &nlon, &lmax, alm);
	int ready;
	#pragma omp atomic read
	ready = sh_expansion_matrix_ready;

	if (!ready)
	{
		#pragma omp critical (sh_expansion_matrix)
		{
			if (!sh_expansion_matrix_ready)
			{
				build_sh_expansion_matrix();

				#pragma omp atomic write
				sh_expansion_matrix_ready = 1;
			}
		}
	}

	// Store 4π-normalized coefficients regardless of the convention used by the wrapper
	memset(alm, 0, VLM_SPHERE_LEN * sizeof(double));

	double *samples = &grid[0][0];
	for (int r = 0; r < n_sh_expansion_rows; r++)
	{
		const double *row = &sh_expansion_matrix[(size_t) r * NLAT * NLON];

		double sum = 0.0;
		for (int i = 0; i < NLAT * NLON; i++)
			sum += row[i] * samples[i];

		alm[sh_expansion_rows[r]] = sum;
	}

	// @TEST - print non-zero coefficients
	// for (int l = 0; l <= LMAX; l++)
	// 	for (int m = 0; m <= l; m++)
	// 	{
	// 		double C_lm = alm[VLM_INDEX(0, l, m)];
	// 		double S_lm = alm[VLM_INDEX(1, l, m)];
	//
	// 		if (fabs(C_lm) > 1E-6 || fabs(S_lm) > 1E-6)
	// 			printf("C_lm[%d,%d] = %f, S_lm[%d,%d] = %f\n", l, m, C_lm, l, m, S_lm);
	// 	}
}

// Compute the center of sphere (sx, sy, sz) in the trap frame
void compute_sphere_center
(
	int sx,
	int sy,
	int sz,
	double (*center)[3]
)
{
	(*center)[0] = SPH_CENTER_X(sx);
	(*center)[1] = SPH_CENTER_Y(sy);
	(*center)[2] = SPH_CENTER_Z(sz);
}

// Find the sphere whose center is nearest to a position in the trap frame; returns 1 if the position lies outside
// of that sphere
int find_nearest_sphere
(
	double position[3],
	int *sx,
	int *sy,
	int *sz
)
{
	// The sphere centers form a cubic lattice, so the nearest center can be found one axis at a time
	*sx = (int) round((position[0] - SPH_CENTER_X(0)) / SPH_SPACING);
	*sy = (int) round((position[1] - SPH_CENTER_Y(0)) / SPH_SPACING);
	*sz = (int) round((position[2] - SPH_CENTER_Z(0)) / SPH_SPACING);

	*sx = MIN(MAX(*sx, 0), NSPH_X - 1);
	*sy = MIN(MAX(*sy, 0), NSPH_Y - 1);
	*sz = MIN(MAX(*sz, 0), NSPH_Z - 1);

	double center[3];
	compute_sphere_center(*sx, *sy, *sz, &center);

	double r_sq = 0.0;
	for (int k = 0; k < 3; k++)
		r_sq += pow(position[k] - center[k], 2);

	return r_sq > SPH_R * SPH_R;
}

// Index of R_l^m in arrays of regular solid harmonics
static inline int solid_harmonic_index
(
	int l,
	int m
)
{
	return l*(l+1)/2 + m;
}

// Normalization of the 4π-normalized real spherical harmonics used by SHTOOLS (norm = 1, csphase = 1), relating
// them to the solid harmonics by r^l Y_lm = N_lm Re(R_l^m) and r^l Y_l(-m) = N_lm Im(R_l^m)
static double sh_normalizations[SOLID_HARMONICS_LEN];

// Computed once when the library is loaded, before any threads can use it
__attribute__((constructor))
static void compute_sh_normalizations
(
)
{
	for (int l = 0; l <= LMAX; l++)
		for (int m = 0; m <= l; m++)
		{
			// (l - m)! / (l + m)!
			double factorial_ratio = 1.0;
			for (int k = l - m + 1; k <= l + m; k++)
				factorial_ratio /= k;

			sh_normalizations[solid_harmonic_index(l, m)] = sqrt((m == 0 ? 1.0 : 2.0) * (2*l + 1) * factorial_ratio);
		}
}

// Components of the symmetric Hessians stored by the expansions: xx, xy, xz, yy, yz and zz
static const int hessian_k[6] = { 0, 0, 0, 1, 1, 2 };
static const int hessian_j[6] = { 0, 1, 2, 1, 2, 2 };

// A regular solid harmonic R_l^m(u) = |u|^l P_l^m(cos(theta)) e^(i m phi), without the Condon-Shortley phase, with its
// gradient and the unique components of its Hessian
struct SolidHarmonic
{
	double complex value;
	double complex grad[3];
	double complex hess[6];
};

// c w B by the product rule, where w = x + i y has gradient (1, i, 0) and no curvature
static inline void multiply_w
(
	double c,
	double complex w,
	struct SolidHarmonic *B,
	int order,
	struct SolidHarmonic *A
)
{
	A->value = c * w * B->value;

	if (order >= 1)
	{
		A->grad[0] = c * (B->value + w * B->grad[0]);
		A->grad[1] = c * (I * B->value + w * B->grad[1]);
		A->grad[2] = c * w * B->grad[2];
	}

	if (order >= 2)
	{
		A->hess[0] = c * (2 * B->grad[0] + w * B->hess[0]);
		A->hess[1] = c * (B->grad[1] + I * B->grad[0] + w * B->hess[1]);
		A->hess[2] = c * (B->grad[2] + w * B->hess[2]);
		A->hess[3] = c * (2 * I * B->grad[1] + w * B->hess[3]);
		A->hess[4] = c * (I * B->grad[2] + w * B->hess[4]);
		A->hess[5] = c * w * B->hess[5];
	}
}

// c z B by the product rule, where z has gradient (0, 0, 1) and no curvature
static inline void multiply_z
(
	double c,
	double z,
	struct SolidHarmonic *B,
	int order,
	struct SolidHarmonic *A
)
{
	A->value = c * z * B->value;

	if (order >= 1)
	{
		A->grad[0] = c * z * B->grad[0];
		A->grad[1] = c * z * B->grad[1];
		A->grad[2] = c * (B->value + z * B->grad[2]);
	}

	if (order >= 2)
	{
		A->hess[0] = c * z * B->hess[0];
		A->hess[1] = c * z * B->hess[1];
		A->hess[2] = c * (B->grad[0] + z * B->hess[2]);
		A->hess[3] = c * z * B->hess[3];
		A->hess[4] = c * (B->grad[1] + z * B->hess[4]);
		A->hess[5] = c * (2 * B->grad[2] + z * B->hess[5]);
	}
}

// A - c |u|^2 B by the product rule, where |u|^2 has gradient 2u and curvature 2 in every direction
static inline void subtract_r_sq
(
	double c,
	double u[3],
	double r_sq,
	struct SolidHarmonic *B,
	int order,
	struct SolidHarmonic *A
)
{
	A->value -= c * r_sq * B->value;

	if (order >= 1)
		for (int k = 0; k < 3; k++)
			A->grad[k] -= c * (2 * u[k] * B->value + r_sq * B->grad[k]);

	if (order >= 2)
		for (int h = 0; h < 6; h++)
		{
			int k = hessian_k[h];
			int j = hessian_j[h];

			A->hess[h] -= c * ((k == j ? 2.0 : 0.0) * B->value + 2 * u[k] * B->grad[j] + 2 * u[j] * B->grad[k] + r_sq * B->hess[h]);
		}
}

// Accumulate weighted sums of the regular solid harmonics at u, along with their gradients (order >= 1) and Hessians
// (order >= 2), where c and s hold the cosine and sine coefficients of each sum (n_sets-by-SOLID_HARMONICS_LEN, with
// the normalizations included). The harmonics are generated by recurrence relations which are polynomial in Cartesian
// coordinates, so there are no singularities at the origin or on the z-axis, one order m at a time so that only the
// last two degrees need to be kept
static void accumulate_solid_harmonics
(
	double u[3],
	int order,
	int n_sets,
	double *c,
	double *s,
	double *phi,
	double (*grad)[3],
	double (*hess)[6]
)
{
	double complex w = u[0] + I * u[1];
	double z = u[2];
	double r_sq = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];

	for (int set = 0; set < n_sets; set++)
	{
		phi[set] = 0.0;

		if (order >= 1)
			for (int k = 0; k < 3; k++)
				grad[set][k] = 0.0;

		if (order >= 2)
			for (int h = 0; h < 6; h++)
				hess[set][h] = 0.0;
	}

	// R_m^m, and three rotating slots for the last two degrees and the current one
	struct SolidHarmonic diagonal[2] = { { 0 } };
	diagonal[0].value = 1.0;
	int d = 0;

	struct SolidHarmonic slots[3];

	for (int m = 0; m <= LMAX; m++)
	{
		// R_m^m = (2m - 1) w R_(m-1)^(m-1)
		if (m > 0)
		{
			multiply_w(2*m - 1, w, &diagonal[d], order, &diagonal[1 - d]);
			d = 1 - d;
		}

		struct SolidHarmonic *previous_2 = NULL;
		struct SolidHarmonic *previous_1 = NULL;

		for (int l = m; l <= LMAX; l++)
		{
			struct SolidHarmonic *current = &slots[(l - m) % 3];

			if (l == m)
			{
				current = &diagonal[d];
			}
			else if (l == m + 1)
			{
				// R_(m+1)^m = (2m + 1) z R_m^m
				multiply_z(2*m + 1, z, &diagonal[d], order, current);
			}
			else
			{
				// (l - m) R_l^m = (2l - 1) z R_(l-1)^m - (l + m - 1) |u|^2 R_(l-2)^m
				multiply_z((2*l - 1.0) / (l - m), z, previous_1, order, current);
				subtract_r_sq((l + m - 1.0) / (l - m), u, r_sq, previous_2, order, current);
			}

			int idx = solid_harmonic_index(l, m);

			for (int set = 0; set < n_sets; set++)
			{
				double c_lm = c[set * SOLID_HARMONICS_LEN + idx];
				double s_lm = s[set * SOLID_HARMONICS_LEN + idx];

				if (c_lm == 0.0 && s_lm == 0.0)
					continue;

				phi[set] += c_lm * creal(current->value) + s_lm * cimag(current->value);

				if (order >= 1)
					for (int k = 0; k < 3; k++)
						grad[set][k] += c_lm * creal(current->grad[k]) + s_lm * cimag(current->grad[k]);

				if (order >= 2)
					for (int h = 0; h < 6; h++)
						hess[set][h] += c_lm * creal(current->hess[h]) + s_lm * cimag(current->hess[h]);
			}

			previous_2 = previous_1;
			previous_1 = current;
		}
	}
}

// Evaluate several weighted sums of electrode potentials at once, along with their gradients and Hessians if requested
// (any output may be NULL), using the spherical harmonics expansions within the sphere nearest to the position and
// sharing the solid harmonics between the sums; weights has length n_sets-by-n_electrodes and the outputs have length
// n_sets. Returns 1 if the position lies outside of the sphere, in which case the expansions are extrapolated
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
)
{
	int order = (H != NULL) ? 2 : ((grad_phi != NULL) ? 1 : 0);

	int sx;
	int sy;
	int sz;
	int outside = find_nearest_sphere(position, &sx, &sy, &sz);

	double center[3];
	compute_sphere_center(sx, sy, sz, &center);

	// Work in coordinates relative to the sphere center, scaled by the sphere radius
	double u[3];
	for (int k = 0; k < 3; k++)
		u[k] = (position[k] - center[k]) / SPH_R;

	// Combine the coefficients of the electrodes for each sum, including the normalizations of the harmonics
	double c[n_sets * SOLID_HARMONICS_LEN];
	double s[n_sets * SOLID_HARMONICS_LEN];
	memset(c, 0, sizeof(c));
	memset(s, 0, sizeof(s));

	for (int set = 0; set < n_sets; set++)
		for (int e = 0; e < n_electrodes; e++)
		{
			double weight = weights[set * n_electrodes + e];

			if (weight == 0.0)
				continue;

			double *alm = VLM_SLICE((*electrodes)[e].Vlm, sx, sy, sz);

			for (int l = 0; l <= LMAX; l++)
				for (int m = 0; m <= l; m++)
				{
					int idx = solid_harmonic_index(l, m);

					c[set * SOLID_HARMONICS_LEN + idx] += weight * sh_normalizations[idx] * alm[VLM_INDEX(0, l, m)];
					s[set * SOLID_HARMONICS_LEN + idx] += weight * sh_normalizations[idx] * alm[VLM_INDEX(1, l, m)];
				}
		}

	double phi_sums[n_sets];
	double grad_sums[n_sets][3];
	double hess_sums[n_sets][6];
	accumulate_solid_harmonics(u, order, n_sets, c, s, phi_sums, grad_sums, hess_sums);

	// Undo the scaling of the coordinates
	for (int set = 0; set < n_sets; set++)
	{
		if (phi != NULL)
			phi[set] = phi_sums[set];

		if (grad_phi != NULL)
			for (int k = 0; k < 3; k++)
				grad_phi[set][k] = grad_sums[set][k] / SPH_R;

		if (H != NULL)
			for (int h = 0; h < 6; h++)
			{
				H[set][hessian_k[h]][hessian_j[h]] = hess_sums[set][h] / (SPH_R * SPH_R);
				H[set][hessian_j[h]][hessian_k[h]] = hess_sums[set][h] / (SPH_R * SPH_R);
			}
	}

	return outside;
}

// Evaluate a weighted sum of electrode potentials, along with its gradient and Hessian if requested (any output may
// be NULL); see evaluate_expansion_sets
int evaluate_expansion
(
	int n_electrodes,
	struct Electrode (*electrodes)[],
	double *weights,
	double position[3],
	double *phi,
	double grad_phi[3],
	double H[3][3]
)
{
	return evaluate_expansion_sets(
		n_electrodes,
		electrodes,
		1,
		weights,
		position,
		phi,
		(double (*)[3]) grad_phi,
		(double (*)[3][3]) H
	);
}

// Evaluate the potential of every electrode at 1 V, with its gradient and Hessian if requested (any output may be NULL),
// sharing the solid harmonics between electrodes; the outputs have length n_electrodes
int evaluate_electrode_derivatives_all
(
	struct Trap *trap,
	double position[3],
	double *phi,
	double (*grad_phi)[3],
	double (*H)[3][3]
)
{
	int n_e = trap->n_electrodes;

	double *weights = calloc(n_e * n_e, sizeof(double));
	for (int e = 0; e < n_e; e++)
		weights[e * n_e + e] = 1.0;

	int outside = evaluate_expansion_sets(n_e, trap->electrodes, n_e, weights, position, phi, grad_phi, H);

	free(weights);

	return outside;
}

// Gather the expansion coefficients of every electrode into one array of length n_electrodes-by-n_spheres-by-
// SOLID_HARMONICS_LEN-by-2 (cosine and sine), with the normalizations of the harmonics included, as used by the OpenCL
// kernels
void pack_expansion_coefficients
(
	struct Trap *trap,
	double *coefficients
)
{
	int n_spheres = NSPH_X * NSPH_Y * NSPH_Z;

	#pragma omp parallel for collapse(2)
	for (int e = 0; e < trap->n_electrodes; e++)
		for (int sphere = 0; sphere < n_spheres; sphere++)
		{
			double *alm = &(*(*trap->electrodes)[e].Vlm)[sphere * VLM_SPHERE_LEN];
			double *packed = &coefficients[((size_t) e * n_spheres + sphere) * SOLID_HARMONICS_LEN * 2];

			for (int l = 0; l <= LMAX; l++)
				for (int m = 0; m <= l; m++)
				{
					int idx = solid_harmonic_index(l, m);

					packed[2 * idx] = sh_normalizations[idx] * alm[VLM_INDEX(0, l, m)];
					packed[2 * idx + 1] = sh_normalizations[idx] * alm[VLM_INDEX(1, l, m)];
				}
		}
}

// Evaluate the potential of an electrode at 1 V and its gradient and Hessian from its spherical harmonics
// expansion, skipping any outputs which are NULL
int evaluate_electrode_derivatives
(
	struct Electrode *electrode,
	double position[3],
	double *phi,
	double grad_phi[3],
	double H[3][3]
)
{
	double weight = 1.0;

	return evaluate_expansion(1, (struct Electrode (*)[]) electrode, &weight, position, phi, grad_phi, H);
}

// Evaluate the potential of an electrode at 1 V from its spherical harmonics expansion
int evaluate_electrode_potential
(
	struct Electrode *electrode,
	double position[3],
	double *phi
)
{
	return evaluate_electrode_derivatives(electrode, position, phi, NULL, NULL);
}

// Evaluate the gradient of the potential of an electrode at 1 V from its spherical harmonics expansion
int evaluate_electrode_gradient
(
	struct Electrode *electrode,
	double position[3],
	double grad_phi[3]
)
{
	return evaluate_electrode_derivatives(electrode, position, NULL, grad_phi, NULL);
}

// Evaluate the Hessian of the potential of an electrode at 1 V from its spherical harmonics expansion
int evaluate_electrode_hessian_analytical
(
	struct Electrode *electrode,
	double position[3],
	double H[3][3]
)
{
	return evaluate_electrode_derivatives(electrode, position, NULL, NULL, H);
}

// Evaluate the Hessian of the potential of an electrode at 1 V by central differences of its gradient
void evaluate_electrode_hessian_finite_differences
(
	struct Electrode *electrode,
	double position[3],
	double H[3][3]
)
{
	// @TODO - factor this out
	double h = 1E-5;

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			H[i][j] = 0;

	for (int j = 0; j < 3; j++)
	{
		double p1[3] = { position[0], position[1], position[2] };
		double p2[3] = { position[0], position[1], position[2] };

		p1[j] += h;
		p2[j] -= h;

		double g1[3];
		double g2[3];

		evaluate_electrode_gradient(electrode, p1, g1);
		evaluate_electrode_gradient(electrode, p2, g2);

		for (int i = 0; i < 3; i++)
			H[i][j] = (g1[i] - g2[i]) / (2 * h);
	}
}

// Evaluate the electric potential (V) when the given voltages are applied to the electrodes, along with its gradient
// (V/µm) and Hessian (V/µm^2); voltages has length n_electrodes and any output may be NULL
int evaluate_trap_potential
(
	struct Trap *trap,
	double *voltages,
	double position[3],
	double *phi,
	double grad_phi[3],
	double H[3][3]
)
{
	return evaluate_expansion(trap->n_electrodes, trap->electrodes, voltages, position, phi, grad_phi, H);
}

// Prefactor (J µm^2) relating the pseudopotential to the squared gradient of the RF potential per volt (1/µm^2):
// U_pond = q^2 |E_RF|^2 / (4 m Omega_RF^2), where E_RF = -V_RF grad(phi_RF); 0 without an RF drive
static double pseudopotential_prefactor
(
	struct Trap *trap,
	const double m,
	const double q
)
{
	if (trap->n_electrodes_rf == 0 || trap->V_RF == 0.0 || trap->Omega_RF == 0.0)
		return 0.0;

	return pow(q * trap->V_RF, 2) / (4 * m * pow(trap->Omega_RF * LENGTH_SCALE, 2));
}

// Pseudopotential gradient 2 c (grad(phi_RF) . H_RF) from the RF potential's gradient and Hessian
static void pseudopotential_gradient
(
	double prefactor,
	double g[3],
	double H_rf[3][3],
	double grad_U_pond[3]
)
{
	for (int i = 0; i < 3; i++)
	{
		grad_U_pond[i] = 0.0;

		for (int k = 0; k < 3; k++)
			grad_U_pond[i] += 2 * prefactor * g[k] * H_rf[k][i];
	}
}

// Evaluate the ponderomotive pseudopotential (J) of the RF drive for an ion of mass m (kg) and charge q (C), along
// with its gradient (J/µm) and Hessian (J/µm^2); any output may be NULL
int evaluate_pseudopotential
(
	struct Trap *trap,
	const double m,
	const double q,
	double position[3],
	double *U_pond,
	double grad_U_pond[3],
	double H_U_pond[3][3]
)
{
	if (U_pond != NULL)
		*U_pond = 0.0;

	if (grad_U_pond != NULL)
		for (int k = 0; k < 3; k++)
			grad_U_pond[k] = 0.0;

	if (H_U_pond != NULL)
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				H_U_pond[i][j] = 0.0;

	double prefactor = pseudopotential_prefactor(trap, m, q);

	if (prefactor == 0.0)
	{
		int sx;
		int sy;
		int sz;

		return find_nearest_sphere(position, &sx, &sy, &sz);
	}

	// Every RF electrode is driven in phase with the same amplitude
	double rf_weights[trap->n_electrodes];
	for (int e = 0; e < trap->n_electrodes; e++)
		rf_weights[e] = (e < trap->n_electrodes_rf) ? 1.0 : 0.0;

	double g[3];
	double H_rf[3][3];
	int outside = evaluate_expansion(trap->n_electrodes, trap->electrodes, rf_weights, position, NULL, g, H_rf);

	if (U_pond != NULL)
		*U_pond = prefactor * (g[0]*g[0] + g[1]*g[1] + g[2]*g[2]);

	if (grad_U_pond != NULL)
		pseudopotential_gradient(prefactor, g, H_rf, grad_U_pond);

	if (H_U_pond != NULL)
	{
		// The Hessian needs third derivatives of phi_RF, so differentiate the analytical gradient numerically
		double h = 1E-3;

		for (int j = 0; j < 3; j++)
		{
			double p_plus[3] = { position[0], position[1], position[2] };
			double p_minus[3] = { position[0], position[1], position[2] };

			p_plus[j] += h;
			p_minus[j] -= h;

			double g_plus[3];
			double g_minus[3];
			double H_plus[3][3];
			double H_minus[3][3];
			evaluate_expansion(trap->n_electrodes, trap->electrodes, rf_weights, p_plus, NULL, g_plus, H_plus);
			evaluate_expansion(trap->n_electrodes, trap->electrodes, rf_weights, p_minus, NULL, g_minus, H_minus);

			double grad_plus[3];
			double grad_minus[3];
			pseudopotential_gradient(prefactor, g_plus, H_plus, grad_plus);
			pseudopotential_gradient(prefactor, g_minus, H_minus, grad_minus);

			for (int i = 0; i < 3; i++)
				H_U_pond[i][j] = (grad_plus[i] - grad_minus[i]) / (2 * h);
		}

		for (int i = 0; i < 3; i++)
			for (int j = i+1; j < 3; j++)
			{
				double H_ij = 0.5 * (H_U_pond[i][j] + H_U_pond[j][i]);
				H_U_pond[i][j] = H_ij;
				H_U_pond[j][i] = H_ij;
			}
	}

	return outside;
}

// Evaluate the total potential energy (J) of an ion of mass m (kg) and charge q (C), which is the sum of its
// electrostatic potential energy in the DC potential and the RF pseudopotential, along with its gradient (J/µm) and
// Hessian (J/µm^2); any output may be NULL
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
)
{
	int n_e = trap->n_electrodes;
	double prefactor = pseudopotential_prefactor(trap, m, q);

	// The DC potential and (for the pseudopotential) the RF potential share their solid harmonics
	int n_sets = (prefactor != 0.0) ? 2 : 1;
	double weights[2 * n_e];
	for (int e = 0; e < n_e; e++)
	{
		weights[e] = voltages[e];
		weights[n_e + e] = (e < trap->n_electrodes_rf) ? 1.0 : 0.0;
	}

	// The pseudopotential needs the gradient of the RF potential, and the pseudopotential gradient its Hessian
	int need_hessian = (H_U != NULL) || (grad_U != NULL && n_sets == 2);
	int need_gradient = (grad_U != NULL) || need_hessian || (U != NULL && n_sets == 2);

	double phi[2];
	double grad_phi[2][3];
	double H_phi[2][3][3];
	int outside = evaluate_expansion_sets(n_e, trap->electrodes, n_sets, weights, position, phi, need_gradient ? grad_phi : NULL, need_hessian ? H_phi : NULL);

	if (U != NULL)
	{
		*U = q * phi[0];

		if (n_sets == 2)
			*U += prefactor * (grad_phi[1][0]*grad_phi[1][0] + grad_phi[1][1]*grad_phi[1][1] + grad_phi[1][2]*grad_phi[1][2]);
	}

	if (grad_U != NULL)
	{
		double grad_U_pond[3] = { 0.0, 0.0, 0.0 };
		if (n_sets == 2)
			pseudopotential_gradient(prefactor, grad_phi[1], H_phi[1], grad_U_pond);

		for (int k = 0; k < 3; k++)
			grad_U[k] = q * grad_phi[0][k] + grad_U_pond[k];
	}

	if (H_U != NULL)
	{
		double H_U_pond[3][3];
		evaluate_pseudopotential(trap, m, q, position, NULL, NULL, H_U_pond);

		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				H_U[i][j] = q * H_phi[0][i][j] + H_U_pond[i][j];
	}

	return outside;
}

// Compute the secular frequencies (rad/s) and principal axes (rows of axes) of an ion at a position from the
// curvature of its potential energy; directions with negative curvature are flagged by negative frequencies
int compute_normal_modes
(
	struct Trap *trap,
	const double m,
	const double q,
	double *voltages,
	double position[3],
	double (*omega)[3],
	double (*axes)[3][3]
)
{
	double H_U[3][3];
	int outside = evaluate_potential_energy(trap, m, q, voltages, position, NULL, NULL, H_U);

	// Convert the curvature from J/µm^2 to N/m
	double K[3][3];
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			K[i][j] = H_U[i][j] / (LENGTH_SCALE * LENGTH_SCALE);

	double k[3];
	diagonalize_matrix_3d(&K, &k, axes);

	for (int i = 0; i < 3; i++)
		(*omega)[i] = (k[i] >= 0) ? sqrt(k[i] / m) : -sqrt(-k[i] / m);

	return outside;
}

// Find the stationary point of the potential energy nearest to the given position by Newton's method, updating the
// position in place; returns 0 on convergence
int find_potential_minimum
(
	struct Trap *trap,
	const double m,
	const double q,
	double *voltages,
	double (*position)[3],
	int max_iterations
)
{
	// Largest step allowed in each iteration (µm)
	double max_step = 2.0;

	for (int it = 0; it < max_iterations; it++)
	{
		double grad_U[3];
		double H_U[3][3];
		evaluate_potential_energy(trap, m, q, voltages, *position, NULL, grad_U, H_U);

		// Solve H_U step = -grad_U by Cramer's rule
		double det = (
			H_U[0][0] * (H_U[1][1] * H_U[2][2] - H_U[1][2] * H_U[2][1]) -
			H_U[0][1] * (H_U[1][0] * H_U[2][2] - H_U[1][2] * H_U[2][0]) +
			H_U[0][2] * (H_U[1][0] * H_U[2][1] - H_U[1][1] * H_U[2][0])
		);

		if (fabs(det) < 1E-300)
		{
			printf("ERROR: Potential energy has a singular Hessian at (%f, %f, %f)\n", (*position)[0], (*position)[1], (*position)[2]);
			return 1;
		}

		double step[3];
		for (int k = 0; k < 3; k++)
		{
			double M[3][3];
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 3; j++)
					M[i][j] = (j == k) ? -grad_U[i] : H_U[i][j];

			step[k] = (
				M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
				M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
				M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
			) / det;
		}

		double step_norm = sqrt(step[0]*step[0] + step[1]*step[1] + step[2]*step[2]);
		double step_scale = (step_norm > max_step) ? max_step / step_norm : 1.0;

		for (int k = 0; k < 3; k++)
			(*position)[k] += step_scale * step[k];

		if (step_norm < 1E-9)
			return 0;
	}

	return 1;
}

// Find the point of the RF null line (the minimum of the pseudopotential in the yz-plane) at the x-coordinate of a
// position by Newton's method, updating its y and z coordinates in place; returns 0 on convergence
int find_rf_null
(
	struct Trap *trap,
	double (*position)[3],
	int max_iterations
)
{
	if (trap->n_electrodes_rf == 0)
	{
		printf("ERROR: Trap has no RF electrodes, so it has no RF null\n");
		return 1;
	}

	// The location of the null does not depend on the drive or on the ion, so use unit values to keep numbers moderate
	struct Trap rf_trap = *trap;
	rf_trap.V_RF = 1.0;
	rf_trap.Omega_RF = 1.0 / LENGTH_SCALE;

	// Largest step allowed in each iteration (µm)
	double max_step = 2.0;

	for (int it = 0; it < max_iterations; it++)
	{
		double grad_U[3];
		double H_U[3][3];
		evaluate_pseudopotential(&rf_trap, 1.0, 1.0, *position, NULL, grad_U, H_U);

		double det = H_U[1][1] * H_U[2][2] - H_U[1][2] * H_U[2][1];

		if (fabs(det) < 1E-300)
		{
			printf("ERROR: Pseudopotential has a singular Hessian at (%f, %f, %f)\n", (*position)[0], (*position)[1], (*position)[2]);
			return 1;
		}

		double step_y = -(H_U[2][2] * grad_U[1] - H_U[1][2] * grad_U[2]) / det;
		double step_z = -(H_U[1][1] * grad_U[2] - H_U[2][1] * grad_U[1]) / det;

		double step_norm = sqrt(step_y*step_y + step_z*step_z);
		double step_scale = (step_norm > max_step) ? max_step / step_norm : 1.0;

		(*position)[1] += step_scale * step_y;
		(*position)[2] += step_scale * step_z;

		if (step_norm < 1E-9)
			return 0;
	}

	return 1;
}
