#include "test_electrodynamics.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "electrodynamics.h"
#include "defs.h"
#include "data_structures.h"
#include "ti_utils.h"
#include "traps.h"
#include "ions.h"
#include "test_utils.h"

int test_all_electrodynamics
(
	double atol
)
{
	printf("- Testing electrodynamics module...\n");

	int result = 0;

	result += test_solver_relaxation(atol);
	result += test_solver_spectral(atol);
	result += test_sample_dh1(atol);
	result += test_expand_spherical_harmonics(atol);
	result += test_evaluate_electrode_potential_analytical(atol);
	result += test_evaluate_expansion(atol);
	result += test_evaluate_electrode_hessian(atol);
	result += test_expansion_accuracy(atol);
	result += test_pseudopotential(atol);

	return result;
}

// A 20 µm square electrode centered at the origin of the trap frame, as in trap.data
static double square_vertices[4][3] =
{
	{ 10.0, 10.0, 0.0 },
	{ -10.0, 10.0, 0.0 },
	{ -10.0, -10.0, 0.0 },
	{ 10.0, -10.0, 0.0 }
};

static struct Electrode make_square_electrode
(
)
{
	struct Electrode electrode;

	electrode.n_vertices = 4;
	electrode.vertices = square_vertices;
	electrode.n_edges = 0;
	electrode.edges = NULL;
	electrode.Vlm_len = NSPH_X*NSPH_Y*NSPH_Z*VLM_SPHERE_LEN;
	electrode.Vlm = calloc(electrode.Vlm_len, sizeof(double));

	return electrode;
}

int test_solver_relaxation
(
	double atol
)
{
	// A small grounded box, so that the test is fast; electrodes are centered in it as in the full relaxation grid
	const int NX = 81;
	const int NY = 81;
	const int NZ = 61;

	struct Electrode electrode = make_square_electrode();
	double electrode_position[3] = { 0.0, 0.0, 0.0 };

	// 1. tiopt result
	double (*V_solver_relaxation)[NX][NY][NZ] = calloc(NX * NY * NZ, sizeof(double));

	int dx = 1;
	int dy = 1;
	int dz = 1;

	int max_iterations = (int) 1E4;
	int iterations = solver_relaxation(
		NX, NY, NZ,
		&electrode,
		electrode_position,
		V_solver_relaxation,
		dx, dy, dz,
		max_iterations,
		1
	);

	// 2. Analytical result in the gapless plane approximation, which neglects the grounded walls of the box; above
	// the electrode and far from the walls, the two agree to within the discretization of the electrode on the grid
	double error = 0.0;
	for (int z = 2; z <= 10; z++)
		for (int x = -5; x <= 5; x++)
			for (int y = -5; y <= 5; y++)
			{
				double position[3] = { x, y, z };

				double V_analytical;
				evaluate_electrode_potential_analytical(&electrode, electrode_position, position, &V_analytical);

				error = MAX(error, fabs((*V_solver_relaxation)[x + NX/2][y + NY/2][z] - V_analytical) / V_analytical);
			}

	int result = check_error("solver_relaxation agrees with gapless plane solution", error, 2E-2);

	// The solution must satisfy the discretized Laplace equation everywhere off the electrode and the boundary
	double residual = 0.0;
	for (int x = 1; x < NX-1; x++)
		for (int y = 1; y < NY-1; y++)
			for (int z = 1; z < NZ-1; z++)
				residual = MAX(residual, fabs(
					(*V_solver_relaxation)[x-1][y][z] + (*V_solver_relaxation)[x+1][y][z] +
					(*V_solver_relaxation)[x][y-1][z] + (*V_solver_relaxation)[x][y+1][z] +
					(*V_solver_relaxation)[x][y][z-1] + (*V_solver_relaxation)[x][y][z+1] -
					6 * (*V_solver_relaxation)[x][y][z]
				));

	result += check_error("solver_relaxation solves discrete Laplace equation", residual, 1E-5);
	result += check_error("solver_relaxation converges before iteration limit", iterations, max_iterations - 1);

	free(electrode.Vlm);
	free(V_solver_relaxation);

	return result;
}

int test_solver_spectral
(
	double atol
)
{
	// The same small grounded box as in test_solver_relaxation
	const int NX = 81;
	const int NY = 81;
	const int NZ = 61;

	struct Electrode electrode = make_square_electrode();
	double electrode_position[3] = { 0.0, 0.0, 0.0 };

	// 1. Relaxation result
	double (*V_solver_relaxation)[NX][NY][NZ] = calloc(NX * NY * NZ, sizeof(double));
	solver_relaxation(
		NX, NY, NZ,
		&electrode,
		electrode_position,
		V_solver_relaxation,
		1, 1, 1,
		(int) 1E4,
		1
	);

	// 2. tiopt result, which solves the same discrete problem exactly
	double (*V_solver_spectral)[NX][NY][NZ] = calloc(NX * NY * NZ, sizeof(double));
	int status = solver_spectral(
		NX, NY, NZ,
		&electrode,
		electrode_position,
		V_solver_spectral
	);

	int result = check_error("solver_spectral handles electrode in trap surface", status, 0);

	double error = 0.0;
	double residual = 0.0;
	for (int x = 1; x < NX-1; x++)
		for (int y = 1; y < NY-1; y++)
			for (int z = 1; z < NZ-1; z++)
			{
				error = MAX(error, fabs((*V_solver_spectral)[x][y][z] - (*V_solver_relaxation)[x][y][z]));

				residual = MAX(residual, fabs(
					(*V_solver_spectral)[x-1][y][z] + (*V_solver_spectral)[x+1][y][z] +
					(*V_solver_spectral)[x][y-1][z] + (*V_solver_spectral)[x][y+1][z] +
					(*V_solver_spectral)[x][y][z-1] + (*V_solver_spectral)[x][y][z+1] -
					6 * (*V_solver_spectral)[x][y][z]
				));
			}

	// The relaxation solution is only converged to within RELAXATION_TOLERANCE per cell
	result += check_error("solver_spectral agrees with solver_relaxation", error, 1E-5);
	result += check_error("solver_spectral solves discrete Laplace equation", residual, 1E3 * atol);

	// Electrodes raised above the trap surface are not handled
	double raised_position[3] = { 0.0, 0.0, 5.0 };
	status = solver_spectral(
		NX, NY, NZ,
		&electrode,
		raised_position,
		V_solver_spectral
	);

	result += check_error("solver_spectral rejects raised electrode", status != 1, 0);

	free(electrode.Vlm);
	free(V_solver_relaxation);
	free(V_solver_spectral);

	return result;
}

// @TODO - generalize beyond the relaxation solver parameters
int test_sample_dh1
(
	double atol
)
{
	// The grid only needs to contain one sphere
	const int NX = 48;
	const int NY = 48;
	const int NZ = 48;

	double (*V)[NX][NY][NZ] = calloc(NX * NY * NZ, sizeof(double));

	double dx = 1.0;
	double dy = 1.0;
	double dz = 1.0;

	// Define V(x, y, z) = x + y + z, which trilinear interpolation reproduces exactly
	for (int x = 0; x < NX; x++)
		for (int y = 0; y < NY; y++)
			for (int z = 0; z < NZ; z++)
				(*V)[x][y][z] = x * dx + y * dy + z * dz;

	double x_c = NX/2;
	double y_c = NY/2;
	double z_c = NZ/2;

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
			double theta = (double)j/NLON * 2 * PI;

			double x = x_c + R_sin_phi * cos(theta);
			double y = y_c + R_sin_phi * sin(theta);
			double z = z_c + R_cos_phi;

			grid_analytical[i][j] = x + y + z;
		}
	}

	// 2. tiopt result
	double grid_sample_dh1[NLAT][NLON];

	sample_dh1(
		NX, NY, NZ,
		V,
		x_c, y_c, z_c,
		grid_sample_dh1,
		dx, dy, dz
	);

	double error = 0.0;
	for (int i = 0; i < NLAT; i++)
		for (int j = 0; j < NLON; j++)
			error += fabs(grid_sample_dh1[i][j] - grid_analytical[i][j]);

	int result = check_error("sample_dh1 samples linear function exactly", error, atol);

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

	// Define grid(phi, theta) = 2 * Y_0,0 + 4 * Y_1,-1 + 8 * Y_2,2 with 4π-normalized real spherical harmonics, where phi
	// is the colatitude and theta is the longitude
	for (int i = 0; i < NLAT; i++)
	{
		double phi = (double)i/NLAT * PI;

		for (int j = 0; j < NLON; j++)
		{
			double theta = (double)j/NLON * 2 * PI;

			grid[i][j] = 2 * 1.0 + 4 * sqrt(3) * sin(phi) * sin(theta) + 8 * sqrt(15) / 2 * pow(sin(phi), 2) * cos(2 * theta);
		}
	}

	// 1. Analytical result
	double alm_analytical[VLM_SPHERE_LEN] = { 0 };
	alm_analytical[VLM_INDEX(0, 0, 0)] = 2.0; // Y_0,0
	alm_analytical[VLM_INDEX(1, 1, 1)] = 4.0; // Y_1,-1 is the sine harmonic of degree 1 and order 1
	alm_analytical[VLM_INDEX(0, 2, 2)] = 8.0; // Y_2,2

	// 2. tiopt result
	double alm_expand_spherical_harmonics[VLM_SPHERE_LEN];
	expand_spherical_harmonics(
		grid,
		alm_expand_spherical_harmonics
	);

	double error = 0.0;
	for (int idx = 0; idx < VLM_SPHERE_LEN; idx++)
		error += fabs(alm_expand_spherical_harmonics[idx] - alm_analytical[idx]);

	return check_error("expand_spherical_harmonics recovers coefficients", error, atol);
}

int test_evaluate_electrode_potential_analytical
(
	double atol
)
{
	struct Electrode electrode = make_square_electrode();
	double electrode_position[3] = { 25.0, 0.0, 0.0 };

	// The solid angle of a square of side a seen from a height h above its center is 4 asin(a^2 / (a^2 + 4 h^2))
	double a = 20.0;
	double error = 0.0;
	for (double h = 1.0; h < 500.0; h *= 1.5)
	{
		double position[3] = { 25.0, 0.0, h };

		double phi;
		evaluate_electrode_potential_analytical(&electrode, electrode_position, position, &phi);

		error = MAX(error, fabs(phi - 4 * asin(a*a / (a*a + 4*h*h)) / (2 * PI)));
	}

	int result = check_error("analytical potential of square electrode", error, atol);

	// Gapless potentials superpose, so an L-shaped (non-convex) electrode is the sum of two rectangles
	struct Electrode l_electrode = electrode;
	double l_vertices[6][3] = { { 0, 0, 0 }, { 30, 0, 0 }, { 30, 10, 0 }, { 10, 10, 0 }, { 10, 30, 0 }, { 0, 30, 0 } };
	l_electrode.n_vertices = 6;
	l_electrode.vertices = l_vertices;

	struct Electrode rectangle_1 = electrode;
	double rectangle_1_vertices[4][3] = { { 0, 0, 0 }, { 30, 0, 0 }, { 30, 10, 0 }, { 0, 10, 0 } };
	rectangle_1.vertices = rectangle_1_vertices;

	struct Electrode rectangle_2 = electrode;
	double rectangle_2_vertices[4][3] = { { 0, 10, 0 }, { 10, 10, 0 }, { 10, 30, 0 }, { 0, 30, 0 } };
	rectangle_2.vertices = rectangle_2_vertices;

	double origin[3] = { 0.0, 0.0, 0.0 };
	double superposition_error = 0.0;
	for (double x = -20; x <= 50; x += 7)
		for (double y = -20; y <= 50; y += 7)
			for (double z = 1; z <= 60; z += 11)
			{
				double position[3] = { x, y, z };

				double phi_l;
				double phi_1;
				double phi_2;
				evaluate_electrode_potential_analytical(&l_electrode, origin, position, &phi_l);
				evaluate_electrode_potential_analytical(&rectangle_1, origin, position, &phi_1);
				evaluate_electrode_potential_analytical(&rectangle_2, origin, position, &phi_2);

				superposition_error = MAX(superposition_error, fabs(phi_l - phi_1 - phi_2));
			}

	result += check_error("analytical potential of non-convex electrode", superposition_error, atol);

	free(electrode.Vlm);

	return result;
}

struct HarmonicPolynomialData
{
	double origin[3];
};

// A harmonic polynomial of degree 4, which the expansion up to LMAX represents exactly in every sphere
static double harmonic_polynomial
(
	double position[3],
	void *f_data
)
{
	struct HarmonicPolynomialData *data = f_data;

	double x = position[0] - data->origin[0];
	double y = position[1] - data->origin[1];
	double z = position[2] - data->origin[2];

	return (
		0.3 + 0.02*x - 0.01*y + 0.05*z + 1E-3*(x*x - y*y) + 2E-3*x*z + 5E-4*(2*z*z - x*x - y*y)
		+ 1E-5*(x*x*x - 3*x*y*y) + 1E-6*(x*x*x*x - 6*x*x*y*y + y*y*y*y) + 1E-7*z*(2*z*z - 3*x*x - 3*y*y)
	);
}

int test_evaluate_expansion
(
	double atol
)
{
	struct Electrode electrode = make_square_electrode();

	struct HarmonicPolynomialData data = { { 10.0, -5.0, 40.0 } };
	build_expansion_from_function(&electrode, harmonic_polynomial, &data);

	// 1. Analytical result and 2. tiopt result, compared at points throughout both layers of spheres
	double error = 0.0;
	double scale = 0.0;
	srand(1);
	for (int t = 0; t < 1000; t++)
	{
		double position[3] =
		{
			-200 + 400.0 * rand() / RAND_MAX,
			-200 + 400.0 * rand() / RAND_MAX,
			25 + 40.0 * rand() / RAND_MAX
		};

		double phi;
		evaluate_electrode_potential(&electrode, position, &phi);

		double phi_analytical = harmonic_polynomial(position, &data);

		error = MAX(error, fabs(phi - phi_analytical));
		scale = MAX(scale, fabs(phi_analytical));
	}

	int result = check_error("expansion reproduces harmonic polynomial", error / scale, atol);

	// The gradient of the expansion must match the derivative of the polynomial (by Richardson-extrapolated central
	// differences of the polynomial itself)
	double gradient_error = 0.0;
	double gradient_scale = 0.0;
	for (int t = 0; t < 200; t++)
	{
		double position[3] =
		{
			-150 + 300.0 * rand() / RAND_MAX,
			-150 + 300.0 * rand() / RAND_MAX,
			30 + 30.0 * rand() / RAND_MAX
		};

		double grad_phi[3];
		evaluate_electrode_gradient(&electrode, position, grad_phi);

		for (int k = 0; k < 3; k++)
		{
			double h = 1E-2;
			double f[4];
			double offsets[4] = { -2*h, -h, h, 2*h };
			for (int o = 0; o < 4; o++)
			{
				double p[3] = { position[0], position[1], position[2] };
				p[k] += offsets[o];
				f[o] = harmonic_polynomial(p, &data);
			}

			double derivative = (f[0] - 8*f[1] + 8*f[2] - f[3]) / (12 * h);

			gradient_error = MAX(gradient_error, fabs(grad_phi[k] - derivative));
			gradient_scale = MAX(gradient_scale, fabs(derivative));
		}
	}

	result += check_error("expansion gradient matches polynomial derivative", gradient_error / gradient_scale, 1E3 * atol);

	free(electrode.Vlm);

	return result;
}

int test_evaluate_electrode_hessian
(
	double atol
)
{
	struct Electrode electrode = make_square_electrode();
	double electrode_position[3] = { 0.0, 0.0, 0.0 };

	solve_trap_electrodynamics(
		&(struct Trap) { .n_electrodes = 1, .electrodes = (struct Electrode (*)[]) &electrode, .electrode_positions = (double (*)[][3]) &electrode_position },
		ANALYTICAL,
		NULL
	);

	double error = 0.0;
	double laplacian = 0.0;
	double scale = 0.0;
	for (double x = -40; x <= 40; x += 8)
		for (double z = 40; z <= 65; z += 5)
		{
			double position[3] = { x, 0.3 * x, z };

			double H_analytical[3][3];
			double H_finite_differences[3][3];
			evaluate_electrode_hessian_analytical(&electrode, position, H_analytical);
			evaluate_electrode_hessian_finite_differences(&electrode, position, H_finite_differences);

			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 3; j++)
				{
					error = MAX(error, fabs(H_analytical[i][j] - H_finite_differences[i][j]));
					scale = MAX(scale, fabs(H_analytical[i][j]));
				}

			// Every term of the expansion is harmonic
			laplacian = MAX(laplacian, fabs(H_analytical[0][0] + H_analytical[1][1] + H_analytical[2][2]));
		}

	int result = check_error("analytical Hessian matches finite differences", error / scale, 1E-6);
	result += check_error("expansion satisfies Laplace's equation", laplacian / scale, atol);

	free(electrode.Vlm);

	return result;
}

int test_expansion_accuracy
(
	double atol
)
{
	struct Electrode electrode = make_square_electrode();
	double electrode_position[3] = { 25.0, 0.0, 0.0 };

	solve_trap_electrodynamics(
		&(struct Trap) { .n_electrodes = 1, .electrodes = (struct Electrode (*)[]) &electrode, .electrode_positions = (double (*)[][3]) &electrode_position },
		ANALYTICAL,
		NULL
	);

	// Truncating the expansion at LMAX limits its accuracy, which is best around the height of the ions
	double error = 0.0;
	for (double x = -60; x <= 110; x += 1.7)
		for (double y = -40; y <= 40; y += 3.1)
			for (double z = 45; z <= 60; z += 5)
			{
				double position[3] = { x, y, z };

				double phi;
				double phi_analytical;
				evaluate_electrode_potential(&electrode, position, &phi);
				evaluate_electrode_potential_analytical(&electrode, electrode_position, position, &phi_analytical);

				error = MAX(error, fabs(phi - phi_analytical) / phi_analytical);
			}

	int result = check_error("expansion of electrode potential near ion height", error, 1E-4);

	free(electrode.Vlm);

	return result;
}

int test_pseudopotential
(
	double atol
)
{
	struct Trap trap;
	generate_trap_from_preset(LINEAR_SURFACE_TRAP, 7, 50.0, &trap);
	solve_trap_electrodynamics(&trap, ANALYTICAL, NULL);

	struct IonParameters ion;
	generate_ion_from_preset("40Ca+", &ion);

	double m = ion_mass(&ion);
	double q = ion_charge(&ion);

	// The preset places the RF null at ION_HEIGHT above the trap center
	double null[3] = { 0.0, 3.0, ION_HEIGHT - 4.0 };
	find_rf_null(&trap, &null, 100);

	int result = check_error("RF null of linear trap preset at ion height", sqrt(null[1]*null[1] + pow(null[2] - ION_HEIGHT, 2)), 1E-3);

	// At the null, the pseudopotential Hessian is 2 c H_RF^2 exactly, where c = (q V_RF)^2 / (4 m (Omega_RF L)^2)
	double *rf_weights = calloc(trap.n_electrodes, sizeof(double));
	rf_weights[0] = 1.0;
	rf_weights[1] = 1.0;

	double grad_rf[3];
	double H_rf[3][3];
	evaluate_trap_potential(&trap, rf_weights, null, NULL, grad_rf, H_rf);

	double H_U_pond[3][3];
	evaluate_pseudopotential(&trap, m, q, null, NULL, NULL, H_U_pond);

	double c = pow(q * trap.V_RF, 2) / (4 * m * pow(trap.Omega_RF * LENGTH_SCALE, 2));
	double error = 0.0;
	double scale = 0.0;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
		{
			double H_exact = 0.0;
			for (int k = 0; k < 3; k++)
				H_exact += 2 * c * H_rf[i][k] * H_rf[k][j];

			error = MAX(error, fabs(H_U_pond[i][j] - H_exact));
			scale = MAX(scale, fabs(H_exact));
		}

	result += check_error("pseudopotential Hessian at RF null", error / scale, 1E-6);

	// For infinitely long rails spanning heights y_1 to y_2 off the trap axis, phi_RF = (2/PI) (atan(y_2/z) - atan(y_1/z))
	// on the axis, which sets the radial secular frequency q V_RF |phi_zz| / (sqrt(2) m Omega_RF)
	double y_1 = (*trap.electrode_positions)[0][1] - (*trap.electrodes)[0].vertices[0][1];
	double y_2 = (*trap.electrode_positions)[0][1] + (*trap.electrodes)[0].vertices[0][1];
	double z = ION_HEIGHT;
	double phi_zz = 2 / PI * (2 * y_2 * z / pow(z*z + y_2*y_2, 2) - 2 * y_1 * z / pow(z*z + y_1*y_1, 2));
	double omega_radial = q * trap.V_RF * fabs(phi_zz) / (LENGTH_SCALE * LENGTH_SCALE) / (SQRT2 * m * trap.Omega_RF);

	double *voltages = calloc(trap.n_electrodes, sizeof(double));
	double omega[3];
	double axes[3][3];
	compute_normal_modes(&trap, m, q, voltages, null, &omega, &axes);

	// The rails are finite, so allow for a small deviation from the two-dimensional result
	double frequency_error = MAX(fabs(omega[1] - omega_radial), fabs(omega[2] - omega_radial)) / omega_radial;
	result += check_error("radial frequencies match infinite rail model", frequency_error, 5E-2);

	free(voltages);
	free(rf_weights);
	free_trap(&trap);

	return result;
}
