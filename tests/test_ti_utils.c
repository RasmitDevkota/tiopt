#include "test_ti_utils.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"
#include "ti_utils.h"
#include "quantum.h"
#include "test_utils.h"

int test_all_ti_utils
(
	double atol
)
{
	printf("- Testing utilities...\n");

	int result = 0;

	result += test_dot_and_cross(atol);
	result += test_grad_1d(atol);
	result += test_interpolation(atol);
	result += test_point_in_polygon_zslice(atol);
	result += test_compute_hessian_3d(atol);
	result += test_diagonalize_matrix_3d(atol);
	result += test_find_step_times(atol);

	return result;
}

int test_dot_and_cross
(
	double atol
)
{
	double v1[3] = { 1.0, 2.0, 3.0 };
	double v2[3] = { -4.0, 0.5, 2.0 };

	// The result is overwritten rather than accumulated
	double v1_dot_v2 = 123.0;
	dot(3, &v1, &v2, &v1_dot_v2);

	double v1_cross_v2[3];
	cross(&v1, &v2, &v1_cross_v2);

	double error = fabs(v1_dot_v2 - 3.0);
	error += fabs(v1_cross_v2[0] - 2.5) + fabs(v1_cross_v2[1] + 14.0) + fabs(v1_cross_v2[2] - 8.5);

	return check_error("dot and cross products", error, atol);
}

int test_grad_1d
(
	double atol
)
{
	// Centered differences of a periodic function have error dx^2 f'''/6
	const int N = 1000;
	double dx = 1.0 / N;

	double (*f)[N] = malloc(N * sizeof(double));
	double (*grad_f)[N] = malloc(N * sizeof(double));
	for (int k = 0; k < N; k++)
		(*f)[k] = sin(2 * PI * k * dx);

	grad_1d(N, f, grad_f, dx);

	double error = 0.0;
	for (int k = 0; k < N; k++)
		error = MAX(error, fabs((*grad_f)[k] - 2 * PI * cos(2 * PI * k * dx)));

	free(grad_f);
	free(f);

	return check_error("grad_1d of periodic function", error, 1.1 * pow(2 * PI, 3) * dx*dx / 6);
}

int test_interpolation
(
	double atol
)
{
	// Linear interpolation reproduces linear functions exactly, including at grid points
	const int N = 10;

	double (*f_1d)[N] = malloc(N * sizeof(double));
	double (*f_2d)[N][N] = malloc(N * N * sizeof(double));
	double (*f_3d)[N][N][N] = malloc(N * N * N * sizeof(double));

	for (int x = 0; x < N; x++)
	{
		(*f_1d)[x] = 3.0 * x - 1.0;

		for (int y = 0; y < N; y++)
		{
			(*f_2d)[x][y] = 3.0 * x - 2.0 * y + 0.5;

			for (int z = 0; z < N; z++)
				(*f_3d)[x][y][z] = 3.0 * x - 2.0 * y + 0.25 * z + 0.5;
		}
	}

	double error = 0.0;
	for (double x = 0.0; x <= N - 1; x += 0.37)
	{
		double f_x;
		interpolate_1d(N, f_1d, x, &f_x, 1.0);
		error = MAX(error, fabs(f_x - (3.0 * x - 1.0)));

		double y = 0.61 * x;
		double p_2d[2] = { x, y };
		double f_xy;
		interpolate_2d(N, N, f_2d, p_2d, &f_xy, 1.0, 1.0);
		error = MAX(error, fabs(f_xy - (3.0 * x - 2.0 * y + 0.5)));

		double z = (N - 1) - x;
		double p_3d[3] = { x, y, z };
		double f_xyz;
		interpolate_3d(N, N, N, f_3d, p_3d, &f_xyz, 1.0, 1.0, 1.0);
		error = MAX(error, fabs(f_xyz - (3.0 * x - 2.0 * y + 0.25 * z + 0.5)));
	}

	// Integer points used to be counted twice by interpolate_1d
	double f_integer;
	interpolate_1d(N, f_1d, 4.0, &f_integer, 1.0);
	error = MAX(error, fabs(f_integer - 11.0));

	free(f_3d);
	free(f_2d);
	free(f_1d);

	return check_error("linear interpolation in 1D, 2D and 3D", error, atol);
}

int test_point_in_polygon_zslice
(
	double atol
)
{
	// L-shaped polygon in the z = 0 plane
	double vertices[6][3] = { { 0, 0, 0 }, { 30, 0, 0 }, { 30, 10, 0 }, { 10, 10, 0 }, { 10, 30, 0 }, { 0, 30, 0 } };

	int inside_points[4][3] = { { 5, 5, 0 }, { 25, 5, 0 }, { 5, 25, 0 }, { 9, 29, 0 } };
	int outside_points[5][3] = { { 20, 20, 0 }, { 35, 5, 0 }, { -1, 5, 0 }, { 5, 5, 1 }, { 25, 25, 0 } };

	int n_wrong = 0;
	for (int check_bounding_box = 0; check_bounding_box < 2; check_bounding_box++)
	{
		for (int p = 0; p < 4; p++)
			n_wrong += !point_in_polygon_zslice(inside_points[p][0], inside_points[p][1], inside_points[p][2], 6, vertices, check_bounding_box);

		for (int p = 0; p < 5; p++)
			n_wrong += point_in_polygon_zslice(outside_points[p][0], outside_points[p][1], outside_points[p][2], 6, vertices, check_bounding_box);
	}

	return check_error("point_in_polygon_zslice classifies points", n_wrong, 0);
}

int test_compute_hessian_3d
(
	double atol
)
{
	// Central differences are exact for quadratic functions
	const int N = 12;
	double (*f)[N][N][N] = malloc(N * N * N * sizeof(double));

	// Use a grid spacing of 0.5, i.e. 2 grid indices per unit
	double d = 2.0;

	double H_analytical[3][3] = { { 2.0, 0.5, -1.0 }, { 0.5, -4.0, 0.25 }, { -1.0, 0.25, 6.0 } };
	for (int x = 0; x < N; x++)
		for (int y = 0; y < N; y++)
			for (int z = 0; z < N; z++)
			{
				double p[3] = { x / d, y / d, z / d };

				(*f)[x][y][z] = 0.0;
				for (int i = 0; i < 3; i++)
					for (int j = 0; j < 3; j++)
						(*f)[x][y][z] += 0.5 * H_analytical[i][j] * p[i] * p[j];
			}

	double p_rel[3] = { 3.0, 2.5, 3.5 };
	double h[3] = { 1.0, 2.0, 1.0 };
	double H[3][3];
	compute_hessian_3d(N, N, N, f, p_rel, &H, h, d, d, d);

	double error = 0.0;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			error = MAX(error, fabs(H[i][j] - H_analytical[i][j]));

	free(f);

	return check_error("compute_hessian_3d of quadratic function", error, 1E3 * atol);
}

int test_diagonalize_matrix_3d
(
	double atol
)
{
	double A[3][3] = { { 4.0, 1.0, -2.0 }, { 1.0, 2.0, 0.0 }, { -2.0, 0.0, 3.0 } };

	double eigenvalues[3];
	double eigenvectors[3][3];
	diagonalize_matrix_3d(&A, &eigenvalues, &eigenvectors);

	// Check A v = lambda v, orthonormality and ascending order
	double error = 0.0;
	for (int k = 0; k < 3; k++)
	{
		for (int i = 0; i < 3; i++)
		{
			double A_v = 0.0;
			for (int j = 0; j < 3; j++)
				A_v += A[i][j] * eigenvectors[k][j];

			error = MAX(error, fabs(A_v - eigenvalues[k] * eigenvectors[k][i]));
		}

		for (int l = 0; l < 3; l++)
		{
			double v_k_dot_v_l = 0.0;
			for (int i = 0; i < 3; i++)
				v_k_dot_v_l += eigenvectors[k][i] * eigenvectors[l][i];

			error = MAX(error, fabs(v_k_dot_v_l - (k == l ? 1.0 : 0.0)));
		}
	}

	if (eigenvalues[0] > eigenvalues[1] || eigenvalues[1] > eigenvalues[2])
		error = HUGE_VAL;

	// The trace is invariant
	error = MAX(error, fabs(eigenvalues[0] + eigenvalues[1] + eigenvalues[2] - 9.0));

	return check_error("diagonalize_matrix_3d of symmetric matrix", error, 1E3 * atol);
}

int test_find_step_times
(
	double atol
)
{
	struct PulseProgram *pulse_program = allocate_pulse_program(1, 3);

	// Two pulses sharing a boundary and one overlapping them
	double times[3] = { 1E-6, 3E-6, 2E-6 };
	double durations[3] = { 2E-6, 1E-6, 4E-6 };
	for (int p = 0; p < 3; p++)
	{
		pulse_program->pulse_times[p] = times[p];
		pulse_program->pulse_durations[p] = durations[p];
	}

	int n_event_times;
	double *event_times = find_event_times(pulse_program, &n_event_times);

	double expected_event_times[5] = { 0.0, 1E-6, 2E-6, 3E-6, 4E-6 };
	double error = (n_event_times == 6) ? 0.0 : HUGE_VAL;
	for (int i = 0; i < MIN(n_event_times, 5); i++)
		error = MAX(error, fabs(event_times[i] - expected_event_times[i]));
	if (n_event_times == 6)
		error = MAX(error, fabs(event_times[5] - 6E-6));

	int result = check_error("find_event_times sorts and merges pulse boundaries", error / 1E-6, atol);

	int n_step_times;
	int *step_times = find_step_times(pulse_program, 2E-6, &n_step_times);

	// Times of 1 µs and 3 µs round to steps 1 and 2 (half away from zero) or 1, merging with the steps of 2 µs and 4 µs
	int expected_step_times[4] = { 0, 1, 2, 3 };
	int n_wrong = (n_step_times == 4) ? 0 : 1;
	for (int i = 0; i < MIN(n_step_times, 4); i++)
		n_wrong += (step_times[i] != expected_step_times[i]);

	result += check_error("find_step_times converts to distinct steps", n_wrong, 0);

	free(step_times);
	free(event_times);
	free_pulse_program(pulse_program);

	return result;
}
