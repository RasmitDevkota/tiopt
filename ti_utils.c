#include "ti_utils.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <gsl/gsl_eigen.h>

#include "defs.h"

// Compute dot product of vectors
void dot
(
	const int len,
	double (*v1)[len],
	double (*v2)[len],
	double *v1_dot_v2
)
{
	*v1_dot_v2 = 0.0;

	for (int k = 0; k < len; k++)
	{
		*v1_dot_v2 += (*v1)[k] * (*v2)[k];
	}
}

// Compute cross product of 3D vectors
void cross
(
	double (*v1)[3],
	double (*v2)[3],
	double (*v1_cross_v2)[3]
)
{
	(*v1_cross_v2)[0] = (*v1)[1] * (*v2)[2] - (*v1)[2] * (*v2)[1];
	(*v1_cross_v2)[1] = (*v1)[2] * (*v2)[0] - (*v1)[0] * (*v2)[2];
	(*v1_cross_v2)[2] = (*v1)[0] * (*v2)[1] - (*v1)[1] * (*v2)[0];
}

// Compute centered difference gradient of a periodic function sampled with spacing dx
void grad_1d
(
	const int f_len,
	double (*f)[f_len],
	double (*grad_f)[f_len],
	const double dx
)
{
	for (int k = 0; k < f_len; k++)
	{
		// Compute the next index
		const int idx_next = (k+1) % f_len;

		// Compute the previous index by a safer method in case of negative overflow
		const int idx_prev = ((k-1) % f_len + f_len) % f_len;

		(*grad_f)[k] = ((*f)[idx_next] - (*f)[idx_prev]) / (2 * dx);
	}
}

// @TODO - implement a more robust interpolation algorithm
// Compute weighted centered-averaging interpolation
void interpolate_1d
(
	const int f_len,
	double (*f)[f_len],
	double x_rel,
	double *f_x_rel,
	const double dx
)
{
	double x = x_rel * dx;

	if (x < 0 || x > f_len-1)
	{
		printf("Cannot interpolate value of function at point index %.13f - too close to edge of function at %d.\n", x, f_len);
		return;
	}

	double x_0 = floor(x);
	double x_1 = ceil(x);
	double x_d = (x_0 == x_1) ? 0.0 : (x - x_0)/(x_1 - x_0);

	*f_x_rel = (*f)[(int)x_0] * (1 - x_d) + (*f)[(int)x_1] * x_d;
}

// @TODO - implement a more robust interpolation algorithm
// Compute weighted centered-averaging interpolation
void interpolate_2d
(
	const int f_len_x,
	const int f_len_y,
	double (*f)[f_len_x][f_len_y],
	double p_rel[2],
	double *f_p_rel,
	double dx,
	double dy
)
{
	double x = p_rel[0] * dx;
	double y = p_rel[1] * dy;

	if (
		(x < 0 || x > f_len_x-1) ||
		(y < 0 || y > f_len_y-1)
	)
	{
		printf("Cannot interpolate value of function at point index (%.13f,%.13f) - too close to edge of function at (%d,%d).\n", x, y, f_len_x, f_len_y);
		return;
	}

	double x_0 = floor(x);
	double y_0 = floor(y);
	double x_1 = ceil(x);
	double y_1 = ceil(y);
	double x_d = (x_0 == x_1) ? 0.0 : (x - x_0)/(x_1 - x_0);
	double y_d = (y_0 == y_1) ? 0.0 : (y - y_0)/(y_1 - y_0);

	// Interpolate along the x-axis
	double f_0 = (*f)[(int)x_0][(int)y_0] * (1 - x_d) + (*f)[(int)x_1][(int)y_0] * x_d;
	double f_1 = (*f)[(int)x_0][(int)y_1] * (1 - x_d) + (*f)[(int)x_1][(int)y_1] * x_d;

	// Interpolate along the y-axis
	*f_p_rel = f_0 * (1 - y_d) + f_1 * y_d;
}

// @TODO - implement a more robust interpolation algorithm
// Compute weighted centered-averaging interpolation
void interpolate_3d
(
	const int f_len_x,
	const int f_len_y,
	const int f_len_z,
	double (*f)[f_len_x][f_len_y][f_len_z],
	double p_rel[3],
	double *f_p_rel,
	double dx,
	double dy,
	double dz
)
{
	double x = p_rel[0] * dx;
	double y = p_rel[1] * dy;
	double z = p_rel[2] * dz;

	if (
		(x < 0 || x > f_len_x-1) ||
		(y < 0 || y > f_len_y-1) ||
		(z < 0 || z > f_len_z-1)
	)
	{
		printf("Cannot interpolate value of function at point index (%.13f,%.13f,%.13f) - too close to edge of function at (%d,%d,%d).\n", x, y, z, f_len_x, f_len_y, f_len_z);
		return;
	}

	double x_0 = floor(x);
	double y_0 = floor(y);
	double z_0 = floor(z);
	double x_1 = ceil(x);
	double y_1 = ceil(y);
	double z_1 = ceil(z);
	double x_d = (x_0 == x_1) ? 0.0 : (x - x_0)/(x_1 - x_0);
	double y_d = (y_0 == y_1) ? 0.0 : (y - y_0)/(y_1 - y_0);
	double z_d = (z_0 == z_1) ? 0.0 : (z - z_0)/(z_1 - z_0);

	// Interpolate along the x-axis
	double f_00 = (*f)[(int)x_0][(int)y_0][(int)z_0] * (1 - x_d) + (*f)[(int)x_1][(int)y_0][(int)z_0] * x_d;
	double f_01 = (*f)[(int)x_0][(int)y_0][(int)z_1] * (1 - x_d) + (*f)[(int)x_1][(int)y_0][(int)z_1] * x_d;
	double f_10 = (*f)[(int)x_0][(int)y_1][(int)z_0] * (1 - x_d) + (*f)[(int)x_1][(int)y_1][(int)z_0] * x_d;
	double f_11 = (*f)[(int)x_0][(int)y_1][(int)z_1] * (1 - x_d) + (*f)[(int)x_1][(int)y_1][(int)z_1] * x_d;

	// Interpolate along the y-axis
	double f_0 = f_00 * (1 - y_d) + f_10 * y_d;
	double f_1 = f_01 * (1 - y_d) + f_11 * y_d;

	// Interpolate along the z-axis
	*f_p_rel = f_0 * (1 - z_d) + f_1 * z_d;
}

// Check whether the grid point (x, y, z) lies inside a polygon in its z-slice, optionally checking the
// bounding box of the polygon first
int point_in_polygon_zslice
(
	int x,
	int y,
	int z,
	int n_vertices,
	double (*vertices)[3],
	int check_bounding_box
)
{
	if (check_bounding_box)
	{
		int min_X = vertices[0][0];
		int max_X = vertices[0][0];
		int min_Y = vertices[0][1];
		int max_Y = vertices[0][1];
		int min_Z = vertices[0][2];
		int max_Z = vertices[0][2];

		for (int i = 1; i < n_vertices; i++)
		{
			min_X = MIN(vertices[i][0], min_X);
			max_X = MAX(vertices[i][0], max_X);
			min_Y = MIN(vertices[i][1], min_Y);
			max_Y = MAX(vertices[i][1], max_Y);
			min_Z = MIN(vertices[i][2], min_Z);
			max_Z = MAX(vertices[i][2], max_Z);
		}

		if (x < min_X || x > max_X || y < min_Y || y > max_Y || z < min_Z || z > max_Z)
			return 0;
	}
	else
	{
		int min_Z = vertices[0][2];
		int max_Z = vertices[0][2];

		for (int i = 1; i < n_vertices; i++)
		{
			min_Z = MIN(vertices[i][2], min_Z);
			max_Z = MAX(vertices[i][2], max_Z);
		}

		if (z < min_Z || z > max_Z)
			return 0;
	}

	// https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
	int inside = 0;
	for (int i = 0, j = n_vertices - 1; i < n_vertices; j = i++)
		if (
			(vertices[i][1] > y) != (vertices[j][1] > y) &&
			x < (vertices[j][0] - vertices[i][0]) * (y - vertices[i][1]) / (vertices[j][1] - vertices[i][1]) + vertices[i][0]
		)
			inside = 1 - inside;

	return inside;
}

// Compute the Hessian by central differences, where h holds the step sizes in units of grid indices
void compute_hessian_3d
(
	const int f_len_x,
	const int f_len_y,
	const int f_len_z,
	double (*f)[f_len_x][f_len_y][f_len_z],
	double p_rel[3],
	double (*H)[3][3],
	double h[3],
	double dx,
	double dy,
	double dz
)
{
	int x = (int) round(p_rel[0] * dx);
	int y = (int) round(p_rel[1] * dy);
	int z = (int) round(p_rel[2] * dz);

	int xph = x + (int) h[0];
	int xmh = x - (int) h[0];
	int yph = y + (int) h[1];
	int ymh = y - (int) h[1];
	int zph = z + (int) h[2];
	int zmh = z - (int) h[2];

	if (
		(xmh < 0 || xph > f_len_x-1) ||
		(ymh < 0 || yph > f_len_y-1) ||
		(zmh < 0 || zph > f_len_z-1)
	)
	{
		printf("Cannot compute value of Hessian at point index (%d,%d,%d) - too close to edge of function at (%d,%d,%d).\n", x, y, z, f_len_x, f_len_y, f_len_z);
		return;
	}

	double f_xyz = (*f)[x][y][z];
	double f_xph = (*f)[xph][y][z];
	double f_xmh = (*f)[xmh][y][z];
	double f_yph = (*f)[x][yph][z];
	double f_ymh = (*f)[x][ymh][z];
	double f_zph = (*f)[x][y][zph];
	double f_zmh = (*f)[x][y][zmh];

	double f_xphyph = (*f)[xph][yph][z];
	double f_xmhyph = (*f)[xmh][yph][z];
	double f_xphymh = (*f)[xph][ymh][z];
	double f_xmhymh = (*f)[xmh][ymh][z];

	double f_xphzph = (*f)[xph][y][zph];
	double f_xmhzph = (*f)[xmh][y][zph];
	double f_xphzmh = (*f)[xph][y][zmh];
	double f_xmhzmh = (*f)[xmh][y][zmh];

	double f_yphzph = (*f)[x][yph][zph];
	double f_ymhzph = (*f)[x][ymh][zph];
	double f_yphzmh = (*f)[x][yph][zmh];
	double f_ymhzmh = (*f)[x][ymh][zmh];

	// Convert the step sizes from grid indices to physical units
	double h_x = h[0] / dx;
	double h_y = h[1] / dy;
	double h_z = h[2] / dz;

	(*H)[0][0] = (f_xph - 2 * f_xyz + f_xmh)/pow(h_x, 2);
	(*H)[1][1] = (f_yph - 2 * f_xyz + f_ymh)/pow(h_y, 2);
	(*H)[2][2] = (f_zph - 2 * f_xyz + f_zmh)/pow(h_z, 2);

	(*H)[0][1] = (f_xphyph - f_xphymh - f_xmhyph + f_xmhymh)/(4 * h_x * h_y);
	(*H)[0][2] = (f_xphzph - f_xphzmh - f_xmhzph + f_xmhzmh)/(4 * h_x * h_z);

	(*H)[1][2] = (f_yphzph - f_yphzmh - f_ymhzph + f_ymhzmh)/(4 * h_y * h_z);

	(*H)[1][0] = (*H)[0][1];
	(*H)[2][0] = (*H)[0][2];
	(*H)[2][1] = (*H)[1][2];
}

// Diagonalize a symmetric 3x3 matrix, returning eigenvalues in ascending order and the
// corresponding normalized eigenvectors as the rows of eigenvectors
int diagonalize_matrix_3d
(
	double (*A)[3][3],
	double (*eigenvalues)[3],
	double (*eigenvectors)[3][3]
)
{
	gsl_matrix *A_gsl = gsl_matrix_alloc(3, 3);
	gsl_vector *eval = gsl_vector_alloc(3);
	gsl_matrix *evec = gsl_matrix_alloc(3, 3);
	gsl_eigen_symmv_workspace *workspace = gsl_eigen_symmv_alloc(3);

	// Symmetrize to protect against round-off in the input
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			gsl_matrix_set(A_gsl, i, j, 0.5 * ((*A)[i][j] + (*A)[j][i]));

	int status = gsl_eigen_symmv(A_gsl, eval, evec, workspace);
	gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_VAL_ASC);

	for (int k = 0; k < 3; k++)
	{
		(*eigenvalues)[k] = gsl_vector_get(eval, k);

		for (int i = 0; i < 3; i++)
			(*eigenvectors)[k][i] = gsl_matrix_get(evec, i, k);
	}

	gsl_eigen_symmv_free(workspace);
	gsl_matrix_free(evec);
	gsl_vector_free(eval);
	gsl_matrix_free(A_gsl);

	return status;
}

static int compare_doubles
(
	const void *a,
	const void *b
)
{
	double da = *(const double *) a;
	double db = *(const double *) b;

	return (da > db) - (da < db);
}

// Find the sorted, distinct times at which any pulse in a pulse program starts or stops
double* find_event_times
(
	struct PulseProgram *pulse_program,
	int *n_event_times
)
{
	double *event_times = malloc((2 * pulse_program->n_pulses + 1) * sizeof(double));

	int n = 0;
	event_times[n++] = 0.0;
	for (int p = 0; p < pulse_program->n_pulses; p++)
	{
		event_times[n++] = pulse_program->pulse_times[p];
		event_times[n++] = pulse_program->pulse_times[p] + pulse_program->pulse_durations[p];
	}

	qsort(event_times, n, sizeof(double), compare_doubles);

	// Remove duplicates, treating times closer than a femtosecond as equal
	int n_distinct = 0;
	for (int i = 0; i < n; i++)
		if (n_distinct == 0 || event_times[i] - event_times[n_distinct-1] > 1E-15)
			event_times[n_distinct++] = event_times[i];

	*n_event_times = n_distinct;

	return event_times;
}

// Find the sorted, distinct timestep indices (for timestep dt) at which any pulse starts or stops
int* find_step_times
(
	struct PulseProgram *pulse_program,
	const double dt,
	int *n_step_times
)
{
	int n_event_times;
	double *event_times = find_event_times(pulse_program, &n_event_times);

	int *step_times = malloc(n_event_times * sizeof(int));

	int n_distinct = 0;
	for (int i = 0; i < n_event_times; i++)
	{
		int step = (int) round(event_times[i] / dt);

		if (n_distinct == 0 || step != step_times[n_distinct-1])
			step_times[n_distinct++] = step;
	}

	free(event_times);

	*n_step_times = n_distinct;

	return step_times;
}
