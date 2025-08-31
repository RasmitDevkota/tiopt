#include "ti_utils.h"

#include <stdio.h>
#include <math.h>

#include "defs.h"

// Compute centered difference gradient
void grad(
	const int f_len,
	double (*f)[f_len],
	double (*grad_f)[f_len],
	const double dx
)
{
	for (int k = 0; k < f_len; k++)
	{
		// compute the next index
		const int idx_next = (k+1) % f_len;

		// compute the previous index by a safer method in case of negative overflow
		const int idx_prev = ((k-1) % f_len + f_len) % f_len;

		(*grad_f)[k] = (2 * dx) * ((*f)[idx_next] - (*f)[idx_prev]);
	}
}

// @TODO - implement a more robust interpolation algorithm
// Compute weighted centered-averaging interpolation
void interpolate_1d(
	const int f_len,
	double (*f)[f_len],
	double x_rel,
	double *f_x_rel,
	const double dx
)
{
	double x_idx_prev = floor(x_rel * dx);
	double x_idx_next = ceil(x_rel * dx);
	double distance_prev = x_rel - x_idx_prev;
	double distance_next = x_idx_next - x_rel;

	double weight_prev = 1-distance_prev;
	double weight_next = 1-distance_next;

	double f_prev = (*f)[(int)x_idx_prev];
	double f_next = (*f)[(int)x_idx_next];

	*f_x_rel = weight_prev * f_prev + weight_next * f_next;
}

// @TODO - implement a more robust interpolation algorithm
// Compute weighted centered-averaging interpolation
void interpolate_3d(
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
		(x < 1 || x > f_len_x-1) ||
		(y < 1 || y > f_len_y-1) ||
		(z < 1 || z > f_len_z-1)
	)
	{
		printf("Cannot interpolate value of function at point (%f,%f,%f) - too close to edge of function.\n", x, y, z);
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

int point_in_polygon_zslice(
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

