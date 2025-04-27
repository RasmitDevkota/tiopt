#include "ti_utils.h"

#include <math.h>

// Compute centered difference gradient
void grad(
	const int f_len,
	double (*f)[f_len],
	double (*grad_f)[f_len],
	const double dx
) {
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
) {
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

