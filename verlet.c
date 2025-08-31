#include "verlet.h"
#include "ti_utils.h"

#include <stdlib.h>
#include <stdio.h>

void velocity_verlet(
	const double m,
	const double q,
	const int V_len,
	double (*V)[V_len], // @TODO - do we need to pass the Trap instead since we only store V for each electrode separately?
	double (*pos)[3],
	double (*vel)[3],
	double (*acc)[3],
	const int n_it,
	const double dt
) {
	// @TODO - consider passing grad_V around instead
	// compute grad_V
	double (*grad_V)[3] = calloc(3, sizeof(double));
	grad(V_len, V, grad_V, dt); // @TODO - choose a better value for dx than dt

	for (int t = 0; t < n_it; t++)
	{
		velocity_verlet_update(m, q, V_len, V, grad_V, pos, vel, acc, dt);

		printf("Position at step %d, time %f: (%f,%f,%f)\n", t, (double)(t+1) * dt, (*pos)[0], (*pos)[1], (*pos)[2]);
	}

	free(grad_V);

	printf("Finished!\n");
}

void velocity_verlet_update(
	const double m,
	const double q,
	const int V_len,
	double (*V)[V_len],
	double (*grad_V)[V_len],
	double (*pos)[3],
	double (*vel)[3],
	double (*acc)[3],
	const double dt
) {
	for (int k = 0; k < 3; k++)
	{
		(*pos)[k] += (*vel)[k] * dt + 0.5 * (*acc)[k] * dt * dt;
	}

	// we have to keep the old and new accelerations separate for later use
	// in updating the velocity
	double (*new_acc)[3] = calloc(3, sizeof(double));
	for (int k = 0; k < 3; k++)
	{
		(*new_acc)[k] = -q * (*grad_V)[k]/m;
	}

	for (int k = 0; k < 3; k++)
	{
		*(vel)[k] += ((*acc)[k]+(*new_acc)[k]) * (dt * 0.5);
		
		// we don't need to track the old acceleration after we use it here,
		// so update the primary acceleration array
		*(acc)[k] = (*new_acc)[k];
	}

	free(new_acc);
}

