#include "verlet.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_legendre.h>

#include "defs.h"
#include "data_structures.h"
#include "ti_utils.h"

void velocity_verlet(
	struct Trap *trap,
	const double m,
	const double q,
	double (*pos)[3],
	double (*vel)[3],
	double (*acc)[3],
	const int n_it,
	const double dt
)
{
	for (int t = 0; t < n_it; t++)
	{
		velocity_verlet_update(trap, m, q, pos, vel, acc, dt);

		printf("Position at step %d, time %f: (%f,%f,%f)\n", t, (double)(t+1) * dt, (*pos)[0], (*pos)[1], (*pos)[2]);
	}

	printf("Finished!\n");
}

void velocity_verlet_update(
	struct Trap *trap,
	const double m,
	const double q,
	double (*pos)[3],
	double (*vel)[3],
	double (*acc)[3],
	const double dt
)
{
	for (int k = 0; k < 3; k++)
	{
		(*pos)[k] += (*vel)[k] * dt + 0.5 * (*acc)[k] * dt * dt;
	}

	double (*grad_V)[3] = calloc(3, sizeof(double));
	for (int e = 0; e < trap->n_electrodes; e++)
	{
		int Vlm_len = (*trap->electrodes)[e].Vlm_len;
		double (*Vlm)[Vlm_len] = (*trap->electrodes)[e].Vlm;
		for (int sx = 0; sx < NSPH_X; sx++)
			for (int sy = 0; sy < NSPH_Y; sy++)
				for (int sz = 0; sz < NSPH_Z; sz++)
				{
					double x_c = sx * SPH_SPACING + SPH_R;
					double y_c = sy * SPH_SPACING + SPH_R;
					double z_c = sz * SPH_SPACING + SPH_Z_MIN + 1; // # @TEST - check the +1
					double dx = (*pos)[0] - x_c;
					double dy = (*pos)[1] - y_c;
					double dz = (*pos)[2] - z_c;
					double r = sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2));
					if (r <= SPH_R)
					{
						double costheta = dz/r;
						double phi = atan2(dy, dx);

						double *Ylm_thetaphi = calloc(gsl_sf_legendre_array_n(LMAX), sizeof(double));
						gsl_sf_legendre_array_e(GSL_SF_LEGENDRE_SPHARM, LMAX, costheta, -1, Ylm_thetaphi);

						double (*Vlm_sphere)[(LMAX+1)*(LMAX+1)*2] = VLM_SLICE(&Vlm, sx, sy, sz);

						// @TODO - handle imaginary parts
						for (int l = 0; l <= LMAX; l++)
						{
							// Add dR*Y terms
							double dRl_dx = l * dx * pow(r, l-2);
							double dRl_dy = l * dy * pow(r, l-2);
							for (int m = -l; m < 0; m++)
							{
								(*grad_V)[0] += (*Vlm_sphere)[2 * (l * (LMAX + 1) + m)] * dRl_dx * Ylm_thetaphi[l*(l+1)/2 + m];
								(*grad_V)[1] += (*Vlm_sphere)[2 * (l * (LMAX + 1) + m)] * dRl_dy * Ylm_thetaphi[l*(l+1)/2 + m];
								(*grad_V)[2] += 0.0;
							}

							// Add R*dY terms
							double Rl = pow(r, l);

							for (int m = -l; m < 0; m++)
							{
								(*grad_V)[0] += pow(-1, m) * (*Vlm_sphere)[2 * (l * (LMAX + 1) + m)] * Rl * (
									m * (sin((abs(m)-1) * phi)) / (sin(abs(m) * phi))
								) * Ylm_thetaphi[l*(l+1)/2 + m];
								(*grad_V)[1] += pow(-1, m) * (*Vlm_sphere)[2 * (l * (LMAX + 1) + m)] * Rl * (
									m * (cos((abs(m)-1) * phi)) / (sin(abs(m) * phi))
								) * Ylm_thetaphi[l*(l+1)/2 + m];
							}
							for (int m = 1; m <= l; m++)
							{
								(*grad_V)[0] += (*Vlm_sphere)[2 * (l * (LMAX + 1) + m)] * (
									m * (cos((m-1) * phi)) / (cos(m * phi))
								) * Ylm_thetaphi[l*(l+1)/2 + m];
								(*grad_V)[1] += (*Vlm_sphere)[2 * (l * (LMAX + 1) + m)] * (
									-1 * m * (sin((m-1) * phi)) / (cos(m * phi))
								) * Ylm_thetaphi[l*(l+1)/2 + m];
							}
							for (int m = -l; m < l; m++)
							{
								double Nlm_ratio = sqrt(
									gsl_sf_fact(l-m) / gsl_sf_fact(l+m) * gsl_sf_fact(l+abs(m)+1) / gsl_sf_fact(l-abs(m)-1)
								);
								(*grad_V)[2] += -1 * sqrt(2) * cos(m * phi) * Nlm_ratio * Ylm_thetaphi[l*(l+1)/2 + m];
							}
						}
					}
				}
	}

	// We have to keep the old and new accelerations separate for later use in updating the velocity
	double (*new_acc)[3] = calloc(3, sizeof(double));
	for (int k = 0; k < 3; k++)
	{
		(*new_acc)[k] = -q * (*grad_V)[k]/m;
	}

	for (int k = 0; k < 3; k++)
	{
		*(vel)[k] += ((*acc)[k]+(*new_acc)[k]) * (dt * 0.5);
		
		// We don't need to track the old acceleration after we use it here, so update the primary acceleration array
		*(acc)[k] = (*new_acc)[k];
	}

	free(grad_V);
	free(new_acc);
}

