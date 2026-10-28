#include "verlet.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "defs.h"
#include "data_structures.h"
#include "ti_utils.h"
#include "electrodynamics.h"
#include "waveforms.h"
#include "acceleration.h"

// Compute the acceleration (µm/s^2) of an ion of mass m (kg) and charge q (C) at a position (µm) and time t (s), with
// the given DC voltages on the electrodes; the RF drive is either included explicitly (full_rf) or through the
// pseudopotential approximation
void compute_acceleration(
	struct Trap *trap,
	const double m,
	const double q,
	double *voltages,
	const double t,
	const int full_rf,
	double (*pos)[3],
	double (*acc)[3]
)
{
	double grad_U[3];

	if (full_rf)
	{
		double voltages_t[trap->n_electrodes];
		for (int e = 0; e < trap->n_electrodes; e++)
			voltages_t[e] = voltages[e] + ((e < trap->n_electrodes_rf) ? trap->V_RF * cos(trap->Omega_RF * t) : 0.0);

		double grad_phi[3];
		evaluate_trap_potential(trap, voltages_t, *pos, NULL, grad_phi, NULL);

		for (int k = 0; k < 3; k++)
			grad_U[k] = q * grad_phi[k];
	}
	else
	{
		evaluate_potential_energy(trap, m, q, voltages, *pos, NULL, grad_U, NULL);
	}

	// The gradient is in J/µm, so the acceleration is converted from m/s^2 to µm/s^2
	for (int k = 0; k < 3; k++)
		(*acc)[k] = -grad_U[k] / (m * LENGTH_SCALE * LENGTH_SCALE);
}

// Integrate the motion of an ion without reporting progress, as for one member of an ensemble
static void integrate_trajectory(
	struct Trap *trap,
	const double m,
	const double q,
	struct TransportWaveform *waveform,
	const int full_rf,
	double (*pos)[3],
	double (*vel)[3],
	const int n_it,
	const double dt
)
{
	double voltages[trap->n_electrodes];
	for (int e = 0; e < trap->n_electrodes; e++)
		voltages[e] = 0.0;

	double t_0 = (waveform != NULL && waveform->n_waveform_steps > 0) ? waveform->times[0] : 0.0;

	if (waveform != NULL)
		waveform_voltages_at_time(waveform, t_0, voltages);

	double acc[3];
	compute_acceleration(trap, m, q, voltages, t_0, full_rf, pos, &acc);

	for (int t = 0; t < n_it; t++)
	{
		double time = t_0 + (double)(t+1) * dt;

		if (waveform != NULL)
			waveform_voltages_at_time(waveform, time, voltages);

		velocity_verlet_update(trap, m, q, voltages, time, full_rf, pos, vel, &acc, dt);
	}
}

// Integrate the motion of an ensemble of independent ions (e.g. sampled from a thermal state) for n_it steps of dt
// (s), updating their positions (µm) and velocities (µm/s) in place, on the OpenCL device if one is selected and
// otherwise in parallel over the ions
void simulate_ensemble(
	struct Trap *trap,
	const double m,
	const double q,
	struct TransportWaveform *waveform,
	const int full_rf,
	int n_particles,
	double (*positions)[3],
	double (*velocities)[3],
	const int n_it,
	const double dt
)
{
	if (get_compute_device() == OPENCL_DEVICE)
	{
		int n_spheres = NSPH_X * NSPH_Y * NSPH_Z;
		double *coefficients = malloc((size_t) trap->n_electrodes * n_spheres * SOLID_HARMONICS_LEN * 2 * sizeof(double));
		pack_expansion_coefficients(trap, coefficients);

		int n_steps = (waveform != NULL) ? waveform->n_waveform_steps : 0;
		double t_0 = (n_steps > 0) ? waveform->times[0] : 0.0;

		int status = opencl_simulate_ensemble(
			trap->n_electrodes, trap->n_electrodes_rf,
			coefficients,
			trap->V_RF, trap->Omega_RF,
			m, q,
			n_steps,
			(n_steps > 0) ? waveform->times : NULL,
			(n_steps > 0) ? waveform->voltages : NULL,
			full_rf,
			n_particles,
			positions,
			velocities,
			n_it,
			t_0,
			dt
		);

		free(coefficients);

		if (status == 0)
			return;
	}

	#pragma omp parallel for schedule(dynamic)
	for (int p = 0; p < n_particles; p++)
		integrate_trajectory(trap, m, q, waveform, full_rf, &positions[p], &velocities[p], n_it, dt);
}

// Integrate the motion of an ion for n_it steps of dt (s), starting at the first time of the waveform (or at 0 if the
// waveform is NULL, in which case all electrodes are held at 0 V DC); positions and velocities, if not NULL, record the
// trajectory and must have length (n_it + 1)-by-3
void velocity_verlet(
	struct Trap *trap,
	const double m,
	const double q,
	struct TransportWaveform *waveform,
	const int full_rf,
	double (*pos)[3],
	double (*vel)[3],
	double (*acc)[3],
	const int n_it,
	const double dt,
	double (*positions)[3],
	double (*velocities)[3]
)
{
	double *voltages = calloc(trap->n_electrodes, sizeof(double));

	double t_0 = (waveform != NULL && waveform->n_waveform_steps > 0) ? waveform->times[0] : 0.0;

	if (waveform != NULL)
		waveform_voltages_at_time(waveform, t_0, voltages);

	// The acceleration must be consistent with the initial position
	compute_acceleration(trap, m, q, voltages, t_0, full_rf, pos, acc);

	for (int k = 0; k < 3; k++)
	{
		if (positions != NULL)
			positions[0][k] = (*pos)[k];

		if (velocities != NULL)
			velocities[0][k] = (*vel)[k];
	}

	for (int t = 0; t < n_it; t++)
	{
		double time = t_0 + (double)(t+1) * dt;

		// The new acceleration is evaluated with the voltages at the end of the step
		if (waveform != NULL)
			waveform_voltages_at_time(waveform, time, voltages);

		velocity_verlet_update(trap, m, q, voltages, time, full_rf, pos, vel, acc, dt);

		for (int k = 0; k < 3; k++)
		{
			if (positions != NULL)
				positions[t+1][k] = (*pos)[k];

			if (velocities != NULL)
				velocities[t+1][k] = (*vel)[k];
		}

		if ((t+1) % MAX(n_it/10, 1) == 0)
			printf("Position at step %d, time %e: (%f,%f,%f)\n", t+1, time, (*pos)[0], (*pos)[1], (*pos)[2]);
	}

	printf("Finished!\n");

	free(voltages);
}

// Advance the ion by one step of dt, where voltages and t are those at the end of the step
void velocity_verlet_update(
	struct Trap *trap,
	const double m,
	const double q,
	double *voltages,
	const double t,
	const int full_rf,
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

	// We have to keep the old and new accelerations separate for later use in updating the velocity
	double new_acc[3];
	compute_acceleration(trap, m, q, voltages, t, full_rf, pos, &new_acc);

	for (int k = 0; k < 3; k++)
	{
		(*vel)[k] += ((*acc)[k] + new_acc[k]) * (dt * 0.5);

		// We don't need to track the old acceleration after we use it here, so update the primary acceleration array
		(*acc)[k] = new_acc[k];
	}
}
