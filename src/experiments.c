#include "experiments.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"
#include "electrodynamics.h"
#include "waveforms.h"
#include "verlet.h"
#include "analysis.h"
#include "ions.h"

// Run an experiment on an ion in a trap whose electrodynamics have been solved, adding a dimensionless cost to *cost:
// - IDLE holds the ion in a static well at the trap center; the cost penalizes misplacement of the well (per µm),
//   errors in the axial frequency (relative), unstable modes, and heating from voltage noise (per quantum/ms)
// - BACK_AND_FORTH transports the ion away from the trap center and back; the cost is the number of motional quanta
//   excited along the axis plus the final position error (per µm)
void run_experiment
(
	struct Trap *trap,
	struct IonParameters *ion,
	enum Experiment experiment,
	double *cost
)
{
	double m = ion_mass(ion);
	double q = ion_charge(ion);

	double axis[3] = { 1.0, 0.0, 0.0 };

	// Wells sit on the RF null at the trap center
	double center[3] = { 0.0, 0.0, ION_HEIGHT };
	if (trap->n_electrodes_rf > 0)
		find_rf_null(trap, &center, 100);

	if (experiment == IDLE)
	{
		printf("Running idle experiment...\n");

		double *voltages = calloc(trap->n_electrodes, sizeof(double));
		double well[1][3] = { { center[0], center[1], center[2] } };

		double experiment_cost = 0.0;

		if (solve_waveform(trap, m, q, 1, 1, well, AXIAL_FREQUENCY, WAVEFORM_ROUGHNESS_WEIGHT, voltages) != 0)
		{
			experiment_cost += 1E3;
		}
		else
		{
			// Where the ion actually sits
			double minimum[3] = { center[0], center[1], center[2] };
			if (find_potential_minimum(trap, m, q, voltages, &minimum, 100) != 0)
				experiment_cost += 1E2;

			double position_error = sqrt(pow(minimum[0] - center[0], 2) + pow(minimum[1] - center[1], 2) + pow(minimum[2] - center[2], 2));

			double omega[3];
			double axes[3][3];
			compute_normal_modes(trap, m, q, voltages, minimum, &omega, &axes);

			// The mode with the largest overlap with the trap axis is the axial mode
			int axial_mode = 0;
			for (int k = 0; k < 3; k++)
				if (fabs(axes[k][0]) > fabs(axes[axial_mode][0]))
					axial_mode = k;

			double frequency_error = fabs(omega[axial_mode] - AXIAL_FREQUENCY) / AXIAL_FREQUENCY;

			double unstable_penalty = 0.0;
			for (int k = 0; k < 3; k++)
				if (omega[k] < 0)
					unstable_penalty += 10.0;

			// Typical voltage noise from DACs and filters, 1 nV/sqrt(Hz)
			double gamma = heating_rate_voltage_noise(trap, m, q, minimum, AXIAL_FREQUENCY, axis, 1E-18);

			printf("Idle well at (%f, %f, %f), secular frequencies (%f, %f, %f) MHz, voltage noise heating %e quanta/s\n", minimum[0], minimum[1], minimum[2], omega[0] / (2 * PI * 1E6), omega[1] / (2 * PI * 1E6), omega[2] / (2 * PI * 1E6), gamma);

			experiment_cost += position_error + frequency_error + unstable_penalty + gamma * 1E-3;
		}

		printf("Idle experiment cost: %f\n", experiment_cost);
		*cost += experiment_cost;

		free(voltages);
	}
	else if (experiment == BACK_AND_FORTH)
	{
		printf("Running back and forth experiment...\n");

		double distance = 50.0;
		double duration = 2 * distance / (0.5 * TRANSPORT_SPEED_LIMIT);

		// Waveform steps must be much shorter than the axial period to avoid exciting the ion
		int n_steps = (int) ceil(duration * AXIAL_FREQUENCY / (2 * PI) * 10);

		double away[3] = { center[0] + distance, center[1], center[2] };
		if (trap->n_electrodes_rf > 0)
			find_rf_null(trap, &away, 100);

		struct TransportWaveform *waveforms = malloc(2 * sizeof(struct TransportWaveform));

		struct TransportWaveform *waveform_out = generate_transport_waveform(trap, ion, center, away, 0.5 * duration, n_steps, AXIAL_FREQUENCY);
		struct TransportWaveform *waveform_back = generate_transport_waveform(trap, ion, away, center, 0.5 * duration, n_steps, AXIAL_FREQUENCY);

		double experiment_cost = 0.0;

		if (waveform_out == NULL || waveform_back == NULL)
		{
			experiment_cost += 1E3;
		}
		else
		{
			waveforms[0] = *waveform_out;
			waveforms[1] = *waveform_back;
			struct TransportWaveform *waveform = compose_waveforms(2, waveforms, NULL);

			// Integrate with a timestep resolving the fastest secular motion
			double omega_max = AXIAL_FREQUENCY;
			double *voltages = calloc(trap->n_electrodes, sizeof(double));
			double omega[3];
			double axes[3][3];
			compute_normal_modes(trap, m, q, voltages, center, &omega, &axes);
			for (int k = 0; k < 3; k++)
				omega_max = MAX(omega_max, fabs(omega[k]));
			free(voltages);

			double dt = 0.05 / omega_max;
			int n_it = (int) ceil(duration / dt);

			double pos[3] = { center[0], center[1], center[2] };
			double vel[3] = { 0.0, 0.0, 0.0 };
			double acc[3];
			velocity_verlet(trap, m, q, waveform, 0, &pos, &vel, &acc, n_it, dt, NULL, NULL);

			double quanta = motional_quanta(m, AXIAL_FREQUENCY, axis, pos, vel, center);
			double position_error = sqrt(pow(pos[0] - center[0], 2) + pow(pos[1] - center[1], 2) + pow(pos[2] - center[2], 2));

			printf("Back and forth transport excited %f quanta and left the ion %f µm from the trap center\n", quanta, position_error);

			experiment_cost += quanta + position_error;

			free_transport_waveform(waveform);
		}

		printf("Back and forth experiment cost: %f\n", experiment_cost);
		*cost += experiment_cost;

		free_transport_waveform(waveform_out);
		free_transport_waveform(waveform_back);
		free(waveforms);
	}
	else
	{
		printf("run_experiment does not know experiment %d!\n", experiment);
	}
}
