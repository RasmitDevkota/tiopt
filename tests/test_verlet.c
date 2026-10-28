#include "test_verlet.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"
#include "electrodynamics.h"
#include "verlet.h"
#include "waveforms.h"
#include "traps.h"
#include "ions.h"
#include "test_utils.h"

int test_all_verlet
(
	double atol
)
{
	printf("- Testing velocity Verlet integrator...\n");

	int result = 0;

	result += test_velocity_verlet_uniform_field(atol);
	result += test_velocity_verlet_harmonic(atol);
	result += test_velocity_verlet_full_rf(atol);

	return result;
}

// A trap with a single DC electrode whose potential is given by a function, held at 1 V by a one-step waveform
static void make_function_trap
(
	double (*f)(double position[3], void *f_data),
	void *f_data,
	struct Trap *trap,
	struct TransportWaveform **waveform
)
{
	memset(trap, 0, sizeof(struct Trap));

	allocate_trap(trap, 1);
	trap->n_electrodes_rf = 0;
	trap->n_electrodes_dc = 1;

	build_expansion_from_function(&(*trap->electrodes)[0], f, f_data);

	*waveform = allocate_transport_waveform(1, 1, 0);
	(*waveform)->voltages[0] = 1.0;
}

// phi = E_0 x
static double uniform_field
(
	double position[3],
	void *f_data
)
{
	return *(double *) f_data * position[0];
}

// phi = (kappa / 2) (x^2 - (z - ION_HEIGHT)^2), which confines along x
static double saddle
(
	double position[3],
	void *f_data
)
{
	double kappa = *(double *) f_data;

	return 0.5 * kappa * (position[0]*position[0] - pow(position[2] - ION_HEIGHT, 2));
}

int test_velocity_verlet_uniform_field
(
	double atol
)
{
	struct IonParameters ion;
	generate_ion_from_preset("40Ca+", &ion);

	double m = ion_mass(&ion);
	double q = ion_charge(&ion);

	double E_0 = 1E-6;

	struct Trap trap;
	struct TransportWaveform *waveform;
	make_function_trap(uniform_field, &E_0, &trap, &waveform);

	double pos[3] = { 0.0, 0.0, ION_HEIGHT };
	double vel[3] = { 1E5, 0.0, 0.0 };
	double acc[3];

	int n_it = 1000;
	double dt = 1E-9;
	velocity_verlet(&trap, m, q, waveform, 0, &pos, &vel, &acc, n_it, dt, NULL, NULL);

	// 1. Analytical result, which velocity Verlet reproduces exactly for constant forces
	double a = -q * E_0 / (m * LENGTH_SCALE * LENGTH_SCALE);
	double t = n_it * dt;
	double x_analytical = 1E5 * t + 0.5 * a * t*t;
	double v_analytical = 1E5 + a * t;

	// 2. tiopt result
	double error = fabs(pos[0] - x_analytical) / fabs(x_analytical) + fabs(vel[0] - v_analytical) / fabs(v_analytical);

	free_transport_waveform(waveform);
	free_trap(&trap);

	return check_error("velocity_verlet in uniform field", error, 1E3 * atol);
}

int test_velocity_verlet_harmonic
(
	double atol
)
{
	struct IonParameters ion;
	generate_ion_from_preset("40Ca+", &ion);

	double m = ion_mass(&ion);
	double q = ion_charge(&ion);

	// Curvature (V/µm^2) giving a 1 MHz secular frequency along x
	double omega = 2 * PI * 1E6;
	double kappa = m * omega*omega * LENGTH_SCALE*LENGTH_SCALE / q;

	struct Trap trap;
	struct TransportWaveform *waveform;
	make_function_trap(saddle, &kappa, &trap, &waveform);

	int n_steps_per_period = 200;
	int n_periods = 3;
	int n_it = n_steps_per_period * n_periods;
	double dt = 2 * PI / omega / n_steps_per_period;

	double x_0 = 1.0;
	double pos[3] = { x_0, 0.0, ION_HEIGHT };
	double vel[3] = { 0.0, 0.0, 0.0 };
	double acc[3];

	double (*positions)[3] = malloc((n_it + 1) * sizeof(double[3]));
	double (*velocities)[3] = malloc((n_it + 1) * sizeof(double[3]));
	velocity_verlet(&trap, m, q, waveform, 0, &pos, &vel, &acc, n_it, dt, positions, velocities);

	// The ion returns to its starting point every period, up to the phase error of order (omega dt)^2 / 24 per radian
	double period_error = 0.0;
	for (int p = 1; p <= n_periods; p++)
		period_error = MAX(period_error, fabs(positions[p * n_steps_per_period][0] - x_0) / x_0);

	int result = check_error("velocity_verlet harmonic oscillation period", period_error, 1E-3);

	// Velocity Verlet is symplectic: it conserves a modified energy 1/2 v^2 + 1/2 omega^2 x^2 (1 - (omega dt)^2 / 4), so
	// the true energy oscillates by (omega dt)^2 / 4 instead of drifting
	double energy_error = 0.0;
	double E_0 = 0.5 * omega*omega * x_0*x_0;
	for (int t = 0; t <= n_it; t++)
	{
		double E = 0.5 * velocities[t][0]*velocities[t][0] + 0.5 * omega*omega * positions[t][0]*positions[t][0];
		energy_error = MAX(energy_error, fabs(E - E_0) / E_0);
	}

	result += check_error("velocity_verlet conserves energy", energy_error, pow(omega * dt, 2) / 4 * 1.01);

	free(velocities);
	free(positions);
	free_transport_waveform(waveform);
	free_trap(&trap);

	return result;
}

int test_velocity_verlet_full_rf
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

	double null[3] = { 0.0, 0.0, ION_HEIGHT };
	find_rf_null(&trap, &null, 100);

	struct TransportWaveform *waveform = generate_transport_waveform(&trap, &ion, null, null, 0.0, 1, AXIAL_FREQUENCY);

	double omega[3];
	double axes[3][3];
	compute_normal_modes(&trap, m, q, waveform->voltages, null, &omega, &axes);

	// The mode along z is one of the two radial modes, which are well separated from the axial mode
	int z_mode = 0;
	for (int k = 0; k < 3; k++)
		if (fabs(axes[k][2]) > fabs(axes[z_mode][2]))
			z_mode = k;

	// Displace the ion along z and integrate with the RF drive included explicitly, resolving each RF period
	int n_steps_per_rf_period = 100;
	double T_RF = 2 * PI / trap.Omega_RF;
	double dt = T_RF / n_steps_per_rf_period;
	int n_rf_periods = (int) ceil(20 * trap.Omega_RF / omega[z_mode]);
	int n_it = n_rf_periods * n_steps_per_rf_period;

	double displacement = 0.1;
	double pos[3] = { null[0], null[1], null[2] + displacement };
	double vel[3] = { 0.0, 0.0, 0.0 };
	double acc[3];

	double (*positions)[3] = malloc((n_it + 1) * sizeof(double[3]));
	velocity_verlet(&trap, m, q, waveform, 1, &pos, &vel, &acc, n_it, dt, positions, NULL);

	// Remove the micromotion by averaging over each RF period, then count zero crossings of the secular motion
	int n_crossings = 0;
	double t_first = 0.0;
	double t_last = 0.0;
	double previous = 0.0;
	double max_displacement = 0.0;
	for (int p = 0; p < n_rf_periods; p++)
	{
		double average = 0.0;
		for (int s = 0; s < n_steps_per_rf_period; s++)
		{
			average += positions[p * n_steps_per_rf_period + s][2] - null[2];
			max_displacement = MAX(max_displacement, fabs(positions[p * n_steps_per_rf_period + s][2] - null[2]));
		}
		average /= n_steps_per_rf_period;

		if (p > 0 && (average > 0) != (previous > 0))
		{
			// Interpolate the time of the crossing between the centers of the two RF periods
			double t_crossing = (p - 0.5 + previous / (previous - average)) * T_RF;

			if (n_crossings == 0)
				t_first = t_crossing;
			t_last = t_crossing;

			n_crossings++;
		}

		previous = average;
	}

	double omega_secular = (n_crossings > 1) ? PI * (n_crossings - 1) / (t_last - t_first) : 0.0;

	// Mathieu parameter of the RF drive along z, q_z = 2 q V_RF |phi_RF,zz| / (m Omega_RF^2)
	double *rf_weights = calloc(trap.n_electrodes, sizeof(double));
	for (int e = 0; e < trap.n_electrodes_rf; e++)
		rf_weights[e] = 1.0;

	double grad_rf[3];
	double H_rf[3][3];
	evaluate_trap_potential(&trap, rf_weights, null, NULL, grad_rf, H_rf);

	double q_z = 2 * q * trap.V_RF * fabs(H_rf[2][2]) / (LENGTH_SCALE * LENGTH_SCALE) / (m * trap.Omega_RF * trap.Omega_RF);

	// The pseudopotential approximation omega = q_z Omega_RF / (2 sqrt(2)) is the lowest order of the Mathieu
	// characteristic exponent beta^2 = q^2 / (2 - q^2), so the true secular frequency is higher by about q_z^2 / 4
	double frequency_excess = omega_secular / omega[z_mode] - 1;
	int result = check_error("full RF secular frequency exceeds pseudopotential by Mathieu correction", fabs(frequency_excess - 0.25 * q_z*q_z), 1E-2);

	// The ion starts at rest when the RF field defocuses it along z, i.e. at a minimum of the micromotion envelope
	// z(t) = A cos(omega t) (1 - (q_z / 2) cos(Omega_RF t)), so its largest displacement is (1 + q_z/2) / (1 - q_z/2) times
	// the initial displacement
	double expected_ratio = (1 + 0.5 * q_z) / (1 - 0.5 * q_z);
	result += check_error("full RF micromotion envelope", fabs(max_displacement / displacement - expected_ratio) / expected_ratio, 3E-2);

	free(rf_weights);

	free(positions);
	free_transport_waveform(waveform);
	free_trap(&trap);

	return result;
}
