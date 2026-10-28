#include "test_analysis.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"
#include "electrodynamics.h"
#include "analysis.h"
#include "traps.h"
#include "ions.h"
#include "ion_heating/ion_heating.h"
#include "test_utils.h"

int test_all_analysis
(
	double atol
)
{
	printf("- Testing analysis module...\n");

	int result = 0;

	result += test_heating_rate(atol);
	result += test_heating_rate_rf_noise(atol);
	result += test_heating_rate_voltage_noise(atol);
	result += test_motional_quanta(atol);
	result += test_ion_heating(atol);
	result += test_compute_trap_depth(atol);

	return result;
}

int test_heating_rate
(
	double atol
)
{
	double q = E_CHARGE;
	double m = 40 * AMU;
	double Omega_RF = 2 * PI * 40E6;
	double omega = 2 * PI * 1E6;
	double dE2_dz = 3E14;
	double S_V_N = 1E-12;
	double V_0 = 40.0;

	// 1. Analytical result: fractional amplitude noise modulates the pseudopotential q^2 E^2 / (4 m Omega^2), so the force
	// noise is S_F = (q^2 dE2_dz / (2 m Omega^2))^2 S_V_N / V_0^2 and the heating rate is S_F / (4 m hbar omega)
	double S_F = pow(q*q * dE2_dz / (2 * m * Omega_RF*Omega_RF), 2) * S_V_N / (V_0*V_0);
	double gamma_analytical = S_F / (4 * m * HBAR * omega);

	// 2. tiopt result
	double gamma = heating_rate(q, m, Omega_RF, omega, dE2_dz, S_V_N, V_0);

	return check_error("heating_rate from RF amplitude noise", fabs(gamma - gamma_analytical) / gamma_analytical, atol);
}

int test_heating_rate_rf_noise
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

	double omega = 2 * PI * 4E6;
	double axis[3] = { 0.0, 0.0, 1.0 };
	double S_V_N = 1E-12;

	// The RF field gradient, and so the heating, vanishes at the RF null
	double null[3] = { 0.0, 0.0, ION_HEIGHT };
	find_rf_null(&trap, &null, 100);

	double gamma_null = heating_rate_rf_noise(&trap, m, q, null, omega, axis, S_V_N);

	// Away from the null, compare with a numerical derivative of |E_RF|^2 along z
	double position[3] = { null[0], null[1], null[2] + 2.0 };
	double gamma = heating_rate_rf_noise(&trap, m, q, position, omega, axis, S_V_N);

	double *rf_weights = calloc(trap.n_electrodes, sizeof(double));
	rf_weights[0] = 1.0;
	rf_weights[1] = 1.0;

	double h = 1E-3;
	double E2[2];
	for (int s = 0; s < 2; s++)
	{
		double p[3] = { position[0], position[1], position[2] + (s == 0 ? h : -h) };

		double g[3];
		evaluate_trap_potential(&trap, rf_weights, p, NULL, g, NULL);

		E2[s] = pow(trap.V_RF / LENGTH_SCALE, 2) * (g[0]*g[0] + g[1]*g[1] + g[2]*g[2]);
	}

	double dE2_dz = (E2[0] - E2[1]) / (2 * h * LENGTH_SCALE);
	double gamma_finite_differences = heating_rate(q, m, trap.Omega_RF, omega, dE2_dz, S_V_N, trap.V_RF);

	int result = check_error("RF noise heating vanishes at RF null", gamma_null / gamma, 1E-6);
	result += check_error("RF noise heating away from RF null", fabs(gamma - gamma_finite_differences) / gamma, 1E-5);

	free(rf_weights);
	free_trap(&trap);

	return result;
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

int test_heating_rate_voltage_noise
(
	double atol
)
{
	struct IonParameters ion;
	generate_ion_from_preset("40Ca+", &ion);

	double m = ion_mass(&ion);
	double q = ion_charge(&ion);

	// A single DC electrode producing a uniform field E_0 per volt along x
	double E_0 = 2E-3;

	struct Trap trap;
	memset(&trap, 0, sizeof(struct Trap));
	allocate_trap(&trap, 1);
	trap.n_electrodes_dc = 1;
	build_expansion_from_function(&(*trap.electrodes)[0], uniform_field, &E_0);

	double position[3] = { 0.0, 0.0, ION_HEIGHT };
	double axis[3] = { 2.0, 0.0, 0.0 };
	double omega = 2 * PI * 1E6;
	double S_V = 1E-18;

	// 1. Analytical result, Gamma = q^2 S_E / (4 m hbar omega) with S_E = (E_0 / L)^2 S_V
	double gamma_analytical = q*q * pow(E_0 / LENGTH_SCALE, 2) * S_V / (4 * m * HBAR * omega);

	// 2. tiopt result
	double gamma = heating_rate_voltage_noise(&trap, m, q, position, omega, axis, S_V);

	// Noise along a perpendicular mode does not heat it
	double perpendicular_axis[3] = { 0.0, 1.0, 0.0 };
	double gamma_perpendicular = heating_rate_voltage_noise(&trap, m, q, position, omega, perpendicular_axis, S_V);

	free_trap(&trap);

	int result = check_error("voltage noise heating of uniform field", fabs(gamma - gamma_analytical) / gamma_analytical, 1E3 * atol);
	result += check_error("voltage noise does not heat perpendicular mode", gamma_perpendicular / gamma_analytical, 1E3 * atol);

	return result;
}

int test_motional_quanta
(
	double atol
)
{
	double m = 40 * AMU;
	double omega = 2 * PI * 1E6;

	// A coherent displacement x_0 holds m omega x_0^2 / (2 hbar) quanta, whether as position or as velocity
	double x_0 = 0.05;
	double axis[3] = { 1.0, 0.0, 0.0 };
	double equilibrium[3] = { 10.0, 0.0, ION_HEIGHT };
	double position[3] = { 10.0 + x_0, 3.0, ION_HEIGHT };
	double velocity[3] = { 0.0, 1E5, 0.0 };
	double velocity_only[3] = { omega * x_0, 0.0, 0.0 };

	double quanta_analytical = m * omega * pow(x_0 * LENGTH_SCALE, 2) / (2 * HBAR);

	double error = fabs(motional_quanta(m, omega, axis, position, velocity, equilibrium) - quanta_analytical) / quanta_analytical;
	error += fabs(motional_quanta(m, omega, axis, equilibrium, velocity_only, equilibrium) - quanta_analytical) / quanta_analytical;

	return check_error("motional_quanta of displaced ion", error, atol);
}

static void path_origin
(
	double t,
	void *ctx,
	double r[3]
)
{
	r[0] = 0.0;
	r[1] = 0.0;
	r[2] = 0.0;
}

struct OscillatingFieldData
{
	double amplitude;
	double omega;
	double offset;
};

static void oscillating_field
(
	double t,
	const double r[3],
	void *ctx,
	double e[3]
)
{
	struct OscillatingFieldData *data = ctx;

	e[0] = data->amplitude * sin(data->omega * t) + data->offset;
	e[1] = data->offset;
	e[2] = 0.0;
}

int test_ion_heating
(
	double atol
)
{
	spectrum_opts_t options;
	ion_spectrum_opts_default(&options);
	options.dt = 1E-8;
	options.n = 8192;

	// A tone exactly on a frequency bin carries power amplitude^2 / 2 in the one-sided spectrum
	double amplitude = 100.0;
	double f_tone = 64 / (options.n * options.dt);

	double *e = malloc(options.n * sizeof(double));
	for (int i = 0; i < options.n; i++)
		e[i] = amplitude * sin(2 * PI * f_tone * i * options.dt);

	double *f_hz;
	double *psd;
	int n_bins = ion_compute_psd(e, &options, &f_hz, &psd);

	double power = 0.0;
	for (int k = 0; k < n_bins; k++)
		power += psd[k] / (options.n * options.dt);

	int result = check_error("ion_compute_psd conserves power of tone", fabs(power - 0.5 * amplitude*amplitude) / (0.5 * amplitude*amplitude), 2E-2);

	free(psd);
	free(f_hz);
	free(e);

	// A field along x only heats the mode along x, and a constant field is removed before computing the spectrum
	ion_params3_t ion_3d;
	ion_3d.q = E_CHARGE;
	ion_3d.m = 40 * AMU;
	ion_3d.omega_sec[0] = 2 * PI * 1E6;
	ion_3d.omega_sec[1] = 2 * PI * 1.2E6;
	ion_3d.omega_sec[2] = 2 * PI * 0.8E6;
	ion_3d.axes = NULL;

	struct OscillatingFieldData field_data = { amplitude, ion_3d.omega_sec[0], 0.0 };
	double gamma[3];
	ion_heating_rates_from_path_field_3d(0.0, &ion_3d, &options, path_origin, NULL, oscillating_field, &field_data, gamma);

	result += check_error("field along x only heats x mode", (gamma[0] > 0.0 && gamma[1] < 1E-6 * gamma[0] && gamma[2] == 0.0) ? 0 : 1, 0);

	struct OscillatingFieldData constant_field_data = { 0.0, 0.0, 50.0 };
	ion_heating_rates_from_path_field_3d(0.0, &ion_3d, &options, path_origin, NULL, oscillating_field, &constant_field_data, gamma);

	result += check_error("constant field does not heat", gamma[0] + gamma[1] + gamma[2], atol);

	return result;
}

int test_compute_trap_depth
(
	double atol
)
{
	// Two RF rails between y = a and y = b and between y = -b and y = -a, long enough to be treated as infinite
	double a = 30.0;
	double b = 100.0;
	double half_length = 1E5;

	struct Trap trap;
	memset(&trap, 0, sizeof(struct Trap));
	allocate_trap(&trap, 2);
	trap.n_electrodes_rf = 2;
	trap.V_RF = 50.0;
	trap.Omega_RF = 2 * PI * 30E6;

	set_rectangular_electrode(&trap, 0, -half_length, half_length, a, b);
	set_rectangular_electrode(&trap, 1, -half_length, half_length, -b, -a);

	double m = 40 * AMU;
	double q = E_CHARGE;

	// 1. Analytical result: above the center, the potential of an infinite strip from y = a to y = b in the gapless plane
	// is (atan(b / z) - atan(a / z)) / PI, so the field of the rails is E_z = (2 / PI) (a / (z^2 + a^2) - b / (z^2 + b^2))
	// per volt, which vanishes at the RF null z = sqrt(a b) and peaks above it
	double rf_null[3] = { 0.0, 0.0, sqrt(a * b) };
	double prefactor = pow(q * trap.V_RF, 2) / (4 * m * pow(trap.Omega_RF * LENGTH_SCALE, 2));

	double U_barrier = 0.0;
	for (int i = 0; i <= 100000; i++)
	{
		double z = rf_null[2] + 5 * ION_HEIGHT * i / 100000.0;
		double E_z = 2 / PI * (a / (z*z + a*a) - b / (z*z + b*b));

		U_barrier = MAX(U_barrier, prefactor * E_z*E_z);
	}

	double depth_analytical = U_barrier / E_CHARGE;

	// 2. tiopt result
	double depth = compute_trap_depth(&trap, m, q, rf_null);

	free_trap(&trap);

	return check_error("compute_trap_depth of infinite rails", fabs(depth - depth_analytical) / depth_analytical, 1E-3);
}
