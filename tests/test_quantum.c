#include "test_quantum.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"
#include "quantum.h"
#include "ions.h"
#include "test_utils.h"

int test_all_quantum
(
	double atol
)
{
	printf("- Testing quantum dynamics module...\n");

	int result = 0;

	result += test_displacement_matrix_element(atol);
	result += test_laser_coupling(atol);
	result += test_simulate_rabi_oscillations(atol);
	result += test_simulate_dissipation(atol);
	result += test_simulate_ramsey(atol);
	result += test_simulate_molmer_sorensen_gate(atol);

	return result;
}

// A pulse program of consecutive single-ion pulses with the given detunings from the S-D transition
static struct PulseProgram* make_pulse_program
(
	struct IonParameters *ion,
	int n_ions,
	int n_pulses,
	double *detunings,
	double *durations,
	double *phases
)
{
	struct PulseProgram *pulse_program = allocate_pulse_program(n_ions, n_pulses);

	double time = 0.0;
	for (int p = 0; p < n_pulses; p++)
	{
		pulse_program->pulse_frequencies[p] = ion->omega_SD + detunings[p];
		pulse_program->pulse_times[p] = time;
		pulse_program->pulse_durations[p] = durations[p];
		pulse_program->pulse_phases[p] = (phases != NULL) ? phases[p] : 0.0;
		pulse_program->pulse_n_targets_list[p] = n_ions;
		pulse_program->pulse_targets_list[p] = malloc(n_ions * sizeof(int));

		for (int i = 0; i < n_ions; i++)
			pulse_program->pulse_targets_list[p][i] = i + 1;

		time += durations[p];
	}

	return pulse_program;
}

int test_displacement_matrix_element
(
	double atol
)
{
	double alpha[2] = { 0.3, -0.7 };
	double alpha_sq = alpha[0]*alpha[0] + alpha[1]*alpha[1];

	// <1|D(alpha)|0> = alpha e^(-|alpha|^2/2) and <0|D(alpha)|1> = -alpha* e^(-|alpha|^2/2)
	double D_10[2];
	double D_01[2];
	displacement_matrix_element(1, 0, alpha[0], alpha[1], &D_10);
	displacement_matrix_element(0, 1, alpha[0], alpha[1], &D_01);

	double error = fabs(D_10[0] - alpha[0] * exp(-0.5 * alpha_sq)) + fabs(D_10[1] - alpha[1] * exp(-0.5 * alpha_sq));
	error += fabs(D_01[0] + alpha[0] * exp(-0.5 * alpha_sq)) + fabs(D_01[1] - alpha[1] * exp(-0.5 * alpha_sq));

	// D is unitary, so each of its columns is normalized and different columns are orthogonal
	for (int n = 0; n < 5; n++)
	{
		double norm = 0.0;
		double overlap[2] = { 0.0, 0.0 };
		for (int m = 0; m < 60; m++)
		{
			double D_mn[2];
			double D_m0[2];
			displacement_matrix_element(m, n, alpha[0], alpha[1], &D_mn);
			displacement_matrix_element(m, 0, alpha[0], alpha[1], &D_m0);

			norm += D_mn[0]*D_mn[0] + D_mn[1]*D_mn[1];
			overlap[0] += D_m0[0] * D_mn[0] + D_m0[1] * D_mn[1];
			overlap[1] += D_m0[0] * D_mn[1] - D_m0[1] * D_mn[0];
		}

		error = MAX(error, fabs(norm - 1));

		if (n > 0)
			error = MAX(error, hypot(overlap[0], overlap[1]));
	}

	return check_error("displacement matrix elements", error, 1E3 * atol);
}

int test_laser_coupling
(
	double atol
)
{
	struct IonParameters ion;
	generate_ion_from_preset("40Ca+", &ion);

	double m = ion_mass(&ion);
	double omega = 2 * PI * 1E6;
	double wavelength = 2 * PI * C_LIGHT / ion.omega_SD;

	// A laser along the mode gives eta = k sqrt(hbar / (2 m omega))
	double k = 2 * PI / wavelength;
	double k_vector[3] = { 0.0, 0.6 * k, 0.8 * k };
	double mode_axis[3] = { 0.0, 3.0, 4.0 };
	double eta = lamb_dicke_parameter(k_vector, mode_axis, m, omega);

	double error = fabs(eta - k * sqrt(HBAR / (2 * m * omega))) / eta;

	// At twice the saturation intensity, the Rabi frequency equals the linewidth
	double gamma = ion.gamma_SP;
	double wavelength_SP = 2 * PI * C_LIGHT / ion.omega_SP;
	double rabi_frequency = rabi_frequency_from_intensity(gamma, wavelength_SP, 2 * saturation_intensity(gamma, wavelength_SP));
	error += fabs(rabi_frequency - gamma) / gamma;

	// The saturation intensity of the 87Rb D2 cycling transition is 16.69 W/m^2 (D. A. Steck, Rubidium 87 D Line Data)
	error += fabs(saturation_intensity(2 * PI * 6.0666E6, 780.241E-9) - 16.69) / 16.69 > 1E-3;

	int result = check_error("Lamb-Dicke parameter and Rabi frequency", error, 1E3 * atol);

	// Without motion, the carrier flops as sin^2(Omega e^(-eta^2/2) t / 2)
	double rabi = 2 * PI * 100E3;
	double flop_error = 0.0;
	for (double t = 0.0; t < 30E-6; t += 1.3E-6)
		flop_error = MAX(flop_error, fabs(thermal_rabi_excitation(rabi, 0.1, 0, 0.0, 0.0, t, 10) - pow(sin(0.5 * rabi * exp(-0.005) * t), 2)));

	result += check_error("thermal_rabi_excitation for ground state", flop_error, atol);

	return result;
}

int test_simulate_rabi_oscillations
(
	double atol
)
{
	struct IonParameters ion;
	generate_ion_from_preset("40Ca+", &ion);
	ion.gamma_SD = 0.0;

	double nu = 2 * PI * 1E6;
	double rabi_frequency = 2 * PI * 100E3;
	double eta = 0.1;

	// Detuned carrier pulse on a thermal state
	struct MotionalMode mode = { nu, 20, 0.5, { 0.0, 0.0 }, 0.0 };

	double detuning = 2 * PI * 5E3;
	double duration = 30E-6;
	struct PulseProgram *pulse_program = make_pulse_program(&ion, 1, 1, &detuning, &duration, NULL);

	int n_samples = 61;
	double sample_times[61];
	double excited_populations[61];
	double mean_phonon_numbers[61];
	simulate_pulse_program(&ion, &mode, pulse_program, &rabi_frequency, &eta, 0.0, n_samples, sample_times, excited_populations, mean_phonon_numbers, NULL);

	double error = 0.0;
	for (int s = 0; s < n_samples; s++)
		error = MAX(error, fabs(excited_populations[s] - thermal_rabi_excitation(rabi_frequency, eta, 0, mode.n_bar, detuning, sample_times[s], mode.n_max)));

	int result = check_error("simulated thermal carrier matches analytical result", error, 1E-6);

	free_pulse_program(pulse_program);

	// Blue sideband from the motional ground state flops at eta Omega e^(-eta^2/2) and adds one phonon per excitation
	mode.n_bar = 0.0;
	mode.n_max = 10;

	double blue_detuning = nu;
	pulse_program = make_pulse_program(&ion, 1, 1, &blue_detuning, &duration, NULL);
	simulate_pulse_program(&ion, &mode, pulse_program, &rabi_frequency, &eta, 0.0, n_samples, sample_times, excited_populations, mean_phonon_numbers, NULL);

	error = 0.0;
	double sideband_rabi_frequency = rabi_frequency * eta * exp(-0.5 * eta*eta);
	for (int s = 0; s < n_samples; s++)
	{
		double P_e = pow(sin(0.5 * sideband_rabi_frequency * sample_times[s]), 2);

		error = MAX(error, fabs(excited_populations[s] - P_e) + fabs(mean_phonon_numbers[s] - P_e));
	}

	result += check_error("simulated blue sideband from ground state", error, 1E-6);

	// Red sideband cannot excite the motional ground state
	double red_detuning = -nu;
	free_pulse_program(pulse_program);
	pulse_program = make_pulse_program(&ion, 1, 1, &red_detuning, &duration, NULL);
	simulate_pulse_program(&ion, &mode, pulse_program, &rabi_frequency, &eta, 0.0, n_samples, sample_times, excited_populations, mean_phonon_numbers, NULL);

	error = 0.0;
	for (int s = 0; s < n_samples; s++)
		error = MAX(error, excited_populations[s]);

	result += check_error("red sideband does not excite ground state", error, atol);

	free_pulse_program(pulse_program);

	return result;
}

int test_simulate_dissipation
(
	double atol
)
{
	struct IonParameters ion;
	generate_ion_from_preset("40Ca+", &ion);

	double nu = 2 * PI * 1E6;
	double eta = 0.05;

	// Heating raises the mean phonon number linearly
	ion.gamma_SD = 0.0;
	struct MotionalMode mode = { nu, 40, 0.2, { 0.0, 0.0 }, 1E4 };

	double detuning = 0.0;
	double duration = 100E-6;
	double rabi_frequency = 0.0;
	struct PulseProgram *pulse_program = make_pulse_program(&ion, 1, 1, &detuning, &duration, NULL);

	int n_samples = 11;
	double sample_times[11];
	double excited_populations[11];
	double mean_phonon_numbers[11];
	simulate_pulse_program(&ion, &mode, pulse_program, &rabi_frequency, &eta, 0.0, n_samples, sample_times, excited_populations, mean_phonon_numbers, NULL);

	double error = 0.0;
	for (int s = 0; s < n_samples; s++)
		error = MAX(error, fabs(mean_phonon_numbers[s] - (mode.n_bar + mode.heating_rate * sample_times[s])));

	int result = check_error("motional heating", error, 1E-6);

	free_pulse_program(pulse_program);

	// After a pi pulse, the excited state decays at gamma_SD
	ion.gamma_SD = 1E4;
	mode.heating_rate = 0.0;
	mode.n_bar = 0.0;
	mode.n_max = 2;

	double rabi_frequencies[2] = { 2 * PI * 5E6, 0.0 };
	double detunings[2] = { 0.0, 0.0 };
	double durations[2] = { 0.5 / 5E6, 200E-6 };
	double etas[2] = { 0.0, 0.0 };
	pulse_program = make_pulse_program(&ion, 1, 2, detunings, durations, NULL);

	n_samples = 11;
	simulate_pulse_program(&ion, &mode, pulse_program, rabi_frequencies, etas, 0.0, n_samples, sample_times, excited_populations, mean_phonon_numbers, NULL);

	// The pi pulse is short compared with the lifetime, so only allow for decay during it
	error = 0.0;
	for (int s = 1; s < n_samples; s++)
		error = MAX(error, fabs(excited_populations[s] - exp(-ion.gamma_SD * (sample_times[s] - 0.5 * durations[0]))));

	result += check_error("spontaneous decay", error, ion.gamma_SD * durations[0]);

	free_pulse_program(pulse_program);

	return result;
}

int test_simulate_ramsey
(
	double atol
)
{
	struct IonParameters ion;
	generate_ion_from_preset("40Ca+", &ion);
	ion.gamma_SD = 0.0;

	struct MotionalMode mode = { 2 * PI * 1E6, 0, 0.0, { 0.0, 0.0 }, 0.0 };

	// pi/2 pulse, free evolution, and pi/2 pulse: with dephasing, the coherence and hence the excitation decay; the
	// pulses are short so that little dephasing happens during them
	double rabi = 2 * PI * 25E6;
	double t_pi_2 = 0.25 / 25E6;
	double wait = 20E-6;
	double dephasing_rate = 3E4;

	double rabi_frequencies[3] = { rabi, 0.0, rabi };
	double detunings[3] = { 0.0, 0.0, 0.0 };
	double durations[3] = { t_pi_2, wait, t_pi_2 };
	double etas[3] = { 0.0, 0.0, 0.0 };
	struct PulseProgram *pulse_program = make_pulse_program(&ion, 1, 3, detunings, durations, NULL);

	double sample_times[2];
	double excited_populations[2];
	double mean_phonon_numbers[2];
	simulate_pulse_program(&ion, &mode, pulse_program, rabi_frequencies, etas, dephasing_rate, 2, sample_times, excited_populations, mean_phonon_numbers, NULL);

	// Dephasing also acts during the pulses
	double coherence_time = wait + 2 * t_pi_2;
	double expected = 0.5 * (1 + exp(-dephasing_rate * wait));
	int result = check_error("Ramsey contrast decays with dephasing", fabs(excited_populations[1] - expected), dephasing_rate * (coherence_time - wait));

	// Shifting the phase of the second pulse by PI returns the ion to the ground state without dephasing
	pulse_program->pulse_phases[2] = PI;
	simulate_pulse_program(&ion, &mode, pulse_program, rabi_frequencies, etas, 0.0, 2, sample_times, excited_populations, mean_phonon_numbers, NULL);
	result += check_error("Ramsey phase reverses excitation", excited_populations[1], 1E-6);

	free_pulse_program(pulse_program);

	return result;
}

int test_simulate_molmer_sorensen_gate
(
	double atol
)
{
	struct IonParameters ion;
	generate_ion_from_preset("40Ca+", &ion);
	ion.gamma_SD = 0.0;

	double nu = 2 * PI * 1E6;

	// Bichromatic light detuned by delta from both sidebands entangles the ions after 2 PI / delta when eta Omega = delta / 2
	double delta = 2 * PI * 20E3;
	double eta = 0.05;

	struct PulseProgram *pulse_program = allocate_pulse_program(2, 2);
	for (int p = 0; p < 2; p++)
	{
		pulse_program->pulse_times[p] = 0.0;
		pulse_program->pulse_durations[p] = 2 * PI / delta;
		pulse_program->pulse_n_targets_list[p] = 2;
		pulse_program->pulse_targets_list[p] = malloc(2 * sizeof(int));
		pulse_program->pulse_targets_list[p][0] = 1;
		pulse_program->pulse_targets_list[p][1] = 2;
	}
	pulse_program->pulse_frequencies[0] = ion.omega_SD + nu + delta;
	pulse_program->pulse_frequencies[1] = ion.omega_SD - nu - delta;

	struct MotionalMode mode = { nu, 12, 0.0, { 0.0, 0.0 }, 0.0 };

	double rabi_frequencies[2] = { delta / (2 * eta), delta / (2 * eta) };
	double etas[4] = { eta, eta, eta, eta };

	int n_motion = mode.n_max + 1;
	int dim = 4 * n_motion;
	double (*rho)[2] = malloc(dim * dim * sizeof(double[2]));

	double sample_times[2];
	double excited_populations[4];
	double mean_phonon_numbers[2];
	simulate_pulse_program(&ion, &mode, pulse_program, rabi_frequencies, etas, 0.0, 2, sample_times, excited_populations, mean_phonon_numbers, rho);

	// Fidelity with the Bell state (|gg> + e^(i phase) |ee>) / sqrt(2) for the best phase
	double P_gg = 0.0;
	double P_ee = 0.0;
	double coherence[2] = { 0.0, 0.0 };
	for (int n = 0; n < n_motion; n++)
	{
		P_gg += rho[n * dim + n][0];
		P_ee += rho[(3 * n_motion + n) * dim + 3 * n_motion + n][0];
		coherence[0] += rho[n * dim + 3 * n_motion + n][0];
		coherence[1] += rho[n * dim + 3 * n_motion + n][1];
	}

	double fidelity = 0.5 * (P_gg + P_ee) + hypot(coherence[0], coherence[1]);

	int result = check_error("Molmer-Sorensen gate creates Bell state", 1 - fidelity, 1E-3);
	result += check_error("Molmer-Sorensen gate disentangles motion", mean_phonon_numbers[1], 1E-3);

	free(rho);
	free_pulse_program(pulse_program);

	return result;
}
