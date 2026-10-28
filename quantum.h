#pragma once

#include "data_structures.h"

struct PulseProgram* allocate_pulse_program
(
	int n_ions,
	int n_pulses
);

void free_pulse_program
(
	struct PulseProgram *pulse_program
);

double lamb_dicke_parameter
(
	double k_vector[3],
	double mode_axis[3],
	const double m,
	const double omega
);

double saturation_intensity
(
	const double gamma,
	const double wavelength
);

double rabi_frequency_from_intensity
(
	const double gamma,
	const double wavelength,
	const double intensity
);

void displacement_matrix_element
(
	int m,
	int n,
	double alpha_real,
	double alpha_imag,
	double (*element)[2]
);

void motional_coupling
(
	int n_from,
	int n_to,
	const double eta,
	double (*coupling)[2]
);

double thermal_rabi_excitation
(
	const double rabi_frequency,
	const double eta,
	const int sideband,
	const double n_bar,
	const double detuning,
	const double t,
	const int n_max
);

int simulate_pulse_program
(
	struct IonParameters *ion,
	struct MotionalMode *mode,
	struct PulseProgram *pulse_program,
	double *rabi_frequencies, // implicitly, length n_pulses (rad/s)
	double *lamb_dicke_parameters, // implicitly, length n_pulses-by-n_ions
	const double dephasing_rate,
	const int n_samples,
	double *sample_times, // implicitly, length n_samples
	double *excited_populations, // implicitly, length n_samples-by-n_ions
	double *mean_phonon_numbers, // implicitly, length n_samples
	double (*rho_final)[2] // implicitly, length dim-by-dim with dim = 2^n_ions * (n_max + 1); may be NULL
);
