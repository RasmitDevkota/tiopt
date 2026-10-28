#include "quantum.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#include <gsl/gsl_sf_laguerre.h>

#include "defs.h"
#include "data_structures.h"
#include "ti_utils.h"

// Allocate a pulse program for n_pulses pulses on n_ions ions
struct PulseProgram* allocate_pulse_program
(
	int n_ions,
	int n_pulses
)
{
	struct PulseProgram *pulse_program = malloc(sizeof(struct PulseProgram));

	pulse_program->n_ions = n_ions;
	pulse_program->n_pulses = n_pulses;

	pulse_program->pulse_frequencies = calloc(n_pulses, sizeof(double));
	pulse_program->pulse_times = calloc(n_pulses, sizeof(double));
	pulse_program->pulse_powers = calloc(n_pulses, sizeof(double));
	pulse_program->pulse_durations = calloc(n_pulses, sizeof(double));
	pulse_program->pulse_phases = calloc(n_pulses, sizeof(double));
	pulse_program->pulse_n_targets_list = calloc(n_pulses, sizeof(int));
	pulse_program->pulse_targets_list = calloc(n_pulses, sizeof(int *));

	return pulse_program;
}

// Free a pulse program, including the targets of each pulse
void free_pulse_program
(
	struct PulseProgram *pulse_program
)
{
	if (pulse_program == NULL)
		return;

	for (int p = 0; p < pulse_program->n_pulses; p++)
		free(pulse_program->pulse_targets_list[p]);

	free(pulse_program->pulse_targets_list);
	free(pulse_program->pulse_n_targets_list);
	free(pulse_program->pulse_phases);
	free(pulse_program->pulse_durations);
	free(pulse_program->pulse_powers);
	free(pulse_program->pulse_times);
	free(pulse_program->pulse_frequencies);
	free(pulse_program);
}

// Lamb-Dicke parameter eta = |k . b| sqrt(hbar / (2 m omega)) of a laser with wavevector k (1/m) coupling to a mode with
// direction b, mass m (kg) and frequency omega (rad/s)
double lamb_dicke_parameter
(
	double k_vector[3],
	double mode_axis[3],
	const double m,
	const double omega
)
{
	double axis_norm = sqrt(mode_axis[0]*mode_axis[0] + mode_axis[1]*mode_axis[1] + mode_axis[2]*mode_axis[2]);

	double k_dot_b = (k_vector[0] * mode_axis[0] + k_vector[1] * mode_axis[1] + k_vector[2] * mode_axis[2]) / axis_norm;

	return fabs(k_dot_b) * sqrt(HBAR / (2 * m * omega));
}

// Saturation intensity (W/m^2) of a two-level transition with linewidth gamma (1/s) and wavelength (m)
double saturation_intensity
(
	const double gamma,
	const double wavelength
)
{
	return PI * H_PLANCK * C_LIGHT * gamma / (3 * pow(wavelength, 3));
}

// Rabi frequency (rad/s) of a two-level transition with linewidth gamma (1/s) and wavelength (m) driven at an intensity
// (W/m^2), Omega = gamma sqrt(I / (2 I_sat)); polarization and Clebsch-Gordan factors are taken to be unity
double rabi_frequency_from_intensity
(
	const double gamma,
	const double wavelength,
	const double intensity
)
{
	return gamma * sqrt(intensity / (2 * saturation_intensity(gamma, wavelength)));
}

// Matrix element <m|D(alpha)|n> of the displacement operator between Fock states
void displacement_matrix_element
(
	int m,
	int n,
	double alpha_real,
	double alpha_imag,
	double (*element)[2]
)
{
	double x = alpha_real*alpha_real + alpha_imag*alpha_imag;

	if (x == 0.0)
	{
		(*element)[0] = (m == n) ? 1.0 : 0.0;
		(*element)[1] = 0.0;
		return;
	}

	int n_low = MIN(m, n);
	int dn = abs(m - n);

	// sqrt(n_low! / n_high!) |alpha|^dn e^(-|alpha|^2 / 2) L_(n_low)^(dn)(|alpha|^2)
	double magnitude = exp(0.5 * (lgamma(n_low + 1.0) - lgamma(n_low + dn + 1.0)) + 0.5 * dn * log(x) - 0.5 * x) * gsl_sf_laguerre_n(n_low, dn, x);

	// The phase is that of alpha^dn for m >= n and of (-alpha*)^dn for m < n
	double phase = (m >= n) ? dn * atan2(alpha_imag, alpha_real) : dn * atan2(alpha_imag, -alpha_real);

	(*element)[0] = magnitude * cos(phase);
	(*element)[1] = magnitude * sin(phase);
}

// Matrix element <n_to|exp(i eta (a + a^dagger))|n_from> through which a laser couples motional states
void motional_coupling
(
	int n_from,
	int n_to,
	const double eta,
	double (*coupling)[2]
)
{
	displacement_matrix_element(n_to, n_from, 0.0, eta, coupling);
}

// Thermal populations of the Fock states n = 0, ..., n_max
static double thermal_population
(
	const double n_bar,
	const int n
)
{
	if (n_bar <= 0.0)
		return (n == 0) ? 1.0 : 0.0;

	return exp(n * log(n_bar / (n_bar + 1)) - log(n_bar + 1));
}

// Excited state population after driving an ion in a thermal motional state for a time t on a motional sideband
// (0 for the carrier, +1 for the first blue sideband, -1 for the first red sideband, and so on) with the given
// detuning from that sideband, within the resolved sideband approximation
double thermal_rabi_excitation
(
	const double rabi_frequency,
	const double eta,
	const int sideband,
	const double n_bar,
	const double detuning,
	const double t,
	const int n_max
)
{
	double P_e = 0.0;
	double norm = 0.0;

	for (int n = 0; n <= n_max; n++)
	{
		double p_n = thermal_population(n_bar, n);
		norm += p_n;

		if (n + sideband < 0)
			continue;

		double coupling[2];
		motional_coupling(n, n + sideband, eta, &coupling);

		double rabi_frequency_n_sq = pow(rabi_frequency, 2) * (coupling[0]*coupling[0] + coupling[1]*coupling[1]);
		double W_sq = rabi_frequency_n_sq + detuning*detuning;

		if (W_sq > 0.0)
			P_e += p_n * rabi_frequency_n_sq / W_sq * pow(sin(0.5 * sqrt(W_sq) * t), 2);
	}

	return P_e / norm;
}

// A term (Omega/2) e^(i phi) <n_to|D(i eta)|n_from> |e, n_to><g, n_from| of the Hamiltonian, rotating at e^(-i Delta t)
struct CouplingTerm
{
	int pulse;
	int from;
	int to;
	double complex amplitude;
};

struct MasterEquation
{
	int n_ions;
	int n_motion; // number of Fock states kept
	int dim;
	int n_pulses;
	int n_terms;
	struct CouplingTerm *terms; // implicitly, length n_terms
	double *detunings; // detuning of each pulse from the sideband it drives (rad/s); implicitly, length n_pulses
	int *active; // whether each pulse is on during the current segment; implicitly, length n_pulses
	double complex *phases; // e^(-i Delta t) of each active pulse at the current time; implicitly, length n_pulses
	double decay_rate;
	double dephasing_rate;
	double heating_rate;
	// The entries of the Hamiltonian in each row, as the term they come from, the column they are in, and whether they
	// are the Hermitian conjugate of the term; implicitly, length dim+1 for row_start and 2 n_terms for the others
	int *row_start;
	int *row_term;
	int *row_column;
	int *row_conjugate;
	int *qubits; // configuration of the qubits (one bit per ion) of each basis state; implicitly, length dim
	int *phonons; // Fock state of each basis state; implicitly, length dim
	double complex *K; // workspace for H rho; implicitly, length dim-by-dim
};

// Organize the coupling terms by the row of the Hamiltonian they appear in, so that H rho can be computed one row at a
// time (and in parallel)
static void build_hamiltonian_rows
(
	struct MasterEquation *eq
)
{
	eq->row_start = calloc(eq->dim + 1, sizeof(int));
	eq->row_term = malloc(MAX(2 * eq->n_terms, 1) * sizeof(int));
	eq->row_column = malloc(MAX(2 * eq->n_terms, 1) * sizeof(int));
	eq->row_conjugate = malloc(MAX(2 * eq->n_terms, 1) * sizeof(int));

	for (int k = 0; k < eq->n_terms; k++)
	{
		eq->row_start[eq->terms[k].to + 1]++;
		eq->row_start[eq->terms[k].from + 1]++;
	}

	for (int a = 0; a < eq->dim; a++)
		eq->row_start[a + 1] += eq->row_start[a];

	int *filled = calloc(eq->dim, sizeof(int));
	for (int k = 0; k < eq->n_terms; k++)
	{
		int a = eq->terms[k].to;
		int i = eq->row_start[a] + filled[a]++;
		eq->row_term[i] = k;
		eq->row_column[i] = eq->terms[k].from;
		eq->row_conjugate[i] = 0;

		int b = eq->terms[k].from;
		i = eq->row_start[b] + filled[b]++;
		eq->row_term[i] = k;
		eq->row_column[i] = eq->terms[k].to;
		eq->row_conjugate[i] = 1;
	}

	free(filled);
}

// Time derivative of the density matrix in the interaction picture, from the Lindblad master equation; since H and rho
// are Hermitian, -i [H, rho] = -i (K - K^dagger) with K = H rho, and only the upper triangle needs to be computed
static void compute_master_equation_derivative
(
	struct MasterEquation *eq,
	double t,
	double complex *rho,
	double complex *drho
)
{
	int dim = eq->dim;
	int n_motion = eq->n_motion;
	double complex *K = eq->K;

	for (int p = 0; p < eq->n_pulses; p++)
		eq->phases[p] = eq->active[p] ? cexp(-I * eq->detunings[p] * t) : 0.0;

	// K = H rho, one row at a time
	#pragma omp parallel for schedule(static) if (dim >= 48)
	for (int a = 0; a < dim; a++)
	{
		double complex *K_a = &K[(size_t) a * dim];

		for (int j = 0; j < dim; j++)
			K_a[j] = 0.0;

		for (int k = eq->row_start[a]; k < eq->row_start[a + 1]; k++)
		{
			struct CouplingTerm *term = &eq->terms[eq->row_term[k]];

			double complex h = term->amplitude * eq->phases[term->pulse];
			if (h == 0.0)
				continue;

			if (eq->row_conjugate[k])
				h = conj(h);

			const double complex *rho_b = &rho[(size_t) eq->row_column[k] * dim];
			for (int j = 0; j < dim; j++)
				K_a[j] += h * rho_b[j];
		}
	}

	int all_ions = (1 << eq->n_ions) - 1;

	// Coherent evolution, and the spontaneous decay and dephasing of each qubit and the heating of the motional mode
	#pragma omp parallel for schedule(dynamic, 4) if (dim >= 48)
	for (int a = 0; a < dim; a++)
	{
		int c_a = eq->qubits[a];
		int n_a = eq->phonons[a];

		for (int b = a; b < dim; b++)
		{
			int c_b = eq->qubits[b];
			int n_b = eq->phonons[b];

			double complex rho_ab = rho[(size_t) a * dim + b];
			double complex value = -I * (K[(size_t) a * dim + b] - conj(K[(size_t) b * dim + a]));

			if (eq->decay_rate > 0.0)
			{
				value -= 0.5 * eq->decay_rate * (__builtin_popcount(c_a) + __builtin_popcount(c_b)) * rho_ab;

				// Decay from |e> to |g> of each ion which is in |g> in both states
				int ground_in_both = ~(c_a | c_b) & all_ions;
				for (int i = 0; i < eq->n_ions; i++)
					if ((ground_in_both >> i) & 1)
						value += eq->decay_rate * rho[(size_t) ((c_a | (1 << i)) * n_motion + n_a) * dim + (c_b | (1 << i)) * n_motion + n_b];
			}

			if (eq->dephasing_rate > 0.0)
				value -= eq->dephasing_rate * __builtin_popcount(c_a ^ c_b) * rho_ab;

			if (eq->heating_rate > 0.0)
			{
				value -= eq->heating_rate * (n_a + n_b + 1) * rho_ab;

				if (n_a >= 1 && n_b >= 1)
					value += eq->heating_rate * sqrt((double) n_a * n_b) * rho[(size_t) (a - 1) * dim + (b - 1)];

				if (n_a + 1 < n_motion && n_b + 1 < n_motion)
					value += eq->heating_rate * sqrt((n_a + 1.0) * (n_b + 1.0)) * rho[(size_t) (a + 1) * dim + (b + 1)];
			}

			drho[(size_t) a * dim + b] = value;

			if (b != a)
				drho[(size_t) b * dim + a] = conj(value);
		}
	}
}

// Coefficients of the Dormand-Prince 5(4) method (J. R. Dormand and P. J. Prince, J. Comput. Appl. Math. 6, 19 (1980))
static const double dormand_prince_c[7] = { 0.0, 1.0/5, 3.0/10, 4.0/5, 8.0/9, 1.0, 1.0 };
static const double dormand_prince_a[7][6] =
{
	{ 0.0 },
	{ 1.0/5 },
	{ 3.0/40, 9.0/40 },
	{ 44.0/45, -56.0/15, 32.0/9 },
	{ 19372.0/6561, -25360.0/2187, 64448.0/6561, -212.0/729 },
	{ 9017.0/3168, -355.0/33, 46732.0/5247, 49.0/176, -5103.0/18656 },
	{ 35.0/384, 0.0, 500.0/1113, 125.0/192, -2187.0/6784, 11.0/84 }
};
static const double dormand_prince_e[7] = { 71.0/57600, 0.0, -71.0/16695, 71.0/1920, -17253.0/339200, 22.0/525, -1.0/40 };

// Integrate the master equation from t to t_end with adaptive Dormand-Prince steps, starting with step *h (which is
// updated to the last step size used); k[0] must hold the derivative at t on entry. Returns the number of steps taken
static int integrate_master_equation
(
	struct MasterEquation *eq,
	double t,
	double t_end,
	double *h,
	double complex *rho,
	double complex *k[7],
	double complex *rho_stage
)
{
	size_t len = (size_t) eq->dim * eq->dim;
	int n_steps = 0;

	while (t < t_end)
	{
		double step = MIN(*h, t_end - t);
		int last_step = (step >= t_end - t);

		// Stages 2 through 7, where the last stage is evaluated at the fifth-order solution
		for (int stage = 1; stage < 7; stage++)
		{
			#pragma omp parallel for schedule(static) if (len >= 4096)
			for (size_t i = 0; i < len; i++)
			{
				double complex sum = 0.0;
				for (int j = 0; j < stage; j++)
					sum += dormand_prince_a[stage][j] * k[j][i];

				rho_stage[i] = rho[i] + step * sum;
			}

			compute_master_equation_derivative(eq, t + dormand_prince_c[stage] * step, rho_stage, k[stage]);
		}

		// Error of the fourth-order solution relative to the fifth-order one
		double error = 0.0;
		#pragma omp parallel for reduction(max:error) schedule(static) if (len >= 4096)
		for (size_t i = 0; i < len; i++)
		{
			double complex difference = 0.0;
			for (int j = 0; j < 7; j++)
				difference += dormand_prince_e[j] * k[j][i];

			double scale = QUANTUM_TOLERANCE * (1 + MAX(cabs(rho[i]), cabs(rho_stage[i])));
			error = MAX(error, step * cabs(difference) / scale);
		}

		if (error <= 1.0)
		{
			// Accept the step; the derivative at its end is the first stage of the next step
			memcpy(rho, rho_stage, len * sizeof(double complex));
			double complex *swap = k[0];
			k[0] = k[6];
			k[6] = swap;

			t = last_step ? t_end : t + step;
			n_steps++;
		}

		// Standard step size control, which only grows the step if it was not shortened to reach t_end
		double factor = (error > 0.0) ? 0.9 * pow(error, -0.2) : 5.0;
		factor = MIN(MAX(factor, 0.2), 5.0);
		if (!(error <= 1.0 && last_step && factor > 1.0))
			*h = step * factor;

		if (!(*h > 0.0) || isnan(error))
		{
			printf("ERROR: Adaptive integration of the master equation failed at t = %e\n", t);
			break;
		}
	}

	return n_steps;
}

static int compare_times
(
	const void *a,
	const void *b
)
{
	double da = *(const double *) a;
	double db = *(const double *) b;

	return (da > db) - (da < db);
}

// Simulate the qubits of a string of ions sharing one motional mode during a pulse program, by integrating the Lindblad
// master equation in the interaction picture with adaptive steps (to a local tolerance of QUANTUM_TOLERANCE). Each
// pulse drives the motional sideband nearest to its detuning from the S-D transition (keeping all orders in the
// Lamb-Dicke parameter, but neglecting off-resonant sidebands), with the given Rabi frequency, phase and per-ion
// Lamb-Dicke parameters. The ions start in the ground state and the motion in a displaced thermal state; the D state
// decays at gamma_SD, the qubits dephase at the given rate, and the mode heats at its heating rate. Populations and the
// mean phonon number are sampled at n_samples evenly spaced times spanning the pulse program. Returns 0 on success
int simulate_pulse_program
(
	struct IonParameters *ion,
	struct MotionalMode *mode,
	struct PulseProgram *pulse_program,
	double *rabi_frequencies,
	double *lamb_dicke_parameters,
	const double dephasing_rate,
	const int n_samples,
	double *sample_times,
	double *excited_populations,
	double *mean_phonon_numbers,
	double (*rho_final)[2]
)
{
	int n_ions = pulse_program->n_ions;
	int n_pulses = pulse_program->n_pulses;
	int n_motion = mode->n_max + 1;

	if (n_ions < 1 || n_ions > 8 || mode->n_max < 0)
	{
		printf("ERROR: Quantum simulations support 1 to 8 ions and need at least one motional state\n");
		return 1;
	}

	int dim = (1 << n_ions) * n_motion;
	size_t len = (size_t) dim * dim;

	struct MasterEquation eq;
	eq.n_ions = n_ions;
	eq.n_motion = n_motion;
	eq.dim = dim;
	eq.n_pulses = n_pulses;
	eq.decay_rate = ion->gamma_SD;
	eq.dephasing_rate = dephasing_rate;
	eq.heating_rate = mode->heating_rate;
	eq.detunings = calloc(MAX(n_pulses, 1), sizeof(double));
	eq.active = calloc(MAX(n_pulses, 1), sizeof(int));
	eq.phases = calloc(MAX(n_pulses, 1), sizeof(double complex));
	eq.qubits = malloc(dim * sizeof(int));
	eq.phonons = malloc(dim * sizeof(int));
	eq.K = malloc(len * sizeof(double complex));

	for (int a = 0; a < dim; a++)
	{
		eq.qubits[a] = a / n_motion;
		eq.phonons[a] = a % n_motion;
	}

	// The detuning of each pulse from the qubit transition determines which sideband it drives
	int *sidebands = calloc(MAX(n_pulses, 1), sizeof(int));
	double *max_couplings = calloc(MAX(n_pulses, 1), sizeof(double));
	int max_terms = 0;
	for (int p = 0; p < n_pulses; p++)
	{
		double detuning = pulse_program->pulse_frequencies[p] - ion->omega_SD;

		sidebands[p] = (mode->omega > 0.0) ? (int) lround(detuning / mode->omega) : 0;
		eq.detunings[p] = detuning - sidebands[p] * mode->omega;

		max_terms += pulse_program->pulse_n_targets_list[p] * (1 << (n_ions - 1)) * n_motion;
	}

	eq.terms = malloc(MAX(max_terms, 1) * sizeof(struct CouplingTerm));
	eq.n_terms = 0;

	for (int p = 0; p < n_pulses; p++)
	{
		int s = sidebands[p];

		for (int t = 0; t < pulse_program->pulse_n_targets_list[p]; t++)
		{
			int i = pulse_program->pulse_targets_list[p][t] - 1;

			if (i < 0 || i >= n_ions)
			{
				printf("ERROR: Pulse %d targets ion %d, but there are only %d ions\n", p, i + 1, n_ions);
				continue;
			}

			double eta = lamb_dicke_parameters[p * n_ions + i];

			for (int c = 0; c < (1 << n_ions); c++)
			{
				if ((c >> i) & 1)
					continue;

				for (int n = MAX(0, -s); n < n_motion && n + s < n_motion; n++)
				{
					double coupling[2];
					motional_coupling(n, n + s, eta, &coupling);

					struct CouplingTerm *term = &eq.terms[eq.n_terms++];
					term->pulse = p;
					term->from = c * n_motion + n;
					term->to = (c | (1 << i)) * n_motion + n + s;
					term->amplitude = 0.5 * rabi_frequencies[p] * cexp(I * pulse_program->pulse_phases[p]) * (coupling[0] + I * coupling[1]);

					max_couplings[p] = MAX(max_couplings[p], cabs(term->amplitude));
				}
			}
		}
	}

	build_hamiltonian_rows(&eq);

	// Initial state: every qubit in the ground state, and the motion in a displaced thermal state
	double complex *rho = calloc(len, sizeof(double complex));

	double complex *displaced = malloc((size_t) n_motion * n_motion * sizeof(double complex));
	for (int m_1 = 0; m_1 < n_motion; m_1++)
		for (int n = 0; n < n_motion; n++)
		{
			double D[2];
			displacement_matrix_element(m_1, n, mode->alpha[0], mode->alpha[1], &D);
			displaced[m_1 * n_motion + n] = D[0] + I * D[1];
		}

	double trace = 0.0;
	for (int n = 0; n < n_motion; n++)
	{
		double p_n = thermal_population(mode->n_bar, n);

		if (p_n < 1E-300)
			continue;

		for (int m_1 = 0; m_1 < n_motion; m_1++)
			for (int m_2 = 0; m_2 < n_motion; m_2++)
				rho[m_1 * dim + m_2] += p_n * displaced[m_1 * n_motion + n] * conj(displaced[m_2 * n_motion + n]);
	}

	free(displaced);

	for (int m = 0; m < n_motion; m++)
		trace += creal(rho[m * dim + m]);

	for (size_t i = 0; i < len; i++)
		rho[i] /= trace;

	if (1 - trace > 1E-3)
		printf("WARNING: Truncating the motional basis at n_max = %d discards %.2e of the initial state\n", mode->n_max, 1 - trace);

	// Sample times span the pulse program
	double t_end = 0.0;
	for (int p = 0; p < n_pulses; p++)
		t_end = MAX(t_end, pulse_program->pulse_times[p] + pulse_program->pulse_durations[p]);

	for (int s = 0; s < n_samples; s++)
		sample_times[s] = (n_samples > 1) ? t_end * s / (n_samples - 1) : t_end;

	// The Hamiltonian only changes when pulses start or stop, so integrate between those times and the sample times
	int n_event_times;
	double *event_times = find_event_times(pulse_program, &n_event_times);

	int n_breakpoints = 0;
	double *breakpoints = malloc((n_event_times + n_samples + 1) * sizeof(double));
	for (int i = 0; i < n_event_times; i++)
		if (event_times[i] <= t_end)
			breakpoints[n_breakpoints++] = event_times[i];
	for (int s = 0; s < n_samples; s++)
		breakpoints[n_breakpoints++] = sample_times[s];

	qsort(breakpoints, n_breakpoints, sizeof(double), compare_times);

	double complex *k[7];
	for (int stage = 0; stage < 7; stage++)
		k[stage] = malloc(len * sizeof(double complex));
	double complex *rho_stage = malloc(len * sizeof(double complex));

	printf("Simulating %d pulses on %d ions with %d motional states...\n", n_pulses, n_ions, n_motion);

	int n_steps = 0;
	int sample = 0;
	double t = 0.0;
	double h = 0.0;
	for (int b = 0; b <= n_breakpoints; b++)
	{
		double t_next = (b < n_breakpoints) ? breakpoints[b] : t_end;

		if (t_next > t)
		{
			// Pulses which are on during this segment, and an estimate of the fastest rate for the first step
			double t_mid = 0.5 * (t + t_next);
			double rate = eq.n_ions * (eq.decay_rate + eq.dephasing_rate) + eq.heating_rate * (2 * n_motion + 1);
			for (int p = 0; p < n_pulses; p++)
			{
				eq.active[p] = t_mid >= pulse_program->pulse_times[p] && t_mid < pulse_program->pulse_times[p] + pulse_program->pulse_durations[p];

				if (eq.active[p])
					rate += fabs(eq.detunings[p]) + 2 * max_couplings[p] * pulse_program->pulse_n_targets_list[p];
			}

			// Nothing evolves in the interaction picture without pulses or dissipation
			if (rate > 0.0)
			{
				if (h <= 0.0)
					h = QUANTUM_STEP_FACTOR / rate;

				compute_master_equation_derivative(&eq, t, rho, k[0]);
				n_steps += integrate_master_equation(&eq, t, t_next, &h, rho, k, rho_stage);
			}

			t = t_next;
		}

		// Record every sample which has been reached
		while (sample < n_samples && sample_times[sample] <= t)
		{
			double n_mean = 0.0;
			for (int i = 0; i < n_ions; i++)
				excited_populations[sample * n_ions + i] = 0.0;

			for (int a = 0; a < dim; a++)
			{
				double population = creal(rho[(size_t) a * dim + a]);

				n_mean += eq.phonons[a] * population;

				for (int i = 0; i < n_ions; i++)
					if ((eq.qubits[a] >> i) & 1)
						excited_populations[sample * n_ions + i] += population;
			}

			mean_phonon_numbers[sample] = n_mean;

			sample++;
		}
	}

	printf("Finished simulation in %d adaptive steps\n", n_steps);

	// Check that the truncated motional basis was large enough
	double final_trace = 0.0;
	double highest_population = 0.0;
	for (int a = 0; a < dim; a++)
	{
		final_trace += creal(rho[(size_t) a * dim + a]);

		if (eq.phonons[a] == n_motion - 1)
			highest_population += creal(rho[(size_t) a * dim + a]);
	}

	if (fabs(final_trace - 1) > 1E-3 || highest_population > 1E-3)
		printf("WARNING: Motional basis may be too small (trace %.6f, population %.2e in n_max = %d)\n", final_trace, highest_population, mode->n_max);

	if (rho_final != NULL)
		for (size_t i = 0; i < len; i++)
		{
			rho_final[i][0] = creal(rho[i]);
			rho_final[i][1] = cimag(rho[i]);
		}

	free(rho_stage);
	for (int stage = 0; stage < 7; stage++)
		free(k[stage]);
	free(breakpoints);
	free(event_times);
	free(rho);
	free(eq.row_conjugate);
	free(eq.row_column);
	free(eq.row_term);
	free(eq.row_start);
	free(eq.terms);
	free(max_couplings);
	free(sidebands);
	free(eq.K);
	free(eq.phonons);
	free(eq.qubits);
	free(eq.phases);
	free(eq.active);
	free(eq.detunings);

	return 0;
}
