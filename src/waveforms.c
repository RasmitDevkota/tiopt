#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <nlopt.h>
#include <gsl/gsl_linalg.h>

#include "waveforms.h"
#include "defs.h"
#include "data_structures.h"
#include "ti_utils.h"
#include "electrodynamics.h"
#include "ions.h"

// Allocate a waveform with n_waveform_steps steps for n_electrodes electrodes and n_wells wells
struct TransportWaveform* allocate_transport_waveform
(
	int n_waveform_steps,
	int n_electrodes,
	int n_wells
)
{
	struct TransportWaveform *waveform = malloc(sizeof(struct TransportWaveform));

	waveform->n_waveform_steps = n_waveform_steps;
	waveform->n_electrodes = n_electrodes;
	waveform->n_wells = n_wells;

	waveform->voltages = calloc(n_waveform_steps * n_electrodes, sizeof(double));
	waveform->times = calloc(n_waveform_steps, sizeof(double));
	waveform->well_positions = calloc(n_waveform_steps * n_wells, sizeof(double[3]));

	return waveform;
}

// Free a waveform and its arrays
void free_transport_waveform
(
	struct TransportWaveform *waveform
)
{
	if (waveform == NULL)
		return;

	free(waveform->voltages);
	free(waveform->times);
	free(waveform->well_positions);
	free(waveform);
}

// Cost function for the DC voltages of every step of a waveform, laid out as n_steps-by-n_voltages
double waveform_cost_function(
	unsigned n,
	const double *x,
	double *grad,
	void *cost_data
)
{
	struct WaveformCostData *w = cost_data;

	int n_v = w->n_voltages;

	double cost = 0.0;

	if (grad)
		for (unsigned i = 0; i < n; i++)
			grad[i] = 0.0;

	// Penalize large voltages
	for (unsigned i = 0; i < n; i++)
	{
		cost += w->voltage_weight * x[i]*x[i];

		if (grad)
			grad[i] += 2 * w->voltage_weight * x[i];
	}

	// Penalize waveform roughness: sum(V(t+1)-2V(t)+V(t-1))^2
	for (int t = 1; t < w->n_steps-1; t++)
		for (int e = 0; e < n_v; e++)
		{
			int k = t*n_v + e;

			double d2 = x[k + n_v] - 2*x[k] + x[k - n_v];

			cost += w->roughness_weight * d2*d2;

			if (grad)
			{
				grad[k + n_v] += 2 * w->roughness_weight * d2;
				grad[k] -= 4 * w->roughness_weight * d2;
				grad[k - n_v] += 2 * w->roughness_weight * d2;
			}
		}

	return cost;
}

// Vector-valued equality constraint for nlopt which requires the voltages of one step to satisfy that step's linear
// well constraints
void waveform_well_constraints
(
	unsigned m_constraints,
	double *result,
	unsigned n,
	const double *x,
	double *grad,
	void *constraint_data
)
{
	struct WaveformConstraintData *d = constraint_data;

	for (unsigned r = 0; r < m_constraints; r++)
	{
		double value = -d->targets[r];

		for (int v = 0; v < d->n_voltages; v++)
			value += d->rows[r * d->n_voltages + v] * x[d->voltage_offset + v];

		result[r] = value;

		if (grad)
		{
			for (unsigned i = 0; i < n; i++)
				grad[r * n + i] = 0.0;

			for (int v = 0; v < d->n_voltages; v++)
				grad[r * n + d->voltage_offset + v] = d->rows[r * d->n_voltages + v];
		}
	}
}

// Inequality constraint for nlopt which limits the total voltage applied to the electrodes at one step
double transport_waveform_constraint(
	unsigned n,
	const double *x,
	double *grad,
	void *constraint_data
)
{
	struct WaveformConstraintData *d = constraint_data;

	// Smooth approximation of |V| so that gradient-based optimizers can be used
	double epsilon = 1E-6;

	double constraint_value = 0.0;

	constraint_value -= fabs(d->max_voltage_total);

	if (grad)
		for (unsigned i = 0; i < n; i++)
			grad[i] = 0.0;

	for (int v = 0; v < d->n_voltages; v++)
	{
		double V = x[d->voltage_offset + v];
		double abs_V = sqrt(V*V + epsilon*epsilon);

		constraint_value += abs_V;

		if (grad)
			grad[d->voltage_offset + v] = V / abs_V;
	}

	return constraint_value;
}

// Compute the gradient and Hessian of the potential created by the voltages V (length n_electrodes)
void waveform_potential_derivatives
(
	struct Trap *trap,
	double *V,
	double position[3],
	double *grad_phi,
	double H[3][3]
)
{
	evaluate_trap_potential(trap, V, position, NULL, grad_phi, H);
}

// Build the constraints, linear in the DC voltages, which place a potential well at a position: no net force on the
// ion, the trap axis (x) as a principal axis, and the target axial curvature. The rows are scaled by ION_HEIGHT so that
// every constraint is measured in volts. rows has length N_WELL_CONSTRAINTS-by-n_electrodes_dc
void build_well_constraints
(
	struct Trap *trap,
	const double m,
	const double q,
	double position[3],
	const double omega_axial,
	double *rows,
	double *targets
)
{
	int n_dc = trap->n_electrodes_dc;
	int n_rf = trap->n_electrodes_rf;

	double L_c = ION_HEIGHT;

	// Every electrode's derivatives come from the same solid harmonics
	double (*grad_phi_all)[3] = malloc(trap->n_electrodes * sizeof(double[3]));
	double (*H_all)[3][3] = malloc(trap->n_electrodes * sizeof(double[3][3]));
	evaluate_electrode_derivatives_all(trap, position, NULL, grad_phi_all, H_all);

	for (int d = 0; d < n_dc; d++)
	{
		double *grad_phi = grad_phi_all[n_rf + d];
		double (*H)[3] = H_all[n_rf + d];

		rows[CONSTRAINT_EX_FIELD_X * n_dc + d] = grad_phi[0] * L_c;
		rows[CONSTRAINT_EX_FIELD_Y * n_dc + d] = grad_phi[1] * L_c;
		rows[CONSTRAINT_EX_FIELD_Z * n_dc + d] = grad_phi[2] * L_c;
		rows[CONSTRAINT_HESSIAN_XY * n_dc + d] = H[0][1] * L_c*L_c;
		rows[CONSTRAINT_HESSIAN_XZ * n_dc + d] = H[0][2] * L_c*L_c;
		rows[CONSTRAINT_AXIAL_CURVATURE * n_dc + d] = H[0][0] * L_c*L_c;
	}

	free(H_all);
	free(grad_phi_all);

	// The DC potential must cancel the force and the off-axis curvature of the pseudopotential (converted from J/µm and
	// J/µm^2 to V/µm and V/µm^2), and make up the rest of the axial curvature m omega^2
	double grad_U_pond[3];
	double H_U_pond[3][3];
	evaluate_pseudopotential(trap, m, q, position, NULL, grad_U_pond, H_U_pond);

	double axial_curvature = m * omega_axial*omega_axial * LENGTH_SCALE*LENGTH_SCALE;

	targets[CONSTRAINT_EX_FIELD_X] = -grad_U_pond[0] / q * L_c;
	targets[CONSTRAINT_EX_FIELD_Y] = -grad_U_pond[1] / q * L_c;
	targets[CONSTRAINT_EX_FIELD_Z] = -grad_U_pond[2] / q * L_c;
	targets[CONSTRAINT_HESSIAN_XY] = -H_U_pond[0][1] / q * L_c*L_c;
	targets[CONSTRAINT_HESSIAN_XZ] = -H_U_pond[0][2] / q * L_c*L_c;
	targets[CONSTRAINT_AXIAL_CURVATURE] = (axial_curvature - H_U_pond[0][0]) / q * L_c*L_c;
}

// Add a value to element (i, j), with i >= j, of a symmetric matrix in GSL's banded format
static void band_add
(
	gsl_matrix *AB,
	int i,
	int j,
	double value
)
{
	gsl_matrix_set(AB, j, i - j, gsl_matrix_get(AB, j, i - j) + value);
}

// Solve for the DC voltages which place potential wells at the given positions at every step, by minimizing the
// (heavily weighted) violations of the well constraints together with the voltages and the roughness of the waveform
// in a single regularized least squares problem. well_positions has length n_steps-by-n_wells-by-3, and voltages
// (the output) has length n_steps-by-n_electrodes, with RF electrodes held at 0 V DC
int solve_waveform
(
	struct Trap *trap,
	const double m,
	const double q,
	int n_steps,
	int n_wells,
	double (*well_positions)[3],
	const double omega_axial,
	const double roughness_weight,
	double *voltages
)
{
	int n_dc = trap->n_electrodes_dc;
	int n_rf = trap->n_electrodes_rf;
	int n_e = trap->n_electrodes;

	if (n_dc <= 0 || n_steps <= 0)
	{
		printf("ERROR: Waveforms need at least one DC electrode and one step\n");
		return 1;
	}

	int n_rows = N_WELL_CONSTRAINTS * n_wells;
	int N = n_steps * n_dc;

	// The roughness penalty couples each step to the two steps on either side
	int p = (n_steps >= 3) ? 2 * n_dc : n_dc - 1;

	gsl_matrix *AB = gsl_matrix_calloc(N, p + 1);
	gsl_vector *rhs = gsl_vector_calloc(N);
	gsl_vector *x = gsl_vector_alloc(N);

	printf("Building constraints for %d waveform steps with %d wells...\n", n_steps, n_wells);

	// For every well position, add the derivative and second derivative conditions; the steps fill separate blocks of
	// the matrix, so they are built in parallel
	#pragma omp parallel for schedule(dynamic)
	for (int s = 0; s < n_steps; s++)
	{
		double *rows = malloc(MAX(n_rows, 1) * n_dc * sizeof(double));
		double *targets = malloc(MAX(n_rows, 1) * sizeof(double));

		for (int w = 0; w < n_wells; w++)
			build_well_constraints(
				trap,
				m, q,
				well_positions[s * n_wells + w],
				omega_axial,
				&rows[w * N_WELL_CONSTRAINTS * n_dc],
				&targets[w * N_WELL_CONSTRAINTS]
			);

		// Diagonal block W_c A^T A + W_v I and right hand side W_c A^T b
		for (int a = 0; a < n_dc; a++)
		{
			for (int b = 0; b <= a; b++)
			{
				double AtA = 0.0;
				for (int r = 0; r < n_rows; r++)
					AtA += rows[r * n_dc + a] * rows[r * n_dc + b];

				band_add(AB, s * n_dc + a, s * n_dc + b, WAVEFORM_CONSTRAINT_WEIGHT * AtA + (a == b ? WAVEFORM_VOLTAGE_WEIGHT : 0.0));
			}

			double Atb = 0.0;
			for (int r = 0; r < n_rows; r++)
				Atb += rows[r * n_dc + a] * targets[r];

			gsl_vector_set(rhs, s * n_dc + a, WAVEFORM_CONSTRAINT_WEIGHT * Atb);
		}

		free(targets);
		free(rows);
	}

	// Roughness penalty W_r D^T D, where D takes the second difference along the steps for each electrode
	double c[3] = { 1.0, -2.0, 1.0 };
	for (int t = 1; t < n_steps-1; t++)
		for (int a = 0; a < n_dc; a++)
			for (int i = 0; i < 3; i++)
				for (int j = 0; j <= i; j++)
					band_add(AB, (t - 1 + i) * n_dc + a, (t - 1 + j) * n_dc + a, roughness_weight * c[i] * c[j]);

	printf("Solving for waveform voltages...\n");

	int status = gsl_linalg_cholesky_band_decomp(AB);
	if (status == 0)
		status = gsl_linalg_cholesky_band_solve(AB, rhs, x);

	if (status != 0)
		printf("ERROR: Failed to solve for waveform voltages (GSL error %d)\n", status);
	else
		for (int s = 0; s < n_steps; s++)
			for (int e = 0; e < n_e; e++)
				voltages[s * n_e + e] = (e < n_rf) ? 0.0 : gsl_vector_get(x, s * n_dc + (e - n_rf));

	gsl_vector_free(x);
	gsl_vector_free(rhs);
	gsl_matrix_free(AB);

	return status;
}

// Refine a waveform with nlopt, enforcing its well constraints (within WAVEFORM_CONSTRAINT_TOLERANCE) while bounding
// the voltage on each electrode and, if max_voltage_total > 0, the total voltage at each step; starts from the current
// voltages of the waveform, which are replaced if the optimization succeeds. Returns 0 on success
int optimize_primitive
(
	struct Trap *trap,
	struct IonParameters *ion,
	struct TransportWaveform *waveform,
	const double omega_axial,
	const double max_voltage_individual,
	const double max_voltage_total,
	int max_iterations
)
{
	double m = ion_mass(ion);
	double q = ion_charge(ion);

	int n_steps = waveform->n_waveform_steps;
	int n_wells = waveform->n_wells;
	int n_dc = trap->n_electrodes_dc;
	int n_rf = trap->n_electrodes_rf;
	int n_e = trap->n_electrodes;
	int n_rows = N_WELL_CONSTRAINTS * n_wells;

	double min_cost = 99999;
	const int dim = n_steps * n_dc;

	printf("- Constructing optimization problem...\n");

	// nlopt setup
	nlopt_opt opt = nlopt_create(NLOPT_LD_SLSQP, dim);

	printf("Setting constraints...\n");
	struct WaveformConstraintData *constraint_data = calloc(n_steps, sizeof(struct WaveformConstraintData));

	double *tolerances = malloc(MAX(n_rows, 1) * sizeof(double));
	for (int r = 0; r < n_rows; r++)
		tolerances[r] = WAVEFORM_CONSTRAINT_TOLERANCE;

	for (int t = 0; t < n_steps; t++)
	{
		struct WaveformConstraintData *cx = &constraint_data[t];

		cx->n_voltages = n_dc;
		cx->voltage_offset = t * n_dc;
		cx->n_rows = n_rows;
		cx->rows = malloc(MAX(n_rows, 1) * n_dc * sizeof(double));
		cx->targets = malloc(MAX(n_rows, 1) * sizeof(double));
		cx->max_voltage_individual = max_voltage_individual;
		cx->max_voltage_total = max_voltage_total;

		for (int w = 0; w < n_wells; w++)
			build_well_constraints(
				trap,
				m, q,
				waveform->well_positions[t * n_wells + w],
				omega_axial,
				&cx->rows[w * N_WELL_CONSTRAINTS * n_dc],
				&cx->targets[w * N_WELL_CONSTRAINTS]
			);

		if (n_rows > 0)
			nlopt_add_equality_mconstraint(
				opt,
				n_rows,
				waveform_well_constraints,
				cx,
				tolerances
			);

		if (max_voltage_total > 0.0)
			nlopt_add_inequality_constraint(
				opt,
				transport_waveform_constraint,
				cx,
				1E-8
			);
	}

	printf("Setting bounds...\n");
	double *lower_bounds = malloc(dim * sizeof(double));
	double *upper_bounds = malloc(dim * sizeof(double));
	for (int i = 0; i < dim; i++)
	{
		lower_bounds[i] = (max_voltage_individual > 0.0) ? -max_voltage_individual : -HUGE_VAL;
		upper_bounds[i] = (max_voltage_individual > 0.0) ? max_voltage_individual : HUGE_VAL;
	}
	nlopt_set_lower_bounds(opt, lower_bounds);
	nlopt_set_upper_bounds(opt, upper_bounds);

	printf("Setting stopping criteria...\n");
	nlopt_set_ftol_rel(opt, 1E-12);
	nlopt_set_xtol_rel(opt, 1E-10);
	nlopt_set_maxeval(opt, max_iterations);

	printf("Constructing initial guess...\n");
	// Start from the current waveform, clipped to the bounds
	double *x = malloc(dim * sizeof(double));
	for (int t = 0; t < n_steps; t++)
		for (int d = 0; d < n_dc; d++)
			x[t * n_dc + d] = MIN(MAX(waveform->voltages[t * n_e + n_rf + d], lower_bounds[t * n_dc + d]), upper_bounds[t * n_dc + d]);

	printf("Setting objectives...\n");
	struct WaveformCostData cost_data = { n_steps, n_dc, WAVEFORM_VOLTAGE_WEIGHT, WAVEFORM_ROUGHNESS_WEIGHT };
	nlopt_set_min_objective(opt, waveform_cost_function, &cost_data);

	printf("Optimizing...\n");
	nlopt_result result = nlopt_optimize(opt, x, &min_cost);
	if (result < 0)
	{
		printf("nlopt failed with code %d!\n", result);
	}
	else
	{
		printf("Found minimum cost %0.10g after %d evaluations\n", min_cost, nlopt_get_numevals(opt));

		for (int t = 0; t < n_steps; t++)
			for (int d = 0; d < n_dc; d++)
				waveform->voltages[t * n_e + n_rf + d] = x[t * n_dc + d];
	}

	for (int t = 0; t < n_steps; t++)
	{
		free(constraint_data[t].rows);
		free(constraint_data[t].targets);
	}

	free(x);
	free(upper_bounds);
	free(lower_bounds);
	free(tolerances);
	free(constraint_data);
	nlopt_destroy(opt);

	return result < 0;
}

// Fraction of a transport completed after a fraction u of its duration, following a minimum-jerk profile so that the
// velocity and acceleration of the well vanish at both ends
static double transport_profile
(
	double u
)
{
	return u*u*u * (10 - 15*u + 6*u*u);
}

// Generate a waveform which transports a single ion between two positions over the given duration
struct TransportWaveform* generate_transport_waveform
(
	struct Trap *trap,
	struct IonParameters *ion,
	double start[3],
	double end[3],
	const double duration,
	const int n_steps,
	const double omega_axial
)
{
	return generate_transport_primitive(trap, ion, start, end, duration, n_steps, omega_axial, WAVEFORM_ROUGHNESS_WEIGHT);
}

// Generate the waveform which moves a single well from start to end (µm) over a duration (s) along a minimum-jerk
// profile, with the given weight on the roughness of the voltages between steps
struct TransportWaveform* generate_transport_primitive
(
	struct Trap *trap,
	struct IonParameters *ion,
	double start[3],
	double end[3],
	const double duration,
	const int n_steps,
	const double omega_axial,
	const double roughness_weight
)
{
	struct TransportWaveform *waveform = allocate_transport_waveform(n_steps, trap->n_electrodes, 1);

	for (int s = 0; s < n_steps; s++)
	{
		double u = (n_steps > 1) ? (double) s / (n_steps - 1) : 0.0;
		double fraction = transport_profile(u);

		waveform->times[s] = u * duration;

		for (int k = 0; k < 3; k++)
			waveform->well_positions[s][k] = start[k] + fraction * (end[k] - start[k]);
	}

	if (
		solve_waveform(
			trap,
			ion_mass(ion), ion_charge(ion),
			n_steps,
			1,
			waveform->well_positions,
			omega_axial,
			roughness_weight,
			waveform->voltages
		) != 0
	)
	{
		free_transport_waveform(waveform);
		return NULL;
	}

	return waveform;
}

// Convert a transport program into a waveform with a well for every ion, where each transport moves its target ions to
// its destination along a minimum-jerk profile at the speed limit (starting late enough to arrive at its arrival time,
// if one is given) while the other ions are held in place
struct TransportWaveform* transport_program_to_waveform
(
	struct Trap *trap,
	struct IonParameters *ion,
	struct TransportProgram *transport_program,
	int *n_waveform_steps_list,
	const double omega_axial
)
{
	int n_ions = transport_program->n_ions;

	// Prepare waveform object
	int n_waveform_steps = 0;
	for (int t = 0; t < transport_program->n_transports; t++)
		n_waveform_steps += n_waveform_steps_list[t];

	struct TransportWaveform *waveform = allocate_transport_waveform(n_waveform_steps, trap->n_electrodes, n_ions);

	// Track where every ion is
	double (*ion_positions)[3] = malloc(n_ions * sizeof(double[3]));
	for (int i = 0; i < n_ions; i++)
		for (int k = 0; k < 3; k++)
			ion_positions[i][k] = (transport_program->initial_positions != NULL) ? transport_program->initial_positions[i][k] : 0.0;

	double time = 0.0;
	int step = 0;

	// For every destination, move the target ions from their current positions
	for (int t = 0; t < transport_program->n_transports; t++)
	{
		int n_targets = transport_program->transport_n_targets_list[t];
		int *targets = transport_program->transport_targets_list[t];
		double *destination = transport_program->destinations[t];

		// The farthest target sets the duration at the speed limit
		double distance = 0.0;
		for (int i = 0; i < n_targets; i++)
		{
			double *position = ion_positions[targets[i] - 1];

			distance = MAX(distance, sqrt(pow(destination[0] - position[0], 2) + pow(destination[1] - position[1], 2) + pow(destination[2] - position[2], 2)));
		}

		double duration = distance / TRANSPORT_SPEED_LIMIT;

		double start_time = time;
		if (transport_program->arrival_times != NULL && transport_program->arrival_times[t] - duration > time)
			start_time = transport_program->arrival_times[t] - duration;

		int n_steps = n_waveform_steps_list[t];
		for (int s = 0; s < n_steps; s++, step++)
		{
			double u = (n_steps > 1) ? (double) s / (n_steps - 1) : 1.0;
			double fraction = transport_profile(u);

			waveform->times[step] = start_time + u * duration;

			for (int i = 0; i < n_ions; i++)
				for (int k = 0; k < 3; k++)
					waveform->well_positions[step * n_ions + i][k] = ion_positions[i][k];

			for (int i = 0; i < n_targets; i++)
			{
				int ion_idx = targets[i] - 1;

				for (int k = 0; k < 3; k++)
					waveform->well_positions[step * n_ions + ion_idx][k] = ion_positions[ion_idx][k] + fraction * (destination[k] - ion_positions[ion_idx][k]);
			}
		}

		for (int i = 0; i < n_targets; i++)
			for (int k = 0; k < 3; k++)
				ion_positions[targets[i] - 1][k] = destination[k];

		time = start_time + duration;
	}

	free(ion_positions);

	if (
		solve_waveform(
			trap,
			ion_mass(ion), ion_charge(ion),
			n_waveform_steps,
			n_ions,
			waveform->well_positions,
			omega_axial,
			WAVEFORM_ROUGHNESS_WEIGHT,
			waveform->voltages
		) != 0
	)
	{
		free_transport_waveform(waveform);
		return NULL;
	}

	return waveform;
}

// Compute the largest residual force on an ion at any well of a waveform (N), and the largest relative error of the
// axial secular frequency from its target
void evaluate_waveform_errors
(
	struct Trap *trap,
	struct IonParameters *ion,
	struct TransportWaveform *waveform,
	const double omega_axial,
	double *max_force,
	double *max_frequency_error
)
{
	double m = ion_mass(ion);
	double q = ion_charge(ion);

	*max_force = 0.0;
	*max_frequency_error = 0.0;

	for (int s = 0; s < waveform->n_waveform_steps; s++)
		for (int w = 0; w < waveform->n_wells; w++)
		{
			double grad_U[3];
			double H_U[3][3];
			evaluate_potential_energy(
				trap,
				m, q,
				&waveform->voltages[s * waveform->n_electrodes],
				waveform->well_positions[s * waveform->n_wells + w],
				NULL,
				grad_U,
				H_U
			);

			// Convert from J/µm to N
			double force = sqrt(grad_U[0]*grad_U[0] + grad_U[1]*grad_U[1] + grad_U[2]*grad_U[2]) / LENGTH_SCALE;
			*max_force = MAX(*max_force, force);

			double k_axial = H_U[0][0] / (LENGTH_SCALE * LENGTH_SCALE);
			double omega = (k_axial > 0) ? sqrt(k_axial / m) : 0.0;
			*max_frequency_error = MAX(*max_frequency_error, fabs(omega - omega_axial) / omega_axial);
		}
}

// Linearly interpolate the voltages of a waveform at a time, holding the first and last steps outside of the waveform
void waveform_voltages_at_time
(
	struct TransportWaveform *waveform,
	const double t,
	double *voltages
)
{
	int n_steps = waveform->n_waveform_steps;
	int n_e = waveform->n_electrodes;

	if (n_steps == 0)
	{
		for (int e = 0; e < n_e; e++)
			voltages[e] = 0.0;
		return;
	}

	int lo = 0;
	int hi = n_steps - 1;

	if (t <= waveform->times[0])
		hi = 0;
	else if (t >= waveform->times[n_steps-1])
		lo = n_steps - 1;
	else
	{
		// Binary search for the interval containing t
		while (hi - lo > 1)
		{
			int mid = (lo + hi) / 2;

			if (waveform->times[mid] <= t)
				lo = mid;
			else
				hi = mid;
		}
	}

	double span = waveform->times[hi] - waveform->times[lo];
	double fraction = (hi > lo && span > 0.0) ? (t - waveform->times[lo]) / span : 1.0;

	for (int e = 0; e < n_e; e++)
		voltages[e] = (1 - fraction) * waveform->voltages[lo * n_e + e] + fraction * waveform->voltages[hi * n_e + e];
}

// Interpolate a coarse waveform onto the steps of a fine waveform (allocated by the caller), where fine step s
// corresponds to coarse step s * dx
void interpolate_waveform
(
	struct Trap *trap,
	struct TransportWaveform *waveform_coarse,
	struct TransportWaveform *waveform_fine,
	double dx
)
{
	int n_coarse = waveform_coarse->n_waveform_steps;
	int n_e = trap->n_electrodes;
	int n_wells = MIN(waveform_coarse->n_wells, waveform_fine->n_wells);

	double (*column)[n_coarse] = malloc(n_coarse * sizeof(double));

	for (int s = 0; s < waveform_fine->n_waveform_steps; s++)
	{
		for (int e = 0; e < n_e; e++)
		{
			for (int c = 0; c < n_coarse; c++)
				(*column)[c] = waveform_coarse->voltages[c * n_e + e];

			interpolate_1d(
				n_coarse,
				column,
				s,
				&waveform_fine->voltages[s * n_e + e],
				dx
			);
		}

		interpolate_1d(
			n_coarse,
			(double (*)[n_coarse]) waveform_coarse->times,
			s,
			&waveform_fine->times[s],
			dx
		);

		for (int w = 0; w < n_wells; w++)
			for (int k = 0; k < 3; k++)
			{
				for (int c = 0; c < n_coarse; c++)
					(*column)[c] = waveform_coarse->well_positions[c * waveform_coarse->n_wells + w][k];

				interpolate_1d(
					n_coarse,
					column,
					s,
					&waveform_fine->well_positions[s * waveform_fine->n_wells + w][k],
					dx
				);
			}
	}

	free(column);
}

// Concatenate waveforms in time, scaling the voltages of each (waveform_scales may be NULL for unit scales)
struct TransportWaveform* compose_waveforms
(
	int n_waveforms,
	struct TransportWaveform *waveforms,
	double *waveform_scales
)
{
	int n_waveform_steps = 0;
	for (int w = 0; w < n_waveforms; w++)
		n_waveform_steps += waveforms[w].n_waveform_steps;

	int n_e = (n_waveforms > 0) ? waveforms[0].n_electrodes : 0;
	int n_wells = (n_waveforms > 0) ? waveforms[0].n_wells : 0;

	struct TransportWaveform *composite_waveform = allocate_transport_waveform(n_waveform_steps, n_e, n_wells);

	int step = 0;
	double time_offset = 0.0;
	for (int w = 0; w < n_waveforms; w++)
	{
		double scale = (waveform_scales != NULL) ? waveform_scales[w] : 1.0;

		// Each waveform starts where the previous one ended
		if (waveforms[w].n_waveform_steps > 0)
			time_offset = (step > 0 ? composite_waveform->times[step-1] : 0.0) - waveforms[w].times[0];

		for (int substep = 0; substep < waveforms[w].n_waveform_steps; substep++, step++)
		{
			composite_waveform->times[step] = waveforms[w].times[substep] + time_offset;

			for (int e = 0; e < n_e; e++)
				composite_waveform->voltages[step * n_e + e] = scale * waveforms[w].voltages[substep * n_e + e];

			for (int i = 0; i < MIN(n_wells, waveforms[w].n_wells); i++)
				for (int k = 0; k < 3; k++)
					composite_waveform->well_positions[step * n_wells + i][k] = waveforms[w].well_positions[substep * waveforms[w].n_wells + i][k];
		}
	}

	return composite_waveform;
}

// Estimate the secular frequencies of an ideal linear Paul trap with axial (z) DC confinement and radial (x, y) RF
// confinement, where Z_0 (µm) is the characteristic distance between the ion and the electrodes; for a real trap, see
// compute_normal_modes
void compute_secular_frequencies
(
	double q,
	double m,
	double V_DC,
	double V_RF,
	double Omega,
	double Z_0,
	double *omega
)
{
	double Z_0_m = Z_0 * LENGTH_SCALE;

	double a_z = 4 * q * V_DC / (m * pow(Z_0_m, 2) * pow(Omega, 2));
	double a_x = -0.5 * a_z;
	double a_y = -0.5 * a_z;

	double q_x = 2 * q * V_RF / (m * pow(Z_0_m, 2) * pow(Omega, 2));
	double q_y = -q_x;
	double q_z = 0.0;

	omega[0] = Omega/2 * sqrt(a_x + 0.5 * pow(q_x, 2));
	omega[1] = Omega/2 * sqrt(a_y + 0.5 * pow(q_y, 2));
	omega[2] = Omega/2 * sqrt(a_z + 0.5 * pow(q_z, 2));
}

// Compute the pseudopotential U_pond = V_rf^2 |grad(U_rf)|^2 / (4 m omega_rf^2) (J) from samples of the RF potential
// energy per volt U_rf (J/V) taken along a periodic line with spacing dx (m)
void compute_U_pond
(
	double m,
	double omega_rf,
	double V_rf,
	const int len,
	double *U_rf,
	double *U_pond,
	double dx
)
{
	double (*grad_U_rf)[len] = malloc(len * sizeof(double));

	grad_1d(
		len,
		(double (*)[len]) U_rf,
		grad_U_rf,
		dx
	);

	for (int k = 0; k < len; k++)
		U_pond[k] = pow(V_rf * (*grad_U_rf)[k], 2) / (4 * m * pow(omega_rf, 2));

	free(grad_U_rf);
}
