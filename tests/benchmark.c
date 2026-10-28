#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#include "defs.h"
#include "data_structures.h"
#include "electrodynamics.h"
#include "verlet.h"
#include "waveforms.h"
#include "quantum.h"
#include "traps.h"
#include "ions.h"
#include "bindings.h"
#include "acceleration.h"

// Everything shared by the benchmarks, set up once
struct BenchmarkContext
{
	struct Trap trap;
	struct IonParameters ion;
	double m;
	double q;
	double start[3];
	double end[3];
	struct TransportWaveform *waveform;
	int n_points;
	double (*positions)[3];
	double *U;
};

static double wall_time
(
)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);

	return t.tv_sec + 1E-9 * t.tv_nsec;
}

// Silence the progress output of the library while a benchmark runs
static int quiet_stdout
(
)
{
	fflush(stdout);
	int saved = dup(fileno(stdout));

	if (freopen("/dev/null", "w", stdout) == NULL)
		return -1;

	return saved;
}

static void restore_stdout
(
	int saved
)
{
	if (saved < 0)
		return;

	fflush(stdout);
	dup2(saved, fileno(stdout));
	close(saved);
}

static void benchmark_analytical_solve
(
	struct BenchmarkContext *context
)
{
	struct Trap trap;
	generate_trap_from_preset(LINEAR_SURFACE_TRAP, 7, 50.0, &trap);
	solve_trap_electrodynamics(&trap, ANALYTICAL, NULL);
	free_trap(&trap);
}

static void benchmark_spectral_solve
(
	struct BenchmarkContext *context
)
{
	double (*V)[RELAXATION_NX][RELAXATION_NY][RELAXATION_NZ] = calloc(RELAXATION_NX * RELAXATION_NY * RELAXATION_NZ, sizeof(double));

	solver_spectral(
		RELAXATION_NX, RELAXATION_NY, RELAXATION_NZ,
		&(*context->trap.electrodes)[3],
		(*context->trap.electrode_positions)[3],
		V
	);

	free(V);
}

static void benchmark_relaxation_sweeps
(
	struct BenchmarkContext *context
)
{
	double (*V)[RELAXATION_NX][RELAXATION_NY][RELAXATION_NZ] = calloc(RELAXATION_NX * RELAXATION_NY * RELAXATION_NZ, sizeof(double));

	solver_relaxation(
		RELAXATION_NX, RELAXATION_NY, RELAXATION_NZ,
		&(*context->trap.electrodes)[3],
		(*context->trap.electrode_positions)[3],
		V,
		1, 1, 1,
		50,
		1
	);

	free(V);
}

static void benchmark_potential_map
(
	struct BenchmarkContext *context
)
{
	double voltages[context->trap.n_electrodes];
	memset(voltages, 0, sizeof(voltages));

	evaluate_potential_energy_batch(&context->trap, context->m, context->q, voltages, context->n_points, context->positions, context->U);
}

static void benchmark_waveform
(
	struct BenchmarkContext *context
)
{
	struct TransportWaveform *waveform = generate_transport_waveform(&context->trap, &context->ion, context->start, context->end, 50E-6, 501, AXIAL_FREQUENCY);
	free_transport_waveform(waveform);
}

static void benchmark_trajectory
(
	struct BenchmarkContext *context
)
{
	double pos[3] = { context->start[0], context->start[1], context->start[2] };
	double vel[3] = { 0.0, 0.0, 0.0 };
	double acc[3];

	velocity_verlet(&context->trap, context->m, context->q, context->waveform, 0, &pos, &vel, &acc, 10000, 5E-9, NULL, NULL);
}

static void benchmark_ensemble
(
	struct BenchmarkContext *context
)
{
	int n_particles = 1024;
	double (*positions)[3] = malloc(n_particles * sizeof(double[3]));
	double (*velocities)[3] = malloc(n_particles * sizeof(double[3]));

	for (int p = 0; p < n_particles; p++)
		for (int k = 0; k < 3; k++)
		{
			positions[p][k] = context->start[k] + 0.3 * sin(3.1 * p + k);
			velocities[p][k] = 1E4 * cos(1.7 * p + k);
		}

	simulate_ensemble(&context->trap, context->m, context->q, context->waveform, 0, n_particles, positions, velocities, 1000, 5E-9);

	free(velocities);
	free(positions);
}

// A Molmer-Sorensen gate on n_ions ions with a heated mode
static void benchmark_ms_gate
(
	struct BenchmarkContext *context,
	int n_ions,
	int n_max
)
{
	double nu = 2 * PI * 1E6;
	double delta = 2 * PI * 20E3;
	double eta = 0.05;

	struct PulseProgram *pulse_program = allocate_pulse_program(n_ions, 2);
	for (int k = 0; k < 2; k++)
	{
		pulse_program->pulse_durations[k] = 2 * PI / delta;
		pulse_program->pulse_n_targets_list[k] = n_ions;
		pulse_program->pulse_targets_list[k] = malloc(n_ions * sizeof(int));

		for (int i = 0; i < n_ions; i++)
			pulse_program->pulse_targets_list[k][i] = i + 1;
	}
	pulse_program->pulse_frequencies[0] = context->ion.omega_SD + nu + delta;
	pulse_program->pulse_frequencies[1] = context->ion.omega_SD - nu - delta;

	struct MotionalMode mode = { nu, n_max, 0.5, { 0.0, 0.0 }, 1E3 };

	double rabi_frequencies[2] = { delta / (2 * eta), delta / (2 * eta) };
	double lamb_dicke_parameters[2 * n_ions];
	for (int i = 0; i < 2 * n_ions; i++)
		lamb_dicke_parameters[i] = eta;

	double sample_times[2];
	double excited_populations[2 * n_ions];
	double mean_phonon_numbers[2];
	simulate_pulse_program(&context->ion, &mode, pulse_program, rabi_frequencies, lamb_dicke_parameters, 100.0, 2, sample_times, excited_populations, mean_phonon_numbers, NULL);

	free_pulse_program(pulse_program);
}

static void benchmark_ms_gate_2
(
	struct BenchmarkContext *context
)
{
	benchmark_ms_gate(context, 2, 12);
}

static void benchmark_ms_gate_3
(
	struct BenchmarkContext *context
)
{
	benchmark_ms_gate(context, 3, 25);
}

// Time a benchmark (s)
static double time_benchmark
(
	void (*benchmark)(struct BenchmarkContext *context),
	struct BenchmarkContext *context
)
{
	int saved = quiet_stdout();

	double t_0 = wall_time();
	benchmark(context);
	double t_1 = wall_time();

	restore_stdout(saved);

	return t_1 - t_0;
}

// Run every benchmark serially, on every thread, and on the OpenCL device (if there is one), printing a table of the
// times taken
int main
(
)
{
	struct BenchmarkContext context;

	int saved = quiet_stdout();

	generate_trap_from_preset(LINEAR_SURFACE_TRAP, 7, 50.0, &context.trap);
	solve_trap_electrodynamics(&context.trap, ANALYTICAL, NULL);

	generate_ion_from_preset("40Ca+", &context.ion);
	context.m = ion_mass(&context.ion);
	context.q = ion_charge(&context.ion);

	double start[3] = { -50.0, 0.0, ION_HEIGHT };
	double end[3] = { 50.0, 0.0, ION_HEIGHT };
	find_rf_null(&context.trap, &start, 100);
	find_rf_null(&context.trap, &end, 100);
	memcpy(context.start, start, sizeof(start));
	memcpy(context.end, end, sizeof(end));

	context.waveform = generate_transport_waveform(&context.trap, &context.ion, context.start, context.end, 50E-6, 501, AXIAL_FREQUENCY);

	// A 200-by-200 map of the potential energy in the xz-plane
	context.n_points = 40000;
	context.positions = malloc(context.n_points * sizeof(double[3]));
	context.U = malloc(context.n_points * sizeof(double));
	for (int p = 0; p < context.n_points; p++)
	{
		context.positions[p][0] = -100.0 + 200.0 * (p % 200) / 199.0;
		context.positions[p][1] = 0.0;
		context.positions[p][2] = ION_HEIGHT - 15.0 + 30.0 * (p / 200) / 199.0;
	}

	int has_opencl = opencl_available();

	restore_stdout(saved);

	const int n_benchmarks = 9;
	const char *names[] =
	{
		"Analytical solve (linear trap)",
		"Spectral solve (1 electrode)",
		"Relaxation (50 sweeps)",
		"Potential energy map (40000 points)",
		"Transport waveform (501 steps)",
		"Trajectory (10000 steps)",
		"Ensemble (1024 ions, 1000 steps)",
		"MS gate (2 ions, dim 52)",
		"MS gate (3 ions, dim 208)",
	};
	void (*benchmarks[])(struct BenchmarkContext *context) =
	{
		benchmark_analytical_solve,
		benchmark_spectral_solve,
		benchmark_relaxation_sweeps,
		benchmark_potential_map,
		benchmark_waveform,
		benchmark_trajectory,
		benchmark_ensemble,
		benchmark_ms_gate_2,
		benchmark_ms_gate_3,
	};
	// Only some benchmarks have OpenCL implementations
	int opencl_benchmarks[] = { 0, 0, 1, 1, 0, 0, 1, 0, 0 };

	int n_threads = get_num_threads();

	char threads_label[32];
	snprintf(threads_label, sizeof(threads_label), "%d threads", n_threads);

	printf("%-38s %12s %12s", "Benchmark (s)", "1 thread", threads_label);
	if (has_opencl)
		printf(" %12s", "OpenCL");
	printf("\n");

	for (int b = 0; b < n_benchmarks; b++)
	{
		set_compute_device(CPU_DEVICE);

		set_num_threads(1);
		double t_serial = time_benchmark(benchmarks[b], &context);

		set_num_threads(n_threads);
		double t_parallel = time_benchmark(benchmarks[b], &context);

		printf("%-38s %12.4f %12.4f", names[b], t_serial, t_parallel);

		if (has_opencl && opencl_benchmarks[b])
		{
			set_compute_device(OPENCL_DEVICE);
			printf(" %12.4f", time_benchmark(benchmarks[b], &context));
			set_compute_device(CPU_DEVICE);
		}

		printf("\n");
		fflush(stdout);
	}

	if (has_opencl)
		printf("OpenCL device: %s\n", opencl_device_name());
	else
		printf("No OpenCL device with double precision was found; see README for how to enable GPUs\n");

	free(context.U);
	free(context.positions);
	free_transport_waveform(context.waveform);
	free_trap(&context.trap);
	release_opencl();

	return 0;
}
