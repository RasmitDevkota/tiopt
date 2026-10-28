#include "test_acceleration.h"

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
#include "bindings.h"
#include "acceleration.h"
#include "test_utils.h"

int test_all_acceleration
(
	double atol
)
{
	printf("- Testing acceleration module...\n");

	int result = 0;

	// Every test shares one solved trap, since solving it is the slowest part
	struct Trap trap;
	generate_trap_from_preset(LINEAR_SURFACE_TRAP, 3, 50.0, &trap);
	solve_trap_electrodynamics(&trap, ANALYTICAL, NULL);

	result += test_thread_consistency(&trap, atol);
	result += test_simulate_ensemble(&trap, atol);
	result += test_opencl(&trap, atol);

	free_trap(&trap);

	return result;
}

// Positions spread through the region above the trap where the ion is held
static void fill_positions
(
	int n_points,
	double (*positions)[3]
)
{
	for (int p = 0; p < n_points; p++)
	{
		positions[p][0] = -80.0 + 160.0 * (p % 37) / 36.0;
		positions[p][1] = -15.0 + 30.0 * ((7 * p) % 11) / 10.0;
		positions[p][2] = ION_HEIGHT - 10.0 + 20.0 * ((3 * p) % 13) / 12.0;
	}
}

// DC voltages which are different on every electrode
static void fill_voltages
(
	struct Trap *trap,
	double *voltages
)
{
	for (int e = 0; e < trap->n_electrodes; e++)
		voltages[e] = (e < trap->n_electrodes_rf) ? 0.0 : 0.5 * sin(1.0 + e);
}

// A small thermal-like ensemble about the RF null at the trap center
static void fill_ensemble
(
	int n_particles,
	double (*positions)[3],
	double (*velocities)[3]
)
{
	for (int p = 0; p < n_particles; p++)
		for (int k = 0; k < 3; k++)
		{
			positions[p][k] = ((k == 2) ? ION_HEIGHT : 0.0) + 0.3 * sin(3.1 * p + k);
			velocities[p][k] = 1E4 * cos(1.7 * p + k);
		}
}

// The maximum difference between two arrays relative to the largest magnitude in the first
static double relative_difference
(
	int n,
	double *a,
	double *b
)
{
	double difference = 0.0;
	double scale = 0.0;

	for (int i = 0; i < n; i++)
	{
		difference = MAX(difference, fabs(a[i] - b[i]));
		scale = MAX(scale, fabs(a[i]));
	}

	return (scale > 0.0) ? difference / scale : difference;
}

int test_thread_consistency
(
	struct Trap *trap,
	double atol
)
{
	struct IonParameters ion;
	generate_ion_from_preset("40Ca+", &ion);

	double m = ion_mass(&ion);
	double q = ion_charge(&ion);

	int n_points = 500;
	double (*positions)[3] = malloc(n_points * sizeof(double[3]));
	fill_positions(n_points, positions);

	double voltages[trap->n_electrodes];
	fill_voltages(trap, voltages);

	int n_threads = get_num_threads();

	// 1. Serial result
	double *U_serial = malloc(n_points * sizeof(double));
	set_num_threads(1);
	evaluate_potential_energy_batch(trap, m, q, voltages, n_points, positions, U_serial);

	// 2. tiopt result with every thread
	double *U_parallel = malloc(n_points * sizeof(double));
	set_num_threads(MAX(n_threads, 4));
	evaluate_potential_energy_batch(trap, m, q, voltages, n_points, positions, U_parallel);

	set_num_threads(n_threads);

	// Every point is evaluated independently, so the results are identical
	int result = check_error("evaluate_potential_energy_batch is independent of thread count", relative_difference(n_points, U_serial, U_parallel), 0.0);

	free(U_parallel);
	free(U_serial);
	free(positions);

	return result;
}

int test_simulate_ensemble
(
	struct Trap *trap,
	double atol
)
{
	struct IonParameters ion;
	generate_ion_from_preset("40Ca+", &ion);

	double m = ion_mass(&ion);
	double q = ion_charge(&ion);

	int n_particles = 8;
	int n_it = 500;
	double dt = 5E-9;

	struct TransportWaveform *waveform = allocate_transport_waveform(1, trap->n_electrodes, 0);
	fill_voltages(trap, waveform->voltages);

	// 1. Each ion integrated on its own
	double (*positions)[3] = malloc(n_particles * sizeof(double[3]));
	double (*velocities)[3] = malloc(n_particles * sizeof(double[3]));
	fill_ensemble(n_particles, positions, velocities);

	for (int p = 0; p < n_particles; p++)
	{
		double acc[3];
		velocity_verlet(trap, m, q, waveform, 0, &positions[p], &velocities[p], &acc, n_it, dt, NULL, NULL);
	}

	// 2. tiopt result
	double (*ensemble_positions)[3] = malloc(n_particles * sizeof(double[3]));
	double (*ensemble_velocities)[3] = malloc(n_particles * sizeof(double[3]));
	fill_ensemble(n_particles, ensemble_positions, ensemble_velocities);

	enum ComputeDevice device = get_compute_device();
	set_compute_device(CPU_DEVICE);
	simulate_ensemble(trap, m, q, waveform, 0, n_particles, ensemble_positions, ensemble_velocities, n_it, dt);
	set_compute_device(device);

	double error = relative_difference(3 * n_particles, (double *) positions, (double *) ensemble_positions);
	error += relative_difference(3 * n_particles, (double *) velocities, (double *) ensemble_velocities);

	free(ensemble_velocities);
	free(ensemble_positions);
	free(velocities);
	free(positions);
	free_transport_waveform(waveform);

	return check_error("simulate_ensemble matches velocity_verlet", error, 0.0);
}

int test_opencl
(
	struct Trap *trap,
	double atol
)
{
	if (!opencl_available())
	{
		printf("  SKIPPED OpenCL tests (no OpenCL device with double precision; see README)\n");
		return 0;
	}

	printf("  Testing OpenCL device %s\n", opencl_device_name());

	struct IonParameters ion;
	generate_ion_from_preset("40Ca+", &ion);

	double m = ion_mass(&ion);
	double q = ion_charge(&ion);

	enum ComputeDevice device = get_compute_device();
	int result = 0;

	// Potential maps
	int n_points = 500;
	double (*positions)[3] = malloc(n_points * sizeof(double[3]));
	fill_positions(n_points, positions);

	double voltages[trap->n_electrodes];
	fill_voltages(trap, voltages);

	double *U[2];
	double *phi[2];
	double (*grad_phi[2])[3];
	for (int d = 0; d < 2; d++)
	{
		set_compute_device((d == 0) ? CPU_DEVICE : OPENCL_DEVICE);

		U[d] = malloc(n_points * sizeof(double));
		phi[d] = malloc(n_points * sizeof(double));
		grad_phi[d] = malloc(n_points * sizeof(double[3]));

		evaluate_potential_energy_batch(trap, m, q, voltages, n_points, positions, U[d]);
		evaluate_trap_potential_batch(trap, voltages, n_points, positions, phi[d], grad_phi[d]);
	}

	result += check_error("OpenCL potential energy map matches CPU", relative_difference(n_points, U[0], U[1]), 1E3 * atol);
	result += check_error("OpenCL potential map matches CPU", relative_difference(n_points, phi[0], phi[1]), 1E3 * atol);
	result += check_error("OpenCL field map matches CPU", relative_difference(3 * n_points, (double *) grad_phi[0], (double *) grad_phi[1]), 1E3 * atol);

	for (int d = 0; d < 2; d++)
	{
		free(U[d]);
		free(phi[d]);
		free(grad_phi[d]);
	}
	free(positions);

	// Ensembles, in the pseudopotential approximation and with the RF drive
	struct TransportWaveform *waveform = allocate_transport_waveform(1, trap->n_electrodes, 0);
	fill_voltages(trap, waveform->voltages);

	for (int full_rf = 0; full_rf < 2; full_rf++)
	{
		int n_particles = 16;
		int n_it = full_rf ? 2000 : 500;
		double dt = full_rf ? 2.5E-10 : 5E-9;

		double (*ensemble_positions[2])[3];
		double (*ensemble_velocities[2])[3];
		for (int d = 0; d < 2; d++)
		{
			set_compute_device((d == 0) ? CPU_DEVICE : OPENCL_DEVICE);

			ensemble_positions[d] = malloc(n_particles * sizeof(double[3]));
			ensemble_velocities[d] = malloc(n_particles * sizeof(double[3]));
			fill_ensemble(n_particles, ensemble_positions[d], ensemble_velocities[d]);

			simulate_ensemble(trap, m, q, waveform, full_rf, n_particles, ensemble_positions[d], ensemble_velocities[d], n_it, dt);
		}

		// Rounding differences grow slowly over the trajectories
		double error = relative_difference(3 * n_particles, (double *) ensemble_positions[0], (double *) ensemble_positions[1]);
		error += relative_difference(3 * n_particles, (double *) ensemble_velocities[0], (double *) ensemble_velocities[1]);

		result += check_error(full_rf ? "OpenCL ensemble with RF drive matches CPU" : "OpenCL ensemble in pseudopotential matches CPU", error, 1E3 * atol);

		for (int d = 0; d < 2; d++)
		{
			free(ensemble_positions[d]);
			free(ensemble_velocities[d]);
		}
	}

	free_transport_waveform(waveform);

	// Relaxation in a small box, for a fixed number of iterations so that both devices do the same sweeps
	const int NX = 41;
	const int NY = 41;
	const int NZ = 31;

	double (*V[2])[NX][NY][NZ];
	for (int d = 0; d < 2; d++)
	{
		set_compute_device((d == 0) ? CPU_DEVICE : OPENCL_DEVICE);

		V[d] = calloc(NX * NY * NZ, sizeof(double));

		double electrode_position[3] = { 0.0, 0.0, 0.0 };
		solver_relaxation(NX, NY, NZ, &(*trap->electrodes)[0], electrode_position, V[d], 1, 1, 1, 50, 1);
	}

	result += check_error("OpenCL relaxation matches CPU", relative_difference(NX * NY * NZ, &(*V[0])[0][0][0], &(*V[1])[0][0][0]), 1E3 * atol);

	for (int d = 0; d < 2; d++)
		free(V[d]);

	set_compute_device(device);

	return result;
}
