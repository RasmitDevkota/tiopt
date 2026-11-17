#include "test_waveforms.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"
#include "electrodynamics.h"
#include "waveforms.h"
#include "transport.h"
#include "traps.h"
#include "ions.h"
#include "test_utils.h"

int test_all_waveforms
(
	double atol
)
{
	printf("- Testing waveforms module...\n");

	int result = 0;

	result += test_solve_waveform_static(atol);
	result += test_generate_transport_waveform(atol);
	result += test_waveform_utilities(atol);
	result += test_optimize_primitive(atol);
	result += test_transport_program_to_waveform(atol);
	result += test_schedule_trace_to_waveform(atol);
	result += test_compute_secular_frequencies(atol);
	result += test_compute_U_pond(atol);

	return result;
}

// The linear surface trap preset with its electrodynamics solved
static void make_test_trap
(
	struct Trap *trap,
	struct IonParameters *ion
)
{
	generate_trap_from_preset(LINEAR_SURFACE_TRAP, 7, 50.0, trap);
	solve_trap_electrodynamics(trap, ANALYTICAL, NULL);

	generate_ion_from_preset("40Ca+", ion);
}

int test_solve_waveform_static
(
	double atol
)
{
	struct Trap trap;
	struct IonParameters ion;
	make_test_trap(&trap, &ion);

	double m = ion_mass(&ion);
	double q = ion_charge(&ion);

	double well[1][3] = { { 0.0, 0.0, ION_HEIGHT } };
	find_rf_null(&trap, &well[0], 100);

	struct TransportWaveform *waveform = generate_transport_waveform(&trap, &ion, well[0], well[0], 0.0, 1, AXIAL_FREQUENCY);

	// A force of 1E-20 N displaces the ion by about 4 nm at 1 MHz, well within its ground state spread of 11 nm
	double max_force;
	double max_frequency_error;
	evaluate_waveform_errors(&trap, &ion, waveform, AXIAL_FREQUENCY, &max_force, &max_frequency_error);

	int result = check_error("static well has no residual force", max_force, 1E-20);
	result += check_error("static well has target axial frequency", max_frequency_error, 1E-4);

	// The ion settles at the target, with the trap axis as a principal axis
	double minimum[3] = { well[0][0] + 1.0, well[0][1] - 0.5, well[0][2] + 0.5 };
	find_potential_minimum(&trap, m, q, waveform->voltages, &minimum, 100);

	double position_error = sqrt(pow(minimum[0] - well[0][0], 2) + pow(minimum[1] - well[0][1], 2) + pow(minimum[2] - well[0][2], 2));
	result += check_error("static well minimum at target", position_error, 1E-5);

	double omega[3];
	double axes[3][3];
	compute_normal_modes(&trap, m, q, waveform->voltages, minimum, &omega, &axes);

	int axial_mode = 0;
	for (int k = 0; k < 3; k++)
		if (fabs(omega[k] - AXIAL_FREQUENCY) < fabs(omega[axial_mode] - AXIAL_FREQUENCY))
			axial_mode = k;

	result += check_error("static well axial mode along trap axis", 1 - fabs(axes[axial_mode][0]), 1E-6);

	// The radial modes are confined by the pseudopotential
	int n_unstable = 0;
	for (int k = 0; k < 3; k++)
		n_unstable += (omega[k] <= 0.0);

	result += check_error("static well is stable", n_unstable, 0);

	free_transport_waveform(waveform);
	free_trap(&trap);

	return result;
}

int test_generate_transport_waveform
(
	double atol
)
{
	struct Trap trap;
	struct IonParameters ion;
	make_test_trap(&trap, &ion);

	double start[3] = { -40.0, 0.0, ION_HEIGHT };
	double end[3] = { 40.0, 0.0, ION_HEIGHT };
	find_rf_null(&trap, &start, 100);
	find_rf_null(&trap, &end, 100);

	int n_steps = 201;
	double duration = 40E-6;
	struct TransportWaveform *waveform = generate_transport_waveform(&trap, &ion, start, end, duration, n_steps, AXIAL_FREQUENCY);

	double max_force;
	double max_frequency_error;
	evaluate_waveform_errors(&trap, &ion, waveform, AXIAL_FREQUENCY, &max_force, &max_frequency_error);

	int result = check_error("transport waveform has no residual force", max_force, 1E-19);
	result += check_error("transport waveform has target axial frequency", max_frequency_error, 1E-4);

	// The wells follow the minimum-jerk profile, which is symmetric about the middle of the transport
	double profile_error = 0.0;
	for (int k = 0; k < 3; k++)
	{
		profile_error = MAX(profile_error, fabs(waveform->well_positions[0][k] - start[k]));
		profile_error = MAX(profile_error, fabs(waveform->well_positions[n_steps-1][k] - end[k]));
		profile_error = MAX(profile_error, fabs(waveform->well_positions[n_steps/2][k] - 0.5 * (start[k] + end[k])));
	}

	profile_error = MAX(profile_error, fabs(waveform->times[n_steps-1] - duration) / duration);

	result += check_error("transport waveform follows minimum-jerk profile", profile_error, atol);

	free_transport_waveform(waveform);
	free_trap(&trap);

	return result;
}

int test_waveform_utilities
(
	double atol
)
{
	// A waveform with 5 steps on 2 electrodes whose voltages are linear in time
	struct TransportWaveform *waveform = allocate_transport_waveform(5, 2, 1);
	for (int s = 0; s < 5; s++)
	{
		waveform->times[s] = 1E-6 * s;
		waveform->voltages[2*s] = 1.0 + s;
		waveform->voltages[2*s + 1] = -2.0 * s;
		waveform->well_positions[s][0] = 10.0 * s;
	}

	// Interpolation in time, holding the ends
	double voltages[2];
	double error = 0.0;
	double times[4] = { 2.5E-6, 3E-6, -1E-6, 9E-6 };
	double expected[4][2] = { { 3.5, -5.0 }, { 4.0, -6.0 }, { 1.0, 0.0 }, { 5.0, -8.0 } };
	for (int i = 0; i < 4; i++)
	{
		waveform_voltages_at_time(waveform, times[i], voltages);
		error = MAX(error, fabs(voltages[0] - expected[i][0]) + fabs(voltages[1] - expected[i][1]));
	}

	int result = check_error("waveform_voltages_at_time interpolates and holds", error, atol);

	// Interpolating onto twice as many steps keeps linear waveforms linear
	struct Trap trap = { .n_electrodes = 2 };
	struct TransportWaveform *waveform_fine = allocate_transport_waveform(9, 2, 1);
	interpolate_waveform(&trap, waveform, waveform_fine, 0.5);

	error = 0.0;
	for (int s = 0; s < 9; s++)
	{
		error = MAX(error, fabs(waveform_fine->voltages[2*s] - (1.0 + 0.5 * s)));
		error = MAX(error, fabs(waveform_fine->voltages[2*s + 1] + s));
		error = MAX(error, fabs(waveform_fine->times[s] - 0.5E-6 * s) / 1E-6);
		error = MAX(error, fabs(waveform_fine->well_positions[s][0] - 5.0 * s));
	}

	result += check_error("interpolate_waveform onto finer steps", error, atol);

	// Composition concatenates the waveforms in time and scales their voltages
	struct TransportWaveform waveforms[2] = { *waveform, *waveform_fine };
	double scales[2] = { 1.0, -2.0 };
	struct TransportWaveform *composite = compose_waveforms(2, waveforms, scales);

	error = abs(composite->n_waveform_steps - 14);
	error += fabs(composite->times[5] - composite->times[4]) / 1E-6;
	error += fabs(composite->times[13] - 8E-6) / 1E-6;
	error += fabs(composite->voltages[2*7] - (-2.0 * waveform_fine->voltages[2*2]));

	result += check_error("compose_waveforms concatenates and scales", error, atol);

	free_transport_waveform(composite);
	free_transport_waveform(waveform_fine);
	free_transport_waveform(waveform);

	return result;
}

int test_optimize_primitive
(
	double atol
)
{
	struct Trap trap;
	struct IonParameters ion;
	make_test_trap(&trap, &ion);

	double m = ion_mass(&ion);
	double q = ion_charge(&ion);

	double start[3] = { -10.0, 0.0, ION_HEIGHT };
	double end[3] = { 10.0, 0.0, ION_HEIGHT };
	find_rf_null(&trap, &start, 100);
	find_rf_null(&trap, &end, 100);

	int n_steps = 5;
	struct TransportWaveform *waveform = generate_transport_waveform(&trap, &ion, start, end, 10E-6, n_steps, AXIAL_FREQUENCY);

	double peak_voltage = 0.0;
	for (int i = 0; i < n_steps * trap.n_electrodes; i++)
		peak_voltage = MAX(peak_voltage, fabs(waveform->voltages[i]));

	// Require the waveform to use lower voltages than the unconstrained solution
	double max_voltage = 0.9 * peak_voltage;
	int status = optimize_primitive(&trap, &ion, waveform, AXIAL_FREQUENCY, max_voltage, 0.0, 5000);

	double new_peak_voltage = 0.0;
	for (int i = 0; i < n_steps * trap.n_electrodes; i++)
		new_peak_voltage = MAX(new_peak_voltage, fabs(waveform->voltages[i]));

	int result = check_error("optimize_primitive succeeds", status, 0);
	result += check_error("optimize_primitive respects voltage limits", MAX(new_peak_voltage - max_voltage, 0.0), 1E-9);

	// The well constraints still hold (in the scaled units used by the optimizer, volts)
	double constraint_error = 0.0;
	double *rows = malloc(N_WELL_CONSTRAINTS * trap.n_electrodes_dc * sizeof(double));
	double targets[N_WELL_CONSTRAINTS];
	for (int s = 0; s < n_steps; s++)
	{
		build_well_constraints(&trap, m, q, waveform->well_positions[s], AXIAL_FREQUENCY, rows, targets);

		for (int r = 0; r < N_WELL_CONSTRAINTS; r++)
		{
			double value = -targets[r];
			for (int d = 0; d < trap.n_electrodes_dc; d++)
				value += rows[r * trap.n_electrodes_dc + d] * waveform->voltages[s * trap.n_electrodes + trap.n_electrodes_rf + d];

			constraint_error = MAX(constraint_error, fabs(value));
		}
	}

	result += check_error("optimize_primitive keeps well constraints", constraint_error, 10 * WAVEFORM_CONSTRAINT_TOLERANCE);

	free(rows);
	free_transport_waveform(waveform);
	free_trap(&trap);

	return result;
}

int test_transport_program_to_waveform
(
	double atol
)
{
	struct Trap trap;
	struct IonParameters ion;
	make_test_trap(&trap, &ion);

	// One ion moving to two destinations in turn
	double positions[3][3] = { { -30.0, 0.0, ION_HEIGHT }, { 30.0, 0.0, ION_HEIGHT }, { 0.0, 0.0, ION_HEIGHT } };
	for (int p = 0; p < 3; p++)
		find_rf_null(&trap, &positions[p], 100);

	int n_targets_list[2] = { 1, 1 };
	int targets[1] = { 1 };
	int *targets_list[2] = { targets, targets };

	struct TransportProgram transport_program;
	transport_program.n_ions = 1;
	transport_program.n_transports = 2;
	transport_program.arrival_times = NULL;
	transport_program.transport_n_targets_list = n_targets_list;
	transport_program.transport_targets_list = targets_list;
	transport_program.destinations = &positions[1];
	transport_program.initial_positions = &positions[0];

	int n_waveform_steps_list[2] = { 61, 31 };
	struct TransportWaveform *waveform = transport_program_to_waveform(&trap, &ion, &transport_program, n_waveform_steps_list, AXIAL_FREQUENCY);

	double error = abs(waveform->n_waveform_steps - 92) + abs(waveform->n_wells - 1);

	for (int k = 0; k < 3; k++)
	{
		error += fabs(waveform->well_positions[0][k] - positions[0][k]);
		error += fabs(waveform->well_positions[60][k] - positions[1][k]);
		error += fabs(waveform->well_positions[91][k] - positions[2][k]);
	}

	// Transports run back to back at the speed limit
	double distance = sqrt(pow(positions[1][0] - positions[0][0], 2) + pow(positions[1][2] - positions[0][2], 2)) + sqrt(pow(positions[2][0] - positions[1][0], 2) + pow(positions[2][2] - positions[1][2], 2));
	error += fabs(waveform->times[91] - distance / TRANSPORT_SPEED_LIMIT) / waveform->times[91];

	for (int s = 1; s < waveform->n_waveform_steps; s++)
		if (waveform->times[s] < waveform->times[s-1])
			error += 1.0;

	int result = check_error("transport_program_to_waveform follows destinations", error, atol);

	double max_force;
	double max_frequency_error;
	evaluate_waveform_errors(&trap, &ion, waveform, AXIAL_FREQUENCY, &max_force, &max_frequency_error);
	result += check_error("transport_program_to_waveform wells have target frequency", max_frequency_error, 1E-4);

	free_transport_waveform(waveform);
	free_trap(&trap);

	return result;
}

int test_schedule_trace_to_waveform
(
	double atol
)
{
	struct Trap trap;
	struct IonParameters ion;
	make_test_trap(&trap, &ion);

	// Three zones along the axis, far enough apart (two electrodes) for separate wells, with ion 1 in the first and ion 2
	// in the last
	struct ZoneNode zones[3] = { { 0 } };
	double xs[3] = { -100.0, 0.0, 100.0 };
	for (int z = 0; z < 3; z++)
	{
		zones[z].position[0] = xs[z];
		zones[z].position[2] = ION_HEIGHT;
		find_rf_null(&trap, &zones[z].position, 100);
	}

	struct TrapGraph trap_graph = { 3, 0, zones, NULL };

	double initial_positions[2][3];
	memcpy(initial_positions[0], zones[0].position, sizeof(double[3]));
	memcpy(initial_positions[1], zones[2].position, sizeof(double[3]));

	struct TransportProgram transport_program = { 0 };
	transport_program.n_ions = 2;
	transport_program.initial_positions = initial_positions;

	// Ion 1 moves to the middle zone and back (into storage), and ion 2 moves to the middle zone while ion 1 is leaving
	// it, so that the last two moves overlap
	struct IonMove moves[3] = {
		{ 1, 0, 1, 0, 0.0, 30E-6 },
		{ 1, 1, 0, -1, 40E-6, 70E-6 },
		{ 2, 2, 1, 1, 50E-6, 80E-6 },
	};
	struct ScheduleTrace trace = { 3, 3, moves, 0, NULL };

	struct TransportWaveform *waveform = schedule_trace_to_waveform(&trap, &ion, &trap_graph, &transport_program, &trace, 31, AXIAL_FREQUENCY);
	if (waveform == NULL)
	{
		free_trap(&trap);
		return check_error("schedule_trace_to_waveform follows the moves", HUGE_VAL, 0);
	}

	// 31 steps over each of the first two moves, and 10 more for the last one, which shares the times of the second from
	// 50 µs to 70 µs
	double error = abs(waveform->n_waveform_steps - 72) + abs(waveform->n_wells - 2);

	for (int s = 1; s < waveform->n_waveform_steps; s++)
		if (waveform->times[s] <= waveform->times[s-1])
			error += 1.0;

	// Every move reaches its destination when it arrives, and the other ion is where it should be
	for (int s = 0; s < waveform->n_waveform_steps; s++)
	{
		double t = waveform->times[s];
		double (*wells)[3] = &waveform->well_positions[2 * s];

		for (int m = 0; m < 3; m++)
			if (fabs(t - moves[m].arrive) < 1E-12)
				for (int k = 0; k < 3; k++)
					error += fabs(wells[moves[m].ion - 1][k] - zones[moves[m].to_zone].position[k]);

		// Ion 2 waits in its zone until 50 µs, and both ions move at once from 50 µs to 70 µs
		if (t <= 50E-6 + 1E-12)
			error += fabs(wells[1][0] - zones[2].position[0]);
		if (t > 50E-6 + 1E-9 && t < 70E-6 - 1E-9)
			error += (fabs(wells[0][0] - zones[0].position[0]) < 1E-9) + (fabs(wells[1][0] - zones[2].position[0]) < 1E-9);
	}

	int result = check_error("schedule_trace_to_waveform follows the moves", error, atol);

	double max_force;
	double max_frequency_error;
	evaluate_waveform_errors(&trap, &ion, waveform, AXIAL_FREQUENCY, &max_force, &max_frequency_error);
	// Two wells 100 µm apart in this seven-electrode trap only reach the target frequency to about 6E-3, with these moves
	// one after another (as transport_program_to_waveform finds too) or overlapping
	result += check_error("schedule_trace_to_waveform wells have target frequency", max_frequency_error, 1E-2);

	free_transport_waveform(waveform);
	free_trap(&trap);

	return result;
}

int test_compute_secular_frequencies
(
	double atol
)
{
	struct IonParameters ion;
	generate_ion_from_preset("40Ca+", &ion);

	double m = ion_mass(&ion);
	double q = ion_charge(&ion);

	double V_RF = 100.0;
	double Omega = 2 * PI * 20E6;
	double Z_0 = 500.0;

	// 1. Analytical result: without DC confinement, the radial frequencies are Omega q_Mathieu / (2 sqrt(2))
	double q_mathieu = 2 * q * V_RF / (m * pow(Z_0 * LENGTH_SCALE, 2) * pow(Omega, 2));
	double omega_radial = Omega * q_mathieu / (2 * SQRT2);

	// 2. tiopt result
	double omega[3];
	compute_secular_frequencies(q, m, 0.0, V_RF, Omega, Z_0, omega);

	double error = (fabs(omega[0] - omega_radial) + fabs(omega[1] - omega_radial) + fabs(omega[2])) / omega_radial;

	// The DC potential is harmonic, so it redistributes the curvature without changing the sum of omega^2
	double omega_dc[3];
	compute_secular_frequencies(q, m, 2.0, V_RF, Omega, Z_0, omega_dc);
	error += fabs(omega_dc[0]*omega_dc[0] + omega_dc[1]*omega_dc[1] + omega_dc[2]*omega_dc[2] - 2 * omega_radial*omega_radial) / (omega_radial*omega_radial);

	return check_error("compute_secular_frequencies of ideal Paul trap", error, 1E3 * atol);
}

int test_compute_U_pond
(
	double atol
)
{
	struct IonParameters ion;
	generate_ion_from_preset("40Ca+", &ion);

	double m = ion_mass(&ion);

	const int N = 400;
	double dx = 1E-7;
	double V_rf = 50.0;
	double omega_rf = 2 * PI * 30E6;
	double amplitude = 1E-21;

	double *U_rf = malloc(N * sizeof(double));
	double *U_pond = malloc(N * sizeof(double));
	for (int k = 0; k < N; k++)
		U_rf[k] = amplitude * sin(2 * PI * k / N);

	compute_U_pond(m, omega_rf, V_rf, N, U_rf, U_pond, dx);

	// The centered difference has a relative error of (2 PI / N)^2 / 6 in the gradient
	double error = 0.0;
	double scale = pow(V_rf * amplitude * 2 * PI / (N * dx), 2) / (4 * m * omega_rf*omega_rf);
	for (int k = 0; k < N; k++)
	{
		double U_pond_analytical = scale * pow(cos(2 * PI * k / N), 2);
		error = MAX(error, fabs(U_pond[k] - U_pond_analytical) / scale);
	}

	free(U_pond);
	free(U_rf);

	return check_error("compute_U_pond from sampled RF potential", error, 2.2 * pow(2 * PI / N, 2) / 6);
}
