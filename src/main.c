#include "main.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <nlopt.h>

#include "defs.h"
#include "io.h"
#include "data_structures.h"
#include "electrodynamics.h"
#include "verlet.h"
#include "experiments.h"
#include "transport.h"
#include "traps.h"
#include "ions.h"
#include "waveforms.h"
#include "photonics.h"
#include "quantum.h"
#include "analysis.h"
#include "circuits.h"
#include "codesign.h"
#include "acceleration.h"

// Number of DC segments on each side of the trap optimized by the optimization loop
#ifndef OPTIMIZATION_N_SEGMENTS
#define OPTIMIZATION_N_SEGMENTS 7
#endif

// Cost function passed to optimization routine
double cost_function(
	unsigned n,
	const double *x,
	double *grad,
	void *trap_obj
)
{
	// The purpose of this function is to compute an aggregate cost of a trap design and transport configuration
	double cost = 0.0;

	struct Trap *trap = (struct Trap *)trap_obj;

	// Update trap based on optimization parameters: the width of the DC segments (µm) and the RF voltage (V)
	free_trap(trap);
	if (generate_trap_from_preset(LINEAR_SURFACE_TRAP, OPTIMIZATION_N_SEGMENTS, x[0], trap) != 0)
		return 1E6;

	trap->V_RF = x[1];

	// Solve for the potential energy contribution of each electrode individually
	solve_trap_electrodynamics(trap, ANALYTICAL, NULL);

	// Perform ion transport experiments
	struct IonParameters ion;
	generate_ion_from_preset("40Ca+", &ion);

	int experiment_list[2] = { IDLE, BACK_AND_FORTH };
	for (int e = 0; e < 2; e++)
		run_experiment(trap, &ion, experiment_list[e], &cost);

	printf("Cost for segment width %f µm and RF voltage %f V: %f\n", x[0], x[1], cost);

	return cost;
}

// Optimize the geometry and RF drive of a linear surface trap based on transport experiments
static void optimize_trap
(
	int max_evaluations
)
{
	double min_cost = 99999;
	const int dim = 2;

	printf("- Constructing optimization problem...\n");

	// nlopt setup
	nlopt_opt opt = nlopt_create(NLOPT_LN_COBYLA, dim);

	printf("Setting bounds...\n");
	double lower_bounds[2] = { 20.0, 10.0 };
	double upper_bounds[2] = { 120.0, 100.0 };
	nlopt_set_lower_bounds(opt, lower_bounds);
	nlopt_set_upper_bounds(opt, upper_bounds);

	printf("Setting stopping criteria...\n");
	nlopt_set_xtol_abs1(opt, 0.5);
	nlopt_set_maxeval(opt, max_evaluations);

	printf("Constructing initial guess...\n");
	struct Trap trap;
	generate_trap_from_preset(LINEAR_SURFACE_TRAP, OPTIMIZATION_N_SEGMENTS, 50.0, &trap);

	double x[2] = { 50.0, trap.V_RF };

	printf("Setting objectives...\n");
	nlopt_set_min_objective(opt, cost_function, &trap);

	printf("Optimizing...\n");
	if (nlopt_optimize(opt, x, &min_cost) < 0)
	{
		printf("nlopt failed!\n");
	}
	else
	{
		printf("Found minimum at f(");
		for (int k = 0; k < dim; k++)
		{
			printf("%g", x[k]);

			if (k < dim-1)
			{
				printf(",");
			}
		}
		printf(") = %0.10g\n", min_cost);
	}

	// Free malloc'd memory
	free_trap(&trap);
	nlopt_destroy(opt);
}

// Solve the electrodynamics of the trap described by a trap file
static void solve_trap_file
(
	char *trap_filename
)
{
	struct Trap trap;
	if (generate_trap_from_file(trap_filename, &trap) != 0)
		return;

	printf("Enter 0 to use the relaxation solver (slow) or 1 to use the analytical solver: ");

	int solver = 1;
	if (scanf("%d", &solver) != 1)
		solver = 1;

	solve_trap_electrodynamics(&trap, (solver == 0) ? RELAXATION : ANALYTICAL, (solver == 0) ? "V.txt" : NULL);

	// Report the potential of every electrode above its center
	for (int e = 0; e < trap.n_electrodes; e++)
	{
		double position[3] = { (*trap.electrode_positions)[e][0], (*trap.electrode_positions)[e][1], ION_HEIGHT };

		double phi;
		double grad_phi[3];
		evaluate_electrode_derivatives(&(*trap.electrodes)[e], position, &phi, grad_phi, NULL);

		printf("Electrode %d at ion height: phi = %f V/V, grad(phi) = (%e, %e, %e) V/V/µm\n", e, phi, grad_phi[0], grad_phi[1], grad_phi[2]);
	}

	free_trap(&trap);
}

// Optimize a transport waveform on the linear surface trap preset and simulate the ion following it
static void simulate_transport
(
)
{
	struct Trap trap;
	generate_trap_from_preset(LINEAR_SURFACE_TRAP, 7, 50.0, &trap);
	solve_trap_electrodynamics(&trap, ANALYTICAL, NULL);

	struct IonParameters ion;
	generate_ion_from_preset("40Ca+", &ion);

	double m = ion_mass(&ion);
	double q = ion_charge(&ion);

	// Move a well along the RF null
	double start[3] = { -50.0, 0.0, ION_HEIGHT };
	double end[3] = { 50.0, 0.0, ION_HEIGHT };
	find_rf_null(&trap, &start, 100);
	find_rf_null(&trap, &end, 100);

	double duration = 50E-6;

	// Steps must be much shorter than the axial period, since stepping the voltages near the axial period excites the ion
	int n_steps = 501;
	double max_voltage = 10.0;

	struct TransportWaveform *waveform = generate_transport_waveform(&trap, &ion, start, end, duration, n_steps, AXIAL_FREQUENCY);
	if (waveform == NULL)
	{
		free_trap(&trap);
		return;
	}

	double peak_voltage = 0.0;
	for (int i = 0; i < waveform->n_waveform_steps * waveform->n_electrodes; i++)
		peak_voltage = MAX(peak_voltage, fabs(waveform->voltages[i]));

	printf("Peak voltage of waveform: %f V\n", peak_voltage);

	// Only refine the waveform with nlopt when it violates the voltage limits, since this is much slower
	if (peak_voltage > max_voltage)
	{
		printf("Refining waveform with voltage limits...\n");
		optimize_primitive(&trap, &ion, waveform, AXIAL_FREQUENCY, max_voltage, 0.0, 500);
	}

	double max_force;
	double max_frequency_error;
	evaluate_waveform_errors(&trap, &ion, waveform, AXIAL_FREQUENCY, &max_force, &max_frequency_error);
	printf("Waveform has a maximum residual force of %e N and a maximum axial frequency error of %e\n", max_force, max_frequency_error);

	double pos[3] = { start[0], start[1], start[2] };
	double vel[3] = { 0.0, 0.0, 0.0 };
	double acc[3];

	double dt = 5E-9;
	int n_it = (int) ceil(1.2 * duration / dt);
	velocity_verlet(&trap, m, q, waveform, 0, &pos, &vel, &acc, n_it, dt, NULL, NULL);

	double axis[3] = { 1.0, 0.0, 0.0 };
	printf("Transport excited %f axial quanta\n", motional_quanta(m, AXIAL_FREQUENCY, axis, pos, vel, end));

	free_transport_waveform(waveform);
	free_trap(&trap);
}

// Design a grating coupler which addresses an ion at the trap center and evaluate the light it delivers
static void design_grating
(
)
{
	struct Trap trap;
	if (generate_trap_from_preset(LINEAR_SURFACE_TRAP_WITH_GRATINGS, 7, 50.0, &trap) != 0)
		return;

	struct IonParameters ion;
	generate_ion_from_preset("40Ca+", &ion);

	struct GratingCoupler *grating = &(*trap.grating_couplers)[0];

	solve_waveguide_mode(&grating->waveguide);

	double theta;
	double k_hat[3];
	compute_grating_emission(grating, &theta, &k_hat);

	double ion_position[3] = { 0.0, 0.0, ION_HEIGHT };
	double intensity;
	double k_vector[3];
	evaluate_grating_beam(grating, ion_position, &intensity, &k_vector);

	double wavelength = 2 * PI * C_LIGHT / ion.omega_SD;
	double rabi_frequency = rabi_frequency_from_intensity(ion.gamma_SD, wavelength, intensity);

	double axis[3] = { 1.0, 0.0, 0.0 };
	double eta = lamb_dicke_parameter(k_vector, axis, ion_mass(&ion), AXIAL_FREQUENCY);

	printf("Waveguide effective index: %f\n", grating->waveguide.n_eff);
	printf("Grating period: %f µm, emission angle: %f degrees, azimuth: %f degrees\n", grating->period, theta * 180 / PI, grating->azimuth * 180 / PI);
	printf("Intensity at the ion: %e W/m^2\n", intensity);
	printf("Rabi frequency of the S-D transition: 2 pi x %f kHz\n", rabi_frequency / (2 * PI * 1E3));
	printf("Lamb-Dicke parameter of the axial mode: %f\n", eta);

	free_trap(&trap);
}

// Simulate Rabi oscillations of one ion and a Molmer-Sorensen gate on two ions
static void simulate_pulses
(
)
{
	struct IonParameters ion;
	generate_ion_from_preset("40Ca+", &ion);

	struct MotionalMode mode = { AXIAL_FREQUENCY, 20, 0.05, { 0.0, 0.0 }, 0.0 };

	// Rabi oscillations on the carrier
	struct PulseProgram *pulse_program = allocate_pulse_program(1, 1);
	pulse_program->pulse_frequencies[0] = ion.omega_SD;
	pulse_program->pulse_durations[0] = 20E-6;
	pulse_program->pulse_n_targets_list[0] = 1;
	pulse_program->pulse_targets_list[0] = malloc(sizeof(int));
	pulse_program->pulse_targets_list[0][0] = 1;

	double rabi_frequencies[1] = { 2 * PI * 100E3 };
	double lamb_dicke_parameters[1] = { 0.1 };

	int n_samples = 11;
	double sample_times[11];
	double excited_populations[11];
	double mean_phonon_numbers[11];
	simulate_pulse_program(&ion, &mode, pulse_program, rabi_frequencies, lamb_dicke_parameters, 0.0, n_samples, sample_times, excited_populations, mean_phonon_numbers, NULL);

	for (int s = 0; s < n_samples; s++)
		printf("t = %6.2f µs: P_D = %f\n", sample_times[s] * 1E6, excited_populations[s]);

	free_pulse_program(pulse_program);

	// Molmer-Sorensen gate, with bichromatic light detuned by delta from the sidebands and eta Omega = delta / 2
	double delta = 2 * PI * 20E3;
	double eta = 0.05;

	struct PulseProgram *gate = allocate_pulse_program(2, 2);
	for (int p = 0; p < 2; p++)
	{
		gate->pulse_durations[p] = 2 * PI / delta;
		gate->pulse_n_targets_list[p] = 2;
		gate->pulse_targets_list[p] = malloc(2 * sizeof(int));
		gate->pulse_targets_list[p][0] = 1;
		gate->pulse_targets_list[p][1] = 2;
	}
	gate->pulse_frequencies[0] = ion.omega_SD + AXIAL_FREQUENCY + delta;
	gate->pulse_frequencies[1] = ion.omega_SD - AXIAL_FREQUENCY - delta;

	struct MotionalMode gate_mode = { AXIAL_FREQUENCY, 12, 0.0, { 0.0, 0.0 }, 0.0 };

	double gate_rabi_frequencies[2] = { delta / (2 * eta), delta / (2 * eta) };
	double gate_lamb_dicke_parameters[4] = { eta, eta, eta, eta };

	double gate_sample_times[2];
	double gate_excited_populations[4];
	double gate_mean_phonon_numbers[2];
	simulate_pulse_program(&ion, &gate_mode, gate, gate_rabi_frequencies, gate_lamb_dicke_parameters, 0.0, 2, gate_sample_times, gate_excited_populations, gate_mean_phonon_numbers, NULL);

	printf("After the gate: P_D = %f for ion 1 and %f for ion 2, mean phonon number %f\n", gate_excited_populations[2], gate_excited_populations[3], gate_mean_phonon_numbers[1]);

	free_pulse_program(gate);
}

// Co-design a trap with integrated photonics for a circuit: a basic design of everything, a pass optimizing each part
// a little on its own, and a pass optimizing each part for the quantum computing metrics of the whole design
static void run_codesign_flow
(
	char *qasm_filename
)
{
	struct Codesign *codesign = create_codesign("40Ca+", qasm_filename, NULL, NULL);
	if (codesign == NULL)
		return;

	run_codesign(codesign, 2, 10);

	printf("\nHistory of the co-design flow:\n");
	for (int r = 0; r < codesign->n_records; r++)
	{
		struct CodesignRecord *record = &codesign->records[r];

		printf("Pass %d, stage %d (%s): circuit fidelity %.6f, circuit duration %.3f ms\n", record->pass, record->stage, codesign_stage_name(record->stage), record->metrics.circuit_fidelity, record->metrics.circuit_duration * 1E3);
	}

	if (write_trap_to_file("codesign.trap", &codesign->trap) == 0)
		printf("Wrote the co-designed trap to codesign.trap\n");

	free_codesign(codesign);
}

// Main optimization loop
int main()
{
	printf(
		"|=---------------------------------------------------------------------------------------------------|\n"
		"|                                                                                                    |\n"
		"|                                   ████████╗██╗ ██████╗ ██████╗ ████████╗                           |\n"
		"|                                   ╚══██╔══╝██║██╔═══██╗██╔══██╗╚══██╔══╝                           |\n"
		"|                                      ██║   ██║██║   ██║██████╔╝   ██║                              |\n"
		"|                                      ██║   ██║██║   ██║██╔═══╝    ██║                              |\n"
		"|                                      ██║   ██║╚██████╔╝██║        ██║                              |\n"
		"|                                      ╚═╝   ╚═╝ ╚═════╝ ╚═╝        ╚═╝                              |\n"
		"|                                                                                                    |\n"
		"|                                                                                                    |\n"
		"|                                                                                                    |\n"
		"|                                                 XXXXXXX                                            |\n"
		"|                                               XX       XX                                          |\n"
		"|                                              X           X             \\                           |\n"
		"|                                             XX           XX   ==========》                         |\n"
		"|                                              X           X             /                           |\n"
		"|                                               XX       XX                                          |\n"
		"|                                                 XXXXXXX                                            |\n"
		"|                                                                                                    |\n"
		"|                                                                                                    |\n"
		"|                                 ########################################################           |\n"
		"|                                ##+++++++##+++++++##+++++++##+++++++##+++++++##+++++++###           |\n"
		"|                              ##+++++++##+++++++##+++++++##+++++++##+++++++##+++++++##.##           |\n"
		"|                            ##+++++++##+++++++##+++++++##+++++++##+++++++##+++++++##...##           |\n"
		"|                          ##+++++++##+++++++##+++++++##+++++++##+++++++##+++++++##.....##           |\n"
		"|                        #########################################################......##           |\n"
		"|                      #########################################################......##             |\n"
		"|                    #########################################################......##               |\n"
		"|                  #########################################################......##                 |\n"
		"|                ##+++++++##+++++++##+++++++##+++++++##+++++++##+++++++##.......##                   |\n"
		"|              ##+++++++##+++++++##+++++++##+++++++##+++++++##+++++++##.......##                     |\n"
		"|            ##+++++++##+++++++##+++++++##+++++++##+++++++##+++++++##.......##                       |\n"
		"|          ########################################################.......##                         |\n"
		"|          ##....................................................##.....##                           |\n"
		"|          ##....................................................##...##                             |\n"
		"|          ##....................................................##.##                               |\n"
		"|          #########################################################                                 |\n"
		"|                                                                                                    |\n"
		"|----------------------------------------------------------------------------------------------------|\n"
	);

	printf("\n");

	// @TODO - turn this into its own file/option
	// printf("SPH_R=%f\n", SPH_R);
	// printf("SPH_Z_MIN=%f\n", SPH_Z_MIN);
	// printf("SPH_SPACING=%f\n", SPH_SPACING);
	// printf("NSPH_X=%d, NSPH_Y=%d\n", NSPH_X, NSPH_Y);
	// printf("RELAXATION_NX=%d, RELAXATION_NY=%d, RELAXATION_NZ=%d\n", RELAXATION_NX, RELAXATION_NY, RELAXATION_NZ);

	// Accelerators are chosen through the environment, e.g. TIOPT_DEVICE=opencl and OMP_NUM_THREADS=8
	char *device = getenv("TIOPT_DEVICE");
	if (device != NULL && strcmp(device, "opencl") == 0)
		set_compute_device(OPENCL_DEVICE);

	printf("Running on %d threads%s%s\n", get_num_threads(), (get_compute_device() == OPENCL_DEVICE) ? " and OpenCL device " : "", (get_compute_device() == OPENCL_DEVICE) ? opencl_device_name() : "");

	int option = -1;

	printf("Welcome to tiopt!\n");

	while (option != 0)
	{
		printf("\nEnter a number to get started!\n");
		printf("(0) Quit\n");
		printf("(1) Get ion transport schedule for default inputs.\n");
		printf("(2) Get ion transport schedule for test qasm input.\n");
		printf("(3) Solve trap electrodynamics for trap.data.\n");
		printf("(4) Optimize and simulate an ion transport waveform.\n");
		printf("(5) Optimize trap geometry and RF voltage.\n");
		printf("(6) Design a grating coupler addressing the trap center.\n");
		printf("(7) Simulate Rabi oscillations and a Molmer-Sorensen gate.\n");
		printf("(8) Co-design a trap with integrated photonics for a GHZ circuit.\n");
		printf("Enter your option and press enter: ");

		int result = scanf("%d", &option);
		printf("\n");

		if (result != 1)
		{
			printf("Invalid input.\n");
			printf("\n---------------------------------------------------------------------------------\n");

			// Discard the rest of the line
			int i;
			while ((i = getchar()) != '\n' && i != EOF);

			if (i == EOF)
				break;

			continue;
		}

		switch (option)
		{
			case 0:
				printf("Quitting tiopt...\n");
				break;
			case 1:
				free_transport_program(simple_transport_schedule());
				break;
			case 2:
				free_transport_program(simple_transport_schedule_qasm_input());
				break;
			case 3:
				solve_trap_file("trap.data");
				break;
			case 4:
				simulate_transport();
				break;
			case 5:
				optimize_trap(20);
				break;
			case 6:
				design_grating();
				break;
			case 7:
				simulate_pulses();
				break;
			case 8:
				run_codesign_flow("examples/circuits/ghz3.qasm");
				break;
			default:
				printf("Not an option.\n");
				break;
		}

		printf("\n---------------------------------------------------------------------------------\n");
	}

	return 0;
}
