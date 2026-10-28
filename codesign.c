#include "codesign.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <math.h>

#include <nlopt.h>

#include "defs.h"
#include "data_structures.h"
#include "electrodynamics.h"
#include "traps.h"
#include "ions.h"
#include "photonics.h"
#include "waveforms.h"
#include "verlet.h"
#include "quantum.h"
#include "analysis.h"
#include "circuits.h"
#include "transport.h"

// Every operation of the circuit is run in one of these zones, which are spaced evenly along the trap axis: ions are
// loaded into and wait in the first zone, and gates and detection happen in the gate and measurement zones, with empty
// zones in between to keep the wells apart
#define N_CODESIGN_ZONES 5

// Number of single-qubit rotations which, with one Molmer-Sorensen gate, make up a controlled Pauli gate
#define ROTATIONS_PER_TWO_QUBIT_GATE 4

// Shuttles take at least this many axial periods in the first pass, since faster shuttles are far from adiabatic
#define MIN_SHUTTLE_PERIODS 10.0

// Parameters optimized by each stage, with their bounds; parameters spanning orders of magnitude are optimized in their
// logarithms
struct CodesignVariable
{
	enum CodesignStage stage;
	const char *name;
	size_t offset; // offset of the parameter within struct CodesignParameters
	double lower_bound;
	double upper_bound;
	int logarithmic;
};

static const struct CodesignVariable codesign_variables[] =
{
	{ TRAP_STAGE, "segment_width", offsetof(struct CodesignParameters, segment_width), 30.0, 80.0, 0 },
	{ TRAP_STAGE, "V_RF", offsetof(struct CodesignParameters, V_RF), 30.0, 400.0, 1 },
	{ TRAP_STAGE, "Omega_RF", offsetof(struct CodesignParameters, Omega_RF), 2 * PI * 15E6, 2 * PI * 80E6, 1 },
	{ PHOTONICS_STAGE, "grating_length", offsetof(struct CodesignParameters, grating_length), 8.0, 60.0, 0 },
	{ PHOTONICS_STAGE, "grating_width", offsetof(struct CodesignParameters, grating_width), 4.0, 40.0, 0 },
	{ PHOTONICS_STAGE, "rabi_fraction", offsetof(struct CodesignParameters, rabi_fraction), 0.01, 1.0, 1 },
	{ PHOTONICS_STAGE, "detector_size", offsetof(struct CodesignParameters, detector_size), 10.0, 150.0, 0 },
	{ PHOTONICS_STAGE, "detection_time", offsetof(struct CodesignParameters, detection_time), 2E-6, 1E-3, 1 },
	{ TRANSPORT_STAGE, "zone_spacing", offsetof(struct CodesignParameters, zone_spacing), 40.0, 110.0, 0 },
	{ TRANSPORT_STAGE, "shuttle_duration", offsetof(struct CodesignParameters, shuttle_duration), 2E-6, 2E-4, 1 },
	{ WAVEFORM_STAGE, "omega_axial", offsetof(struct CodesignParameters, omega_axial), 2 * PI * 0.5E6, 2 * PI * 3E6, 1 },
	{ WAVEFORM_STAGE, "log_roughness_weight", offsetof(struct CodesignParameters, log_roughness_weight), -2.0, 6.0, 0 },
};

#define N_CODESIGN_VARIABLES ((int) (sizeof(codesign_variables) / sizeof(codesign_variables[0])))

// Get the name of a stage of the co-design flow
const char* codesign_stage_name
(
	enum CodesignStage stage
)
{
	const char *names[N_CODESIGN_STAGES] = { "basic design", "trap", "photonics", "transport", "waveforms", "metrics" };

	return (stage >= 0 && stage < N_CODESIGN_STAGES) ? names[stage] : "unknown";
}

// Fill in the basic design (stage 0) of the trap, photonics, transport program and waveforms
void default_codesign_parameters
(
	struct CodesignParameters *parameters
)
{
	parameters->segment_width = 50.0;
	parameters->V_RF = 35.0;
	parameters->Omega_RF = 2 * PI * 40E6;

	parameters->grating_length = 20.0;
	parameters->grating_width = 8.0;
	parameters->rabi_fraction = 0.2;
	parameters->detector_size = 50.0;
	parameters->detection_time = 50E-6;

	parameters->zone_spacing = 100.0;
	parameters->shuttle_duration = 20E-6;

	parameters->omega_axial = AXIAL_FREQUENCY;
	parameters->log_roughness_weight = log10(WAVEFORM_ROUGHNESS_WEIGHT);
}

// A room-temperature surface trap with filtered DC electronics and a sideband-cooled ion
void default_codesign_environment
(
	struct CodesignEnvironment *environment
)
{
	environment->n_segments = 15;
	environment->max_voltage = 20.0;
	environment->qubit_power = 1E-3;
	environment->detection_power = 1E-5;
	environment->field_noise = 1E-12;
	environment->field_noise_exponent = 1.0;
	environment->voltage_noise = 1E-19;
	environment->dephasing_rate = 100.0;
	environment->n_bar_cooled = 0.05;
	environment->position_drift = 0.1;
	environment->detector_efficiency = 0.8;
	environment->dark_count_density = 1E-3;
	environment->target_radial_frequency = 2 * PI * 4E6;
	environment->max_mathieu_q = 0.4;
	environment->min_trap_depth = 0.02;
}

static double* parameter_pointer
(
	struct CodesignParameters *parameters,
	int v
)
{
	return (double *) ((char *) parameters + codesign_variables[v].offset);
}

// Map a parameter onto [0, 1] within its bounds, and back
static double normalize_parameter
(
	int v,
	double value
)
{
	const struct CodesignVariable *variable = &codesign_variables[v];

	if (variable->logarithmic)
		return log(value / variable->lower_bound) / log(variable->upper_bound / variable->lower_bound);

	return (value - variable->lower_bound) / (variable->upper_bound - variable->lower_bound);
}

static double denormalize_parameter
(
	int v,
	double u
)
{
	const struct CodesignVariable *variable = &codesign_variables[v];

	u = MIN(MAX(u, 0.0), 1.0);

	if (variable->logarithmic)
		return variable->lower_bound * pow(variable->upper_bound / variable->lower_bound, u);

	return variable->lower_bound + u * (variable->upper_bound - variable->lower_bound);
}

// Position of a zone along the trap axis, with the zones centered on the trap
static double zone_x
(
	struct Codesign *codesign,
	int zone
)
{
	return (zone - 0.5 * (codesign->n_zones - 1)) * codesign->parameters.zone_spacing;
}

// How far the outermost zone lies beyond the region where wells can be made: over the segmented electrodes (less a
// segment at each end) and within the spherical harmonics expansions (µm); zero or less if every zone fits
static double zone_overflow
(
	struct Codesign *codesign
)
{
	double segments = 0.5 * codesign->environment.n_segments * codesign->parameters.segment_width - codesign->parameters.segment_width;
	double expansions = MIN(-SPH_CENTER_X(0), SPH_CENTER_X(NSPH_X - 1));

	return zone_x(codesign, codesign->n_zones - 1) - MIN(segments, expansions);
}

// Summarize the circuit in a QASM file by the layers of the transport program which runs it on the zones of the trap,
// with distances in zone spacings
static int summarize_circuit
(
	struct Codesign *codesign,
	char *qasm_filename
)
{
	struct CircuitGraph *circuit_graph = parse_qasm(qasm_filename);

	if (circuit_graph == NULL || circuit_graph->n_ions <= 0 || circuit_graph->n_instruction_nodes <= 0)
	{
		printf("ERROR: Failed to read a circuit from '%s'\n", qasm_filename);

		if (circuit_graph != NULL)
			free_circuit_graph(circuit_graph);

		return 1;
	}

	int n_ions = circuit_graph->n_ions;

	// A zone where ions are loaded and wait, a gate zone and a measurement zone, separated by empty zones which ions
	// pass through
	enum ZoneType zone_types[N_CODESIGN_ZONES] = { LOAD_ZONE | STORAGE_ZONE, EMPTY_ZONE, GATE_ZONE, EMPTY_ZONE, MEASURE_ZONE };
	int zone_capacities[N_CODESIGN_ZONES] = { n_ions, 1, 2, 1, n_ions };

	codesign->n_zones = N_CODESIGN_ZONES;
	codesign->gate_zone = 2;
	codesign->measure_zone = 4;

	// Zones one unit apart, so that distances come out in zone spacings
	struct TrapGraph trap_graph;
	generate_linear_trap_graph(N_CODESIGN_ZONES, zone_types, zone_capacities, 0.0, N_CODESIGN_ZONES - 1, ION_HEIGHT, &trap_graph);

	struct TransportProgram *transport_program = calloc(1, sizeof(struct TransportProgram));
	if (generate_transport_schedule(&trap_graph, circuit_graph, transport_program) == NULL)
	{
		printf("ERROR: Failed to schedule the circuit on the trap\n");

		free_trap_graph(&trap_graph);
		free_circuit_graph(circuit_graph);
		free_transport_program(transport_program);

		return 1;
	}

	int n_layers = 0;
	for (int i = 0; i < circuit_graph->n_instruction_nodes; i++)
		n_layers = MAX(n_layers, circuit_graph->layer_idxs[i]);

	codesign->n_ions = n_ions;
	codesign->n_layers = n_layers;
	codesign->layer_distances = calloc(n_layers, sizeof(double));
	codesign->layer_single_qubit_gates = calloc(n_layers, sizeof(int));
	codesign->layer_two_qubit_gates = calloc(n_layers, sizeof(int));
	codesign->layer_measurements = calloc(n_layers, sizeof(int));

	double *ion_x = malloc(n_ions * sizeof(double));
	for (int i = 0; i < n_ions; i++)
		ion_x[i] = transport_program->initial_positions[i][0];

	// The zones occupied at once by the most wells, which must all be held while ions wait for others
	codesign->n_static_wells = 1;
	codesign->static_well_zones[0] = (int) lround(ion_x[0]);

	for (int layer = 1; layer <= n_layers; layer++)
	{
		for (int t = 0; t < transport_program->n_transports; t++)
		{
			if (circuit_graph->layer_idxs[t] != layer)
				continue;

			struct InstructionNode *node = &circuit_graph->instruction_nodes[t];

			for (int i = 0; i < transport_program->transport_n_targets_list[t]; i++)
			{
				int ion = transport_program->transport_targets_list[t][i] - 1;

				codesign->layer_distances[layer - 1] = MAX(codesign->layer_distances[layer - 1], fabs(transport_program->destinations[t][0] - ion_x[ion]));
				ion_x[ion] = transport_program->destinations[t][0];
			}

			if (node->operation_type & DETECT)
				codesign->layer_measurements[layer - 1] += node->n_target_ions;
			else if (node->n_target_ions >= 2)
				codesign->layer_two_qubit_gates[layer - 1]++;
			else
				codesign->layer_single_qubit_gates[layer - 1]++;

			int occupied[N_CODESIGN_ZONES] = { 0 };
			for (int i = 0; i < n_ions; i++)
				occupied[MIN(MAX((int) lround(ion_x[i]), 0), N_CODESIGN_ZONES - 1)] = 1;

			int n_occupied = 0;
			for (int z = 0; z < N_CODESIGN_ZONES; z++)
				n_occupied += occupied[z];

			if (n_occupied > codesign->n_static_wells)
			{
				codesign->n_static_wells = 0;
				for (int z = 0; z < N_CODESIGN_ZONES; z++)
					if (occupied[z])
						codesign->static_well_zones[codesign->n_static_wells++] = z;
			}
		}
	}

	free(ion_x);
	free_transport_program(transport_program);
	free_trap_graph(&trap_graph);
	free_circuit_graph(circuit_graph);

	return 0;
}

// Make the trap match the trap parameters (solving its electrodynamics again only if its geometry changed), find the
// RF nulls at the gate and measurement zones, and set up its photonics to address them
static int update_trap
(
	struct Codesign *codesign
)
{
	struct CodesignParameters *parameters = &codesign->parameters;
	struct CodesignEnvironment *environment = &codesign->environment;

	if (codesign->solved_segment_width != parameters->segment_width)
	{
		free_trap(&codesign->trap);

		codesign->solved_segment_width = 0.0;

		if (generate_trap_from_preset(LINEAR_SURFACE_TRAP, environment->n_segments, parameters->segment_width, &codesign->trap) != 0)
			return 1;

		solve_trap_electrodynamics(&codesign->trap, ANALYTICAL, NULL);

		codesign->solved_segment_width = parameters->segment_width;
		codesign->primitive_status = -1;
	}

	codesign->trap.V_RF = parameters->V_RF;
	codesign->trap.Omega_RF = parameters->Omega_RF;

	double gate_position[3] = { zone_x(codesign, codesign->gate_zone), 0.0, ION_HEIGHT };
	double measure_position[3] = { zone_x(codesign, codesign->measure_zone), 0.0, ION_HEIGHT };
	find_rf_null(&codesign->trap, &gate_position, 100);
	find_rf_null(&codesign->trap, &measure_position, 100);

	memcpy(codesign->gate_position, gate_position, sizeof(gate_position));
	memcpy(codesign->measure_position, measure_position, sizeof(measure_position));

	if (add_trap_photonics(&codesign->trap, codesign->gate_position, codesign->measure_position) != 0)
		return 1;

	struct GratingCoupler *qubit_grating = &(*codesign->trap.grating_couplers)[QUBIT_BEAM];
	qubit_grating->length = parameters->grating_length;
	qubit_grating->width = parameters->grating_width;
	qubit_grating->power = environment->qubit_power;

	(*codesign->trap.grating_couplers)[DETECTION_BEAM].power = environment->detection_power;

	// Dark counts scale with the area of the detector
	struct Detector *detector = &(*codesign->trap.detectors)[0];
	detector->width = parameters->detector_size;
	detector->length = parameters->detector_size;
	detector->quantum_efficiency = environment->detector_efficiency;
	detector->dark_count_rate = environment->dark_count_density * parameters->detector_size * parameters->detector_size;

	return 0;
}

// Radial confinement, depth, heating and static voltages of the trap at the gate zone
static void evaluate_trap_metrics
(
	struct Codesign *codesign,
	struct QCMetrics *metrics
)
{
	struct Trap *trap = &codesign->trap;
	struct CodesignParameters *parameters = &codesign->parameters;
	struct CodesignEnvironment *environment = &codesign->environment;

	double m = ion_mass(&codesign->ion);
	double q = ion_charge(&codesign->ion);

	// The radial modes are the two stiffest modes of the pseudopotential alone
	double voltages[trap->n_electrodes];
	memset(voltages, 0, sizeof(voltages));

	double omega[3];
	double axes[3][3];
	compute_normal_modes(trap, m, q, voltages, codesign->gate_position, &omega, &axes);

	double omega_sorted[3] = { omega[0], omega[1], omega[2] };
	for (int i = 0; i < 3; i++)
		for (int j = i + 1; j < 3; j++)
			if (omega_sorted[j] < omega_sorted[i])
			{
				double swap = omega_sorted[i];
				omega_sorted[i] = omega_sorted[j];
				omega_sorted[j] = swap;
			}

	metrics->radial_frequency = omega_sorted[1];
	metrics->mathieu_q = 2 * SQRT2 * omega_sorted[2] / parameters->Omega_RF;
	metrics->trap_depth = compute_trap_depth(trap, m, q, codesign->gate_position);

	// Heating of the axial mode by electric field noise from the electrode surfaces and by voltage noise on the DC
	// electrodes
	double axis[3] = { 1.0, 0.0, 0.0 };
	double S_E = environment->field_noise * pow(2 * PI * 1E6 / parameters->omega_axial, environment->field_noise_exponent);
	metrics->heating_rate = heating_rate_field_noise(m, q, parameters->omega_axial, S_E);
	metrics->heating_rate += heating_rate_voltage_noise(trap, m, q, codesign->gate_position, parameters->omega_axial, axis, environment->voltage_noise);

	// Voltages which hold wells in every zone the circuit occupies at once (as when ions wait while others are gated);
	// wells closer together than several ion heights need large voltages to be controlled separately
	int n_wells = codesign->n_static_wells;
	double well_positions[N_CODESIGN_ZONES][3];
	for (int w = 0; w < n_wells; w++)
	{
		well_positions[w][0] = zone_x(codesign, codesign->static_well_zones[w]);
		well_positions[w][1] = 0.0;
		well_positions[w][2] = ION_HEIGHT;
		find_rf_null(trap, &well_positions[w], 100);
	}

	metrics->static_voltage = HUGE_VAL;
	if (solve_waveform(trap, m, q, 1, n_wells, well_positions, parameters->omega_axial, 0.0, voltages) == 0)
	{
		metrics->static_voltage = 0.0;
		for (int e = trap->n_electrodes_rf; e < trap->n_electrodes; e++)
			metrics->static_voltage = MAX(metrics->static_voltage, fabs(voltages[e]));
	}
}

// Light delivered to the gate and measurement zones, and the fidelity of detection
static void evaluate_photonics_metrics
(
	struct Codesign *codesign,
	struct QCMetrics *metrics
)
{
	struct Trap *trap = &codesign->trap;
	struct IonParameters *ion = &codesign->ion;
	struct CodesignParameters *parameters = &codesign->parameters;
	struct CodesignEnvironment *environment = &codesign->environment;

	double m = ion_mass(ion);

	struct GratingCoupler *qubit_grating = &(*trap->grating_couplers)[QUBIT_BEAM];

	double intensity;
	double k_vector[3];
	evaluate_grating_beam(qubit_grating, codesign->gate_position, &intensity, &k_vector);

	double wavelength_SD = 2 * PI * C_LIGHT / ion->omega_SD;
	metrics->rabi_frequency = parameters->rabi_fraction * rabi_frequency_from_intensity(ion->gamma_SD, wavelength_SD, intensity);

	double axis[3] = { 1.0, 0.0, 0.0 };
	metrics->lamb_dicke_parameter = lamb_dicke_parameter(k_vector, axis, m, parameters->omega_axial);

	// Ions waiting in the neighboring zones see the tail of the beam
	metrics->crosstalk = 0.0;
	for (int side = -1; side <= 1; side += 2)
	{
		double neighbor[3] = { codesign->gate_position[0] + side * parameters->zone_spacing, codesign->gate_position[1], codesign->gate_position[2] };

		double neighbor_intensity;
		evaluate_grating_beam(qubit_grating, neighbor, &neighbor_intensity, NULL);

		metrics->crosstalk = MAX(metrics->crosstalk, neighbor_intensity / intensity);
	}

	// Rabi frequencies scale as the square root of the intensity, so the relative change of the Rabi frequency when
	// the ion drifts (or is left off target by transport) is half the change in the logarithm of the intensity, in the
	// most sensitive direction
	double drift = sqrt(pow(environment->position_drift, 2) + pow(codesign->position_error, 2));
	metrics->rabi_sensitivity = 0.0;
	for (int k = 0; k < 3; k++)
	{
		double displaced[3] = { codesign->gate_position[0], codesign->gate_position[1], codesign->gate_position[2] };
		displaced[k] += drift;

		double displaced_intensity;
		evaluate_grating_beam(qubit_grating, displaced, &displaced_intensity, NULL);

		metrics->rabi_sensitivity = MAX(metrics->rabi_sensitivity, 0.5 * fabs(log(displaced_intensity / intensity)));
	}

	// Detection by fluorescence on the S-P transition, driven half a linewidth below resonance; the dark state decays
	// at the rate of the D state
	struct GratingCoupler *detection_grating = &(*trap->grating_couplers)[DETECTION_BEAM];

	double detection_intensity;
	evaluate_grating_beam(detection_grating, codesign->measure_position, &detection_intensity, NULL);

	double wavelength_SP = 2 * PI * C_LIGHT / ion->omega_SP;
	double s = detection_intensity / saturation_intensity(ion->gamma_SP, wavelength_SP);
	double scattering_rate = compute_scattering_rate(ion->gamma_SP, s, -0.5 * ion->gamma_SP);

	struct Detector *detector = &(*trap->detectors)[0];
	double photon_rate = scattering_rate * compute_collection_efficiency(detector, codesign->measure_position);

	metrics->detection_fidelity = compute_detection_fidelity(photon_rate, detector->dark_count_rate, ion->gamma_SD, parameters->detection_time, NULL);
}

// Optimize and simulate the shuttle primitive, which moves an ion from the zone before the gate zone into the gate
// zone; the results are kept until a parameter the primitive depends on changes. Returns 1 if the waveform could not
// be found
static int update_primitive
(
	struct Codesign *codesign,
	double radial_frequency
)
{
	struct Trap *trap = &codesign->trap;
	struct IonParameters *ion = &codesign->ion;
	struct CodesignParameters *parameters = &codesign->parameters;

	double key[7] =
	{
		parameters->segment_width,
		parameters->V_RF,
		parameters->Omega_RF,
		parameters->zone_spacing,
		parameters->shuttle_duration,
		parameters->omega_axial,
		parameters->log_roughness_weight
	};

	if (codesign->primitive_status >= 0 && memcmp(key, codesign->primitive_key, sizeof(key)) == 0)
		return codesign->primitive_status;

	memcpy(codesign->primitive_key, key, sizeof(key));
	codesign->primitive_status = 1;

	double m = ion_mass(ion);
	double q = ion_charge(ion);

	double start[3] = { zone_x(codesign, codesign->gate_zone - 1), 0.0, ION_HEIGHT };
	find_rf_null(trap, &start, 100);

	double *end = codesign->gate_position;

	// Steps much shorter than the axial period, so that stepping the voltages does not excite the ion
	double duration = parameters->shuttle_duration;
	double period = 2 * PI / parameters->omega_axial;
	int n_steps = (int) MIN(MAX(ceil(20 * duration / period), 51), 2001);

	struct TransportWaveform *waveform = generate_transport_primitive(trap, ion, start, end, duration, n_steps, parameters->omega_axial, pow(10, parameters->log_roughness_weight));

	if (waveform == NULL)
		return 1;

	codesign->peak_voltage = 0.0;
	for (int s = 0; s < n_steps; s++)
		for (int e = trap->n_electrodes_rf; e < trap->n_electrodes; e++)
			codesign->peak_voltage = MAX(codesign->peak_voltage, fabs(waveform->voltages[s * trap->n_electrodes + e]));

	double max_force;
	evaluate_waveform_errors(trap, ion, waveform, parameters->omega_axial, &max_force, &codesign->frequency_error);

	// Equilibria of the ion at the start and end of the shuttle
	double initial[3] = { start[0], start[1], start[2] };
	double final[3] = { end[0], end[1], end[2] };
	find_potential_minimum(trap, m, q, waveform->voltages, &initial, 100);
	find_potential_minimum(trap, m, q, &waveform->voltages[(n_steps - 1) * trap->n_electrodes], &final, 100);

	codesign->position_error = sqrt(pow(final[0] - end[0], 2) + pow(final[1] - end[1], 2) + pow(final[2] - end[2], 2));

	// Follow an ion which starts at rest through the shuttle and a few periods after it, with steps resolving the
	// fastest secular motion
	double omega_max = MAX(parameters->omega_axial, radial_frequency);
	double dt = 2 * PI / omega_max / 50;
	int n_it = (int) ceil((duration + 3 * period) / dt);

	double position[1][3] = { { initial[0], initial[1], initial[2] } };
	double velocity[1][3] = { { 0.0, 0.0, 0.0 } };
	simulate_ensemble(trap, m, q, waveform, 0, 1, position, velocity, n_it, dt);

	double axis[3] = { 1.0, 0.0, 0.0 };
	codesign->shuttle_excitation = motional_quanta(m, parameters->omega_axial, axis, position[0], velocity[0], final);

	free_transport_waveform(waveform);

	codesign->primitive_status = isfinite(codesign->shuttle_excitation) ? 0 : 1;

	return codesign->primitive_status;
}

// Fidelity with a Bell state of two ions after a Molmer-Sorensen gate, simulated with the master equation: bichromatic
// light detuned by delta from both sidebands of the axial mode, with eta Omega = delta / 2 so that the ions entangle
// after one loop in phase space
static double simulate_two_qubit_gate
(
	struct IonParameters *ion,
	struct CodesignEnvironment *environment,
	const double omega_axial,
	const double eta,
	const double rabi_frequency,
	const double n_bar,
	const double heating_rate
)
{
	double delta = 2 * eta * rabi_frequency;

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
	pulse_program->pulse_frequencies[0] = ion->omega_SD + omega_axial + delta;
	pulse_program->pulse_frequencies[1] = ion->omega_SD - omega_axial - delta;

	// Enough Fock states for the thermal distribution (whose tail beyond n_max is below 1E-4), the heating during the
	// gate and the displacement of the motion by the gate
	double n_bar_final = n_bar + heating_rate * 2 * PI / delta;
	int n_max = (int) ceil(log(1E-4) / log(n_bar_final / (n_bar_final + 1))) + 6;
	n_max = MIN(MAX(n_max, 8), 50);

	struct MotionalMode mode = { omega_axial, n_max, n_bar, { 0.0, 0.0 }, heating_rate };

	double rabi_frequencies[2] = { rabi_frequency, rabi_frequency };
	double etas[4] = { eta, eta, eta, eta };

	int n_motion = n_max + 1;
	int dim = 4 * n_motion;
	double (*rho)[2] = malloc((size_t) dim * dim * sizeof(double[2]));

	double sample_times[2];
	double excited_populations[4];
	double mean_phonon_numbers[2];
	simulate_pulse_program(ion, &mode, pulse_program, rabi_frequencies, etas, environment->dephasing_rate, 2, sample_times, excited_populations, mean_phonon_numbers, rho);

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

	free(rho);
	free_pulse_program(pulse_program);

	return 0.5 * (P_gg + P_ee) + hypot(coherence[0], coherence[1]);
}

// Gate durations and errors, the duration of the circuit and the probability that it runs without error, simulating
// the two-qubit gates with the master equation if simulate_gates is set and estimating their errors otherwise
static void evaluate_circuit_metrics
(
	struct Codesign *codesign,
	int simulate_gates,
	struct QCMetrics *metrics
)
{
	struct CodesignParameters *parameters = &codesign->parameters;
	struct CodesignEnvironment *environment = &codesign->environment;

	double gamma = environment->dephasing_rate;
	double eta = metrics->lamb_dicke_parameter;
	double nu = parameters->omega_axial;

	// The light is split between the two tones of the two-qubit gates
	double rabi_frequency = metrics->rabi_frequency;
	double rabi_frequency_tone = rabi_frequency / SQRT2;

	metrics->single_qubit_time = 0.5 * PI / rabi_frequency;
	metrics->two_qubit_time = 2 * PI / (2 * eta * rabi_frequency_tone);

	double t_1 = metrics->single_qubit_time;
	double t_2 = metrics->two_qubit_time;
	double t_detect = parameters->detection_time;

	// Run through the layers of the circuit: the ions are cooled at the start and are not cooled again, so every shuttle
	// adds its excitation (treated as thermal) and the mode heats throughout. Operations in a layer share the single gate
	// zone and measurement zone, so they run one after another
	double time = 0.0;
	double n_shuttles = 0.0;
	double n_bar_sum = 0.0;
	int n_single_qubit_gates = 0;
	int n_two_qubit_gates = 0;
	int n_measurements = 0;
	for (int layer = 0; layer < codesign->n_layers; layer++)
	{
		time += codesign->layer_distances[layer] * parameters->shuttle_duration;
		n_shuttles += codesign->layer_distances[layer];

		int n_single = codesign->layer_single_qubit_gates[layer];
		int n_two = codesign->layer_two_qubit_gates[layer];
		int n_detect = codesign->layer_measurements[layer];

		double n_bar = environment->n_bar_cooled + metrics->shuttle_excitation * n_shuttles + metrics->heating_rate * time;
		n_bar_sum += n_two * n_bar;

		n_single_qubit_gates += n_single + ROTATIONS_PER_TWO_QUBIT_GATE * n_two;
		n_two_qubit_gates += n_two;
		n_measurements += n_detect;

		time += n_single * t_1 + n_two * (t_2 + ROTATIONS_PER_TWO_QUBIT_GATE * t_1) + n_detect * t_detect;
	}

	metrics->circuit_duration = time;
	metrics->gate_n_bar = (n_two_qubit_gates > 0) ? n_bar_sum / n_two_qubit_gates : environment->n_bar_cooled + metrics->heating_rate * time;

	double n_bar = metrics->gate_n_bar;

	// Spectator ions in the neighboring zones are rotated by the tail of the beam
	int n_spectators = MIN(codesign->n_ions - 1, 2);
	double spectator_error = n_spectators * pow(sin(0.25 * PI * sqrt(metrics->crosstalk)), 2);

	// Single-qubit rotations by PI / 2 err with the spread of the carrier Rabi frequency over the thermal state
	// (Debye-Waller factor), with drifts of the ion within the beam, and with dephasing
	double thermal_spread = eta*eta * sqrt(n_bar * (n_bar + 1));
	double single_qubit_error = pow(0.25 * PI * thermal_spread, 2) + pow(0.25 * PI * metrics->rabi_sensitivity, 2) + 0.5 * gamma * t_1 + spectator_error;

	metrics->single_qubit_fidelity = 1 - single_qubit_error;

	// Two-qubit gates also err with the off-resonant carrier excitation of both ions by both tones (which the
	// simulation, in the resolved sideband approximation, leaves out) and with drifts of eta Omega from its calibration
	double carrier_error = 2 * pow(rabi_frequency_tone / nu, 2);
	double drift_error = pow(0.5 * PI * metrics->rabi_sensitivity, 2);

	double gate_fidelity;
	if (simulate_gates)
		gate_fidelity = simulate_two_qubit_gate(&codesign->ion, environment, nu, eta, rabi_frequency_tone, n_bar, metrics->heating_rate);
	else
		gate_fidelity = 1 - 0.5 * metrics->heating_rate * t_2 - 0.5 * gamma * t_2 - 0.25 * PI*PI * pow(eta, 4) * (n_bar*n_bar + n_bar);

	metrics->two_qubit_fidelity = gate_fidelity * (1 - carrier_error) * (1 - drift_error) * (1 - spectator_error);

	// Every ion also dephases while it waits for the rest of the circuit
	double idle_fidelity = 1 - 0.5 * (1 - exp(-gamma * time));

	double log_fidelity = 0.0;
	log_fidelity += n_single_qubit_gates * log(MAX(metrics->single_qubit_fidelity, 1E-300));
	log_fidelity += n_two_qubit_gates * log(MAX(metrics->two_qubit_fidelity, 1E-300));
	log_fidelity += n_measurements * log(MAX(metrics->detection_fidelity, 1E-300));
	log_fidelity += codesign->n_ions * log(MAX(idle_fidelity, 1E-300));

	metrics->circuit_fidelity = exp(log_fidelity);
}

// Penalty for designs which cannot be run: unstable or shallow traps, and waveforms beyond the voltage limit
static double feasibility_penalty
(
	struct Codesign *codesign,
	struct QCMetrics *metrics
)
{
	struct CodesignEnvironment *environment = &codesign->environment;

	double penalty = 0.0;

	if (!(metrics->radial_frequency > 0.0))
		penalty += 1E2;

	if (metrics->mathieu_q > environment->max_mathieu_q)
		penalty += 1E2 * pow(metrics->mathieu_q / environment->max_mathieu_q - 1, 2);

	if (metrics->trap_depth < environment->min_trap_depth)
		penalty += 1E2 * pow(1 - metrics->trap_depth / environment->min_trap_depth, 2);

	if (metrics->peak_voltage > environment->max_voltage)
		penalty += 1E2 * pow(metrics->peak_voltage / environment->max_voltage - 1, 2);

	if (metrics->static_voltage > environment->max_voltage)
		penalty += 1E2 * pow(metrics->static_voltage / environment->max_voltage - 1, 2);

	return penalty;
}

// Evaluate every figure of merit of the design (see evaluate_circuit_metrics for simulate_gates); returns 1 if the
// design cannot be evaluated, in which case metrics->objective is a large penalty
int evaluate_codesign
(
	struct Codesign *codesign,
	int simulate_gates,
	struct QCMetrics *metrics
)
{
	memset(metrics, 0, sizeof(struct QCMetrics));

	double overflow = zone_overflow(codesign);
	if (overflow > 0.0)
	{
		metrics->objective = 1E3 * (1 + overflow);
		return 1;
	}

	if (update_trap(codesign) != 0)
	{
		metrics->objective = 1E3;
		return 1;
	}

	evaluate_trap_metrics(codesign, metrics);

	if (update_primitive(codesign, metrics->radial_frequency) != 0)
	{
		metrics->objective = 1E3;
		return 1;
	}

	metrics->shuttle_excitation = codesign->shuttle_excitation;
	metrics->position_error = codesign->position_error;
	metrics->frequency_error = codesign->frequency_error;
	metrics->peak_voltage = codesign->peak_voltage;

	evaluate_photonics_metrics(codesign, metrics);
	evaluate_circuit_metrics(codesign, simulate_gates, metrics);

	metrics->objective = -log(MAX(metrics->circuit_fidelity, 1E-300)) + feasibility_penalty(codesign, metrics);

	return 0;
}

// Objective of a stage in the first pass, which only looks at the part of the design the stage is responsible for
static double local_objective
(
	struct Codesign *codesign,
	enum CodesignStage stage,
	struct QCMetrics *metrics
)
{
	struct CodesignParameters *parameters = &codesign->parameters;
	struct CodesignEnvironment *environment = &codesign->environment;

	double objective = feasibility_penalty(codesign, metrics);

	if (stage == TRAP_STAGE)
	{
		// Radial confinement near the target, wells in every zone which leave room within the voltage limit for
		// transport, and little heating over the time of a gate
		objective += pow(metrics->radial_frequency / environment->target_radial_frequency - 1, 2);
		objective += 10 * pow(MAX(metrics->static_voltage / (0.5 * environment->max_voltage) - 1, 0.0), 2);
		objective += metrics->heating_rate * GATE_DURATION;
	}
	else if (stage == PHOTONICS_STAGE)
	{
		// Errors of single operations
		objective += (1 - metrics->single_qubit_fidelity) + (1 - metrics->two_qubit_fidelity) + (1 - metrics->detection_fidelity);
	}
	else if (stage == TRANSPORT_STAGE)
	{
		// Fraction of the circuit spent shuttling, with shuttles kept adiabatic and spectators kept out of the beam
		double transport_time = 0.0;
		for (int layer = 0; layer < codesign->n_layers; layer++)
			transport_time += codesign->layer_distances[layer] * parameters->shuttle_duration;

		objective += transport_time / MAX(metrics->circuit_duration, 1E-12);
		objective += 10 * pow(MAX(1 - parameters->shuttle_duration * parameters->omega_axial / (2 * PI * MIN_SHUTTLE_PERIODS), 0.0), 2);
		objective += 1E2 * metrics->crosstalk;
	}
	else if (stage == WAVEFORM_STAGE)
	{
		// Quiet shuttles which deliver the ion to its target in a well of the intended frequency
		objective += metrics->shuttle_excitation;
		objective += pow(metrics->position_error / 0.1, 2);
		objective += pow(metrics->frequency_error / 0.05, 2);
	}

	return objective;
}

struct StageObjectiveData
{
	struct Codesign *codesign;
	enum CodesignStage stage;
	int qc_aware;
	int n_variables;
	int variables[N_CODESIGN_VARIABLES];
	int n_evaluations;
};

static double stage_objective
(
	unsigned n,
	const double *u,
	double *grad,
	void *stage_data
)
{
	struct StageObjectiveData *data = (struct StageObjectiveData *) stage_data;
	struct Codesign *codesign = data->codesign;

	for (int i = 0; i < data->n_variables; i++)
		*parameter_pointer(&codesign->parameters, data->variables[i]) = denormalize_parameter(data->variables[i], u[i]);

	struct QCMetrics metrics;
	int status = evaluate_codesign(codesign, data->qc_aware, &metrics);

	double objective = metrics.objective;
	if (status == 0 && !data->qc_aware)
		objective = local_objective(codesign, data->stage, &metrics);

	data->n_evaluations++;

	printf("  %s evaluation %d: objective %.6e (circuit fidelity %.6f)\n", codesign_stage_name(data->stage), data->n_evaluations, objective, metrics.circuit_fidelity);

	return objective;
}

// Append the current design to the history of the flow
static void record_codesign
(
	struct Codesign *codesign,
	int pass,
	enum CodesignStage stage,
	int n_evaluations,
	double objective
)
{
	if (codesign->n_records >= codesign->max_records)
	{
		codesign->max_records = 2 * codesign->max_records + N_CODESIGN_STAGES;
		codesign->records = realloc(codesign->records, codesign->max_records * sizeof(struct CodesignRecord));
	}

	struct CodesignRecord *record = &codesign->records[codesign->n_records];

	record->pass = pass;
	record->stage = stage;
	record->n_evaluations = n_evaluations;
	record->objective = objective;
	record->parameters = codesign->parameters;
	record->metrics = codesign->metrics;

	// Readers polling the history from another thread only see complete records
	int n_records = codesign->n_records + 1;

	#pragma omp atomic write
	codesign->n_records = n_records;
}

// Optimize the parameters of one stage a little (with at most max_evaluations evaluations of the design), with either
// the stage's own objective or the QC-aware objective of the whole design, leaving the best parameters found in the
// design and the metrics of the design in codesign->metrics; the objective reached and the number of evaluations made
// are returned through objective and n_evaluations if they are not NULL, and the result of NLopt is returned
int optimize_codesign_stage
(
	struct Codesign *codesign,
	enum CodesignStage stage,
	int qc_aware,
	int max_evaluations,
	double *objective,
	int *n_evaluations
)
{
	struct StageObjectiveData data = { codesign, stage, qc_aware, 0, { 0 }, 0 };

	for (int v = 0; v < N_CODESIGN_VARIABLES; v++)
		if (codesign_variables[v].stage == stage)
			data.variables[data.n_variables++] = v;

	if (data.n_variables == 0)
		return 0;

	double u[N_CODESIGN_VARIABLES];
	double lower_bounds[N_CODESIGN_VARIABLES];
	double upper_bounds[N_CODESIGN_VARIABLES];
	double steps[N_CODESIGN_VARIABLES];
	for (int i = 0; i < data.n_variables; i++)
	{
		u[i] = MIN(MAX(normalize_parameter(data.variables[i], *parameter_pointer(&codesign->parameters, data.variables[i])), 0.0), 1.0);
		lower_bounds[i] = 0.0;
		upper_bounds[i] = 1.0;
		steps[i] = 0.1;
	}

	printf("Optimizing %s with the %s objective...\n", codesign_stage_name(stage), qc_aware ? "QC-aware" : "local");

	nlopt_opt opt = nlopt_create(NLOPT_LN_COBYLA, data.n_variables);
	nlopt_set_lower_bounds(opt, lower_bounds);
	nlopt_set_upper_bounds(opt, upper_bounds);
	nlopt_set_initial_step(opt, steps);
	nlopt_set_maxeval(opt, max_evaluations);
	nlopt_set_xtol_abs1(opt, 1E-3);
	nlopt_set_min_objective(opt, stage_objective, &data);

	double best_objective = HUGE_VAL;
	int result = nlopt_optimize(opt, u, &best_objective);

	nlopt_destroy(opt);

	// NLopt leaves the best point it found in u
	for (int i = 0; i < data.n_variables; i++)
		*parameter_pointer(&codesign->parameters, data.variables[i]) = denormalize_parameter(data.variables[i], u[i]);

	evaluate_codesign(codesign, 1, &codesign->metrics);

	if (objective != NULL)
		*objective = best_objective;

	if (n_evaluations != NULL)
		*n_evaluations = data.n_evaluations;

	printf("Finished %s stage after %d evaluations: objective %.6e, circuit fidelity %.6f\n", codesign_stage_name(stage), data.n_evaluations, best_objective, codesign->metrics.circuit_fidelity);

	return result;
}

// Set up the co-design of a trap for the circuit in a QASM file (stage 0), starting from the given environment and
// parameters (or the defaults, if they are NULL); returns NULL if the circuit cannot be read or the design evaluated
struct Codesign* create_codesign
(
	char *element,
	char *qasm_filename,
	struct CodesignEnvironment *environment,
	struct CodesignParameters *parameters
)
{
	struct Codesign *codesign = calloc(1, sizeof(struct Codesign));

	if (generate_ion_from_preset(element, &codesign->ion) != 0)
	{
		free(codesign);
		return NULL;
	}

	if (environment != NULL)
		codesign->environment = *environment;
	else
		default_codesign_environment(&codesign->environment);

	if (parameters != NULL)
		codesign->parameters = *parameters;
	else
		default_codesign_parameters(&codesign->parameters);

	codesign->primitive_status = -1;

	printf("Stage 0: basic design of the trap, photonics, transport program and waveforms\n");

	if (summarize_circuit(codesign, qasm_filename) != 0)
	{
		free_codesign(codesign);
		return NULL;
	}

	evaluate_codesign(codesign, 1, &codesign->metrics);

	record_codesign(codesign, 0, BASIC_DESIGN_STAGE, 1, codesign->metrics.objective);

	printf("Basic design: circuit fidelity %.6f over %.3f ms\n", codesign->metrics.circuit_fidelity, codesign->metrics.circuit_duration * 1E3);

	return codesign;
}

// Run the co-design flow: the first pass optimizes stages 1-4 with their own objectives and computes the metrics of
// the design (stage 5), and every later pass repeats stages 1-5 with the QC-aware objective; every stage takes at most
// max_evaluations evaluations of the design
int run_codesign
(
	struct Codesign *codesign,
	int n_passes,
	int max_evaluations
)
{
	// Room for the whole history up front, so that readers polling it from another thread never see it move
	int max_records = codesign->n_records + n_passes * (N_CODESIGN_STAGES - 1);
	if (max_records > codesign->max_records)
	{
		codesign->max_records = max_records;
		codesign->records = realloc(codesign->records, max_records * sizeof(struct CodesignRecord));
	}

	for (int pass = 0; pass < n_passes; pass++)
	{
		int qc_aware = (pass > 0);

		printf("Pass %d of the co-design flow (%s objectives)\n", pass, qc_aware ? "QC-aware" : "local");

		for (enum CodesignStage stage = TRAP_STAGE; stage <= WAVEFORM_STAGE; stage++)
		{
			double objective;
			int n_evaluations;
			optimize_codesign_stage(codesign, stage, qc_aware, max_evaluations, &objective, &n_evaluations);

			record_codesign(codesign, pass, stage, n_evaluations, objective);
		}

		evaluate_codesign(codesign, 1, &codesign->metrics);
		record_codesign(codesign, pass, METRICS_STAGE, 1, codesign->metrics.objective);

		struct QCMetrics *metrics = &codesign->metrics;
		printf("Metrics after pass %d:\n", pass);
		printf("  Radial frequency 2 pi x %.3f MHz, Mathieu q %.3f, trap depth %.3f eV, heating rate %.1f quanta/s\n", metrics->radial_frequency / (2 * PI * 1E6), metrics->mathieu_q, metrics->trap_depth, metrics->heating_rate);
		printf("  Rabi frequency 2 pi x %.1f kHz, Lamb-Dicke parameter %.4f, crosstalk %.2e, detection fidelity %.6f\n", metrics->rabi_frequency / (2 * PI * 1E3), metrics->lamb_dicke_parameter, metrics->crosstalk, metrics->detection_fidelity);
		printf("  Shuttle excitation %.4f quanta, position error %.4f µm, peak voltage %.3f V\n", metrics->shuttle_excitation, metrics->position_error, metrics->peak_voltage);
		printf("  Gate times %.2f µs and %.2f µs, fidelities %.6f and %.6f\n", metrics->single_qubit_time * 1E6, metrics->two_qubit_time * 1E6, metrics->single_qubit_fidelity, metrics->two_qubit_fidelity);
		printf("  Circuit duration %.3f ms, circuit fidelity %.6f\n", metrics->circuit_duration * 1E3, metrics->circuit_fidelity);
	}

	return 0;
}

// Free a co-design, including its trap and history
void free_codesign
(
	struct Codesign *codesign
)
{
	if (codesign == NULL)
		return;

	free(codesign->layer_distances);
	free(codesign->layer_single_qubit_gates);
	free(codesign->layer_two_qubit_gates);
	free(codesign->layer_measurements);
	free(codesign->records);
	free_trap(&codesign->trap);
	free(codesign);
}
