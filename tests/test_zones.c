#include "test_zones.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"
#include "zones.h"
#include "traps.h"
#include "io.h"
#include "ions.h"
#include "photonics.h"
#include "electrodynamics.h"
#include "analysis.h"
#include "circuits.h"
#include "transport.h"
#include "codesign.h"
#include "test_utils.h"

int test_all_zones
(
	double atol
)
{
	printf("- Testing zones, trap presets and zone layouts...\n");

	int result = 0;

	result += test_parse_zone_layout(atol);
	result += test_generate_trap_from_layout(atol);
	result += test_species_photonics(atol);
	result += test_junction_traps(atol);
	result += test_point_trap(atol);
	result += test_rf_null_path(atol);
	result += test_zone_trap_file(atol);
	result += test_parallel_schedule(atol);
	result += test_codesign_layout(atol);
	result += test_required_nsph_x(atol);

	return result;
}

int test_parse_zone_layout
(
	double atol
)
{
	int n_zones;
	enum ZoneType *zone_types;
	int *zone_capacities;

	// Case and spaces are ignored, capacities follow ":" and counts follow "x" or "*"
	int status = parse_zone_layout("Load+Storage, empty, gate:3 x2 , measure*2", 4, &n_zones, &zone_types, &zone_capacities);

	enum ZoneType expected_types[6] = { LOAD_ZONE | STORAGE_ZONE, EMPTY_ZONE, GATE_ZONE, GATE_ZONE, MEASURE_ZONE, MEASURE_ZONE };
	int expected_capacities[6] = { 4, 1, 3, 3, 4, 4 };

	int n_wrong = status + abs(n_zones - 6);
	for (int z = 0; z < MIN(n_zones, 6); z++)
		n_wrong += (zone_types[z] != expected_types[z]) + (zone_capacities[z] != expected_capacities[z]);

	free(zone_types);
	free(zone_capacities);

	// Unknown types, empty entries, and counts or capacities below one are rejected
	char *invalid_layouts[5] = { "gate, warp", "gate x0", "gate:0", "", "load,,gate" };
	for (int i = 0; i < 5; i++)
	{
		n_wrong += (parse_zone_layout(invalid_layouts[i], 4, &n_zones, &zone_types, &zone_capacities) == 0);
		n_wrong += (zone_types != NULL) + (zone_capacities != NULL);
	}

	// Zone types are written back the way they are parsed
	char text[64];
	format_zone_types(LOAD_ZONE | STORAGE_ZONE, text, sizeof(text));
	n_wrong += (strcmp(text, "load+storage") != 0);

	return check_error("parse_zone_layout parses layouts", n_wrong, 0.0);
}

int test_generate_trap_from_layout
(
	double atol
)
{
	struct Trap trap;
	int status = generate_trap_from_layout("load, storage, gate x2, measure", 0, 80.0, 50.0, "40Ca+", 1, &trap);

	int result = check_error("generate_trap_from_layout succeeds", status, 0.0);

	// The zones are spaced evenly along the trap axis and centered on it, and the segments reach beyond the outermost
	// zones
	int n_wrong = abs(trap.n_zones - 5);
	for (int z = 0; z < MIN(trap.n_zones, 5); z++)
	{
		struct ZoneNode *zone = &(*trap.zones)[z];
		n_wrong += (fabs(zone->position[0] - (z - 2) * 80.0) > atol) + (fabs(zone->position[2] - ION_HEIGHT) > atol);
	}

	int n_segments = (trap.n_electrodes_dc - 1) / 2;
	n_wrong += (0.5 * n_segments * 50.0 < 160.0 + 50.0);

	result += check_error("trap from layout has its zones", n_wrong, 0.0);

	// The loading zone gets the photoionization and cooling beams, each gate zone the qubit and quench beams, and the
	// measurement zone the detection and repump beams and a detector
	int expected_counts[N_PHOTONICS_BEAMS] = { 2, 2, 2, 2, 1, 1 };
	int counts[N_PHOTONICS_BEAMS] = { 0 };
	for (int g = 0; g < trap.n_grating_couplers; g++)
		if ((*trap.grating_couplers)[g].beam >= 0 && (*trap.grating_couplers)[g].beam < N_PHOTONICS_BEAMS)
			counts[(*trap.grating_couplers)[g].beam]++;

	n_wrong = abs(trap.n_grating_couplers - 10) + abs(trap.n_detectors - 1);
	for (int b = 0; b < N_PHOTONICS_BEAMS; b++)
		n_wrong += abs(counts[b] - expected_counts[b]);

	if (trap.n_detectors > 0)
		n_wrong += ((*trap.detectors)[0].zone != 4);

	result += check_error("trap from layout has photonics for every zone", n_wrong, 0.0);

	// Every beam is focused on the zone it addresses, so its intensity peaks there
	double error = 0.0;
	for (int g = 0; g < trap.n_grating_couplers; g++)
	{
		struct GratingCoupler *grating = &(*trap.grating_couplers)[g];
		double *target = (*trap.zones)[grating->zone].position;

		double intensity;
		evaluate_grating_beam(grating, target, &intensity, NULL);

		for (int k = 0; k < 2; k++)
		{
			double displaced[3] = { target[0], target[1], target[2] };
			displaced[k] += 2.0;

			double displaced_intensity;
			evaluate_grating_beam(grating, displaced, &displaced_intensity, NULL);

			if (displaced_intensity >= intensity)
				error += 1.0;
		}
	}

	result += check_error("trap from layout focuses every beam on its zone", error, 0.0);

	free_trap(&trap);

	return result;
}

int test_species_photonics
(
	double atol
)
{
	// A zone which loads, gates and measures needs every beam a species uses: all six for the species with an optical
	// qubit, and only the qubit (Raman), detection, repump and photoionization beams for 9Be+
	char *elements[4] = { "40Ca+", "88Sr+", "171Yb+", "9Be+" };
	int expected_gratings[4] = { 6, 6, 6, 4 };

	int n_wrong = 0;
	for (int i = 0; i < 4; i++)
	{
		struct Trap trap;
		memset(&trap, 0, sizeof(struct Trap));

		double position[3] = { 0.0, 0.0, ION_HEIGHT };
		allocate_zones(&trap, 1);
		set_trap_zone(&trap, 0, LOAD_ZONE | GATE_ZONE | MEASURE_ZONE, 1, position);

		n_wrong += (add_zone_photonics(&trap, elements[i]) != 0);
		n_wrong += abs(trap.n_grating_couplers - expected_gratings[i]) + abs(trap.n_detectors - 1);

		for (int g = 0; g < trap.n_grating_couplers; g++)
		{
			struct GratingCoupler *grating = &(*trap.grating_couplers)[g];

			struct Waveguide waveguide;
			double power;
			n_wrong += (get_photonics_beam(elements[i], grating->beam, &waveguide, &power) != 0);
			n_wrong += (fabs(grating->waveguide.wavelength - waveguide.wavelength) > atol);
		}

		free_trap(&trap);
	}

	struct Trap trap;
	memset(&trap, 0, sizeof(struct Trap));
	n_wrong += (add_zone_photonics(&trap, "12C+") == 0);
	free_trap(&trap);

	return check_error("add_zone_photonics designs every beam of every species", n_wrong, 0.0);
}

int test_junction_traps
(
	double atol
)
{
	enum TrapPreset presets[3] = { X_JUNCTION_TRAP, Y_JUNCTION_TRAP, T_JUNCTION_TRAP };
	int n_arms[3] = { 4, 3, 3 };
	double arm_angles[3][4] = { { 0.0, 0.5 * PI, PI, 1.5 * PI }, { 0.0, 2 * PI / 3, 4 * PI / 3 }, { 0.0, 0.5 * PI, PI } };

	int n_wrong = 0;
	double error = 0.0;
	for (int p = 0; p < 3; p++)
	{
		struct Trap trap;
		n_wrong += (generate_trap_from_preset(presets[p], 2, 40.0, &trap) != 0);

		// One RF electrode for each gap between arms, a central DC electrode, and the segments of every arm, with a
		// junction zone at the center
		n_wrong += abs(trap.n_electrodes_rf - n_arms[p]) + abs(trap.n_electrodes_dc - (1 + 2 * n_arms[p] * 2));
		n_wrong += (trap.n_zones != 1) || ((*trap.zones)[0].zone_type != JUNCTION_ZONE);

		solve_trap_electrodynamics(&trap, ANALYTICAL, NULL);

		// Away from the junction, the RF null runs along the axis of every arm at about the height of the ions
		for (int a = 0; a < n_arms[p]; a++)
		{
			double direction[3] = { cos(arm_angles[p][a]), sin(arm_angles[p][a]), 0.0 };
			double position[3] = { 150.0 * direction[0], 150.0 * direction[1], ION_HEIGHT };

			n_wrong += (find_rf_null_across(&trap, &position, direction, 100) != 0);

			double off_axis = fabs(position[1] * direction[0] - position[0] * direction[1]);
			error = MAX(error, MAX(fabs(position[2] - ION_HEIGHT), off_axis));
		}

		free_trap(&trap);
	}

	// Neighboring arms must be at most 180 degrees apart
	struct Trap trap;
	double bent_arms[2] = { 0.0, 0.5 * PI };
	n_wrong += (generate_junction_trap(2, bent_arms, 2, 40.0, &trap) == 0);
	free_trap(&trap);

	int result = check_error("junction presets have their electrodes", n_wrong, 0.0);
	result += check_error("junction presets have RF nulls along their arms (µm)", error, 3.0);

	return result;
}

int test_point_trap
(
	double atol
)
{
	struct Trap trap;
	int n_wrong = (generate_trap_from_preset(POINT_TRAP, 4, 40.0, &trap) != 0);
	n_wrong += abs(trap.n_electrodes_rf - 1) + abs(trap.n_electrodes_dc - 5) + abs(trap.n_zones - 1);

	solve_trap_electrodynamics(&trap, ANALYTICAL, NULL);

	// The ring holds the ion above its center in all three directions with the RF alone
	struct IonParameters ion;
	generate_ion_from_preset("40Ca+", &ion);

	double position[3] = { 0.0, 0.0, ION_HEIGHT };
	n_wrong += (find_rf_null(&trap, &position, 100) != 0);

	double voltages[trap.n_electrodes];
	memset(voltages, 0, sizeof(voltages));

	double omega[3];
	double axes[3][3];
	compute_normal_modes(&trap, ion_mass(&ion), ion_charge(&ion), voltages, position, &omega, &axes);

	for (int k = 0; k < 3; k++)
		n_wrong += !(omega[k] > 0.0);

	int result = check_error("point trap confines in every direction", n_wrong, 0.0);
	result += check_error("point trap has its RF null at the height of the ions (µm)", fabs(position[2] - ION_HEIGHT) + hypot(position[0], position[1]), 0.5);

	free_trap(&trap);

	return result;
}

int test_rf_null_path
(
	double atol
)
{
	struct IonParameters ion;
	generate_ion_from_preset("40Ca+", &ion);
	double m = ion_mass(&ion);
	double q = ion_charge(&ion);

	// Along the axis of a linear trap the RF null is a line, so the path is symmetric and has no barrier
	struct Trap linear_trap;
	generate_trap_from_preset(LINEAR_SURFACE_TRAP, 7, 50.0, &linear_trap);
	solve_trap_electrodynamics(&linear_trap, ANALYTICAL, NULL);

	double axis_waypoints[2][3] = { { -100.0, 0.0, ION_HEIGHT }, { 100.0, 0.0, ION_HEIGHT } };
	double positions[41][3];
	double energies[41];
	double linear_barrier = compute_rf_null_path(&linear_trap, m, q, 2, axis_waypoints, 41, positions, energies);

	int result = check_error("compute_rf_null_path along a linear trap (eV)", fabs(linear_barrier) + fabs(positions[20][2] - ION_HEIGHT) * 1E-6, 1E-6);

	free_trap(&linear_trap);

	// Turning through an X junction, the pseudopotential rises above that of the arms
	struct Trap junction_trap;
	generate_trap_from_preset(X_JUNCTION_TRAP, 2, 40.0, &junction_trap);
	solve_trap_electrodynamics(&junction_trap, ANALYTICAL, NULL);

	double turn_waypoints[3][3] = { { 150.0, 0.0, ION_HEIGHT }, { 0.0, 0.0, ION_HEIGHT }, { 0.0, 150.0, ION_HEIGHT } };
	double junction_barrier = compute_rf_null_path(&junction_trap, m, q, 3, turn_waypoints, 41, NULL, NULL);

	result += check_error("compute_rf_null_path finds the barrier of an X junction", (junction_barrier > 1E-3) ? 0.0 : 1.0, 0.0);

	free_trap(&junction_trap);

	return result;
}

int test_zone_trap_file
(
	double atol
)
{
	struct Trap trap;
	generate_trap_from_layout("load+storage, gate, measure", 3, 100.0, 50.0, "40Ca+", 1, &trap);

	char *filename = "zones_test.trap";
	int n_wrong = write_trap_to_file(filename, &trap);

	struct Trap loaded;
	n_wrong += generate_trap_from_file(filename, &loaded);
	remove(filename);

	if (n_wrong == 0)
	{
		n_wrong += abs(loaded.n_zones - trap.n_zones) + abs(loaded.n_grating_couplers - trap.n_grating_couplers) + abs(loaded.n_detectors - trap.n_detectors);

		for (int z = 0; z < MIN(loaded.n_zones, trap.n_zones); z++)
		{
			struct ZoneNode *a = &(*trap.zones)[z];
			struct ZoneNode *b = &(*loaded.zones)[z];

			n_wrong += (a->zone_type != b->zone_type) + (a->zone_capacity != b->zone_capacity);
			for (int k = 0; k < 3; k++)
				n_wrong += (fabs(a->position[k] - b->position[k]) > atol);
		}

		for (int g = 0; g < MIN(loaded.n_grating_couplers, trap.n_grating_couplers); g++)
			n_wrong += ((*trap.grating_couplers)[g].beam != (*loaded.grating_couplers)[g].beam) + ((*trap.grating_couplers)[g].zone != (*loaded.grating_couplers)[g].zone);

		for (int d = 0; d < MIN(loaded.n_detectors, trap.n_detectors); d++)
			n_wrong += ((*trap.detectors)[d].zone != (*loaded.detectors)[d].zone);
	}

	free_trap(&loaded);
	free_trap(&trap);

	return check_error("trap files keep zones and the beams and zones of the photonics", n_wrong, 0.0);
}

int test_parallel_schedule
(
	double atol
)
{
	// With two gate zones, the two gates of the second layer (on qubits 1 and 2, and on qubits 3 and 4) each run in a
	// gate zone of their own
	enum ZoneType zone_types[4] = { LOAD_ZONE | STORAGE_ZONE, GATE_ZONE, GATE_ZONE, MEASURE_ZONE };
	int zone_capacities[4] = { 4, 2, 2, 4 };

	struct TrapGraph trap_graph;
	generate_linear_trap_graph(4, zone_types, zone_capacities, -150.0, 150.0, ION_HEIGHT, &trap_graph);

	struct CircuitGraph *circuit_graph = parse_qasm("quantumcircuits/test_0000.qasm");
	struct TransportProgram *transport_program = calloc(1, sizeof(struct TransportProgram));

	int n_wrong = (generate_transport_schedule(&trap_graph, circuit_graph, transport_program) == NULL);

	if (n_wrong == 0)
	{
		double x_first = transport_program->destinations[4][0];
		double x_second = transport_program->destinations[5][0];

		n_wrong += (circuit_graph->layer_idxs[4] != 2) + (circuit_graph->layer_idxs[5] != 2);
		n_wrong += (fabs(x_first - x_second) < 1.0);
		n_wrong += (fabs(fabs(x_first) - 50.0) > atol) + (fabs(fabs(x_second) - 50.0) > atol);

		// The measurement of every qubit happens in the measurement zone
		n_wrong += (fabs(transport_program->destinations[10][0] - 150.0) > atol);
	}

	free_transport_program(transport_program);
	free_circuit_graph(circuit_graph);
	free_trap_graph(&trap_graph);

	return check_error("generate_transport_schedule runs gates in several gate zones at once", n_wrong, 0.0);
}

int test_codesign_layout
(
	double atol
)
{
	// Two gate zones let the gates of a layer run at once; the zones are brought closer together to fit within the
	// spherical harmonics expansions
	struct Codesign *codesign = create_codesign("40Ca+", "quantumcircuits/test_0000.qasm", "load+storage, empty, gate, empty, gate, empty, measure", NULL, NULL);

	if (codesign == NULL)
		return check_error("create_codesign accepts a layout with two gate zones", HUGE_VAL, 0.0);

	int n_wrong = abs(codesign->n_zones - 7) + abs(codesign->gate_zone - 2) + abs(codesign->measure_zone - 6) + abs(codesign->trap.n_zones - 7);

	int most_gate_zones = 0;
	for (int layer = 0; layer < codesign->n_layers; layer++)
		most_gate_zones = MAX(most_gate_zones, codesign->layer_gate_zones[layer]);
	n_wrong += (most_gate_zones != 2);

	int n_qubit_gratings = 0;
	for (int g = 0; g < codesign->trap.n_grating_couplers; g++)
		n_qubit_gratings += ((*codesign->trap.grating_couplers)[g].beam == QUBIT_BEAM);
	n_wrong += (n_qubit_gratings != 2);

	double outermost = 0.5 * (codesign->n_zones - 1) * codesign->parameters.zone_spacing;
	n_wrong += (outermost > MIN(-SPH_CENTER_X(0), SPH_CENTER_X(NSPH_X - 1)) + atol);
	n_wrong += !(codesign->metrics.circuit_fidelity > 0.0 && codesign->metrics.circuit_fidelity <= 1.0);

	free_codesign(codesign);

	int result = check_error("create_codesign uses a layout with two gate zones", n_wrong, 0.0);

	// Layouts without a gate zone cannot run the circuit, and layouts which do not fit even at the smallest zone
	// spacing are rejected
	n_wrong = (create_codesign("40Ca+", "quantumcircuits/test_0000.qasm", "load+storage, measure", NULL, NULL) != NULL);
	n_wrong += (create_codesign("40Ca+", "quantumcircuits/test_0000.qasm", "load+storage, empty x20, gate, empty x20, measure", NULL, NULL) != NULL);

	result += check_error("create_codesign rejects layouts which cannot run the circuit", n_wrong, 0.0);

	return result;
}

int test_required_nsph_x
(
	double atol
)
{
	// The expansions of this build reach exactly as far as their own number of spheres, and longer traps need more
	int n_wrong = (required_nsph_x(SPH_CENTER_X(0), SPH_CENTER_X(NSPH_X - 1)) != NSPH_X);
	n_wrong += (required_nsph_x(2 * SPH_CENTER_X(0), 2 * SPH_CENTER_X(NSPH_X - 1)) <= NSPH_X);

	return check_error("required_nsph_x matches the expansions of this build", n_wrong, 0.0);
}
