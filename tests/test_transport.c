#include "test_transport.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"
#include "transport.h"
#include "circuits.h"
#include "zones.h"
#include "schedule_check.h"
#include "test_utils.h"
#include <dirent.h>

int test_all_transport
(
	double atol
)
{
	printf("- Testing transport scheduling...\n");

	int result = 0;

	result += test_add_qasm_targets(atol);
	result += test_parse_qasm(atol);
	result += test_compute_circuit_layers(atol);
	result += test_generate_transport_schedule(atol);
	result += test_parse_qasm_rejects_unknown_gates(atol);
	result += test_parse_parameterized_gates(atol);
	result += test_fast_schedule_matches_baseline(atol);
	result += test_schedule_rounds_fit_zones(atol);
	result += test_schedule_trace_times(atol);
	result += test_trap_graph_routes(atol);
	result += test_schedule_through_junction(atol);
	result += test_three_ion_schedule_by_hand(atol);
	result += test_schedule_check_finds_violations(atol);
	result += test_qccd_cost_model_by_hand(atol);
	result += test_lookahead_layers(atol);
	result += test_commuting_layers(atol);
	result += test_gates_take_chain_in_turn(atol);
	result += test_mid_chain_swap_cost(atol);
	result += test_cycle_breaker_in_storage_zones(atol);
	result += test_stay_evicts_ion_needed_last(atol);

	return result;
}

int test_add_qasm_targets
(
	double atol
)
{
	int targets[4];
	int n_targets = 0;

	char operand_1[] = "qreg[2]";
	char operand_2[] = "qreg[2]";
	char operand_3[] = "qreg[9]";
	char operand_4[] = "qreg";

	n_targets = add_qasm_targets(operand_1, 4, targets, n_targets);
	int n_wrong = (n_targets != 1 || targets[0] != 3);

	// Duplicates and out of range qubits are ignored
	n_targets = add_qasm_targets(operand_2, 4, targets, n_targets);
	n_targets = add_qasm_targets(operand_3, 4, targets, n_targets);
	n_wrong += (n_targets != 1);

	// A register without an index refers to every qubit, adding those which are not already targeted in order
	n_targets = add_qasm_targets(operand_4, 4, targets, n_targets);
	int expected_targets[4] = { 3, 1, 2, 4 };
	n_wrong += (n_targets != 4);
	for (int t = 0; t < MIN(n_targets, 4); t++)
		n_wrong += (targets[t] != expected_targets[t]);

	return check_error("add_qasm_targets parses operands", n_wrong, 0);
}

int test_parse_qasm
(
	double atol
)
{
	struct CircuitGraph *circuit_graph = parse_qasm("quantumcircuits/test_0000.qasm");

	if (circuit_graph == NULL)
		return check_error("parse_qasm reads test circuit", HUGE_VAL, 0);

	// h, x, y, z on each qubit, then three layers of pairs of two-qubit gates, then a measurement of every qubit, which
	// measures each qubit on its own
	int expected_n_targets[14] = { 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1 };
	int expected_targets[14][2] = { { 1 }, { 2 }, { 3 }, { 4 }, { 1, 2 }, { 3, 4 }, { 1, 3 }, { 2, 4 }, { 1, 4 }, { 2, 3 }, { 1 }, { 2 }, { 3 }, { 4 } };
	int expected_layers[14] = { 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5 };
	int expected_n_prev_nodes[14] = { 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1 };

	int n_wrong = (circuit_graph->n_ions != 4) + (circuit_graph->n_instruction_nodes != 14);

	for (int i = 0; i < MIN(circuit_graph->n_instruction_nodes, 14); i++)
	{
		struct InstructionNode *node = &circuit_graph->instruction_nodes[i];

		n_wrong += (node->n_target_ions != expected_n_targets[i]);
		n_wrong += (node->operation_type != ((i >= 10) ? DETECT : QUBIT));
		n_wrong += (node->graphindex != i + 1);
		n_wrong += (node->n_prev_nodes != expected_n_prev_nodes[i]);
		n_wrong += (circuit_graph->layer_idxs[i] != expected_layers[i]);

		for (int t = 0; t < MIN(node->n_target_ions, 2); t++)
			n_wrong += (node->target_ions[t] != expected_targets[i][t]);
	}

	// The measurement of qubit 0 follows the CZ on qubits 0 and 3, its last gate
	if (circuit_graph->n_instruction_nodes == 14)
		n_wrong += (circuit_graph->instruction_nodes[10].prev_nodes[0].graphindex != 9);

	// The CZ on qubits 0 and 3 follows the CY on qubits 0 and 2 and the CZ on qubits 1 and 3
	struct InstructionNode *cz = &circuit_graph->instruction_nodes[8];
	int prev_graphindices = 0;
	for (int p = 0; p < cz->n_prev_nodes; p++)
		prev_graphindices += cz->prev_nodes[p].graphindex;
	n_wrong += (prev_graphindices != 7 + 8);

	free_circuit_graph(circuit_graph);

	return check_error("parse_qasm builds circuit graph", n_wrong, 0);
}

int test_compute_circuit_layers
(
	double atol
)
{
	// A diamond-shaped graph listed out of order: node 1 depends on nodes 2 and 3, which both depend on node 4
	struct InstructionNode nodes[4];
	memset(nodes, 0, sizeof(nodes));
	for (int i = 0; i < 4; i++)
		nodes[i].graphindex = i + 1;

	struct InstructionNode prev_1[2] = { nodes[1], nodes[2] };
	struct InstructionNode prev_2[1] = { nodes[3] };
	struct InstructionNode prev_3[1] = { nodes[3] };

	nodes[0].n_prev_nodes = 2;
	nodes[0].prev_nodes = prev_1;
	nodes[1].n_prev_nodes = 1;
	nodes[1].prev_nodes = prev_2;
	nodes[2].n_prev_nodes = 1;
	nodes[2].prev_nodes = prev_3;

	int layer_idxs[4];
	struct CircuitGraph circuit_graph = { 1, 4, nodes, layer_idxs };
	compute_circuit_layers(&circuit_graph);

	int expected_layers[4] = { 3, 2, 2, 1 };
	int n_wrong = 0;
	for (int i = 0; i < 4; i++)
		n_wrong += (layer_idxs[i] != expected_layers[i]);

	return check_error("compute_circuit_layers orders dependencies", n_wrong, 0);
}

int test_generate_transport_schedule
(
	double atol
)
{
	// Load, empty, storage, empty, gate, empty and measure zones along the axis, as in simple_transport_schedule
	enum ZoneType zone_types[7] = { LOAD_ZONE, EMPTY_ZONE, STORAGE_ZONE, EMPTY_ZONE, GATE_ZONE, EMPTY_ZONE, MEASURE_ZONE };
	int zone_capacities[7] = { 5, 3, 10, 3, 2, 3, 4 };

	struct TrapGraph trap_graph;
	generate_linear_trap_graph(7, zone_types, zone_capacities, -150.0, 150.0, ION_HEIGHT, &trap_graph);

	struct CircuitGraph *circuit_graph = parse_qasm("quantumcircuits/test_0000.qasm");

	// Times only count distance here
	struct TransportCostModel cost_model = distance_transport_cost_model();
	// Rounds, whose ions go back into storage after each layer (SCHEDULE_STAY leaves them in place)
	struct ScheduleOptions options = { SCHEDULE_ROUNDS, DEFAULT_LOOKAHEAD_LAYERS, 0, &cost_model };

	struct TransportProgram *transport_program = malloc(sizeof(struct TransportProgram));
	struct ScheduleTrace trace = { 0 };
	transport_program = generate_transport_schedule_options(&trap_graph, circuit_graph, transport_program, &options, &trace);

	if (transport_program == NULL)
	{
		free_schedule_trace(&trace);
		free_circuit_graph(circuit_graph);
		free_trap_graph(&trap_graph);
		return check_error("generate_transport_schedule succeeds", HUGE_VAL, 0);
	}

	int n_wrong = (transport_program->n_ions != 4) + (transport_program->n_transports != 14);

	// Ions start in the load zone, and are transported to the gate zone for gates and to the measure zone for detection
	for (int i = 0; i < 4; i++)
		n_wrong += (fabs(transport_program->initial_positions[i][0] + 150.0) > atol);

	for (int t = 0; t < MIN(transport_program->n_transports, 14); t++)
	{
		double x_expected = (t >= 10) ? 150.0 : 50.0;

		n_wrong += (fabs(transport_program->destinations[t][0] - x_expected) > atol);
		n_wrong += (fabs(transport_program->destinations[t][2] - ION_HEIGHT) > atol);
	}

	// The gate zone only holds two ions, so each layer runs in rounds which fit in it: the four single-qubit gates of
	// the first layer in two rounds, each pair of two-qubit gates in two more, and the measurements of every qubit in one
	int expected_rounds[14] = { 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9 };
	n_wrong += (trace.n_rounds != 9);
	for (int t = 0; t < MIN(transport_program->n_transports, 14); t++)
		n_wrong += (trace.transport_rounds[t] != expected_rounds[t]);

	// Transports in the same round arrive together (no round holds two gates on several ions in one zone), and later
	// rounds arrive later
	for (int t = 1; t < MIN(transport_program->n_transports, 14); t++)
	{
		int same_round = trace.transport_rounds[t] == trace.transport_rounds[t-1];

		if (same_round)
			n_wrong += (fabs(transport_program->arrival_times[t] - transport_program->arrival_times[t-1]) > atol);
		else
			n_wrong += (transport_program->arrival_times[t] <= transport_program->arrival_times[t-1]);
	}

	// The first round starts after the 200 µm transport from the load zone to the gate zone, with the second ion one
	// 50 µm segment behind the first
	n_wrong += (fabs(transport_program->arrival_times[0] - 250.0 / TRANSPORT_SPEED_LIMIT) > atol);

	free_schedule_trace(&trace);
	free_transport_program(transport_program);
	free_circuit_graph(circuit_graph);
	free_trap_graph(&trap_graph);

	return check_error("generate_transport_schedule fills transport program", n_wrong, 0);
}

int test_parse_parameterized_gates
(
	double atol
)
{
	// Rotations, phases and two-qubit interactions are read whatever their parameters
	char *filename = "parameterized_test.qasm";
	FILE *fp = fopen(filename, "w");
	fprintf(fp, "qubit[3] qreg;\nbit[3] creg;\nrz(0.5) qreg[0];\nt qreg[1];\ncp(1.570796) qreg[0], qreg[1];\nrzz(0.2) qreg[1], qreg[2];\nswap qreg[0], qreg[2];\nmeasure qreg -> creg;\n");
	fclose(fp);

	struct CircuitGraph *circuit_graph = parse_qasm(filename);
	remove(filename);

	int n_wrong = (circuit_graph == NULL);
	if (circuit_graph != NULL)
	{
		// The measurement of the register measures each of its three qubits on its own
		int expected_targets[8] = { 1, 1, 2, 2, 2, 1, 1, 1 };
		n_wrong += (circuit_graph->n_instruction_nodes != 8);
		for (int i = 0; i < MIN(circuit_graph->n_instruction_nodes, 8); i++)
			n_wrong += (circuit_graph->instruction_nodes[i].n_target_ions != expected_targets[i]);

		free_circuit_graph(circuit_graph);
	}

	return check_error("parse_qasm reads parameterized gates", n_wrong, 0);
}

int test_fast_schedule_matches_baseline
(
	double atol
)
{
	enum ZoneType zone_types[7] = { LOAD_ZONE, EMPTY_ZONE, STORAGE_ZONE, EMPTY_ZONE, GATE_ZONE, EMPTY_ZONE, MEASURE_ZONE };
	int zone_capacities[7] = { 5, 3, 10, 3, 2, 3, 4 };

	struct CircuitGraph *circuit_graph = parse_qasm("quantumcircuits/test_0000.qasm");

	struct TransportProgram *programs[2];
	struct ScheduleTrace traces[2] = { { 0 }, { 0 } };
	enum ScheduleStrategy strategies[2] = { SCHEDULE_BASELINE, SCHEDULE_FAST };

	for (int s = 0; s < 2; s++)
	{
		struct TrapGraph trap_graph;
		generate_linear_trap_graph(7, zone_types, zone_capacities, -150.0, 150.0, ION_HEIGHT, &trap_graph);

		programs[s] = calloc(1, sizeof(struct TransportProgram));
		if (generate_transport_schedule_strategy(&trap_graph, circuit_graph, programs[s], strategies[s], &traces[s]) == NULL)
		{
			free_transport_program(programs[s]);
			programs[s] = NULL;
		}

		free_trap_graph(&trap_graph);
	}

	int n_wrong = (programs[0] == NULL) + (programs[1] == NULL);
	if (n_wrong == 0)
	{
		int n = programs[0]->n_transports;

		n_wrong += (traces[0].n_moves != traces[1].n_moves);
		n_wrong += (memcmp(programs[0]->arrival_times, programs[1]->arrival_times, n * sizeof(double)) != 0);
		n_wrong += (memcmp(programs[0]->destinations, programs[1]->destinations, n * sizeof(double[3])) != 0);

		if (traces[0].n_moves == traces[1].n_moves)
			n_wrong += (memcmp(traces[0].moves, traces[1].moves, traces[0].n_moves * sizeof(struct IonMove)) != 0);
	}

	for (int s = 0; s < 2; s++)
	{
		free_transport_program(programs[s]);
		free_schedule_trace(&traces[s]);
	}
	free_circuit_graph(circuit_graph);

	return check_error("SCHEDULE_FAST makes the same schedule as SCHEDULE_BASELINE", n_wrong, 0);
}

// Follow the ions of a schedule which starts with every ion in the first zone, checking each round once all of its
// transports are done: every operation must find its ions at its destination, and no zone may be over capacity. Returns
// the number of problems found
static int count_invalid
(
	struct TrapGraph *trap_graph,
	struct CircuitGraph *circuit_graph,
	struct TransportProgram *program,
	struct ScheduleTrace *trace,
	double atol
)
{
	int n_wrong = 0;
	int n_ions = circuit_graph->n_ions;
	int *ion_zones = calloc(n_ions, sizeof(int));
	int *occupancy = calloc(trap_graph->n_zones, sizeof(int));
	occupancy[0] = n_ions;

	for (int m = 0; m < trace->n_moves; m++)
	{
		struct IonMove *move = &trace->moves[m];
		ion_zones[move->ion - 1] = move->to_zone;
		occupancy[move->from_zone]--;
		occupancy[move->to_zone]++;

		int last = (move->transport >= 0) && (m + 1 == trace->n_moves || trace->moves[m + 1].transport < 0 ||
			trace->transport_rounds[trace->moves[m + 1].transport] != trace->transport_rounds[move->transport]);
		if (!last)
			continue;

		int round = trace->transport_rounds[move->transport];
		for (int t = 0; t < program->n_transports; t++)
		{
			if (trace->transport_rounds[t] != round)
				continue;

			struct InstructionNode *node = &circuit_graph->instruction_nodes[t];
			for (int i = 0; i < node->n_target_ions; i++)
				n_wrong += (fabs(trap_graph->zone_nodes[ion_zones[node->target_ions[i] - 1]].position[0] - program->destinations[t][0]) > atol);
		}

		for (int z = 0; z < trap_graph->n_zones; z++)
			n_wrong += (occupancy[z] > trap_graph->zone_nodes[z].zone_capacity);
	}

	free(occupancy);
	free(ion_zones);

	return n_wrong;
}

int test_schedule_rounds_fit_zones
(
	double atol
)
{
	// The first layer has four single-qubit gates but the gate zone only holds two ions, so the strategies which split
	// layers into rounds run it in two rounds, with every operation finding its ions in a zone of its type
	enum ZoneType zone_types[7] = { LOAD_ZONE, EMPTY_ZONE, STORAGE_ZONE, EMPTY_ZONE, GATE_ZONE, EMPTY_ZONE, MEASURE_ZONE };
	int zone_capacities[7] = { 5, 3, 10, 3, 2, 3, 4 };

	struct CircuitGraph *circuit_graph = parse_qasm("quantumcircuits/test_0000.qasm");

	enum ScheduleStrategy strategies[3] = { SCHEDULE_ROUNDS, SCHEDULE_LOOKAHEAD, SCHEDULE_LAYER_ASSIGN };
	int n_wrong = 0;

	for (int s = 0; s < 3; s++)
	{
		struct TrapGraph trap_graph;
		generate_linear_trap_graph(7, zone_types, zone_capacities, -150.0, 150.0, ION_HEIGHT, &trap_graph);

		struct TransportProgram *program = calloc(1, sizeof(struct TransportProgram));
		struct ScheduleTrace trace = { 0 };

		if (generate_transport_schedule_strategy(&trap_graph, circuit_graph, program, strategies[s], &trace) == NULL)
		{
			n_wrong++;
		}
		else
		{
			n_wrong += (trace.transport_rounds[0] != 1) + (trace.transport_rounds[3] != 2);

			n_wrong += count_invalid(&trap_graph, circuit_graph, program, &trace, atol);
		}

		free_schedule_trace(&trace);
		free_transport_program(program);
		free_trap_graph(&trap_graph);
	}

	free_circuit_graph(circuit_graph);

	return check_error("schedules split layers into rounds which fit in the zones", n_wrong, 0);
}

int test_parse_qasm_rejects_unknown_gates
(
	double atol
)
{
	// A gate which the parser does not know would be left out of the schedule, so the circuit is rejected instead
	char *filename = "unknown_gate_test.qasm";
	FILE *fp = fopen(filename, "w");
	fprintf(fp, "// Comments, headers and declarations are skipped\nOPENQASM 3;\nqubit[2] qreg;\nbit[2] creg;\nh qreg[0];\nu3(0.1) qreg[0];\ncx qreg[0], qreg[1];\nmeasure qreg -> creg;\n");
	fclose(fp);

	struct CircuitGraph *circuit_graph = parse_qasm(filename);
	remove(filename);

	int n_wrong = (circuit_graph != NULL);
	if (circuit_graph != NULL)
		free_circuit_graph(circuit_graph);

	return check_error("parse_qasm rejects circuits with unknown gates", n_wrong, 0);
}

int test_schedule_trace_times
(
	double atol
)
{
	enum ZoneType zone_types[7] = { LOAD_ZONE, EMPTY_ZONE, STORAGE_ZONE, EMPTY_ZONE, GATE_ZONE, EMPTY_ZONE, MEASURE_ZONE };
	int zone_capacities[7] = { 5, 3, 10, 3, 2, 3, 4 };

	// Zones 50 µm apart, from -150 µm to 150 µm
	struct TrapGraph trap_graph;
	generate_linear_trap_graph(7, zone_types, zone_capacities, -150.0, 150.0, ION_HEIGHT, &trap_graph);

	struct CircuitGraph *circuit_graph = parse_qasm("quantumcircuits/test_0000.qasm");

	// Times only count distance here
	struct TransportCostModel cost_model = distance_transport_cost_model();
	struct ScheduleOptions options = { SCHEDULE_ROUNDS, 1, 0, &cost_model };

	struct TransportProgram *transport_program = malloc(sizeof(struct TransportProgram));
	struct ScheduleTrace trace = { 0 };
	if (generate_transport_schedule_options(&trap_graph, circuit_graph, transport_program, &options, &trace) == NULL)
	{
		free(transport_program);
		free_schedule_trace(&trace);
		free_circuit_graph(circuit_graph);
		free_trap_graph(&trap_graph);
		return check_error("time_schedule_trace times every move", HUGE_VAL, 0);
	}

	double eps = 1E-12;
	double segment = 50.0 / TRANSPORT_SPEED_LIMIT;
	int n_wrong = 0;
	int n_releases = 0;

	for (int m = 0; m < trace.n_moves; m++)
	{
		struct IonMove *move = &trace.moves[m];

		// A move takes at least as long as its distance at the speed limit
		double distance = fabs(trap_graph.zone_nodes[move->to_zone].position[0] - trap_graph.zone_nodes[move->from_zone].position[0]);
		n_wrong += (move->arrive - move->depart < distance / TRANSPORT_SPEED_LIMIT - eps);

		// An ion makes one move at a time
		for (int p = m - 1; p >= 0; p--)
			if (trace.moves[p].ion == move->ion)
			{
				n_wrong += (move->depart < trace.moves[p].arrive - eps);
				break;
			}

		// Ions arrive by the time their operation starts
		if (move->transport >= 0)
			n_wrong += (move->arrive > transport_program->arrival_times[move->transport] + eps);
		else
			n_releases++;
	}

	n_wrong += (n_releases == 0);

	// Every step of a move lies within it, and every move along a segment takes a segment's length at the speed limit
	int *n_swaps = calloc(trace.n_moves, sizeof(int));
	for (int k = 0; k < trace.n_steps; k++)
	{
		struct TransportStep *step = &trace.steps[k];
		if (step->type == STEP_OPERATION)
			continue;

		struct IonMove *move = &trace.moves[step->index];
		n_wrong += (step->ion != move->ion) + (step->start < move->depart - eps) + (step->end > move->arrive + eps);

		if (step->type == STEP_MOVE)
			n_wrong += (abs(step->to_zone - step->zone) != 1) + (fabs(step->end - step->start - segment) > eps);

		if (step->type == STEP_ION_SWAP)
			n_swaps[step->index]++;
	}

	// Worked out by hand: in the first round, ions 1 and 2 leave the chain in the load zone (zone 0) for the gate zone
	// (zone 4), ion 2 one segment behind ion 1, so the operations start at 250 µs and end at 350 µs. In the second round,
	// ions 1 and 2 go back into storage (zone 2): ion 2 merged into the front of the chain in the gate zone, which faces
	// storage, so it goes first and arrives at 450 µs, and ion 1 follows a segment behind, at 500 µs. Ions 3 and 4 leave
	// the load zone at 350 µs, and have to pass through the chain in storage: each merges into its front, swaps with
	// ions 2 and 1 to its back, splits off and goes on, so ion 3 waits in zone 1 until storage is free at 500 µs and
	// arrives at 650 µs, and ion 4 arrives one segment later, at 700 µs
	double expected_departures[6] = { 0.0, 0.0, 350.0, 350.0, 350.0, 350.0 };
	double expected_arrivals[6] = { 200.0, 250.0, 500.0, 450.0, 650.0, 700.0 };
	int expected_ions[6] = { 1, 2, 1, 2, 3, 4 };
	int expected_swaps[6] = { 0, 0, 0, 0, 2, 2 };
	for (int m = 0; m < MIN(trace.n_moves, 6); m++)
	{
		n_wrong += (trace.moves[m].ion != expected_ions[m]);
		n_wrong += (fabs(trace.moves[m].depart - expected_departures[m] / TRANSPORT_SPEED_LIMIT) > eps);
		n_wrong += (fabs(trace.moves[m].arrive - expected_arrivals[m] / TRANSPORT_SPEED_LIMIT) > eps);
		n_wrong += (n_swaps[m] != expected_swaps[m]);
	}
	n_wrong += (trace.n_moves < 6);
	n_wrong += (fabs(transport_program->arrival_times[0] - 250.0 / TRANSPORT_SPEED_LIMIT) > eps);
	n_wrong += (fabs(transport_program->arrival_times[2] - 700.0 / TRANSPORT_SPEED_LIMIT) > eps);

	free(n_swaps);
	free_schedule_trace(&trace);
	free_transport_program(transport_program);
	free_circuit_graph(circuit_graph);
	free_trap_graph(&trap_graph);

	return check_error("time_schedule_trace times every move", n_wrong, 0);
}

// A T junction: storage on the left, a gate zone on the right, and a measurement zone up the third arm, whose segment
// is longer than the distance between the centers of its zones
static void generate_t_junction_graph
(
	struct TrapGraph *trap_graph
)
{
	enum ZoneType zone_types[4] = { LOAD_ZONE | STORAGE_ZONE, JUNCTION_ZONE, GATE_ZONE, MEASURE_ZONE };
	int zone_capacities[4] = { 4, 0, 2, 4 };
	double positions[4][3] = { { -200.0, 0.0, ION_HEIGHT }, { 0.0, 0.0, ION_HEIGHT }, { 200.0, 0.0, ION_HEIGHT }, { 0.0, 200.0, ION_HEIGHT } };
	int segments[3][2] = { { 0, 1 }, { 1, 2 }, { 1, 3 } };
	double lengths[3] = { 0.0, 0.0, 300.0 };

	generate_trap_graph(4, zone_types, zone_capacities, positions, 3, segments, lengths, trap_graph);
}

int test_trap_graph_routes
(
	double atol
)
{
	struct TrapGraph trap_graph;
	generate_t_junction_graph(&trap_graph);

	int n_wrong = 0;
	int path[4];

	// Straight through the junction, and up its third arm
	n_wrong += (trap_graph_route(&trap_graph, 0, 2, path) != 3) || path[0] != 0 || path[1] != 1 || path[2] != 2;
	n_wrong += (fabs(trap_graph_distance(&trap_graph, 0, 2) - 400.0) > atol);
	n_wrong += (fabs(trap_graph_distance(&trap_graph, 2, 3) - 500.0) > atol);
	n_wrong += (trap_graph_route(&trap_graph, 3, 0, path) != 3) || path[1] != 1 || path[2] != 0;

	// The junction joins all three arms
	n_wrong += (trap_graph_neighbors(&trap_graph, 1, path) != 3);
	n_wrong += (trap_graph_neighbors(&trap_graph, 3, path) != 1) || path[0] != 1;

	// A zone with no segments has no path to the others
	struct TrapGraph unjoined;
	enum ZoneType zone_types[2] = { LOAD_ZONE, GATE_ZONE };
	int zone_capacities[2] = { 2, 2 };
	double positions[2][3] = { { 0.0, 0.0, ION_HEIGHT }, { 100.0, 0.0, ION_HEIGHT } };
	generate_trap_graph(2, zone_types, zone_capacities, positions, 0, NULL, NULL, &unjoined);
	n_wrong += (trap_graph_distance(&unjoined, 0, 1) != HUGE_VAL) + (trap_graph_route(&unjoined, 0, 1, path) != 0);

	free_trap_graph(&unjoined);
	free_trap_graph(&trap_graph);

	return check_error("trap graphs route along their segments", n_wrong, 0);
}

int test_schedule_through_junction
(
	double atol
)
{
	// Two ions are entangled in the gate zone and measured up the third arm, crossing the junction each way
	char *filename = "junction_test.qasm";
	FILE *fp = fopen(filename, "w");
	fprintf(fp, "qubit[2] qreg;\nbit[2] creg;\nh qreg[0];\ncx qreg[0], qreg[1];\nmeasure qreg -> creg;\n");
	fclose(fp);

	struct CircuitGraph *circuit_graph = parse_qasm(filename);
	remove(filename);

	struct TrapGraph trap_graph;
	generate_t_junction_graph(&trap_graph);

	struct TransportCostModel cost_model = distance_transport_cost_model();
	cost_model.junction_y = 30E-6;
	struct ScheduleOptions options = { SCHEDULE_ROUNDS, 1, 0, &cost_model };

	struct TransportProgram *program = calloc(1, sizeof(struct TransportProgram));
	struct ScheduleTrace trace = { 0 };
	int n_wrong = (circuit_graph == NULL);

	if (circuit_graph == NULL || generate_transport_schedule_options(&trap_graph, circuit_graph, program, &options, &trace) == NULL)
		n_wrong++;
	else
	{
		int n_crossings = 0;
		for (int k = 0; k < trace.n_steps; k++)
		{
			struct TransportStep *step = &trace.steps[k];

			// Ions move only along segments, never rest in the junction, and take 30 µs to cross it
			if (step->type == STEP_MOVE)
				n_wrong += (trap_graph.segment_lengths[step->zone * 4 + step->to_zone] < 0.0);
			if (step->type == STEP_MERGE)
				n_wrong += (step->zone == 1);
			if (step->type == STEP_JUNCTION)
			{
				n_wrong += (step->zone != 1) + (fabs(step->end - step->start - 30E-6) > 1E-12);
				n_crossings++;
			}
		}

		// Ion 1 crosses to the gate zone and back for the h gate, both ions do for the cx gate, and both cross up to the
		// measurement zone and back into storage at the end
		n_wrong += (n_crossings != 10);

		// The move from storage up the third arm takes its 200 µm and 300 µm segments at the speed limit, and the
		// crossing
		for (int m = 0; m < trace.n_moves; m++)
			if (trace.moves[m].transport == 2 && trace.moves[m].from_zone == 0)
				n_wrong += (trace.moves[m].arrive - trace.moves[m].depart < 500.0 / TRANSPORT_SPEED_LIMIT + 30E-6 - 1E-12);
	}

	free_schedule_trace(&trace);
	free_transport_program(program);
	free_trap_graph(&trap_graph);
	if (circuit_graph != NULL)
		free_circuit_graph(circuit_graph);

	return check_error("schedules move ions through junctions along the segments", n_wrong, 0);
}

// Whether two schedules are exactly the same
static int same_schedule
(
	struct TransportProgram *a,
	struct ScheduleTrace *trace_a,
	struct TransportProgram *b,
	struct ScheduleTrace *trace_b
)
{
	if (a->n_transports != b->n_transports || trace_a->n_moves != trace_b->n_moves || trace_a->n_rounds != trace_b->n_rounds)
		return 0;

	return memcmp(trace_a->transport_rounds, trace_b->transport_rounds, a->n_transports * sizeof(int)) == 0 &&
		memcmp(a->arrival_times, b->arrival_times, a->n_transports * sizeof(double)) == 0 &&
		memcmp(a->destinations, b->destinations, a->n_transports * sizeof(double[3])) == 0 &&
		memcmp(trace_a->moves, trace_b->moves, trace_a->n_moves * sizeof(struct IonMove)) == 0;
}

int test_lookahead_layers
(
	double atol
)
{
	// Load and storage, two gate zones which hold four ions, and a measure zone; in the circuit, qubit 1 waits for one
	// layer between its two gates with qubit 2
	enum ZoneType zone_types[4] = { LOAD_ZONE | STORAGE_ZONE, GATE_ZONE, GATE_ZONE, MEASURE_ZONE };
	int zone_capacities[4] = { 4, 4, 4, 4 };

	const char *circuits[2] = { "quantumcircuits/lookahead_gap.qasm", "quantumcircuits/test_0000.qasm" };
	enum ScheduleStrategy strategies[2] = { SCHEDULE_LOOKAHEAD, SCHEDULE_LAYER_ASSIGN };
	int n_wrong = 0;

	for (int c = 0; c < 2; c++)
	{
		struct CircuitGraph *circuit_graph = parse_qasm(circuits[c]);

		for (int s = 0; s < 2; s++)
		{
			struct TransportProgram *programs[4];
			struct ScheduleTrace traces[4];
			int scheduled = 1;

			// With the strategy alone, and looking 1 (as by default), 2, and 0 (which is taken as 1) layers ahead
			int depths[4] = { 1, 1, 2, 0 };
			for (int k = 0; k < 4; k++)
			{
				struct TrapGraph trap_graph;
				generate_linear_trap_graph(4, zone_types, zone_capacities, 0.0, 300.0, ION_HEIGHT, &trap_graph);

				struct ScheduleOptions options = { strategies[s], depths[k] };
				programs[k] = calloc(1, sizeof(struct TransportProgram));
				traces[k] = (struct ScheduleTrace) { 0 };

				struct TransportProgram *result = (k == 0)
					? generate_transport_schedule_strategy(&trap_graph, circuit_graph, programs[k], strategies[s], &traces[k])
					: generate_transport_schedule_options(&trap_graph, circuit_graph, programs[k], &options, &traces[k]);

				if (result == NULL)
				{
					n_wrong += 100;
					scheduled = 0;
				}
				else
					n_wrong += count_invalid(&trap_graph, circuit_graph, programs[k], &traces[k], atol);

				free_trap_graph(&trap_graph);
			}

			if (scheduled)
			{
				n_wrong += !same_schedule(programs[0], &traces[0], programs[1], &traces[1]);
				n_wrong += !same_schedule(programs[0], &traces[0], programs[3], &traces[3]);
			}

			// Looking 1 layer ahead, qubit 1 goes back into storage between its gates, and looking 2 layers ahead it
			// stays in the gate zone
			if (c == 0 && scheduled)
			{
				for (int k = 1; k <= 2; k++)
				{
					int n_back = 0;
					for (int m = 0; m < traces[k].n_moves; m++)
					{
						struct IonMove *move = &traces[k].moves[m];
						n_back += (move->ion == 1 && move->transport < 0 && move->arrive <= programs[k]->arrival_times[2]);
					}
					n_wrong += (k == 1) ? (n_back == 0) : (n_back != 0);
				}

				// So the last gate starts sooner
				n_wrong += (programs[2]->arrival_times[2] >= programs[1]->arrival_times[2]);
			}

			for (int k = 0; k < 4; k++)
			{
				free_schedule_trace(&traces[k]);
				if (scheduled)
					free_transport_program(programs[k]);
			}
		}

		free_circuit_graph(circuit_graph);
	}

	return check_error("lookahead_layers keeps ions for operations further ahead", n_wrong, 0);
}

// Pairs of operations on the same ion which are in the same layer (or round), or in the opposite order to the circuit
// while not both diagonal
static int count_order_violations
(
	struct CircuitGraph *circuit_graph,
	const int *layers
)
{
	int n_violations = 0;
	for (int i = 0; i < circuit_graph->n_instruction_nodes; i++)
		for (int j = i + 1; j < circuit_graph->n_instruction_nodes; j++)
		{
			struct InstructionNode *a = &circuit_graph->instruction_nodes[i];
			struct InstructionNode *b = &circuit_graph->instruction_nodes[j];

			int share = 0;
			for (int s = 0; s < a->n_target_ions; s++)
				for (int t = 0; t < b->n_target_ions; t++)
					share |= (a->target_ions[s] == b->target_ions[t]);

			if (share)
				n_violations += (layers[i] == layers[j]) || (layers[i] > layers[j] && !(a->diagonal && b->diagonal));
		}

	return n_violations;
}

static int count_layers
(
	struct CircuitGraph *circuit_graph
)
{
	int n_layers = 0;
	for (int i = 0; i < circuit_graph->n_instruction_nodes; i++)
		n_layers = MAX(n_layers, circuit_graph->layer_idxs[i]);

	return n_layers;
}

int test_commuting_layers
(
	double atol
)
{
	int n_wrong = 0;

	// Gates which are diagonal in the computational basis, whatever their parameters
	const char *diagonal[] = { "rz(0.5)", "rzz(0.2)", "cz", "cp(pi/4)", "crz(0.1)", "z", "s", "sdg", "t", "tdg", "p(0.3)" };
	const char *not_diagonal[] = { "h", "x", "cx", "rx(0.1)", "rxx(0.2)", "swap", "measure", "zz" };
	for (size_t g = 0; g < sizeof(diagonal) / sizeof(diagonal[0]); g++)
		n_wrong += !is_diagonal_gate(diagonal[g]);
	for (size_t g = 0; g < sizeof(not_diagonal) / sizeof(not_diagonal[0]); g++)
		n_wrong += is_diagonal_gate(not_diagonal[g]);

	// No benchmark circuit takes more layers with commuting operations moved, and every layering keeps the order the
	// circuit allows
	struct dirent **entries;
	int n_entries = scandir("quantumcircuits/benchmarks", &entries, NULL, alphasort);
	int n_circuits = 0;
	for (int e = 0; e < n_entries; e++)
	{
		if (strstr(entries[e]->d_name, ".qasm") != NULL)
		{
			char filename[512];
			snprintf(filename, sizeof(filename), "quantumcircuits/benchmarks/%s", entries[e]->d_name);
			struct CircuitGraph *circuit_graph = parse_qasm(filename);

			if (circuit_graph == NULL)
				n_wrong++;
			else
			{
				int n_layers = count_layers(circuit_graph);
				n_wrong += count_order_violations(circuit_graph, circuit_graph->layer_idxs);

				compute_commuting_layers(circuit_graph);
				n_wrong += (count_layers(circuit_graph) > n_layers);
				n_wrong += count_order_violations(circuit_graph, circuit_graph->layer_idxs);

				// QAOA MaxCut on 6 qubits packs its three rzz gates per qubit into 3 layers rather than 6, in each of its
				// two steps
				if (strcmp(entries[e]->d_name, "qaoa_06.qasm") == 0)
					n_wrong += (n_layers != 16) + (count_layers(circuit_graph) != 10);

				free_circuit_graph(circuit_graph);
				n_circuits++;
			}
		}

		free(entries[e]);
	}
	free(entries);
	n_wrong += (n_circuits != 21);

	// Scheduling with commuting operations moved, on a line of five gate zones: the schedule is valid, runs the
	// operations in an order the circuit allows, takes fewer rounds, and leaves the layers of the circuit as they were
	struct CircuitGraph *circuit_graph = parse_qasm("quantumcircuits/benchmarks/qaoa_06.qasm");
	int n_nodes = circuit_graph->n_instruction_nodes;
	int *layers = malloc(n_nodes * sizeof(int));
	memcpy(layers, circuit_graph->layer_idxs, n_nodes * sizeof(int));

	int n_zones;
	enum ZoneType *zone_types;
	int *zone_capacities;
	parse_zone_layout("load, storage x2, gate x5, measure", circuit_graph->n_ions, &n_zones, &zone_types, &zone_capacities);

	int n_rounds[2] = { 0, 0 };
	for (int commute = 0; commute < 2; commute++)
	{
		struct TrapGraph trap_graph;
		generate_linear_trap_graph(n_zones, zone_types, zone_capacities, 0.0, (n_zones - 1) * 100.0, ION_HEIGHT, &trap_graph);

		struct ScheduleOptions options = { SCHEDULE_LAYER_ASSIGN, 1, commute };
		struct TransportProgram *program = calloc(1, sizeof(struct TransportProgram));
		struct ScheduleTrace trace = { 0 };

		if (generate_transport_schedule_options(&trap_graph, circuit_graph, program, &options, &trace) == NULL)
		{
			n_wrong += 100;
			free(program);
		}
		else
		{
			n_wrong += count_invalid(&trap_graph, circuit_graph, program, &trace, atol);
			n_wrong += count_order_violations(circuit_graph, trace.transport_rounds);
			n_rounds[commute] = trace.n_rounds;
			free_transport_program(program);
		}

		n_wrong += (memcmp(layers, circuit_graph->layer_idxs, n_nodes * sizeof(int)) != 0);

		free_schedule_trace(&trace);
		free_trap_graph(&trap_graph);
	}
	n_wrong += (n_rounds[1] >= n_rounds[0]);

	free(zone_types);
	free(zone_capacities);
	free(layers);
	free_circuit_graph(circuit_graph);

	return check_error("compute_commuting_layers moves commuting operations into fewer layers", n_wrong, 0);
}

// Schedule a circuit written to a file on a trap graph with a cost model (or, if it is NULL, one which only counts
// distance), returning NULL if it fails
static struct TransportProgram* schedule_text
(
	const char *qasm,
	struct TrapGraph *trap_graph,
	enum ScheduleStrategy strategy,
	const struct TransportCostModel *cost_model,
	struct CircuitGraph **circuit_graph,
	struct ScheduleTrace *trace
)
{
	struct TransportCostModel distance_model = distance_transport_cost_model();
	struct ScheduleOptions options = { strategy, 1, 0, (cost_model != NULL) ? cost_model : &distance_model };

	char *filename = "check_test.qasm";
	FILE *fp = fopen(filename, "w");
	fputs(qasm, fp);
	fclose(fp);

	*circuit_graph = parse_qasm(filename);
	remove(filename);
	if (*circuit_graph == NULL)
		return NULL;

	struct TransportProgram *program = calloc(1, sizeof(struct TransportProgram));
	if (generate_transport_schedule_options(trap_graph, *circuit_graph, program, &options, trace) == NULL)
	{
		free_transport_program(program);
		return NULL;
	}

	return program;
}

// Storage (zone 1), an empty zone, a gate zone and a measurement zone 100 µm apart, holding 3, 2, 2 and 3 ions
static void generate_check_graph
(
	struct TrapGraph *trap_graph
)
{
	enum ZoneType zone_types[4] = { LOAD_ZONE | STORAGE_ZONE, EMPTY_ZONE, GATE_ZONE, MEASURE_ZONE };
	int zone_capacities[4] = { 3, 2, 2, 3 };
	generate_linear_trap_graph(4, zone_types, zone_capacities, 0.0, 300.0, ION_HEIGHT, trap_graph);
}

int test_three_ion_schedule_by_hand
(
	double atol
)
{
	// Three ions start in storage (zone 1) in the chain [1, 2, 3], whose front faces the empty zone (zone 2); zones are
	// 100 µm apart, so a segment takes 100 µs at the speed limit. A cx gate on ions 1 and 2 and a measurement of ion 3
	// run in the same round:
	// - Ion 1 is at the front, so it splits off at 0 and reaches the gate zone (zone 3) at 200 µs
	// - Ion 2 splits off next, and follows a segment behind, waiting for each segment and zone to be free; it reaches
	//   the gate zone at 300 µs and merges into the front of the chain there: [2, 1]
	// - Ion 3 is left alone in storage, so it sets off without splitting once storage is free (at 100 µs), and follows
	//   a segment behind ion 2. To reach the measurement zone (zone 4) it passes the chain in the gate zone: it merges
	//   into its front at 400 µs, swaps past ions 2 and 1 to the back, splits off, and arrives at 500 µs
	// - The gate (GATE_DURATION, 100 µs) and the measurement (DETECTION_DURATION, 200 µs) start at 500 µs, so the round
	//   ends at 700 µs
	// - Back into storage, the ions which pass no chain and need no swaps go first: ion 2, at the front of the gate
	//   chain [2, 1], splits off at 700 µs and arrives at 900 µs; ion 1, now alone, follows a segment behind (1000 µs)
	//   and merges into the front of storage: [1, 2]; ion 3 no longer has to pass a chain, waits for the gate zone to be
	//   free at 800 µs and arrives at 1100 µs, merging into the front: [3, 1, 2]
	struct TrapGraph trap_graph;
	generate_check_graph(&trap_graph);

	struct CircuitGraph *circuit_graph;
	struct ScheduleTrace trace = { 0 };
	struct TransportProgram *program = schedule_text("qubit[3] qreg;\nbit[3] creg;\ncx qreg[0], qreg[1];\nmeasure qreg[2] -> creg[2];\n",
		&trap_graph, SCHEDULE_ROUNDS, NULL, &circuit_graph, &trace);

	int n_wrong = (program == NULL);
	if (program != NULL)
	{
		double us = 1E-6;
		double eps = 1E-12;

		n_wrong += (program->n_ions != 3) + (trace.n_moves != 6) + (trace.n_rounds != 1);
		n_wrong += (fabs(program->arrival_times[0] - 500 * us) > eps) + (fabs(program->arrival_times[1] - 500 * us) > eps);

		int expected_ions[6] = { 1, 2, 3, 1, 2, 3 };
		double expected_departures[6] = { 0, 0, 100, 700, 700, 700 };
		double expected_arrivals[6] = { 200, 300, 500, 1000, 900, 1100 };
		int expected_swaps[6] = { 0, 0, 2, 0, 0, 0 };
		int swaps[6] = { 0 };
		int n_splits = 0;
		int n_merges = 0;
		for (int k = 0; k < trace.n_steps; k++)
		{
			struct TransportStep *step = &trace.steps[k];
			if (step->type == STEP_ION_SWAP && step->index < 6)
				swaps[step->index]++;
			n_splits += (step->type == STEP_SPLIT);
			n_merges += (step->type == STEP_MERGE);
		}

		for (int m = 0; m < MIN(trace.n_moves, 6); m++)
		{
			n_wrong += (trace.moves[m].ion != expected_ions[m]) + (swaps[m] != expected_swaps[m]);
			n_wrong += (fabs(trace.moves[m].depart - expected_departures[m] * us) > eps);
			n_wrong += (fabs(trace.moves[m].arrive - expected_arrivals[m] * us) > eps);
		}
		n_wrong += (n_splits != 4) + (n_merges != 4);

		n_wrong += (check_schedule_trace(&trap_graph, circuit_graph, program, &trace, NULL, NULL, 0) != 0);

		free_transport_program(program);
	}

	free_schedule_trace(&trace);
	if (circuit_graph != NULL)
		free_circuit_graph(circuit_graph);
	free_trap_graph(&trap_graph);

	return check_error("a three-ion schedule matches its timing by hand and passes the checker", n_wrong, 0);
}

// Remove the steps of a trace for which a predicate holds
static void remove_steps
(
	struct ScheduleTrace *trace,
	int (*remove)(struct TransportStep *step, int arg),
	int arg
)
{
	int n = 0;
	for (int k = 0; k < trace->n_steps; k++)
		if (!remove(&trace->steps[k], arg))
			trace->steps[n++] = trace->steps[k];

	trace->n_steps = n;
}

static int is_swap
(
	struct TransportStep *step,
	int arg
)
{
	return step->type == STEP_ION_SWAP;
}

// Steps of the move arg in zone 3, except moves along segments
static int passes_storage
(
	struct TransportStep *step,
	int arg
)
{
	return step->index == arg && step->zone == 2 && step->type != STEP_MOVE && step->type != STEP_OPERATION;
}

// Whether checking a trace finds violations, one of which is described by a phrase
static int finds
(
	struct TrapGraph *trap_graph,
	struct CircuitGraph *circuit_graph,
	struct TransportProgram *program,
	struct ScheduleTrace *trace,
	const char *phrase
)
{
	char message[4096];
	int n = check_schedule_trace(trap_graph, circuit_graph, program, trace, NULL, message, sizeof(message));

	return n > 0 && strstr(message, phrase) != NULL;
}

int test_schedule_check_finds_violations
(
	double atol
)
{
	int n_wrong = 0;

	struct TrapGraph trap_graph;
	generate_check_graph(&trap_graph);

	// The schedule of the three-ion test, where ion 3 passes through the chain of ions 2 and 1 in the gate zone
	struct CircuitGraph *circuit_graph;
	struct ScheduleTrace trace = { 0 };
	struct TransportProgram *program = schedule_text("qubit[3] qreg;\nbit[3] creg;\ncx qreg[0], qreg[1];\nmeasure qreg[2] -> creg[2];\n",
		&trap_graph, SCHEDULE_ROUNDS, NULL, &circuit_graph, &trace);

	if (program == NULL)
		n_wrong++;
	else
	{
		struct ScheduleTrace broken = trace;
		broken.steps = malloc(trace.n_steps * sizeof(struct TransportStep));

		// Ion 3 leaves the gate chain from its front, which it merged into, without swapping past ions 2 and 1
		memcpy(broken.steps, trace.steps, trace.n_steps * sizeof(struct TransportStep));
		remove_steps(&broken, is_swap, 0);
		n_wrong += !finds(&trap_graph, circuit_graph, program, &broken, "from the middle, or from the end");

		// Ion 2 sets off while ion 1 is still on the first segment
		memcpy(broken.steps, trace.steps, trace.n_steps * sizeof(struct TransportStep));
		broken.n_steps = trace.n_steps;
		for (int k = 0; k < broken.n_steps; k++)
			if (broken.steps[k].type == STEP_MOVE && broken.steps[k].ion == 2 && broken.steps[k].zone == 0)
			{
				broken.steps[k].start -= 50E-6;
				broken.steps[k].end -= 50E-6;
			}
		n_wrong += !finds(&trap_graph, circuit_graph, program, &broken, "take up the segment between zones 1 and 2 at once");

		// Ion 1 jumps straight from storage to the gate zone
		memcpy(broken.steps, trace.steps, trace.n_steps * sizeof(struct TransportStep));
		for (int k = 0; k < broken.n_steps; k++)
			if (broken.steps[k].type == STEP_MOVE && broken.steps[k].ion == 1 && broken.steps[k].zone == 0)
				broken.steps[k].to_zone = 2;
		n_wrong += !finds(&trap_graph, circuit_graph, program, &broken, "which are not joined by a segment");

		// The gate starts before ion 2 arrives
		memcpy(broken.steps, trace.steps, trace.n_steps * sizeof(struct TransportStep));
		double arrival = program->arrival_times[0];
		program->arrival_times[0] -= 250E-6;
		n_wrong += !finds(&trap_graph, circuit_graph, program, &broken, "starts before ion 2 arrives");
		program->arrival_times[0] = arrival;

		// The gate zone only holds one ion
		trap_graph.zone_nodes[2].zone_capacity = 1;
		n_wrong += !finds(&trap_graph, circuit_graph, program, &trace, "more than its capacity of 1");
		trap_graph.zone_nodes[2].zone_capacity = 2;

		free(broken.steps);
		free_transport_program(program);
	}

	free_schedule_trace(&trace);
	if (circuit_graph != NULL)
		free_circuit_graph(circuit_graph);
	free_trap_graph(&trap_graph);

	// In the schedule of time_schedule_trace's test, ion 3 passes through the chain of ions 1 and 2 in storage on its
	// way to the gate zone; without merging into it and swapping past them, it goes straight through them
	enum ZoneType zone_types[7] = { LOAD_ZONE, EMPTY_ZONE, STORAGE_ZONE, EMPTY_ZONE, GATE_ZONE, EMPTY_ZONE, MEASURE_ZONE };
	int zone_capacities[7] = { 5, 3, 10, 3, 2, 3, 4 };
	generate_linear_trap_graph(7, zone_types, zone_capacities, -150.0, 150.0, ION_HEIGHT, &trap_graph);

	circuit_graph = parse_qasm("quantumcircuits/test_0000.qasm");
	trace = (struct ScheduleTrace) { 0 };
	program = calloc(1, sizeof(struct TransportProgram));
	struct TransportCostModel cost_model = distance_transport_cost_model();
	struct ScheduleOptions options = { SCHEDULE_ROUNDS, 1, 0, &cost_model };
	if (generate_transport_schedule_options(&trap_graph, circuit_graph, program, &options, &trace) == NULL)
		n_wrong++;
	else
	{
		n_wrong += (check_schedule_trace(&trap_graph, circuit_graph, program, &trace, NULL, NULL, 0) != 0);

		remove_steps(&trace, passes_storage, 4);
		n_wrong += !finds(&trap_graph, circuit_graph, program, &trace, "passes through the chain of 2 ions in zone 3");
	}

	free_schedule_trace(&trace);
	free_transport_program(program);
	free_circuit_graph(circuit_graph);
	free_trap_graph(&trap_graph);

	return check_error("check_schedule_trace finds ions passing, colliding, jumping, early and over capacity", n_wrong, 0);
}

int test_qccd_cost_model_by_hand
(
	double atol
)
{
	// The default cost model is that of tiopt.qccd: ions move at 20 µm/µs (5 µs for each 100 µm segment), a split or
	// merge takes 80 µs, and an FM gate on a chain of 2 ions takes 100 µs. A cx gate on ions 1 and 2, which start in
	// storage in the chain [1, 2]:
	// - Ion 1 splits off from 0 to 80 µs (adding 0.1 quanta to it and to the chain it leaves), and moves to the gate zone
	//   from 80 µs to 90 µs, gaining 0.01 quanta and 5 µs of background heating (100 quanta/s) on each segment: 0.121
	// - Ion 2, left alone, takes the chain's 0.1 quanta and its heating since 0 (0.008), sets off when ion 1 leaves the
	//   empty zone at 85 µs (gaining 5 µs of heating while it waits), reaches the gate zone at 95 µs (0.1295 quanta)
	//   and merges into the chain there from 95 µs to 175 µs, which then holds 0.121 + 0.0005 (heating from 90 µs to
	//   95 µs) + 0.1295 + 0.1 (the merge) quanta
	// - The gate starts at 175 µs, after 80 µs more of heating, with n_bar = 0.359, and has fidelity
	//   1 - 1E-6/µs 100 µs - 1E-4 2 / ln 2 (2 n_bar + 1)
	struct TrapGraph trap_graph;
	generate_check_graph(&trap_graph);

	struct TransportCostModel cost_model = default_transport_cost_model();
	struct CircuitGraph *circuit_graph;
	struct ScheduleTrace trace = { 0 };
	struct TransportProgram *program = schedule_text("qubit[2] qreg;\ncx qreg[0], qreg[1];\n", &trap_graph, SCHEDULE_ROUNDS, &cost_model, &circuit_graph, &trace);

	int n_wrong = (program == NULL);
	if (program != NULL)
	{
		double us = 1E-6;
		double n_bar = 0.121 + 0.0005 + 0.1295 + 0.1 + 0.008;
		double fidelity = 1.0 - 1E-4 - 1E-4 * 2.0 / log(2.0) * (2.0 * n_bar + 1.0);

		n_wrong += (fabs(program->arrival_times[0] - 175 * us) > 1E-12);
		n_wrong += (fabs(trace.moves[0].arrive - 90 * us) > 1E-12) + (fabs(trace.moves[1].arrive - 175 * us) > 1E-12);
		n_wrong += (fabs(trace.operation_n_bars[0] - n_bar) > 1E-9);
		n_wrong += (fabs(trace.max_n_bar - n_bar) > 1E-9);
		n_wrong += (fabs(trace.fidelity - fidelity) > 1E-12);
		n_wrong += (check_schedule_trace(&trap_graph, circuit_graph, program, &trace, &cost_model, NULL, 0) != 0);

		free_transport_program(program);
	}

	free_schedule_trace(&trace);
	if (circuit_graph != NULL)
		free_circuit_graph(circuit_graph);
	free_trap_graph(&trap_graph);

	return check_error("the default cost model times, heats and scores a schedule as tiopt.qccd does", n_wrong, 0);
}

int test_gates_take_chain_in_turn
(
	double atol
)
{
	// Four ions start in storage, and a gate zone holds all four: the two cx gates of the first layer both run there,
	// one after the other (Mølmer-Sørensen gates drive the motion of the whole chain), while the single-qubit gates of
	// the second layer, which also all run there, happen together
	enum ZoneType zone_types[3] = { LOAD_ZONE | STORAGE_ZONE, GATE_ZONE, MEASURE_ZONE };
	int zone_capacities[3] = { 4, 4, 4 };
	struct TrapGraph trap_graph;
	generate_linear_trap_graph(3, zone_types, zone_capacities, 0.0, 200.0, ION_HEIGHT, &trap_graph);

	struct TransportCostModel cost_model = default_transport_cost_model();
	struct CircuitGraph *circuit_graph;
	struct ScheduleTrace trace = { 0 };
	struct TransportProgram *program = schedule_text(
		"qubit[4] qreg;\ncx qreg[0], qreg[1];\ncx qreg[2], qreg[3];\nh qreg[0];\nh qreg[1];\nh qreg[2];\nh qreg[3];\n",
		&trap_graph, SCHEDULE_ROUNDS, &cost_model, &circuit_graph, &trace
	);

	int n_wrong = (program == NULL);
	if (program != NULL)
	{
		struct TransportStep *ops[6] = { NULL };
		for (int k = 0; k < trace.n_steps; k++)
			if (trace.steps[k].type == STEP_OPERATION && trace.steps[k].index < 6)
				ops[trace.steps[k].index] = &trace.steps[k];

		for (int i = 0; i < 6; i++)
			n_wrong += (ops[i] == NULL || ops[i]->zone != 1);

		if (n_wrong == 0)
		{
			// The gates share a round and the zone, and the second starts when the first ends
			n_wrong += (trace.transport_rounds[0] != trace.transport_rounds[1]);
			struct TransportStep *first = (ops[0]->start <= ops[1]->start) ? ops[0] : ops[1];
			struct TransportStep *second = (first == ops[0]) ? ops[1] : ops[0];
			n_wrong += (fabs(second->start - first->end) > 1E-12);
			n_wrong += (fabs(program->arrival_times[second->index] - second->start) > 1E-12);

			// The single-qubit gates run together
			for (int i = 3; i < 6; i++)
				n_wrong += (trace.transport_rounds[i] != trace.transport_rounds[2]) + (fabs(ops[i]->start - ops[2]->start) > 1E-12);
		}

		n_wrong += (check_schedule_trace(&trap_graph, circuit_graph, program, &trace, &cost_model, NULL, 0) != 0);

		free_transport_program(program);
	}

	free_schedule_trace(&trace);
	if (circuit_graph != NULL)
		free_circuit_graph(circuit_graph);
	free_trap_graph(&trap_graph);

	return check_error("gates on several ions take the chain of their zone one after another", n_wrong, 0);
}

int test_mid_chain_swap_cost
(
	double atol
)
{
	// Four ions start in storage in the chain [1, 2, 3, 4], whose front faces the gate zone. Ion 3 goes first (it needs
	// the fewest swaps to reach the front): it swaps with ion 2 in the middle of the chain, which splits the pair off on
	// both sides, rotates it, and merges it back on both sides (2 x 80 + 42 + 2 x 80 = 362 µs), and then with ion 1 at
	// the end of the chain, which takes one split and one merge (80 + 42 + 80 = 202 µs)
	enum ZoneType zone_types[3] = { LOAD_ZONE | STORAGE_ZONE, GATE_ZONE, MEASURE_ZONE };
	int zone_capacities[3] = { 4, 2, 4 };
	struct TrapGraph trap_graph;
	generate_linear_trap_graph(3, zone_types, zone_capacities, 0.0, 200.0, ION_HEIGHT, &trap_graph);

	struct TransportCostModel cost_model = default_transport_cost_model();
	struct CircuitGraph *circuit_graph;
	struct ScheduleTrace trace = { 0 };
	struct TransportProgram *program = schedule_text("qubit[4] qreg;\nx qreg[2];\nx qreg[3];\n", &trap_graph,
		SCHEDULE_ROUNDS, &cost_model, &circuit_graph, &trace);

	int n_wrong = (program == NULL);
	if (program != NULL)
	{
		double us = 1E-6;
		double expected[2] = { 362 * us, 202 * us };
		int expected_others[2] = { 2, 1 };
		int n_swaps = 0;

		for (int k = 0; k < trace.n_steps; k++)
		{
			struct TransportStep *step = &trace.steps[k];
			if (step->type != STEP_ION_SWAP || step->ion != 3)
				continue;

			if (n_swaps < 2)
				n_wrong += (fabs(step->end - step->start - expected[n_swaps]) > 1E-12) + (step->other_ion != expected_others[n_swaps]);
			n_swaps++;
		}

		n_wrong += (n_swaps != 2);
		n_wrong += (check_schedule_trace(&trap_graph, circuit_graph, program, &trace, &cost_model, NULL, 0) != 0);

		free_transport_program(program);
	}

	free_schedule_trace(&trace);
	if (circuit_graph != NULL)
		free_circuit_graph(circuit_graph);
	free_trap_graph(&trap_graph);

	return check_error("a physical swap in the middle of a chain takes two splits and two merges", n_wrong, 0);
}

int test_cycle_breaker_in_storage_zones
(
	double atol
)
{
	// Two gate zones which can also store ions hold [1, 2] and [3, 4] after the first layer; the second layer pairs
	// ions across them, so the zones would exchange ions while both are full. One ion has to leave for storage first,
	// although it waits in a zone which can store it
	enum ZoneType zone_types[4] = { LOAD_ZONE | STORAGE_ZONE, GATE_ZONE | STORAGE_ZONE, GATE_ZONE | STORAGE_ZONE, MEASURE_ZONE };
	int zone_capacities[4] = { 4, 2, 2, 4 };
	struct TrapGraph trap_graph;
	generate_linear_trap_graph(4, zone_types, zone_capacities, 0.0, 300.0, ION_HEIGHT, &trap_graph);

	struct TransportCostModel cost_model = default_transport_cost_model();
	struct CircuitGraph *circuit_graph;
	struct ScheduleTrace trace = { 0 };
	struct TransportProgram *program = schedule_text(
		"qubit[4] qreg;\ncx qreg[0], qreg[1];\ncx qreg[2], qreg[3];\ncx qreg[0], qreg[2];\ncx qreg[1], qreg[3];\n",
		&trap_graph, SCHEDULE_LAYER_ASSIGN, &cost_model, &circuit_graph, &trace
	);

	int n_wrong = (program == NULL);
	if (program != NULL)
	{
		char message[1024];
		int n_violations = check_schedule_trace(&trap_graph, circuit_graph, program, &trace, &cost_model, message, sizeof(message));
		if (n_violations != 0)
			printf("%s", message);
		n_wrong += n_violations;

		// One ion of the second layer goes back to storage before it goes on
		int n_evicted = 0;
		for (int m = 0; m < trace.n_moves; m++)
			n_evicted += (trace.moves[m].transport < 0 && trace.moves[m].to_zone == 0);
		n_wrong += (n_evicted != 1);

		free_transport_program(program);
	}

	free_schedule_trace(&trace);
	if (circuit_graph != NULL)
		free_circuit_graph(circuit_graph);
	free_trap_graph(&trap_graph);

	return check_error("full zones which can store ions exchange ions through storage", n_wrong, 0);
}

int test_stay_evicts_ion_needed_last
(
	double atol
)
{
	// A gate zone holds three ions. The first layer brings ions 1 and 2 (for a cx) and ion 3 (for an x) into it, and
	// with SCHEDULE_STAY they stay there. The cx on ions 3 and 4 of the second layer needs room for ion 4, so one of
	// ions 1 and 2 moves back into storage: ion 2, which no later operation needs (ion 1 has an x in the third layer,
	// which it takes where it is)
	enum ZoneType zone_types[2] = { LOAD_ZONE | STORAGE_ZONE, GATE_ZONE };
	int zone_capacities[2] = { 4, 3 };
	struct TrapGraph trap_graph;
	generate_linear_trap_graph(2, zone_types, zone_capacities, 0.0, 100.0, ION_HEIGHT, &trap_graph);

	struct TransportCostModel cost_model = default_transport_cost_model();
	struct CircuitGraph *circuit_graph;
	struct ScheduleTrace trace = { 0 };
	struct TransportProgram *program = schedule_text(
		"qubit[4] qreg;\ncx qreg[0], qreg[1];\nx qreg[2];\ncx qreg[2], qreg[3];\nx qreg[0];\n",
		&trap_graph, SCHEDULE_STAY, &cost_model, &circuit_graph, &trace
	);

	int n_wrong = (program == NULL);
	if (program != NULL)
	{
		int n_back = 0;
		int evicted = 0;
		for (int m = 0; m < trace.n_moves; m++)
		{
			struct IonMove *move = &trace.moves[m];
			if (move->transport < 0)
			{
				n_back++;
				evicted = move->ion;
				n_wrong += (move->from_zone != 1) + (move->to_zone != 0);
			}

			// Ion 1 takes its x where it is
			if (move->transport == 3)
				n_wrong += (move->from_zone != 1) + (move->to_zone != 1);
		}

		n_wrong += (n_back != 1) + (evicted != 2);
		n_wrong += (check_schedule_trace(&trap_graph, circuit_graph, program, &trace, &cost_model, NULL, 0) != 0);

		free_transport_program(program);
	}

	free_schedule_trace(&trace);
	if (circuit_graph != NULL)
		free_circuit_graph(circuit_graph);
	free_trap_graph(&trap_graph);

	return check_error("with SCHEDULE_STAY, ions stay where they are, and the one needed last makes room", n_wrong, 0);
}
