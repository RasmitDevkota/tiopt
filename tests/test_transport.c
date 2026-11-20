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
	result += test_lookahead_layers(atol);
	result += test_commuting_layers(atol);

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

	// h, x, y, z on each qubit, then three layers of pairs of two-qubit gates, then a measurement of every qubit
	int expected_n_targets[11] = { 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4 };
	int expected_targets[11][4] = { { 1 }, { 2 }, { 3 }, { 4 }, { 1, 2 }, { 3, 4 }, { 1, 3 }, { 2, 4 }, { 1, 4 }, { 2, 3 }, { 1, 2, 3, 4 } };
	int expected_layers[11] = { 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5 };
	int expected_n_prev_nodes[11] = { 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2 };

	int n_wrong = (circuit_graph->n_ions != 4) + (circuit_graph->n_instruction_nodes != 11);

	for (int i = 0; i < MIN(circuit_graph->n_instruction_nodes, 11); i++)
	{
		struct InstructionNode *node = &circuit_graph->instruction_nodes[i];

		n_wrong += (node->n_target_ions != expected_n_targets[i]);
		n_wrong += (node->operation_type != ((i == 10) ? DETECT : QUBIT));
		n_wrong += (node->graphindex != i + 1);
		n_wrong += (node->n_prev_nodes != expected_n_prev_nodes[i]);
		n_wrong += (circuit_graph->layer_idxs[i] != expected_layers[i]);

		for (int t = 0; t < MIN(node->n_target_ions, 4); t++)
			n_wrong += (node->target_ions[t] != expected_targets[i][t]);
	}

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

	struct TransportProgram *transport_program = malloc(sizeof(struct TransportProgram));
	struct ScheduleTrace trace = { 0 };
	transport_program = generate_transport_schedule_strategy(&trap_graph, circuit_graph, transport_program, DEFAULT_SCHEDULE_STRATEGY, &trace);

	if (transport_program == NULL)
	{
		free_schedule_trace(&trace);
		free_circuit_graph(circuit_graph);
		free_trap_graph(&trap_graph);
		return check_error("generate_transport_schedule succeeds", HUGE_VAL, 0);
	}

	int n_wrong = (transport_program->n_ions != 4) + (transport_program->n_transports != 11);

	// Ions start in the load zone, and are transported to the gate zone for gates and to the measure zone for detection
	for (int i = 0; i < 4; i++)
		n_wrong += (fabs(transport_program->initial_positions[i][0] + 150.0) > atol);

	for (int t = 0; t < MIN(transport_program->n_transports, 11); t++)
	{
		double x_expected = (t == 10) ? 150.0 : 50.0;

		n_wrong += (fabs(transport_program->destinations[t][0] - x_expected) > atol);
		n_wrong += (fabs(transport_program->destinations[t][2] - ION_HEIGHT) > atol);
	}

	// The gate zone only holds two ions, so each layer runs in rounds which fit in it: the four single-qubit gates of
	// the first layer in two rounds, each pair of two-qubit gates in two more, and the measurement of every qubit in one
	int expected_rounds[11] = { 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9 };
	n_wrong += (trace.n_rounds != 9);
	for (int t = 0; t < MIN(transport_program->n_transports, 11); t++)
		n_wrong += (trace.transport_rounds[t] != expected_rounds[t]);

	// Transports in the same round arrive together, and later rounds arrive later
	for (int t = 1; t < MIN(transport_program->n_transports, 11); t++)
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
		int expected_targets[6] = { 1, 1, 2, 2, 2, 3 };
		n_wrong += (circuit_graph->n_instruction_nodes != 6);
		for (int i = 0; i < MIN(circuit_graph->n_instruction_nodes, 6); i++)
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

	struct TransportProgram *transport_program = malloc(sizeof(struct TransportProgram));
	struct ScheduleTrace trace = { 0 };
	if (generate_transport_schedule_strategy(&trap_graph, circuit_graph, transport_program, SCHEDULE_ROUNDS, &trace) == NULL)
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
	// ions 1 and 2 go back into storage (zone 2), arriving at 450 µs and 500 µs, and ions 3 and 4 leave the load zone at
	// 350 µs. They have to pass through the chain in storage: each merges into its front, swaps with ions 1 and 2 to its
	// back, splits off and goes on, so ion 3 waits in zone 1 until storage is free at 500 µs and arrives at 650 µs, and
	// ion 4 arrives one segment later, at 700 µs
	double expected_departures[6] = { 0.0, 0.0, 350.0, 350.0, 350.0, 350.0 };
	double expected_arrivals[6] = { 200.0, 250.0, 450.0, 500.0, 650.0, 700.0 };
	int expected_ions[6] = { 1, 2, 1, 2, 3, 4 };
	int expected_swaps[6] = { 0, 0, 1, 0, 2, 2 };
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

	struct TransportCostModel cost_model = default_transport_cost_model();
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
