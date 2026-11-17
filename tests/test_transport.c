#include "test_transport.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"
#include "transport.h"
#include "circuits.h"
#include "test_utils.h"

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

	// The first round starts after the 200 µm transport from the load zone to the gate zone
	n_wrong += (fabs(transport_program->arrival_times[0] - 200.0 / TRANSPORT_SPEED_LIMIT) > atol);

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

			// Follow the ions, checking each round once all of its transports are done
			int n_ions = circuit_graph->n_ions;
			int *ion_zones = calloc(n_ions, sizeof(int));
			int *occupancy = calloc(trap_graph.n_zones, sizeof(int));
			occupancy[0] = n_ions;

			for (int m = 0; m < trace.n_moves; m++)
			{
				struct IonMove *move = &trace.moves[m];
				ion_zones[move->ion - 1] = move->to_zone;
				occupancy[move->from_zone]--;
				occupancy[move->to_zone]++;

				int last = (move->transport >= 0) && (m + 1 == trace.n_moves || trace.moves[m + 1].transport < 0 ||
					trace.transport_rounds[trace.moves[m + 1].transport] != trace.transport_rounds[move->transport]);
				if (!last)
					continue;

				int round = trace.transport_rounds[move->transport];
				for (int t = 0; t < program->n_transports; t++)
				{
					if (trace.transport_rounds[t] != round)
						continue;

					struct InstructionNode *node = &circuit_graph->instruction_nodes[t];
					for (int i = 0; i < node->n_target_ions; i++)
						n_wrong += (fabs(trap_graph.zone_nodes[ion_zones[node->target_ions[i] - 1]].position[0] - program->destinations[t][0]) > atol);
				}

				for (int z = 0; z < trap_graph.n_zones; z++)
					n_wrong += (occupancy[z] > trap_graph.zone_nodes[z].zone_capacity);
			}

			free(occupancy);
			free(ion_zones);
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
	int n_wrong = 0;
	int n_releases = 0;

	for (int m = 0; m < trace.n_moves; m++)
	{
		struct IonMove *move = &trace.moves[m];

		// Every move takes as long as its distance at the speed limit
		double distance = fabs(trap_graph.zone_nodes[move->to_zone].position[0] - trap_graph.zone_nodes[move->from_zone].position[0]);
		n_wrong += (fabs(move->arrive - move->depart - distance / TRANSPORT_SPEED_LIMIT) > eps);

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

	// In the second round, ions 1 and 2 leave the gate zone (50 µm) for storage (-50 µm) once the first round is done,
	// and only then do ions 3 and 4 come from the load zone (-150 µm), so the round arrives 300 µm of transport after the
	// operations of the first round
	int n_checked = 0;
	int in_second_round = 0;
	for (int m = 0; m < trace.n_moves; m++)
	{
		struct IonMove *move = &trace.moves[m];
		if (move->transport >= 0 && trace.transport_rounds[move->transport] == 2)
		{
			n_wrong += (fabs(move->depart - (transport_program->arrival_times[2] - 200.0 / TRANSPORT_SPEED_LIMIT)) > eps);
			n_checked++;
			in_second_round = 1;
		}

		// Moves back into storage before the transports of the second round
		if (move->transport < 0 && !in_second_round)
		{
			n_wrong += (move->depart <= transport_program->arrival_times[0]);
			n_wrong += (fabs(move->arrive - move->depart - 100.0 / TRANSPORT_SPEED_LIMIT) > eps);
			n_wrong += (fabs(move->arrive - (transport_program->arrival_times[2] - 200.0 / TRANSPORT_SPEED_LIMIT)) > eps);
			n_checked++;
		}
	}
	n_wrong += (n_checked != 4);

	free_schedule_trace(&trace);
	free_transport_program(transport_program);
	free_circuit_graph(circuit_graph);
	free_trap_graph(&trap_graph);

	return check_error("time_schedule_trace times every move", n_wrong, 0);
}
