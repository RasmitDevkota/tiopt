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
	transport_program = generate_transport_schedule(&trap_graph, circuit_graph, transport_program);

	if (transport_program == NULL)
	{
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

	// Transports in the same layer arrive together, and later layers arrive later
	for (int t = 1; t < MIN(transport_program->n_transports, 11); t++)
	{
		int same_layer = circuit_graph->layer_idxs[t] == circuit_graph->layer_idxs[t-1];

		if (same_layer)
			n_wrong += (fabs(transport_program->arrival_times[t] - transport_program->arrival_times[t-1]) > atol);
		else
			n_wrong += (transport_program->arrival_times[t] <= transport_program->arrival_times[t-1]);
	}

	// The first layer starts after the 200 µm transport from the load zone to the gate zone
	n_wrong += (fabs(transport_program->arrival_times[0] - 200.0 / TRANSPORT_SPEED_LIMIT) > atol);

	free_transport_program(transport_program);
	free_circuit_graph(circuit_graph);
	free_trap_graph(&trap_graph);

	return check_error("generate_transport_schedule fills transport program", n_wrong, 0);
}
