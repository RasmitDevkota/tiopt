#include "circuits.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"

// Add the qubits referenced by a QASM operand to a list of target ions (which start at 1), where an operand without an
// index (e.g. "qreg") refers to every qubit; returns the new number of targets
int add_qasm_targets
(
	char *token,
	int n_ions,
	int *targets,
	int n_targets
)
{
	char *bracket = strchr(token, '[');

	int first = 0;
	int last = n_ions - 1;
	if (bracket != NULL)
	{
		first = atoi(bracket + 1);
		last = first;
	}

	for (int idx = first; idx <= last; idx++)
	{
		if (idx < 0 || idx >= n_ions)
		{
			printf("ERROR: Qubit index %d in QASM operand '%s' is out of range\n", idx, token);
			continue;
		}

		// Each qubit can only be targeted once by an operation
		int duplicate = 0;
		for (int t = 0; t < n_targets; t++)
			if (targets[t] == idx + 1)
				duplicate = 1;

		if (!duplicate)
			targets[n_targets++] = idx + 1;
	}

	return n_targets;
}

// Set the previous nodes of an instruction node to the most recent nodes acting on each of its targets, where
// last_node_idxs holds the index of the most recent node acting on each qubit (or -1 if there is none)
void set_prev_nodes
(
	struct CircuitGraph *circuit_graph,
	int *last_node_idxs,
	int *targets,
	int n_targets,
	struct InstructionNode *node
)
{
	node->n_prev_nodes = 0;
	node->prev_nodes = (n_targets > 0) ? malloc(n_targets * sizeof(struct InstructionNode)) : NULL;

	for (int t = 0; t < n_targets; t++)
	{
		int prev_idx = last_node_idxs[targets[t] - 1];

		if (prev_idx < 0)
			continue;

		// Several targets may share their previous node, e.g. for consecutive two-qubit gates on the same qubits
		int duplicate = 0;
		for (int p = 0; p < node->n_prev_nodes; p++)
			if (node->prev_nodes[p].graphindex == prev_idx + 1)
				duplicate = 1;

		if (!duplicate)
			node->prev_nodes[node->n_prev_nodes++] = circuit_graph->instruction_nodes[prev_idx];
	}
}

// Assign every instruction node to the earliest layer (starting at 1) after the layers of all of its previous nodes
void compute_circuit_layers
(
	struct CircuitGraph *circuit_graph
)
{
	for (int i = 0; i < circuit_graph->n_instruction_nodes; i++)
		circuit_graph->layer_idxs[i] = 1;

	// Relax the layers until they stop changing, which handles nodes in any order
	for (int it = 0; it < circuit_graph->n_instruction_nodes; it++)
	{
		int changed = 0;

		for (int i = 0; i < circuit_graph->n_instruction_nodes; i++)
		{
			struct InstructionNode *node = &circuit_graph->instruction_nodes[i];

			for (int p = 0; p < node->n_prev_nodes; p++)
			{
				int prev_idx = node->prev_nodes[p].graphindex - 1;

				if (circuit_graph->layer_idxs[prev_idx] + 1 > circuit_graph->layer_idxs[i])
				{
					circuit_graph->layer_idxs[i] = circuit_graph->layer_idxs[prev_idx] + 1;
					changed = 1;
				}
			}
		}

		if (!changed)
			break;
	}
}

// Duration of an operation, using defaults for operations whose duration is not set
static double operation_duration
(
	struct InstructionNode *node
)
{
	if (node->operation_duration > 0.0)
		return node->operation_duration;

	if (node->operation_type & DETECT)
		return DETECTION_DURATION;

	if (node->operation_type & COOL)
		return COOLING_DURATION;

	return GATE_DURATION;
}

// Schedule the arrival time of every transport of a transport program, where each layer of the circuit starts once the
// slowest transport into it (at TRANSPORT_SPEED_LIMIT) has finished, and lasts as long as its slowest operation
void schedule_arrival_times
(
	struct CircuitGraph *circuit_graph,
	struct TransportProgram *transport_program
)
{
	int n_ions = transport_program->n_ions;

	double (*ion_positions)[3] = malloc(n_ions * sizeof(double[3]));
	for (int i = 0; i < n_ions; i++)
		for (int k = 0; k < 3; k++)
			ion_positions[i][k] = transport_program->initial_positions[i][k];

	int max_layer = 0;
	for (int i = 0; i < circuit_graph->n_instruction_nodes; i++)
		max_layer = MAX(max_layer, circuit_graph->layer_idxs[i]);

	double time = 0.0;
	for (int layer = 1; layer <= max_layer; layer++)
	{
		double max_distance = 0.0;
		double max_duration = 0.0;

		for (int i = 0; i < circuit_graph->n_instruction_nodes; i++)
		{
			if (circuit_graph->layer_idxs[i] != layer)
				continue;

			for (int t = 0; t < transport_program->transport_n_targets_list[i]; t++)
			{
				double *position = ion_positions[transport_program->transport_targets_list[i][t] - 1];
				double *destination = transport_program->destinations[i];

				max_distance = MAX(max_distance, sqrt(pow(destination[0] - position[0], 2) + pow(destination[1] - position[1], 2) + pow(destination[2] - position[2], 2)));
			}

			max_duration = MAX(max_duration, operation_duration(&circuit_graph->instruction_nodes[i]));
		}

		time += max_distance / TRANSPORT_SPEED_LIMIT;

		for (int i = 0; i < circuit_graph->n_instruction_nodes; i++)
		{
			if (circuit_graph->layer_idxs[i] != layer)
				continue;

			transport_program->arrival_times[i] = time;

			for (int t = 0; t < transport_program->transport_n_targets_list[i]; t++)
				for (int k = 0; k < 3; k++)
					ion_positions[transport_program->transport_targets_list[i][t] - 1][k] = transport_program->destinations[i][k];
		}

		time += max_duration;
	}

	free(ion_positions);
}

// Generate a trap graph of zones evenly spaced along the trap axis at the given height, connected in a line
void generate_linear_trap_graph
(
	int n_zones,
	enum ZoneType *zone_types,
	int *zone_capacities,
	double x_min,
	double x_max,
	double height,
	struct TrapGraph *trap_graph
)
{
	trap_graph->n_zones = n_zones;
	trap_graph->n_transport_edges = MAX(n_zones - 1, 0);
	trap_graph->zone_nodes = calloc(n_zones, sizeof(struct ZoneNode));
	trap_graph->transport_edges = calloc(MAX(n_zones - 1, 1), sizeof(struct TransportEdge));

	for (int z = 0; z < n_zones; z++)
	{
		struct ZoneNode *zone = &trap_graph->zone_nodes[z];

		zone->zone_type = zone_types[z];
		zone->zone_capacity = zone_capacities[z];
		zone->zone_occupancy = 0;
		zone->position[0] = (n_zones > 1) ? x_min + (x_max - x_min) * z / (n_zones - 1) : 0.5 * (x_min + x_max);
		zone->position[1] = 0.0;
		zone->position[2] = height;
	}

	for (int t = 0; t < n_zones - 1; t++)
	{
		struct TransportEdge *edge = &trap_graph->transport_edges[t];

		edge->edge_type = LINEAR_SEGMENT;
		edge->n_zones = 2;
		edge->zone_nodes = malloc(2 * sizeof(struct ZoneNode));
		edge->zone_nodes[0] = trap_graph->zone_nodes[t];
		edge->zone_nodes[1] = trap_graph->zone_nodes[t + 1];
	}
}

// Free the zones and transport edges of a trap graph
void free_trap_graph
(
	struct TrapGraph *trap_graph
)
{
	for (int t = 0; t < trap_graph->n_transport_edges; t++)
		free(trap_graph->transport_edges[t].zone_nodes);

	free(trap_graph->transport_edges);
	free(trap_graph->zone_nodes);

	trap_graph->n_zones = 0;
	trap_graph->n_transport_edges = 0;
	trap_graph->zone_nodes = NULL;
	trap_graph->transport_edges = NULL;
}

// Free a circuit graph and its instruction nodes
void free_circuit_graph
(
	struct CircuitGraph *circuit_graph
)
{
	if (circuit_graph == NULL)
		return;

	for (int i = 0; i < circuit_graph->n_instruction_nodes; i++)
	{
		free(circuit_graph->instruction_nodes[i].target_ions);
		free(circuit_graph->instruction_nodes[i].prev_nodes);
	}

	free(circuit_graph->instruction_nodes);
	free(circuit_graph->layer_idxs);
	free(circuit_graph);
}

// Free a transport program and its arrays
void free_transport_program
(
	struct TransportProgram *transport_program
)
{
	if (transport_program == NULL)
		return;

	for (int t = 0; t < transport_program->n_transports; t++)
		free(transport_program->transport_targets_list[t]);

	free(transport_program->transport_targets_list);
	free(transport_program->transport_n_targets_list);
	free(transport_program->arrival_times);
	free(transport_program->destinations);
	free(transport_program->initial_positions);
	free(transport_program);
}
