#include "circuits.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"
#include "transport.h"

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

// Sort nodes by their layers (which start at 1), keeping their order within each layer, so that each layer can be
// visited without scanning every node; allocates order (the indices of the nodes, layer by layer) and layer_starts (of
// length n_layers + 1, where layer l takes up order[layer_starts[l-1]] to order[layer_starts[l]-1]), and returns the
// number of layers
int sort_layers
(
	int n_nodes,
	const int *layer_idxs,
	int **order,
	int **layer_starts
)
{
	int n_layers = 0;
	for (int i = 0; i < n_nodes; i++)
		n_layers = MAX(n_layers, layer_idxs[i]);

	// Counting sort, which is stable
	*layer_starts = calloc(n_layers + 1, sizeof(int));
	for (int i = 0; i < n_nodes; i++)
		(*layer_starts)[layer_idxs[i]]++;

	for (int l = 1; l <= n_layers; l++)
		(*layer_starts)[l] += (*layer_starts)[l - 1];

	int *next = malloc((n_layers + 1) * sizeof(int));
	memcpy(next, *layer_starts, (n_layers + 1) * sizeof(int));

	*order = malloc(MAX(n_nodes, 1) * sizeof(int));
	for (int i = 0; i < n_nodes; i++)
		(*order)[next[layer_idxs[i] - 1]++] = i;

	free(next);

	return n_layers;
}

// Sort the instruction nodes of a circuit by layer (see sort_layers)
int sort_circuit_layers
(
	struct CircuitGraph *circuit_graph,
	int **order,
	int **layer_starts
)
{
	return sort_layers(circuit_graph->n_instruction_nodes, circuit_graph->layer_idxs, order, layer_starts);
}

// Schedule the arrival time of every transport of a transport program, where each layer of the circuit starts once the
// slowest transport into it (at TRANSPORT_SPEED_LIMIT) has finished, and lasts as long as its slowest operation
void schedule_arrival_times
(
	struct CircuitGraph *circuit_graph,
	struct TransportProgram *transport_program
)
{
	schedule_arrival_times_in_rounds(circuit_graph, transport_program, circuit_graph->layer_idxs);
}

// As schedule_arrival_times, but with the transports in the given rounds (which start at 1) instead of the layers of
// the circuit, as when a layer does not fit in the zones at once
void schedule_arrival_times_in_rounds
(
	struct CircuitGraph *circuit_graph,
	struct TransportProgram *transport_program,
	const int *rounds
)
{
	int n_ions = transport_program->n_ions;

	double (*ion_positions)[3] = malloc(n_ions * sizeof(double[3]));
	for (int i = 0; i < n_ions; i++)
		for (int k = 0; k < 3; k++)
			ion_positions[i][k] = transport_program->initial_positions[i][k];

	int *order;
	int *layer_starts;
	int max_layer = sort_layers(circuit_graph->n_instruction_nodes, rounds, &order, &layer_starts);

	double time = 0.0;
	for (int layer = 1; layer <= max_layer; layer++)
	{
		double max_distance = 0.0;
		double max_duration = 0.0;

		for (int o = layer_starts[layer - 1]; o < layer_starts[layer]; o++)
		{
			int i = order[o];

			for (int t = 0; t < transport_program->transport_n_targets_list[i]; t++)
			{
				double *position = ion_positions[transport_program->transport_targets_list[i][t] - 1];
				double *destination = transport_program->destinations[i];

				max_distance = MAX(max_distance, sqrt(pow(destination[0] - position[0], 2) + pow(destination[1] - position[1], 2) + pow(destination[2] - position[2], 2)));
			}

			max_duration = MAX(max_duration, operation_duration(&circuit_graph->instruction_nodes[i]));
		}

		time += max_distance / TRANSPORT_SPEED_LIMIT;

		for (int o = layer_starts[layer - 1]; o < layer_starts[layer]; o++)
		{
			int i = order[o];

			transport_program->arrival_times[i] = time;

			for (int t = 0; t < transport_program->transport_n_targets_list[i]; t++)
				for (int k = 0; k < 3; k++)
					ion_positions[transport_program->transport_targets_list[i][t] - 1][k] = transport_program->destinations[i][k];
		}

		time += max_duration;
	}

	free(order);
	free(layer_starts);
	free(ion_positions);
}

// Distance between the centers of two zones of a trap graph (µm)
static double zone_distance
(
	struct TrapGraph *trap_graph,
	int from_zone,
	int to_zone
)
{
	double *a = trap_graph->zone_nodes[from_zone].position;
	double *b = trap_graph->zone_nodes[to_zone].position;

	return sqrt(pow(b[0] - a[0], 2) + pow(b[1] - a[1], 2) + pow(b[2] - a[2], 2));
}

// Time every move of a schedule trace (whose transport_rounds and n_rounds are set) and the arrival time of every
// transport of its program, at TRANSPORT_SPEED_LIMIT: each round first moves ions back into storage (an ion's moves one
// after another, different ions at once), then moves the ions of its operations into their zones (all at once, from
// where they actually are), and then runs its operations, which all start once the slowest transport has arrived and
// last as long as the slowest one. A move back into storage belongs to the round of the next transport after it in the
// trace, and those after the last transport happen once the last round is done
void time_schedule_trace
(
	struct TrapGraph *trap_graph,
	struct CircuitGraph *circuit_graph,
	struct TransportProgram *transport_program,
	struct ScheduleTrace *trace
)
{
	int n_ions = transport_program->n_ions;
	int n_rounds = trace->n_rounds;
	int *rounds = trace->transport_rounds;

	int n_moves = trace->n_moves;
	int n_nodes = circuit_graph->n_instruction_nodes;

	// Round of every move
	int *move_rounds = malloc(MAX(n_moves, 1) * sizeof(int));
	int next_round = n_rounds + 1;
	for (int m = n_moves - 1; m >= 0; m--)
	{
		if (trace->moves[m].transport >= 0)
			next_round = rounds[trace->moves[m].transport];

		move_rounds[m] = next_round;
	}

	// Moves and instructions of each round, in order
	int *move_starts = calloc(n_rounds + 3, sizeof(int));
	int *node_starts = calloc(n_rounds + 3, sizeof(int));
	for (int m = 0; m < n_moves; m++)
		move_starts[move_rounds[m] + 1]++;
	for (int i = 0; i < n_nodes; i++)
		node_starts[rounds[i] + 1]++;
	for (int r = 1; r <= n_rounds + 2; r++)
	{
		move_starts[r] += move_starts[r - 1];
		node_starts[r] += node_starts[r - 1];
	}

	int *round_moves = malloc(MAX(n_moves, 1) * sizeof(int));
	int *round_nodes = malloc(MAX(n_nodes, 1) * sizeof(int));
	int *next = malloc((n_rounds + 2) * sizeof(int));
	memcpy(next, move_starts, (n_rounds + 2) * sizeof(int));
	for (int m = 0; m < n_moves; m++)
		round_moves[next[move_rounds[m]]++] = m;
	memcpy(next, node_starts, (n_rounds + 2) * sizeof(int));
	for (int i = 0; i < n_nodes; i++)
		round_nodes[next[rounds[i]]++] = i;

	// When each ion is free to make its next move back into storage in the round it last moved in
	double *ion_times = malloc(MAX(n_ions, 1) * sizeof(double));
	int *ion_rounds = calloc(MAX(n_ions, 1), sizeof(int));

	double time = 0.0;
	for (int round = 1; round <= n_rounds + 1; round++)
	{
		// Moves back into storage
		double transport_start = time;
		for (int k = move_starts[round]; k < move_starts[round + 1]; k++)
		{
			struct IonMove *move = &trace->moves[round_moves[k]];
			if (move->transport >= 0)
				continue;

			int ion = move->ion - 1;
			if (ion_rounds[ion] != round)
			{
				ion_rounds[ion] = round;
				ion_times[ion] = time;
			}

			move->depart = ion_times[ion];
			move->arrive = move->depart + zone_distance(trap_graph, move->from_zone, move->to_zone) / TRANSPORT_SPEED_LIMIT;
			ion_times[ion] = move->arrive;
			transport_start = MAX(transport_start, move->arrive);
		}

		// Moves into the zones of the operations
		double max_transport = 0.0;
		for (int k = move_starts[round]; k < move_starts[round + 1]; k++)
		{
			struct IonMove *move = &trace->moves[round_moves[k]];
			if (move->transport < 0)
				continue;

			double duration = zone_distance(trap_graph, move->from_zone, move->to_zone) / TRANSPORT_SPEED_LIMIT;
			move->depart = transport_start;
			move->arrive = transport_start + duration;
			max_transport = MAX(max_transport, duration);
		}

		// Operations
		double max_duration = 0.0;
		for (int k = node_starts[round]; k < node_starts[round + 1]; k++)
		{
			int i = round_nodes[k];

			transport_program->arrival_times[i] = transport_start + max_transport;
			max_duration = MAX(max_duration, operation_duration(&circuit_graph->instruction_nodes[i]));
		}

		time = transport_start + max_transport + max_duration;
	}

	free(ion_rounds);
	free(ion_times);
	free(next);
	free(round_nodes);
	free(round_moves);
	free(node_starts);
	free(move_starts);
	free(move_rounds);
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

// Find the zone of a trap graph with any of the given types which has room for the target ions of an operation and
// lies nearest to them (counting targets already in a zone as making room in it), or NULL if there is none; ion_states
// holds the zone of each ion, and targets start at 1
struct ZoneNode* select_operation_zone
(
	struct TrapGraph *trap_graph,
	enum ZoneType zone_type,
	struct ZoneNode **ion_states,
	int *targets,
	int n_targets
)
{
	// Centroid of the target ions
	double centroid[3] = { 0.0, 0.0, 0.0 };
	for (int t = 0; t < n_targets; t++)
		for (int k = 0; k < 3; k++)
			centroid[k] += ion_states[targets[t] - 1]->position[k] / n_targets;

	struct ZoneNode *nearest = NULL;
	double nearest_distance = HUGE_VAL;
	for (int z = 0; z < trap_graph->n_zones; z++)
	{
		struct ZoneNode *zone = &trap_graph->zone_nodes[z];

		if (!(zone->zone_type & zone_type))
			continue;

		int n_present = 0;
		for (int t = 0; t < n_targets; t++)
			n_present += (ion_states[targets[t] - 1] == zone);

		if (zone->zone_occupancy - n_present + n_targets > zone->zone_capacity)
			continue;

		// Squared distances, which order the zones the same way
		double dx = zone->position[0] - centroid[0];
		double dy = zone->position[1] - centroid[1];
		double dz = zone->position[2] - centroid[2];
		double distance = dx * dx + dy * dy + dz * dz;

		if (distance < nearest_distance)
		{
			nearest = zone;
			nearest_distance = distance;
		}
	}

	return nearest;
}

// Move every ion which waits in a zone where operations happen (a gate, measurement or cooling zone which is not also
// a storage zone) into the nearest storage zone with room, or else the nearest loading or empty zone with room, as
// after a layer of a circuit; ions with nowhere to go stay where they are. ion_states holds the zone of each ion and is
// updated
void release_operation_zones
(
	struct TrapGraph *trap_graph,
	struct ZoneNode **ion_states,
	int n_ions
)
{
	release_ions(trap_graph, ion_states, n_ions, NULL, NULL);
}

// Record a move of an ion in a schedule trace, if there is one
void record_ion_move
(
	struct ScheduleTrace *trace,
	struct TrapGraph *trap_graph,
	int ion,
	struct ZoneNode *from,
	struct ZoneNode *to,
	int transport
)
{
	if (trace == NULL)
		return;

	if (trace->n_moves == trace->max_moves)
	{
		trace->max_moves = MAX(2 * trace->max_moves, 64);
		trace->moves = realloc(trace->moves, trace->max_moves * sizeof(struct IonMove));
	}

	trace->moves[trace->n_moves++] = (struct IonMove) { ion, (int) (from - trap_graph->zone_nodes), (int) (to - trap_graph->zone_nodes), transport, 0.0, 0.0 };
}

// As release_operation_zones, but leaving the ions for which keep is set (if keep is not NULL) where they are, and
// recording the moves in trace (if it is not NULL)
void release_ions
(
	struct TrapGraph *trap_graph,
	struct ZoneNode **ion_states,
	int n_ions,
	const int *keep,
	struct ScheduleTrace *trace
)
{
	enum ZoneType destinations[3] = { STORAGE_ZONE, LOAD_ZONE, EMPTY_ZONE };

	for (int i = 0; i < n_ions; i++)
	{
		struct ZoneNode *zone = ion_states[i];

		if (!(zone->zone_type & (GATE_ZONE | MEASURE_ZONE | COOL_ZONE)) || (zone->zone_type & STORAGE_ZONE))
			continue;

		if (keep != NULL && keep[i])
			continue;

		int target = i + 1;
		for (int d = 0; d < 3; d++)
		{
			struct ZoneNode *destination = select_operation_zone(trap_graph, destinations[d], ion_states, &target, 1);

			if (destination != NULL && destination != zone)
			{
				record_ion_move(trace, trap_graph, target, zone, destination, -1);
				zone->zone_occupancy--;
				destination->zone_occupancy++;
				ion_states[i] = destination;
				break;
			}
		}
	}
}
