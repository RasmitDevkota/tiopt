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

// Layers of the diagonal operations of a batch placed one way (see compute_commuting_layers): 0 at the earliest layer
// after the last operations on their qubits in order, 1 at the earliest free layer after the last non-diagonal
// operations on their qubits in order, and 2 like 1 but with the operations on the busiest qubits first. Fills layers
// with the layer of each operation of the batch and used with the layers each qubit is busy in, and returns the
// highest layer of the batch, with the sum of the last layers of its qubits in sum
static int place_diagonal_batch
(
	struct CircuitGraph *circuit_graph,
	const int *batch,
	int n_batch,
	int method,
	const int *ready,
	const int *last,
	int n_slots,
	char *used,
	int *layers,
	long *sum
)
{
	int n_ions = circuit_graph->n_ions;

	// Order of the operations, by how busy their busiest qubit is in the batch (and then in order) for method 2
	int *order = malloc(n_batch * sizeof(int));
	int *degree = calloc(n_ions, sizeof(int));
	int *key = malloc(n_batch * sizeof(int));
	for (int k = 0; k < n_batch; k++)
	{
		struct InstructionNode *node = &circuit_graph->instruction_nodes[batch[k]];
		for (int t = 0; t < node->n_target_ions; t++)
			degree[node->target_ions[t] - 1]++;
	}
	for (int k = 0; k < n_batch; k++)
	{
		struct InstructionNode *node = &circuit_graph->instruction_nodes[batch[k]];
		key[k] = 0;
		for (int t = 0; t < node->n_target_ions; t++)
			key[k] = MAX(key[k], degree[node->target_ions[t] - 1]);

		int j = k;
		while (method == 2 && j > 0 && key[order[j - 1]] < key[k])
		{
			order[j] = order[j - 1];
			j--;
		}
		order[j] = k;
	}

	int *qubit_last = malloc(n_ions * sizeof(int));
	memcpy(qubit_last, last, n_ions * sizeof(int));

	int highest = 0;
	for (int o = 0; o < n_batch; o++)
	{
		int k = order[o];
		struct InstructionNode *node = &circuit_graph->instruction_nodes[batch[k]];

		int layer = 1;
		for (int t = 0; t < node->n_target_ions; t++)
		{
			int q = node->target_ions[t] - 1;
			layer = MAX(layer, ((method == 0) ? qubit_last[q] : ready[q]) + 1);
		}

		// The earliest layer in which none of the qubits is busy
		for (int busy = 1; busy && method != 0; )
		{
			busy = 0;
			for (int t = 0; t < node->n_target_ions; t++)
				busy |= used[(node->target_ions[t] - 1) * n_slots + layer];
			layer += busy;
		}

		layers[k] = layer;
		highest = MAX(highest, layer);
		for (int t = 0; t < node->n_target_ions; t++)
		{
			int q = node->target_ions[t] - 1;
			used[q * n_slots + layer] = 1;
			qubit_last[q] = MAX(qubit_last[q], layer);
		}
	}

	*sum = 0;
	for (int q = 0; q < n_ions; q++)
		*sum += (degree[q] > 0) ? qubit_last[q] : 0;

	free(qubit_last);
	free(key);
	free(degree);
	free(order);

	return highest;
}

// Assign every instruction node to a layer like compute_circuit_layers, but letting operations which are diagonal in the
// computational basis (which all commute with each other) change places, to fit them into fewer layers: diagonal
// operations wait in a batch until a non-diagonal operation (or the end of the circuit) comes, and then each batch is
// placed in whichever of three ways ends it soonest (see place_diagonal_batch), where none of them puts two operations
// on the same qubit in the same layer, or a diagonal operation before the last non-diagonal operation on its qubits
void compute_commuting_layers
(
	struct CircuitGraph *circuit_graph
)
{
	int n_nodes = circuit_graph->n_instruction_nodes;
	int n_ions = MAX(circuit_graph->n_ions, 1);
	int n_slots = n_nodes + 2;

	int *ready = calloc(n_ions, sizeof(int)); // layer of the last non-diagonal operation on each qubit
	int *last = calloc(n_ions, sizeof(int)); // last layer each qubit is busy in
	char *used = calloc((size_t) n_ions * n_slots, sizeof(char)); // layers each qubit is busy in with diagonal operations
	char *trial = malloc((size_t) n_ions * n_slots * sizeof(char));
	int *batch = malloc(MAX(n_nodes, 1) * sizeof(int));
	int *layers = malloc(MAX(n_nodes, 1) * sizeof(int));
	int *best_layers = malloc(MAX(n_nodes, 1) * sizeof(int));
	int n_batch = 0;

	for (int i = 0; i <= n_nodes; i++)
	{
		struct InstructionNode *node = (i < n_nodes) ? &circuit_graph->instruction_nodes[i] : NULL;

		if (node != NULL && node->diagonal)
		{
			batch[n_batch++] = i;
			continue;
		}

		// Place the waiting diagonal operations
		if (n_batch > 0)
		{
			int best_method = -1;
			int best_highest = 0;
			long best_sum = 0;
			for (int method = 0; method < 3; method++)
			{
				memcpy(trial, used, (size_t) n_ions * n_slots);
				long sum;
				int highest = place_diagonal_batch(
					circuit_graph,
					batch, n_batch,
					method,
					ready, last,
					n_slots, trial,
					layers, &sum
				);

				if (best_method < 0 || highest < best_highest || (highest == best_highest && sum < best_sum))
				{
					best_method = method;
					best_highest = highest;
					best_sum = sum;
					memcpy(best_layers, layers, n_batch * sizeof(int));
				}
			}

			for (int k = 0; k < n_batch; k++)
			{
				struct InstructionNode *batch_node = &circuit_graph->instruction_nodes[batch[k]];
				circuit_graph->layer_idxs[batch[k]] = best_layers[k];

				for (int t = 0; t < batch_node->n_target_ions; t++)
				{
					int q = batch_node->target_ions[t] - 1;
					used[q * n_slots + best_layers[k]] = 1;
					last[q] = MAX(last[q], best_layers[k]);
				}
			}

			n_batch = 0;
		}

		if (node == NULL)
			break;

		// A non-diagonal operation follows everything on its qubits
		int layer = 1;
		for (int t = 0; t < node->n_target_ions; t++)
			layer = MAX(layer, last[node->target_ions[t] - 1] + 1);

		circuit_graph->layer_idxs[i] = layer;
		for (int t = 0; t < node->n_target_ions; t++)
		{
			int q = node->target_ions[t] - 1;
			ready[q] = layer;
			last[q] = layer;
		}
	}

	free(best_layers);
	free(layers);
	free(batch);
	free(trial);
	free(used);
	free(last);
	free(ready);
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

// Cost model of time_schedule_trace when it is given none: the defaults of tiopt.qccd.TimingModel and
// tiopt.qccd.NoiseModel (from P. Murali et al., ISCA 2020), in s rather than µs. A split or merge takes 80 µs, crossing
// a Y or X junction 100 or 120 µs, and ions move at 20 µm/µs; two-qubit gates follow the FM gate model, and ions reach
// the end of their chain with a SWAP gate or physical swaps, whichever costs less fidelity, where gates can be done
// (and with physical swaps elsewhere). Splits, merges and junction crossings add 0.1 quanta, every 100 µm moved 0.01,
// and every chain and moving ion heats by 100 quanta/s; a two-qubit gate fails with probability 1 µs^-1 times its
// duration plus A (2 n_bar + 1), where A = 1E-4 N / ln N
struct TransportCostModel default_transport_cost_model
(
)
{
	struct TransportCostModel cost_model = { 0 };

	cost_model.speed = 20.0E6;
	cost_model.min_move = 1E-6;
	cost_model.split = 80E-6;
	cost_model.merge = 80E-6;
	cost_model.junction_y = 100E-6;
	cost_model.junction_x = 120E-6;
	cost_model.rotation = 42E-6;
	cost_model.swap_gates = SWAP_CHEAPER;
	cost_model.gate_model = GATE_MODEL_FM;
	cost_model.two_qubit = GATE_DURATION;
	cost_model.single_qubit = 5E-6;
	cost_model.measurement = 100E-6;
	cost_model.cooling = 200E-6;
	cost_model.split_heating = 0.1;
	cost_model.merge_heating = 0.1;
	cost_model.junction_heating = 0.1;
	cost_model.move_heating = 0.01;
	cost_model.background_heating = 1E-4 * 1E6;
	cost_model.gate_background_error = 1E-6 * 1E6;
	cost_model.motional_error = 1E-4;
	cost_model.single_qubit_error = 1E-4;
	cost_model.measurement_error = 1E-3;
	cost_model.cooled_n_bar = 0.05;

	return cost_model;
}

// Cost model which only counts the distance ions move: they move at TRANSPORT_SPEED_LIMIT, splits, merges, swaps and
// junction crossings take no time, operations take GATE_DURATION, DETECTION_DURATION and COOLING_DURATION (as the
// scheduler used to assume), and nothing heats or fails
struct TransportCostModel distance_transport_cost_model
(
)
{
	struct TransportCostModel cost_model = { 0 };

	cost_model.speed = TRANSPORT_SPEED_LIMIT;
	cost_model.gate_model = GATE_MODEL_FIXED;
	cost_model.two_qubit = GATE_DURATION;
	cost_model.single_qubit = GATE_DURATION;
	cost_model.measurement = DETECTION_DURATION;
	cost_model.cooling = COOLING_DURATION;

	return cost_model;
}

// Duration of a two-qubit gate between ions separation apart in a chain of n_ions (s), in the gate model of a cost model
// (as tiopt.gates.gate_time_model)
double two_qubit_gate_time
(
	const struct TransportCostModel *cost_model,
	int n_ions,
	int separation
)
{
	int d = MAX(separation, 1);

	switch (cost_model->gate_model)
	{
		case GATE_MODEL_AM1:
			return 1E-6 * MAX(100.0 * d - 22.0, 1.0);
		case GATE_MODEL_AM2:
			return 1E-6 * MAX(38.0 * d + 10.0, 1.0);
		case GATE_MODEL_PM:
			return 1E-6 * MAX(5.0 * d + 160.0, 1.0);
		case GATE_MODEL_FM:
			return 1E-6 * MAX(13.33 * n_ions - 54.0, 100.0);
		default:
			return cost_model->two_qubit;
	}
}

// Chains of ions in the zones of a trap graph, and when each zone, segment and ion is next free, while the moves of a
// schedule are timed
struct ShuttleState
{
	struct TrapGraph *trap_graph;
	const struct TransportCostModel *cost_model;
	struct ScheduleTrace *trace;
	int n_zones;
	int n_ions;
	int *chains; // ions in each zone, from the end which faces its first neighbor; implicitly, length n_zones-by-(n_ions+1)
	int *chain_lengths; // implicitly, length n_zones
	int *first_neighbors; // neighbor of each zone with the lowest index, which the front of its chain faces (-1 if none)
	int *degrees; // number of neighbors of each zone
	double *zone_free; // time from which each zone is free (s)
	double *segment_free; // time from which each segment is free (s); implicitly, length n_zones-by-n_zones
	double *ion_free; // time from which each ion is free (s)
	int *path; // implicitly, length n_zones
	double *energies; // mean phonon number of the chain in each zone
	double *energy_times; // time up to which each chain has been heated (s)
	double *carried; // mean phonon number which each ion carries out of a chain
	double *carried_times; // time up to which each ion out of a chain has been heated (s)
	double fidelity;
	double max_n_bar;
	int *later_gates; // two-qubit (Mølmer-Sørensen) gates which each ion has yet to take part in
};

// Background heating of the chain in a zone up to a time
static void heat_chain
(
	struct ShuttleState *state,
	int zone,
	double t
)
{
	state->energies[zone] += state->cost_model->background_heating * MAX(t - state->energy_times[zone], 0.0);
	state->energy_times[zone] = MAX(t, state->energy_times[zone]);
}

// Background heating of an ion out of a chain up to a time
static void heat_ion
(
	struct ShuttleState *state,
	int ion,
	double t
)
{
	state->carried[ion - 1] += state->cost_model->background_heating * MAX(t - state->carried_times[ion - 1], 0.0);
	state->carried_times[ion - 1] = MAX(t, state->carried_times[ion - 1]);
}

// Fidelity of a two-qubit gate of a duration in the chain of a zone, as heated up to its start
static double two_qubit_gate_fidelity
(
	struct ShuttleState *state,
	int zone,
	double start,
	double duration
)
{
	const struct TransportCostModel *cost_model = state->cost_model;

	heat_chain(state, zone, start);

	double n = MAX(state->chain_lengths[zone], 2);
	double n_bar = state->energies[zone];
	state->max_n_bar = MAX(state->max_n_bar, n_bar);

	return MAX(1.0 - cost_model->gate_background_error * duration - cost_model->motional_error * n / log(n) * (2.0 * n_bar + 1.0), 0.0);
}

// Record a step of a schedule, returning when it ends
static double add_transport_step
(
	struct ShuttleState *state,
	enum TransportStepType type,
	int ion,
	int other_ion,
	int zone,
	int to_zone,
	int index,
	double start,
	double duration
)
{
	struct ScheduleTrace *trace = state->trace;

	if (trace->n_steps == trace->max_steps)
	{
		trace->max_steps = MAX(2 * trace->max_steps, 256);
		trace->steps = realloc(trace->steps, trace->max_steps * sizeof(struct TransportStep));
	}

	trace->steps[trace->n_steps++] = (struct TransportStep) { type, ion, other_ion, zone, to_zone, index, start, start + duration };

	return start + duration;
}

// End of the chain in a zone (0 for the front and 1 for the back) which faces one of its neighbors
static int chain_end_facing
(
	struct ShuttleState *state,
	int zone,
	int neighbor
)
{
	return (neighbor == state->first_neighbors[zone]) ? 0 : 1;
}

static int chain_position
(
	struct ShuttleState *state,
	int zone,
	int ion
)
{
	int *chain = &state->chains[zone * (state->n_ions + 1)];
	for (int k = 0; k < state->chain_lengths[zone]; k++)
		if (chain[k] == ion)
			return k;

	return -1;
}

static void insert_into_chain
(
	struct ShuttleState *state,
	int zone,
	int ion,
	int end
)
{
	int *chain = &state->chains[zone * (state->n_ions + 1)];
	int n = state->chain_lengths[zone]++;

	if (end == 0)
	{
		memmove(&chain[1], &chain[0], n * sizeof(int));
		chain[0] = ion;
	}
	else
		chain[n] = ion;
}

static void remove_from_chain
(
	struct ShuttleState *state,
	int zone,
	int ion
)
{
	int *chain = &state->chains[zone * (state->n_ions + 1)];
	int k = chain_position(state, zone, ion);
	if (k < 0)
		return;

	memmove(&chain[k], &chain[k + 1], (state->chain_lengths[zone] - k - 1) * sizeof(int));
	state->chain_lengths[zone]--;
}

// Whether bringing an ion from one place in the chain of a gate zone to another costs less fidelity with a SWAP gate
// (three two-qubit gates) than by physically swapping it with each ion on the way. The SWAP gate fails now (at the mean
// phonon number of the chain), while the quanta which the physical swaps add (and the background heating while either
// happens) stay in the chain and lower the fidelity of every later two-qubit gate of its ions, counted as if the chain
// stayed together for them (in a chain of two, which shares its quanta with both of its ions, this is exact)
static int swap_gate_is_cheaper
(
	struct ShuttleState *state,
	int zone,
	int position,
	int target,
	double t
)
{
	const struct TransportCostModel *cost_model = state->cost_model;
	int *chain = &state->chains[zone * (state->n_ions + 1)];
	int n = state->chain_lengths[zone];
	int distance = abs(target - position);

	// Infidelity of the SWAP gate
	double n_bar = state->energies[zone] + cost_model->background_heating * MAX(t - state->energy_times[zone], 0.0);
	double gate = two_qubit_gate_time(cost_model, n, distance);
	double size = MAX(n, 2);
	double A = cost_model->motional_error * size / log(size);
	double error = cost_model->gate_background_error * gate + A * (2.0 * n_bar + 1.0);
	double gate_cost = 1.0 - pow(MAX(1.0 - error, 0.0), 3);

	// Infidelity which each quantum added to the chain causes later, in the later gates of its ions (each between two
	// of them)
	int later = 0;
	for (int k = 0; k < n; k++)
		later += state->later_gates[chain[k] - 1];

	double per_quantum = 2.0 * A * 0.5 * later;

	double swap_time = cost_model->split + cost_model->rotation + cost_model->merge;
	double gate_heat = cost_model->background_heating * 3 * gate;
	double swap_heat = cost_model->split_heating + cost_model->merge_heating;
	swap_heat += cost_model->background_heating * swap_time;
	double physical_heat = distance * swap_heat;

	return gate_cost + gate_heat * per_quantum < physical_heat * per_quantum;
}

// Bring an ion to one end of the chain in its zone (0 for the front and 1 for the back), returning when it is there:
// with a SWAP gate with the ion at that end in a zone where gates can be done (if the cost model uses them), and
// otherwise by swapping places with each neighbor on the way (a split, a rotation and a merge each, which heat the
// chain)
static double reorder_to_end
(
	struct ShuttleState *state,
	int ion,
	int zone,
	int end,
	int index,
	double t
)
{
	const struct TransportCostModel *cost_model = state->cost_model;
	int *chain = &state->chains[zone * (state->n_ions + 1)];
	int position = chain_position(state, zone, ion);
	int target = (end == 0) ? 0 : state->chain_lengths[zone] - 1;
	int step = (target > position) ? 1 : -1;

	if (position == target)
		return t;

	int gate_swap = 0;
	if (state->trap_graph->zone_nodes[zone].zone_type & GATE_ZONE)
	{
		if (cost_model->swap_gates == SWAP_CHEAPER)
			gate_swap = swap_gate_is_cheaper(state, zone, position, target, t);
		else
			gate_swap = (cost_model->swap_gates == SWAP_GATES);
	}

	if (gate_swap)
	{
		// Three Mølmer-Sørensen gates exchange the states of the ion and the one at the end
		int other = chain[target];
		double gate = two_qubit_gate_time(cost_model, state->chain_lengths[zone], abs(target - position));
		state->fidelity *= pow(two_qubit_gate_fidelity(state, zone, t, gate), 3);
		t = add_transport_step(state, STEP_GATE_SWAP, ion, other, zone, zone, index, t, 3 * gate);

		chain[target] = ion;
		chain[position] = other;
		state->ion_free[other - 1] = MAX(state->ion_free[other - 1], t);

		return t;
	}

	heat_chain(state, zone, t);
	state->energies[zone] += abs(target - position) * (cost_model->split_heating + cost_model->merge_heating);

	while (position != target)
	{
		int neighbor = chain[position + step];
		t = add_transport_step(
			state,
			STEP_ION_SWAP,
			ion, neighbor,
			zone, zone,
			index,
			t, cost_model->split + cost_model->rotation + cost_model->merge
		);

		chain[position + step] = ion;
		chain[position] = neighbor;
		position += step;
	}

	return t;
}

// Split an ion off the end of the chain in a zone which faces a neighbor (it takes its share of the motional energy of
// the chain, and the split heats both), or let it go if it is alone, returning when it is free
static double split_off
(
	struct ShuttleState *state,
	int ion,
	int zone,
	int toward,
	int index,
	double t
)
{
	const struct TransportCostModel *cost_model = state->cost_model;
	int n = state->chain_lengths[zone];

	heat_chain(state, zone, t);
	double share = state->energies[zone] / MAX(n, 1);
	state->energies[zone] -= share;
	state->carried[ion - 1] = share;
	state->carried_times[ion - 1] = t;

	if (n > 1)
	{
		t = add_transport_step(state, STEP_SPLIT, ion, 0, zone, toward, index, t, cost_model->split);
		state->energies[zone] += cost_model->split_heating;
		state->carried[ion - 1] += cost_model->split_heating;
	}

	// The ion heats on its own from when it is free
	state->carried_times[ion - 1] = t;
	remove_from_chain(state, zone, ion);

	return t;
}

// Merge an ion into the end of the chain in a zone which faces the neighbor it comes from (adding the motional energy it
// carries, and the heating of the merge), or let it stay there if the zone is empty, returning when it is done
static double merge_into
(
	struct ShuttleState *state,
	int ion,
	int zone,
	int from,
	int index,
	double t
)
{
	const struct TransportCostModel *cost_model = state->cost_model;

	heat_ion(state, ion, t);

	if (state->chain_lengths[zone] > 0)
	{
		heat_chain(state, zone, t);
		t = add_transport_step(state, STEP_MERGE, ion, 0, zone, from, index, t, cost_model->merge);
		state->energies[zone] += state->carried[ion - 1] + cost_model->merge_heating;
	}
	else
	{
		state->energies[zone] = state->carried[ion - 1];
		state->energy_times[zone] = t;
	}

	state->carried[ion - 1] = 0.0;
	insert_into_chain(state, zone, ion, chain_end_facing(state, zone, from));

	return t;
}

// Time of a move along a segment
static double segment_move_time
(
	struct ShuttleState *state,
	int from_zone,
	int to_zone
)
{
	double length = state->trap_graph->segment_lengths[from_zone * state->n_zones + to_zone];

	return MAX(length / state->cost_model->speed, state->cost_model->min_move);
}

static int is_junction_zone
(
	struct ShuttleState *state,
	int zone
)
{
	return (state->trap_graph->zone_nodes[zone].zone_type & JUNCTION_ZONE) != 0;
}

// Carry out a move of a schedule trace along the shortest path between its zones, no earlier than a time, returning 1
// if there is no path. The ion first comes to the end of the chain in its zone which faces the way it goes, and splits
// off it if other ions are there; then it moves along each segment of the path, crosses each junction, and passes
// through the chain of every other zone which holds ions (merging into it, swapping its way to the far end, and
// splitting off again), and finally merges into the chain in its destination. One ion at a time moves along a segment,
// and an ion takes up the zone it moves toward from when it sets off until it leaves it (or has merged into the chain
// there), so that moving ions are always at least a segment apart and never pass each other
static int shuttle_ion
(
	struct ShuttleState *state,
	int m,
	double start
)
{
	struct IonMove *move = &state->trace->moves[m];
	const struct TransportCostModel *cost_model = state->cost_model;
	int n = state->n_zones;
	int ion = move->ion;
	int from = move->from_zone;
	int to = move->to_zone;

	double t = MAX(start, state->ion_free[ion - 1]);

	if (from == to)
	{
		move->depart = move->arrive = t;
		state->ion_free[ion - 1] = t;
		return 0;
	}

	int n_path = trap_graph_route(state->trap_graph, from, to, state->path);
	if (n_path < 2)
	{
		printf("ERROR: No path between zones %d and %d for ion %d\n", from + 1, to + 1, ion);
		return 1;
	}

	int *path = state->path;

	if (chain_position(state, from, ion) < 0)
	{
		printf("ERROR: Ion %d is not in zone %d when it moves from there\n", ion, from + 1);
		return 1;
	}

	// Leave the chain in the first zone
	t = MAX(t, state->zone_free[from]);
	move->depart = t;

	if (state->chain_lengths[from] > 1)
		t = reorder_to_end(state, ion, from, chain_end_facing(state, from, path[1]), m, t);
	t = split_off(state, ion, from, path[1], m, t);

	for (int k = 1; k < n_path; k++)
	{
		int a = path[k - 1];
		int b = path[k];

		// Set off once the segment and the zone it leads to are free, giving up the zone it leaves
		double s = MAX(t, MAX(state->segment_free[a * n + b], state->zone_free[b]));
		state->zone_free[a] = MAX(state->zone_free[a], s);

		// The ion heats while it waits and moves, and by the distance it moves
		heat_ion(state, ion, s);
		t = add_transport_step(state, STEP_MOVE, ion, 0, a, b, m, s, segment_move_time(state, a, b));
		state->segment_free[a * n + b] = state->segment_free[b * n + a] = t;
		heat_ion(state, ion, t);
		state->carried[ion - 1] += cost_model->move_heating * state->trap_graph->segment_lengths[a * n + b] / 100.0;

		if (is_junction_zone(state, b))
		{
			if (k == n_path - 1)
			{
				printf("ERROR: Ion %d cannot stay in junction zone %d\n", ion, b + 1);
				return 1;
			}

			if (state->degrees[b] >= 3)
			{
				t = add_transport_step(
					state,
					STEP_JUNCTION,
					ion, 0,
					b, b,
					m,
					t, (state->degrees[b] == 3) ? cost_model->junction_y : cost_model->junction_x
				);
				state->carried[ion - 1] += cost_model->junction_heating;
			}

			continue;
		}

		int occupied = (state->chain_lengths[b] > 0);

		if (k == n_path - 1)
		{
			// Merge into the chain in the destination
			t = merge_into(state, ion, b, a, m, t);
			state->zone_free[b] = MAX(state->zone_free[b], t);
		}
		else if (occupied)
		{
			// Pass through the chain in this zone
			int next = path[k + 1];

			t = merge_into(state, ion, b, a, m, t);
			t = reorder_to_end(state, ion, b, chain_end_facing(state, b, next), m, t);
			t = split_off(state, ion, b, next, m, t);
		}
	}

	move->arrive = t;
	state->ion_free[ion - 1] = t;

	return 0;
}

// Duration of an operation which starts at a time on the chain in a zone (s), multiplying the fidelity of the schedule by
// its own: a gate on several ions takes its Mølmer-Sørensen gates one after another, each lasting as long as the gate
// model gives for the chain and how far apart its ions are in it, and failing as two_qubit_gate_fidelity gives; a
// single-qubit gate takes single_qubit (or no time if it is virtual), a measurement measurement for every ion, and
// cooling brings the chain to cooled_n_bar. An operation whose duration is set takes that long
static double run_operation
(
	struct ShuttleState *state,
	struct InstructionNode *node,
	int zone,
	double start
)
{
	const struct TransportCostModel *cost_model = state->cost_model;
	double duration;

	if ((node->operation_type & QUBIT) && node->n_target_ions >= 2)
	{
		int first = state->n_ions;
		int last = 0;
		for (int t = 0; t < node->n_target_ions; t++)
		{
			int position = chain_position(state, zone, node->target_ions[t]);
			first = MIN(first, position);
			last = MAX(last, position);
		}

		double gate = two_qubit_gate_time(cost_model, state->chain_lengths[zone], last - first);
		int n_gates = MAX(node->n_ms_gates, 1);
		duration = (node->operation_duration > 0.0) ? node->operation_duration : n_gates * gate;

		for (int g = 0; g < n_gates; g++)
			state->fidelity *= two_qubit_gate_fidelity(state, zone, start + g * duration / n_gates, duration / n_gates);

		for (int t = 0; t < node->n_target_ions; t++)
			state->later_gates[node->target_ions[t] - 1] -= n_gates;
	}
	else if (node->operation_type & QUBIT)
	{
		duration = node->virtual_gate ? 0.0 : cost_model->single_qubit;
		if (!node->virtual_gate)
			state->fidelity *= 1.0 - cost_model->single_qubit_error;
	}
	else if (node->operation_type & DETECT)
	{
		duration = cost_model->measurement;
		state->fidelity *= pow(1.0 - cost_model->measurement_error, node->n_target_ions);
	}
	else if (node->operation_type & COOL)
	{
		duration = cost_model->cooling;
		state->energies[zone] = cost_model->cooled_n_bar;
		state->energy_times[zone] = start + duration;
	}
	else
		duration = 0.0;

	if (node->operation_duration > 0.0)
		duration = node->operation_duration;

	return duration;
}

// Whether the moves of a round which are not done yet fit in the zones one after another in the order of the trace,
// starting from the chains as they are (after making move first, if it is not -1), so that no zone ever holds more
// ions than its capacity between moves
static int round_moves_fit
(
	struct ShuttleState *state,
	int *round_moves,
	int n_round_moves,
	int *done,
	int first,
	int *counts
)
{
	struct ZoneNode *zones = state->trap_graph->zone_nodes;
	memcpy(counts, state->chain_lengths, state->n_zones * sizeof(int));

	for (int k = -1; k < n_round_moves; k++)
	{
		int m = (k < 0) ? first : round_moves[k];
		if (m < 0 || (k >= 0 && (done[m] || m == first)))
			continue;

		struct IonMove *move = &state->trace->moves[m];
		if (move->from_zone == move->to_zone)
			continue;

		if (counts[move->to_zone] + 1 > zones[move->to_zone].zone_capacity)
			return 0;

		counts[move->to_zone]++;
		counts[move->from_zone]--;
	}

	return 1;
}

// Time every move of a schedule trace (whose transport_rounds and n_rounds are set) and the arrival time of every
// transport of its program, carrying out each move along the segments of the trap graph (see shuttle_ion) with the
// durations of a cost model (or of default_transport_cost_model if it is NULL), and recording every step in the trace.
// Each round first moves ions back into storage, then moves the ions of its operations into their zones (each time the
// ions which pass through the fewest chains first, then those at the end of their chain, and then the longest moves),
// keeping the moves of each ion in the order of the trace and never taking a move out of order if that would leave the
// rest without room in the zones, and then runs its operations, which all start once every ion of the round has arrived
// and their zones are free, and last as long as the slowest one. A move back into storage belongs to the round of the
// next transport after it in the trace, and those after the last transport happen once the last round is done. Moves
// wait for each other only where they share a segment or a zone, so a round lasts at least as long as when every ion
// moves at once along a straight line. Returns 1 if a move has no path
int time_schedule_trace
(
	struct TrapGraph *trap_graph,
	struct CircuitGraph *circuit_graph,
	struct TransportProgram *transport_program,
	struct ScheduleTrace *trace,
	const struct TransportCostModel *cost_model
)
{
	struct TransportCostModel default_cost_model = default_transport_cost_model();
	if (cost_model == NULL)
		cost_model = &default_cost_model;

	if (trap_graph_routes(trap_graph) != 0)
		return 1;

	int n_ions = transport_program->n_ions;
	int n_zones = trap_graph->n_zones;
	int n_rounds = trace->n_rounds;
	int *rounds = trace->transport_rounds;

	int n_moves = trace->n_moves;
	int n_nodes = circuit_graph->n_instruction_nodes;

	// Zone of each ion before the first move, which is recorded in the trace if the scheduler did not
	if (trace->initial_zones == NULL || trace->n_ions != n_ions)
	{
		free(trace->initial_zones);
		trace->n_ions = n_ions;
		trace->initial_zones = malloc(MAX(n_ions, 1) * sizeof(int));

		for (int i = 0; i < n_ions; i++)
		{
			trace->initial_zones[i] = 0;
			for (int z = 0; z < n_zones; z++)
				if (memcmp(trap_graph->zone_nodes[z].position, transport_program->initial_positions[i], sizeof(double[3])) == 0)
					trace->initial_zones[i] = z;
		}
	}

	trace->n_steps = 0;

	struct ShuttleState state = { trap_graph, cost_model, trace, n_zones, n_ions };
	state.chains = malloc(MAX(n_zones * (n_ions + 1), 1) * sizeof(int));
	state.chain_lengths = calloc(MAX(n_zones, 1), sizeof(int));
	state.first_neighbors = malloc(MAX(n_zones, 1) * sizeof(int));
	state.degrees = malloc(MAX(n_zones, 1) * sizeof(int));
	state.zone_free = calloc(MAX(n_zones, 1), sizeof(double));
	state.segment_free = calloc(MAX(n_zones * n_zones, 1), sizeof(double));
	state.ion_free = calloc(MAX(n_ions, 1), sizeof(double));
	state.path = malloc(MAX(n_zones, 1) * sizeof(int));
	state.energies = calloc(MAX(n_zones, 1), sizeof(double));
	state.energy_times = calloc(MAX(n_zones, 1), sizeof(double));
	state.carried = calloc(MAX(n_ions, 1), sizeof(double));
	state.carried_times = calloc(MAX(n_ions, 1), sizeof(double));
	state.fidelity = 1.0;
	state.max_n_bar = 0.0;
	state.later_gates = calloc(MAX(n_ions, 1), sizeof(int));

	for (int i = 0; i < n_nodes; i++)
	{
		struct InstructionNode *node = &circuit_graph->instruction_nodes[i];
		if ((node->operation_type & QUBIT) && node->n_target_ions >= 2)
			for (int t = 0; t < node->n_target_ions; t++)
				state.later_gates[node->target_ions[t] - 1] += MAX(node->n_ms_gates, 1);
	}

	free(trace->operation_n_bars);
	trace->operation_n_bars = calloc(MAX(n_nodes, 1), sizeof(double));

	for (int z = 0; z < n_zones; z++)
	{
		state.degrees[z] = trap_graph_neighbors(trap_graph, z, state.path);
		state.first_neighbors[z] = (state.degrees[z] > 0) ? state.path[0] : -1;
	}

	// Chains start out in the order of the ions
	for (int i = 0; i < n_ions; i++)
		insert_into_chain(&state, trace->initial_zones[i], i + 1, 1);

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

	// Zone of each operation, where its ions go
	int *op_zones = malloc(MAX(n_nodes, 1) * sizeof(int));
	for (int i = 0; i < n_nodes; i++)
		op_zones[i] = -1;
	for (int m = 0; m < n_moves; m++)
		if (trace->moves[m].transport >= 0)
			op_zones[trace->moves[m].transport] = trace->moves[m].to_zone;

	// Previous move of the ion of each move, which must be done first
	int *previous_moves = malloc(MAX(n_moves, 1) * sizeof(int));
	int *last_moves = malloc(MAX(n_ions, 1) * sizeof(int));
	for (int i = 0; i < n_ions; i++)
		last_moves[i] = -1;
	for (int m = 0; m < n_moves; m++)
	{
		previous_moves[m] = last_moves[trace->moves[m].ion - 1];
		last_moves[trace->moves[m].ion - 1] = m;
	}

	int *done = calloc(MAX(n_moves, 1), sizeof(int));
	int *counts = malloc(MAX(n_zones, 1) * sizeof(int));
	int status = 0;

	double time = 0.0;
	for (int round = 1; round <= n_rounds + 1 && status == 0; round++)
	{
		double round_end = time;

		// Moves of the round, each ion's in the order of the trace: moves back into storage first, then moves into the
		// zones of the operations, where among each the ion which would pass through the fewest chains on its way goes
		// first, then the one which needs the fewest swaps to reach the end of its chain, and then the one with the
		// longest way to go (so that it leads the others), as long as the zones have room
		int *moves_of_round = &round_moves[move_starts[round]];
		int n_moves_of_round = move_starts[round + 1] - move_starts[round];

		for (;;)
		{
			int best = -1;
			int best_transport = 0;
			int best_passes = 0;
			int best_swaps = 0;
			double best_length = 0.0;
			int first = -1;

			// If the rest of the round fits in the order of the trace (as the scheduler made it), a move may only go
			// first if the rest still fits after it; otherwise its destination needs room for it
			int fits_in_order = round_moves_fit(&state, moves_of_round, n_moves_of_round, done, -1, counts);

			for (int k = move_starts[round]; k < move_starts[round + 1]; k++)
			{
				int m = round_moves[k];
				struct IonMove *move = &trace->moves[m];
				if (done[m] || (previous_moves[m] >= 0 && !done[previous_moves[m]]))
					continue;

				if (first < 0)
					first = m;

				if (move->from_zone != move->to_zone)
				{
					if (fits_in_order ? !round_moves_fit(&state, moves_of_round, n_moves_of_round, done, m, counts) :
						state.chain_lengths[move->to_zone] >= trap_graph->zone_nodes[move->to_zone].zone_capacity)
						continue;
				}

				// Chains the ion would pass through, and swaps it needs to leave its own
				int transport = (move->transport >= 0);
				int passes = 0;
				int swaps = 0;
				double length = 0.0;
				int n_path = 0;
				if (move->from_zone != move->to_zone)
					n_path = trap_graph_route(trap_graph, move->from_zone, move->to_zone, state.path);
				if (n_path >= 2)
				{
					int position = chain_position(&state, move->from_zone, move->ion);
					if (position >= 0)
						swaps = (chain_end_facing(&state, move->from_zone, state.path[1]) == 0) ? position : state.chain_lengths[move->from_zone] - 1 - position;

					for (int p = 1; p < n_path - 1; p++)
						passes += (state.chain_lengths[state.path[p]] > 0);

					length = trap_graph_distance(trap_graph, move->from_zone, move->to_zone);
				}

				if (best < 0 || transport < best_transport || (transport == best_transport && (passes < best_passes ||
					(passes == best_passes && (swaps < best_swaps || (swaps == best_swaps && length > best_length))))))
				{
					best = m;
					best_transport = transport;
					best_passes = passes;
					best_swaps = swaps;
					best_length = length;
				}
			}

			// Every move waiting for room: take them in order
			if (best < 0)
				best = first;

			if (best < 0)
				break;

			done[best] = 1;
			status = shuttle_ion(&state, best, time);
			round_end = MAX(round_end, trace->moves[best].arrive);

			if (status != 0)
				break;
		}

		if (status != 0)
			break;

		// Operations, which start together once their ions have arrived and their zones are free
		double op_start = round_end;
		for (int k = node_starts[round]; k < node_starts[round + 1]; k++)
		{
			int zone = op_zones[round_nodes[k]];
			if (zone >= 0)
				op_start = MAX(op_start, state.zone_free[zone]);
		}

		for (int k = node_starts[round]; k < node_starts[round + 1]; k++)
		{
			int i = round_nodes[k];
			struct InstructionNode *node = &circuit_graph->instruction_nodes[i];
			int zone = MAX(op_zones[i], 0);

			heat_chain(&state, zone, op_start);
			trace->operation_n_bars[i] = state.energies[zone];

			double duration = run_operation(&state, node, zone, op_start);
			double end = add_transport_step(&state, STEP_OPERATION, 0, 0, zone, zone, i, op_start, duration);
			transport_program->arrival_times[i] = op_start;

			state.zone_free[zone] = MAX(state.zone_free[zone], end);
			for (int t = 0; t < node->n_target_ions; t++)
				state.ion_free[node->target_ions[t] - 1] = MAX(state.ion_free[node->target_ions[t] - 1], end);

			round_end = MAX(round_end, end);
		}

		time = round_end;
	}

	free(counts);
	free(done);
	free(last_moves);
	free(previous_moves);
	free(op_zones);
	free(next);
	free(round_nodes);
	free(round_moves);
	free(node_starts);
	free(move_starts);
	free(move_rounds);
	trace->fidelity = state.fidelity;
	trace->max_n_bar = state.max_n_bar;

	free(state.later_gates);
	free(state.carried_times);
	free(state.carried);
	free(state.energy_times);
	free(state.energies);
	free(state.path);
	free(state.ion_free);
	free(state.segment_free);
	free(state.zone_free);
	free(state.degrees);
	free(state.first_neighbors);
	free(state.chain_lengths);
	free(state.chains);

	return status;
}

// Generate a trap graph of zones at the given positions, joined by segments between pairs of zones (of the given
// lengths in µm, or of the distance between the centers of the zones if segment_lengths is NULL or a length is 0)
void generate_trap_graph
(
	int n_zones,
	enum ZoneType *zone_types,
	int *zone_capacities,
	double (*positions)[3],
	int n_segments,
	int (*segments)[2],
	double *segment_lengths,
	struct TrapGraph *trap_graph
)
{
	*trap_graph = (struct TrapGraph) { 0 };
	trap_graph->n_zones = n_zones;
	trap_graph->n_transport_edges = n_segments;
	trap_graph->zone_nodes = calloc(MAX(n_zones, 1), sizeof(struct ZoneNode));
	trap_graph->transport_edges = calloc(MAX(n_segments, 1), sizeof(struct TransportEdge));

	for (int z = 0; z < n_zones; z++)
	{
		struct ZoneNode *zone = &trap_graph->zone_nodes[z];

		zone->zone_type = zone_types[z];
		zone->zone_capacity = zone_capacities[z];
		zone->zone_occupancy = 0;
		memcpy(zone->position, positions[z], sizeof(double[3]));
	}

	// Junctions join their segments at their center
	int *degrees = calloc(MAX(n_zones, 1), sizeof(int));
	for (int s = 0; s < n_segments; s++)
	{
		degrees[segments[s][0]]++;
		degrees[segments[s][1]]++;
	}

	for (int s = 0; s < n_segments; s++)
	{
		struct TransportEdge *edge = &trap_graph->transport_edges[s];
		int a = segments[s][0];
		int b = segments[s][1];

		int degree = 0;
		if (zone_types[a] & JUNCTION_ZONE)
			degree = MAX(degree, degrees[a]);
		if (zone_types[b] & JUNCTION_ZONE)
			degree = MAX(degree, degrees[b]);

		edge->edge_type = (degree >= 4) ? FOUR_WAY_JUNCTION_SEGMENT : (degree == 3) ? THREE_WAY_JUNCTION_SEGMENT : LINEAR_SEGMENT;
		edge->n_zones = 2;
		edge->zone_nodes = malloc(2 * sizeof(struct ZoneNode));
		edge->zone_nodes[0] = trap_graph->zone_nodes[a];
		edge->zone_nodes[1] = trap_graph->zone_nodes[b];
		edge->length = (segment_lengths != NULL) ? segment_lengths[s] : 0.0;
	}

	free(degrees);
}

// Zone of a trap graph at the position of a copy of one of its zones, or -1 if there is none
static int find_graph_zone
(
	struct TrapGraph *trap_graph,
	struct ZoneNode *zone
)
{
	for (int z = 0; z < trap_graph->n_zones; z++)
		if (memcmp(trap_graph->zone_nodes[z].position, zone->position, sizeof(double[3])) == 0)
			return z;

	return -1;
}

// Find the segments between the zones of a trap graph (each transport edge joins each of its zones to the next) and the
// shortest path along them between every pair of zones, unless they are already known; returns 1 if a transport edge
// has a zone which is not in the graph
int trap_graph_routes
(
	struct TrapGraph *trap_graph
)
{
	if (trap_graph->route_lengths != NULL)
		return 0;

	int n = trap_graph->n_zones;
	double *segment_lengths = malloc(MAX(n * n, 1) * sizeof(double));
	double *lengths = malloc(MAX(n * n, 1) * sizeof(double));
	int *next = malloc(MAX(n * n, 1) * sizeof(int));

	for (int k = 0; k < n * n; k++)
		segment_lengths[k] = -1.0;

	for (int e = 0; e < trap_graph->n_transport_edges; e++)
	{
		struct TransportEdge *edge = &trap_graph->transport_edges[e];

		for (int k = 0; k + 1 < edge->n_zones; k++)
		{
			int a = find_graph_zone(trap_graph, &edge->zone_nodes[k]);
			int b = find_graph_zone(trap_graph, &edge->zone_nodes[k + 1]);

			if (a < 0 || b < 0)
			{
				printf("ERROR: Transport edge %d joins a zone which is not in the trap graph\n", e + 1);
				free(segment_lengths);
				free(lengths);
				free(next);
				return 1;
			}

			if (a == b)
				continue;

			double *p = trap_graph->zone_nodes[a].position;
			double *q = trap_graph->zone_nodes[b].position;
			double length = (edge->length > 0.0) ? edge->length : sqrt(pow(q[0] - p[0], 2) + pow(q[1] - p[1], 2) + pow(q[2] - p[2], 2));

			if (segment_lengths[a * n + b] < 0.0 || length < segment_lengths[a * n + b])
				segment_lengths[a * n + b] = segment_lengths[b * n + a] = length;
		}
	}

	// Shortest paths (Floyd-Warshall), keeping the first path found among those of the same length
	for (int a = 0; a < n; a++)
		for (int b = 0; b < n; b++)
		{
			int k = a * n + b;

			if (a == b)
			{
				lengths[k] = 0.0;
				next[k] = b;
			}
			else if (segment_lengths[k] >= 0.0)
			{
				lengths[k] = segment_lengths[k];
				next[k] = b;
			}
			else
			{
				lengths[k] = HUGE_VAL;
				next[k] = -1;
			}
		}

	for (int c = 0; c < n; c++)
		for (int a = 0; a < n; a++)
		{
			if (lengths[a * n + c] == HUGE_VAL)
				continue;

			for (int b = 0; b < n; b++)
			{
				double length = lengths[a * n + c] + lengths[c * n + b];

				if (length < lengths[a * n + b] * (1.0 - 1E-12))
				{
					lengths[a * n + b] = length;
					next[a * n + b] = next[a * n + c];
				}
			}
		}

	trap_graph->segment_lengths = segment_lengths;
	trap_graph->route_lengths = lengths;
	trap_graph->route_next = next;

	return 0;
}

// Shortest distance along the segments of a trap graph between two of its zones (µm), or HUGE_VAL if there is no path
double trap_graph_distance
(
	struct TrapGraph *trap_graph,
	int from_zone,
	int to_zone
)
{
	if (trap_graph_routes(trap_graph) != 0)
		return HUGE_VAL;

	return trap_graph->route_lengths[from_zone * trap_graph->n_zones + to_zone];
}

// Fill path (of length at least n_zones) with the zones of the shortest path between two zones of a trap graph, from the
// first to the last, returning their number, or 0 if there is no path
int trap_graph_route
(
	struct TrapGraph *trap_graph,
	int from_zone,
	int to_zone,
	int *path
)
{
	if (trap_graph_routes(trap_graph) != 0)
		return 0;

	int n = trap_graph->n_zones;
	if (trap_graph->route_next[from_zone * n + to_zone] < 0)
		return 0;

	int n_path = 0;
	int zone = from_zone;
	path[n_path++] = zone;

	while (zone != to_zone && n_path < n)
	{
		zone = trap_graph->route_next[zone * n + to_zone];
		path[n_path++] = zone;
	}

	return n_path;
}

// Fill neighbors (of length at least n_zones) with the zones joined to a zone of a trap graph by a segment, in order,
// returning their number
int trap_graph_neighbors
(
	struct TrapGraph *trap_graph,
	int zone,
	int *neighbors
)
{
	if (trap_graph_routes(trap_graph) != 0)
		return 0;

	int n = trap_graph->n_zones;
	int n_neighbors = 0;
	for (int z = 0; z < n; z++)
		if (trap_graph->segment_lengths[zone * n + z] >= 0.0)
			neighbors[n_neighbors++] = z;

	return n_neighbors;
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
	*trap_graph = (struct TrapGraph) { 0 };
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
	free(trap_graph->segment_lengths);
	free(trap_graph->route_lengths);
	free(trap_graph->route_next);

	trap_graph->segment_lengths = NULL;
	trap_graph->route_lengths = NULL;
	trap_graph->route_next = NULL;
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
// lies nearest to them along its segments (the one whose farthest target is nearest, and then whose targets are nearest
// in all, which on a line is the zone nearest the middle of two targets), counting targets already in a zone as making
// room in it, or NULL if there is none; ion_states holds the zone of each ion, and targets start at 1
struct ZoneNode* select_operation_zone
(
	struct TrapGraph *trap_graph,
	enum ZoneType zone_type,
	struct ZoneNode **ion_states,
	int *targets,
	int n_targets
)
{
	struct ZoneNode *nearest = NULL;
	double nearest_distance = HUGE_VAL;
	double nearest_sum = HUGE_VAL;
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

		double distance = 0.0;
		double sum = 0.0;
		for (int t = 0; t < n_targets; t++)
		{
			double d = trap_graph_distance(trap_graph, (int) (ion_states[targets[t] - 1] - trap_graph->zone_nodes), z);
			distance = MAX(distance, d);
			sum += d;
		}

		if (distance == HUGE_VAL)
			continue;

		if (distance < nearest_distance || (distance == nearest_distance && sum < nearest_sum))
		{
			nearest = zone;
			nearest_distance = distance;
			nearest_sum = sum;
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
