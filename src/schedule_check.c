#include "schedule_check.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"
#include "circuits.h"
#include "transport.h"

// Checks a schedule independently of the scheduler which made it, by replaying the steps recorded in its trace against
// the physical rules of shuttling: ions move only along the segments of the trap graph, one at a time on each segment,
// and never pass each other (an ion goes past the ions in a zone only by merging into their chain, swapping its way to
// the far end, and splitting off, where it swaps with SWAP gates only in zones where gates can be done); an ion splits
// off or swaps only at the end of a chain and merges only into its end; no zone holds more ions than its capacity
// between moves; and every operation finds its ions in a zone of its type. Only the segments of the trap graph are
// used, never its routes.

// Times closer than this are the same (s)
#define CHECK_EPS 1E-12

enum CheckIonState
{
	ION_IN_CHAIN, // the ion is in the chain in its zone
	ION_SPLIT, // the ion has split off the chain in its zone, or has arrived there along a segment
};

// Interval of time during which an ion (owner > 0) or an operation (owner < 0) takes up a zone or a segment
struct CheckInterval
{
	int place; // a zone, or n_zones + a * n_zones + b for the segment between zones a < b
	double start;
	double end;
	int owner;
};

struct ScheduleCheck
{
	struct TrapGraph *trap_graph;
	struct ScheduleTrace *trace;
	int n_zones;
	int n_ions;
	int *chains; // implicitly, length n_zones-by-(n_ions+1)
	int *chain_lengths;
	int *ion_zones;
	int *ion_states;
	int *merged; // whether an ion has merged into the chain of its zone during a move, so it may pass through it
	int n_violations;
	char *message;
	int message_size;
	int message_length;
	struct CheckInterval *intervals;
	int n_intervals;
	int max_intervals;
};

// Count a violation, describing the first few in the message
static void violation
(
	struct ScheduleCheck *check,
	const char *format,
	...
)
{
	check->n_violations++;

	if (check->message == NULL || check->message_size <= 0 || check->n_violations > 10)
		return;

	int room = check->message_size - check->message_length;
	if (room <= 1)
		return;

	va_list args;
	va_start(args, format);
	int n = vsnprintf(check->message + check->message_length, room, format, args);
	va_end(args);

	if (n < 0)
		return;

	check->message_length += MIN(n, room - 1);
	if (check->message_length < check->message_size - 1)
	{
		check->message[check->message_length++] = '\n';
		check->message[check->message_length] = '\0';
	}
}

static void add_interval
(
	struct ScheduleCheck *check,
	int place,
	double start,
	double end,
	int owner
)
{
	if (check->n_intervals == check->max_intervals)
	{
		check->max_intervals = MAX(2 * check->max_intervals, 256);
		check->intervals = realloc(check->intervals, check->max_intervals * sizeof(struct CheckInterval));
	}

	check->intervals[check->n_intervals++] = (struct CheckInterval) { place, start, end, owner };
}

static double segment_length
(
	struct ScheduleCheck *check,
	int a,
	int b
)
{
	if (a < 0 || b < 0 || a >= check->n_zones || b >= check->n_zones || a == b)
		return -1.0;

	return check->trap_graph->segment_lengths[a * check->n_zones + b];
}

// Neighbor of a zone with the lowest index, which the front of its chain faces
static int first_neighbor
(
	struct ScheduleCheck *check,
	int zone
)
{
	for (int z = 0; z < check->n_zones; z++)
		if (segment_length(check, zone, z) >= 0.0)
			return z;

	return -1;
}

static int *chain_of
(
	struct ScheduleCheck *check,
	int zone
)
{
	return &check->chains[zone * (check->n_ions + 1)];
}

static int position_in_chain
(
	struct ScheduleCheck *check,
	int zone,
	int ion
)
{
	int *chain = chain_of(check, zone);
	for (int k = 0; k < check->chain_lengths[zone]; k++)
		if (chain[k] == ion)
			return k;

	return -1;
}

// Whether an ion is at the end of the chain in its zone which faces a neighbor
static int at_chain_end
(
	struct ScheduleCheck *check,
	int zone,
	int ion,
	int neighbor
)
{
	int position = position_in_chain(check, zone, ion);
	if (position < 0)
		return 0;

	return (neighbor == first_neighbor(check, zone)) ? (position == 0) : (position == check->chain_lengths[zone] - 1);
}

static void join_chain
(
	struct ScheduleCheck *check,
	int zone,
	int ion,
	int neighbor
)
{
	int *chain = chain_of(check, zone);
	int n = check->chain_lengths[zone]++;

	if (neighbor == first_neighbor(check, zone))
	{
		memmove(&chain[1], &chain[0], n * sizeof(int));
		chain[0] = ion;
	}
	else
		chain[n] = ion;

	check->ion_zones[ion - 1] = zone;
	check->ion_states[ion - 1] = ION_IN_CHAIN;
}

static void leave_chain
(
	struct ScheduleCheck *check,
	int zone,
	int ion
)
{
	int *chain = chain_of(check, zone);
	int k = position_in_chain(check, zone, ion);
	if (k >= 0)
	{
		memmove(&chain[k], &chain[k + 1], (check->chain_lengths[zone] - k - 1) * sizeof(int));
		check->chain_lengths[zone]--;
	}

	check->ion_states[ion - 1] = ION_SPLIT;
}

// Step of a trace, to sort steps by time (and by the order they were planned at the same time)
struct TimedStep
{
	double start;
	int index;
};

static int compare_timed_steps
(
	const void *a,
	const void *b
)
{
	const struct TimedStep *x = a;
	const struct TimedStep *y = b;

	if (x->start != y->start)
		return (x->start > y->start) - (x->start < y->start);

	return (x->index > y->index) - (x->index < y->index);
}

static int compare_intervals
(
	const void *a,
	const void *b
)
{
	const struct CheckInterval *x = a;
	const struct CheckInterval *y = b;

	if (x->place != y->place)
		return (x->place > y->place) - (x->place < y->place);

	return (x->start > y->start) - (x->start < y->start);
}

static enum ZoneType required_zone_type
(
	struct InstructionNode *node
)
{
	if (node->operation_type & QUBIT)
		return GATE_ZONE;
	if (node->operation_type & COOL)
		return COOL_ZONE;
	if (node->operation_type & DETECT)
		return MEASURE_ZONE;

	return 0;
}

// Check a schedule (a transport program and the trace of every move and step of it) against the physical rules of
// shuttling on a trap graph (see the top of this file), and that its steps take at least as long as in a cost model (if
// it is not NULL). Returns the number of violations found, and describes the first few in message (if it is not NULL),
// one per line
int check_schedule_trace
(
	struct TrapGraph *trap_graph,
	struct CircuitGraph *circuit_graph,
	struct TransportProgram *transport_program,
	struct ScheduleTrace *trace,
	const struct TransportCostModel *cost_model,
	char *message,
	int message_size
)
{
	struct ScheduleCheck check = { trap_graph, trace, trap_graph->n_zones, transport_program->n_ions };
	check.message = message;
	check.message_size = message_size;
	if (message != NULL && message_size > 0)
		message[0] = '\0';

	int n_zones = check.n_zones;
	int n_ions = check.n_ions;
	int n_nodes = circuit_graph->n_instruction_nodes;

	if (trap_graph_routes(trap_graph) != 0)
	{
		violation(&check, "the transport edges of the trap graph join zones which are not in it");
		return check.n_violations;
	}

	if (trace->initial_zones == NULL || trace->n_ions != n_ions)
	{
		violation(&check, "the trace does not record where the ions start");
		return check.n_violations;
	}

	check.chains = malloc(MAX(n_zones * (n_ions + 1), 1) * sizeof(int));
	check.chain_lengths = calloc(MAX(n_zones, 1), sizeof(int));
	check.ion_zones = malloc(MAX(n_ions, 1) * sizeof(int));
	check.ion_states = malloc(MAX(n_ions, 1) * sizeof(int));
	check.merged = calloc(MAX(n_ions, 1), sizeof(int));

	// The ions start in their chains in order, between moves
	for (int i = 0; i < n_ions; i++)
	{
		int zone = trace->initial_zones[i];
		if (zone < 0 || zone >= n_zones)
		{
			violation(&check, "ion %d starts outside the trap graph", i + 1);
			zone = 0;
		}

		int *chain = chain_of(&check, zone);
		chain[check.chain_lengths[zone]++] = i + 1;
		check.ion_zones[i] = zone;
		check.ion_states[i] = ION_IN_CHAIN;
	}

	for (int z = 0; z < n_zones; z++)
		if (check.chain_lengths[z] > trap_graph->zone_nodes[z].zone_capacity)
			violation(
				&check,
				"zone %d holds %d ions at the start, more than its capacity of %d",
				z + 1, check.chain_lengths[z], trap_graph->zone_nodes[z].zone_capacity
			);

	// The last step of each move, after which its ion is between moves
	int *last_steps = malloc(MAX(trace->n_moves, 1) * sizeof(int));
	int *n_move_steps = calloc(MAX(trace->n_moves, 1), sizeof(int));
	for (int m = 0; m < trace->n_moves; m++)
		last_steps[m] = -1;

	int *op_steps = calloc(MAX(n_nodes, 1), sizeof(int));

	for (int k = 0; k < trace->n_steps; k++)
	{
		struct TransportStep *step = &trace->steps[k];

		if (step->type == STEP_OPERATION)
		{
			if (step->index < 0 || step->index >= n_nodes)
				violation(&check, "step %d runs operation %d, which is not in the circuit", k, step->index + 1);
			else
				op_steps[step->index]++;
			continue;
		}

		if (step->index < 0 || step->index >= trace->n_moves || step->ion != trace->moves[step->index].ion)
		{
			violation(&check, "step %d of ion %d is not part of one of its moves", k, step->ion);
			continue;
		}

		last_steps[step->index] = k;
		n_move_steps[step->index]++;
	}

	for (int i = 0; i < n_nodes; i++)
		if (op_steps[i] != 1)
			violation(&check, "operation %d runs %d times", i + 1, op_steps[i]);

	for (int m = 0; m < trace->n_moves; m++)
		if (trace->moves[m].from_zone != trace->moves[m].to_zone && n_move_steps[m] == 0)
			violation(
				&check,
				"ion %d jumps from zone %d to zone %d without moving along the segments",
				trace->moves[m].ion, trace->moves[m].from_zone + 1, trace->moves[m].to_zone + 1
			);

	// Replay the steps in order of time
	struct TimedStep *order = malloc(MAX(trace->n_steps, 1) * sizeof(struct TimedStep));
	for (int k = 0; k < trace->n_steps; k++)
		order[k] = (struct TimedStep) { trace->steps[k].start, k };
	qsort(order, trace->n_steps, sizeof(struct TimedStep), compare_timed_steps);

	// When each ion last set off toward a zone, or came to a chain's end in it, and is still there
	double *hold_starts = malloc(MAX(n_ions, 1) * sizeof(double));
	int *hold_zones = malloc(MAX(n_ions, 1) * sizeof(int));
	for (int i = 0; i < n_ions; i++)
		hold_zones[i] = -1;

	// When each ion is next free, for its steps and operations not to overlap
	double *ion_busy = calloc(MAX(n_ions, 1), sizeof(double));
	int *ion_moves = malloc(MAX(n_ions, 1) * sizeof(int));
	for (int i = 0; i < n_ions; i++)
		ion_moves[i] = -1;

	for (int o = 0; o < trace->n_steps; o++)
	{
		int k = order[o].index;
		struct TransportStep *step = &trace->steps[k];
		int zone = step->zone;

		if (step->end < step->start - CHECK_EPS)
			violation(&check, "step %d ends before it starts", k);

		if (zone < 0 || zone >= n_zones || step->to_zone < 0 || step->to_zone >= n_zones)
		{
			violation(&check, "step %d happens outside the trap graph", k);
			continue;
		}

		if (step->type == STEP_OPERATION)
		{
			if (step->index < 0 || step->index >= n_nodes)
				continue;

			struct InstructionNode *node = &circuit_graph->instruction_nodes[step->index];

			if (!(trap_graph->zone_nodes[zone].zone_type & required_zone_type(node)))
				violation(&check, "operation %d runs in zone %d, which is not of its type", step->index + 1, zone + 1);

			if (fabs(step->start - transport_program->arrival_times[step->index]) > CHECK_EPS)
				violation(
					&check,
					"operation %d starts at %.3f us, not at its arrival time",
					step->index + 1, 1E6 * step->start
				);

			for (int t = 0; t < node->n_target_ions; t++)
			{
				int ion = node->target_ions[t];
				if (ion < 1 || ion > n_ions)
					continue;

				if (check.ion_states[ion - 1] != ION_IN_CHAIN || check.ion_zones[ion - 1] != zone)
					violation(
						&check,
						"operation %d runs in zone %d at %.3f us, but ion %d is not in the chain there",
						step->index + 1, zone + 1, 1E6 * step->start, ion
					);

				if (step->start < ion_busy[ion - 1] - CHECK_EPS)
					violation(
						&check,
						"operation %d starts at %.3f us, while ion %d is still busy",
						step->index + 1, 1E6 * step->start, ion
					);

				ion_busy[ion - 1] = MAX(ion_busy[ion - 1], step->end);
			}

			add_interval(&check, zone, step->start, step->end, -(step->index + 1));
			continue;
		}

		int ion = step->ion;
		if (ion < 1 || ion > n_ions || step->index < 0 || step->index >= trace->n_moves || ion != trace->moves[step->index].ion)
			continue;

		struct IonMove *move = &trace->moves[step->index];

		// Steps of an ion one at a time, each move's within the times of the move
		if (step->start < ion_busy[ion - 1] - CHECK_EPS)
			violation(
				&check,
				"ion %d starts a step at %.3f us before its last step or operation is over",
				ion, 1E6 * step->start
			);
		ion_busy[ion - 1] = MAX(ion_busy[ion - 1], step->end);

		if (step->start < move->depart - CHECK_EPS || step->end > move->arrive + CHECK_EPS)
			violation(&check, "a step of ion %d lies outside the times of its move", ion);

		// A new move starts where the ion is, from the chain there
		if (ion_moves[ion - 1] != step->index)
		{
			ion_moves[ion - 1] = step->index;
			check.merged[ion - 1] = 0;

			if (check.ion_zones[ion - 1] != move->from_zone || check.ion_states[ion - 1] != ION_IN_CHAIN)
				violation(
					&check,
					"ion %d starts a move from zone %d, but is not in the chain there",
					ion, move->from_zone + 1
				);

			hold_zones[ion - 1] = check.ion_zones[ion - 1];
			hold_starts[ion - 1] = step->start;
		}

		int here = check.ion_zones[ion - 1];
		double cost;

		switch (step->type)
		{
			case STEP_SPLIT:
				if (check.ion_states[ion - 1] != ION_IN_CHAIN || here != zone)
					violation(&check, "ion %d splits off the chain in zone %d, but is not in it", ion, zone + 1);
				else if (segment_length(&check, zone, step->to_zone) < 0.0)
					violation(
						&check,
						"ion %d splits off the chain in zone %d toward zone %d, which is not a neighbor",
						ion, zone + 1, step->to_zone + 1
					);
				else if (!at_chain_end(&check, zone, ion, step->to_zone))
					violation(
						&check,
						"ion %d splits off the chain in zone %d from the middle, or from the end which does not face zone %d",
						ion, zone + 1, step->to_zone + 1
					);

				leave_chain(&check, here, ion);
				cost = (cost_model != NULL) ? cost_model->split : 0.0;
				break;

			case STEP_MOVE:
			{
				double length = segment_length(&check, zone, step->to_zone);
				if (length < 0.0)
					violation(
						&check,
						"ion %d moves from zone %d to zone %d, which are not joined by a segment",
						ion, zone + 1, step->to_zone + 1
					);

				if (here != zone)
					violation(&check, "ion %d moves from zone %d, but is in zone %d", ion, zone + 1, here + 1);
				else if (check.ion_states[ion - 1] == ION_IN_CHAIN)
				{
					// An ion alone in its zone moves off without splitting
					if (check.chain_lengths[zone] > 1)
						violation(
							&check,
							"ion %d moves out of the chain in zone %d without splitting off it",
							ion, zone + 1
						);
					leave_chain(&check, zone, ion);
				}
				else if (check.chain_lengths[zone] > 0 && !check.merged[ion - 1] && zone != move->from_zone)
				{
					// An ion which passes a zone without merging goes straight through the ions there
					violation(
						&check,
						"ion %d passes through the chain of %d ions in zone %d at %.3f us without merging into it",
						ion, check.chain_lengths[zone], zone + 1, 1E6 * step->start
					);
				}

				// The ion gives up the zone it leaves, and takes up the segment and the zone it moves toward
				if (hold_zones[ion - 1] >= 0)
					add_interval(&check, hold_zones[ion - 1], hold_starts[ion - 1], step->start, ion);

				if (length >= 0.0)
					add_interval(
						&check,
						n_zones + MIN(zone, step->to_zone) * n_zones + MAX(zone, step->to_zone),
						step->start, step->end,
						ion
					);

				hold_zones[ion - 1] = step->to_zone;
				hold_starts[ion - 1] = step->start;
				check.ion_zones[ion - 1] = step->to_zone;
				check.ion_states[ion - 1] = ION_SPLIT;
				check.merged[ion - 1] = 0;

				cost = (cost_model != NULL && length >= 0.0) ? MAX(length / cost_model->speed, cost_model->min_move) : 0.0;
				break;
			}

			case STEP_JUNCTION:
				if (!(trap_graph->zone_nodes[zone].zone_type & JUNCTION_ZONE))
					violation(&check, "ion %d crosses zone %d as a junction, but it is not one", ion, zone + 1);
				if (here != zone || check.ion_states[ion - 1] != ION_SPLIT)
					violation(&check, "ion %d crosses junction %d without being there", ion, zone + 1);

				cost = 0.0;
				if (cost_model != NULL)
				{
					int degree = 0;
					for (int z = 0; z < n_zones; z++)
						degree += (segment_length(&check, zone, z) >= 0.0);
					cost = (degree >= 4) ? cost_model->junction_x : (degree == 3) ? cost_model->junction_y : 0.0;
				}
				break;

			case STEP_MERGE:
				if (here != zone || check.ion_states[ion - 1] != ION_SPLIT)
					violation(&check, "ion %d merges into the chain in zone %d without being there", ion, zone + 1);
				else if (trap_graph->zone_nodes[zone].zone_type & JUNCTION_ZONE)
					violation(&check, "ion %d merges into junction %d, which holds no ions", ion, zone + 1);
				else if (segment_length(&check, zone, step->to_zone) < 0.0)
					violation(
						&check,
						"ion %d merges into the chain in zone %d from zone %d, which is not a neighbor",
						ion, zone + 1, step->to_zone + 1
					);
				else
				{
					join_chain(&check, zone, ion, step->to_zone);
					check.merged[ion - 1] = 1;

					// A chain may hold one ion more than its capacity while an ion passes through it
					int passing = (last_steps[step->index] != k);
					if (check.chain_lengths[zone] > trap_graph->zone_nodes[zone].zone_capacity + passing)
						violation(
							&check,
							"zone %d holds %d ions at %.3f us, more than its capacity of %d",
							zone + 1, check.chain_lengths[zone], 1E6 * step->end, trap_graph->zone_nodes[zone].zone_capacity
						);
				}

				cost = (cost_model != NULL) ? cost_model->merge : 0.0;
				break;

			case STEP_ION_SWAP:
			{
				int a = position_in_chain(&check, zone, ion);
				int b = position_in_chain(&check, zone, step->other_ion);
				if (a < 0 || b < 0 || abs(a - b) != 1)
					violation(
						&check,
						"ions %d and %d swap in zone %d, but are not neighbors in the chain there",
						ion, step->other_ion, zone + 1
					);
				else
				{
					int *chain = chain_of(&check, zone);
					chain[a] = step->other_ion;
					chain[b] = ion;
				}

				if (step->other_ion >= 1 && step->other_ion <= n_ions)
				{
					if (step->start < ion_busy[step->other_ion - 1] - CHECK_EPS)
						violation(
							&check,
							"ion %d swaps with ion %d at %.3f us, while that ion is busy",
							ion, step->other_ion, 1E6 * step->start
						);
					ion_busy[step->other_ion - 1] = MAX(ion_busy[step->other_ion - 1], step->end);
				}

				cost = (cost_model != NULL) ? cost_model->split + cost_model->rotation + cost_model->merge : 0.0;
				break;
			}

			case STEP_GATE_SWAP:
			{
				int a = position_in_chain(&check, zone, ion);
				int b = position_in_chain(&check, zone, step->other_ion);
				if (!(trap_graph->zone_nodes[zone].zone_type & GATE_ZONE))
					violation(
						&check,
						"ions %d and %d swap with a SWAP gate in zone %d, where gates cannot be done",
						ion, step->other_ion, zone + 1
					);
				if (a < 0 || b < 0 || a == b)
					violation(
						&check,
						"ions %d and %d swap with a SWAP gate in zone %d, but are not both in the chain there",
						ion, step->other_ion, zone + 1
					);
				else
				{
					int *chain = chain_of(&check, zone);
					chain[a] = step->other_ion;
					chain[b] = ion;
				}

				if (step->other_ion >= 1 && step->other_ion <= n_ions)
				{
					if (step->start < ion_busy[step->other_ion - 1] - CHECK_EPS)
						violation(
							&check,
							"ion %d swaps with ion %d at %.3f us, while that ion is busy",
							ion, step->other_ion, 1E6 * step->start
						);
					ion_busy[step->other_ion - 1] = MAX(ion_busy[step->other_ion - 1], step->end);
				}

				cost = 0.0;
				if (cost_model != NULL && a >= 0 && b >= 0)
					cost = 3 * two_qubit_gate_time(cost_model, check.chain_lengths[zone], abs(a - b));
				break;
			}

			default:
				violation(&check, "step %d is of an unknown type", k);
				cost = 0.0;
				break;
		}

		if (step->end - step->start < cost * (1.0 - 1E-9) - CHECK_EPS)
			violation(
				&check,
				"a step of ion %d takes %.3f us, less than the %.3f us of the cost model",
				ion, 1E6 * (step->end - step->start), 1E6 * cost
			);

		// At the end of a move, the ion is in the chain in its destination (joining an empty zone without a merge)
		if (k == last_steps[step->index])
		{
			int to = move->to_zone;

			if (check.ion_zones[ion - 1] != to)
				violation(
					&check,
					"ion %d ends a move in zone %d, not in its destination %d",
					ion, check.ion_zones[ion - 1] + 1, to + 1
				);
			else if (check.ion_states[ion - 1] == ION_SPLIT)
			{
				if (trap_graph->zone_nodes[to].zone_type & JUNCTION_ZONE)
					violation(&check, "ion %d stays in junction %d", ion, to + 1);
				else if (check.chain_lengths[to] > 0)
					violation(
						&check,
						"ion %d joins the chain of %d ions in zone %d without merging into it",
						ion, check.chain_lengths[to], to + 1
					);

				join_chain(&check, to, ion, (step->type == STEP_MOVE) ? step->zone : step->to_zone);
				if (check.chain_lengths[to] > trap_graph->zone_nodes[to].zone_capacity)
					violation(
						&check,
						"zone %d holds %d ions, more than its capacity of %d",
						to + 1, check.chain_lengths[to], trap_graph->zone_nodes[to].zone_capacity
					);
			}

			if (fabs(step->end - move->arrive) > CHECK_EPS)
				violation(&check, "ion %d arrives at %.3f us, not when its last step ends", ion, 1E6 * move->arrive);

			if (hold_zones[ion - 1] >= 0)
				add_interval(&check, hold_zones[ion - 1], hold_starts[ion - 1], step->end, ion);
			hold_zones[ion - 1] = -1;
		}
	}

	// Ions must be in a chain between moves
	for (int i = 0; i < n_ions; i++)
		if (check.ion_states[i] != ION_IN_CHAIN)
			violation(&check, "ion %d ends the schedule outside a chain", i + 1);

	// Operations must start once their ions have arrived
	for (int m = 0; m < trace->n_moves; m++)
	{
		struct IonMove *move = &trace->moves[m];
		if (move->transport >= 0 && move->transport < n_nodes && move->arrive > transport_program->arrival_times[move->transport] + CHECK_EPS)
			violation(&check, "operation %d starts before ion %d arrives", move->transport + 1, move->ion);
	}

	// No two ions take up a zone or a segment at once (operations may share a zone with each other)
	qsort(check.intervals, check.n_intervals, sizeof(struct CheckInterval), compare_intervals);
	for (int a = 0; a < check.n_intervals; a++)
	{
		struct CheckInterval *x = &check.intervals[a];

		for (int b = a + 1; b < check.n_intervals && check.intervals[b].place == x->place && check.intervals[b].start < x->end - CHECK_EPS; b++)
		{
			struct CheckInterval *y = &check.intervals[b];
			if (y->owner == x->owner || (x->owner < 0 && y->owner < 0) || y->end <= y->start + CHECK_EPS || x->end <= x->start + CHECK_EPS)
				continue;

			char what[64];
			if (x->place < n_zones)
				snprintf(what, sizeof(what), "zone %d", x->place + 1);
			else
				snprintf(
					what,
					sizeof(what),
					"the segment between zones %d and %d",
					(x->place - n_zones) / n_zones + 1, (x->place - n_zones) % n_zones + 1
				);

			char first[32];
			char second[32];
			snprintf(first, sizeof(first), x->owner > 0 ? "ion %d" : "operation %d", abs(x->owner));
			snprintf(second, sizeof(second), y->owner > 0 ? "ion %d" : "operation %d", abs(y->owner));
			violation(&check, "%s and %s take up %s at once, at %.3f us", first, second, what, 1E6 * y->start);
		}
	}

	free(ion_moves);
	free(ion_busy);
	free(hold_zones);
	free(hold_starts);
	free(order);
	free(op_steps);
	free(n_move_steps);
	free(last_steps);
	free(check.intervals);
	free(check.merged);
	free(check.ion_states);
	free(check.ion_zones);
	free(check.chain_lengths);
	free(check.chains);

	return check.n_violations;
}
