#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>

#include "defs.h"
#include "data_structures.h"
#include "circuits.h"
#include "transport.h"
#include "zones.h"

// Compare the strategies of generate_transport_schedule_strategy on circuits and zone layouts, by the schedules they
// make (following the ions through every move, including the moves back into storage which transport programs do not
// record) and by how long they take to make them

#define ZONE_SPACING 100.0 // distance between neighboring zones (µm)
#define MIN_RUNS 5
#define MAX_RUNS 1000
#define MIN_TIME 0.2 // shortest time spent timing each strategy (s)

// Strategies compared, where the first two must give the same schedule, the lookahead strategies are also run looking
// further ahead than the next layer, and some strategies are also run with commuting (diagonal) operations moved into
// fewer layers, which are compared with the same strategy without (their base)
static const char *strategy_names[] = {
	"baseline", "fast", "rounds", "lookahead", "layer-assign",
	"lookahead-2", "lookahead-3", "lookahead-6", "layer-assign-2", "layer-assign-3", "layer-assign-6",
	"rounds-commute", "layer-assign-commute", "layer-assign-2-commute",
};
static const struct ScheduleOptions strategies[] = {
	{ SCHEDULE_BASELINE, 1 }, { SCHEDULE_FAST, 1 }, { SCHEDULE_ROUNDS, 1 }, { SCHEDULE_LOOKAHEAD, 1 }, { SCHEDULE_LAYER_ASSIGN, 1 },
	{ SCHEDULE_LOOKAHEAD, 2 }, { SCHEDULE_LOOKAHEAD, 3 }, { SCHEDULE_LOOKAHEAD, 6 },
	{ SCHEDULE_LAYER_ASSIGN, 2 }, { SCHEDULE_LAYER_ASSIGN, 3 }, { SCHEDULE_LAYER_ASSIGN, 6 },
	{ SCHEDULE_ROUNDS, 1, 1 }, { SCHEDULE_LAYER_ASSIGN, 1, 1 }, { SCHEDULE_LAYER_ASSIGN, 2, 1 },
};
static const int bases[] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 4, 8 };
#define N_STRATEGIES 14

// Every circuit in this directory, made by quantumcircuits/generate_benchmark_circuits.py
#define CIRCUIT_DIR "quantumcircuits/benchmarks"

static const char *layouts[] = {
	"load+storage, empty, gate, empty, measure",
	"load+storage, gate, gate, measure",
	"load, storage x2, gate x5, measure",
};
#define N_LAYOUTS 3

// Results of one strategy on one circuit and layout
struct ScheduleResult
{
	int scheduled; // whether the scheduler made a schedule
	int n_invalid; // operations whose ions are not all in a zone of the right type when it happens, and zones over capacity
	double makespan; // when the last operation ends (s)
	double distance; // total distance moved by the ions along the segments of the trap (µm)
	int n_moves; // moves of an ion to a different zone
	int n_splits; // splits of an ion off a chain
	int n_merges; // merges of an ion into a chain
	int n_swaps; // swaps of neighboring ions in a chain, to bring an ion to the end of its chain
	double run_time; // median time taken by the scheduler (s)
};

static double wall_time
(
)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);

	return t.tv_sec + 1E-9 * t.tv_nsec;
}

static int compare_doubles
(
	const void *a,
	const void *b
)
{
	double x = *(const double *) a;
	double y = *(const double *) b;

	return (x > y) - (x < y);
}

// Send stdout to /dev/null (silence = 1) or back (silence = 0), to hide the output of the parser and scheduler
static void silence_stdout
(
	int silence
)
{
	static int saved = -1;

	fflush(stdout);

	if (silence && saved < 0)
	{
		saved = dup(1);
		int devnull = open("/dev/null", O_WRONLY);
		dup2(devnull, 1);
		close(devnull);
	}
	else if (!silence && saved >= 0)
	{
		dup2(saved, 1);
		close(saved);
		saved = -1;
	}
}

// Duration of an operation, as in schedule_arrival_times
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

static enum ZoneType operation_zone_type
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

// Follow the ions through the moves of a schedule, checking that every operation finds its ions in a zone of its type
// with room for them, and adding up the distance they move along the segments of the trap; the makespan is when the
// last operation of the program ends, and the steps which carry out the moves are counted
static void evaluate_schedule
(
	struct TrapGraph *trap_graph,
	struct CircuitGraph *circuit_graph,
	struct TransportProgram *transport_program,
	struct ScheduleTrace *trace,
	struct ScheduleResult *result
)
{
	int n_ions = circuit_graph->n_ions;
	int n_zones = trap_graph->n_zones;

	// Rounds of the schedule, which are called layers here
	int *order;
	int *layer_starts;
	int n_layers = sort_layers(circuit_graph->n_instruction_nodes, trace->transport_rounds, &order, &layer_starts);

	int *ion_zones = malloc(n_ions * sizeof(int));
	int *occupancy = calloc(n_zones, sizeof(int));
	for (int i = 0; i < n_ions; i++)
	{
		ion_zones[i] = 0;
		for (int z = 0; z < n_zones; z++)
			if (memcmp(trap_graph->zone_nodes[z].position, transport_program->initial_positions[i], sizeof(double[3])) == 0)
				ion_zones[i] = z;

		occupancy[ion_zones[i]]++;
	}

	// Moves of the targets of each layer still to come
	int *layer_moves = calloc(n_layers + 1, sizeof(int));
	for (int i = 0; i < circuit_graph->n_instruction_nodes; i++)
		layer_moves[trace->transport_rounds[i]] += circuit_graph->instruction_nodes[i].n_target_ions;

	int *op_zones = malloc(circuit_graph->n_instruction_nodes * sizeof(int));

	result->n_invalid = 0;
	result->makespan = 0.0;
	result->distance = 0.0;
	result->n_moves = 0;
	result->n_splits = 0;
	result->n_merges = 0;
	result->n_swaps = 0;

	int layer = 1;
	for (int m = 0; m < trace->n_moves && layer <= n_layers; m++)
	{
		struct IonMove *move = &trace->moves[m];
		int ion = move->ion - 1;
		double distance = trap_graph_distance(trap_graph, move->from_zone, move->to_zone);

		result->distance += distance;
		result->n_moves += (move->from_zone != move->to_zone);

		occupancy[move->from_zone]--;
		occupancy[move->to_zone]++;
		ion_zones[ion] = move->to_zone;

		if (move->transport < 0)
			continue;

		op_zones[move->transport] = move->to_zone;

		if (--layer_moves[layer] > 0)
			continue;

		// Every ion of the layer has arrived, so its operations run now
		for (int o = layer_starts[layer - 1]; o < layer_starts[layer]; o++)
		{
			struct InstructionNode *node = &circuit_graph->instruction_nodes[order[o]];
			int zone = op_zones[order[o]];

			int n_wrong = !(trap_graph->zone_nodes[zone].zone_type & operation_zone_type(node));
			for (int t = 0; t < node->n_target_ions; t++)
				n_wrong += (ion_zones[node->target_ions[t] - 1] != zone);

			result->n_invalid += (n_wrong > 0);
		}

		for (int z = 0; z < n_zones; z++)
			result->n_invalid += (occupancy[z] > trap_graph->zone_nodes[z].zone_capacity);

		layer++;
	}

	// Layers whose ions never all arrived
	result->n_invalid += n_layers + 1 - layer;

	for (int k = 0; k < trace->n_steps; k++)
	{
		struct TransportStep *step = &trace->steps[k];

		result->makespan = MAX(result->makespan, step->end);
		result->n_splits += (step->type == STEP_SPLIT);
		result->n_merges += (step->type == STEP_MERGE);
		result->n_swaps += (step->type == STEP_ION_SWAP);
	}

	free(op_zones);
	free(layer_moves);
	free(occupancy);
	free(ion_zones);
	free(order);
	free(layer_starts);
}

// Operations which run in an order the circuit does not allow: two operations on the same ion in the same round, or two
// operations on the same ion in the opposite order to the circuit, unless both are diagonal (and so commute)
static int count_order_violations
(
	struct CircuitGraph *circuit_graph,
	struct ScheduleTrace *trace
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

			if (!share)
				continue;

			int round_a = trace->transport_rounds[i];
			int round_b = trace->transport_rounds[j];
			n_violations += (round_a == round_b) || (round_a > round_b && !(a->diagonal && b->diagonal));
		}

	return n_violations;
}

// Whether two transport programs and traces are exactly the same
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

	if (memcmp(trace_a->transport_rounds, trace_b->transport_rounds, a->n_transports * sizeof(int)) != 0)
		return 0;

	if (memcmp(a->arrival_times, b->arrival_times, a->n_transports * sizeof(double)) != 0 ||
		memcmp(a->destinations, b->destinations, a->n_transports * sizeof(double[3])) != 0 ||
		memcmp(a->transport_n_targets_list, b->transport_n_targets_list, a->n_transports * sizeof(int)) != 0 ||
		memcmp(trace_a->moves, trace_b->moves, trace_a->n_moves * sizeof(struct IonMove)) != 0)
		return 0;

	for (int t = 0; t < a->n_transports; t++)
		if (memcmp(a->transport_targets_list[t], b->transport_targets_list[t], a->transport_n_targets_list[t] * sizeof(int)) != 0)
			return 0;

	return 1;
}

static struct TransportProgram* run_schedule
(
	struct TrapGraph *trap_graph,
	struct CircuitGraph *circuit_graph,
	const struct ScheduleOptions *options,
	struct ScheduleTrace *trace
)
{
	for (int z = 0; z < trap_graph->n_zones; z++)
		trap_graph->zone_nodes[z].zone_occupancy = 0;

	struct TransportProgram *transport_program = calloc(1, sizeof(struct TransportProgram));

	silence_stdout(1);
	struct TransportProgram *result = generate_transport_schedule_options(trap_graph, circuit_graph, transport_program, options, trace);
	silence_stdout(0);

	if (result == NULL)
	{
		free_transport_program(transport_program);
		return NULL;
	}

	return transport_program;
}

int main
(
	int argc,
	char **argv
)
{
	const char *csv_filename = (argc > 1) ? argv[1] : "schedule_benchmark.csv";
	FILE *csv = fopen(csv_filename, "w");
	if (csv != NULL)
		fprintf(csv, "circuit,qubits,operations,layout,strategy,scheduled,invalid,rounds,makespan_us,distance_um,moves,splits,merges,swaps,run_time_us\n");

	int n_mismatches = 0;
	int n_worse = 0;
	int n_timing = 0;
	int n_order = 0;
	int n_commute_slower = 0;
	int n_slower = 0;

	struct dirent **entries;
	int n_entries = scandir(CIRCUIT_DIR, &entries, NULL, alphasort);
	if (n_entries < 0)
	{
		printf("ERROR: Failed to read the circuits in '%s'\n", CIRCUIT_DIR);
		return 1;
	}

	printf("%-20s %-42s %-22s %7s %7s %12s %13s %7s %7s %12s\n", "circuit", "layout", "strategy", "valid", "rounds", "makespan(us)", "distance(um)", "moves", "swaps", "time(us)");

	for (int c = 0; c < n_entries; c++)
	{
		const char *circuit_name = entries[c]->d_name;
		size_t length = strlen(circuit_name);
		if (length < 5 || strcmp(circuit_name + length - 5, ".qasm") != 0)
			continue;

		char filename[512];
		snprintf(filename, sizeof(filename), "%s/%s", CIRCUIT_DIR, circuit_name);

		silence_stdout(1);
		struct CircuitGraph *circuit_graph = parse_qasm(filename);
		silence_stdout(0);

		if (circuit_graph == NULL)
		{
			printf("ERROR: Failed to read '%s'\n", filename);
			continue;
		}

		for (int l = 0; l < N_LAYOUTS; l++)
		{
			int n_zones;
			enum ZoneType *zone_types;
			int *zone_capacities;
			if (parse_zone_layout(layouts[l], circuit_graph->n_ions, &n_zones, &zone_types, &zone_capacities) != 0)
				continue;

			struct TrapGraph trap_graph;
			generate_linear_trap_graph(n_zones, zone_types, zone_capacities, 0.0, (n_zones - 1) * ZONE_SPACING, ION_HEIGHT, &trap_graph);

			struct TransportProgram *programs[N_STRATEGIES];
			struct ScheduleTrace traces[N_STRATEGIES];
			struct ScheduleResult results[N_STRATEGIES];

			for (int s = 0; s < N_STRATEGIES; s++)
			{
				traces[s] = (struct ScheduleTrace) { 0 };
				programs[s] = run_schedule(&trap_graph, circuit_graph, &strategies[s], &traces[s]);

				struct ScheduleResult *result = &results[s];
				memset(result, 0, sizeof(*result));
				result->scheduled = (programs[s] != NULL);

				if (programs[s] != NULL)
				{
					evaluate_schedule(&trap_graph, circuit_graph, programs[s], &traces[s], result);

					// Every operation must start once all of its ions have arrived, and the program must end with its last
					// operation
					int n_early = 0;
					for (int m = 0; m < traces[s].n_moves; m++)
					{
						struct IonMove *move = &traces[s].moves[m];
						n_early += (move->transport >= 0 && move->arrive > programs[s]->arrival_times[move->transport] + 1E-12);
					}

					double program_makespan = 0.0;
					for (int i = 0; i < circuit_graph->n_instruction_nodes; i++)
						program_makespan = MAX(program_makespan, programs[s]->arrival_times[i] + operation_duration(&circuit_graph->instruction_nodes[i]));

					if (n_early > 0 || program_makespan > result->makespan * (1.0 + 1E-12))
					{
						printf("  TIMING: %s starts %d operations before their ions arrive, or ends at %.2f us after its last step at %.2f us\n", strategy_names[s], n_early, 1E6 * program_makespan, 1E6 * result->makespan);
						n_timing++;
					}
				}

				// Time the scheduler without a trace of our own (it still times every move with a trace of its own)
				double run_times[MAX_RUNS];
				int n_runs = 0;
				double start = wall_time();
				while (n_runs < MAX_RUNS && (n_runs < MIN_RUNS || wall_time() - start < MIN_TIME))
				{
					for (int z = 0; z < trap_graph.n_zones; z++)
						trap_graph.zone_nodes[z].zone_occupancy = 0;

					struct TransportProgram *transport_program = calloc(1, sizeof(struct TransportProgram));

					silence_stdout(1);
					double t0 = wall_time();
					generate_transport_schedule_options(&trap_graph, circuit_graph, transport_program, &strategies[s], NULL);
					run_times[n_runs++] = wall_time() - t0;
					silence_stdout(0);

					free_transport_program(transport_program);
				}

				qsort(run_times, n_runs, sizeof(double), compare_doubles);
				result->run_time = run_times[n_runs / 2];

				int valid = result->scheduled && result->n_invalid == 0;
				int n_rounds = (programs[s] != NULL) ? traces[s].n_rounds : 0;
				printf("%-20s %-42s %-22s %7s %7d %12.2f %13.0f %7d %7d %12.2f\n", circuit_name, layouts[l], strategy_names[s],
					!result->scheduled ? "fail" : valid ? "yes" : "NO", n_rounds, 1E6 * result->makespan, result->distance, result->n_moves, result->n_swaps, 1E6 * result->run_time);

				if (csv != NULL)
					fprintf(csv, "%s,%d,%d,\"%s\",%s,%d,%d,%d,%.6f,%.3f,%d,%d,%d,%d,%.6f\n", circuit_name, circuit_graph->n_ions, circuit_graph->n_instruction_nodes, layouts[l], strategy_names[s],
						result->scheduled, result->n_invalid, n_rounds, 1E6 * result->makespan, result->distance, result->n_moves, result->n_splits, result->n_merges, result->n_swaps, 1E6 * result->run_time);
			}

			// The fast scheduler must make exactly the same schedule as the baseline
			if ((programs[0] == NULL) != (programs[1] == NULL) || (programs[0] != NULL && !same_schedule(programs[0], &traces[0], programs[1], &traces[1])))
			{
				printf("  MISMATCH: fast and baseline schedules differ\n");
				n_mismatches++;
			}

			// The other strategies must always make valid schedules. None is sure to be faster than rounds: the strategies
			// choose zones by distance, but ions which share a segment or pass through a chain wait for each other, so a
			// shorter longest move can still make a longer round (which is counted and reported rather than failed)
			for (int s = 2; s < N_STRATEGIES; s++)
			{
				int valid = results[s].scheduled && results[s].n_invalid == 0;
				int slower = valid && s > 2 && results[s].makespan > results[2].makespan * (1.0 + 1E-12);

				if (!valid)
				{
					printf("  WORSE: %s is invalid\n", strategy_names[s]);
					n_worse++;
				}
				else if (slower)
				{
					printf("  NOTE: %s is slower than rounds\n", strategy_names[s]);
					n_slower++;
				}

				// Moving commuting operations into fewer layers can still lengthen a schedule, which is counted (and
				// reported) rather than failed
				if (valid && bases[s] >= 0 && results[bases[s]].scheduled && results[s].makespan > results[bases[s]].makespan * (1.0 + 1E-12))
				{
					printf("  NOTE: %s is slower than %s\n", strategy_names[s], strategy_names[bases[s]]);
					n_commute_slower++;
				}
			}

			// Every schedule must run the operations of each ion in an order the circuit allows
			for (int s = 0; s < N_STRATEGIES; s++)
			{
				if (programs[s] == NULL)
					continue;

				int n_violations = count_order_violations(circuit_graph, &traces[s]);
				if (n_violations > 0)
				{
					printf("  ORDER: %s runs %d pairs of operations in an order the circuit does not allow\n", strategy_names[s], n_violations);
					n_order++;
				}
			}

			for (int s = 0; s < N_STRATEGIES; s++)
			{
				free_transport_program(programs[s]);
				free_schedule_trace(&traces[s]);
			}

			free_trap_graph(&trap_graph);
			free(zone_types);
			free(zone_capacities);
		}

		free_circuit_graph(circuit_graph);
	}

	for (int c = 0; c < n_entries; c++)
		free(entries[c]);
	free(entries);

	if (csv != NULL)
	{
		fclose(csv);
		printf("\nWrote %s\n", csv_filename);
	}

	printf("%d mismatches between the fast and baseline schedules, %d invalid schedules, %d programs whose arrival times disagree with their moves, %d schedules which break the order of the circuit\n", n_mismatches, n_worse, n_timing, n_order);
	printf("%d schedules slower than rounds, and %d with commuting operations moved slower than without (reported, not failed)\n", n_slower, n_commute_slower);

	return (n_mismatches > 0 || n_worse > 0 || n_timing > 0 || n_order > 0);
}
