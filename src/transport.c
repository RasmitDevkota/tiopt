#include "transport.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"
#include "circuits.h"

// Helper to find an available zone of a given type respecting capacity
struct ZoneNode* select_available_zone
(
    struct TrapGraph *trapGraph,
    enum ZoneType type
)
{
    for (int i = 0; i < trapGraph->n_zones; i++) {
        if ((trapGraph->zone_nodes[i].zone_type & type) &&
            trapGraph->zone_nodes[i].zone_occupancy < trapGraph->zone_nodes[i].zone_capacity) {
            return &trapGraph->zone_nodes[i];
        }
    }

    return NULL;
}

// Whether the scheduler prints the layers and the occupancy of the zones as it goes, which is set by the environment
// variable TIOPT_SCHEDULE_VERBOSE
static int schedule_verbose()
{
    static int verbose = -1;
    if (verbose < 0) {
        verbose = (getenv("TIOPT_SCHEDULE_VERBOSE") != NULL);
    }

    return verbose;
}

static void print_zone_occupancy(struct TrapGraph *trapGraph)
{
    if (!schedule_verbose()) {
        return;
    }

    for (int i = 0; i < trapGraph->n_zones; i++) {
        printf("Zone %d Occupancy: %d\n", i + 1, trapGraph->zone_nodes[i].zone_occupancy);
        printf("Zone %d Capacity: %d\n", i + 1, trapGraph->zone_nodes[i].zone_capacity);
        printf("\n");
    }
}

// Type of zone which an operation happens in, or 0 if there is none
static enum ZoneType operation_zone_type(struct InstructionNode *instr)
{
    if (instr->operation_type & QUBIT) {
        return GATE_ZONE;
    } else if (instr->operation_type & COOL) {
        return COOL_ZONE;
    } else if (instr->operation_type & DETECT) {
        return MEASURE_ZONE;
    }

    return 0;
}

// Record the transport of the target ions of instruction i into a zone, and move them there
static void transport_ions
(
    struct TrapGraph *trapGraph,
    struct ZoneNode **ion_states,
    struct InstructionNode *instr,
    int i,
    struct ZoneNode *instructionZone,
    struct TransportProgram *transportProgram,
    struct ScheduleTrace *trace
)
{
    memcpy(transportProgram->destinations[i], instructionZone->position, sizeof(double[3]));

    for (int t = 0; t < instr->n_target_ions; t++) {
        int ion_id = instr->target_ions[t];

        transportProgram->transport_targets_list[i][t] = ion_id;

        record_ion_move(trace, trapGraph, ion_id, ion_states[ion_id - 1], instructionZone, i);

        // Update occupancy
        ion_states[ion_id - 1]->zone_occupancy--;
        instructionZone->zone_occupancy++;

        ion_states[ion_id - 1] = instructionZone;
    }
}

// Original scheduler, which scans the whole circuit for the instructions of each layer
static int schedule_baseline
(
    struct TrapGraph *trapGraph,
    struct CircuitGraph *circuitGraph,
    struct TransportProgram *transportProgram,
    struct ZoneNode **ion_states,
    int max_layer,
    int *rounds,
    struct ScheduleTrace *trace
)
{
    for (int layer = 1; layer <= max_layer; layer++) {
        for (int i = 0; i < circuitGraph->n_instruction_nodes; i++) {
            if (circuitGraph->layer_idxs[i] != layer) {
                continue;
            }

            struct InstructionNode *instr = &circuitGraph->instruction_nodes[i];
            struct ZoneNode *instructionZone = NULL;

            rounds[i] = layer;

            transportProgram->transport_n_targets_list[i] = instr->n_target_ions;
            transportProgram->transport_targets_list[i] = (int *)malloc(sizeof(int) * transportProgram->transport_n_targets_list[i]);

            // Find instruction zone for each instruction: the nearest one with room for its ions, moving ions out of
            // the zones used earlier in the layer if every zone is full
            enum ZoneType instructionType = operation_zone_type(instr);

            if (instructionType) {
                instructionZone = select_operation_zone(trapGraph, instructionType, ion_states, instr->target_ions, instr->n_target_ions);
                if (!instructionZone) {
                    release_ions(trapGraph, ion_states, transportProgram->n_ions, NULL, trace);
                    instructionZone = select_operation_zone(trapGraph, instructionType, ion_states, instr->target_ions, instr->n_target_ions);
                }
            }

            if (!instructionZone) {
                printf("ERROR: No available instruction zone for instruction %d.\n", instr->graphindex);
                return 1;
            }

            print_zone_occupancy(trapGraph);

            // Move each target ion to a the instruction zone
            transport_ions(trapGraph, ion_states, instr, i, instructionZone, transportProgram, trace);

            print_zone_occupancy(trapGraph);
        }

        // Move ions to storage/empty zones once the layer is done, so that the operations of a layer can use several
        // instruction zones at once
        release_ions(trapGraph, ion_states, transportProgram->n_ions, NULL, trace);

        print_zone_occupancy(trapGraph);
    }

    return 0;
}

// Assign each operation of a layer to a zone of its type, so that the longest distance any target ion moves is as short
// as possible (and then the total distance), with branch and bound over the zones of the operations in order of their
// shortest possible transport, longest first; ions which no operation of the layer targets stay where they are. Returns
// 1 if no assignment fits in the zones
#define LAYER_ASSIGN_MAX_NODES 20000

struct LayerAssignment
{
    struct TrapGraph *trapGraph;
    int n_ops;
    int *op_order; // operations in the order they are assigned
    int *n_targets; // implicitly, length n_ops
    int *n_candidates; // implicitly, length n_ops
    int **candidates; // zones each operation can use, nearest first; implicitly, length n_ops-by-[n_candidates]
    double **costs; // longest distance moved by a target ion for each candidate; implicitly, like candidates
    double **sums; // total distance moved by the target ions for each candidate; implicitly, like candidates
    int *free_capacity; // room left in each zone; implicitly, length n_zones
    int *choice; // implicitly, length n_ops
    int *best_choice; // implicitly, length n_ops
    double best_max;
    double best_sum;
    int n_nodes;
};

static void search_layer_assignment(struct LayerAssignment *a, int depth, double max_cost, double sum_cost)
{
    if (a->n_nodes++ > LAYER_ASSIGN_MAX_NODES) {
        return;
    }

    if (depth == a->n_ops) {
        if (max_cost < a->best_max || (max_cost == a->best_max && sum_cost < a->best_sum)) {
            a->best_max = max_cost;
            a->best_sum = sum_cost;
            memcpy(a->best_choice, a->choice, a->n_ops * sizeof(int));
        }
        return;
    }

    int k = a->op_order[depth];
    for (int c = 0; c < a->n_candidates[k]; c++) {
        // Candidates are sorted by cost, so no later one can do better
        double cost = MAX(max_cost, a->costs[k][c]);
        if (cost > a->best_max) {
            break;
        }

        if (cost == a->best_max && sum_cost + a->sums[k][c] >= a->best_sum) {
            continue;
        }

        int z = a->candidates[k][c];
        if (a->free_capacity[z] < a->n_targets[k]) {
            continue;
        }

        a->free_capacity[z] -= a->n_targets[k];
        a->choice[k] = z;
        search_layer_assignment(a, depth + 1, cost, sum_cost + a->sums[k][c]);
        a->free_capacity[z] += a->n_targets[k];
    }
}

static int assign_layer_zones
(
    struct TrapGraph *trapGraph,
    struct CircuitGraph *circuitGraph,
    struct ZoneNode **ion_states,
    int *ops,
    int n_ops,
    struct ZoneNode **op_zones
)
{
    int n_zones = trapGraph->n_zones;
    struct ZoneNode *zones = trapGraph->zone_nodes;

    struct LayerAssignment a = { trapGraph, n_ops };
    a.op_order = calloc(n_ops, sizeof(int));
    a.n_targets = calloc(n_ops, sizeof(int));
    a.n_candidates = calloc(n_ops, sizeof(int));
    a.candidates = calloc(n_ops, sizeof(int*));
    a.costs = calloc(n_ops, sizeof(double*));
    a.sums = calloc(n_ops, sizeof(double*));
    a.free_capacity = malloc(n_zones * sizeof(int));
    a.choice = malloc(n_ops * sizeof(int));
    a.best_choice = malloc(n_ops * sizeof(int));
    a.best_max = HUGE_VAL;
    a.best_sum = HUGE_VAL;

    // Room in each zone once the target ions of the layer have left it
    for (int z = 0; z < n_zones; z++) {
        a.free_capacity[z] = zones[z].zone_capacity - zones[z].zone_occupancy;
    }

    for (int k = 0; k < n_ops; k++) {
        struct InstructionNode *instr = &circuitGraph->instruction_nodes[ops[k]];
        for (int t = 0; t < instr->n_target_ions; t++) {
            a.free_capacity[ion_states[instr->target_ions[t] - 1] - zones]++;
        }
    }

    int status = 0;
    double *shortest = calloc(n_ops, sizeof(double));

    for (int k = 0; k < n_ops && status == 0; k++) {
        struct InstructionNode *instr = &circuitGraph->instruction_nodes[ops[k]];
        enum ZoneType type = operation_zone_type(instr);

        a.n_targets[k] = instr->n_target_ions;
        a.candidates[k] = malloc(n_zones * sizeof(int));
        a.costs[k] = malloc(n_zones * sizeof(double));
        a.sums[k] = malloc(n_zones * sizeof(double));

        for (int z = 0; z < n_zones; z++) {
            if (!(zones[z].zone_type & type) || a.free_capacity[z] < instr->n_target_ions) {
                continue;
            }

            double cost = 0.0;
            double sum = 0.0;
            for (int t = 0; t < instr->n_target_ions; t++) {
                double *position = ion_states[instr->target_ions[t] - 1]->position;
                double distance = sqrt(pow(zones[z].position[0] - position[0], 2) + pow(zones[z].position[1] - position[1], 2) + pow(zones[z].position[2] - position[2], 2));

                cost = MAX(cost, distance);
                sum += distance;
            }

            // Insert in order of cost, then total distance
            int c = a.n_candidates[k]++;
            while (c > 0 && (a.costs[k][c - 1] > cost || (a.costs[k][c - 1] == cost && a.sums[k][c - 1] > sum))) {
                a.candidates[k][c] = a.candidates[k][c - 1];
                a.costs[k][c] = a.costs[k][c - 1];
                a.sums[k][c] = a.sums[k][c - 1];
                c--;
            }
            a.candidates[k][c] = z;
            a.costs[k][c] = cost;
            a.sums[k][c] = sum;
        }

        if (a.n_candidates[k] == 0) {
            status = 1;
        } else {
            shortest[k] = a.costs[k][0];
        }
    }

    // The operations of each type of zone need no more room than the zones of that type have, which rules out most
    // layers that do not fit without a search
    for (int k = 0; k < n_ops && status == 0; k++) {
        enum ZoneType type = operation_zone_type(&circuitGraph->instruction_nodes[ops[k]]);

        int needed = 0;
        for (int j = 0; j < n_ops; j++) {
            needed += (operation_zone_type(&circuitGraph->instruction_nodes[ops[j]]) == type) ? a.n_targets[j] : 0;
        }

        int room = 0;
        for (int z = 0; z < n_zones; z++) {
            room += (zones[z].zone_type & type) ? MAX(a.free_capacity[z], 0) : 0;
        }

        status = (needed > room);
    }

    if (status == 0) {
        // Operations whose shortest transport is longest go first, since they bound the layer
        for (int k = 0; k < n_ops; k++) {
            int j = k;
            while (j > 0 && shortest[a.op_order[j - 1]] < shortest[k]) {
                a.op_order[j] = a.op_order[j - 1];
                j--;
            }
            a.op_order[j] = k;
        }

        search_layer_assignment(&a, 0, 0.0, 0.0);

        if (a.best_max == HUGE_VAL) {
            status = 1;
        } else {
            for (int k = 0; k < n_ops; k++) {
                op_zones[k] = &zones[a.best_choice[k]];
            }
        }
    }

    for (int k = 0; k < n_ops; k++) {
        free(a.candidates[k]);
        free(a.costs[k]);
        free(a.sums[k]);
    }

    free(shortest);
    free(a.op_order);
    free(a.n_targets);
    free(a.n_candidates);
    free(a.candidates);
    free(a.costs);
    free(a.sums);
    free(a.free_capacity);
    free(a.choice);
    free(a.best_choice);

    return status;
}

// Scheduler which visits the instructions of each layer through the circuit sorted by layer; with SCHEDULE_FAST it gives
// the same schedule as schedule_baseline, while the other strategies split each layer into rounds which fit in the
// zones, SCHEDULE_LOOKAHEAD and SCHEDULE_LAYER_ASSIGN leave ions in their zone for an operation at most lookahead_layers
// layers later, and SCHEDULE_LAYER_ASSIGN also assigns the zones of each round together. Sets the round of each
// instruction, and returns the number of rounds, or -1 if the circuit cannot be scheduled
static int schedule_sorted
(
    struct TrapGraph *trapGraph,
    struct CircuitGraph *circuitGraph,
    struct TransportProgram *transportProgram,
    struct ZoneNode **ion_states,
    enum ScheduleStrategy strategy,
    int lookahead_layers,
    int *rounds,
    struct ScheduleTrace *trace
)
{
    int n_ions = transportProgram->n_ions;
    int n_nodes = circuitGraph->n_instruction_nodes;

    int *order;
    int *layer_starts;
    int max_layer = sort_circuit_layers(circuitGraph, &order, &layer_starts);

    // Operations of each ion in order (which is also the order of their layers), to look ahead to its next operation
    int *ion_op_starts = calloc(n_ions + 1, sizeof(int));
    for (int i = 0; i < n_nodes; i++) {
        for (int t = 0; t < circuitGraph->instruction_nodes[i].n_target_ions; t++) {
            ion_op_starts[circuitGraph->instruction_nodes[i].target_ions[t]]++;
        }
    }
    for (int ion = 1; ion <= n_ions; ion++) {
        ion_op_starts[ion] += ion_op_starts[ion - 1];
    }

    int *ion_ops = malloc(MAX(ion_op_starts[n_ions], 1) * sizeof(int));
    int *ion_next = malloc(n_ions * sizeof(int));
    memcpy(ion_next, ion_op_starts, n_ions * sizeof(int));
    for (int i = 0; i < n_nodes; i++) {
        for (int t = 0; t < circuitGraph->instruction_nodes[i].n_target_ions; t++) {
            int ion = circuitGraph->instruction_nodes[i].target_ions[t] - 1;
            ion_ops[ion_next[ion]++] = i;
        }
    }
    memcpy(ion_next, ion_op_starts, n_ions * sizeof(int));

    int *keep = calloc(n_ions, sizeof(int));
    int *next_ops = malloc(n_ions * sizeof(int));
    int *candidates = malloc(MAX(n_ions, 1) * sizeof(int));
    int *zone_reserved = malloc(MAX(trapGraph->n_zones, 1) * sizeof(int));
    int *pending = malloc(MAX(n_nodes, 1) * sizeof(int));
    int *round_ops = malloc(MAX(n_nodes, 1) * sizeof(int));
    int *trial_ops = malloc(MAX(n_nodes, 1) * sizeof(int));
    struct ZoneNode **op_zones = malloc(MAX(n_nodes, 1) * sizeof(struct ZoneNode*));

    int n_rounds = 0;
    int status = 0;

    for (int layer = 1; layer <= max_layer && status == 0; layer++) {
        int n_pending = layer_starts[layer] - layer_starts[layer - 1];

        for (int k = 0; k < n_pending; k++) {
            int i = order[layer_starts[layer - 1] + k];
            struct InstructionNode *instr = &circuitGraph->instruction_nodes[i];

            pending[k] = i;
            transportProgram->transport_n_targets_list[i] = instr->n_target_ions;
            transportProgram->transport_targets_list[i] = (int *)malloc(sizeof(int) * instr->n_target_ions);

            if (!operation_zone_type(instr)) {
                printf("ERROR: No available instruction zone for instruction %d.\n", instr->graphindex);
                status = 1;
            }
        }

        if (status != 0) {
            break;
        }

        if (strategy == SCHEDULE_FAST) {
            // Ions which already took part in the layer are released along with the others when the zones are full, as
            // in schedule_baseline
            n_rounds++;

            for (int k = 0; k < n_pending; k++) {
                struct InstructionNode *instr = &circuitGraph->instruction_nodes[pending[k]];
                enum ZoneType instructionType = operation_zone_type(instr);

                struct ZoneNode *instructionZone = select_operation_zone(trapGraph, instructionType, ion_states, instr->target_ions, instr->n_target_ions);
                if (!instructionZone) {
                    release_ions(trapGraph, ion_states, n_ions, NULL, trace);
                    instructionZone = select_operation_zone(trapGraph, instructionType, ion_states, instr->target_ions, instr->n_target_ions);
                }

                if (!instructionZone) {
                    printf("ERROR: No available instruction zone for instruction %d.\n", instr->graphindex);
                    status = 1;
                    break;
                }

                rounds[pending[k]] = n_rounds;
                transport_ions(trapGraph, ion_states, instr, pending[k], instructionZone, transportProgram, trace);
            }

            if (status != 0) {
                break;
            }
        }

        // Rounds of the operations of the layer which fit in the zones at once
        while (strategy != SCHEDULE_FAST && n_pending > 0) {
            n_rounds++;

            // When looking ahead, the operations whose ions already wait in a zone where they can happen go first, so
            // that other operations do not take their room
            if (strategy == SCHEDULE_LOOKAHEAD || strategy == SCHEDULE_LAYER_ASSIGN) {
                int n_first = 0;
                for (int k = 0; k < n_pending; k++) {
                    struct InstructionNode *instr = &circuitGraph->instruction_nodes[pending[k]];

                    int waiting = 0;
                    for (int t = 0; t < instr->n_target_ions; t++) {
                        waiting |= (ion_states[instr->target_ions[t] - 1]->zone_type & operation_zone_type(instr)) != 0;
                    }

                    if (waiting) {
                        int op = pending[k];
                        memmove(&pending[n_first + 1], &pending[n_first], (k - n_first) * sizeof(int));
                        pending[n_first++] = op;
                    }
                }
            }

            // Make room by moving the ions which the rest of the layer does not use out of the operation zones
            memset(keep, 0, n_ions * sizeof(int));
            for (int k = 0; k < n_pending; k++) {
                struct InstructionNode *instr = &circuitGraph->instruction_nodes[pending[k]];
                for (int t = 0; t < instr->n_target_ions; t++) {
                    keep[instr->target_ions[t] - 1] = 1;
                }
            }

            int n_round = 0;

            // If ions waiting in the operation zones for later operations of the layer block every operation, move
            // them out too and try again
            for (int attempt = 0; attempt < 2 && n_round == 0; attempt++) {
                if (attempt == 1) {
                    release_ions(trapGraph, ion_states, n_ions, NULL, trace);
                }

                if (strategy == SCHEDULE_LAYER_ASSIGN) {
                    if (assign_layer_zones(trapGraph, circuitGraph, ion_states, pending, n_pending, op_zones) == 0) {
                        memcpy(round_ops, pending, n_pending * sizeof(int));
                        n_round = n_pending;
                    } else {
                        release_ions(trapGraph, ion_states, n_ions, keep, trace);

                        if (assign_layer_zones(trapGraph, circuitGraph, ion_states, pending, n_pending, op_zones) == 0) {
                            memcpy(round_ops, pending, n_pending * sizeof(int));
                            n_round = n_pending;
                        } else {
                            // Take the operations in order while they still fit together
                            for (int k = 0; k < n_pending; k++) {
                                memcpy(trial_ops, round_ops, n_round * sizeof(int));
                                trial_ops[n_round] = pending[k];

                                if (assign_layer_zones(trapGraph, circuitGraph, ion_states, trial_ops, n_round + 1, op_zones) == 0) {
                                    round_ops[n_round++] = pending[k];
                                }
                            }

                            if (n_round > 0) {
                                assign_layer_zones(trapGraph, circuitGraph, ion_states, round_ops, n_round, op_zones);
                            }
                        }
                    }

                    for (int k = 0; k < n_round; k++) {
                        rounds[round_ops[k]] = n_rounds;
                        transport_ions(trapGraph, ion_states, &circuitGraph->instruction_nodes[round_ops[k]], round_ops[k], op_zones[k], transportProgram, trace);
                    }
                } else {
                    // Nearest zone for each operation in turn, leaving the operations which do not fit for the next round
                    int released = 0;

                    for (int k = 0; k < n_pending; k++) {
                        struct InstructionNode *instr = &circuitGraph->instruction_nodes[pending[k]];
                        enum ZoneType instructionType = operation_zone_type(instr);

                        struct ZoneNode *instructionZone = select_operation_zone(trapGraph, instructionType, ion_states, instr->target_ions, instr->n_target_ions);
                        if (!instructionZone && !released) {
                            release_ions(trapGraph, ion_states, n_ions, keep, trace);
                            released = 1;
                            instructionZone = select_operation_zone(trapGraph, instructionType, ion_states, instr->target_ions, instr->n_target_ions);
                        }

                        if (!instructionZone) {
                            continue;
                        }

                        rounds[pending[k]] = n_rounds;
                        round_ops[n_round++] = pending[k];
                        transport_ions(trapGraph, ion_states, instr, pending[k], instructionZone, transportProgram, trace);
                    }
                }
            }

            if (n_round == 0) {
                printf("ERROR: No available instruction zone for instruction %d.\n", circuitGraph->instruction_nodes[pending[0]].graphindex);
                status = 1;
                break;
            }

            // Operations left for the next round
            int n_left = 0;
            for (int k = 0; k < n_pending; k++) {
                int done = 0;
                for (int r = 0; r < n_round; r++) {
                    done |= (round_ops[r] == pending[k]);
                }

                if (!done) {
                    pending[n_left++] = pending[k];
                }
            }
            n_pending = n_left;

            // The ions of this round are done with the layer, so they make room for the next round
            if (n_pending > 0) {
                memset(keep, 1, n_ions * sizeof(int));
                for (int r = 0; r < n_round; r++) {
                    struct InstructionNode *instr = &circuitGraph->instruction_nodes[round_ops[r]];
                    for (int t = 0; t < instr->n_target_ions; t++) {
                        keep[instr->target_ions[t] - 1] = 0;
                    }
                }

                release_ions(trapGraph, ion_states, n_ions, keep, trace);
            }
        }

        if (status != 0) {
            break;
        }

        // Move ions to storage/empty zones once the layer is done, except (when looking ahead) the ions whose next
        // operation is at most lookahead_layers layers later and can happen in the zone they are in
        memset(keep, 0, n_ions * sizeof(int));

        if (strategy == SCHEDULE_LOOKAHEAD || strategy == SCHEDULE_LAYER_ASSIGN) {
            memset(zone_reserved, 0, trapGraph->n_zones * sizeof(int));

            // Ions which could stay, in order of the layer of their next operation (and then by ion), so that the
            // operations which come soonest get the room in each zone first
            int n_candidates = 0;
            for (int ion = 0; ion < n_ions; ion++) {
                while (ion_next[ion] < ion_op_starts[ion + 1] && circuitGraph->layer_idxs[ion_ops[ion_next[ion]]] <= layer) {
                    ion_next[ion]++;
                }

                next_ops[ion] = -1;
                if (ion_next[ion] == ion_op_starts[ion + 1]) {
                    continue;
                }

                int next = ion_ops[ion_next[ion]];
                struct InstructionNode *instr = &circuitGraph->instruction_nodes[next];
                if (circuitGraph->layer_idxs[next] > layer + lookahead_layers || !(ion_states[ion]->zone_type & operation_zone_type(instr))) {
                    continue;
                }

                next_ops[ion] = next;

                int c = n_candidates++;
                while (c > 0 && circuitGraph->layer_idxs[next_ops[candidates[c - 1]]] > circuitGraph->layer_idxs[next]) {
                    candidates[c] = candidates[c - 1];
                    c--;
                }
                candidates[c] = ion;
            }

            for (int c = 0; c < n_candidates; c++) {
                int ion = candidates[c];
                int next = next_ops[ion];
                struct InstructionNode *instr = &circuitGraph->instruction_nodes[next];

                // Only as many ions stay in a zone as leave room for every ion of their next operations, so that the
                // ions they wait for can join them
                int shared = 0;
                for (int other = 0; other < c; other++) {
                    shared |= keep[candidates[other]] && ion_states[candidates[other]] == ion_states[ion] && next_ops[candidates[other]] == next;
                }

                int z = ion_states[ion] - trapGraph->zone_nodes;
                if (shared) {
                    keep[ion] = 1;
                } else if (zone_reserved[z] + instr->n_target_ions <= ion_states[ion]->zone_capacity) {
                    keep[ion] = 1;
                    zone_reserved[z] += instr->n_target_ions;
                }
            }
        }

        release_ions(trapGraph, ion_states, n_ions, keep, trace);
    }

    free(op_zones);
    free(trial_ops);
    free(round_ops);
    free(pending);
    free(zone_reserved);
    free(candidates);
    free(next_ops);
    free(keep);
    free(ion_next);
    free(ion_ops);
    free(ion_op_starts);
    free(order);
    free(layer_starts);

    return (status == 0) ? n_rounds : -1;
}

// Generates transport schedule given circuit data and trap data
struct TransportProgram* generate_transport_schedule
(
    struct TrapGraph *trapGraph,
    struct CircuitGraph *circuitGraph,
    struct TransportProgram* transportProgram
)
{
    return generate_transport_schedule_strategy(trapGraph, circuitGraph, transportProgram, DEFAULT_SCHEDULE_STRATEGY, NULL);
}

// Generates transport schedule given circuit data and trap data with a given strategy, recording every move of the ions
// in trace if it is not NULL
struct TransportProgram* generate_transport_schedule_strategy
(
    struct TrapGraph *trapGraph,
    struct CircuitGraph *circuitGraph,
    struct TransportProgram* transportProgram,
    enum ScheduleStrategy strategy,
    struct ScheduleTrace *trace
)
{
    struct ScheduleOptions options = { strategy, DEFAULT_LOOKAHEAD_LAYERS };

    return generate_transport_schedule_options(trapGraph, circuitGraph, transportProgram, &options, trace);
}

// Generates transport schedule given circuit data and trap data with the given options, recording every move of the ions
// in trace if it is not NULL
struct TransportProgram* generate_transport_schedule_options
(
    struct TrapGraph *trapGraph,
    struct CircuitGraph *circuitGraph,
    struct TransportProgram* transportProgram,
    const struct ScheduleOptions *options,
    struct ScheduleTrace *trace
)
{
    enum ScheduleStrategy strategy = options->strategy;
    int lookahead_layers = MAX(options->lookahead_layers, 1);

    // circuitGraph->n_ions must be initialized
    if (circuitGraph->n_ions <= 0) {
        printf("ERROR: Number of ions in circuit graph must be initialized.\n");
        return NULL;
    }

    // Set number of ions
    transportProgram->n_ions = circuitGraph->n_ions;

    // Set number of transports
    transportProgram->n_transports = circuitGraph->n_instruction_nodes;
    
    // Allocate space for transport program
    transportProgram->transport_n_targets_list = (int *)calloc(transportProgram->n_transports, sizeof(int));
    transportProgram->transport_targets_list = (int **)calloc(transportProgram->n_transports, sizeof(int*));
    transportProgram->arrival_times = (double *)calloc(transportProgram->n_transports, sizeof(double));
    transportProgram->destinations = calloc(transportProgram->n_transports, sizeof(double[3]));
    transportProgram->initial_positions = calloc(transportProgram->n_ions, sizeof(double[3]));

    // Every move is recorded (in a trace of our own if the caller does not want one) so that it can be timed
    struct ScheduleTrace local_trace = { 0 };
    if (trace == NULL) {
        trace = &local_trace;
    }

    // Keeps track of which zone ions are in
    struct ZoneNode **ion_states = malloc(transportProgram->n_ions * sizeof(struct ZoneNode*));
    
    // Assume ions start in load zone
    for (int i = 0; i < transportProgram->n_ions; i++) {
        ion_states[i] = select_available_zone(trapGraph, LOAD_ZONE);
        if (!ion_states[i]) {
            printf("ERROR: No available load zone for ion %d.\n", i + 1);
            free(ion_states);
            return NULL;
        }
        ion_states[i]->zone_occupancy++;
        memcpy(transportProgram->initial_positions[i], ion_states[i]->position, sizeof(double[3]));
    }

    // Find how many layers in circuit (may be redundant)
    int max_layer = 0;
    for (int i = 0; i < circuitGraph->n_instruction_nodes; i++) {
        if (schedule_verbose()) {
            printf("Instruction %d is in layer %d\n", i + 1, circuitGraph->layer_idxs[i]);
        }
        if (circuitGraph->layer_idxs[i] > max_layer) {
            max_layer = circuitGraph->layer_idxs[i];
        }
    }
    
    if (schedule_verbose()) {
        printf("Number of layers in circuit: %d\n\n", max_layer);
    }

    // Empty circuit
    if (max_layer == 0) {
        printf("ERROR: No layers in circuit graph.\n");
        free(ion_states);
        return NULL;
    }

    // Non-empty circuit, with the round of each instruction
    int *rounds = calloc(MAX(circuitGraph->n_instruction_nodes, 1), sizeof(int));
    int n_rounds;
    if (strategy == SCHEDULE_BASELINE) {
        n_rounds = (schedule_baseline(trapGraph, circuitGraph, transportProgram, ion_states, max_layer, rounds, trace) == 0) ? max_layer : -1;
    } else {
        n_rounds = schedule_sorted(trapGraph, circuitGraph, transportProgram, ion_states, strategy, lookahead_layers, rounds, trace);
    }

    free(ion_states);

    if (n_rounds < 0) {
        free(rounds);
        free_schedule_trace(&local_trace);
        return NULL;
    }

    if (schedule_verbose()) {
        printf("\nOUTPUT\n");
        print_zone_occupancy(trapGraph);
        printf("\n");
    }

    free(trace->transport_rounds);
    trace->n_rounds = n_rounds;
    trace->transport_rounds = rounds;

    // Time every move, including the moves back into storage, and the arrival of every transport
    time_schedule_trace(trapGraph, circuitGraph, transportProgram, trace);

    free_schedule_trace(&local_trace);

    return transportProgram;
}

int default_schedule_strategy(void)
{
    return DEFAULT_SCHEDULE_STRATEGY;
}

// Free the moves of a schedule trace
void free_schedule_trace(struct ScheduleTrace *trace)
{
    free(trace->moves);
    free(trace->transport_rounds);

    trace->moves = NULL;
    trace->transport_rounds = NULL;
    trace->n_rounds = 0;
    trace->n_moves = 0;
    trace->max_moves = 0;
}

// Helper to get operation type from QASM code, ignoring the parameters of the gate (e.g. "rz(0.5)"), which do not
// change where it happens
enum OperationType get_operation_type(const char *op)
{
    static const char *gates[] = {
        "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "rx", "ry", "rz", "p",
        "cx", "cy", "cz", "swap", "cp", "crx", "cry", "crz", "rxx", "ryy", "rzz",
    };

    size_t length = strcspn(op, "(");

    for (size_t g = 0; g < sizeof(gates) / sizeof(gates[0]); g++) {
        if (strlen(gates[g]) == length && strncmp(op, gates[g], length) == 0) return QUBIT;
    }

    if (strcmp(op, "measure") == 0) return DETECT;

    return 0;
}

// Helper to parse qubit indices from QASM CODE
int parse_qubit_index(const char *token)
{
    if (strncmp(token, "qreg[", 5) != 0) {
        return -1;
    }

    const char *lbracket = strchr(token, '[');
    if (!lbracket) return -1;

    return atoi(lbracket + 1);
}

// Helper to determine whether a QASM line is an instruction node
int is_instruction_line(const char *line)
{
    // skip leading whitespace
    while (isspace(*line)) line++;

    // ignore empty lines
    if (*line == '\0' || *line == '\n') return 0;

    // ignore declarations
    if (strncmp(line, "qubit", 5) == 0) return 0;
    if (strncmp(line, "bit", 3) == 0) return 0;

    // ignore comments
    if (strncmp(line, "//", 2) == 0) return 0;

    return 1;
}

// Helper to determine number of instruction nodes in QASM code
// int count_instruction_nodes(const char *filename)
// {
//     FILE *fp = fopen(filename, "r");
//     if (!fp) {
//         printf("ERROR: Could not open file\n");
//         return -1;
//     }

//     char line[256];
//     int count = 0;

//     while (fgets(line, sizeof(line), fp)) {
//         if (is_instruction_line(line)) {
//             count++;
//         }
//     }

//     fclose(fp);
//     return count;
// }

// Helper to get find number of qubits from QASM CODE
// int get_max_qubit_index(const char *filename)
// {
//     FILE *fp = fopen(filename, "r");
//     if (!fp) {
//         printf("ERROR: Could not open file\n");
//         return -1;
//     }

//     char line[256];
//     int max_index = -1;

//     while (fgets(line, sizeof(line), fp)) {

//         char *token = strtok(line, " ,;\n");
//         while (token) {

//             int idx = parse_qubit_index(token);
//             if (idx >= 0 && idx > max_index) {
//                 max_index = idx;
//             }

//             token = strtok(NULL, " ,;\n");
//         }
//     }

//     fclose(fp);
//     return max_index;
// }

// Parses QASM file and turns it into circuit graph
struct CircuitGraph* parse_qasm(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        printf("ERROR: Could not open file\n");
        return NULL;
    }

    int nodeCount = 0;
    int max_index = -1;
    char line[256];

    while (fgets(line, sizeof(line), fp)) {
        char line_copy[256];
        strcpy(line_copy, line);

        if (is_instruction_line(line_copy)) {
            nodeCount++;
        }

        char *token = strtok(line, " ,;\n");
        while (token) {
            int idx = parse_qubit_index(token);
            if (idx >= 0 && idx > max_index) {
                max_index = idx;
            }

            

            token = strtok(NULL, " ,;\n");
        }
    }

    struct CircuitGraph *graph = (struct CircuitGraph *)malloc(sizeof(struct CircuitGraph));
    graph->n_ions = max_index + 1;
    graph->n_instruction_nodes = nodeCount;
    graph->instruction_nodes = (struct InstructionNode *)malloc(sizeof(struct InstructionNode) * nodeCount);
    graph->layer_idxs = (int *)malloc(sizeof(int) * nodeCount);

    int *last_node_idxs = (int *)malloc(sizeof(int) * (graph->n_ions > 0 ? graph->n_ions : 1));
    for (int i = 0; i < graph->n_ions; i++) {
        last_node_idxs[i] = -1;
    }

    int count = 0;
    int line_number = 0;

    rewind(fp);

    printf("Parsing QASM file...\n");

    while (fgets(line, sizeof(line), fp)) {
        line_number++;
        printf("\nParsing line: %s", line);

        // skip empty lines
        if (strlen(line) < 2) continue;

        // skip declarations
        if (strstr(line, "qubit") || strstr(line, "bit")) continue;

        // tokenize
        char *token = strtok(line, " \t,;\r\n");
        if (!token) continue;

        // skip comments, headers and declarations, which are not gates
        if (strncmp(token, "//", 2) == 0 || strcmp(token, "OPENQASM") == 0 || strcmp(token, "include") == 0 ||
            strcmp(token, "barrier") == 0 || strcmp(token, "qreg") == 0 || strcmp(token, "creg") == 0) continue;

        // A gate which is not known would otherwise be left out of the schedule
        enum OperationType op_type = get_operation_type(token);
        if (op_type == 0) {
            printf("ERROR: Circuit contains unknown gate '%s' on line %d\n", token, line_number);

            fclose(fp);
            free(last_node_idxs);
            graph->n_instruction_nodes = count;
            free_circuit_graph(graph);

            return NULL;
        }

        // parse targets
        int targets[graph->n_ions > 0 ? graph->n_ions : 1];
        int n_targets = 0;

        while ((token = strtok(NULL, " ,;\n"))) {
            if (strstr(token, "qreg")) {
                n_targets = add_qasm_targets(token, graph->n_ions, targets, n_targets);
            }
        }

        struct InstructionNode node;
        set_prev_nodes(graph, last_node_idxs, targets, n_targets, &node);

        node.operation_type = op_type;
        node.operation_duration = 0.0;
        node.graphindex = count + 1;
        node.n_target_ions = n_targets;
        node.target_ions = malloc(sizeof(int) * n_targets);
        for (int i = 0; i < n_targets; i++) {
            node.target_ions[i] = targets[i];
        }

        printf("Instruction %d: Operation Type: %d, Num Targets: %d, Targets: ",
            node.graphindex,
            node.operation_type,
            node.n_target_ions
        );

        for (int t = 0; t < node.n_target_ions; t++) {
            printf("%d ", node.target_ions[t]);
        }
        printf("\n");

        for (int i = 0; i < n_targets; i++) {
            node.target_ions[i] = targets[i];
        }

        graph->instruction_nodes[count] = node;
        for (int i = 0; i < n_targets; i++) {
            last_node_idxs[targets[i] - 1] = count;
        }
        count++;
    }

    fclose(fp);
    free(last_node_idxs);

    // Only lines with recognized operations become instruction nodes
    graph->n_instruction_nodes = count;
    compute_circuit_layers(graph);

    printf("\nFinished parsing QASM file.\n");

    for (int i = 0; i < graph->n_instruction_nodes; i++) {
        printf("Instruction %d: Operation Type: %d, Num Targets: %d, Targets: ",
            i + 1,
            graph->instruction_nodes[i].operation_type,
            graph->instruction_nodes[i].n_target_ions
        );

        for (int t = 0; t < graph->instruction_nodes[i].n_target_ions; t++) {
            printf("%d ", graph->instruction_nodes[i].target_ions[t]);
        }
        printf("\n");
    }

    printf("\nParsed QASM file: %d instruction nodes, number of qubits: %d\n\n", graph->n_instruction_nodes, graph->n_ions);

    return graph;
}

struct TransportProgram* simple_transport_schedule()
{
    struct ZoneNode L1 = { LOAD_ZONE, 5, 0};
    struct ZoneNode E1 = { EMPTY_ZONE, 3, 0};
    struct ZoneNode S1 = { STORAGE_ZONE, 10, 0};
    struct ZoneNode E2 = { EMPTY_ZONE, 3, 0};
    struct ZoneNode G1 = { GATE_ZONE, 2, 0};
    struct ZoneNode E3 = { EMPTY_ZONE, 3, 0};
    struct ZoneNode M1 = { MEASURE_ZONE, 1, 0};

    struct ZoneNode Edge1Zones[2] = { L1, E1 };
    struct TransportEdge Edge1 = { LINEAR_SEGMENT, 2, Edge1Zones };
    struct ZoneNode Edge2Zones[2] = { E1, S1 };
    struct TransportEdge Edge2 = { LINEAR_SEGMENT, 2, Edge2Zones };
    struct ZoneNode Edge3Zones[2] = { S1, E2 };
    struct TransportEdge Edge3 = { LINEAR_SEGMENT, 2, Edge3Zones };
    struct ZoneNode Edge4Zones[2] = { E2, G1 };
    struct TransportEdge Edge4 = { LINEAR_SEGMENT, 2, Edge4Zones };
    struct ZoneNode Edge5Zones[2] = { G1, E3 };
    struct TransportEdge Edge5 = { LINEAR_SEGMENT, 2, Edge5Zones };
    struct ZoneNode Edge6Zones[2] = { E3, M1 };
    struct TransportEdge Edge6 = { LINEAR_SEGMENT, 2, Edge6Zones };

    struct ZoneNode trapGraphZones[7] = { L1, E1, S1, E2, G1, E3, M1 };
    struct TransportEdge trapGraphEdges[7] = { Edge1, Edge2, Edge3, Edge4, Edge5, Edge6 };
    
    struct TrapGraph trapGraphObject = { 7, 6, trapGraphZones, trapGraphEdges };
    struct TrapGraph *trapGraph = &trapGraphObject;

    int HTargetIons[1] = { 1 };
    struct InstructionNode H = { QUBIT, 0, 1, HTargetIons, 1, 0, NULL};
    int XTargetIons[1] = { 1 };
    struct InstructionNode X = { QUBIT, 0, 1, XTargetIons, 2, 0, NULL};
    int CXTargetIons[2] = { 1, 2 };
    struct InstructionNode CXPrevNodes[2] = { H, X };
    struct InstructionNode CX = { QUBIT, 0, 2, CXTargetIons, 3, 2, CXPrevNodes};

    struct InstructionNode circuitInstructions[3] = { H, X, CX };
    int circuitLayers[3] = { 1, 1, 2 };
    struct CircuitGraph circuitGraphObject = {2, 3, circuitInstructions, circuitLayers};
    struct CircuitGraph *circuitGraph = &circuitGraphObject;

    struct TransportProgram *transportProgram = (struct TransportProgram*)malloc(sizeof(struct TransportProgram));
    transportProgram = generate_transport_schedule(trapGraph, circuitGraph, transportProgram);
    if (transportProgram == NULL) {
        return NULL;
    }
    
    print_transport_program(transportProgram);

    return transportProgram;
}

struct TransportProgram* simple_transport_schedule_qasm_input()
{
    struct ZoneNode L1 = { LOAD_ZONE, 5, 0};
    struct ZoneNode E1 = { EMPTY_ZONE, 3, 0};
    struct ZoneNode S1 = { STORAGE_ZONE, 10, 0};
    struct ZoneNode E2 = { EMPTY_ZONE, 3, 0};
    struct ZoneNode G1 = { GATE_ZONE, 2, 0};
    struct ZoneNode E3 = { EMPTY_ZONE, 3, 0};
    struct ZoneNode M1 = { MEASURE_ZONE, 1, 0};

    struct ZoneNode Edge1Zones[2] = { L1, E1 };
    struct TransportEdge Edge1 = { LINEAR_SEGMENT, 2, Edge1Zones };
    struct ZoneNode Edge2Zones[2] = { E1, S1 };
    struct TransportEdge Edge2 = { LINEAR_SEGMENT, 2, Edge2Zones };
    struct ZoneNode Edge3Zones[2] = { S1, E2 };
    struct TransportEdge Edge3 = { LINEAR_SEGMENT, 2, Edge3Zones };
    struct ZoneNode Edge4Zones[2] = { E2, G1 };
    struct TransportEdge Edge4 = { LINEAR_SEGMENT, 2, Edge4Zones };
    struct ZoneNode Edge5Zones[2] = { G1, E3 };
    struct TransportEdge Edge5 = { LINEAR_SEGMENT, 2, Edge5Zones };
    struct ZoneNode Edge6Zones[2] = { E3, M1 };
    struct TransportEdge Edge6 = { LINEAR_SEGMENT, 2, Edge6Zones };

    struct ZoneNode trapGraphZones[7] = { L1, E1, S1, E2, G1, E3, M1 };
    struct TransportEdge trapGraphEdges[7] = { Edge1, Edge2, Edge3, Edge4, Edge5, Edge6 };
    
    struct TrapGraph trapGraphObject = { 7, 6, trapGraphZones, trapGraphEdges };
    struct TrapGraph *trapGraph = &trapGraphObject;

    struct CircuitGraph *circuitGraph = parse_qasm("tests/quantumcircuits/test_0000.qasm");

    if (!circuitGraph) {
        printf("ERROR: Failed to parse QASM file.\n");
        return NULL;
    }

    struct TransportProgram *transportProgram = (struct TransportProgram*)malloc(sizeof(struct TransportProgram));
    transportProgram = generate_transport_schedule(trapGraph, circuitGraph, transportProgram);

    if (transportProgram == NULL) {
        return NULL;
    }
    
    print_transport_program(transportProgram);

    return transportProgram;
}

void print_transport_program(struct TransportProgram *transportProgram)
{
    printf("Number of ions: %d\n", transportProgram->n_ions);
    printf("Number of transports: %d\n", transportProgram->n_transports);
    
    for (int i = 0; i < transportProgram->n_transports; i++) {
        int n_targets = transportProgram->transport_n_targets_list[i];

        printf("\nTransport %d: %d targets", i + 1, n_targets);

        for (int t = 0; t < n_targets; t++) {
            printf("\n  Target %d: Ion %d", t + 1,
                transportProgram->transport_targets_list[i][t]);
        }

        printf("\n");
    }
}

