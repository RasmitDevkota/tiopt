#include "transport.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

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

// Original scheduler, which runs every layer at once: when the zones are full, it moves every ion out of them,
// including the ions of operations placed earlier in the layer, so its schedules are only valid if every layer fits
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
                    release_ions(trapGraph, ion_states, transportProgram->n_ions, trace);
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
        release_ions(trapGraph, ion_states, transportProgram->n_ions, trace);

        print_zone_occupancy(trapGraph);
    }

    return 0;
}

// Scheduler which splits each layer into rounds of operations which fit in the zones at once: each operation goes to
// the nearest zone of its type with room for its ions, the operations which do not fit wait for the next round, and
// the ions go back into storage after each round. Sets the round of each instruction, and returns the number of
// rounds, or -1 if an operation does not fit in any zone
static int schedule_rounds
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
    int *pending = malloc((circuitGraph->n_instruction_nodes > 0 ? circuitGraph->n_instruction_nodes : 1) * sizeof(int));
    int n_rounds = 0;
    int status = 0;

    for (int layer = 1; layer <= max_layer && status == 0; layer++) {
        int n_pending = 0;
        for (int i = 0; i < circuitGraph->n_instruction_nodes; i++) {
            if (circuitGraph->layer_idxs[i] != layer) {
                continue;
            }

            struct InstructionNode *instr = &circuitGraph->instruction_nodes[i];

            transportProgram->transport_n_targets_list[i] = instr->n_target_ions;
            transportProgram->transport_targets_list[i] = (int *)malloc(sizeof(int) * instr->n_target_ions);
            pending[n_pending++] = i;
        }

        while (n_pending > 0) {
            n_rounds++;

            // Nearest zone for each operation in turn, leaving the operations which do not fit for the next round
            int n_left = 0;
            for (int k = 0; k < n_pending; k++) {
                struct InstructionNode *instr = &circuitGraph->instruction_nodes[pending[k]];
                enum ZoneType instructionType = operation_zone_type(instr);

                if (!instructionType) {
                    printf("ERROR: No available instruction zone for instruction %d.\n", instr->graphindex);
                    status = 1;
                    break;
                }

                struct ZoneNode *instructionZone = select_operation_zone(trapGraph, instructionType, ion_states, instr->target_ions, instr->n_target_ions);
                if (!instructionZone) {
                    pending[n_left++] = pending[k];
                    continue;
                }

                rounds[pending[k]] = n_rounds;
                transport_ions(trapGraph, ion_states, instr, pending[k], instructionZone, transportProgram, trace);
            }

            if (status != 0) {
                break;
            }

            if (n_left == n_pending) {
                printf("ERROR: No available instruction zone for instruction %d.\n", circuitGraph->instruction_nodes[pending[0]].graphindex);
                status = 1;
                break;
            }

            n_pending = n_left;

            // Move ions to storage/empty zones once the round is done, to make room for the next round
            release_ions(trapGraph, ion_states, transportProgram->n_ions, trace);

            print_zone_occupancy(trapGraph);
        }
    }

    free(pending);

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
// and the round of each transport in trace if it is not NULL
struct TransportProgram* generate_transport_schedule_strategy
(
    struct TrapGraph *trapGraph,
    struct CircuitGraph *circuitGraph,
    struct TransportProgram* transportProgram,
    enum ScheduleStrategy strategy,
    struct ScheduleTrace *trace
)
{
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
    int *rounds = calloc(circuitGraph->n_instruction_nodes, sizeof(int));
    int n_rounds;
    if (strategy == SCHEDULE_BASELINE) {
        n_rounds = (schedule_baseline(trapGraph, circuitGraph, transportProgram, ion_states, max_layer, rounds, trace) == 0) ? max_layer : -1;
    } else {
        n_rounds = schedule_rounds(trapGraph, circuitGraph, transportProgram, ion_states, max_layer, rounds, trace);
    }

    free(ion_states);

    if (n_rounds < 0) {
        free(rounds);
        return NULL;
    }

    if (schedule_verbose()) {
        printf("\nOUTPUT\n");
        print_zone_occupancy(trapGraph);
        printf("\n");
    }

    schedule_arrival_times_in_rounds(circuitGraph, transportProgram, rounds);

    if (trace != NULL) {
        free(trace->transport_rounds);
        trace->n_rounds = n_rounds;
        trace->transport_rounds = rounds;
    } else {
        free(rounds);
    }

    return transportProgram;
}

// Free the moves and rounds of a schedule trace
void free_schedule_trace(struct ScheduleTrace *trace)
{
    free(trace->moves);
    free(trace->transport_rounds);

    trace->moves = NULL;
    trace->transport_rounds = NULL;
    trace->n_moves = 0;
    trace->max_moves = 0;
    trace->n_rounds = 0;
}

// Helper to get operation type from QASM code
enum OperationType get_operation_type(const char *op)
{
    if (strcmp(op, "h") == 0) return QUBIT;
    if (strcmp(op, "x") == 0) return QUBIT;
    if (strcmp(op, "y") == 0) return QUBIT;
    if (strcmp(op, "z") == 0) return QUBIT;

    if (strcmp(op, "cx") == 0) return QUBIT;
    if (strcmp(op, "cy") == 0) return QUBIT;
    if (strcmp(op, "cz") == 0) return QUBIT;

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

