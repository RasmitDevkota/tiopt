#include "transport.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "data_structures.h"

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

// Generates transport schedule given circuit data and trap data
struct TransportProgram* generate_transport_schedule
(
    struct TrapGraph *trapGraph,
    struct CircuitGraph *circuitGraph,
    struct TransportProgram* transportProgram
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
    transportProgram->transport_n_targets_list = (int *)malloc(sizeof(int) * transportProgram->n_transports);
    transportProgram->transport_targets_list = (int **)malloc(sizeof(int*) * transportProgram->n_transports);

    // Keeps track of which zone ions are in
    struct ZoneNode **ion_states = malloc(transportProgram->n_ions * sizeof(struct ZoneNode*));
    
    // Assume ions start in load zone
    for (int i = 0; i < transportProgram->n_ions; i++) {
        ion_states[i] = select_available_zone(trapGraph, LOAD_ZONE);
        if (!ion_states[i]) {
            printf("ERROR: No available load zone for ion %d.\n", i + 1);
            return NULL;
        }
        ion_states[i]->zone_occupancy++;
    }

    // Find how many layers in circuit (may be redundant)
    int max_layer = 0;
    for (int i = 0; i < circuitGraph->n_instruction_nodes; i++) {
        printf("Instruction %d is in layer %d\n", i + 1, circuitGraph->layer_idxs[i]);
        if (circuitGraph->layer_idxs[i] > max_layer) {
            max_layer = circuitGraph->layer_idxs[i];
        }
    }
    
    printf("Number of layers in circuit: %d\n\n", max_layer);

    // Empty circuit
    if (max_layer == 0) {
        printf("ERROR: No layers in circuit graph.\n");
        return NULL;
    }

    // Non-empty circuit
    for (int layer = 1; layer <= max_layer; layer++) {
        for (int i = 0; i < circuitGraph->n_instruction_nodes; i++) {
            if (circuitGraph->layer_idxs[i] != layer) {
                continue;
            }

            struct InstructionNode *instr = &circuitGraph->instruction_nodes[i];
            struct ZoneNode *instructionZone = NULL;

            transportProgram->transport_n_targets_list[i] = instr->n_target_ions;
            transportProgram->transport_targets_list[i] = (int *)malloc(sizeof(int) * transportProgram->transport_n_targets_list[i]);
            
            // Find instruction zone for each instruction
            if (instr->operation_type & QUBIT) {
                instructionZone = select_available_zone(trapGraph, GATE_ZONE);
                if (!instructionZone) {
                    printf("ERROR: No available gate zone for instruction %d.\n", instr->graphindex);
                    return NULL;
                }
            } else if (instr->operation_type & COOL) {
                instructionZone = select_available_zone(trapGraph, COOL_ZONE);
                if (!instructionZone) {
                    printf("ERROR: No available cool zone for instruction %d.\n", instr->graphindex);
                    return NULL;
                }
            } else if (instr->operation_type & DETECT) {
                instructionZone = select_available_zone(trapGraph, MEASURE_ZONE);
                if (!instructionZone) {
                    printf("ERROR: No available measure zone for instruction %d.\n", instr->graphindex);
                    return NULL;
                }
            }

            if (!instructionZone) {
                printf("ERROR: No available instruction zone for instruction %d.\n", instr->graphindex);
                return NULL;
            }

            for (int i = 0; i < trapGraph->n_zones; i++) {
                printf("Zone %d Occupancy: %d\n", i + 1, trapGraph->zone_nodes[i].zone_occupancy);
                printf("Zone %d Capacity: %d\n", i + 1, trapGraph->zone_nodes[i].zone_capacity);
                printf("\n");
            }

            // Move each target ion to a the instruction zone
            for (int t = 0; t < instr->n_target_ions; t++) {
                int ion_id = instr->target_ions[t];

                transportProgram->transport_targets_list[i][t] = ion_id;

                // Update occupancy
                ion_states[ion_id - 1]->zone_occupancy--;
                instructionZone->zone_occupancy++;


                ion_states[ion_id - 1] = instructionZone;
            }

            for (int i = 0; i < trapGraph->n_zones; i++) {
                printf("Zone %d Occupancy: %d\n", i + 1, trapGraph->zone_nodes[i].zone_occupancy);
                printf("Zone %d Capacity: %d\n", i + 1, trapGraph->zone_nodes[i].zone_capacity);
                printf("\n");
            }

            // Move ions to storage/empty zones
            for (int t = 0; t < instr->n_target_ions; t++) {
                int ion_id = instr->target_ions[t];

                struct ZoneNode *dest = select_available_zone(trapGraph, STORAGE_ZONE | EMPTY_ZONE);
                if (!dest) {
                    printf("ERROR: No available storage/empty zone after instruction %d.\n", instr->graphindex);
                    continue;
                }

                // Update occupancy
                for (int z = 0; z < trapGraph->n_zones; z++) {
                    if (&trapGraph->zone_nodes[z] == ion_states[ion_id - 1]) {
                        ion_states[ion_id - 1]->zone_occupancy--;
                    }
                    if (&trapGraph->zone_nodes[z] == dest) {
                        dest->zone_occupancy++;
                    }
                }

                ion_states[ion_id - 1] = dest;
            }

            for (int i = 0; i < trapGraph->n_zones; i++) {
                printf("Zone %d Occupancy: %d\n", i + 1, trapGraph->zone_nodes[i].zone_occupancy);
                printf("Zone %d Capacity: %d\n", i + 1, trapGraph->zone_nodes[i].zone_capacity);
                printf("\n");
            }
        }
    }

    printf("\nOUTPUT\n");
    for (int i = 0; i < trapGraph->n_zones; i++) {
        printf("Zone %d Occupancy: %d\n", i + 1, trapGraph->zone_nodes[i].zone_occupancy);
        printf("Zone %d Capacity: %d\n", i + 1, trapGraph->zone_nodes[i].zone_capacity);
        printf("\n");
    }
    printf("\n");

    free(ion_states);
    return transportProgram;
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

    struct InstructionNode **prevs = (struct InstructionNode **)malloc(sizeof(struct InstructionNode *) * graph->n_ions);
    for (int i = 0; i < graph->n_ions; i++) {
        prevs[i] = NULL;
    }

    int count = 0;

    rewind(fp);

    printf("Parsing QASM file...\n");

    while (fgets(line, sizeof(line), fp)) {
        printf("\nParsing line: %s", line);

        // skip empty lines
        if (strlen(line) < 2) continue;

        // skip declarations
        if (strstr(line, "qubit") || strstr(line, "bit")) continue;

        // tokenize
        char *token = strtok(line, " ,;\n");
        if (!token) continue;

        enum OperationType op_type = get_operation_type(token);
        if (op_type == 0) continue;

        // parse targets
        int targets[3]; // Assume 3 or less qubits in one gate
        int n_targets = 0;

        while ((token = strtok(NULL, " ,;\n"))) {
            if (strstr(token, "qreg")) {
                int idx = parse_qubit_index(token);
                if (idx >= 0) {
                    printf("Found target qubit index: %d\n", idx);
                    targets[n_targets++] = idx + 1;
                }
            }
        }

        struct InstructionNode node;
        if (n_targets > 0) {
            struct InstructionNode prevNodes[n_targets];
            int numPrevNodes = 0;
            for (int i = 0; i < n_targets; i++) {
                if (prevs[targets[i] - 1]) {
                    prevNodes[i] = *prevs[targets[i] - 1];
                    numPrevNodes++;
                }
            }

            node.n_prev_nodes = numPrevNodes;
            node.prev_nodes = prevNodes;

            for (int i = 0; i < n_targets; i++) {
                prevs[targets[i] - 1] = &node;
            }
        } else {
            node.n_prev_nodes = 0;
            node.prev_nodes = NULL;
        }

        node.operation_type = op_type;
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

        graph->layer_idxs[count] = count + 1;
        graph->instruction_nodes[count++] = node;
    }

    fclose(fp);

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
