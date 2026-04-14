#include "transport.h"

#include <stdlib.h>
#include <stdio.h>

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
    // transportProgram->n_ions must be initialized
    if (transportProgram->n_ions <= 0) {
        printf("ERROR: Number of ions in transport program must be initialized.\n");
        return NULL;
    }

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
        if (circuitGraph->layer_idxs[i] > max_layer) {
            max_layer = circuitGraph->layer_idxs[i];
        }
    }
    
    printf("Number of layers in circuit: %d\n", max_layer);

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
    struct CircuitGraph circuitGraphObject = {3, circuitInstructions, circuitLayers};
    struct CircuitGraph *circuitGraph = &circuitGraphObject;

    struct TransportProgram *transportProgram = (struct TransportProgram*)malloc(sizeof(struct TransportProgram));
    transportProgram->n_ions = 2;
    transportProgram = generate_transport_schedule(trapGraph, circuitGraph, transportProgram);
    if (transportProgram == NULL) {
        return NULL;
    }
    
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

    return transportProgram;
}
