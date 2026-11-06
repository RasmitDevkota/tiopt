#pragma once

#include "data_structures.h"

// How generate_transport_schedule_strategy assigns operations to zones and moves ions between layers
enum ScheduleStrategy
{
    SCHEDULE_BASELINE, // nearest zone for each operation in turn, and every ion back into storage after each layer
                       // (each layer is one round, even if its operations do not all fit in the zones at once)
    SCHEDULE_FAST, // the same schedule as SCHEDULE_BASELINE, computed without rescanning the circuit for each layer
    SCHEDULE_ROUNDS, // as SCHEDULE_BASELINE, but splitting layers into rounds which fit in the zones
    SCHEDULE_LOOKAHEAD, // as SCHEDULE_ROUNDS, but ions stay in their zone if their next operation is in the next layer
                        // and can happen there
    SCHEDULE_LAYER_ASSIGN, // as SCHEDULE_LOOKAHEAD, but the operations of each round are assigned to zones together,
                           // to minimize the longest transport of the round
};

#ifndef DEFAULT_SCHEDULE_STRATEGY
#define DEFAULT_SCHEDULE_STRATEGY SCHEDULE_ROUNDS
#endif

// A move of an ion from one zone of a trap graph to another
struct IonMove
{
    int ion; // starts at 1
    int from_zone;
    int to_zone;
    int transport; // index of the transport which moves the ion, or -1 for a move back into storage between operations
};

// Every move of the ions made by a schedule, in order, including the moves back into storage which the transport
// program does not record, and the round of each transport: the operations of a round happen at once, after the ions
// of the round have all arrived, and a layer of the circuit takes several rounds if its operations do not all fit in
// the zones at once
struct ScheduleTrace
{
    int n_moves;
    int max_moves;
    struct IonMove *moves; // implicitly, length n_moves
    int n_rounds;
    int *transport_rounds; // implicitly, length n_transports (starts at 1)
};

struct TransportProgram* generate_transport_schedule
(
    struct TrapGraph *trapGraph,
    struct CircuitGraph *circuitGraph,
    struct TransportProgram *transportProgram
);

struct TransportProgram* generate_transport_schedule_strategy
(
    struct TrapGraph *trapGraph,
    struct CircuitGraph *circuitGraph,
    struct TransportProgram *transportProgram,
    enum ScheduleStrategy strategy,
    struct ScheduleTrace *trace
);

void free_schedule_trace(struct ScheduleTrace *trace);

struct TransportProgram* simple_transport_schedule ();

struct CircuitGraph* parse_qasm(const char *filename);

struct ZoneNode* select_available_zone
(
    struct TrapGraph *trapGraph,
    enum ZoneType type
);

struct TransportProgram* simple_transport_schedule_qasm_input();

void print_transport_program(struct TransportProgram *transportProgram);
