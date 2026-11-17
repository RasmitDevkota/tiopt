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
                        // (or within ScheduleOptions.lookahead_layers layers) and can happen there
    SCHEDULE_LAYER_ASSIGN, // as SCHEDULE_LOOKAHEAD, but the operations of each round are assigned to zones together,
                           // to minimize the longest transport of the round
};

#ifndef DEFAULT_SCHEDULE_STRATEGY
#define DEFAULT_SCHEDULE_STRATEGY SCHEDULE_ROUNDS
#endif

// Number of layers ahead which SCHEDULE_LOOKAHEAD and SCHEDULE_LAYER_ASSIGN look for the next operation of an ion when
// deciding whether it stays in its zone after a layer
#ifndef DEFAULT_LOOKAHEAD_LAYERS
#define DEFAULT_LOOKAHEAD_LAYERS 1
#endif

// How generate_transport_schedule_options schedules a circuit
struct ScheduleOptions
{
    enum ScheduleStrategy strategy;
    int lookahead_layers; // with SCHEDULE_LOOKAHEAD and SCHEDULE_LAYER_ASSIGN, an ion stays in its zone after a layer if
                          // its next operation is at most this many layers later and can happen there (at least 1)
    int commute; // move operations which are diagonal in the computational basis (and so commute) into fewer layers
};

// A move of an ion from one zone of a trap graph to another
struct IonMove
{
    int ion; // starts at 1
    int from_zone;
    int to_zone;
    int transport; // index of the transport which moves the ion, or -1 for a move back into storage between operations
    double depart; // time at which the ion leaves from_zone (s)
    double arrive; // time at which the ion reaches to_zone (s)
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

struct TransportProgram* generate_transport_schedule_options
(
    struct TrapGraph *trapGraph,
    struct CircuitGraph *circuitGraph,
    struct TransportProgram *transportProgram,
    const struct ScheduleOptions *options,
    struct ScheduleTrace *trace
);

void free_schedule_trace(struct ScheduleTrace *trace);

// Strategy which generate_transport_schedule uses
int default_schedule_strategy(void);

struct TransportProgram* simple_transport_schedule ();

struct CircuitGraph* parse_qasm(const char *filename);

int is_diagonal_gate(const char *op);

struct ZoneNode* select_available_zone
(
    struct TrapGraph *trapGraph,
    enum ZoneType type
);

struct TransportProgram* simple_transport_schedule_qasm_input();

void print_transport_program(struct TransportProgram *transportProgram);
