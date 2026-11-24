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
    SCHEDULE_STAY, // as SCHEDULE_ROUNDS, but ions stay where they are after each operation instead of going back into
                   // storage (as in tiopt.qccd), so that an operation usually brings one ion to where its partner is;
                   // when no zone of its type has room, the ions which the round does not need are moved into storage,
                   // those whose next operation comes last first
};

#ifndef DEFAULT_SCHEDULE_STRATEGY
#define DEFAULT_SCHEDULE_STRATEGY SCHEDULE_ROUNDS
#endif

// Number of layers ahead which SCHEDULE_LOOKAHEAD and SCHEDULE_LAYER_ASSIGN look for the next operation of an ion when
// deciding whether it stays in its zone after a layer
#ifndef DEFAULT_LOOKAHEAD_LAYERS
#define DEFAULT_LOOKAHEAD_LAYERS 1
#endif

// Durations of two-qubit gates in a chain of N ions whose targets are d apart, in the models of P. Murali et al. (ISCA
// 2020), as in tiopt.gates.gate_time_model, or a fixed duration
enum TransportGateModel
{
    GATE_MODEL_FIXED, // TransportCostModel.two_qubit
    GATE_MODEL_AM1, // 100 d - 22 µs
    GATE_MODEL_AM2, // 38 d + 10 µs
    GATE_MODEL_PM, // 5 d + 160 µs
    GATE_MODEL_FM, // max(13.33 N - 54, 100) µs
};

// How an ion reaches the end of its chain in a zone where gates can be done, to leave it or pass through it
enum TransportSwapMethod
{
    SWAP_PHYSICAL, // swapping places with each ion on the way (a rotation each, with a split and a merge on each side
                   // of the pair where the chain goes on: one at the end of the chain and two in its middle)
    SWAP_GATES, // a SWAP gate (three two-qubit gates) with the ion at the end (the "gate" swaps of tiopt.qccd)
    SWAP_CHEAPER, // whichever costs the schedule less fidelity, by the estimate of swap_gate_is_cheaper in circuits.c
};

// Durations of the steps which move ions along the segments of a trap graph and of the operations (see
// time_schedule_trace), and the heating and errors which give the fidelity of a schedule: the TimingModel and NoiseModel
// of tiopt.qccd (in s rather than µs)
struct TransportCostModel
{
    double speed; // speed at which an ion moves along a segment (µm/s)
    double min_move; // shortest time of a move along a segment (s)
    double split; // splitting an ion off the end of a chain (s)
    double merge; // merging an ion into the end of a chain (s)
    double junction_y; // crossing a junction of three segments (s)
    double junction_x; // crossing a junction of four or more segments (s)
    double rotation; // rotating two neighboring ions to swap them, which also takes a split and a merge on each side
                     // of the pair where the chain goes on (s)
    int swap_gates; // enum TransportSwapMethod: how an ion reaches the end of its chain in a zone where gates can be
                    // done (elsewhere, it always swaps places with each ion on the way)
    int gate_model; // enum TransportGateModel
    double two_qubit; // duration of a two-qubit gate with GATE_MODEL_FIXED (s)
    double single_qubit; // duration of a single-qubit gate which is not virtual (s)
    double measurement; // (s)
    double cooling; // (s)
    double split_heating; // motional quanta added by a split (to the ion and to the chain it leaves)
    double merge_heating; // quanta added to a chain by a merge
    double junction_heating; // quanta added to an ion by crossing a junction
    double move_heating; // quanta added to an ion by every 100 µm it moves
    double background_heating; // quanta added to every chain and moving ion over time (1/s)
    double gate_background_error; // error of a two-qubit gate over its duration (1/s)
    double motional_error; // A0 of the thermal error A (2 n_bar + 1) of a two-qubit gate, with A = A0 N / ln N
    double single_qubit_error;
    double measurement_error;
    double cooled_n_bar; // mean phonon number of a chain after cooling
};

// Cost model which time_schedule_trace uses when it is given none: the defaults of tiopt.qccd.TimingModel and
// tiopt.qccd.NoiseModel, from Murali et al.
struct TransportCostModel default_transport_cost_model(void);

// Cost model in which ions move at TRANSPORT_SPEED_LIMIT, splits, merges, swaps and junction crossings take no time,
// operations take GATE_DURATION, DETECTION_DURATION and COOLING_DURATION, and nothing heats or fails
struct TransportCostModel distance_transport_cost_model(void);

// Duration of a two-qubit gate between ions separation apart in a chain of n_ions (s)
double two_qubit_gate_time(const struct TransportCostModel *cost_model, int n_ions, int separation);

// How generate_transport_schedule_options schedules a circuit
struct ScheduleOptions
{
    enum ScheduleStrategy strategy;
    int lookahead_layers; // with SCHEDULE_LOOKAHEAD and SCHEDULE_LAYER_ASSIGN, an ion stays in its zone after a layer if
                          // its next operation is at most this many layers later and can happen there (at least 1)
    int commute; // move operations which are diagonal in the computational basis (and so commute) into fewer layers
    const struct TransportCostModel *cost_model; // durations of the steps of the moves, or NULL for the default
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

// Steps which carry out the moves of a schedule (see time_schedule_trace)
enum TransportStepType
{
    STEP_SPLIT, // the ion splits off the end of the chain in a zone which faces to_zone
    STEP_MOVE, // the ion moves along the segment from zone to to_zone
    STEP_JUNCTION, // the ion crosses the junction zone
    STEP_MERGE, // the ion merges into the end of the chain in a zone which faces to_zone (where it came from)
    STEP_ION_SWAP, // the ion and its neighbor other_ion in the chain of a zone swap places
    STEP_OPERATION, // the operation index runs on the chain in a zone
    STEP_GATE_SWAP, // a SWAP gate exchanges the states of the ion and other_ion in the chain of a zone, so that they
                    // trade places
};

// Step of a schedule, which happens in a zone (or, for a move, along the segment from zone to to_zone)
struct TransportStep
{
    int type; // enum TransportStepType
    int ion; // starts at 1 (0 for an operation)
    int other_ion; // the neighbor of a swap (starts at 1), or 0
    int zone;
    int to_zone; // the other end of a move, the neighbor of a zone which a split or merge faces, or zone
    int index; // the IonMove which the step is part of, or for an operation, its instruction
    double start; // (s)
    double end; // (s)
};

// Every move of the ions made by a schedule, in order, including the moves back into storage which the transport
// program does not record, and the round of each transport: the operations of a round happen after the ions of the
// round have all arrived (at once, except that operations on several ions take the chain of their zone one after
// another), and a layer of the circuit takes several rounds if its operations do not all fit in the zones at once. The
// steps carry out the moves along the segments of the trap graph: the ions of a zone form a chain, which starts at the
// end facing its neighbor with the lowest index, and starts out in the order of the ions
struct ScheduleTrace
{
    int n_moves;
    int max_moves;
    struct IonMove *moves; // implicitly, length n_moves
    int n_rounds;
    int *transport_rounds; // implicitly, length n_transports (starts at 1)
    int n_ions;
    int *initial_zones; // zone of each ion before the first move; implicitly, length n_ions
    int n_steps;
    int max_steps;
    struct TransportStep *steps; // in the order they were planned, which is the order of time in each zone; implicitly,
                                 // length n_steps
    double fidelity; // product of the fidelities of every operation (and SWAP gate) of the schedule
    double max_n_bar; // largest mean phonon number of a chain at a two-qubit gate
    double *operation_n_bars; // mean phonon number of the chain of each operation when it starts; implicitly, length
                              // n_transports
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

int gate_ms_count(const char *op);

int is_virtual_gate(const char *op);

struct ZoneNode* select_available_zone
(
    struct TrapGraph *trapGraph,
    enum ZoneType type
);

struct TransportProgram* simple_transport_schedule_qasm_input();

void print_transport_program(struct TransportProgram *transportProgram);
