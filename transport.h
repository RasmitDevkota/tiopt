#pragma once

#include "data_structures.h"

struct TransportProgram* generate_transport_schedule
(
    struct TrapGraph *trapGraph,
    struct CircuitGraph *circuitGraph,
    struct TransportProgram *transportProgram
);

struct TransportProgram* simple_transport_schedule ();

struct CircuitGraph* parse_qasm(const char *filename);

struct ZoneNode* select_available_zone
(
    struct TrapGraph *trapGraph,
    enum ZoneType type
);

struct TransportProgram* simple_transport_schedule_qasm_input();

void print_transport_program(struct TransportProgram *transportProgram);
