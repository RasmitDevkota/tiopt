#include "data_structures.h"

struct TransportProgram* generate_transport_schedule
(
    struct TrapGraph *trapGraph,
    struct CircuitGraph *circuitGraph,
    struct TransportProgram *transportProgram
);

struct TransportProgram* simple_transport_schedule ();
