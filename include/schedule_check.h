#pragma once

#include "data_structures.h"
#include "transport.h"

int check_schedule_trace
(
	struct TrapGraph *trap_graph,
	struct CircuitGraph *circuit_graph,
	struct TransportProgram *transport_program,
	struct ScheduleTrace *trace,
	const struct TransportCostModel *cost_model,
	char *message,
	int message_size
);
