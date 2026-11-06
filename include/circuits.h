#pragma once

#include "data_structures.h"

struct ScheduleTrace;

int add_qasm_targets
(
	char *token,
	int n_ions,
	int *targets,
	int n_targets
);

void set_prev_nodes
(
	struct CircuitGraph *circuit_graph,
	int *last_node_idxs,
	int *targets,
	int n_targets,
	struct InstructionNode *node
);

void compute_circuit_layers
(
	struct CircuitGraph *circuit_graph
);

void schedule_arrival_times
(
	struct CircuitGraph *circuit_graph,
	struct TransportProgram *transport_program
);

void schedule_arrival_times_in_rounds
(
	struct CircuitGraph *circuit_graph,
	struct TransportProgram *transport_program,
	const int *rounds
);

void generate_linear_trap_graph
(
	int n_zones,
	enum ZoneType *zone_types,
	int *zone_capacities,
	double x_min,
	double x_max,
	double height,
	struct TrapGraph *trap_graph
);

void free_trap_graph
(
	struct TrapGraph *trap_graph
);

void free_circuit_graph
(
	struct CircuitGraph *circuit_graph
);

void free_transport_program
(
	struct TransportProgram *transport_program
);

struct ZoneNode* select_operation_zone
(
	struct TrapGraph *trap_graph,
	enum ZoneType zone_type,
	struct ZoneNode **ion_states,
	int *targets,
	int n_targets
);

void release_operation_zones
(
	struct TrapGraph *trap_graph,
	struct ZoneNode **ion_states,
	int n_ions
);

void record_ion_move
(
	struct ScheduleTrace *trace,
	struct TrapGraph *trap_graph,
	int ion,
	struct ZoneNode *from,
	struct ZoneNode *to,
	int transport
);

void release_ions
(
	struct TrapGraph *trap_graph,
	struct ZoneNode **ion_states,
	int n_ions,
	struct ScheduleTrace *trace
);
