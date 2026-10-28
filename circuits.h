#pragma once

#include "data_structures.h"

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
