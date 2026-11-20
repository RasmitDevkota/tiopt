#pragma once

#include "data_structures.h"

struct ScheduleTrace;
struct TransportCostModel;

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

void compute_commuting_layers
(
	struct CircuitGraph *circuit_graph
);

void compute_circuit_layers
(
	struct CircuitGraph *circuit_graph
);

int sort_layers
(
	int n_nodes,
	const int *layer_idxs,
	int **order,
	int **layer_starts
);

int sort_circuit_layers
(
	struct CircuitGraph *circuit_graph,
	int **order,
	int **layer_starts
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

int time_schedule_trace
(
	struct TrapGraph *trap_graph,
	struct CircuitGraph *circuit_graph,
	struct TransportProgram *transport_program,
	struct ScheduleTrace *trace,
	const struct TransportCostModel *cost_model
);

void generate_trap_graph
(
	int n_zones,
	enum ZoneType *zone_types,
	int *zone_capacities,
	double (*positions)[3],
	int n_segments,
	int (*segments)[2],
	double *segment_lengths,
	struct TrapGraph *trap_graph
);

int trap_graph_routes
(
	struct TrapGraph *trap_graph
);

double trap_graph_distance
(
	struct TrapGraph *trap_graph,
	int from_zone,
	int to_zone
);

int trap_graph_route
(
	struct TrapGraph *trap_graph,
	int from_zone,
	int to_zone,
	int *path
);

int trap_graph_neighbors
(
	struct TrapGraph *trap_graph,
	int zone,
	int *neighbors
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
	const int *keep,
	struct ScheduleTrace *trace
);
