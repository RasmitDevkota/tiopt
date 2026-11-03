#pragma once

#include "data_structures.h"

int parse_zone_types
(
	const char *text,
	enum ZoneType *zone_type
);

void format_zone_types
(
	enum ZoneType zone_type,
	char *text,
	int size
);

int default_zone_capacity
(
	enum ZoneType zone_type,
	int n_ions
);

int parse_zone_layout
(
	const char *layout,
	int n_ions,
	int *n_zones,
	enum ZoneType **zone_types,
	int **zone_capacities
);

void allocate_zones
(
	struct Trap *trap,
	int n_zones
);

void set_trap_zone
(
	struct Trap *trap,
	int z,
	enum ZoneType zone_type,
	int zone_capacity,
	double position[3]
);

int set_layout_zones
(
	struct Trap *trap,
	const char *layout,
	int n_ions,
	double zone_spacing
);

int find_trap_zone
(
	struct Trap *trap,
	enum ZoneType zone_type,
	int start
);

void generate_trap_graph_from_zones
(
	struct Trap *trap,
	struct TrapGraph *trap_graph
);
