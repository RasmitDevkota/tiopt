#include "zones.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "defs.h"
#include "data_structures.h"

// Number of bits of enum ZoneType, whose names are listed below in the same order
#define N_ZONE_TYPES 8

// Ions which load, storage and measurement zones hold when the number of ions is not known
#define DEFAULT_ION_CAPACITY 100

// Largest number of zones in a layout
#define MAX_LAYOUT_ZONES 1000

static const char *zone_type_names[N_ZONE_TYPES] = { "empty", "load", "cool", "gate", "measure", "storage", "junction", "pre_junction" };

// Parse zone types joined by "+" (e.g. "load+storage"), ignoring case and spaces, into a bitmask of enum ZoneType;
// returns 1 if a type is unknown
int parse_zone_types
(
	const char *text,
	enum ZoneType *zone_type
)
{
	*zone_type = 0;

	int length = strlen(text);
	int start = 0;
	while (start <= length)
	{
		int end = start;
		while (end < length && text[end] != '+')
			end++;

		char name[32];
		int n = 0;
		for (int i = start; i < end; i++)
			if (!isspace((unsigned char) text[i]) && n < 31)
				name[n++] = tolower((unsigned char) text[i]);
		name[n] = '\0';

		int found = 0;
		for (int t = 0; t < N_ZONE_TYPES; t++)
			if (strcmp(name, zone_type_names[t]) == 0)
			{
				*zone_type |= 1 << t;
				found = 1;
			}

		if (!found)
		{
			printf("ERROR: Unknown zone type '%s' (zone types are empty, load, cool, gate, measure, storage, junction and pre_junction)\n", name);
			return 1;
		}

		start = end + 1;
	}

	return 0;
}

// Write the types of a zone joined by "+" (e.g. "load+storage") into text, which holds size characters
void format_zone_types
(
	enum ZoneType zone_type,
	char *text,
	int size
)
{
	text[0] = '\0';

	for (int t = 0; t < N_ZONE_TYPES; t++)
		if (zone_type & (1 << t))
		{
			if (text[0] != '\0')
				strncat(text, "+", size - strlen(text) - 1);

			strncat(text, zone_type_names[t], size - strlen(text) - 1);
		}
}

// Number of ions a zone holds by default: every ion in zones where they are loaded, wait or are measured, two in gate
// and cooling zones (for two-qubit gates and sympathetic cooling), and one in zones which ions only pass through
int default_zone_capacity
(
	enum ZoneType zone_type,
	int n_ions
)
{
	if (zone_type & (LOAD_ZONE | STORAGE_ZONE | MEASURE_ZONE))
		return (n_ions > 0) ? n_ions : DEFAULT_ION_CAPACITY;

	if (zone_type & (GATE_ZONE | COOL_ZONE))
		return 2;

	return 1;
}

// Parse a zone layout into the type and capacity of every zone, in order along the trap. The layout lists entries
// separated by commas, each with zone types joined by "+", an optional capacity after ":" and an optional count after
// "x" or "*", e.g. "load+storage, empty, gate:2 x5, empty, measure"; zones without a capacity get the default one for
// n_ions ions (see default_zone_capacity). Allocates zone_types and zone_capacities, and returns 1 if the layout cannot
// be parsed
int parse_zone_layout
(
	const char *layout,
	int n_ions,
	int *n_zones,
	enum ZoneType **zone_types,
	int **zone_capacities
)
{
	*n_zones = 0;
	*zone_types = NULL;
	*zone_capacities = NULL;

	int length = strlen(layout);
	int start = 0;
	while (start <= length)
	{
		int end = start;
		while (end < length && layout[end] != ',')
			end++;

		char entry[128];
		int n = MIN(end - start, 127);
		memcpy(entry, &layout[start], n);
		entry[n] = '\0';

		start = end + 1;

		// The count is a number at the end of the entry, after an "x" or "*"
		int count = 1;
		int i = n - 1;
		while (i >= 0 && isspace((unsigned char) entry[i]))
			i--;

		int last_digit = i;
		while (i >= 0 && isdigit((unsigned char) entry[i]))
			i--;

		int j = i;
		while (j >= 0 && isspace((unsigned char) entry[j]))
			j--;

		if (last_digit > i && j >= 0 && (entry[j] == 'x' || entry[j] == 'X' || entry[j] == '*'))
		{
			count = atoi(&entry[i + 1]);
			entry[j] = '\0';
		}

		int capacity = 0;
		char *colon = strchr(entry, ':');
		if (colon != NULL)
		{
			capacity = atoi(colon + 1);
			*colon = '\0';
		}

		enum ZoneType zone_type;
		if (parse_zone_types(entry, &zone_type) != 0 || count <= 0 || (colon != NULL && capacity <= 0) || *n_zones + count > MAX_LAYOUT_ZONES)
		{
			printf("ERROR: Failed to parse zone layout '%s' (entries look like 'load+storage', 'gate:2' or 'gate x5', separated by commas)\n", layout);

			free(*zone_types);
			free(*zone_capacities);
			*zone_types = NULL;
			*zone_capacities = NULL;
			*n_zones = 0;

			return 1;
		}

		if (capacity == 0)
			capacity = default_zone_capacity(zone_type, n_ions);

		*zone_types = realloc(*zone_types, (*n_zones + count) * sizeof(enum ZoneType));
		*zone_capacities = realloc(*zone_capacities, (*n_zones + count) * sizeof(int));

		for (int c = 0; c < count; c++)
		{
			(*zone_types)[*n_zones] = zone_type;
			(*zone_capacities)[*n_zones] = capacity;
			(*n_zones)++;
		}
	}

	return 0;
}

// Allocate n_zones zones for a trap, replacing any it already has
void allocate_zones
(
	struct Trap *trap,
	int n_zones
)
{
	free(trap->zones);

	trap->n_zones = n_zones;
	trap->zones = (n_zones > 0) ? calloc(n_zones, sizeof(struct ZoneNode)) : NULL;
}

// Set the type, capacity and position (µm) of zone z of a trap
void set_trap_zone
(
	struct Trap *trap,
	int z,
	enum ZoneType zone_type,
	int zone_capacity,
	double position[3]
)
{
	struct ZoneNode *zone = &(*trap->zones)[z];

	zone->zone_type = zone_type;
	zone->zone_capacity = zone_capacity;
	zone->zone_occupancy = 0;

	for (int k = 0; k < 3; k++)
		zone->position[k] = position[k];
}

// Set the zones of a trap to a zone layout (see parse_zone_layout), spaced evenly by zone_spacing (µm) along the trap
// axis, centered on the trap and at the height of the ions; returns 1 if the layout cannot be parsed
int set_layout_zones
(
	struct Trap *trap,
	const char *layout,
	int n_ions,
	double zone_spacing
)
{
	int n_zones;
	enum ZoneType *zone_types;
	int *zone_capacities;
	if (parse_zone_layout(layout, n_ions, &n_zones, &zone_types, &zone_capacities) != 0)
		return 1;

	allocate_zones(trap, n_zones);

	for (int z = 0; z < n_zones; z++)
	{
		double position[3] = { (z - 0.5 * (n_zones - 1)) * zone_spacing, 0.0, ION_HEIGHT };

		set_trap_zone(trap, z, zone_types[z], zone_capacities[z], position);
	}

	free(zone_types);
	free(zone_capacities);

	return 0;
}

// Find the first zone of a trap from zone start onwards which has any of the given types, or -1 if there is none
int find_trap_zone
(
	struct Trap *trap,
	enum ZoneType zone_type,
	int start
)
{
	for (int z = MAX(start, 0); z < trap->n_zones; z++)
		if ((*trap->zones)[z].zone_type & zone_type)
			return z;

	return -1;
}

// Build the graph of a trap's zones for scheduling transports, with consecutive zones connected
void generate_trap_graph_from_zones
(
	struct Trap *trap,
	struct TrapGraph *trap_graph
)
{
	int n_zones = trap->n_zones;

	trap_graph->n_zones = n_zones;
	trap_graph->n_transport_edges = MAX(n_zones - 1, 0);
	trap_graph->zone_nodes = calloc(MAX(n_zones, 1), sizeof(struct ZoneNode));
	trap_graph->transport_edges = calloc(MAX(n_zones - 1, 1), sizeof(struct TransportEdge));

	for (int z = 0; z < n_zones; z++)
	{
		trap_graph->zone_nodes[z] = (*trap->zones)[z];
		trap_graph->zone_nodes[z].zone_occupancy = 0;
	}

	for (int t = 0; t < n_zones - 1; t++)
	{
		struct TransportEdge *edge = &trap_graph->transport_edges[t];

		edge->edge_type = LINEAR_SEGMENT;
		edge->n_zones = 2;
		edge->zone_nodes = malloc(2 * sizeof(struct ZoneNode));
		edge->zone_nodes[0] = trap_graph->zone_nodes[t];
		edge->zone_nodes[1] = trap_graph->zone_nodes[t + 1];
	}
}
