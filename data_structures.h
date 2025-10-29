#pragma once

#include "defs.h"

// Ion data
struct IonParameters
{
	char *element;
	double atomic_mass;
	double charge;
	// @TODO - add quantum mechanical properties
	double omega_SP;
	double omega_SD;
	double gamma_list;
};

// Abstract trap data
enum ZoneType
{
	EMPTY_ZONE = 1 << 0,
	LOAD_ZONE = 1 << 1,
	COOL_ZONE = 1 << 2,
	GATE_ZONE = 1 << 3,
	MEASURE_ZONE = 1 << 4,
	STORAGE_ZONE = 1 << 5,
};

struct ZoneNode
{
	enum ZoneType zone_type;
	int zone_capacity;
};

enum TransportEdgeType
{
	LINEAR_SEGMENT = 1 << 0,
	T_JUNCTION_SEGMENT = 1 << 1,
	X_JUNCTION_SEGMENT = 1 << 2,
	Y_JUNCTION_SEGMENT = 1 << 3,
};

struct TrapGraph
{
	struct ZoneNode *zone_nodes;
	struct TransportEdge *transport_edges;
};

// Trap data
struct Electrode
{
	int n_vertices; // stores a list of the relative coordinates of all the vertices
	int n_edges; // stores a list of indices of vertices connected by edges, labelled by index
	double (*vertices)[3]; // stores a list of the relative coordinates of all the vertices; implicity, length n_vertices-by-3
	int (*edges)[2]; // stores a list of indices of vertices (labelled by index) connected by edges; implicity, length n_edges-by-2
	int Vlm_len; // must equal NSPH_X*NSPH_Y*NSPH_Z*(LMAX+1)*(LMAX+1)*2
	double (*Vlm)[]; // electric potential in spherical harmonics expansion; implicity, length Vlm_len in each component
};

struct Lasers
{
	int n_lasers;
	double (*laser_frequencies)[]; // implicitly, length n_lasers
	double (*site_counts)[]; // implicitly, length n_lasers
	double *(site_positions)[]; // site positions at which each laser is available; implicitly, length n_lasers-by-[site_count]-by-3
};

struct Trap
{
	int n_electrodes; // non-constant in case we want to add electrodes to a Trap
	int n_electrodes_rf; // non-constant in case we want to add electrodes to a Trap
	int n_electrodes_dc; // non-constant in case we want to add electrodes to a Trap
	struct Electrode (*electrodes)[]; // implicitly, length n_electrodes
	double (*electrode_positions)[][3]; // stores a list of the coordinates of all the electrodes; implicity, length n_electrodes-by-3
	struct Lasers *lasers;
};

// Abstract quantum circuit data
enum OperationType
{
	TRANSPORT = 1 << 0,
	COOL = 1 << 1,
	DETECT = 1 << 2,
	DESHELVE = 1 << 3,
	QUBIT = 1 << 4,
	LASER_PULSE = COOL | DETECT | DESHELVE | QUBIT
};

struct InstructionNode
{
	enum OperationType operation_type;
	double operation_duration;
	int n_target_ions;
	int *target_ions; // implicitly, length n_target_ions
	// @TODO - do we need to store both prev_nodes and next_nodes?
	int graphindex; // used to determine position in node array for writing to layer_idxs
	int n_prev_nodes;
	int n_next_nodes;
	struct InstructionNode *prev_nodes; // implicitly, length n_prev_nodes
	struct InstructionNode *next_nodes; // implicitly, length n_next_nodes
};

struct CircuitGraph
{
	int n_instruction_nodes;
	struct InstructionNode *instruction_nodes; // implicitly, length n_instruction_nodes
	int *layer_idxs; // implicitly, length n_instruction_nodes
};

// Physical quantum circuit data
struct TransportProgram
{
	int n_ions;
	int n_transports;
	double *arrival_times; // implicitly, length n_transports
	int *transport_n_targets_list; // implicitly, length n_transports
	int (*transport_targets_list); // implicitly, length n_transports-by-[transport_n_targets]
	double (*destinations)[3]; // implicitly, length n_transports-by-3
};

struct PulseProgram
{
	int n_ions;
	int n_pulses;
	double *pulse_frequencies; // implicitly, length n_pulses
	double *pulse_times; // implicitly, length n_pulses
	double *pulse_powers; // implicitly, length n_pulses
	double *pulse_durations; // implicitly, length n_pulses
	int *pulse_n_targets_list; // implicitly, length n_pulses
	int (*pulse_targets_list)[]; // implicitly, length n_pulses-by-[pulse_n_targets]
};

struct TransportWaveform
{
	int n_waveform_steps;
	double (*voltages)[]; // stores a list of the voltages applied to each electrode at each timestep; implicity, length n_waveform_steps-by-n_electrodes
};

