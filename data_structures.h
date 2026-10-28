#pragma once

#include "defs.h"

// Ion data
struct IonParameters
{
	char *element;
	double atomic_mass; // mass in atomic mass units
	double charge; // charge in units of the elementary charge
	double omega_SP; // angular frequency of the S-P (cooling and detection) transition (rad/s)
	double omega_SD; // angular frequency of the S-D (qubit) transition (rad/s)
	double gamma_SP; // natural linewidth of the S-P transition (1/s)
	double gamma_SD; // natural linewidth of the S-D transition (1/s)
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
	JUNCTION_ZONE = 1 << 6,
	PRE_JUNCTION_ZONE = 1 << 7,
};

struct ZoneNode
{
	enum ZoneType zone_type;
	int zone_capacity;
	int zone_occupancy;
	double position[3]; // position of the center of the zone in the trap frame (µm)
};

enum TransportEdgeType
{
	LINEAR_SEGMENT = 1 << 0,
	THREE_WAY_JUNCTION_SEGMENT = 1 << 1,
	FOUR_WAY_JUNCTION_SEGMENT = 1 << 2,
};

struct TransportEdge
{
	enum TransportEdgeType edge_type;
	int n_zones;
	struct ZoneNode *zone_nodes; // implicitly, length n_zones
};

struct TrapGraph
{
	int n_zones;
	int n_transport_edges;
	struct ZoneNode *zone_nodes; // implicitly, length n_zones
	struct TransportEdge *transport_edges; // implicitly, length n_transport_edges
};

// Trap data
struct Electrode
{
	int n_vertices; // stores a list of the relative coordinates of all the vertices
	int n_edges; // stores a list of indices of vertices connected by edges, labelled by index
	double (*vertices)[3]; // stores a list of the relative coordinates of all the vertices; implicitly, length n_vertices-by-3
	int (*edges)[2]; // stores a list of indices of vertices (labelled by index) connected by edges; implicitly, length n_edges-by-2
	int Vlm_len; // must equal NSPH_X*NSPH_Y*NSPH_Z*(LMAX+1)*(LMAX+1)*2
	double (*Vlm)[]; // electric potential in spherical harmonics expansion; implicitly, length Vlm_len in each component
};

struct Lasers
{
	int n_lasers;
	double (*laser_frequencies)[]; // implicitly, length n_lasers
	double (*site_counts)[]; // implicitly, length n_lasers
	double *(site_positions)[]; // site positions at which each laser is available; implicitly, length n_lasers-by-[site_count]-by-3
};

// Integrated photonics data
struct Waveguide
{
	double wavelength; // free-space wavelength of the guided light (µm)
	double width; // width of the core (µm)
	double thickness; // thickness of the core (µm)
	double n_core; // refractive index of the core
	double n_cladding; // refractive index of the cladding surrounding the core
	double n_eff; // effective index of the fundamental quasi-TE mode, computed by solve_waveguide_mode
};

struct GratingCoupler
{
	struct Waveguide waveguide; // waveguide feeding the grating
	double position[3]; // point at which the emitted beam leaves the trap surface, in the trap frame (µm)
	double azimuth; // in-plane direction of propagation of the guided mode, measured from the x-axis (rad)
	double period; // grating period (µm)
	double fill_factor; // fraction of each period which is left unetched
	double etch_depth; // depth of the grating teeth (µm)
	double length; // length of the grating along the direction of propagation (µm)
	double width; // width of the grating transverse to the direction of propagation (µm)
	double focal_distance; // distance along the beam from the surface to the focus, or 0 for an unfocused grating (µm)
	double power; // optical power delivered to the grating (W)
	double efficiency; // fraction of the delivered power which is emitted towards the ion
};

// Integrated single-photon detector (e.g. a superconducting nanowire or avalanche photodiode) in the trap surface
struct Detector
{
	double position[3]; // center of the active area in the trap frame (µm)
	double width; // extent of the active area along x (µm)
	double length; // extent of the active area along y (µm)
	double wavelength; // free-space wavelength of the detected light (µm)
	double quantum_efficiency; // probability that a photon reaching the active area is counted
	double dark_count_rate; // rate of counts with no light (1/s)
};

struct Trap
{
	int n_electrodes; // non-constant in case we want to add electrodes to a Trap
	int n_electrodes_rf; // non-constant in case we want to add electrodes to a Trap; electrodes [0, n_electrodes_rf) are RF electrodes
	int n_electrodes_dc; // non-constant in case we want to add electrodes to a Trap; electrodes [n_electrodes_rf, n_electrodes) are DC electrodes
	struct Electrode (*electrodes)[]; // implicitly, length n_electrodes
	double (*electrode_positions)[][3]; // stores a list of the coordinates of all the electrodes; implicitly, length n_electrodes-by-3
	struct Lasers *lasers;
	double V_RF; // amplitude of the RF voltage applied to every RF electrode (V)
	double Omega_RF; // angular frequency of the RF drive (rad/s)
	int n_grating_couplers;
	struct GratingCoupler (*grating_couplers)[]; // implicitly, length n_grating_couplers
	int n_detectors;
	struct Detector (*detectors)[]; // implicitly, length n_detectors
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
	int *target_ions; // implicitly, length n_target_ions (starts at 1)
	// @TODO - do we need to store both prev_nodes and next_nodes?
	int graphindex; // used to determine position in node array for writing to layer_idxs (starts at 1)
	int n_prev_nodes;
	struct InstructionNode *prev_nodes; // implicitly, length n_prev_nodes
};

struct CircuitGraph
{
	int n_ions;
	int n_instruction_nodes;
	struct InstructionNode *instruction_nodes; // implicitly, length n_instruction_nodes
	int *layer_idxs; // implicitly, length n_instruction_nodes (starts at 1)
};

// Physical quantum circuit data
struct TransportProgram
{
	int n_ions;
	int n_transports;
	double *arrival_times; // implicitly, length n_transports
	int *transport_n_targets_list; // implicitly, length n_transports
	int **transport_targets_list; // implicitly, length n_transports-by-[transport_n_targets_list]
	double (*destinations)[3]; // implicitly, length n_transports-by-3
	double (*initial_positions)[3]; // position of each ion before the first transport; implicitly, length n_ions-by-3
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
	int **pulse_targets_list; // implicitly, length n_pulses-by-[pulse_n_targets] (starts at 1)
	double *pulse_phases; // implicitly, length n_pulses
};

struct TransportWaveform
{
	int n_waveform_steps;
	int n_electrodes;
	double *voltages; // stores a list of the voltages applied to each electrode at each timestep; implicitly, length n_waveform_steps-by-n_electrodes
	double *times; // time at which each step of the waveform is applied (s); implicitly, length n_waveform_steps
	int n_wells;
	double (*well_positions)[3]; // target position of each potential well at each timestep; implicitly, length n_waveform_steps-by-n_wells-by-3
};

// Quantum dynamics data
struct MotionalMode
{
	double omega; // angular frequency of the mode (rad/s)
	int n_max; // highest Fock state kept in the truncated motional basis
	double n_bar; // mean phonon number of the initial thermal state
	double alpha[2]; // real and imaginary parts of the coherent displacement of the initial state
	double heating_rate; // motional heating rate (quanta/s)
};
