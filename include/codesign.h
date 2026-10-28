#pragma once

#include "data_structures.h"

// Stages of the co-design flow: after a basic design of everything (stage 0), the trap, photonics, transport program
// and waveforms are each optimized a little (stages 1-4) and the quantum computing metrics of the whole design are
// computed (stage 5); later passes repeat stages 1-5 with the metrics of the whole design as the objective of every
// stage, so that each part is optimized for running circuits rather than for its own figures of merit
enum CodesignStage
{
	BASIC_DESIGN_STAGE,
	TRAP_STAGE,
	PHOTONICS_STAGE,
	TRANSPORT_STAGE,
	WAVEFORM_STAGE,
	METRICS_STAGE,
	N_CODESIGN_STAGES,
};

// Design parameters, grouped by the stage which optimizes them
struct CodesignParameters
{
	// Trap (stage 1)
	double segment_width; // width of each segmented DC electrode along the trap axis (µm)
	double V_RF; // amplitude of the RF voltage (V)
	double Omega_RF; // angular frequency of the RF drive (rad/s)

	// Photonics (stage 2)
	double grating_length; // length of the qubit grating coupler along its waveguide (µm)
	double grating_width; // width of the qubit grating coupler (µm)
	double rabi_fraction; // fraction of the largest Rabi frequency the qubit grating delivers which is used for gates
	double detector_size; // side of the square detector beneath the measurement zone (µm)
	double detection_time; // duration of state detection (s)

	// Transport (stage 3)
	double zone_spacing; // distance between neighboring zones along the trap axis (µm)
	double shuttle_duration; // duration of a shuttle between neighboring zones (s)

	// Waveforms (stage 4)
	double omega_axial; // axial secular frequency of every well (rad/s)
	double log_roughness_weight; // base 10 logarithm of the roughness weight of the waveform solutions
};

// Hardware and noise which the co-design takes as given
struct CodesignEnvironment
{
	int n_segments; // number of segmented DC electrodes on each side of the trap
	double max_voltage; // largest DC voltage the electronics can apply (V)
	double qubit_power; // power delivered to the qubit grating coupler (W)
	double detection_power; // power delivered to the detection grating coupler (W)
	double field_noise; // electric field noise spectral density at the ion at 1 MHz (V^2/m^2/Hz)
	double field_noise_exponent; // the field noise scales as omega^-field_noise_exponent
	double voltage_noise; // voltage noise spectral density of the DC electronics at the secular frequency (V^2/Hz)
	double dephasing_rate; // dephasing rate of the qubit (1/s)
	double n_bar_cooled; // mean phonon number of the axial mode after cooling
	double position_drift; // drift of the ion position relative to the beams between calibrations (µm)
	double detector_efficiency; // quantum efficiency of the detector
	double dark_count_density; // dark count rate per unit area of the detector (1/s/µm^2)
	double target_radial_frequency; // radial secular frequency aimed for by the first pass (rad/s)
	double max_mathieu_q; // largest stability parameter of the radial motion
	double min_trap_depth; // smallest depth of the pseudopotential (eV)
};

// Figures of merit of a design, from the trap up to the circuit it runs
struct QCMetrics
{
	// Trap
	double radial_frequency; // lower radial secular frequency at the gate zone (rad/s)
	double mathieu_q; // stability parameter of the radial motion
	double trap_depth; // depth of the pseudopotential (eV)
	double heating_rate; // heating rate of the axial mode (quanta/s)
	double static_voltage; // largest DC voltage which holds wells in every zone the circuit occupies at once (V)

	// Photonics
	double rabi_frequency; // carrier Rabi frequency used for gates (rad/s)
	double lamb_dicke_parameter; // Lamb-Dicke parameter of the qubit beam and the axial mode
	double crosstalk; // intensity of the qubit beam at the neighboring zones relative to the gate zone
	double rabi_sensitivity; // relative change of the Rabi frequency when the ion drifts from the beam center
	double detection_fidelity;

	// Transport and waveforms
	double shuttle_excitation; // axial quanta excited by a shuttle between neighboring zones
	double position_error; // distance of the ion from the target at the end of a shuttle (µm)
	double frequency_error; // largest relative error of the axial frequency during a shuttle
	double peak_voltage; // largest DC voltage during a shuttle (V)

	// Gates and circuit
	double single_qubit_time; // duration of a PI / 2 rotation (s)
	double two_qubit_time; // duration of a Molmer-Sorensen gate (s)
	double gate_n_bar; // mean phonon number of the axial mode at the two-qubit gates
	double single_qubit_fidelity;
	double two_qubit_fidelity;
	double circuit_duration; // (s)
	double circuit_fidelity; // estimated probability that the circuit runs without error
	double objective; // objective value of the design in the QC-aware passes
};

// State of the design after a stage of a pass of the co-design flow
struct CodesignRecord
{
	int pass;
	enum CodesignStage stage;
	int n_evaluations; // evaluations of the design made by the stage
	double objective; // objective value the stage reached
	struct CodesignParameters parameters;
	struct QCMetrics metrics;
};

struct Codesign
{
	struct IonParameters ion;
	struct CodesignEnvironment environment;
	struct CodesignParameters parameters;
	struct QCMetrics metrics;

	// Circuit, summarized by the layers of its transport program
	int n_ions;
	int n_layers;
	double *layer_distances; // largest distance moved by any ion before each layer, in zone spacings; implicitly, length n_layers
	int *layer_single_qubit_gates; // implicitly, length n_layers
	int *layer_two_qubit_gates; // implicitly, length n_layers
	int *layer_measurements; // implicitly, length n_layers
	int n_zones;
	int gate_zone;
	int measure_zone;
	int n_static_wells; // largest number of zones occupied at once
	int static_well_zones[8]; // zones occupied at once by the most wells; implicitly, length n_static_wells

	// Trap, and results which depend on only some of the parameters, kept between evaluations of the design
	struct Trap trap;
	double solved_segment_width; // segment width of the trap whose electrodynamics are solved, or 0 if none is
	double gate_position[3]; // RF null at the gate zone (µm)
	double measure_position[3]; // RF null at the measurement zone (µm)
	double primitive_key[7]; // parameters the shuttle primitive was simulated with
	int primitive_status; // 0 if the shuttle primitive succeeded, 1 if it failed, and -1 if it has not been simulated
	double shuttle_excitation;
	double position_error;
	double frequency_error;
	double peak_voltage;

	// History of the flow
	int n_records;
	int max_records;
	struct CodesignRecord *records; // implicitly, length max_records
};

void default_codesign_parameters
(
	struct CodesignParameters *parameters
);

void default_codesign_environment
(
	struct CodesignEnvironment *environment
);

struct Codesign* create_codesign
(
	char *element,
	char *qasm_filename,
	struct CodesignEnvironment *environment,
	struct CodesignParameters *parameters
);

void free_codesign
(
	struct Codesign *codesign
);

int evaluate_codesign
(
	struct Codesign *codesign,
	int simulate_gates,
	struct QCMetrics *metrics
);

int optimize_codesign_stage
(
	struct Codesign *codesign,
	enum CodesignStage stage,
	int qc_aware,
	int max_evaluations,
	double *objective,
	int *n_evaluations
);

int run_codesign
(
	struct Codesign *codesign,
	int n_passes,
	int max_evaluations
);

const char* codesign_stage_name
(
	enum CodesignStage stage
);
