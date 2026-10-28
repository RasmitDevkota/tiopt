#include "bindings.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"
#include "electrodynamics.h"
#include "photonics.h"
#include "acceleration.h"
#include "codesign.h"

// Size of a data structure, so that bindings to other languages can check their definitions; returns -1 for unknown names
int tiopt_struct_size
(
	char *name
)
{
	if (strcmp(name, "IonParameters") == 0)
		return sizeof(struct IonParameters);
	if (strcmp(name, "ZoneNode") == 0)
		return sizeof(struct ZoneNode);
	if (strcmp(name, "TransportEdge") == 0)
		return sizeof(struct TransportEdge);
	if (strcmp(name, "TrapGraph") == 0)
		return sizeof(struct TrapGraph);
	if (strcmp(name, "Electrode") == 0)
		return sizeof(struct Electrode);
	if (strcmp(name, "Waveguide") == 0)
		return sizeof(struct Waveguide);
	if (strcmp(name, "GratingCoupler") == 0)
		return sizeof(struct GratingCoupler);
	if (strcmp(name, "Trap") == 0)
		return sizeof(struct Trap);
	if (strcmp(name, "InstructionNode") == 0)
		return sizeof(struct InstructionNode);
	if (strcmp(name, "CircuitGraph") == 0)
		return sizeof(struct CircuitGraph);
	if (strcmp(name, "TransportProgram") == 0)
		return sizeof(struct TransportProgram);
	if (strcmp(name, "PulseProgram") == 0)
		return sizeof(struct PulseProgram);
	if (strcmp(name, "TransportWaveform") == 0)
		return sizeof(struct TransportWaveform);
	if (strcmp(name, "MotionalMode") == 0)
		return sizeof(struct MotionalMode);
	if (strcmp(name, "Detector") == 0)
		return sizeof(struct Detector);
	if (strcmp(name, "CodesignParameters") == 0)
		return sizeof(struct CodesignParameters);
	if (strcmp(name, "CodesignEnvironment") == 0)
		return sizeof(struct CodesignEnvironment);
	if (strcmp(name, "QCMetrics") == 0)
		return sizeof(struct QCMetrics);
	if (strcmp(name, "CodesignRecord") == 0)
		return sizeof(struct CodesignRecord);

	return -1;
}

// Value of a compile-time constant from defs.h, so that bindings use the same values as the library; returns NAN for
// unknown names
double tiopt_constant
(
	char *name
)
{
	if (strcmp(name, "ION_HEIGHT") == 0)
		return ION_HEIGHT;
	if (strcmp(name, "LENGTH_SCALE") == 0)
		return LENGTH_SCALE;
	if (strcmp(name, "SPH_R") == 0)
		return SPH_R;
	if (strcmp(name, "SPH_SPACING") == 0)
		return SPH_SPACING;
	if (strcmp(name, "NSPH_X") == 0)
		return NSPH_X;
	if (strcmp(name, "NSPH_Y") == 0)
		return NSPH_Y;
	if (strcmp(name, "NSPH_Z") == 0)
		return NSPH_Z;
	if (strcmp(name, "SPH_CENTER_X_MIN") == 0)
		return SPH_CENTER_X(0);
	if (strcmp(name, "SPH_CENTER_Y_MIN") == 0)
		return SPH_CENTER_Y(0);
	if (strcmp(name, "SPH_CENTER_Z_MIN") == 0)
		return SPH_CENTER_Z(0);
	if (strcmp(name, "LMAX") == 0)
		return LMAX;
	if (strcmp(name, "RELAXATION_NX") == 0)
		return RELAXATION_NX;
	if (strcmp(name, "RELAXATION_NY") == 0)
		return RELAXATION_NY;
	if (strcmp(name, "RELAXATION_NZ") == 0)
		return RELAXATION_NZ;
	if (strcmp(name, "AXIAL_FREQUENCY") == 0)
		return AXIAL_FREQUENCY;
	if (strcmp(name, "TRANSPORT_SPEED_LIMIT") == 0)
		return TRANSPORT_SPEED_LIMIT;
	if (strcmp(name, "WAVEFORM_ROUGHNESS_WEIGHT") == 0)
		return WAVEFORM_ROUGHNESS_WEIGHT;
	if (strcmp(name, "GATE_DURATION") == 0)
		return GATE_DURATION;
	if (strcmp(name, "DETECTION_DURATION") == 0)
		return DETECTION_DURATION;

	return NAN;
}

// Evaluate weighted sums of electrode potentials and their gradients at many points on the OpenCL device, if it is
// selected; returns 0 if the computation ran on the device
static int evaluate_expansion_batch_opencl
(
	struct Trap *trap,
	int n_sets,
	double *weights,
	int n_points,
	double (*positions)[3],
	double *phi,
	double (*grad_phi)[3]
)
{
	if (get_compute_device() != OPENCL_DEVICE)
		return 1;

	int n_spheres = NSPH_X * NSPH_Y * NSPH_Z;
	double *coefficients = malloc((size_t) trap->n_electrodes * n_spheres * SOLID_HARMONICS_LEN * 2 * sizeof(double));
	pack_expansion_coefficients(trap, coefficients);

	int status = opencl_evaluate_expansion(trap->n_electrodes, coefficients, n_sets, weights, n_points, positions, phi, grad_phi);

	free(coefficients);

	return status;
}

// Evaluate the potential (V) and its gradient (V/µm) at many positions; grad_phi may be NULL
void evaluate_trap_potential_batch
(
	struct Trap *trap,
	double *voltages,
	int n_points,
	double (*positions)[3],
	double *phi,
	double (*grad_phi)[3]
)
{
	double (*grad_buffer)[3] = (grad_phi != NULL) ? grad_phi : malloc(MAX(n_points, 1) * sizeof(double[3]));
	int status = evaluate_expansion_batch_opencl(trap, 1, voltages, n_points, positions, phi, grad_buffer);
	if (grad_phi == NULL)
		free(grad_buffer);

	if (status == 0)
		return;

	#pragma omp parallel for schedule(static)
	for (int p = 0; p < n_points; p++)
		evaluate_trap_potential(trap, voltages, positions[p], &phi[p], (grad_phi != NULL) ? grad_phi[p] : NULL, NULL);
}

// Evaluate the potential energy (J) of an ion, including the pseudopotential, at many positions
void evaluate_potential_energy_batch
(
	struct Trap *trap,
	const double m,
	const double q,
	double *voltages,
	int n_points,
	double (*positions)[3],
	double *U
)
{
	if (get_compute_device() == OPENCL_DEVICE)
	{
		// The DC potential and the gradient of the RF potential give the potential energy
		int n_e = trap->n_electrodes;
		double *weights = malloc(2 * n_e * sizeof(double));
		for (int e = 0; e < n_e; e++)
		{
			weights[e] = voltages[e];
			weights[n_e + e] = (e < trap->n_electrodes_rf) ? 1.0 : 0.0;
		}

		double *phi = malloc(MAX(n_points, 1) * 2 * sizeof(double));
		double (*grad_phi)[3] = malloc(MAX(n_points, 1) * 2 * sizeof(double[3]));

		int status = evaluate_expansion_batch_opencl(trap, 2, weights, n_points, positions, phi, grad_phi);

		if (status == 0)
		{
			double prefactor = (trap->n_electrodes_rf > 0 && trap->V_RF != 0.0 && trap->Omega_RF != 0.0) ? pow(q * trap->V_RF, 2) / (4 * m * pow(trap->Omega_RF * LENGTH_SCALE, 2)) : 0.0;

			for (int p = 0; p < n_points; p++)
			{
				double *g = grad_phi[2 * p + 1];
				U[p] = q * phi[2 * p] + prefactor * (g[0]*g[0] + g[1]*g[1] + g[2]*g[2]);
			}
		}

		free(grad_phi);
		free(phi);
		free(weights);

		if (status == 0)
			return;
	}

	#pragma omp parallel for schedule(static)
	for (int p = 0; p < n_points; p++)
		evaluate_potential_energy(trap, m, q, voltages, positions[p], &U[p], NULL, NULL);
}

// Evaluate the analytical (gapless plane) potential of an electrode at many positions
void evaluate_electrode_potential_analytical_batch
(
	struct Electrode *electrode,
	double electrode_position[3],
	int n_points,
	double (*positions)[3],
	double *phi
)
{
	#pragma omp parallel for schedule(static)
	for (int p = 0; p < n_points; p++)
		evaluate_electrode_potential_analytical(electrode, electrode_position, positions[p], &phi[p]);
}

// Evaluate the intensity (W/m^2) of the beam of a grating coupler at many positions
void evaluate_grating_beam_batch
(
	struct GratingCoupler *grating,
	int n_points,
	double (*positions)[3],
	double *intensity
)
{
	#pragma omp parallel for schedule(static)
	for (int p = 0; p < n_points; p++)
		evaluate_grating_beam(grating, positions[p], &intensity[p], NULL);
}

// Accessors for the parts of a co-design which bindings need, so that they do not depend on the layout of struct
// Codesign

// Number of records in the history, which may be read from another thread while the flow runs
int get_codesign_n_records
(
	struct Codesign *codesign
)
{
	int n_records;

	#pragma omp atomic read
	n_records = codesign->n_records;

	return n_records;
}

// Copy record r of the history of a co-design
void get_codesign_record
(
	struct Codesign *codesign,
	int r,
	struct CodesignRecord *record
)
{
	*record = codesign->records[r];
}

// Copy the parameters, environment and metrics of a co-design, skipping any which are NULL
void get_codesign_state
(
	struct Codesign *codesign,
	struct CodesignParameters *parameters,
	struct CodesignEnvironment *environment,
	struct QCMetrics *metrics
)
{
	if (parameters != NULL)
		*parameters = codesign->parameters;

	if (environment != NULL)
		*environment = codesign->environment;

	if (metrics != NULL)
		*metrics = codesign->metrics;
}

// Replace the parameters and/or environment of a co-design (either can be NULL), clearing the results
// which depend on them
void set_codesign_state
(
	struct Codesign *codesign,
	struct CodesignParameters *parameters,
	struct CodesignEnvironment *environment
)
{
	if (parameters != NULL)
		codesign->parameters = *parameters;

	if (environment != NULL)
	{
		// The trap is generated again if its number of segments changes
		if (environment->n_segments != codesign->environment.n_segments)
			codesign->solved_segment_width = 0.0;

		codesign->environment = *environment;
		codesign->primitive_status = -1;
	}
}

// Get the trap of a co-design, which is replaced whenever the trap stage changes its geometry
struct Trap* get_codesign_trap
(
	struct Codesign *codesign
)
{
	return &codesign->trap;
}

// Zones of the circuit's transport program: the positions of the gate and measurement zones (RF nulls, in µm), and the
// number of zones and of layers
void get_codesign_layout
(
	struct Codesign *codesign,
	double gate_position[3],
	double measure_position[3],
	int *n_zones,
	int *n_layers,
	int *n_ions
)
{
	for (int k = 0; k < 3; k++)
	{
		gate_position[k] = codesign->gate_position[k];
		measure_position[k] = codesign->measure_position[k];
	}

	*n_zones = codesign->n_zones;
	*n_layers = codesign->n_layers;
	*n_ions = codesign->n_ions;
}
