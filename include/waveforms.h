#include "data_structures.h"

typedef struct {
	int n_electrodes;
	int n_steps;
    double max_voltage_individual;
    double max_voltage_total;
    double position[3]; // generalize
    double omega[3]; // generalize
    int voltage_offset; // generalize
	int i;
	int j;
    enum
    {
        CONSTRAINT_EX_FIELD_X,
        CONSTRAINT_EX_FIELD_Y,
        CONSTRAINT_EX_FIELD_Z,
        CONSTRAINT_HESSIAN_XY,
        CONSTRAINT_HESSIAN_XZ,
        CONSTRAINT_HESSIAN_YZ,
        CONSTRAINT_AXIAL_CURVATURE
    } type;
} transport_waveform_constraint_data;

double waveform_cost_function(
    unsigned n,
    const double *x,
    double *grad,
    void *transport_waveform_obj
);

void optimize_primitive
(
	int max_iterations
);

void interpolate_waveform
(
	struct Trap *trap,
	struct TransportWaveform *waveform_coarse,
	struct TransportWaveform *waveform_fine,
	double dx
);

struct TransportWaveform* transport_program_to_waveform
(
	struct Trap *trap,
	struct TransportProgram *transport_program,
	int *n_waveform_steps_list // implicitly, length transport_program->n_transports
);

void compose_waveforms
(
	int n_waveforms,
	struct TransportWaveform *waveforms,
	double *waveform_scales
);

void compute_secular_frequencies
(
	double q,
	double m,
	double V_DC,
	double V_RF,
	double Omega,
	double Z_0,
	double *omega
);

void compute_U_pond
(
	double m,
	double omega_rf,
	double V_rf,
	double* U_rf,
	double* U_pond,
	double dx
);

