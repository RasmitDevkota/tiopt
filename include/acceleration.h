#pragma once

enum ComputeDevice
{
	CPU_DEVICE,
	OPENCL_DEVICE,
};

void set_num_threads
(
	int n_threads
);

int get_num_threads
(
);

void make_fftw_thread_safe
(
);

int set_compute_device
(
	enum ComputeDevice device
);

enum ComputeDevice get_compute_device
(
);

int opencl_available
(
);

char* opencl_device_name
(
);

void release_opencl
(
);

int opencl_relaxation
(
	const int NX,
	const int NY,
	const int NZ,
	unsigned char *fixed,
	double *V,
	int max_iterations,
	int chebyshev_acceleration,
	double rho,
	int *iterations
);

int opencl_evaluate_expansion
(
	int n_electrodes,
	double *coefficients,
	int n_sets,
	double *weights,
	int n_points,
	double (*positions)[3],
	double *phi,
	double (*grad_phi)[3]
);

int opencl_simulate_ensemble
(
	int n_electrodes,
	int n_electrodes_rf,
	double *coefficients,
	double V_RF,
	double Omega_RF,
	const double m,
	const double q,
	int n_waveform_steps,
	double *waveform_times,
	double *waveform_voltages,
	const int full_rf,
	int n_particles,
	double (*positions)[3],
	double (*velocities)[3],
	const int n_it,
	const double t_0,
	const double dt
);
