#include "acceleration.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <dlfcn.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <fftw3.h>

#include "defs.h"
#include "kernels_cl.h"

// Threads ------------------------------------------------------------------------------------------------------------

// Set the number of threads used by parallel regions (0 restores the default, which is one per core)
void set_num_threads
(
	int n_threads
)
{
#ifdef _OPENMP
	omp_set_num_threads((n_threads > 0) ? n_threads : omp_get_num_procs());
#else
	(void) n_threads;
#endif
}

// Get the number of threads used by parallel regions
int get_num_threads
(
)
{
#ifdef _OPENMP
	return omp_get_max_threads();
#else
	return 1;
#endif
}

// SHTOOLS plans FFTs on every call, and the FFTW planner is only safe to call from several threads once it is told so
void make_fftw_thread_safe
(
)
{
	static int done = 0;

	#pragma omp critical (fftw_thread_safety)
	{
		if (!done)
		{
			fftw_make_planner_thread_safe();
			done = 1;
		}
	}
}

// OpenCL -------------------------------------------------------------------------------------------------------------

// Minimal subset of the OpenCL API, which is loaded at runtime so that the library needs no OpenCL headers or libraries
// to build, and runs on machines without GPUs
typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef cl_ulong cl_bitfield;
typedef struct _cl_platform_id *cl_platform_id;
typedef struct _cl_device_id *cl_device_id;
typedef struct _cl_context *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_mem *cl_mem;
typedef struct _cl_program *cl_program;
typedef struct _cl_kernel *cl_kernel;
typedef struct _cl_event *cl_event;

#define CL_SUCCESS 0
#define CL_TRUE 1
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_DOUBLE_FP_CONFIG 0x1032
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_COPY_HOST_PTR (1 << 5)
#define CL_PROGRAM_BUILD_LOG 0x1183

struct OpenCL
{
	int initialized;
	int available;
	void *library;
	cl_context context;
	cl_command_queue queue;
	cl_device_id device;
	cl_program program;
	char device_name[256];

	cl_int (*GetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
	cl_int (*GetDeviceIDs)(cl_platform_id, cl_bitfield, cl_uint, cl_device_id *, cl_uint *);
	cl_int (*GetDeviceInfo)(cl_device_id, cl_uint, size_t, void *, size_t *);
	cl_context (*CreateContext)(const intptr_t *, cl_uint, const cl_device_id *, void *, void *, cl_int *);
	cl_command_queue (*CreateCommandQueue)(cl_context, cl_device_id, cl_bitfield, cl_int *);
	cl_mem (*CreateBuffer)(cl_context, cl_bitfield, size_t, void *, cl_int *);
	cl_int (*EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
	cl_int (*EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
	cl_program (*CreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
	cl_int (*BuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *, void *, void *);
	cl_int (*GetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void *, size_t *);
	cl_kernel (*CreateKernel)(cl_program, const char *, cl_int *);
	cl_int (*SetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
	cl_int (*EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
	cl_int (*Finish)(cl_command_queue);
	cl_int (*ReleaseMemObject)(cl_mem);
	cl_int (*ReleaseKernel)(cl_kernel);
	cl_int (*ReleaseProgram)(cl_program);
	cl_int (*ReleaseCommandQueue)(cl_command_queue);
	cl_int (*ReleaseContext)(cl_context);
};

static struct OpenCL opencl;

static enum ComputeDevice compute_device = CPU_DEVICE;

// Largest number of electrodes the kernels support
// Number of Verlet steps of an ensemble run by each kernel launch
#ifndef OPENCL_ENSEMBLE_STEPS
#define OPENCL_ENSEMBLE_STEPS 100
#endif

#ifndef MAX_ELECTRODES
#define MAX_ELECTRODES 64
#endif

static int load_opencl_function
(
	void **function,
	const char *name
)
{
	*function = dlsym(opencl.library, name);

	return *function == NULL;
}

// Find an OpenCL device with double precision, preferring GPUs; other devices (such as CPU implementations) are only
// used when TIOPT_OPENCL_ANY_DEVICE is set, which is useful for testing. TIOPT_OPENCL_DEVICE selects among the
// suitable devices by index. Returns 0 if a device is ready
static int initialize_opencl
(
)
{
	if (opencl.initialized)
		return !opencl.available;

	opencl.initialized = 1;
	opencl.available = 0;

	opencl.library = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
	if (opencl.library == NULL)
		opencl.library = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);

	if (opencl.library == NULL)
	{
		printf("OpenCL is not available (no OpenCL library was found)\n");
		return 1;
	}

	int missing = 0;
	missing += load_opencl_function((void **) &opencl.GetPlatformIDs, "clGetPlatformIDs");
	missing += load_opencl_function((void **) &opencl.GetDeviceIDs, "clGetDeviceIDs");
	missing += load_opencl_function((void **) &opencl.GetDeviceInfo, "clGetDeviceInfo");
	missing += load_opencl_function((void **) &opencl.CreateContext, "clCreateContext");
	missing += load_opencl_function((void **) &opencl.CreateCommandQueue, "clCreateCommandQueue");
	missing += load_opencl_function((void **) &opencl.CreateBuffer, "clCreateBuffer");
	missing += load_opencl_function((void **) &opencl.EnqueueWriteBuffer, "clEnqueueWriteBuffer");
	missing += load_opencl_function((void **) &opencl.EnqueueReadBuffer, "clEnqueueReadBuffer");
	missing += load_opencl_function((void **) &opencl.CreateProgramWithSource, "clCreateProgramWithSource");
	missing += load_opencl_function((void **) &opencl.BuildProgram, "clBuildProgram");
	missing += load_opencl_function((void **) &opencl.GetProgramBuildInfo, "clGetProgramBuildInfo");
	missing += load_opencl_function((void **) &opencl.CreateKernel, "clCreateKernel");
	missing += load_opencl_function((void **) &opencl.SetKernelArg, "clSetKernelArg");
	missing += load_opencl_function((void **) &opencl.EnqueueNDRangeKernel, "clEnqueueNDRangeKernel");
	missing += load_opencl_function((void **) &opencl.Finish, "clFinish");
	missing += load_opencl_function((void **) &opencl.ReleaseMemObject, "clReleaseMemObject");
	missing += load_opencl_function((void **) &opencl.ReleaseKernel, "clReleaseKernel");
	missing += load_opencl_function((void **) &opencl.ReleaseProgram, "clReleaseProgram");
	missing += load_opencl_function((void **) &opencl.ReleaseCommandQueue, "clReleaseCommandQueue");
	missing += load_opencl_function((void **) &opencl.ReleaseContext, "clReleaseContext");

	if (missing)
	{
		printf("OpenCL is not available (the OpenCL library is missing %d functions)\n", missing);
		return 1;
	}

	cl_platform_id platforms[16];
	cl_uint n_platforms = 0;
	if (opencl.GetPlatformIDs(16, platforms, &n_platforms) != CL_SUCCESS || n_platforms == 0)
	{
		printf("OpenCL is not available (no OpenCL platforms were found)\n");
		return 1;
	}

	int any_device = getenv("TIOPT_OPENCL_ANY_DEVICE") != NULL;
	int requested_device = getenv("TIOPT_OPENCL_DEVICE") != NULL ? atoi(getenv("TIOPT_OPENCL_DEVICE")) : 0;

	// Collect every suitable device, in order of platforms
	int n_suitable = 0;
	for (cl_uint p = 0; p < n_platforms && !opencl.available; p++)
	{
		cl_device_id devices[16];
		cl_uint n_devices = 0;
		if (opencl.GetDeviceIDs(platforms[p], any_device ? CL_DEVICE_TYPE_ALL : CL_DEVICE_TYPE_GPU, 16, devices, &n_devices) != CL_SUCCESS)
			continue;

		for (cl_uint d = 0; d < n_devices; d++)
		{
			cl_ulong double_config = 0;
			opencl.GetDeviceInfo(devices[d], CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(double_config), &double_config, NULL);

			if (double_config == 0)
				continue;

			if (n_suitable++ == requested_device)
			{
				opencl.device = devices[d];
				opencl.available = 1;
				break;
			}
		}
	}

	if (!opencl.available)
	{
		printf("OpenCL is not available (no %sdevice with double precision was found)\n", any_device ? "" : "GPU ");
		return 1;
	}

	opencl.GetDeviceInfo(opencl.device, CL_DEVICE_NAME, sizeof(opencl.device_name) - 1, opencl.device_name, NULL);

	cl_int error;
	opencl.context = opencl.CreateContext(NULL, 1, &opencl.device, NULL, NULL, &error);
	if (error == CL_SUCCESS)
		opencl.queue = opencl.CreateCommandQueue(opencl.context, opencl.device, 0, &error);

	// Compile the kernels with the constants of this build
	if (error == CL_SUCCESS)
	{
		const char *source = kernels_cl;
		opencl.program = opencl.CreateProgramWithSource(opencl.context, 1, &source, NULL, &error);
	}

	if (error == CL_SUCCESS)
	{
		char options[512];
		snprintf(
			options, sizeof(options),
			"-DLMAX=%d -DNSPH_X=%d -DNSPH_Y=%d -DNSPH_Z=%d -DLENGTH_SCALE=%.17g -DMAX_ELECTRODES=%d",
			LMAX, NSPH_X, NSPH_Y, NSPH_Z, LENGTH_SCALE, MAX_ELECTRODES
		);

		error = opencl.BuildProgram(opencl.program, 1, &opencl.device, options, NULL, NULL);
		if (error != CL_SUCCESS)
		{
			char build_log[16384];
			build_log[0] = '\0';
			opencl.GetProgramBuildInfo(opencl.program, opencl.device, CL_PROGRAM_BUILD_LOG, sizeof(build_log) - 1, build_log, NULL);
			printf("ERROR: Failed to compile OpenCL kernels:\n%s\n", build_log);
		}
	}

	if (error != CL_SUCCESS)
	{
		printf("OpenCL is not available (setting up %s failed with error %d)\n", opencl.device_name, error);
		opencl.available = 0;
		return 1;
	}

	printf("Using OpenCL device %s\n", opencl.device_name);

	return 0;
}

// Choose the device used for the computations which support acceleration; returns 1 (and keeps using the CPU) if no
// OpenCL device is available
int set_compute_device
(
	enum ComputeDevice device
)
{
	int status = 0;

	#pragma omp critical (opencl)
	{
		if (device == OPENCL_DEVICE && initialize_opencl() != 0)
			status = 1;
		else
			compute_device = device;
	}

	return status;
}

// Get the device which relaxation, potential maps and ensembles run on
enum ComputeDevice get_compute_device
(
)
{
	return compute_device;
}

// Check whether an OpenCL device with double precision can be used, initializing OpenCL if needed
int opencl_available
(
)
{
	int available;

	#pragma omp critical (opencl)
	available = (initialize_opencl() == 0);

	return available;
}

// Get the name of the OpenCL device, or an empty string if there is none
char* opencl_device_name
(
)
{
	return opencl.available ? opencl.device_name : "";
}

// Release the OpenCL context, queue and program, if they were created
void release_opencl
(
)
{
	#pragma omp critical (opencl)
	{
		if (opencl.available)
		{
			opencl.ReleaseProgram(opencl.program);
			opencl.ReleaseCommandQueue(opencl.queue);
			opencl.ReleaseContext(opencl.context);
		}

		opencl.initialized = 0;
		opencl.available = 0;
		compute_device = CPU_DEVICE;
	}
}

// Create a buffer, optionally initialized from host memory; returns NULL on failure
static cl_mem create_buffer
(
	cl_bitfield flags,
	size_t size,
	void *host_data
)
{
	cl_int error;
	cl_mem buffer = opencl.CreateBuffer(opencl.context, flags | (host_data != NULL ? CL_MEM_COPY_HOST_PTR : 0), (size > 0) ? size : 1, host_data, &error);

	return (error == CL_SUCCESS) ? buffer : NULL;
}

static size_t round_up
(
	size_t n,
	size_t multiple
)
{
	return ((n + multiple - 1) / multiple) * multiple;
}

// Successive overrelaxation of the potential V (with cells marked in fixed held constant) on the OpenCL device, checking
// the residual every 10 iterations; returns 0 if the computation ran on the device
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
)
{
	int status = 1;

	#pragma omp critical (opencl)
	{
		if (compute_device == OPENCL_DEVICE && initialize_opencl() == 0)
		{
			size_t n_cells = (size_t) NX * NY * NZ;
			size_t n_interior = (size_t) (NX - 2) * (NY - 2) * (NZ - 2);
			size_t local_size = 256;
			size_t n_groups = round_up(n_interior, local_size) / local_size;

			cl_int error;
			cl_mem V_buffer = create_buffer(CL_MEM_READ_WRITE, n_cells * sizeof(double), V);
			cl_mem fixed_buffer = create_buffer(CL_MEM_READ_ONLY, n_cells, fixed);
			cl_mem maxima_buffer = create_buffer(CL_MEM_READ_WRITE, n_groups * sizeof(double), NULL);
			cl_kernel sweep = opencl.CreateKernel(opencl.program, "relaxation_sweep", &error);
			cl_kernel residual = opencl.CreateKernel(opencl.program, "relaxation_residual", &error);
			double *maxima = malloc(n_groups * sizeof(double));

			if (V_buffer != NULL && fixed_buffer != NULL && maxima_buffer != NULL && sweep != NULL && residual != NULL)
			{
				opencl.SetKernelArg(sweep, 0, sizeof(cl_mem), &V_buffer);
				opencl.SetKernelArg(sweep, 1, sizeof(cl_mem), &fixed_buffer);
				opencl.SetKernelArg(sweep, 2, sizeof(int), &NX);
				opencl.SetKernelArg(sweep, 3, sizeof(int), &NY);
				opencl.SetKernelArg(sweep, 4, sizeof(int), &NZ);

				opencl.SetKernelArg(residual, 0, sizeof(cl_mem), &V_buffer);
				opencl.SetKernelArg(residual, 1, sizeof(cl_mem), &fixed_buffer);
				opencl.SetKernelArg(residual, 2, sizeof(int), &NX);
				opencl.SetKernelArg(residual, 3, sizeof(int), &NY);
				opencl.SetKernelArg(residual, 4, sizeof(int), &NZ);
				opencl.SetKernelArg(residual, 5, sizeof(cl_mem), &maxima_buffer);
				opencl.SetKernelArg(residual, 6, local_size * sizeof(double), NULL);

				size_t sweep_size[3] = { NX - 2, NY - 2, (NZ - 1) / 2 };
				size_t residual_size = n_groups * local_size;

				double omega = chebyshev_acceleration ? 1.0 : 2.0 / (1.0 + sqrt(1.0 - rho*rho));

				status = 0;
				int it;
				for (it = 0; it < max_iterations && status == 0; it++)
				{
					for (int p = 0; p < 2; p++)
					{
						opencl.SetKernelArg(sweep, 5, sizeof(int), &p);
						opencl.SetKernelArg(sweep, 6, sizeof(double), &omega);
						if (opencl.EnqueueNDRangeKernel(opencl.queue, sweep, 3, NULL, sweep_size, NULL, 0, NULL, NULL) != CL_SUCCESS)
							status = 1;

						// Chebyshev acceleration (Numerical Recipes, Section 19.5)
						if (chebyshev_acceleration)
							omega = (it == 0 && p == 0) ? 1 / (1 - 0.5 * rho*rho) : 1 / (1 - 0.25 * rho*rho * omega);
					}

					if ((it + 1) % 10 == 0 || it == max_iterations - 1)
					{
						opencl.EnqueueNDRangeKernel(opencl.queue, residual, 1, NULL, &residual_size, &local_size, 0, NULL, NULL);
						opencl.EnqueueReadBuffer(opencl.queue, maxima_buffer, CL_TRUE, 0, n_groups * sizeof(double), maxima, 0, NULL, NULL);

						double xi_max = 0.0;
						for (size_t g = 0; g < n_groups; g++)
							xi_max = fmax(xi_max, maxima[g]);

						if (xi_max / 6.0 < RELAXATION_TOLERANCE)
						{
							printf("Stopping relaxation loop early at iteration %d due to small residuals\n", it);
							break;
						}
					}
				}

				if (status == 0)
					status = opencl.EnqueueReadBuffer(opencl.queue, V_buffer, CL_TRUE, 0, n_cells * sizeof(double), V, 0, NULL, NULL) != CL_SUCCESS;

				*iterations = it;
			}

			free(maxima);
			if (sweep != NULL)
				opencl.ReleaseKernel(sweep);
			if (residual != NULL)
				opencl.ReleaseKernel(residual);
			if (V_buffer != NULL)
				opencl.ReleaseMemObject(V_buffer);
			if (fixed_buffer != NULL)
				opencl.ReleaseMemObject(fixed_buffer);
			if (maxima_buffer != NULL)
				opencl.ReleaseMemObject(maxima_buffer);
		}
	}

	return status;
}

// Coordinates of the center of the first sphere, padded to the size of an OpenCL double3
static void sphere_lattice_origin
(
	double (*center_0)[4]
)
{
	(*center_0)[0] = SPH_CENTER_X(0);
	(*center_0)[1] = SPH_CENTER_Y(0);
	(*center_0)[2] = SPH_CENTER_Z(0);
	(*center_0)[3] = 0.0;
}

// Evaluate up to two weighted sums of electrode potentials (V) and their gradients (V/µm) at many points on the OpenCL
// device, where coefficients holds the normalized expansion coefficients of every electrode (see
// pack_expansion_coefficients); returns 0 if the computation ran on the device
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
)
{
	int status = 1;

	if (n_electrodes > MAX_ELECTRODES || n_sets > 2 || n_points == 0)
		return 1;

	#pragma omp critical (opencl)
	{
		if (compute_device == OPENCL_DEVICE && initialize_opencl() == 0)
		{
			size_t coefficients_size = (size_t) n_electrodes * NSPH_X * NSPH_Y * NSPH_Z * SOLID_HARMONICS_LEN * 2 * sizeof(double);

			cl_int error;
			cl_mem coefficients_buffer = create_buffer(CL_MEM_READ_ONLY, coefficients_size, coefficients);
			cl_mem weights_buffer = create_buffer(CL_MEM_READ_ONLY, (size_t) n_sets * n_electrodes * sizeof(double), weights);
			cl_mem positions_buffer = create_buffer(CL_MEM_READ_ONLY, (size_t) n_points * 3 * sizeof(double), positions);
			cl_mem phi_buffer = create_buffer(CL_MEM_READ_WRITE, (size_t) n_points * n_sets * sizeof(double), NULL);
			cl_mem grad_buffer = create_buffer(CL_MEM_READ_WRITE, (size_t) n_points * n_sets * 3 * sizeof(double), NULL);
			cl_kernel kernel = opencl.CreateKernel(opencl.program, "evaluate_expansion_points", &error);

			if (coefficients_buffer != NULL && weights_buffer != NULL && positions_buffer != NULL && phi_buffer != NULL && grad_buffer != NULL && kernel != NULL)
			{
				double center_0[4];
				sphere_lattice_origin(&center_0);
				double spacing = SPH_SPACING;
				double radius = SPH_R;

				opencl.SetKernelArg(kernel, 0, sizeof(cl_mem), &coefficients_buffer);
				opencl.SetKernelArg(kernel, 1, sizeof(int), &n_electrodes);
				opencl.SetKernelArg(kernel, 2, sizeof(cl_mem), &weights_buffer);
				opencl.SetKernelArg(kernel, 3, sizeof(int), &n_sets);
				opencl.SetKernelArg(kernel, 4, sizeof(cl_mem), &positions_buffer);
				opencl.SetKernelArg(kernel, 5, sizeof(int), &n_points);
				opencl.SetKernelArg(kernel, 6, sizeof(center_0), center_0);
				opencl.SetKernelArg(kernel, 7, sizeof(double), &spacing);
				opencl.SetKernelArg(kernel, 8, sizeof(double), &radius);
				opencl.SetKernelArg(kernel, 9, sizeof(cl_mem), &phi_buffer);
				opencl.SetKernelArg(kernel, 10, sizeof(cl_mem), &grad_buffer);

				size_t local_size = 64;
				size_t global_size = round_up(n_points, local_size);

				if (opencl.EnqueueNDRangeKernel(opencl.queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL) == CL_SUCCESS)
				{
					status = 0;
					status += opencl.EnqueueReadBuffer(opencl.queue, phi_buffer, CL_TRUE, 0, (size_t) n_points * n_sets * sizeof(double), phi, 0, NULL, NULL) != CL_SUCCESS;

					if (grad_phi != NULL)
						status += opencl.EnqueueReadBuffer(opencl.queue, grad_buffer, CL_TRUE, 0, (size_t) n_points * n_sets * 3 * sizeof(double), grad_phi, 0, NULL, NULL) != CL_SUCCESS;
				}
			}

			if (kernel != NULL)
				opencl.ReleaseKernel(kernel);
			cl_mem buffers[5] = { coefficients_buffer, weights_buffer, positions_buffer, phi_buffer, grad_buffer };
			for (int b = 0; b < 5; b++)
				if (buffers[b] != NULL)
					opencl.ReleaseMemObject(buffers[b]);
		}
	}

	return status;
}

// Integrate the motion of an ensemble of ions with velocity Verlet on the OpenCL device, updating their positions (µm)
// and velocities (µm/s) in place; returns 0 if the computation ran on the device
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
)
{
	int status = 1;

	if (n_electrodes > MAX_ELECTRODES || n_particles == 0)
		return 1;

	#pragma omp critical (opencl)
	{
		if (compute_device == OPENCL_DEVICE && initialize_opencl() == 0)
		{
			size_t coefficients_size = (size_t) n_electrodes * NSPH_X * NSPH_Y * NSPH_Z * SOLID_HARMONICS_LEN * 2 * sizeof(double);
			size_t state_size = (size_t) n_particles * 3 * sizeof(double);

			// A waveform with no steps holds every electrode at 0 V
			double zero_time = 0.0;
			double *zero_voltages = calloc(n_electrodes, sizeof(double));
			int n_steps = (n_waveform_steps > 0) ? n_waveform_steps : 1;

			cl_int error;
			cl_mem coefficients_buffer = create_buffer(CL_MEM_READ_ONLY, coefficients_size, coefficients);
			cl_mem times_buffer = create_buffer(CL_MEM_READ_ONLY, n_steps * sizeof(double), (n_waveform_steps > 0) ? waveform_times : &zero_time);
			cl_mem voltages_buffer = create_buffer(CL_MEM_READ_ONLY, (size_t) n_steps * n_electrodes * sizeof(double), (n_waveform_steps > 0) ? waveform_voltages : zero_voltages);
			cl_mem positions_buffer = create_buffer(CL_MEM_READ_WRITE, state_size, positions);
			cl_mem velocities_buffer = create_buffer(CL_MEM_READ_WRITE, state_size, velocities);
			cl_mem accelerations_buffer = create_buffer(CL_MEM_READ_WRITE, state_size, NULL);
			cl_kernel initial = opencl.CreateKernel(opencl.program, "ensemble_acceleration", &error);
			cl_kernel step = opencl.CreateKernel(opencl.program, "ensemble_verlet_steps", &error);

			if (coefficients_buffer != NULL && times_buffer != NULL && voltages_buffer != NULL && positions_buffer != NULL && velocities_buffer != NULL && accelerations_buffer != NULL && initial != NULL && step != NULL)
			{
				double center_0[4];
				sphere_lattice_origin(&center_0);
				double spacing = SPH_SPACING;
				double radius = SPH_R;

				// Arguments shared by both kernels, which come first
				cl_kernel kernels[2] = { initial, step };
				for (int k = 0; k < 2; k++)
				{
					opencl.SetKernelArg(kernels[k], 0, sizeof(cl_mem), &coefficients_buffer);
					opencl.SetKernelArg(kernels[k], 1, sizeof(int), &n_electrodes);
					opencl.SetKernelArg(kernels[k], 2, sizeof(int), &n_electrodes_rf);
					opencl.SetKernelArg(kernels[k], 3, sizeof(double), &V_RF);
					opencl.SetKernelArg(kernels[k], 4, sizeof(double), &Omega_RF);
					opencl.SetKernelArg(kernels[k], 5, sizeof(double), &m);
					opencl.SetKernelArg(kernels[k], 6, sizeof(double), &q);
					opencl.SetKernelArg(kernels[k], 7, sizeof(cl_mem), &times_buffer);
					opencl.SetKernelArg(kernels[k], 8, sizeof(cl_mem), &voltages_buffer);
					opencl.SetKernelArg(kernels[k], 9, sizeof(int), &n_steps);
					opencl.SetKernelArg(kernels[k], 10, sizeof(int), &full_rf);
				}

				opencl.SetKernelArg(initial, 11, sizeof(double), &t_0);
				opencl.SetKernelArg(initial, 12, sizeof(center_0), center_0);
				opencl.SetKernelArg(initial, 13, sizeof(double), &spacing);
				opencl.SetKernelArg(initial, 14, sizeof(double), &radius);
				opencl.SetKernelArg(initial, 15, sizeof(cl_mem), &positions_buffer);
				opencl.SetKernelArg(initial, 16, sizeof(cl_mem), &accelerations_buffer);
				opencl.SetKernelArg(initial, 17, sizeof(int), &n_particles);

				opencl.SetKernelArg(step, 11, sizeof(double), &t_0);
				opencl.SetKernelArg(step, 12, sizeof(double), &dt);
				opencl.SetKernelArg(step, 13, sizeof(center_0), center_0);
				opencl.SetKernelArg(step, 14, sizeof(double), &spacing);
				opencl.SetKernelArg(step, 15, sizeof(double), &radius);
				opencl.SetKernelArg(step, 16, sizeof(cl_mem), &positions_buffer);
				opencl.SetKernelArg(step, 17, sizeof(cl_mem), &velocities_buffer);
				opencl.SetKernelArg(step, 18, sizeof(cl_mem), &accelerations_buffer);
				opencl.SetKernelArg(step, 19, sizeof(int), &n_particles);

				size_t local_size = 64;
				size_t global_size = round_up(n_particles, local_size);

				status = opencl.EnqueueNDRangeKernel(opencl.queue, initial, 1, NULL, &global_size, &local_size, 0, NULL, NULL) != CL_SUCCESS;

				for (int first_step = 0; first_step < n_it && status == 0; first_step += OPENCL_ENSEMBLE_STEPS)
				{
					int n_steps = MIN(OPENCL_ENSEMBLE_STEPS, n_it - first_step);
					opencl.SetKernelArg(step, 20, sizeof(int), &first_step);
					opencl.SetKernelArg(step, 21, sizeof(int), &n_steps);
					status = opencl.EnqueueNDRangeKernel(opencl.queue, step, 1, NULL, &global_size, &local_size, 0, NULL, NULL) != CL_SUCCESS;

					// Keep the queue from growing without bound
					if ((first_step / OPENCL_ENSEMBLE_STEPS) % 100 == 99)
						opencl.Finish(opencl.queue);
				}

				if (status == 0)
				{
					status += opencl.EnqueueReadBuffer(opencl.queue, positions_buffer, CL_TRUE, 0, state_size, positions, 0, NULL, NULL) != CL_SUCCESS;
					status += opencl.EnqueueReadBuffer(opencl.queue, velocities_buffer, CL_TRUE, 0, state_size, velocities, 0, NULL, NULL) != CL_SUCCESS;
				}
			}

			free(zero_voltages);
			if (initial != NULL)
				opencl.ReleaseKernel(initial);
			if (step != NULL)
				opencl.ReleaseKernel(step);
			cl_mem buffers[6] = { coefficients_buffer, times_buffer, voltages_buffer, positions_buffer, velocities_buffer, accelerations_buffer };
			for (int b = 0; b < 6; b++)
				if (buffers[b] != NULL)
					opencl.ReleaseMemObject(buffers[b]);
		}
	}

	return status;
}
