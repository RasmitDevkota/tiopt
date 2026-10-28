// OpenCL kernels for tiopt, which acceleration.c compiles at runtime with the constants from defs.h (LMAX, NSPH_X,
// NSPH_Y, NSPH_Z, LENGTH_SCALE and MAX_ELECTRODES) defined as build options

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define SOLID_HARMONICS_LEN ((LMAX + 1) * (LMAX + 2) / 2)

// Relaxation ---------------------------------------------------------------------------------------------------------

// One half-sweep of red-black successive overrelaxation, updating the interior cells of the given parity
__kernel void relaxation_sweep
(
	__global double *V,
	__global const uchar *fixed,
	const int NX,
	const int NY,
	const int NZ,
	const int parity,
	const double omega
)
{
	int x = get_global_id(0) + 1;
	int y = get_global_id(1) + 1;
	int z = 1 + 2 * (int) get_global_id(2) + ((x + y + 1 + parity) & 1);

	if (x >= NX-1 || y >= NY-1 || z >= NZ-1)
		return;

	size_t i = ((size_t) x * NY + y) * NZ + z;

	if (fixed[i])
		return;

	size_t stride_x = (size_t) NY * NZ;

	double xi = V[i - stride_x] + V[i + stride_x] + V[i - NZ] + V[i + NZ] + V[i - 1] + V[i + 1] - 6 * V[i];

	V[i] += omega * xi / 6;
}

// Largest residual of the discretized Laplace equation within each work group
__kernel void relaxation_residual
(
	__global const double *V,
	__global const uchar *fixed,
	const int NX,
	const int NY,
	const int NZ,
	__global double *group_maxima,
	__local double *scratch
)
{
	size_t gid = get_global_id(0);
	size_t n_interior = (size_t) (NX - 2) * (NY - 2) * (NZ - 2);

	double value = 0.0;
	if (gid < n_interior)
	{
		int z = 1 + gid % (NZ - 2);
		size_t rest = gid / (NZ - 2);
		int y = 1 + rest % (NY - 2);
		int x = 1 + rest / (NY - 2);

		size_t i = ((size_t) x * NY + y) * NZ + z;
		size_t stride_x = (size_t) NY * NZ;

		if (!fixed[i])
			value = fabs(V[i - stride_x] + V[i + stride_x] + V[i - NZ] + V[i + NZ] + V[i - 1] + V[i + 1] - 6 * V[i]);
	}

	size_t lid = get_local_id(0);
	scratch[lid] = value;
	barrier(CLK_LOCAL_MEM_FENCE);

	for (size_t stride = get_local_size(0) / 2; stride > 0; stride /= 2)
	{
		if (lid < stride)
			scratch[lid] = fmax(scratch[lid], scratch[lid + stride]);
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (lid == 0)
		group_maxima[get_group_id(0)] = scratch[0];
}

// Spherical harmonics expansions ------------------------------------------------------------------------------------

// A regular solid harmonic with its gradient and the unique components of its Hessian (xx, xy, xz, yy, yz, zz)
typedef struct
{
	double2 value;
	double2 grad[3];
	double2 hess[6];
} Harmonic;

constant int hessian_k[6] = { 0, 0, 0, 1, 1, 2 };
constant int hessian_j[6] = { 0, 1, 2, 1, 2, 2 };

inline double2 cmul
(
	double2 a,
	double2 b
)
{
	return (double2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// c f B by the product rule, where f is a factor of the recurrence relations with known derivatives
Harmonic product
(
	double c,
	Harmonic f,
	Harmonic B,
	int order
)
{
	Harmonic A;

	A.value = c * cmul(f.value, B.value);

	if (order >= 1)
		for (int k = 0; k < 3; k++)
			A.grad[k] = c * (cmul(f.grad[k], B.value) + cmul(f.value, B.grad[k]));

	if (order >= 2)
		for (int h = 0; h < 6; h++)
		{
			int k = hessian_k[h];
			int j = hessian_j[h];

			A.hess[h] = c * (cmul(f.hess[h], B.value) + cmul(f.grad[k], B.grad[j]) + cmul(f.grad[j], B.grad[k]) + cmul(f.value, B.hess[h]));
		}

	return A;
}

Harmonic zero_harmonic
(
)
{
	Harmonic A;

	A.value = (double2)(0.0, 0.0);
	for (int k = 0; k < 3; k++)
		A.grad[k] = (double2)(0.0, 0.0);
	for (int h = 0; h < 6; h++)
		A.hess[h] = (double2)(0.0, 0.0);

	return A;
}

// Accumulate two weighted sums of electrode expansions at the scaled offset u from the center of a sphere, combining the
// coefficients of the electrodes with weights[set * n_electrodes + e] as each solid harmonic is generated
void accumulate_expansion
(
	__global const double *coefficients,
	const int n_electrodes,
	const int sphere,
	const double *weights,
	const double3 u,
	const int order,
	double value[2],
	double grad[2][3],
	double hess[2][6]
)
{
	Harmonic w = zero_harmonic();
	w.value = (double2)(u.x, u.y);
	w.grad[0] = (double2)(1.0, 0.0);
	w.grad[1] = (double2)(0.0, 1.0);

	Harmonic z = zero_harmonic();
	z.value = (double2)(u.z, 0.0);
	z.grad[2] = (double2)(1.0, 0.0);

	Harmonic s = zero_harmonic();
	s.value = (double2)(u.x * u.x + u.y * u.y + u.z * u.z, 0.0);
	s.grad[0] = (double2)(2 * u.x, 0.0);
	s.grad[1] = (double2)(2 * u.y, 0.0);
	s.grad[2] = (double2)(2 * u.z, 0.0);
	s.hess[0] = (double2)(2.0, 0.0);
	s.hess[3] = (double2)(2.0, 0.0);
	s.hess[5] = (double2)(2.0, 0.0);

	for (int set = 0; set < 2; set++)
	{
		value[set] = 0.0;
		for (int k = 0; k < 3; k++)
			grad[set][k] = 0.0;
		for (int h = 0; h < 6; h++)
			hess[set][h] = 0.0;
	}

	Harmonic diagonal = zero_harmonic();
	diagonal.value = (double2)(1.0, 0.0);

	for (int m = 0; m <= LMAX; m++)
	{
		// R_m^m = (2m - 1) w R_(m-1)^(m-1)
		if (m > 0)
			diagonal = product(2*m - 1, w, diagonal, order);

		Harmonic previous_2 = zero_harmonic();
		Harmonic previous_1 = zero_harmonic();
		Harmonic current = diagonal;

		for (int l = m; l <= LMAX; l++)
		{
			if (l == m + 1)
			{
				// R_(m+1)^m = (2m + 1) z R_m^m
				current = product(2*m + 1, z, diagonal, order);
			}
			else if (l >= m + 2)
			{
				// (l - m) R_l^m = (2l - 1) z R_(l-1)^m - (l + m - 1) |u|^2 R_(l-2)^m
				Harmonic A = product((2*l - 1.0) / (l - m), z, previous_1, order);
				Harmonic B = product((l + m - 1.0) / (l - m), s, previous_2, order);

				current.value = A.value - B.value;
				for (int k = 0; k < 3; k++)
					current.grad[k] = A.grad[k] - B.grad[k];
				for (int h = 0; h < 6; h++)
					current.hess[h] = A.hess[h] - B.hess[h];
			}

			// Combine the coefficients of every electrode for this harmonic, which already include its normalization
			int idx = l*(l+1)/2 + m;
			for (int set = 0; set < 2; set++)
			{
				double c = 0.0;
				double s_coefficient = 0.0;
				for (int e = 0; e < n_electrodes; e++)
				{
					double weight = weights[set * n_electrodes + e];

					if (weight != 0.0)
					{
						size_t offset = (((size_t) e * NSPH_X * NSPH_Y * NSPH_Z + sphere) * SOLID_HARMONICS_LEN + idx) * 2;
						c += weight * coefficients[offset];
						s_coefficient += weight * coefficients[offset + 1];
					}
				}

				value[set] += c * current.value.x + s_coefficient * current.value.y;

				if (order >= 1)
					for (int k = 0; k < 3; k++)
						grad[set][k] += c * current.grad[k].x + s_coefficient * current.grad[k].y;

				if (order >= 2)
					for (int h = 0; h < 6; h++)
						hess[set][h] += c * current.hess[h].x + s_coefficient * current.hess[h].y;
			}

			previous_2 = previous_1;
			previous_1 = current;
		}
	}
}

// Nearest sphere to a position, and the offset from its center scaled by its radius, as in find_nearest_sphere
int nearest_sphere
(
	const double3 p,
	const double3 center_0,
	const double spacing,
	const double radius,
	double3 *u
)
{
	int sx = clamp((int) round((p.x - center_0.x) / spacing), 0, NSPH_X - 1);
	int sy = clamp((int) round((p.y - center_0.y) / spacing), 0, NSPH_Y - 1);
	int sz = clamp((int) round((p.z - center_0.z) / spacing), 0, NSPH_Z - 1);

	double3 center = center_0 + spacing * (double3)(sx, sy, sz);
	*u = (p - center) / radius;

	return (sx * NSPH_Y + sy) * NSPH_Z + sz;
}

// Potential and gradient of up to two weighted sums of electrode potentials at many points
__kernel void evaluate_expansion_points
(
	__global const double *coefficients,
	const int n_electrodes,
	__global const double *set_weights,
	const int n_sets,
	__global const double *positions,
	const int n_points,
	const double3 center_0,
	const double spacing,
	const double radius,
	__global double *phi,
	__global double *grad_phi
)
{
	int p = get_global_id(0);

	if (p >= n_points)
		return;

	double weights[2 * MAX_ELECTRODES];
	for (int i = 0; i < 2 * n_electrodes; i++)
		weights[i] = (i < n_sets * n_electrodes) ? set_weights[i] : 0.0;

	double3 u;
	int sphere = nearest_sphere((double3)(positions[3*p], positions[3*p + 1], positions[3*p + 2]), center_0, spacing, radius, &u);

	double value[2];
	double grad[2][3];
	double hess[2][6];
	accumulate_expansion(coefficients, n_electrodes, sphere, weights, u, 1, value, grad, hess);

	for (int set = 0; set < n_sets; set++)
	{
		phi[p * n_sets + set] = value[set];

		for (int k = 0; k < 3; k++)
			grad_phi[(p * n_sets + set) * 3 + k] = grad[set][k] / radius;
	}
}

// Ion dynamics -------------------------------------------------------------------------------------------------------

// Electrode voltages of a waveform at a time, interpolated linearly and held outside of the waveform
void interpolate_waveform
(
	__global const double *times,
	__global const double *voltages,
	const int n_steps,
	const int n_electrodes,
	const double t,
	double *out
)
{
	int lo = 0;
	int hi = n_steps - 1;

	if (t <= times[0])
		hi = 0;
	else if (t >= times[n_steps - 1])
		lo = n_steps - 1;
	else
		while (hi - lo > 1)
		{
			int mid = (lo + hi) / 2;

			if (times[mid] <= t)
				lo = mid;
			else
				hi = mid;
		}

	double span = times[hi] - times[lo];
	double fraction = (hi > lo && span > 0.0) ? (t - times[lo]) / span : 1.0;

	for (int e = 0; e < n_electrodes; e++)
		out[e] = (1 - fraction) * voltages[lo * n_electrodes + e] + fraction * voltages[hi * n_electrodes + e];
}

// Acceleration (µm/s^2) of an ion at a position, as in compute_acceleration
double3 ion_acceleration
(
	__global const double *coefficients,
	const int n_electrodes,
	const int n_electrodes_rf,
	const double V_RF,
	const double Omega_RF,
	const double m,
	const double q,
	__global const double *waveform_times,
	__global const double *waveform_voltages,
	const int n_waveform_steps,
	const int full_rf,
	const double t,
	const double3 position,
	const double3 center_0,
	const double spacing,
	const double radius
)
{
	double weights[2 * MAX_ELECTRODES];
	interpolate_waveform(waveform_times, waveform_voltages, n_waveform_steps, n_electrodes, t, weights);

	for (int e = 0; e < n_electrodes; e++)
	{
		if (full_rf && e < n_electrodes_rf)
			weights[e] += V_RF * cos(Omega_RF * t);

		weights[n_electrodes + e] = (!full_rf && e < n_electrodes_rf) ? 1.0 : 0.0;
	}

	double3 u;
	int sphere = nearest_sphere(position, center_0, spacing, radius, &u);

	double value[2];
	double grad[2][3];
	double hess[2][6];
	accumulate_expansion(coefficients, n_electrodes, sphere, weights, u, full_rf ? 1 : 2, value, grad, hess);

	// Gradient of the potential energy in J/µm
	double3 grad_U = q * (double3)(grad[0][0], grad[0][1], grad[0][2]) / radius;

	if (!full_rf && n_electrodes_rf > 0 && V_RF != 0.0 && Omega_RF != 0.0)
	{
		double prefactor = pow(q * V_RF, 2) / (4 * m * pow(Omega_RF * LENGTH_SCALE, 2));

		double g[3] = { grad[1][0] / radius, grad[1][1] / radius, grad[1][2] / radius };
		double H[3][3];
		for (int h = 0; h < 6; h++)
		{
			H[hessian_k[h]][hessian_j[h]] = hess[1][h] / (radius * radius);
			H[hessian_j[h]][hessian_k[h]] = hess[1][h] / (radius * radius);
		}

		grad_U.x += 2 * prefactor * (g[0] * H[0][0] + g[1] * H[1][0] + g[2] * H[2][0]);
		grad_U.y += 2 * prefactor * (g[0] * H[0][1] + g[1] * H[1][1] + g[2] * H[2][1]);
		grad_U.z += 2 * prefactor * (g[0] * H[0][2] + g[1] * H[1][2] + g[2] * H[2][2]);
	}

	return -grad_U / (m * LENGTH_SCALE * LENGTH_SCALE);
}

// Initial accelerations of an ensemble of ions
__kernel void ensemble_acceleration
(
	__global const double *coefficients,
	const int n_electrodes,
	const int n_electrodes_rf,
	const double V_RF,
	const double Omega_RF,
	const double m,
	const double q,
	__global const double *waveform_times,
	__global const double *waveform_voltages,
	const int n_waveform_steps,
	const int full_rf,
	const double t,
	const double3 center_0,
	const double spacing,
	const double radius,
	__global const double *positions,
	__global double *accelerations,
	const int n_particles
)
{
	int p = get_global_id(0);

	if (p >= n_particles)
		return;

	double3 position = (double3)(positions[3*p], positions[3*p + 1], positions[3*p + 2]);
	double3 acceleration = ion_acceleration(coefficients, n_electrodes, n_electrodes_rf, V_RF, Omega_RF, m, q, waveform_times, waveform_voltages, n_waveform_steps, full_rf, t, position, center_0, spacing, radius);

	accelerations[3*p] = acceleration.x;
	accelerations[3*p + 1] = acceleration.y;
	accelerations[3*p + 2] = acceleration.z;
}

// Several velocity Verlet steps of an ensemble of ions, from step first_step to first_step + n_steps (exclusive), where
// step i ends at time t_0 + (i + 1) dt; each launch runs a bounded number of steps so that kernels stay short enough
// for GPUs which also drive a display
__kernel void ensemble_verlet_steps
(
	__global const double *coefficients,
	const int n_electrodes,
	const int n_electrodes_rf,
	const double V_RF,
	const double Omega_RF,
	const double m,
	const double q,
	__global const double *waveform_times,
	__global const double *waveform_voltages,
	const int n_waveform_steps,
	const int full_rf,
	const double t_0,
	const double dt,
	const double3 center_0,
	const double spacing,
	const double radius,
	__global double *positions,
	__global double *velocities,
	__global double *accelerations,
	const int n_particles,
	const int first_step,
	const int n_steps
)
{
	int p = get_global_id(0);

	if (p >= n_particles)
		return;

	double3 position = (double3)(positions[3*p], positions[3*p + 1], positions[3*p + 2]);
	double3 velocity = (double3)(velocities[3*p], velocities[3*p + 1], velocities[3*p + 2]);
	double3 acceleration = (double3)(accelerations[3*p], accelerations[3*p + 1], accelerations[3*p + 2]);

	for (int it = first_step; it < first_step + n_steps; it++)
	{
		double t = t_0 + (double)(it + 1) * dt;

		position += velocity * dt + 0.5 * acceleration * dt * dt;

		double3 new_acceleration = ion_acceleration(coefficients, n_electrodes, n_electrodes_rf, V_RF, Omega_RF, m, q, waveform_times, waveform_voltages, n_waveform_steps, full_rf, t, position, center_0, spacing, radius);

		velocity += (acceleration + new_acceleration) * (0.5 * dt);
		acceleration = new_acceleration;
	}

	positions[3*p] = position.x;
	positions[3*p + 1] = position.y;
	positions[3*p + 2] = position.z;
	velocities[3*p] = velocity.x;
	velocities[3*p + 1] = velocity.y;
	velocities[3*p + 2] = velocity.z;
	accelerations[3*p] = acceleration.x;
	accelerations[3*p + 1] = acceleration.y;
	accelerations[3*p + 2] = acceleration.z;
}
