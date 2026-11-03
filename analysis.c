#include "analysis.h"

#include <stdlib.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"
#include "electrodynamics.h"

// Heating rate (quanta/s) of a mode of frequency omega (rad/s) due to noise in the amplitude V_0 (V) of the RF drive,
// which modulates the pseudopotential; dE2_dz is the gradient of the squared RF field amplitude along the mode (V^2/m^3)
// and S_V_N is the power spectral density of the amplitude noise at omega (V^2/Hz)
double heating_rate
(
	const double q,
	const double m,
	const double Omega_RF,
	const double omega,
	const double dE2_dz,
	const double S_V_N,
	const double V_0
)
{
	return (pow(q, 4) / (16 * pow(m, 3) * pow(Omega_RF, 4) * HBAR * omega)) * pow(dE2_dz, 2) * S_V_N / pow(V_0, 2);
}

// Heating rate (quanta/s) due to RF amplitude noise for a mode along axis at a position in a trap, which vanishes at
// the RF null
double heating_rate_rf_noise
(
	struct Trap *trap,
	const double m,
	const double q,
	double position[3],
	const double omega,
	double axis[3],
	const double S_V_N
)
{
	if (trap->n_electrodes_rf == 0 || trap->V_RF == 0.0)
		return 0.0;

	double *rf_weights = calloc(trap->n_electrodes, sizeof(double));
	for (int e = 0; e < trap->n_electrodes_rf; e++)
		rf_weights[e] = 1.0;

	double g[3];
	double H[3][3];
	evaluate_trap_potential(trap, rf_weights, position, NULL, g, H);

	free(rf_weights);

	double axis_norm = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);

	// |E|^2 = V_RF^2 |grad(phi_RF)|^2, so its derivative along the axis is 2 V_RF^2 grad(phi_RF) . H a (converted from
	// µm to m)
	double dE2_dz = 0.0;
	for (int k = 0; k < 3; k++)
		for (int j = 0; j < 3; j++)
			dE2_dz += 2 * pow(trap->V_RF, 2) * g[k] * H[k][j] * axis[j] / axis_norm;

	dE2_dz /= pow(LENGTH_SCALE, 3);

	return heating_rate(q, m, trap->Omega_RF, omega, dE2_dz, S_V_N, trap->V_RF);
}

// Heating rate (quanta/s) of a mode of frequency omega (rad/s) due to electric field noise along it with power spectral
// density S_E (V^2/m^2/Hz) (Q. A. Turchette et al., Phys. Rev. A 61, 063418 (2000))
double heating_rate_field_noise
(
	const double m,
	const double q,
	const double omega,
	const double S_E
)
{
	return q*q * S_E / (4 * m * HBAR * omega);
}

// Heating rate (quanta/s) due to uncorrelated voltage noise with power spectral density S_V (V^2/Hz) on every DC
// electrode, for a mode along axis at a position in a trap
double heating_rate_voltage_noise
(
	struct Trap *trap,
	const double m,
	const double q,
	double position[3],
	const double omega,
	double axis[3],
	const double S_V
)
{
	double axis_norm = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);

	double S_E = 0.0;
	for (int e = trap->n_electrodes_rf; e < trap->n_electrodes; e++)
	{
		double grad_phi[3];
		evaluate_electrode_gradient(&(*trap->electrodes)[e], position, grad_phi);

		double E_axis = (grad_phi[0] * axis[0] + grad_phi[1] * axis[1] + grad_phi[2] * axis[2]) / axis_norm / LENGTH_SCALE;

		S_E += E_axis*E_axis * S_V;
	}

	return heating_rate_field_noise(m, q, omega, S_E);
}

// Number of motional quanta in the classical oscillation of an ion along a mode, from its position (µm) and velocity
// (µm/s) relative to its equilibrium position (µm)
double motional_quanta
(
	const double m,
	const double omega,
	double axis[3],
	double position[3],
	double velocity[3],
	double equilibrium[3]
)
{
	double axis_norm = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);

	double x = 0.0;
	double v = 0.0;
	for (int k = 0; k < 3; k++)
	{
		x += (position[k] - equilibrium[k]) * axis[k] / axis_norm * LENGTH_SCALE;
		v += velocity[k] * axis[k] / axis_norm * LENGTH_SCALE;
	}

	double E = 0.5 * m * v*v + 0.5 * m * omega*omega * x*x;

	return E / (HBAR * omega);
}

// Depth (eV) of the pseudopotential well of an RF null, i.e. the height of the barrier the ion must cross to escape
// along the vertical through the null, which is the lowest escape path of surface traps. The potentials of the RF
// electrodes are evaluated analytically (in the gapless plane approximation), since the barrier usually lies above the
// region covered by the spherical harmonics expansions
double compute_trap_depth
(
	struct Trap *trap,
	const double m,
	const double q,
	double rf_null[3]
)
{
	if (trap->n_electrodes_rf == 0 || trap->V_RF == 0.0 || trap->Omega_RF == 0.0)
		return 0.0;

	// Pseudopotential q^2 V_RF^2 |grad(phi_RF)|^2 / (4 m Omega_RF^2), with the gradient converted from 1/µm to 1/m
	double prefactor = pow(q * trap->V_RF, 2) / (4 * m * pow(trap->Omega_RF * LENGTH_SCALE, 2));

	double h = 1E-3;
	int n_samples = 400;
	double z_max = rf_null[2] + 5 * ION_HEIGHT;

	double U_null = 0.0;
	double U_barrier = 0.0;
	for (int i = 0; i <= n_samples; i++)
	{
		double position[3] = { rf_null[0], rf_null[1], rf_null[2] + (z_max - rf_null[2]) * i / n_samples };

		// Gradient of the RF potential by central differences
		double grad_phi[3] = { 0.0, 0.0, 0.0 };
		for (int e = 0; e < trap->n_electrodes_rf; e++)
			for (int k = 0; k < 3; k++)
			{
				double p_plus[3] = { position[0], position[1], position[2] };
				double p_minus[3] = { position[0], position[1], position[2] };
				p_plus[k] += h;
				p_minus[k] -= h;

				double phi_plus;
				double phi_minus;
				evaluate_electrode_potential_analytical(&(*trap->electrodes)[e], (*trap->electrode_positions)[e], p_plus, &phi_plus);
				evaluate_electrode_potential_analytical(&(*trap->electrodes)[e], (*trap->electrode_positions)[e], p_minus, &phi_minus);

				grad_phi[k] += (phi_plus - phi_minus) / (2 * h);
			}

		double U = prefactor * (grad_phi[0]*grad_phi[0] + grad_phi[1]*grad_phi[1] + grad_phi[2]*grad_phi[2]);

		if (i == 0)
			U_null = U;

		U_barrier = MAX(U_barrier, U);
	}

	return (U_barrier - U_null) / E_CHARGE;
}

// Follow the RF null along a path through waypoints (e.g. from one arm of a junction through its center and out along
// another), finding the minimum of the pseudopotential in the plane across the path at n_samples points spaced evenly
// along it; the points (µm) and the pseudopotential there (eV) are returned in positions and energies (each of length
// n_samples, or NULL, with NAN where no minimum is found), and the height of the highest barrier along the path above
// the lower of its two ends (eV) is returned. The trap's electrodynamics must be solved, and the path must lie within
// the spherical harmonics expansions
double compute_rf_null_path
(
	struct Trap *trap,
	const double m,
	const double q,
	int n_waypoints,
	double (*waypoints)[3],
	int n_samples,
	double (*positions)[3],
	double *energies
)
{
	if (n_waypoints < 2 || n_samples < 2)
		return 0.0;

	// Distance along the path to each waypoint
	double *distances = malloc(n_waypoints * sizeof(double));
	distances[0] = 0.0;
	for (int w = 1; w < n_waypoints; w++)
	{
		double d_squared = 0.0;
		for (int k = 0; k < 3; k++)
			d_squared += pow(waypoints[w][k] - waypoints[w - 1][k], 2);

		distances[w] = distances[w - 1] + sqrt(d_squared);
	}

	double (*points)[3] = malloc(n_samples * sizeof(double[3]));
	int *found = calloc(n_samples, sizeof(int));

	// Walk along the path from its start and then back from its end, starting the search at each point from the
	// height of the null at the previous point (which follows the null as it rises or falls near a junction) or else
	// from the height of the path, so that every point can be reached from whichever side leads to it
	for (int pass = 0; pass < 2; pass++)
	{
		int has_height = 0;
		double height = 0.0;

		for (int j = 0; j < n_samples; j++)
		{
			int i = (pass == 0) ? j : n_samples - 1 - j;

			if (found[i])
			{
				height = points[i][2];
				has_height = 1;
				continue;
			}

			double s = distances[n_waypoints - 1] * i / (n_samples - 1);

			int leg = 0;
			while (leg < n_waypoints - 2 && s > distances[leg + 1])
				leg++;

			double leg_length = distances[leg + 1] - distances[leg];
			double fraction = (leg_length > 0.0) ? (s - distances[leg]) / leg_length : 0.0;

			double point[3];
			double direction[3];
			for (int k = 0; k < 3; k++)
			{
				point[k] = waypoints[leg][k] + fraction * (waypoints[leg + 1][k] - waypoints[leg][k]);
				direction[k] = waypoints[leg + 1][k] - waypoints[leg][k];
			}

			double path_height = point[2];
			if (has_height)
				point[2] = height;

			int status = find_rf_null_across(trap, &point, direction, 200);
			if (status != 0 && has_height)
			{
				for (int k = 0; k < 3; k++)
					point[k] = waypoints[leg][k] + fraction * (waypoints[leg + 1][k] - waypoints[leg][k]);

				point[2] = path_height;
				status = find_rf_null_across(trap, &point, direction, 200);
			}

			if (status == 0)
			{
				for (int k = 0; k < 3; k++)
					points[i][k] = point[k];

				found[i] = 1;
				height = point[2];
				has_height = 1;
			}
		}
	}

	// Points where the null cannot be found are left out
	double U_first = NAN;
	double U_last = NAN;
	double U_highest = -HUGE_VAL;
	for (int i = 0; i < n_samples; i++)
	{
		double U = NAN;
		if (found[i])
		{
			evaluate_pseudopotential(trap, m, q, points[i], &U, NULL, NULL);
			U /= E_CHARGE;

			U_first = isnan(U_first) ? U : U_first;
			U_last = U;
			U_highest = MAX(U_highest, U);
		}

		if (positions != NULL)
			for (int k = 0; k < 3; k++)
				positions[i][k] = found[i] ? points[i][k] : NAN;

		if (energies != NULL)
			energies[i] = U;
	}

	free(found);
	free(points);
	free(distances);

	return isnan(U_first) ? NAN : U_highest - MIN(U_first, U_last);
}
