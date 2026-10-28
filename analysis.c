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
