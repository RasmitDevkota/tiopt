#pragma once

#include "data_structures.h"

double heating_rate
(
	const double q,
	const double m,
	const double Omega_RF,
	const double omega,
	const double dE2_dz,
	const double S_V_N,
	const double V_0
);

double heating_rate_rf_noise
(
	struct Trap *trap,
	const double m,
	const double q,
	double position[3],
	const double omega,
	double axis[3],
	const double S_V_N
);

double heating_rate_field_noise
(
	const double m,
	const double q,
	const double omega,
	const double S_E
);

double heating_rate_voltage_noise
(
	struct Trap *trap,
	const double m,
	const double q,
	double position[3],
	const double omega,
	double axis[3],
	const double S_V
);

double motional_quanta
(
	const double m,
	const double omega,
	double axis[3],
	double position[3],
	double velocity[3],
	double equilibrium[3]
);

double compute_trap_depth
(
	struct Trap *trap,
	const double m,
	const double q,
	double rf_null[3]
);
