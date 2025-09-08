#include "analysis.h"

#include <math.h>

double heating_rate(

)
{
	double dE2dz = 0.0;

	return (pow(q, 4) / (16 * pow(m, 3) * pow(Omega_RF, 4) * HBAR * omega_z)) * pow(dE2dz, 2) * S_V_N / V_0^2;
}


