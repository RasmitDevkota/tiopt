#include "ions.h"

#include <stdio.h>
#include <string.h>

#include "defs.h"
#include "data_structures.h"

// Fill in the parameters of a commonly trapped ion species (wavelengths are in vacuum, and linewidths are the inverse
// lifetimes of the P1/2 and D5/2 states); returns 1 if the species is unknown
int generate_ion_from_preset
(
	char *element,
	struct IonParameters *ion
)
{
	if (strcmp(element, "40Ca+") == 0)
	{
		ion->element = "40Ca+";
		ion->atomic_mass = 39.962591;
		ion->charge = 1.0;
		ion->omega_SP = 2 * PI * C_LIGHT / 396.959E-9;
		ion->omega_SD = 2 * PI * C_LIGHT / 729.347E-9;
		ion->gamma_SP = 1.0 / 7.098E-9;
		ion->gamma_SD = 1.0 / 1.168;
	}
	else if (strcmp(element, "88Sr+") == 0)
	{
		ion->element = "88Sr+";
		ion->atomic_mass = 87.905612;
		ion->charge = 1.0;
		ion->omega_SP = 2 * PI * C_LIGHT / 421.671E-9;
		ion->omega_SD = 2 * PI * C_LIGHT / 674.025E-9;
		ion->gamma_SP = 1.0 / 7.39E-9;
		ion->gamma_SD = 1.0 / 0.390;
	}
	else if (strcmp(element, "171Yb+") == 0)
	{
		ion->element = "171Yb+";
		ion->atomic_mass = 170.936323;
		ion->charge = 1.0;
		ion->omega_SP = 2 * PI * C_LIGHT / 369.525E-9;
		ion->omega_SD = 2 * PI * C_LIGHT / 411.042E-9;
		ion->gamma_SP = 1.0 / 8.12E-9;
		ion->gamma_SD = 1.0 / 7.2E-3;
	}
	else if (strcmp(element, "9Be+") == 0)
	{
		// 9Be+ has no metastable D states, so its hyperfine qubit takes the place of the S-D transition
		ion->element = "9Be+";
		ion->atomic_mass = 9.012183;
		ion->charge = 1.0;
		ion->omega_SP = 2 * PI * C_LIGHT / 313.132E-9;
		ion->omega_SD = 2 * PI * 1.250E9;
		ion->gamma_SP = 1.0 / 8.2E-9;
		ion->gamma_SD = 0.0;
	}
	else
	{
		printf("ERROR: Unknown ion species '%s'\n", element);
		return 1;
	}

	return 0;
}

// Mass of an ion (kg)
double ion_mass
(
	struct IonParameters *ion
)
{
	return ion->atomic_mass * AMU;
}

// Charge of an ion (C)
double ion_charge
(
	struct IonParameters *ion
)
{
	return ion->charge * E_CHARGE;
}
