#pragma once

#include "data_structures.h"

int generate_ion_from_preset
(
	char *element,
	struct IonParameters *ion
);

double ion_mass
(
	struct IonParameters *ion
);

double ion_charge
(
	struct IonParameters *ion
);
