#include "data_structures.h"

void velocity_verlet(
	struct Trap *trap,
	const double m,
	const double q,
	double (*pos)[3],
	double (*vel)[3],
	double (*acc)[3],
	const int n_it,
	const double dt
);

void velocity_verlet_update(
	struct Trap *trap,
	const double m,
	const double q,
	double (*pos)[3],
	double (*vel)[3],
	double (*acc)[3],
	const double dt
);

