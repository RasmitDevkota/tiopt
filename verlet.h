#pragma once

#include "data_structures.h"

void compute_acceleration(
	struct Trap *trap,
	const double m,
	const double q,
	double *voltages,
	const double t,
	const int full_rf,
	double (*pos)[3],
	double (*acc)[3]
);

void velocity_verlet(
	struct Trap *trap,
	const double m,
	const double q,
	struct TransportWaveform *waveform,
	const int full_rf,
	double (*pos)[3],
	double (*vel)[3],
	double (*acc)[3],
	const int n_it,
	const double dt,
	double (*positions)[3],
	double (*velocities)[3]
);

void velocity_verlet_update(
	struct Trap *trap,
	const double m,
	const double q,
	double *voltages,
	const double t,
	const int full_rf,
	double (*pos)[3],
	double (*vel)[3],
	double (*acc)[3],
	const double dt
);

void simulate_ensemble(
	struct Trap *trap,
	const double m,
	const double q,
	struct TransportWaveform *waveform,
	const int full_rf,
	int n_particles,
	double (*positions)[3],
	double (*velocities)[3],
	const int n_it,
	const double dt
);
