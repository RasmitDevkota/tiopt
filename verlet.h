void velocity_verlet(
	const double m,
	const double q,
	const int V_len,
	double (*V)[V_len],
	double (*pos)[3],
	double (*vel)[3],
	double (*acc)[3],
	const int n_it,
	const double dt
);

void velocity_verlet_update(
	const double m,
	const double q,
	const int V_len,
	double (*V)[V_len],
	double (*grad_V)[V_len],
	double (*pos)[3],
	double (*vel)[3],
	double (*acc)[3],
	const double dt
);

