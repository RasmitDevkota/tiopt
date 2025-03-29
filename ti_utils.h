void grad(
	const int f_len,
	double (*f)[f_len],
	double (*grad_f)[f_len],
	const double dx
);

void interpolate_1d(
	const int f_len,
	double (*f)[f_len],
	double x_rel,
	double *f_x_rel,
	const double dx
);

