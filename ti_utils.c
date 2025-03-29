void grad(
    const int f_len,
	double (*f)[f_len],
	double (*grad_f)[f_len],
	const double dx
) {
	for (int k = 0; k < f_len; k++)
	{
		// compute the next index
		const int idx_next = (k+1) % f_len;

		// compute the previous index by a safer method in case of negative overflow
		const int idx_prev = ((k-1) % f_len + f_len) % f_len;

	    (*grad_f)[k] = (2 * dx) * ((*f)[idx_next] - (*f)[idx_prev]);
	}
}

