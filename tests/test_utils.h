#pragma once

#include "data_structures.h"

int check_error
(
	const char *test_name,
	double error,
	double tolerance
);

void build_expansion_from_function
(
	struct Electrode *electrode,
	double (*f)(double position[3], void *f_data),
	void *f_data
);
