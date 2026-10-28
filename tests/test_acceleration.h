#pragma once

#include "data_structures.h"

int test_all_acceleration
(
	double atol
);

int test_thread_consistency
(
	struct Trap *trap,
	double atol
);

int test_simulate_ensemble
(
	struct Trap *trap,
	double atol
);

int test_opencl
(
	struct Trap *trap,
	double atol
);
