#include "test_utils.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"
#include "electrodynamics.h"

// Report whether a test passed, i.e. whether its error is within tolerance (NaN errors fail); returns 0 if it passed
// and 1 if it failed
int check_error
(
	const char *test_name,
	double error,
	double tolerance
)
{
	int result = (error <= tolerance) ? 0 : 1;

	printf("  %s %s (error %.3e, tolerance %.3e)\n", (result == 0) ? "PASSED" : "FAILED", test_name, error, tolerance);

	return result;
}

// Fill the spherical harmonics expansions of an electrode (whose Vlm must already be allocated) by sampling a function
// of position on every sphere, so that tests can construct potentials with known forms
void build_expansion_from_function
(
	struct Electrode *electrode,
	double (*f)(double position[3], void *f_data),
	void *f_data
)
{
	for (int sx = 0; sx < NSPH_X; sx++)
		for (int sy = 0; sy < NSPH_Y; sy++)
			for (int sz = 0; sz < NSPH_Z; sz++)
			{
				double grid[NLAT][NLON];

				for (int i = 0; i < NLAT; i++)
				{
					double phi = (double)i/NLAT * PI;

					for (int j = 0; j < NLON; j++)
					{
						double theta = (double)j/NLON * 2 * PI;

						double position[3] =
						{
							SPH_CENTER_X(sx) + SPH_R * sin(phi) * cos(theta),
							SPH_CENTER_Y(sy) + SPH_R * sin(phi) * sin(theta),
							SPH_CENTER_Z(sz) + SPH_R * cos(phi)
						};

						grid[i][j] = f(position, f_data);
					}
				}

				expand_spherical_harmonics(grid, VLM_SLICE(electrode->Vlm, sx, sy, sz));
			}
}
