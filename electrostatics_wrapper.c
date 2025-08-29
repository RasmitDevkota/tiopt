#include "electrostatics_wrapper.h"

#include <stdlib.h>

#include "defs.h"
#include "data_structures.h"
#include "electrostatics.h"

void solve_trap_electrostatics(struct Trap *trap)
{
	for (int e = 0; e < trap->n_electrodes; e++)
	{
		printf("Solving electrostatics for electrode %d/%d\n", e+1, trap->n_electrodes);

		struct Electrode electrode = (*trap->electrodes)[e];

		char output_filename[64];
		sprintf(
			output_filename,
			"sl_outputs/electrode.vtk"
		);

		sparselizard_wrapper(electrode.electrode_mesh_filename, output_filename, electrode.Vlm);
	}
}

