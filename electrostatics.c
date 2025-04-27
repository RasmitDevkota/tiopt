#include "electrostatics.h"

#include "data_structures.h"
#include "fem_wrapper.h"

void solve_trap_electrostatics(struct Trap *trap)
{
	for (int e = 0; e < trap->n_electrodes; e++)
	{
		printf("Solving electrostatics for electrode %d/%d\n", e+1, trap->n_electrodes);

		struct Electrode electrode = (*trap->electrodes)[e];
		char *electrode_mesh_filename = electrode.electrode_mesh_filename;

		char *output_filename;
		sparselizard_wrapper(electrode_mesh_filename, output_filename);

		// %TODO - read output file and populate electrode->V
	}
}

