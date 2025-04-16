#include "electrostatics.h"

// #include <sparselizard.h>

#include "data_structures.h"

void solve_trap_electrostatics(struct Trap *trap)
{
	for (int e = 0; e < trap->n_electrodes; e++)
	{
		struct Electrode electrode = (*trap->electrodes)[e];
		char* electrode_mesh_filename = electrode.electrode_mesh_filename;

		char* output_filename;
		// sparselizard_wrapper(electrode_mesh_filename, output_filename);

		// %TODO - read output file and populate electrode->V
	}
}

extern "C"
{
	// %TODO - find a way to directly recover the electric potential and electric field data
	void sparselizard_wrapper(char *mesh_filename, char *output_filename)
	{
		// int vol = 1, sur = 2;              // Disk volume and boundary as set in mesh file
		// sl::mesh mymesh(mesh_filename);
		//
		// sl::field v("h1");                     // Nodal shape functions for the electric potential
		// v.setorder(vol, 2);                // Interpolation order 2 on the whole domain
		// v.setconstraint(sur, 1);           // Force 1 V on the disk boundary
		//
		// sl::formulation elec;                  // Electrostatics with 0 charge density
		// elec += sl::integral(vol, -8.854e-12 * grad(dof(v)) * grad(tf(v)));
		//
		// elec.generate();                   // Generate, solve and save solution to field v
		// vec solV = sl::solve(elec.A(), elec.b());
		// v.setdata(vol, solV);
		//
		// v.write(vol, output_filename, 2); // Write the electric field to ParaView format
	}
}
