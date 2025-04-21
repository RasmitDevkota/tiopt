#include "fem_wrapper.h"

#include <stdio.h>

#include <sparselizard.h>

using namespace sl;

extern "C"
{
	// %TODO - find a way to directly recover the electric potential and electric field data
	void sparselizard_wrapper(char *mesh_filename, char *output_filename)
	{
		// printf("Sparselizard wrapper is not implemented!\n");

		int vol = 1, sur = 2;              // Disk volume and boundary as set in mesh file
		mesh mymesh(mesh_filename);

		field v("h1");                     // Nodal shape functions for the electric potential
		v.setorder(vol, 2);                // Interpolation order 2 on the whole domain
		v.setconstraint(sur, 1);           // Force 1 V on the disk boundary

		formulation elec;                  // Electrostatics with 0 charge density
		elec += integral(vol, -8.854e-12 * grad(dof(v)) * grad(tf(v)));

		elec.generate();                   // Generate, solve and save solution to field v
		vec solV = solve(elec.A(), elec.b());
		v.setdata(vol, solV);

		v.write(vol, output_filename, 2); // Write the electric field to ParaView format
	}
}

