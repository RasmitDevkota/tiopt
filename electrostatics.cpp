#include "sparselizard.h"


using namespace sl;

extern "C" { 
void electrostatic(char* meshfile, char* outputfile) {
	int vol = 1, sur = 2;              // Disk volume and boundary as set in ’disk.geo’
	mesh mymesh(meshfile);

	field v("h1");                     // Nodal shape functions for the electric potential
	v.setorder(vol, 2);                // Interpolation order 2 on the whole domain
	v.setconstraint(sur, 1);           // Force 1 V on the disk boundary

	formulation elec;                  // Electrostatics with 0 charge density
	elec += integral(vol, -8.854e-12 * grad(dof(v)) * grad(tf(v)));

	elec.generate();                      // Generate, solve and save solution to field v
	vec solV = solve(elec.A(), elec.b());
	v.setdata(vol, solV);

	v.write(vol, outputfile, 2); // Write the electric field to ParaView format
}
}
