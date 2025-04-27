#include "fem_wrapper.h"

#include <stdio.h>
#include <math.h>

#include <sparselizard.h>

using namespace sl;

#define PI 3.14159265358979

extern "C"
{
	// %TODO - find a way to directly recover the electric potential and electric field data
	void sparselizard_wrapper(char *mesh_filename, char *output_filename)
	{
		// Trapping region volume and electrode electrode_surfaceface as set in mesh file
		int sim_volume = 1;
		int electrode_surface = 1;
		mesh electrode_mesh(mesh_filename);

		// Nodal shape functions for the electric potential
		field V("h1");
		// Interpolation order 2 on the whole domain
		V.setorder(sim_volume, 2);
		// Force 1 V on the electrode electrode_surfaceface
		V.setconstraint(electrode_surface, 1);

		// Electrostatics with 0 charge density
		formulation elec;
		elec += integral(sim_volume, -8.854e-12 * grad(dof(V)) * grad(tf(V)));

		// Generate, solve and save solution to field v
		elec.generate();
		vec V_solution = solve(elec.A(), elec.b());
		V.setdata(sim_volume, V_solution);

		printf("Solved electrostatics!\n", mesh_filename);

		sparselizard_sample_dh1(&V, sim_volume, 40, 40);
		// sparselizard_sample_dh1(&V_solution, sim_volume, 40, 40);
	}

	void sparselizard_sample_dh1(void *_f, int sample_volume, int nlat, int nlon)
	{
		field *f = (field *)f;

		double grid[nlat][nlon];

		for (int i = 0; i < nlat; i++) {
			double phi = i/nlat * PI;

			// Precompute trigonometric quantities to reduce cost
			double sin_phi = sin(phi);
			double cos_phi = cos(phi);

			for (int j = 0; j < nlon; j++) {
				double theta = j/nlon * PI;
				auto vector = f->interpolate(sample_volume, { sin_phi * cos(theta), sin_phi * sin(theta), cos_phi });
				auto size = vector.size();
				std::cout << size << std::endl;
				grid[i][j] = vector[0];
			}
		}
	}
}

