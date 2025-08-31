#include "electrodynamics_external.hpp"

#include <stdio.h>
#include <math.h>

#include <sparselizard.h>

extern "C"
{
	void sparselizard_wrapper(
		char *mesh_filename,
		char *output_filename,
		double (*Vlm)[(LMAX+1)*(LMAX+1)*2]
	) {
		// Physical group tags as set in mesh file
		int electrode_surface = 3;
		int electrode_volume = 4;
		int sim_volume = 5;

		// Option 1: Load existing mesh from file
		mesh electrode_mesh(mesh_filename);

		// // Option 2: Construct new mesh using Sparselizard library
		// shape shape_electrode_surface("quadrangle", electrode_surface, {-10,-10,0, 10,-10,0, 10,10,0, -10,10,0}, {10,10,10,10});
		// shape shape_electrode_volume = shape_electrode_surface.extrude(electrode_volume, 10.0, 2);
		//
		// int sim_surface = 6;
		// double z = 5.0;
		// shape shape_sim_surface("quadrangle", sim_surface, {-50,-50,z, 50,-50,z, 50,50,z, -50,50,z}, {50,50,50,50});
		// shape shape_sim_volume = shape_sim_surface.extrude(sim_volume, 50.0, 10);
		//
		// mesh electrode_mesh({ shape_electrode_volume, shape_sim_volume });

		// Option 2b:
		// shape shape_electrode_surface("quadrangle", electrode_surface, {-10,-10,0, 10,-10,0, 10,10,0, -10,10,0}, {10,10,10,10});
		// shape shape_sim_volume = shape_electrode_surface.extrude(sim_volume, 50.0, 50);
		//
		// mesh electrode_mesh({ shape_electrode_surface, shape_sim_volume });

		for (int d=0; d <= 3; d++)
		{
			std::vector<int> physregs = electrode_mesh.getphysicalregionnumbers(d);
			for (long unsigned int r=0; r < physregs.size(); r++)
			{
				printf("%d -> %d\n", d, physregs[r]);
			}
		}

		printf("Constructing problem...\n");

		// Nodal shape functions for the electric potential
		field V("h1", {1});
		field V_lin = V.harmonic(1);

		// Interpolation order 2 on the whole domain
		V.setorder(sim_volume, 1);

		// Force 1 V on the electrode volume
		V.setconstraint(electrode_volume, 1);

		// Define the speed of light [m/s] and the fundamental frequency [Hz].
		// The effective length of an electrode is on the order of 10-100 um. This corresponds to around a fundamental frequency on the order of THz:
		double c = 3E8, f_0 = 1E12;
		setfundamentalfrequency(f_0);

		formulation elec;

		// Option 1: Formulate electrostatics problem with 0 charge density
		// elec += sl::integral(sim_volume, -55.26349406 * sl::grad(sl::dof(V)) * sl::grad(sl::tf(V)));
		// Option 2: Formulate electrodynamics problem (Lorenz gauge) with 0 charge density
		elec += integral(sim_volume, -grad(dof(V))*grad(tf(V)) - 1/(c*c)*dtdt(dof(V))*tf(V));

		elec.generate();

		printf("Solving electrodynamics problem...\n");

		// Solve and save solution to field V
		vec V_solution = sl::solve(elec.A(), elec.b());
		V.setdata(sim_volume, V_solution);

		printf("Writing electric potential to ParaView format...\n");

		// Option 1: Write the static electric potential to GMSH post-processing format
		// V.write(sim_volume, "sl_outputs/V.pos", 2);
		// Option 2: Write one timestep of the electric potential time series to GMSH post-processing format
		V.write(sim_volume, "sl_outputs/V.pos", 2, 1);

		printf("Sampling field...\n");

		// Sample field based on method by Driscoll and Healy (1994)
		double grid[NLAT][NLON];
		sparselizard_sample_dh1(&V, sim_volume, grid);

		printf("Expanding in spherical harmonics basis...\n");

		// compute the spherical harmonics expansion of the potential energy contribution
		expand_spherical_harmonics_cpp(grid, Vlm);
	}

	void sparselizard_sample_dh1(
		void *_f,
		int sample_volume,
		double grid[NLAT][NLON]
	) {
		field *f = (field *)_f;

		// std::vector<double> sphere_center_offset = { 0.0, 0.0, SPH_Z_MIN };
		std::vector<double> sphere_center_offset = { 0.0, 0.0, 25.0 };

		for (int i = 0; i < NLAT; i++) {
			double phi = (double)i/NLAT * PI;

			// Precompute trigonometric quantities to reduce cost
			// double R_sin_phi = 0.1 * SPH_R * sin(phi);
			// double R_cos_phi = 0.1 * SPH_R * cos(phi);
			double R_sin_phi = 0.1 * 10.0 * sin(phi);
			double R_cos_phi = 0.1 * 10.0 * cos(phi);

			for (int j = 0; j < NLON; j++) {
				double theta = (double)j/NLON * PI;

				std::vector<double> sample_coord_rel = { R_sin_phi * cos(theta), R_sin_phi * sin(theta), R_cos_phi };
				std::vector<double> sample_coord = { 0.0, 0.0, 0.0 };
				for (int d = 0; d < 3; d++)
				{
					sample_coord[d] = sphere_center_offset[d] + sample_coord_rel[d];
				}

				std::vector<double> f_interp = { 0.0 };
				std::vector<bool> is_found;
				f->interpolate(sample_volume, sample_coord, f_interp, is_found);
				if (!is_found[0])
				{
					// printf("ERROR: Failed to interpolate value of V(%f,%f,%f)\n", SPH_R, phi, theta);
					printf("ERROR: Failed to interpolate value of V(%f,%f,%f)\n", 25.0, phi, theta);
				}
				else
				{
					// printf("V(%f,%f,%f)=%f\n", SPH_R, phi, theta, f_interp[0]);
					printf("V(%f,%f,%f)=%f\n", 25.0, phi, theta, f_interp[0]);
				}
				grid[i][j] = f_interp[0];
			}
		}
	}

	void expand_spherical_harmonics_cpp(
		double grid[NLAT][NLON],
		double (*alm)[(LMAX+1)*(LMAX+1)*2]
	) {
		// Define new variables so that Fortran recognizes the types
		int nlat = NLAT;
		int nlon = NLON;
		int lmax = LMAX;

		// Call the Fortran function
		// compute_shcoeffs_cmplx((double*) grid, &nlat, &nlon, &lmax, *alm);
		compute_shcoeffs_real((double*) grid, &nlat, &nlon, &lmax, *alm);

		// Print first few coefficients
		for (int l = 0; l <= LMAX; l++) {
			for (int m = -l; m <= l; m++) {
				double alm_real = 0.0;
				double alm_imag = 0.0;

				if (m < 0)
				{
					// Y_l,-m = (-1)^m Y_l,m*
					alm_real = pow(-1, m) * (*alm)[2 * (l * (LMAX + 1) + m)];
					alm_imag = pow(-1, m) * -1 * (*alm)[2 * (l * (LMAX + 1) + m) + 1];
				}
				else
				{
					alm_real = (*alm)[2 * (l * (LMAX + 1) + m)];
					alm_imag = (*alm)[2 * (l * (LMAX + 1) + m) + 1];
				}

				printf("a_lm[%d,+%d] = (%f, %f)\n", l, m, alm_real, alm_imag);
			}
		}
	}
}

