#include "trap_geometry.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <gmshc.h>

#include "defs.h"

// generate the initial mesh object needed from the trap electrodes
void generate_mesh(
	struct Trap *trap
) {
	const double lc = 1E-1;
	const double electrode_thickness = 10E-1;
	for (int e = 0; e < trap->n_electrodes; e++)
	{
		printf("Generating mesh for electrode %d/%d\n", e+1, trap->n_electrodes);

		int ierr;

		int argc = 0;
		char** argv = NULL;
		gmshInitialize(argc, argv, 1, 0, &ierr);

		// Format unique electrode_id for model and mesh file
		char electrode_id[15];
		sprintf(
			electrode_id,
			"e_%2s-%6d",
			e < trap->n_electrodes_rf ? "rf" : "dc",
			e
		);

		// Replace whitespace padding (default behavior of sprintf) with '0' padding
		for (int p = 0; electrode_id[p] != '\0'; p++)
		{
			if (electrode_id[p] == ' ')
				electrode_id[p] = '0';
		}
		printf("%s\n", electrode_id);

		gmshModelAdd(electrode_id, &ierr);

		// 0D - Electrode vertices
		int physgroup_vertices[(*trap->electrodes)[e].n_vertices];
		for (int v = 0; v < (*trap->electrodes)[e].n_vertices; v++)
		{
			int vertex_v = gmshModelOccAddPoint(
				(*trap->electrodes)[e].vertices[v][0], (*trap->electrodes)[e].vertices[v][1], (*trap->electrodes)[e].vertices[v][2],
				lc,
				v+1,
				&ierr
			);

			physgroup_vertices[v] = vertex_v;
		}
		printf("gmshModelOccAddPoint Result: %d\n", ierr);

		// 1D - Electrode edges
		int physgroup_edges[(*trap->electrodes)[e].n_edges];
		for (int d = 0; d < (*trap->electrodes)[e].n_edges; d++)
		{
			int v_i = (*trap->electrodes)[e].edges[d][0];
			int v_f = (*trap->electrodes)[e].edges[d][1];

			int edge_e = gmshModelOccAddLine(
				physgroup_vertices[v_i], physgroup_vertices[v_f],
				d+1,
				&ierr
			);

			physgroup_edges[d] = edge_e;
		}
		printf("gmshModelOccAddLine Result: %d\n", ierr);

		// "2D" - Electrode curve loops (doesn't conflict with 2D tags though)
		int curveloop_definition[(*trap->electrodes)[e].n_edges];
		for (int d = 0; d < (*trap->electrodes)[e].n_edges; d++)
		{
			curveloop_definition[d] = physgroup_edges[d];
		}

		int curveloop = gmshModelOccAddCurveLoop(
			curveloop_definition,
			sizeof(curveloop_definition)/sizeof(curveloop_definition[0]),
			2*e+1,
			&ierr
		);
		printf("gmshModelOccAddCurveLoop Result: %d\n", ierr);

		// 2D - Electrode surfaces
		const int surface_definition[1] = { curveloop };

		int surface = gmshModelOccAddPlaneSurface(
			surface_definition,
			sizeof(surface_definition)/sizeof(surface_definition[0]),
			2*e+1,
			&ierr
		);
		printf("gmshModelOccAddSurface Result: %d\n", ierr);

		int physgroup_surfaces[6] = { surface, 0, 0, 0, 0, 0 };

		// 3D - Spheres
		int sphere = gmshModelOccAddSphere(
			0.0, 0.0, SPH_Z_MIN,
			SPH_R,
			2*e+1,
			-0.5*M_PI, 0.5*M_PI, 2*M_PI,
			&ierr
		);
		printf("gmshModelOccAddSphere Result: %d\n", ierr);

		int physgroup_volumes_spheres[1] = { sphere };

		// 3D - Electrode extrusions
		const int extrusion_definition[2] = { 2, surface };
		size_t n_out_dim_tags;
		int *out_dim_tags;

		gmshModelOccExtrude(
			extrusion_definition,
			2,
			0.0, 0.0, electrode_thickness,
			&out_dim_tags, &n_out_dim_tags,
			NULL, 0,
			NULL, 0,
			0,
			&ierr
		);
		printf("gmshModelOccExtrude Result: %d\n", ierr);

		int physgroup_volumes_extrusions[1] = { 0 };

		if (n_out_dim_tags % 2 != 0)
		{
			printf("ERROR: gmshModelOccExtrude did not return an even number of elements a.k.a. (dim, tag) pairs\n");
		}

		int num_surfaces_added = 0;
		int num_volumes_added = 0;
		int extrusion = 0;
		for (int t = 0; t < (int)n_out_dim_tags/2; t++)
		{
			int tag_dim = out_dim_tags[2*t];
			int tag = out_dim_tags[2*t+1];
			switch (tag_dim)
			{
				case 0:
					printf("WARNING: gmshModelOccExtrude added a vertex\n");
					break;
				case 1:
					printf("WARNING: gmshModelOccExtrude added an edge\n");
					break;
				case 2:
					if (num_surfaces_added >= 5)
					{
						printf("ERROR: gmshModelOccExtrude added more than five surfaces\n");
						break;
					}

					// Start at 1 because there is already one surface
					physgroup_surfaces[1+num_surfaces_added++] = tag;
					break;
				case 3:
					if (num_volumes_added >= 1)
					{
						printf("ERROR: gmshModelOccExtrude added more than one volume\n");
						break;
					}

					// Fixed at 0 because there are no previously-existing extrusions and there shouldn't be any more
					physgroup_volumes_extrusions[num_volumes_added++] = tag;
					extrusion = tag;
					break;
			}
		}
		if (num_surfaces_added < 5)
		{
			printf("ERROR: gmshModelOccExtrude only added %d surfaces, expected 5\n", num_surfaces_added);
		}
		if (num_volumes_added < 1)
		{
			printf("ERROR: gmshModelOccExtrude did not add a new volume\n");
		}
		printf("Extrusion tag: %d\n", extrusion);

		// TODO - synchronization currently performed after every electrode, determine most efficient rate
		gmshModelOccSynchronize(&ierr);
		printf("gmshModelOccSynchronize Result: %d\n", ierr);

		// Add physical groups to model
		gmshModelAddPhysicalGroup(
			0,
			physgroup_vertices,
			sizeof(physgroup_vertices)/sizeof(physgroup_vertices[0]),
			5*e+1,
			"vertices",
			&ierr
		);
		gmshModelAddPhysicalGroup(
			1,
			physgroup_edges,
			sizeof(physgroup_edges)/sizeof(physgroup_edges[0]),
			5*e+2,
			"edges",
			&ierr
		);
		gmshModelAddPhysicalGroup(
			2,
			physgroup_surfaces,
			sizeof(physgroup_surfaces)/sizeof(physgroup_surfaces[0]),
			5*e+3,
			"surfaces",
			&ierr
		);
		gmshModelAddPhysicalGroup(
			3,
			physgroup_volumes_spheres,
			sizeof(physgroup_volumes_spheres)/sizeof(physgroup_volumes_spheres[0]),
			5*e+4,
			"volumes_spheres",
			&ierr
		);
		gmshModelAddPhysicalGroup(
			3,
			physgroup_volumes_extrusions,
			sizeof(physgroup_volumes_extrusions)/sizeof(physgroup_volumes_extrusions[0]),
			5*e+5,
			"volumes_extrusions",
			&ierr
		);

		// Generate mesh
		gmshModelMeshGenerate(3, &ierr);

		// Save mesh to file (forcing msh file version 2.0 for Sparselizard compatibility)
		gmshOptionSetNumber("Mesh.MshFileVersion", 2.0, &ierr);

		sprintf((*trap->electrodes)[e].electrode_mesh_filename, "meshes/%s.msh", electrode_id);
		gmshWrite((*trap->electrodes)[e].electrode_mesh_filename, &ierr);

		gmshFinalize(&ierr);

		gmshFree(out_dim_tags);
	}
}

