#include "trap_geometry.h"

#include <stdio.h>
#include <math.h>

#include <gmshc.h>

// generate the initial mesh object needed from the trap electrodes
void generate_mesh(
	struct Trap *trap
) {
	const double lc = 1E-1;
	for (int e = 0; e < trap->n_electrodes; e++)
	{
		printf("Generating mesh for electrode %d/%d\n", e+1, trap->n_electrodes);

		int ierr;

		int argc = 0;
		char** argv = NULL;
		gmshInitialize(argc, argv, 1, 0, &ierr);

		// Format unique electrode_id for model and mesh file
		char electrode_id[11];
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

		gmshModelAdd(electrode_id, &ierr);

		// 0D - Vertices
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

		// 1D - Edges
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

		// 2D - Curve loops
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

		// 2D - Surfaces
		const int surface_definition[1] = { curveloop };
		int surface = gmshModelOccAddPlaneSurface(
			surface_definition,
			sizeof(surface_definition)/sizeof(surface_definition[0]),
			2*e+2,
			&ierr
		);
		int physgroup_surfaces[1] = { surface };
		printf("gmshModelOccAddSurface Result: %d\n", ierr);

		// 3D - Spheres
		int sphere = gmshModelOccAddSphere(
			0.0, 0.0, 75.0,
			70.0,
			e+1,
			-0.5*M_PI, 0.5*M_PI, 2*M_PI,
			&ierr
		);
		int physgroup_volumes[1] = { sphere };
		printf("gmshModelOccAddSphere Result: %d\n", ierr);

		// TODO - synchronization currently performed after every electrode, determine most efficient rate
		gmshModelOccSynchronize(&ierr);
		printf("gmshModelOccSynchronize Result: %d\n", ierr);

		// Add physical groups to model
		gmshModelAddPhysicalGroup(
			0,
			physgroup_vertices,
			sizeof(physgroup_vertices)/sizeof(physgroup_vertices[0]),
			4*e+1,
			"vertices",
			&ierr
		);
		gmshModelAddPhysicalGroup(
			1,
			physgroup_edges,
			sizeof(physgroup_edges)/sizeof(physgroup_edges[0]),
			4*e+2,
			"edges",
			&ierr
		);
		gmshModelAddPhysicalGroup(
			2,
			physgroup_surfaces,
			sizeof(physgroup_surfaces)/sizeof(physgroup_surfaces[0]),
			4*e+3,
			"surfaces",
			&ierr
		);
		gmshModelAddPhysicalGroup(
			3,
			physgroup_volumes,
			sizeof(physgroup_volumes)/sizeof(physgroup_volumes[0]),
			4*e+4,
			"volumes",
			&ierr
		);

		// Generate mesh
		gmshModelMeshGenerate(3, &ierr);

		// Save mesh to file (forcing msh file version 2.0 for Sparselizard compatibility)
		gmshOptionSetNumber("Mesh.MshFileVersion", 2.0, &ierr);

		sprintf((*trap->electrodes)[e].electrode_mesh_filename, "meshes/%s.msh", electrode_id);
		gmshWrite((*trap->electrodes)[e].electrode_mesh_filename, &ierr);

		gmshFinalize(&ierr);
	}
}

