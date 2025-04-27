#include "trap_geometry.h"

#include <stdio.h>
#include <math.h>

#include <gmshc.h>

// generate the initial mesh object needed from the trap electrodes
void generate_mesh(
	struct Trap *trap
) {
	const double lc = 1E-2;
	for (int e = 0; e < trap->n_electrodes; e++)
	{
		printf("Generating mesh for electrode %d/%d\n", e+1, trap->n_electrodes);

		int ierr;

		int argc;
		char** argv;
		gmshInitialize(argc, argv, 1, 0, &ierr);

		char electrode_id[11];
		sprintf(
			electrode_id,
			"e_%2s-%6d",
			e < trap->n_electrodes_rf ? "rf" : "dc",
			e
		);

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
			int vertex_v = gmshModelGeoAddPoint((*trap->electrodes)[e].vertices[v][0], (*trap->electrodes)[e].vertices[v][1], 0.0, lc, v+1, &ierr);

			physgroup_vertices[v] = vertex_v;
		}

		// 1D - Edges
		int physgroup_edges[(*trap->electrodes)[e].n_edges];
		for (int d = 0; d < (*trap->electrodes)[e].n_edges; d++)
		{
			int edge_e = gmshModelGeoAddLine((*trap->electrodes)[e].edges[d][0]+1, (*trap->electrodes)[e].edges[d][1]+1, d+1, &ierr);

			physgroup_edges[e] = edge_e;
		}

		// 2D - Curve loops
		// %TODO - make sure that the edge list has the appropriate order and signs
		//       - this can either be a user expectation at input or enforced here
		int curveloop_definition[(*trap->electrodes)[e].n_edges];
		for (int d = 0; d < (*trap->electrodes)[e].n_edges; d++)
		{
			curveloop_definition[d] = d+1;
		}
		int curveloop = gmshModelGeoAddCurveLoop(curveloop_definition, sizeof(curveloop_definition)/sizeof(curveloop_definition[0]), e+1, 0, &ierr);

		// 2D - Surfaces
		const int surface_definition[] = { 1 };
		int surface = gmshModelGeoAddPlaneSurface(surface_definition, sizeof(surface_definition)/sizeof(surface_definition[0]), e+1, &ierr);
		int physgroup_surfaces[1] = { surface };

		// // 3D - Cubes
		// int box = gmshModelOccAddBox(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, &ierr);
		// // 3D - Spheres
		// int sphere = gmshModelOccAddSphere(0.0, 0.0, 75.0, 70.0, 0, 2*M_PI, 0, M_PI, &ierr);
		// int physgroup_volumes[1] = { sphere };

		// TODO - synchronization currently performed after every electrode, determine most efficient rate
		gmshModelGeoSynchronize(&ierr);
		// gmshModelOccSynchronize(&ierr);

		gmshModelAddPhysicalGroup(0, physgroup_vertices, sizeof(physgroup_vertices)/sizeof(physgroup_vertices[0]), 4*e+0, "vertices", &ierr);
		gmshModelAddPhysicalGroup(1, physgroup_edges, sizeof(physgroup_edges)/sizeof(physgroup_edges[0]), 4*e+1, "edges", &ierr);
		gmshModelAddPhysicalGroup(2, physgroup_surfaces, sizeof(physgroup_surfaces)/sizeof(physgroup_surfaces[0]), 4*e+2, "surfaces", &ierr);
		// gmshModelAddPhysicalGroup(3, physgroup_volumes, sizeof(physgroup_volumes)/sizeof(physgroup_volumes[0]), 4*e+3, "volumes", &ierr);

		gmshModelMeshGenerate(2, &ierr);

		gmshOptionSetNumber("Mesh.MshFileVersion", 2.0, &ierr);

		sprintf((*trap->electrodes)[e].electrode_mesh_filename, "meshes/%s.msh", electrode_id);
		gmshWrite((*trap->electrodes)[e].electrode_mesh_filename, &ierr);

		gmshFinalize(&ierr);
	}
}

