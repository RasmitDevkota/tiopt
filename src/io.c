#include "io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "defs.h"
#include "data_structures.h"
#include "traps.h"

// Check that an index read from the trap file lies within [0, n)
static int check_index
(
	const char *name,
	int index,
	int n
)
{
	if (index < 0 || index >= n)
	{
		printf("ERROR: %s %d is out of range (must be between 0 and %d) in trap file\n", name, index, n - 1);
		return 1;
	}

	return 0;
}

// @TODO - currently not at the standard level of safety for an input parser,
//         needs many security improvements
int generate_trap_from_file(
	char *trap_filename,
	struct Trap *trap
) {
	FILE *trap_fptr = fopen(trap_filename, "r");

	if (trap_fptr == NULL)
	{
		printf("ERROR: Failed to read trap file '%s'\n", trap_filename);

		return 1;
	}

	memset(trap, 0, sizeof(struct Trap));

	char line[255+2];
	int line_num = 0;
	int status = 0;

	// @TODO - use something safer than sscanf or add additional checks
	while (status == 0 && fgets(line, sizeof(line), trap_fptr) != NULL)
	{
		line_num++;

		char target[9+1];
		target[0] = '\0';
		sscanf(line, "%9s", target);

		if (strncmp("#", target, 1) == 0 || target[0] == '\0')
		{
			continue;
		}
		else if (strcmp(target, "trap") == 0)
		{
			char parameter[19+1];
			parameter[0] = '\0';
			sscanf(line, "trap %19s", parameter);

			if (strcmp(parameter, "n_electrodes") == 0)
			{
				int n_electrodes = 0;
				sscanf(line, "trap n_electrodes %d", &n_electrodes);

				if (n_electrodes <= 0 || trap->electrodes != NULL)
				{
					printf("ERROR: Invalid or repeated number of electrodes on line %d of trap file\n", line_num);
					status = 1;
					continue;
				}

				// Since we know the number of electrodes now, we perform the common memory allocations
				allocate_trap(trap, n_electrodes);
			}
			else if (strcmp(parameter, "n_electrodes_rf") == 0)
			{
				sscanf(line, "trap n_electrodes_rf %d", &(trap->n_electrodes_rf));
			}
			else if (strcmp(parameter, "n_electrodes_dc") == 0)
			{
				sscanf(line, "trap n_electrodes_dc %d", &(trap->n_electrodes_dc));
			}
			else if (strcmp(parameter, "electrode_positions") == 0)
			{
				int electrode_num = -1;

				sscanf(line, "trap electrode_positions %d", &electrode_num);
				if (check_index("Electrode", electrode_num, trap->n_electrodes))
				{
					status = 1;
					continue;
				}

				sscanf(
					line,
					"trap electrode_positions %*d (%lf %lf %lf)",
					&(*trap->electrode_positions)[electrode_num][0],
					&(*trap->electrode_positions)[electrode_num][1],
					&(*trap->electrode_positions)[electrode_num][2]
				);
			}
			else if (strcmp(parameter, "V_RF") == 0)
			{
				sscanf(line, "trap V_RF %lf", &(trap->V_RF));
			}
			else if (strcmp(parameter, "Omega_RF") == 0)
			{
				sscanf(line, "trap Omega_RF %lf", &(trap->Omega_RF));
			}
			else if (strcmp(parameter, "n_grating_couplers") == 0)
			{
				int n_grating_couplers = 0;
				sscanf(line, "trap n_grating_couplers %d", &n_grating_couplers);

				if (n_grating_couplers < 0 || trap->grating_couplers != NULL)
				{
					printf("ERROR: Invalid or repeated number of grating couplers on line %d of trap file\n", line_num);
					status = 1;
					continue;
				}

				allocate_grating_couplers(trap, n_grating_couplers);
			}
			else if (strcmp(parameter, "n_detectors") == 0)
			{
				int n_detectors = 0;
				sscanf(line, "trap n_detectors %d", &n_detectors);

				if (n_detectors < 0 || trap->detectors != NULL)
				{
					printf("ERROR: Invalid or repeated number of detectors on line %d of trap file\n", line_num);
					status = 1;
					continue;
				}

				allocate_detectors(trap, n_detectors);
			}
			else
			{
				printf("ERROR: Invalid trap parameter '%s' detected in trap file\n", parameter);
				status = 1;
			}
		}
		else if (strcmp(target, "electrode") == 0)
		{
			char parameter[10+1];
			parameter[0] = '\0';
			sscanf(line, "electrode %10s", parameter);

			int electrode_num = -1;
			sscanf(line, "electrode %*s %d", &electrode_num);

			if (strcmp(parameter, "surface") != 0 && strcmp(parameter, "s") != 0 && check_index("Electrode", electrode_num, trap->n_electrodes))
			{
				status = 1;
				continue;
			}

			if (strcmp(parameter, "n_vertices") == 0)
			{
				struct Electrode *electrode = &(*trap->electrodes)[electrode_num];

				sscanf(line, "electrode n_vertices %*d %d", &electrode->n_vertices);

				free(electrode->vertices);
				electrode->vertices = calloc(electrode->n_vertices, sizeof(double[3]));
			}
			else if (strcmp(parameter, "n_edges") == 0)
			{
				struct Electrode *electrode = &(*trap->electrodes)[electrode_num];

				sscanf(line, "electrode n_edges %*d %d", &electrode->n_edges);

				free(electrode->edges);
				electrode->edges = calloc(electrode->n_edges, sizeof(int[2]));
			}
			else if (strcmp(parameter, "vertex") == 0 || strcmp(parameter, "v") == 0)
			{
				struct Electrode *electrode = &(*trap->electrodes)[electrode_num];

				int vertex_num = -1;

				sscanf(line, "electrode %*s %*d %d", &vertex_num);
				if (check_index("Vertex", vertex_num, electrode->n_vertices))
				{
					status = 1;
					continue;
				}

				sscanf(
					line,
					"electrode %*s %*d %*d (%lf %lf %lf)",
					&electrode->vertices[vertex_num][0],
					&electrode->vertices[vertex_num][1],
					&electrode->vertices[vertex_num][2]
				);
			}
			else if (strcmp(parameter, "edge") == 0 || strcmp(parameter, "e") == 0)
			{
				struct Electrode *electrode = &(*trap->electrodes)[electrode_num];

				int edge_num = -1;

				sscanf(line, "electrode %*s %*d %d", &edge_num);
				if (check_index("Edge", edge_num, electrode->n_edges))
				{
					status = 1;
					continue;
				}

				sscanf(
					line,
					"electrode %*s %*d %*d (%d %d)",
					&electrode->edges[edge_num][0],
					&electrode->edges[edge_num][1]
				);
			}
			else if (strcmp(parameter, "surface") == 0 || strcmp(parameter, "s") == 0)
			{
				printf("WARNING: Surface definition found but tiopt currently only supports a single surface; ignoring.\n");
			}
			else
			{
				printf("ERROR: Invalid electrode parameter '%s' detected in trap file\n", parameter);
				status = 1;
			}
		}
		else if (strcmp(target, "grating") == 0)
		{
			char parameter[19+1];
			parameter[0] = '\0';
			sscanf(line, "grating %19s", parameter);

			int grating_num = -1;
			sscanf(line, "grating %*s %d", &grating_num);

			if (check_index("Grating coupler", grating_num, trap->n_grating_couplers))
			{
				status = 1;
				continue;
			}

			struct GratingCoupler *grating = &(*trap->grating_couplers)[grating_num];

			// Scalar parameters of the grating and of the waveguide feeding it
			char *scalar_names[] = { "azimuth", "period", "fill_factor", "etch_depth", "length", "width", "focal_distance", "power", "efficiency", "wavelength", "waveguide_width", "waveguide_thickness", "n_core", "n_cladding" };
			double *scalar_values[] = { &grating->azimuth, &grating->period, &grating->fill_factor, &grating->etch_depth, &grating->length, &grating->width, &grating->focal_distance, &grating->power, &grating->efficiency, &grating->waveguide.wavelength, &grating->waveguide.width, &grating->waveguide.thickness, &grating->waveguide.n_core, &grating->waveguide.n_cladding };
			int n_scalars = sizeof(scalar_names) / sizeof(scalar_names[0]);

			int found = 0;
			for (int k = 0; k < n_scalars; k++)
				if (strcmp(parameter, scalar_names[k]) == 0)
				{
					sscanf(line, "grating %*s %*d %lf", scalar_values[k]);
					found = 1;
				}

			if (strcmp(parameter, "position") == 0)
			{
				sscanf(
					line,
					"grating position %*d (%lf %lf %lf)",
					&grating->position[0],
					&grating->position[1],
					&grating->position[2]
				);
			}
			else if (!found)
			{
				printf("ERROR: Invalid grating parameter '%s' detected in trap file\n", parameter);
				status = 1;
			}
		}
		else if (strcmp(target, "detector") == 0)
		{
			char parameter[19+1];
			parameter[0] = '\0';
			sscanf(line, "detector %19s", parameter);

			int detector_num = -1;
			sscanf(line, "detector %*s %d", &detector_num);

			if (check_index("Detector", detector_num, trap->n_detectors))
			{
				status = 1;
				continue;
			}

			struct Detector *detector = &(*trap->detectors)[detector_num];

			char *scalar_names[] = { "width", "length", "wavelength", "quantum_efficiency", "dark_count_rate" };
			double *scalar_values[] = { &detector->width, &detector->length, &detector->wavelength, &detector->quantum_efficiency, &detector->dark_count_rate };
			int n_scalars = sizeof(scalar_names) / sizeof(scalar_names[0]);

			int found = 0;
			for (int k = 0; k < n_scalars; k++)
				if (strcmp(parameter, scalar_names[k]) == 0)
				{
					sscanf(line, "detector %*s %*d %lf", scalar_values[k]);
					found = 1;
				}

			if (strcmp(parameter, "position") == 0)
			{
				sscanf(
					line,
					"detector position %*d (%lf %lf %lf)",
					&detector->position[0],
					&detector->position[1],
					&detector->position[2]
				);
			}
			else if (!found)
			{
				printf("ERROR: Invalid detector parameter '%s' detected in trap file\n", parameter);
				status = 1;
			}
		}
		else
		{
			printf("ERROR: Invalid target '%s' detected in trap file\n", target);
			status = 1;
		}
	}

	fclose(trap_fptr);

	if (status == 0 && trap->n_electrodes_rf + trap->n_electrodes_dc != trap->n_electrodes)
	{
		printf("WARNING: Numbers of RF and DC electrodes do not add up to the number of electrodes; treating electrodes [%d, %d) as DC\n", trap->n_electrodes_rf, trap->n_electrodes);
		trap->n_electrodes_dc = trap->n_electrodes - trap->n_electrodes_rf;
	}

	return status;
}

// Write a trap in the format read by generate_trap_from_file
int write_trap_to_file
(
	char *trap_filename,
	struct Trap *trap
)
{
	FILE *trap_fptr = fopen(trap_filename, "w");

	if (trap_fptr == NULL)
	{
		printf("ERROR: Failed to write trap file '%s'\n", trap_filename);

		return 1;
	}

	fprintf(trap_fptr, "trap n_electrodes %d\n", trap->n_electrodes);
	fprintf(trap_fptr, "trap n_electrodes_rf %d\n", trap->n_electrodes_rf);
	fprintf(trap_fptr, "trap n_electrodes_dc %d\n", trap->n_electrodes_dc);
	fprintf(trap_fptr, "trap V_RF %.17g\n", trap->V_RF);
	fprintf(trap_fptr, "trap Omega_RF %.17g\n", trap->Omega_RF);
	fprintf(trap_fptr, "\n");

	for (int e = 0; e < trap->n_electrodes; e++)
		fprintf(
			trap_fptr,
			"trap electrode_positions %d (%.17g %.17g %.17g)\n",
			e,
			(*trap->electrode_positions)[e][0],
			(*trap->electrode_positions)[e][1],
			(*trap->electrode_positions)[e][2]
		);

	for (int e = 0; e < trap->n_electrodes; e++)
	{
		struct Electrode *electrode = &(*trap->electrodes)[e];

		fprintf(trap_fptr, "\n");
		fprintf(trap_fptr, "electrode n_vertices %d %d\n", e, electrode->n_vertices);
		fprintf(trap_fptr, "electrode n_edges %d %d\n", e, electrode->n_edges);

		for (int v = 0; v < electrode->n_vertices; v++)
			fprintf(
				trap_fptr,
				"electrode v %d %d (%.17g %.17g %.17g)\n",
				e,
				v,
				electrode->vertices[v][0],
				electrode->vertices[v][1],
				electrode->vertices[v][2]
			);

		for (int i = 0; i < electrode->n_edges; i++)
			fprintf(trap_fptr, "electrode e %d %d (%d %d)\n", e, i, electrode->edges[i][0], electrode->edges[i][1]);
	}

	if (trap->n_grating_couplers > 0)
	{
		fprintf(trap_fptr, "\n");
		fprintf(trap_fptr, "trap n_grating_couplers %d\n", trap->n_grating_couplers);
	}

	for (int g = 0; g < trap->n_grating_couplers; g++)
	{
		struct GratingCoupler *grating = &(*trap->grating_couplers)[g];

		fprintf(trap_fptr, "\n");
		fprintf(trap_fptr, "grating position %d (%.17g %.17g %.17g)\n", g, grating->position[0], grating->position[1], grating->position[2]);
		fprintf(trap_fptr, "grating azimuth %d %.17g\n", g, grating->azimuth);
		fprintf(trap_fptr, "grating period %d %.17g\n", g, grating->period);
		fprintf(trap_fptr, "grating fill_factor %d %.17g\n", g, grating->fill_factor);
		fprintf(trap_fptr, "grating etch_depth %d %.17g\n", g, grating->etch_depth);
		fprintf(trap_fptr, "grating length %d %.17g\n", g, grating->length);
		fprintf(trap_fptr, "grating width %d %.17g\n", g, grating->width);
		fprintf(trap_fptr, "grating focal_distance %d %.17g\n", g, grating->focal_distance);
		fprintf(trap_fptr, "grating power %d %.17g\n", g, grating->power);
		fprintf(trap_fptr, "grating efficiency %d %.17g\n", g, grating->efficiency);
		fprintf(trap_fptr, "grating wavelength %d %.17g\n", g, grating->waveguide.wavelength);
		fprintf(trap_fptr, "grating waveguide_width %d %.17g\n", g, grating->waveguide.width);
		fprintf(trap_fptr, "grating waveguide_thickness %d %.17g\n", g, grating->waveguide.thickness);
		fprintf(trap_fptr, "grating n_core %d %.17g\n", g, grating->waveguide.n_core);
		fprintf(trap_fptr, "grating n_cladding %d %.17g\n", g, grating->waveguide.n_cladding);
	}

	if (trap->n_detectors > 0)
	{
		fprintf(trap_fptr, "\n");
		fprintf(trap_fptr, "trap n_detectors %d\n", trap->n_detectors);
	}

	for (int d = 0; d < trap->n_detectors; d++)
	{
		struct Detector *detector = &(*trap->detectors)[d];

		fprintf(trap_fptr, "\n");
		fprintf(trap_fptr, "detector position %d (%.17g %.17g %.17g)\n", d, detector->position[0], detector->position[1], detector->position[2]);
		fprintf(trap_fptr, "detector width %d %.17g\n", d, detector->width);
		fprintf(trap_fptr, "detector length %d %.17g\n", d, detector->length);
		fprintf(trap_fptr, "detector wavelength %d %.17g\n", d, detector->wavelength);
		fprintf(trap_fptr, "detector quantum_efficiency %d %.17g\n", d, detector->quantum_efficiency);
		fprintf(trap_fptr, "detector dark_count_rate %d %.17g\n", d, detector->dark_count_rate);
	}

	fclose(trap_fptr);

	return 0;
}
