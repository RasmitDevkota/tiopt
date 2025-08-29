#include "io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "defs.h"
#include "data_structures.h"

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

	char line[80+2];
	
	// @TODO - use something safer than sscanf or add additional checks
	while (fgets(line, sizeof(line), trap_fptr) != NULL)
	{
		char target[9+1];
		sscanf(line, "%s.*", target);

		if (strncmp("#", target, 1) == 0 || target[0] == '\0')
		{
			continue;
		}
		else if (feof(trap_fptr))
		{
			break;
		}
		else if (strcmp(target, "trap") == 0)
		{
			char parameter[19+1];
			sscanf(line, "trap %s.*", parameter);

			if (strcmp(parameter, "n_electrodes") == 0)
			{
				sscanf(line, "trap n_electrodes %d.*", &(trap->n_electrodes));

				// Since we know the number of electrodes now, we perform the common memory allocations
				trap->electrode_positions = malloc(trap->n_electrodes * sizeof(double[3]));

				trap->electrodes = malloc(trap->n_electrodes * sizeof(struct Electrode));

				for (int electrode_num = 0; electrode_num < trap->n_electrodes; electrode_num++)
				{
					(*trap->electrodes)[electrode_num].Vlm_len = (LMAX+1)*(LMAX+1)*2;
					(*trap->electrodes)[electrode_num].Vlm = malloc((*trap->electrodes)[electrode_num].Vlm_len * sizeof(double));
				}
			}
			else if (strcmp(parameter, "n_electrodes_rf") == 0)
			{
				sscanf(line, "trap n_electrodes_rf %d.*", &(trap->n_electrodes_rf));
			}
			else if (strcmp(parameter, "n_electrodes_dc") == 0)
			{
				sscanf(line, "trap n_electrodes_dc %d.*", &(trap->n_electrodes_dc));
			}
			else if (strcmp(parameter, "electrode_positions") == 0)
			{
				int electrode_num;

				sscanf(line, "trap electrode_positions %d.*", &electrode_num);
				sscanf(
					line,
					"trap electrode_positions %*d (%lf %lf %lf).*",
					&(*trap->electrode_positions)[electrode_num][0],
					&(*trap->electrode_positions)[electrode_num][1],
					&(*trap->electrode_positions)[electrode_num][2]
				);
			}
			else
			{
				printf("ERROR: Invalid trap parameter '%s' detected in trap file\n", parameter);
				return 1;
			}
		}
		else if (strcmp(target, "electrode") == 0)
		{
			char parameter[10+1];
			sscanf(line, "electrode %s.*", parameter);

			if (strcmp(parameter, "n_vertices") == 0)
			{
				int electrode_num;
				
				sscanf(line, "electrode n_vertices %d.*", &electrode_num);
				sscanf(line, "electrode n_vertices %*d %d.*", &(*trap->electrodes)[electrode_num].n_vertices);

				(*trap->electrodes)[electrode_num].vertices = malloc((*trap->electrodes)[electrode_num].n_vertices * sizeof(double[3]));
			}
			else if (strcmp(parameter, "n_edges") == 0)
			{
				int electrode_num;
				
				sscanf(line, "electrode n_edges %d.*", &electrode_num);
				sscanf(line, "electrode n_edges %*d %d.*", &(*trap->electrodes)[electrode_num].n_edges);

				(*trap->electrodes)[electrode_num].edges = malloc((*trap->electrodes)[electrode_num].n_edges * sizeof(int[2]));
			}
			else if (strcmp(parameter, "vertex") == 0 || strcmp(parameter, "v") == 0)
			{
				int electrode_num;
				int vertex_num;

				sscanf(line, "electrode %*s %d %d.*", &electrode_num, &vertex_num);
				sscanf(
					line,
					"electrode %*s %*d %*d (%lf %lf %lf).*",
					&(*trap->electrodes)[electrode_num].vertices[vertex_num][0],
					&(*trap->electrodes)[electrode_num].vertices[vertex_num][1],
					&(*trap->electrodes)[electrode_num].vertices[vertex_num][2]
				);
			}
			else if (strcmp(parameter, "edge") == 0 || strcmp(parameter, "e") == 0)
			{
				int electrode_num;
				int edge_num;

				sscanf(line, "electrode %*s %d %d.*", &electrode_num, &edge_num);
				sscanf(
					line,
					"electrode %*s %*d %*d (%d %d).*",
					&(*trap->electrodes)[electrode_num].edges[edge_num][0],
					&(*trap->electrodes)[electrode_num].edges[edge_num][1]
				);
			}
			else if (strcmp(parameter, "surface") == 0 || strcmp(parameter, "s") == 0)
			{
				printf("WARNING: Surface definition found but tiopt currently only supports a single surface; ignoring.");
			}
			else
			{
				printf("ERROR: Invalid electrode parameter '%s' detected in trap file\n", parameter);
				return 1;
			}
		}
		else
		{
			printf("ERROR: Invalid target '%s' detected in trap file\n", target);
			return 1;
		}
	}

	fclose(trap_fptr);

	return 0;
}

