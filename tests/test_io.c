#include "test_io.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"
#include "io.h"
#include "traps.h"
#include "test_utils.h"

int test_all_io
(
	double atol
)
{
	printf("- Testing input and output...\n");

	int result = 0;

	result += test_generate_trap_from_file(atol);
	result += test_write_trap_to_file(atol);
	result += test_invalid_trap_file(atol);

	return result;
}

int test_generate_trap_from_file
(
	double atol
)
{
	struct Trap trap;
	if (generate_trap_from_file("../trap.data", &trap) != 0)
		return check_error("generate_trap_from_file reads trap.data", HUGE_VAL, atol);

	// 1. Expected contents of trap.data
	double positions[2][3] = { { 25, 0, 0 }, { 50, 0, 0 } };
	double vertices[4][3] = { { 10, 10, 0 }, { -10, 10, 0 }, { -10, -10, 0 }, { 10, -10, 0 } };
	int edges[4][2] = { { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 } };

	// 2. tiopt result
	double error = 0.0;
	error += abs(trap.n_electrodes - 2) + abs(trap.n_electrodes_rf - 0) + abs(trap.n_electrodes_dc - 2);

	for (int e = 0; e < MIN(trap.n_electrodes, 2); e++)
	{
		struct Electrode *electrode = &(*trap.electrodes)[e];

		for (int k = 0; k < 3; k++)
			error += fabs((*trap.electrode_positions)[e][k] - positions[e][k]);

		error += abs(electrode->n_vertices - 4) + abs(electrode->n_edges - 4);

		for (int v = 0; v < MIN(electrode->n_vertices, 4); v++)
			for (int k = 0; k < 3; k++)
				error += fabs(electrode->vertices[v][k] - vertices[v][k]);

		for (int i = 0; i < MIN(electrode->n_edges, 4); i++)
			error += abs(electrode->edges[i][0] - edges[i][0]) + abs(electrode->edges[i][1] - edges[i][1]);

		error += abs(electrode->Vlm_len - NSPH_X*NSPH_Y*NSPH_Z*VLM_SPHERE_LEN);
	}

	free_trap(&trap);

	return check_error("generate_trap_from_file reads trap.data", error, atol);
}

int test_write_trap_to_file
(
	double atol
)
{
	struct Trap trap;
	generate_trap_from_preset(LINEAR_SURFACE_TRAP_WITH_PHOTONICS, 3, 40.0, &trap);

	char *trap_filename = "test_write_trap_to_file.data";
	write_trap_to_file(trap_filename, &trap);

	struct Trap trap_read;
	int status = generate_trap_from_file(trap_filename, &trap_read);
	remove(trap_filename);

	if (status != 0)
	{
		free_trap(&trap);
		return check_error("write_trap_to_file round trip", HUGE_VAL, atol);
	}

	double error = 0.0;
	error += abs(trap.n_electrodes - trap_read.n_electrodes) + abs(trap.n_electrodes_rf - trap_read.n_electrodes_rf) + abs(trap.n_electrodes_dc - trap_read.n_electrodes_dc);
	error += fabs(trap.V_RF - trap_read.V_RF) + fabs(trap.Omega_RF - trap_read.Omega_RF) / trap.Omega_RF;
	error += abs(trap.n_grating_couplers - trap_read.n_grating_couplers);
	error += abs(trap.n_detectors - trap_read.n_detectors);

	for (int e = 0; e < MIN(trap.n_electrodes, trap_read.n_electrodes); e++)
	{
		struct Electrode *electrode = &(*trap.electrodes)[e];
		struct Electrode *electrode_read = &(*trap_read.electrodes)[e];

		for (int k = 0; k < 3; k++)
			error += fabs((*trap.electrode_positions)[e][k] - (*trap_read.electrode_positions)[e][k]);

		error += abs(electrode->n_vertices - electrode_read->n_vertices);
		for (int v = 0; v < MIN(electrode->n_vertices, electrode_read->n_vertices); v++)
			for (int k = 0; k < 3; k++)
				error += fabs(electrode->vertices[v][k] - electrode_read->vertices[v][k]);
	}

	for (int g = 0; g < MIN(trap.n_grating_couplers, trap_read.n_grating_couplers); g++)
	{
		struct GratingCoupler *grating = &(*trap.grating_couplers)[g];
		struct GratingCoupler *grating_read = &(*trap_read.grating_couplers)[g];

		for (int k = 0; k < 3; k++)
			error += fabs(grating->position[k] - grating_read->position[k]);

		error += fabs(grating->azimuth - grating_read->azimuth) + fabs(grating->period - grating_read->period);
		error += fabs(grating->fill_factor - grating_read->fill_factor) + fabs(grating->etch_depth - grating_read->etch_depth);
		error += fabs(grating->length - grating_read->length) + fabs(grating->width - grating_read->width);
		error += fabs(grating->focal_distance - grating_read->focal_distance) + fabs(grating->power - grating_read->power);
		error += fabs(grating->efficiency - grating_read->efficiency);
		error += fabs(grating->waveguide.wavelength - grating_read->waveguide.wavelength) + fabs(grating->waveguide.width - grating_read->waveguide.width);
		error += fabs(grating->waveguide.thickness - grating_read->waveguide.thickness) + fabs(grating->waveguide.n_core - grating_read->waveguide.n_core);
		error += fabs(grating->waveguide.n_cladding - grating_read->waveguide.n_cladding);
	}

	for (int d = 0; d < MIN(trap.n_detectors, trap_read.n_detectors); d++)
	{
		struct Detector *detector = &(*trap.detectors)[d];
		struct Detector *detector_read = &(*trap_read.detectors)[d];

		for (int k = 0; k < 3; k++)
			error += fabs(detector->position[k] - detector_read->position[k]);

		error += fabs(detector->width - detector_read->width) + fabs(detector->length - detector_read->length);
		error += fabs(detector->wavelength - detector_read->wavelength) + fabs(detector->quantum_efficiency - detector_read->quantum_efficiency);
		error += fabs(detector->dark_count_rate - detector_read->dark_count_rate);
	}

	free_trap(&trap_read);
	free_trap(&trap);

	return check_error("write_trap_to_file round trip", error, atol);
}

int test_invalid_trap_file
(
	double atol
)
{
	char *trap_filename = "test_invalid_trap_file.data";

	// A vertex index beyond the number of vertices must be rejected rather than written out of bounds
	FILE *f = fopen(trap_filename, "w");
	fprintf(f, "trap n_electrodes 1\n");
	fprintf(f, "electrode n_vertices 0 3\n");
	fprintf(f, "electrode v 0 7 (1 2 3)\n");
	fclose(f);

	struct Trap trap;
	int status = generate_trap_from_file(trap_filename, &trap);
	free_trap(&trap);
	remove(trap_filename);

	int result = check_error("generate_trap_from_file rejects out of range vertex", status == 1 ? 0 : 1, 0);

	// Electrodes cannot be described before they are allocated
	f = fopen(trap_filename, "w");
	fprintf(f, "electrode n_vertices 0 3\n");
	fclose(f);

	status = generate_trap_from_file(trap_filename, &trap);
	free_trap(&trap);
	remove(trap_filename);

	result += check_error("generate_trap_from_file rejects undeclared electrode", status == 1 ? 0 : 1, 0);

	// Missing files are reported
	status = generate_trap_from_file("this_file_does_not_exist.data", &trap);
	result += check_error("generate_trap_from_file reports missing file", status == 1 ? 0 : 1, 0);

	return result;
}
