#include "traps.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"
#include "photonics.h"
#include "electrodynamics.h"

// Allocate the electrodes of a trap, including storage for their spherical harmonics expansions
void allocate_trap
(
	struct Trap *trap,
	int n_electrodes
)
{
	trap->n_electrodes = n_electrodes;

	trap->electrode_positions = calloc(n_electrodes, sizeof(double[3]));

	trap->electrodes = calloc(n_electrodes, sizeof(struct Electrode));

	for (int e = 0; e < n_electrodes; e++)
	{
		(*trap->electrodes)[e].Vlm_len = NSPH_X*NSPH_Y*NSPH_Z*VLM_SPHERE_LEN;

		(*trap->electrodes)[e].Vlm = calloc((*trap->electrodes)[e].Vlm_len, sizeof(double));
	}
}

// Allocate n_grating_couplers grating couplers for a trap, all set to zero
void allocate_grating_couplers
(
	struct Trap *trap,
	int n_grating_couplers
)
{
	trap->n_grating_couplers = n_grating_couplers;

	trap->grating_couplers = calloc(n_grating_couplers, sizeof(struct GratingCoupler));
}

// Allocate n_detectors detectors for a trap, all set to zero
void allocate_detectors
(
	struct Trap *trap,
	int n_detectors
)
{
	trap->n_detectors = n_detectors;

	trap->detectors = calloc(n_detectors, sizeof(struct Detector));
}

// Add integrated photonics for 40Ca+ to a trap, replacing any it already has: a grating coupler for each beam of enum
// PhotonicsBeam, focused on the gate or measurement zone from below, and a detector beneath the measurement zone.
// Returns 1 if a grating coupler cannot be designed
int add_trap_photonics
(
	struct Trap *trap,
	double gate_position[3],
	double measure_position[3]
)
{
	free(trap->grating_couplers);
	free(trap->detectors);

	allocate_grating_couplers(trap, N_PHOTONICS_BEAMS);
	allocate_detectors(trap, 1);

	// Waveguides of silicon nitride for the red and infrared beams and of alumina for the near-ultraviolet beam, which
	// silicon nitride absorbs, in silica cladding (dimensions in µm)
	double wavelengths[N_PHOTONICS_BEAMS] = { 0.729347, 0.396959, 0.866214, 0.854209 };
	double n_cores[N_PHOTONICS_BEAMS] = { 2.0, 1.65, 2.0, 2.0 };
	double thicknesses[N_PHOTONICS_BEAMS] = { 0.15, 0.15, 0.2, 0.2 };
	double widths[N_PHOTONICS_BEAMS] = { 0.6, 0.6, 0.8, 0.8 };
	double powers[N_PHOTONICS_BEAMS] = { 1E-3, 1E-5, 1E-4, 1E-4 };
	int at_gate_zone[N_PHOTONICS_BEAMS] = { 1, 0, 0, 1 };

	// Every beam arrives diagonally to the trap axis, so that it has components along the axial mode and across the
	// radial modes, and from a different quadrant so that the gratings do not overlap
	double azimuths[N_PHOTONICS_BEAMS] = { 0.25 * PI, 0.75 * PI, -0.25 * PI, -0.75 * PI };

	// Horizontal distance from each grating to the point beneath the zone it addresses
	double offset = 0.75 * ION_HEIGHT;

	for (int b = 0; b < N_PHOTONICS_BEAMS; b++)
	{
		struct GratingCoupler *grating = &(*trap->grating_couplers)[b];

		double *target = at_gate_zone[b] ? gate_position : measure_position;

		grating->waveguide.wavelength = wavelengths[b];
		grating->waveguide.width = widths[b];
		grating->waveguide.thickness = thicknesses[b];
		grating->waveguide.n_core = n_cores[b];
		grating->waveguide.n_cladding = 1.455;

		grating->position[0] = target[0] - offset * cos(azimuths[b]);
		grating->position[1] = target[1] - offset * sin(azimuths[b]);
		grating->position[2] = 0.0;

		grating->fill_factor = 0.5;
		grating->etch_depth = 0.5 * thicknesses[b];
		grating->length = 20.0;
		grating->width = 8.0;
		grating->power = powers[b];
		grating->efficiency = 0.3;

		if (design_grating_coupler(grating, target) != 0)
		{
			printf("ERROR: Failed to design grating coupler %d for trap photonics\n", b);
			return 1;
		}
	}

	struct Detector *detector = &(*trap->detectors)[0];

	detector->position[0] = measure_position[0];
	detector->position[1] = measure_position[1];
	detector->position[2] = 0.0;
	detector->width = 50.0;
	detector->length = 50.0;
	detector->wavelength = wavelengths[DETECTION_BEAM];
	detector->quantum_efficiency = 0.8;
	detector->dark_count_rate = 10.0;

	return 0;
}

// Define electrode e as a polygon with the given vertices relative to the given position, connected in order
void set_polygon_electrode
(
	struct Trap *trap,
	int e,
	int n_vertices,
	double (*vertices)[3],
	double position[3]
)
{
	struct Electrode *electrode = &(*trap->electrodes)[e];

	free(electrode->vertices);
	free(electrode->edges);

	electrode->n_vertices = n_vertices;
	electrode->n_edges = n_vertices;
	electrode->vertices = malloc(n_vertices * sizeof(double[3]));
	electrode->edges = malloc(n_vertices * sizeof(int[2]));

	for (int v = 0; v < n_vertices; v++)
	{
		for (int k = 0; k < 3; k++)
			electrode->vertices[v][k] = vertices[v][k];

		electrode->edges[v][0] = v;
		electrode->edges[v][1] = (v + 1) % n_vertices;
	}

	for (int k = 0; k < 3; k++)
		(*trap->electrode_positions)[e][k] = position[k];
}

// Define electrode e as a rectangle on the trap surface, positioned at its center
void set_rectangular_electrode
(
	struct Trap *trap,
	int e,
	double x_min,
	double x_max,
	double y_min,
	double y_max
)
{
	double position[3] = { 0.5 * (x_min + x_max), 0.5 * (y_min + y_max), 0.0 };

	double half_width = 0.5 * (x_max - x_min);
	double half_length = 0.5 * (y_max - y_min);

	// Counterclockwise, starting from the upper right corner (as in trap.data)
	double vertices[4][3] =
	{
		{ half_width, half_length, 0.0 },
		{ -half_width, half_length, 0.0 },
		{ -half_width, -half_length, 0.0 },
		{ half_width, -half_length, 0.0 }
	};

	set_polygon_electrode(trap, e, 4, vertices, position);
}

// Find the height of the RF null above the origin of the trap frame, where the derivative of the RF potential along
// z vanishes, using the analytical (gapless plane) potentials of the RF electrodes; assumes the trap is symmetric
// about y = 0
static double compute_rf_null_height_analytical
(
	struct Trap *trap
)
{
	double h = 1E-4;

	double z_low = 1.0;
	double z_high = 10 * ION_HEIGHT;

	// The RF potential rises from the grounded surface to a maximum at the null and then decays
	for (int it = 0; it < 100; it++)
	{
		double z_mid = 0.5 * (z_low + z_high);

		double dphi_dz = 0.0;
		for (int e = 0; e < trap->n_electrodes_rf; e++)
		{
			double p_plus[3] = { 0.0, 0.0, z_mid + h };
			double p_minus[3] = { 0.0, 0.0, z_mid - h };

			double phi_plus;
			double phi_minus;
			evaluate_electrode_potential_analytical(&(*trap->electrodes)[e], (*trap->electrode_positions)[e], p_plus, &phi_plus);
			evaluate_electrode_potential_analytical(&(*trap->electrodes)[e], (*trap->electrode_positions)[e], p_minus, &phi_minus);

			dphi_dz += (phi_plus - phi_minus) / (2 * h);
		}

		if (dphi_dz > 0.0)
			z_low = z_mid;
		else
			z_high = z_mid;
	}

	return 0.5 * (z_low + z_high);
}

// Construct a seed trap design from a preset; the trap must not already hold any allocations
int generate_trap_from_preset
(
	enum TrapPreset trap_preset,
	int n_segments,
	double segment_width,
	struct Trap *trap
)
{
	memset(trap, 0, sizeof(struct Trap));

	if (trap_preset == LINEAR_SURFACE_TRAP || trap_preset == LINEAR_SURFACE_TRAP_WITH_GRATINGS || trap_preset == LINEAR_SURFACE_TRAP_WITH_PHOTONICS)
	{
		// Five-wire surface trap: two RF rails surround a central DC electrode, with segmented DC electrodes outside
		// of the rails. For infinitely long rails in the gapless plane approximation, the RF null lies at a height
		// sqrt(y_center * y_rail), so we start from that and then rescale the transverse dimensions until the RF null
		// of the finite trap lies at ION_HEIGHT
		double y_center = 0.5 * ION_HEIGHT;
		double y_rail = ION_HEIGHT * ION_HEIGHT / y_center;
		double y_segment = y_rail + 2 * ION_HEIGHT;

		// The rails extend beyond the segments to reduce end effects in the region the ions occupy
		double x_segments = 0.5 * n_segments * segment_width;
		double x_rail = x_segments + 2 * ION_HEIGHT;

		allocate_trap(trap, 3 + 2 * n_segments);
		trap->n_electrodes_rf = 2;
		trap->n_electrodes_dc = 1 + 2 * n_segments;

		trap->V_RF = 40.0;
		trap->Omega_RF = 2 * PI * 40E6;

		double y_scale = 1.0;
		for (int it = 0; it < 8; it++)
		{
			// RF electrodes come first
			set_rectangular_electrode(trap, 0, -x_rail, x_rail, y_scale * y_center, y_scale * y_rail);
			set_rectangular_electrode(trap, 1, -x_rail, x_rail, -y_scale * y_rail, -y_scale * y_center);

			// Central DC electrode
			set_rectangular_electrode(trap, 2, -x_rail, x_rail, -y_scale * y_center, y_scale * y_center);

			// Segmented DC electrodes, first along the top and then along the bottom
			for (int s = 0; s < n_segments; s++)
			{
				double x_min = -x_segments + s * segment_width;
				double x_max = x_min + segment_width;

				set_rectangular_electrode(trap, 3 + s, x_min, x_max, y_scale * y_rail, y_scale * y_segment);
				set_rectangular_electrode(trap, 3 + n_segments + s, x_min, x_max, -y_scale * y_segment, -y_scale * y_rail);
			}

			double rf_null_height = compute_rf_null_height_analytical(trap);

			if (fabs(rf_null_height - ION_HEIGHT) < 1E-6)
				break;

			y_scale *= ION_HEIGHT / rf_null_height;
		}

		if (trap_preset == LINEAR_SURFACE_TRAP_WITH_GRATINGS)
		{
			// A focusing silicon nitride grating coupler beneath the central electrode addresses an ion at the trap
			// center with 729 nm light (the 40Ca+ S-D quadrupole transition)
			allocate_grating_couplers(trap, 1);

			struct GratingCoupler *grating = &(*trap->grating_couplers)[0];

			grating->waveguide.wavelength = 0.729;
			grating->waveguide.width = 0.6;
			grating->waveguide.thickness = 0.15;
			grating->waveguide.n_core = 2.0;
			grating->waveguide.n_cladding = 1.455;

			grating->position[0] = -0.75 * ION_HEIGHT;
			grating->position[1] = 0.0;
			grating->position[2] = 0.0;

			grating->fill_factor = 0.5;
			grating->etch_depth = 0.15;
			grating->length = 20.0;
			grating->width = 8.0;
			grating->power = 1E-3;
			grating->efficiency = 0.3;

			double target[3] = { 0.0, 0.0, ION_HEIGHT };
			if (design_grating_coupler(grating, target) != 0)
			{
				printf("ERROR: Failed to design grating coupler for trap preset\n");
				return 1;
			}
		}

		if (trap_preset == LINEAR_SURFACE_TRAP_WITH_PHOTONICS)
		{
			// Gates at the trap center and detection two segments away from it, both at the RF null
			double gate_position[3] = { 0.0, 0.0, ION_HEIGHT };
			double measure_position[3] = { 2 * segment_width, 0.0, ION_HEIGHT };

			if (add_trap_photonics(trap, gate_position, measure_position) != 0)
				return 1;
		}

		return 0;
	}

	printf("ERROR: Unknown trap preset %d\n", trap_preset);

	return 1;
}

// Free the electrodes, grating couplers and detectors of a trap
void free_trap
(
	struct Trap *trap
)
{
	if (trap->electrodes != NULL)
	{
		for (int e = 0; e < trap->n_electrodes; e++)
		{
			free((*trap->electrodes)[e].vertices);
			free((*trap->electrodes)[e].edges);
			free((*trap->electrodes)[e].Vlm);
		}
	}

	free(trap->electrodes);
	free(trap->electrode_positions);
	free(trap->grating_couplers);
	free(trap->detectors);

	trap->electrodes = NULL;
	trap->electrode_positions = NULL;
	trap->grating_couplers = NULL;
	trap->detectors = NULL;
	trap->n_electrodes = 0;
	trap->n_electrodes_rf = 0;
	trap->n_electrodes_dc = 0;
	trap->n_grating_couplers = 0;
	trap->n_detectors = 0;
}
