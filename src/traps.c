#include "traps.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"
#include "photonics.h"
#include "electrodynamics.h"
#include "zones.h"

// Directions in the trap surface from which beams arrive at a zone (rad): diagonal to the trap axis first, so that
// beams have components along the axial mode and across the radial modes, then across and along the axis
#define N_BEAM_AZIMUTHS 8

static const double beam_azimuths[N_BEAM_AZIMUTHS] = { 0.25 * PI, 0.75 * PI, -0.25 * PI, -0.75 * PI, 0.5 * PI, -0.5 * PI, 0.0, PI };

// Direction which each beam of enum PhotonicsBeam takes if it is free, so that the beams of a zone come from different
// quadrants
static const int preferred_azimuths[N_PHOTONICS_BEAMS] = { 0, 1, 2, 3, 0, 3 };

// Vertices of the polygons which approximate circles in the point trap
#define N_CIRCLE_VERTICES 48

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

// Allocate n_grating_couplers grating couplers for a trap, all set to zero and addressing no particular beam or zone
void allocate_grating_couplers
(
	struct Trap *trap,
	int n_grating_couplers
)
{
	trap->n_grating_couplers = n_grating_couplers;

	trap->grating_couplers = calloc(n_grating_couplers, sizeof(struct GratingCoupler));

	for (int g = 0; g < n_grating_couplers; g++)
	{
		(*trap->grating_couplers)[g].beam = -1;
		(*trap->grating_couplers)[g].zone = -1;
	}
}

// Allocate n_detectors detectors for a trap, all set to zero and beneath no particular zone
void allocate_detectors
(
	struct Trap *trap,
	int n_detectors
)
{
	trap->n_detectors = n_detectors;

	trap->detectors = calloc(n_detectors, sizeof(struct Detector));

	for (int d = 0; d < n_detectors; d++)
		(*trap->detectors)[d].zone = -1;
}

// Check whether a zone needs a beam of enum PhotonicsBeam: the qubit and quench beams for gates and qubit reset in gate
// zones, the detection and repump beams for cooling and detection in measurement, cooling and loading zones, and the
// photoionization beams in loading zones
static int zone_needs_beam
(
	enum ZoneType zone_type,
	enum PhotonicsBeam beam
)
{
	if (beam == QUBIT_BEAM || beam == QUENCH_BEAM)
		return (zone_type & GATE_ZONE) != 0;

	if (beam == DETECTION_BEAM || beam == REPUMP_BEAM)
		return (zone_type & (MEASURE_ZONE | COOL_ZONE | LOAD_ZONE)) != 0;

	if (beam == PHOTOIONIZATION_BEAM || beam == IONIZATION_BEAM)
		return (zone_type & LOAD_ZONE) != 0;

	return 0;
}

// Add integrated photonics for a species to every zone of a trap, replacing any photonics it already has: a grating
// coupler focused on the zone from below for every beam its types need (see zone_needs_beam and get_photonics_beam),
// listed in the order of enum PhotonicsBeam and then of the zones, and a detector beneath every measurement zone.
// Returns 1 if the species is unknown or a grating coupler cannot be designed
int add_zone_photonics
(
	struct Trap *trap,
	const char *element
)
{
	struct Waveguide detection_waveguide;
	double detection_power;
	if (get_photonics_beam(element, DETECTION_BEAM, &detection_waveguide, &detection_power) != 0)
	{
		printf("ERROR: No integrated photonics for ion species '%s'\n", (element != NULL) ? element : "(none)");
		return 1;
	}

	free(trap->grating_couplers);
	free(trap->detectors);
	trap->grating_couplers = NULL;
	trap->detectors = NULL;

	int n_grating_couplers = 0;
	int n_detectors = 0;
	for (int z = 0; z < trap->n_zones; z++)
	{
		enum ZoneType zone_type = (*trap->zones)[z].zone_type;

		for (int b = 0; b < N_PHOTONICS_BEAMS; b++)
		{
			struct Waveguide waveguide;
			double power;
			if (zone_needs_beam(zone_type, b) && get_photonics_beam(element, b, &waveguide, &power) == 0)
				n_grating_couplers++;
		}

		if (zone_type & MEASURE_ZONE)
			n_detectors++;
	}

	allocate_grating_couplers(trap, n_grating_couplers);
	allocate_detectors(trap, n_detectors);

	// Directions already taken by the beams of each zone
	int (*taken)[N_BEAM_AZIMUTHS] = calloc(MAX(trap->n_zones, 1), sizeof(int[N_BEAM_AZIMUTHS]));

	// Horizontal distance from each grating to the point beneath the zone it addresses
	double offset = 0.75 * ION_HEIGHT;

	int g = 0;
	for (int b = 0; b < N_PHOTONICS_BEAMS; b++)
	{
		for (int z = 0; z < trap->n_zones; z++)
		{
			struct ZoneNode *zone = &(*trap->zones)[z];

			struct Waveguide waveguide;
			double power;
			if (!zone_needs_beam(zone->zone_type, b) || get_photonics_beam(element, b, &waveguide, &power) != 0)
				continue;

			int slot = preferred_azimuths[b];
			for (int k = 0; k < N_BEAM_AZIMUTHS && taken[z][slot]; k++)
				slot = k;

			taken[z][slot] = 1;

			struct GratingCoupler *grating = &(*trap->grating_couplers)[g];

			grating->waveguide = waveguide;

			grating->position[0] = zone->position[0] - offset * cos(beam_azimuths[slot]);
			grating->position[1] = zone->position[1] - offset * sin(beam_azimuths[slot]);
			grating->position[2] = 0.0;

			grating->fill_factor = 0.5;
			grating->etch_depth = 0.5 * waveguide.thickness;
			grating->length = 20.0;
			grating->width = 8.0;
			grating->power = power;
			grating->efficiency = 0.3;
			grating->beam = b;
			grating->zone = z;

			if (design_grating_coupler(grating, zone->position) != 0)
			{
				printf("ERROR: Failed to design grating coupler %d (beam %d for zone %d) for trap photonics\n", g, b, z);
				free(taken);
				return 1;
			}

			g++;
		}
	}

	free(taken);

	int d = 0;
	for (int z = 0; z < trap->n_zones; z++)
	{
		struct ZoneNode *zone = &(*trap->zones)[z];

		if (!(zone->zone_type & MEASURE_ZONE))
			continue;

		struct Detector *detector = &(*trap->detectors)[d++];

		detector->position[0] = zone->position[0];
		detector->position[1] = zone->position[1];
		detector->position[2] = 0.0;
		detector->width = 50.0;
		detector->length = 50.0;
		detector->wavelength = detection_waveguide.wavelength;
		detector->quantum_efficiency = 0.8;
		detector->dark_count_rate = 10.0;
		detector->zone = z;
	}

	return 0;
}

// Give a trap a gate zone and a measurement zone at the given positions, replacing any zones it already has, and add
// integrated photonics for 40Ca+ to them (see add_zone_photonics): grating couplers for the qubit and quench beams
// focused on the gate zone and for the detection and repump beams focused on the measurement zone, and a detector
// beneath the measurement zone. Returns 1 if a grating coupler cannot be designed
int add_trap_photonics
(
	struct Trap *trap,
	double gate_position[3],
	double measure_position[3]
)
{
	allocate_zones(trap, 2);
	set_trap_zone(trap, 0, GATE_ZONE, default_zone_capacity(GATE_ZONE, 0), gate_position);
	set_trap_zone(trap, 1, MEASURE_ZONE, default_zone_capacity(MEASURE_ZONE, 0), measure_position);

	return add_zone_photonics(trap, "40Ca+");
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

// Define electrode e as a polygon with the given absolute vertices, positioned at the mean of its vertices
static void set_absolute_polygon_electrode
(
	struct Trap *trap,
	int e,
	int n_vertices,
	double (*vertices)[3]
)
{
	double position[3] = { 0.0, 0.0, 0.0 };
	for (int v = 0; v < n_vertices; v++)
		for (int k = 0; k < 3; k++)
			position[k] += vertices[v][k] / n_vertices;

	double (*relative)[3] = malloc(n_vertices * sizeof(double[3]));
	for (int v = 0; v < n_vertices; v++)
		for (int k = 0; k < 3; k++)
			relative[v][k] = vertices[v][k] - position[k];

	set_polygon_electrode(trap, e, n_vertices, relative, position);

	free(relative);
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

// Find the height of the RF null above the point (x, y) of the trap surface, where the derivative of the RF potential
// along z vanishes, using the analytical (gapless plane) potentials of the RF electrodes; assumes that the RF null lies
// directly above the point (as it does above the axis of a symmetric linear trap or of an arm of a junction)
static double compute_rf_null_height_analytical
(
	struct Trap *trap,
	double x,
	double y
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
			double p_plus[3] = { x, y, z_mid + h };
			double p_minus[3] = { x, y, z_mid - h };

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

// Build a five-wire linear surface trap into a trap without electrodes: two RF rails surround a central DC electrode,
// with n_segments segmented DC electrodes on either side of the rails, and the rails reach rail_extension beyond the
// segments at either end
static void build_linear_trap
(
	struct Trap *trap,
	int n_segments,
	double segment_width,
	double rail_extension
)
{
	// For infinitely long rails in the gapless plane approximation, the RF null lies at a height sqrt(y_center *
	// y_rail), so we start from that and then rescale the transverse dimensions until the RF null of the finite trap
	// lies at ION_HEIGHT
	double y_center = 0.5 * ION_HEIGHT;
	double y_rail = ION_HEIGHT * ION_HEIGHT / y_center;
	double y_segment = y_rail + 2 * ION_HEIGHT;

	// The rails extend beyond the segments to reduce end effects in the region the ions occupy
	double x_segments = 0.5 * n_segments * segment_width;
	double x_rail = x_segments + rail_extension;

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

		double rf_null_height = compute_rf_null_height_analytical(trap, 0.0, 0.0);

		if (fabs(rf_null_height - ION_HEIGHT) < 1E-6)
			break;

		y_scale *= ION_HEIGHT / rf_null_height;
	}
}

// Point in the trap surface at a distance t along an arm with unit direction d and a distance u to its left (along d
// turned counterclockwise)
static void arm_point
(
	double t,
	double u,
	double d[2],
	double point[3]
)
{
	point[0] = t * d[0] - u * d[1];
	point[1] = t * d[1] + u * d[0];
	point[2] = 0.0;
}

// Construct a surface trap junction where n_arms five-wire arms meet, pointing along the given angles from the x-axis
// (rad, in any order, with no more than PI between neighboring arms), each with n_segments segmented DC electrodes on
// either side of its RF rails; e.g. an X junction has arms at 0, PI / 2, PI and 3 PI / 2. The rails of neighboring arms
// join into one RF electrode for each gap between the arms, and their DC electrodes meet along the bisector of the gap,
// so the junction has n_arms RF electrodes, a central DC electrode shaped like a star, and 2 n_arms n_segments
// segmented DC electrodes; it gets a junction zone at its center. The trap must not already hold any allocations, and
// it returns 1 if the arms are invalid
int generate_junction_trap
(
	int n_arms,
	double *arm_angles,
	int n_segments,
	double segment_width,
	struct Trap *trap
)
{
	memset(trap, 0, sizeof(struct Trap));

	if (n_arms < 2 || n_segments < 1 || segment_width <= 0.0)
	{
		printf("ERROR: A junction needs at least two arms and one segment of positive width on each side of every arm\n");
		return 1;
	}

	// Arms in counterclockwise order, with the gap from each arm to the next
	double *angles = malloc(n_arms * sizeof(double));
	for (int i = 0; i < n_arms; i++)
		angles[i] = fmod(fmod(arm_angles[i], 2 * PI) + 2 * PI, 2 * PI);

	for (int i = 0; i < n_arms; i++)
		for (int j = i + 1; j < n_arms; j++)
			if (angles[j] < angles[i])
			{
				double swap = angles[i];
				angles[i] = angles[j];
				angles[j] = swap;
			}

	double *gaps = malloc(n_arms * sizeof(double));
	double (*directions)[2] = malloc(n_arms * sizeof(double[2]));
	for (int i = 0; i < n_arms; i++)
	{
		gaps[i] = (i + 1 < n_arms) ? angles[i + 1] - angles[i] : angles[0] + 2 * PI - angles[i];
		directions[i][0] = cos(angles[i]);
		directions[i][1] = sin(angles[i]);

		if (gaps[i] < 1E-3 || gaps[i] > PI + 1E-9)
		{
			printf("ERROR: Neighboring arms of a junction must be between 0 and 180 degrees apart\n");
			free(angles);
			free(gaps);
			free(directions);
			return 1;
		}
	}

	// Transverse dimensions, which are rescaled until the RF null lies at ION_HEIGHT along the arms: the rails lie
	// between a and b from the axis of each arm (for infinitely long rails, the RF null lies at a height sqrt(a b)) and
	// the segmented DC electrodes between b and c, which are narrower than in the linear trap so that the segments
	// start closer to the junction
	double a_0 = 0.6 * ION_HEIGHT;
	double b_0 = ION_HEIGHT * ION_HEIGHT / a_0;
	double c_0 = b_0 + ION_HEIGHT;

	int n_electrodes = n_arms + 1 + 2 * n_arms * n_segments;
	allocate_trap(trap, n_electrodes);
	trap->n_electrodes_rf = n_arms;
	trap->n_electrodes_dc = n_electrodes - n_arms;

	trap->V_RF = 40.0;
	trap->Omega_RF = 2 * PI * 40E6;

	double (*vertices)[3] = malloc(MAX(3 * n_arms, 6) * sizeof(double[3]));

	double scale = 1.0;
	for (int it = 0; it < 8; it++)
	{
		double a = scale * a_0;
		double b = scale * b_0;
		double c = scale * c_0;

		// The electrodes of neighboring arms meet along the bisector of the gap between them, at a distance u
		// cot(gap / 2) along either arm from the junction for a distance u from the arm's axis; the segments of each
		// side start where its outer edge meets the bisector, and the rails and central electrode extend beyond the
		// segments to reduce end effects
		double t_segments = 0.0;
		for (int i = 0; i < n_arms; i++)
			t_segments = MAX(t_segments, c / tan(0.5 * gaps[i]));

		double length = t_segments + n_segments * segment_width + 2 * ION_HEIGHT;

		// RF electrodes, one for the rails on either side of each gap
		for (int i = 0; i < n_arms; i++)
		{
			int next = (i + 1) % n_arms;
			double cot_half_gap = 1.0 / tan(0.5 * gaps[i]);

			arm_point(a * cot_half_gap, a, directions[i], vertices[0]);
			arm_point(length, a, directions[i], vertices[1]);
			arm_point(length, b, directions[i], vertices[2]);
			arm_point(b * cot_half_gap, b, directions[i], vertices[3]);
			arm_point(length, -b, directions[next], vertices[4]);
			arm_point(length, -a, directions[next], vertices[5]);

			set_absolute_polygon_electrode(trap, i, 6, vertices);
		}

		// Central DC electrode, which follows the axis of every arm and meets the rails at the inner corners of the
		// gaps
		for (int i = 0; i < n_arms; i++)
		{
			arm_point(length, -a, directions[i], vertices[3 * i]);
			arm_point(length, a, directions[i], vertices[3 * i + 1]);
			arm_point(a / tan(0.5 * gaps[i]), a, directions[i], vertices[3 * i + 2]);
		}

		set_absolute_polygon_electrode(trap, n_arms, 3 * n_arms, vertices);

		// Segmented DC electrodes of each arm, first on its left (counterclockwise) side and then on its right; the
		// first segment of each side reaches back to the bisector of its gap
		for (int i = 0; i < n_arms; i++)
		{
			for (int side = 0; side < 2; side++)
			{
				double sign = (side == 0) ? 1.0 : -1.0;
				double gap = (side == 0) ? gaps[i] : gaps[(i + n_arms - 1) % n_arms];

				double t_b = b / tan(0.5 * gap);
				double t_c = c / tan(0.5 * gap);

				for (int s = 0; s < n_segments; s++)
				{
					int e = n_arms + 1 + (2 * i + side) * n_segments + s;

					double t_min = t_c + s * segment_width;
					double t_max = t_min + segment_width;

					arm_point((s == 0) ? t_b : t_min, sign * b, directions[i], vertices[0]);
					arm_point(t_max, sign * b, directions[i], vertices[1]);
					arm_point(t_max, sign * c, directions[i], vertices[2]);
					arm_point(t_min, sign * c, directions[i], vertices[3]);

					set_absolute_polygon_electrode(trap, e, 4, vertices);
				}
			}
		}

		// The RF null along the middle of the first arm's segments
		double t_null = t_segments + 0.5 * n_segments * segment_width;
		double rf_null_height = compute_rf_null_height_analytical(trap, t_null * directions[0][0], t_null * directions[0][1]);

		if (fabs(rf_null_height - ION_HEIGHT) < 1E-6)
			break;

		scale *= ION_HEIGHT / rf_null_height;
	}

	free(vertices);
	free(angles);
	free(gaps);
	free(directions);

	double center[3] = { 0.0, 0.0, ION_HEIGHT };
	allocate_zones(trap, 1);
	set_trap_zone(trap, 0, JUNCTION_ZONE, default_zone_capacity(JUNCTION_ZONE, 0), center);

	return 0;
}

// Build a point trap into a trap without electrodes: a ring of RF around a central DC electrode, which confines an ion
// in all three directions above its center, surrounded by n_segments DC electrodes in sectors of an annulus
// segment_width wide which compensate stray fields; every electrode is approximated by a polygon
static void build_point_trap
(
	struct Trap *trap,
	int n_segments,
	double segment_width
)
{
	allocate_trap(trap, 2 + n_segments);
	trap->n_electrodes_rf = 1;
	trap->n_electrodes_dc = 1 + n_segments;

	trap->V_RF = 40.0;
	trap->Omega_RF = 2 * PI * 40E6;

	int n_sector_vertices = N_CIRCLE_VERTICES / MAX(n_segments, 1) + 1;
	double (*vertices)[3] = malloc(MAX(2 * N_CIRCLE_VERTICES + 2, 2 * n_sector_vertices) * sizeof(double[3]));

	// In the gapless plane approximation, the RF null of a ring lies at a height proportional to the ring's size, so
	// one rescaling puts it at ION_HEIGHT; the loop checks it
	double r_inner = 0.5 * ION_HEIGHT;
	double r_outer = 3 * r_inner;
	for (int it = 0; it < 8; it++)
	{
		// The ring is one polygon which runs around its outer edge, crosses to its inner edge and runs back around
		// the other way, so that the solid angle of the hole is subtracted
		for (int v = 0; v <= N_CIRCLE_VERTICES; v++)
		{
			double phi = 2 * PI * v / N_CIRCLE_VERTICES;

			vertices[v][0] = r_outer * cos(phi);
			vertices[v][1] = r_outer * sin(phi);
			vertices[v][2] = 0.0;

			vertices[2 * N_CIRCLE_VERTICES + 1 - v][0] = r_inner * cos(phi);
			vertices[2 * N_CIRCLE_VERTICES + 1 - v][1] = r_inner * sin(phi);
			vertices[2 * N_CIRCLE_VERTICES + 1 - v][2] = 0.0;
		}

		double origin[3] = { 0.0, 0.0, 0.0 };
		set_polygon_electrode(trap, 0, 2 * N_CIRCLE_VERTICES + 2, vertices, origin);

		// Central DC electrode
		for (int v = 0; v < N_CIRCLE_VERTICES; v++)
		{
			double phi = 2 * PI * v / N_CIRCLE_VERTICES;

			vertices[v][0] = r_inner * cos(phi);
			vertices[v][1] = r_inner * sin(phi);
			vertices[v][2] = 0.0;
		}

		set_polygon_electrode(trap, 1, N_CIRCLE_VERTICES, vertices, origin);

		// Compensation electrodes around the ring
		for (int s = 0; s < n_segments; s++)
		{
			for (int v = 0; v < n_sector_vertices; v++)
			{
				double phi = 2 * PI * (s + (double) v / (n_sector_vertices - 1)) / n_segments;

				vertices[v][0] = (r_outer + segment_width) * cos(phi);
				vertices[v][1] = (r_outer + segment_width) * sin(phi);
				vertices[v][2] = 0.0;

				vertices[2 * n_sector_vertices - 1 - v][0] = r_outer * cos(phi);
				vertices[2 * n_sector_vertices - 1 - v][1] = r_outer * sin(phi);
				vertices[2 * n_sector_vertices - 1 - v][2] = 0.0;
			}

			set_absolute_polygon_electrode(trap, 2 + s, 2 * n_sector_vertices, vertices);
		}

		double rf_null_height = compute_rf_null_height_analytical(trap, 0.0, 0.0);

		if (fabs(rf_null_height - ION_HEIGHT) < 1E-6)
			break;

		r_inner *= ION_HEIGHT / rf_null_height;
		r_outer *= ION_HEIGHT / rf_null_height;
	}

	free(vertices);

	double center[3] = { 0.0, 0.0, ION_HEIGHT };
	allocate_zones(trap, 1);
	set_trap_zone(trap, 0, GATE_ZONE | MEASURE_ZONE, 1, center);
}

// Construct a seed trap design from a preset; the trap must not already hold any allocations. The linear presets have
// n_segments segmented DC electrodes of segment_width on each side, the junction presets have them on each side of
// every arm (see generate_junction_trap), and the point trap has n_segments compensation electrodes segment_width wide
// around its ring (see build_point_trap)
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
		build_linear_trap(trap, n_segments, segment_width, 2 * ION_HEIGHT);

		if (trap_preset == LINEAR_SURFACE_TRAP_WITH_GRATINGS)
		{
			// A focusing silicon nitride grating coupler beneath the central electrode addresses a gate zone at the
			// trap center with 729 nm light (the 40Ca+ S-D quadrupole transition)
			double center[3] = { 0.0, 0.0, ION_HEIGHT };
			allocate_zones(trap, 1);
			set_trap_zone(trap, 0, GATE_ZONE, default_zone_capacity(GATE_ZONE, 0), center);

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
			grating->beam = QUBIT_BEAM;
			grating->zone = 0;

			if (design_grating_coupler(grating, center) != 0)
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

	if (trap_preset == X_JUNCTION_TRAP)
	{
		double arm_angles[4] = { 0.0, 0.5 * PI, PI, 1.5 * PI };
		return generate_junction_trap(4, arm_angles, n_segments, segment_width, trap);
	}

	if (trap_preset == Y_JUNCTION_TRAP)
	{
		double arm_angles[3] = { 0.0, 2 * PI / 3, 4 * PI / 3 };
		return generate_junction_trap(3, arm_angles, n_segments, segment_width, trap);
	}

	if (trap_preset == T_JUNCTION_TRAP)
	{
		double arm_angles[3] = { 0.0, 0.5 * PI, PI };
		return generate_junction_trap(3, arm_angles, n_segments, segment_width, trap);
	}

	if (trap_preset == POINT_TRAP)
	{
		build_point_trap(trap, n_segments, segment_width);
		return 0;
	}

	printf("ERROR: Unknown trap preset %d\n", trap_preset);

	return 1;
}

// Construct a five-wire linear surface trap for a zone layout (see parse_zone_layout), whose zones are spaced evenly by
// zone_spacing (µm) along the trap axis and centered on the trap, with enough DC segments of segment_width to make
// wells beyond the outermost zones, rails which reach far enough beyond them to keep the RF null on the axis, and with integrated photonics for the species in element addressing every zone if
// photonics is set (see add_zone_photonics; element may be NULL otherwise). Zones without a capacity in the layout hold
// n_ions ions, or a default number if n_ions is 0. The trap must not already hold any allocations, and it returns 1 if
// the layout cannot be parsed or the photonics cannot be designed
int generate_trap_from_layout
(
	const char *layout,
	int n_ions,
	double zone_spacing,
	double segment_width,
	const char *element,
	int photonics,
	struct Trap *trap
)
{
	memset(trap, 0, sizeof(struct Trap));

	if (set_layout_zones(trap, layout, n_ions, zone_spacing) != 0)
		return 1;

	double span = (trap->n_zones - 1) * zone_spacing;
	int n_segments = (int) ceil(span / segment_width - 1E-9) + 2;

	// Near the ends of the rails, the RF null rises off the axis and the ions feel the RF, so the rails reach five ion
	// heights beyond the outermost zones (or two beyond the segments, if that is further)
	double rail_extension = MAX(2 * ION_HEIGHT, 0.5 * span + 5 * ION_HEIGHT - 0.5 * n_segments * segment_width);

	build_linear_trap(trap, n_segments, segment_width, rail_extension);

	if (photonics && add_zone_photonics(trap, element) != 0)
		return 1;

	return 0;
}

// Free the electrodes, grating couplers, detectors and zones of a trap
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
	free(trap->zones);

	trap->electrodes = NULL;
	trap->electrode_positions = NULL;
	trap->grating_couplers = NULL;
	trap->detectors = NULL;
	trap->zones = NULL;
	trap->n_electrodes = 0;
	trap->n_electrodes_rf = 0;
	trap->n_electrodes_dc = 0;
	trap->n_grating_couplers = 0;
	trap->n_detectors = 0;
	trap->n_zones = 0;
}
