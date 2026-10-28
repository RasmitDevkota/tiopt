#pragma once

#include "data_structures.h"

enum Polarization
{
	TE,
	TM,
};

int solve_slab_mode
(
	double n_core,
	double n_cover,
	double n_substrate,
	double thickness,
	double wavelength,
	enum Polarization polarization,
	int mode_order,
	double *n_eff
);

int solve_waveguide_mode
(
	struct Waveguide *waveguide
);

int compute_grating_effective_index
(
	struct GratingCoupler *grating,
	double *n_eff_grating
);

int compute_grating_emission
(
	struct GratingCoupler *grating,
	double *theta,
	double (*k_hat)[3]
);

int design_grating_coupler
(
	struct GratingCoupler *grating,
	double target[3]
);

void compute_gaussian_beam
(
	double waist_aperture,
	double focal_distance,
	double wavelength,
	double *waist,
	double *waist_distance
);

int evaluate_grating_beam
(
	struct GratingCoupler *grating,
	double position[3],
	double *intensity,
	double (*k_vector)[3]
);

double compute_collection_efficiency
(
	struct Detector *detector,
	double position[3]
);

double compute_scattering_rate
(
	const double gamma,
	const double s,
	const double detuning
);

double compute_detection_fidelity
(
	const double photon_rate,
	const double dark_count_rate,
	const double decay_rate,
	const double detection_time,
	int *threshold
);

int compute_multilayer_response
(
	int n_layers,
	double *n_real,
	double *n_imag,
	double *thicknesses,
	double wavelength,
	double theta_incident,
	enum Polarization polarization,
	double (*r)[2],
	double (*t)[2],
	double *R,
	double *T
);

double compute_grating_directionality
(
	double n_box,
	double box_thickness,
	double n_substrate_real,
	double n_substrate_imag,
	double wavelength,
	double theta_box,
	enum Polarization polarization
);

int fit_polynomial
(
	int n_points,
	double *x,
	double *y,
	int degree,
	double *coefficients,
	double *chi_sq
);

int fit_gaussian
(
	int n_points,
	double *x,
	double *y,
	double (*parameters)[4]
);

int fit_waveguide_dispersion
(
	struct Waveguide *waveguide,
	double wavelength_min,
	double wavelength_max,
	int n_samples,
	int degree,
	double *coefficients,
	double *n_group
);
