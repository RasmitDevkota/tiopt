#include "photonics.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>

#include <gsl/gsl_multifit.h>
#include <gsl/gsl_multifit_nlinear.h>

#include "defs.h"
#include "data_structures.h"
#include "ti_utils.h"
#include "electrodynamics.h"

// Dispersion relation of an asymmetric three-layer slab waveguide, which vanishes at the effective index of each guided
// mode and decreases monotonically with the effective index n_eff
static double slab_dispersion
(
	double n_eff,
	double n_core,
	double n_cover,
	double n_substrate,
	double thickness,
	double wavelength,
	enum Polarization polarization,
	int mode_order
)
{
	double k_0 = 2 * PI / wavelength;

	double kappa = k_0 * sqrt(MAX(n_core*n_core - n_eff*n_eff, 0.0));
	double gamma_cover = k_0 * sqrt(MAX(n_eff*n_eff - n_cover*n_cover, 0.0));
	double gamma_substrate = k_0 * sqrt(MAX(n_eff*n_eff - n_substrate*n_substrate, 0.0));

	// TM modes see the boundary conditions on the normal component of D
	if (polarization == TM)
	{
		gamma_cover *= pow(n_core / n_cover, 2);
		gamma_substrate *= pow(n_core / n_substrate, 2);
	}

	return kappa * thickness - mode_order * PI - atan2(gamma_cover, kappa) - atan2(gamma_substrate, kappa);
}

// Solve for the effective index of a guided mode of an asymmetric slab waveguide by bisection; returns 1 if the mode
// is below cutoff
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
)
{
	double n_low = MAX(n_cover, n_substrate);
	double n_high = n_core;

	if (n_high <= n_low || thickness <= 0.0 || wavelength <= 0.0)
		return 1;

	if (slab_dispersion(n_low, n_core, n_cover, n_substrate, thickness, wavelength, polarization, mode_order) <= 0.0)
		return 1;

	for (int it = 0; it < 200 && n_high - n_low > 1E-15; it++)
	{
		double n_mid = 0.5 * (n_low + n_high);

		if (slab_dispersion(n_mid, n_core, n_cover, n_substrate, thickness, wavelength, polarization, mode_order) > 0.0)
			n_low = n_mid;
		else
			n_high = n_mid;
	}

	*n_eff = 0.5 * (n_low + n_high);

	return 0;
}

// Solve for the effective index of the fundamental quasi-TE mode of a rectangular channel waveguide surrounded by
// cladding, using the effective index method
int solve_waveguide_mode
(
	struct Waveguide *waveguide
)
{
	// Vertical confinement: a slab of the core thickness, with the electric field in the plane of the slab
	double n_vertical;
	if (
		solve_slab_mode(
			waveguide->n_core, waveguide->n_cladding, waveguide->n_cladding,
			waveguide->thickness,
			waveguide->wavelength,
			TE,
			0,
			&n_vertical
		) != 0
	)
		return 1;

	// Lateral confinement: a slab of the core width whose core index is the vertical effective index, with the
	// electric field now normal to its interfaces
	return solve_slab_mode(
		n_vertical, waveguide->n_cladding, waveguide->n_cladding,
		waveguide->width,
		waveguide->wavelength,
		TM,
		0,
		&waveguide->n_eff
	);
}

// Compute the effective index of the grating region, treating the unetched and etched sections as slabs (the grating
// is much wider than the wavelength) and averaging them by the fill factor
int compute_grating_effective_index
(
	struct GratingCoupler *grating,
	double *n_eff_grating
)
{
	struct Waveguide *waveguide = &grating->waveguide;

	double n_unetched;
	if (
		solve_slab_mode(
			waveguide->n_core, waveguide->n_cladding, waveguide->n_cladding,
			waveguide->thickness,
			waveguide->wavelength,
			TE,
			0,
			&n_unetched
		) != 0
	)
		return 1;

	// A fully etched section is just cladding
	double n_etched = waveguide->n_cladding;
	double remaining_thickness = waveguide->thickness - grating->etch_depth;
	if (
		remaining_thickness > 1E-9 &&
		solve_slab_mode(
			waveguide->n_core, waveguide->n_cladding, waveguide->n_cladding,
			remaining_thickness,
			waveguide->wavelength,
			TE,
			0,
			&n_etched
		) != 0
	)
		return 1;

	*n_eff_grating = grating->fill_factor * n_unetched + (1 - grating->fill_factor) * n_etched;

	return 0;
}

// Compute the emission angle of a grating coupler into the vacuum above the trap, measured from the surface normal
// towards the direction of propagation of the guided mode, using the first-order phase matching condition
// sin(theta) = n_eff - wavelength / period (the cladding above the grating only displaces the beam); returns 1 if no
// diffraction order radiates
int compute_grating_emission
(
	struct GratingCoupler *grating,
	double *theta,
	double (*k_hat)[3]
)
{
	double n_eff;
	if (compute_grating_effective_index(grating, &n_eff) != 0 || grating->period <= 0.0)
		return 1;

	double sin_theta = n_eff - grating->waveguide.wavelength / grating->period;

	if (fabs(sin_theta) >= 1.0)
		return 1;

	*theta = asin(sin_theta);

	if (k_hat != NULL)
	{
		(*k_hat)[0] = sin_theta * cos(grating->azimuth);
		(*k_hat)[1] = sin_theta * sin(grating->azimuth);
		(*k_hat)[2] = cos(*theta);
	}

	return 0;
}

// Choose the azimuth, period and focal distance of a grating coupler such that it focuses onto a target point
int design_grating_coupler
(
	struct GratingCoupler *grating,
	double target[3]
)
{
	double d[3];
	for (int k = 0; k < 3; k++)
		d[k] = target[k] - grating->position[k];

	if (d[2] <= 0.0)
	{
		printf("ERROR: Grating couplers can only address points above the trap surface\n");
		return 1;
	}

	double rho = sqrt(d[0]*d[0] + d[1]*d[1]);

	grating->azimuth = (rho > 0.0) ? atan2(d[1], d[0]) : 0.0;
	grating->focal_distance = sqrt(rho*rho + d[2]*d[2]);

	double n_eff;
	if (compute_grating_effective_index(grating, &n_eff) != 0)
	{
		printf("ERROR: Waveguide in grating coupler does not support a guided mode\n");
		return 1;
	}

	double sin_theta = rho / grating->focal_distance;

	grating->period = grating->waveguide.wavelength / (n_eff - sin_theta);

	return 0;
}

// Compute the waist (and its distance from the aperture) of a Gaussian beam with the given waist at an aperture which
// focuses it with the given focal distance (S. A. Self, Appl. Opt. 22, 658 (1983)); a focal distance of 0 leaves the
// beam unfocused
void compute_gaussian_beam
(
	double waist_aperture,
	double focal_distance,
	double wavelength,
	double *waist,
	double *waist_distance
)
{
	if (focal_distance <= 0.0)
	{
		*waist = waist_aperture;
		*waist_distance = 0.0;
		return;
	}

	double z_R = PI * waist_aperture * waist_aperture / wavelength;

	*waist = waist_aperture / sqrt(1 + pow(z_R / focal_distance, 2));
	*waist_distance = focal_distance / (1 + pow(focal_distance / z_R, 2));
}

// Evaluate the intensity (W/m^2) and wavevector (1/m) of the beam emitted by a grating coupler at a position in the
// trap frame, modelling the beam as an elliptical Gaussian beam launched from the grating aperture
int evaluate_grating_beam
(
	struct GratingCoupler *grating,
	double position[3],
	double *intensity,
	double (*k_vector)[3]
)
{
	*intensity = 0.0;

	double theta;
	double k_hat[3];
	if (compute_grating_emission(grating, &theta, &k_hat) != 0)
		return 1;

	// Beam frame: the beam axis, the direction in the trap surface transverse to the grating, and a third direction
	double e_perp[3] = { -sin(grating->azimuth), cos(grating->azimuth), 0.0 };
	double e_par[3];
	cross(&e_perp, &k_hat, &e_par);

	double wavelength = grating->waveguide.wavelength;

	// The emitting aperture is the grating projected onto the plane transverse to the beam
	double waist_aperture_par = 0.5 * grating->length * cos(theta);
	double waist_aperture_perp = 0.5 * grating->width;

	double waist_par;
	double waist_perp;
	double waist_distance_par;
	double waist_distance_perp;
	compute_gaussian_beam(waist_aperture_par, grating->focal_distance, wavelength, &waist_par, &waist_distance_par);
	compute_gaussian_beam(waist_aperture_perp, grating->focal_distance, wavelength, &waist_perp, &waist_distance_perp);

	double d[3];
	for (int k = 0; k < 3; k++)
		d[k] = position[k] - grating->position[k];

	double z;
	double rho_par;
	double rho_perp;
	dot(3, &d, &k_hat, &z);
	dot(3, &d, &e_par, &rho_par);
	dot(3, &d, &e_perp, &rho_perp);

	double z_R_par = PI * waist_par * waist_par / wavelength;
	double z_R_perp = PI * waist_perp * waist_perp / wavelength;
	double w_par = waist_par * sqrt(1 + pow((z - waist_distance_par) / z_R_par, 2));
	double w_perp = waist_perp * sqrt(1 + pow((z - waist_distance_perp) / z_R_perp, 2));

	double power = grating->power * grating->efficiency;

	// Intensity in W/µm^2, converted to W/m^2
	*intensity = 2 * power / (PI * w_par * w_perp) * exp(-2 * pow(rho_par / w_par, 2) - 2 * pow(rho_perp / w_perp, 2)) / (LENGTH_SCALE * LENGTH_SCALE);

	if (k_vector != NULL)
		for (int k = 0; k < 3; k++)
			(*k_vector)[k] = 2 * PI / (wavelength * LENGTH_SCALE) * k_hat[k];

	return 0;
}

// Fraction of the photons emitted by an ion at a position (µm) which are counted by a detector in the trap surface:
// the solid angle subtended by its active area divided by 4 PI (for isotropic emission, neglecting the dipole pattern
// of the transition and reflections from the surface) times its quantum efficiency
double compute_collection_efficiency
(
	struct Detector *detector,
	double position[3]
)
{
	double half_width = 0.5 * detector->width;
	double half_length = 0.5 * detector->length;

	double vertices[4][3] =
	{
		{ half_width, half_length, 0.0 },
		{ -half_width, half_length, 0.0 },
		{ -half_width, -half_length, 0.0 },
		{ half_width, -half_length, 0.0 }
	};

	struct Electrode active_area = { 0 };
	active_area.n_vertices = 4;
	active_area.vertices = vertices;

	// The potential of an electrode in the gapless plane approximation is the solid angle it subtends divided by 2 PI
	double phi;
	evaluate_electrode_potential_analytical(&active_area, detector->position, position, &phi);

	return detector->quantum_efficiency * 0.5 * phi;
}

// Rate (1/s) at which a two-level transition of linewidth gamma (1/s) scatters photons when driven with saturation
// parameter s = I / I_sat and the given detuning (rad/s)
double compute_scattering_rate
(
	const double gamma,
	const double s,
	const double detuning
)
{
	return 0.5 * gamma * s / (1 + s + 4 * detuning*detuning / (gamma*gamma));
}

// Fidelity of state detection by thresholding photon counts, 1 - (P(dark | bright) + P(bright | dark)) / 2, where an ion
// in the bright state gives counts at photon_rate + dark_count_rate (1/s) and an ion in the dark state gives counts at
// dark_count_rate until it decays to the bright state at decay_rate (1/s), for a detection time (s). The threshold
// (the number of counts at or above which the ion is declared bright) is chosen to maximize the fidelity, and is
// returned through threshold if it is not NULL
double compute_detection_fidelity
(
	const double photon_rate,
	const double dark_count_rate,
	const double decay_rate,
	const double detection_time,
	int *threshold
)
{
	// The dark state decays at a time distributed exponentially over the detection, after which it counts like the
	// bright state; the decay time is integrated over by the midpoint rule, with the probability of no decay last
	const int n_nodes = 64;
	double weights[n_nodes + 1];
	double means[n_nodes + 1];

	double p_decay = 1 - exp(-decay_rate * detection_time);
	for (int j = 0; j < n_nodes; j++)
	{
		// Decay times at the midpoints of n_nodes intervals of equal probability
		double u = (j + 0.5) / n_nodes;
		double t_decay = (decay_rate > 0.0) ? -log(1 - u * p_decay) / decay_rate : detection_time;

		weights[j] = p_decay / n_nodes;
		means[j] = dark_count_rate * detection_time + photon_rate * (detection_time - t_decay);
	}
	weights[n_nodes] = 1 - p_decay;
	means[n_nodes] = dark_count_rate * detection_time;

	double mean_bright = (photon_rate + dark_count_rate) * detection_time;

	// Poisson probabilities of n counts (kept as logarithms, since they underflow for large means), and the cumulative
	// probabilities of fewer than n counts, built up one count at a time
	double log_pmf_bright = -mean_bright;
	double cdf_bright = 0.0;
	double log_pmf_dark[n_nodes + 1];
	double cdf_dark[n_nodes + 1];
	for (int j = 0; j <= n_nodes; j++)
	{
		log_pmf_dark[j] = -means[j];
		cdf_dark[j] = 0.0;
	}

	double best_fidelity = 0.5;
	int best_threshold = 1;

	// Thresholds above the mean bright count misidentify the bright state more often than not
	int max_threshold = (int) ceil(mean_bright) + 1;
	for (int n = 1; n <= max_threshold; n++)
	{
		// Move the probability of n - 1 counts into the cumulative probabilities
		cdf_bright += exp(log_pmf_bright);
		log_pmf_bright += log(mean_bright) - log(n);

		double p_bright_given_dark = 0.0;
		for (int j = 0; j <= n_nodes; j++)
		{
			cdf_dark[j] += exp(log_pmf_dark[j]);
			log_pmf_dark[j] += log(means[j]) - log(n);

			p_bright_given_dark += weights[j] * MAX(1 - cdf_dark[j], 0.0);
		}

		double fidelity = 1 - 0.5 * (MIN(cdf_bright, 1.0) + p_bright_given_dark);

		if (fidelity > best_fidelity)
		{
			best_fidelity = fidelity;
			best_threshold = n;
		}
	}

	if (threshold != NULL)
		*threshold = best_threshold;

	return best_fidelity;
}

// Compute the complex reflection and transmission coefficients, and the reflectance and transmittance, of a stack of
// layers using the characteristic matrix method; layer 0 is the (lossless) incident medium, layer n_layers-1 is the
// substrate, and thicknesses are only used for the layers in between (all lengths in µm, angles in rad)
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
)
{
	if (n_layers < 2)
		return 1;

	// The tangential component of the wavevector, n sin(theta), is conserved across the stack
	double complex n_sin_theta = n_real[0] * sin(theta_incident);

	double complex eta[n_layers];
	double complex q[n_layers];
	for (int j = 0; j < n_layers; j++)
	{
		double complex n = n_real[j] + I * n_imag[j];

		// q = n cos(theta), choosing the branch which decays (or is absorbed) away from the incident medium
		q[j] = csqrt(n*n - n_sin_theta*n_sin_theta);
		if (cimag(q[j]) < 0.0 || (cimag(q[j]) == 0.0 && creal(q[j]) < 0.0))
			q[j] = -q[j];

		// Tilted optical admittances in units of the admittance of free space
		eta[j] = (polarization == TE) ? q[j] : n*n / q[j];
	}

	// Characteristic matrix of the films, using the e^(i (k z - omega t)) convention
	double complex M[2][2] = { { 1.0, 0.0 }, { 0.0, 1.0 } };
	for (int j = 1; j < n_layers - 1; j++)
	{
		double complex delta = 2 * PI * q[j] * thicknesses[j] / wavelength;

		double complex M_j[2][2] =
		{
			{ ccos(delta), -I * csin(delta) / eta[j] },
			{ -I * eta[j] * csin(delta), ccos(delta) }
		};

		double complex M_new[2][2];
		for (int a = 0; a < 2; a++)
			for (int b = 0; b < 2; b++)
				M_new[a][b] = M[a][0] * M_j[0][b] + M[a][1] * M_j[1][b];

		for (int a = 0; a < 2; a++)
			for (int b = 0; b < 2; b++)
				M[a][b] = M_new[a][b];
	}

	double complex eta_0 = eta[0];
	double complex eta_s = eta[n_layers - 1];

	double complex B = M[0][0] + M[0][1] * eta_s;
	double complex C = M[1][0] + M[1][1] * eta_s;

	double complex r_complex = (eta_0 * B - C) / (eta_0 * B + C);
	double complex t_complex = 2 * eta_0 / (eta_0 * B + C);

	if (r != NULL)
	{
		(*r)[0] = creal(r_complex);
		(*r)[1] = cimag(r_complex);
	}

	if (t != NULL)
	{
		(*t)[0] = creal(t_complex);
		(*t)[1] = cimag(t_complex);
	}

	*R = pow(cabs(r_complex), 2);
	*T = creal(eta_s) / creal(eta_0) * pow(cabs(t_complex), 2);

	return 0;
}

// Estimate the fraction of the power radiated by a grating which leaves upwards, accounting for interference between
// the upward emission and the downward emission reflected by the substrate beneath a buried oxide of the given
// thickness (µm); theta_box is the emission angle inside the oxide
double compute_grating_directionality
(
	double n_box,
	double box_thickness,
	double n_substrate_real,
	double n_substrate_imag,
	double wavelength,
	double theta_box,
	enum Polarization polarization
)
{
	double n_real[2] = { n_box, n_substrate_real };
	double n_imag[2] = { 0.0, n_substrate_imag };
	double thicknesses[2] = { 0.0, 0.0 };

	double r[2];
	double t[2];
	double R;
	double T;
	compute_multilayer_response(2, n_real, n_imag, thicknesses, wavelength, theta_box, polarization, &r, &t, &R, &T);

	// Phase accumulated by the reflected beam on its round trip through the oxide
	double delta = 4 * PI * n_box * box_thickness * cos(theta_box) / wavelength;

	double complex upward = 1.0 + (r[0] + I * r[1]) * cexp(I * delta);

	// The grating radiates equally up and down, and the power not reflected by the substrate is lost into it
	return pow(cabs(upward), 2) / (pow(cabs(upward), 2) + (1 - R));
}

// Fit y = sum_k coefficients[k] x^k by linear least squares
int fit_polynomial
(
	int n_points,
	double *x,
	double *y,
	int degree,
	double *coefficients,
	double *chi_sq
)
{
	int n_coefficients = degree + 1;

	if (n_points < n_coefficients)
		return 1;

	gsl_matrix *X = gsl_matrix_alloc(n_points, n_coefficients);
	gsl_vector *Y = gsl_vector_alloc(n_points);
	gsl_vector *c = gsl_vector_alloc(n_coefficients);
	gsl_matrix *covariance = gsl_matrix_alloc(n_coefficients, n_coefficients);

	for (int i = 0; i < n_points; i++)
	{
		gsl_vector_set(Y, i, y[i]);

		for (int k = 0; k < n_coefficients; k++)
			gsl_matrix_set(X, i, k, pow(x[i], k));
	}

	gsl_multifit_linear_workspace *workspace = gsl_multifit_linear_alloc(n_points, n_coefficients);

	double chi_sq_fit;
	int status = gsl_multifit_linear(X, Y, c, covariance, &chi_sq_fit, workspace);

	for (int k = 0; k < n_coefficients; k++)
		coefficients[k] = gsl_vector_get(c, k);

	if (chi_sq != NULL)
		*chi_sq = chi_sq_fit;

	gsl_multifit_linear_free(workspace);
	gsl_matrix_free(covariance);
	gsl_vector_free(c);
	gsl_vector_free(Y);
	gsl_matrix_free(X);

	return status;
}

struct GaussianFitData
{
	int n_points;
	double *x;
	double *y;
};

static int gaussian_residuals
(
	const gsl_vector *p,
	void *data,
	gsl_vector *f
)
{
	struct GaussianFitData *d = data;

	double amplitude = gsl_vector_get(p, 0);
	double center = gsl_vector_get(p, 1);
	double sigma = gsl_vector_get(p, 2);
	double offset = gsl_vector_get(p, 3);

	for (int i = 0; i < d->n_points; i++)
		gsl_vector_set(f, i, amplitude * exp(-0.5 * pow((d->x[i] - center) / sigma, 2)) + offset - d->y[i]);

	return GSL_SUCCESS;
}

static int gaussian_jacobian
(
	const gsl_vector *p,
	void *data,
	gsl_matrix *J
)
{
	struct GaussianFitData *d = data;

	double amplitude = gsl_vector_get(p, 0);
	double center = gsl_vector_get(p, 1);
	double sigma = gsl_vector_get(p, 2);

	for (int i = 0; i < d->n_points; i++)
	{
		double u = (d->x[i] - center) / sigma;
		double g = exp(-0.5 * u*u);

		gsl_matrix_set(J, i, 0, g);
		gsl_matrix_set(J, i, 1, amplitude * g * u / sigma);
		gsl_matrix_set(J, i, 2, amplitude * g * u*u / sigma);
		gsl_matrix_set(J, i, 3, 1.0);
	}

	return GSL_SUCCESS;
}

// Fit y = amplitude * exp(-(x - center)^2 / (2 sigma^2)) + offset by nonlinear least squares, returning the parameters
// in the order (amplitude, center, sigma, offset)
int fit_gaussian
(
	int n_points,
	double *x,
	double *y,
	double (*parameters)[4]
)
{
	if (n_points < 4)
		return 1;

	// Initial guess from the extremes and the second moment of the data
	int i_max = 0;
	double y_min = y[0];
	for (int i = 0; i < n_points; i++)
	{
		if (y[i] > y[i_max])
			i_max = i;

		y_min = MIN(y[i], y_min);
	}

	double weight_sum = 0.0;
	double moment = 0.0;
	for (int i = 0; i < n_points; i++)
	{
		weight_sum += y[i] - y_min;
		moment += (y[i] - y_min) * pow(x[i] - x[i_max], 2);
	}

	double sigma_guess = (weight_sum > 0.0) ? sqrt(moment / weight_sum) : fabs(x[n_points-1] - x[0]) / 4;
	if (sigma_guess <= 0.0)
		sigma_guess = 1.0;

	double p_initial[4] = { y[i_max] - y_min, x[i_max], sigma_guess, y_min };

	struct GaussianFitData data = { n_points, x, y };

	gsl_multifit_nlinear_fdf fdf;
	fdf.f = gaussian_residuals;
	fdf.df = gaussian_jacobian;
	fdf.fvv = NULL;
	fdf.n = n_points;
	fdf.p = 4;
	fdf.params = &data;

	gsl_multifit_nlinear_parameters fdf_parameters = gsl_multifit_nlinear_default_parameters();
	gsl_multifit_nlinear_workspace *workspace = gsl_multifit_nlinear_alloc(gsl_multifit_nlinear_trust, &fdf_parameters, n_points, 4);

	gsl_vector_view p_view = gsl_vector_view_array(p_initial, 4);
	gsl_multifit_nlinear_init(&p_view.vector, &fdf, workspace);

	int info;
	int status = gsl_multifit_nlinear_driver(200, 1E-12, 1E-12, 1E-12, NULL, NULL, &info, workspace);

	gsl_vector *p = gsl_multifit_nlinear_position(workspace);
	for (int k = 0; k < 4; k++)
		(*parameters)[k] = gsl_vector_get(p, k);

	(*parameters)[2] = fabs((*parameters)[2]);

	gsl_multifit_nlinear_free(workspace);

	return status;
}

// Fit the effective index of a waveguide over a range of wavelengths with a polynomial in the normalized wavelength
// u = (wavelength - wavelength_center) / (half of the range), and compute the group index n_g = n_eff - wavelength
// dn_eff/dwavelength at the center of the range (only geometric dispersion is included, as the indices are constant)
int fit_waveguide_dispersion
(
	struct Waveguide *waveguide,
	double wavelength_min,
	double wavelength_max,
	int n_samples,
	int degree,
	double *coefficients,
	double *n_group
)
{
	double wavelength_center = 0.5 * (wavelength_min + wavelength_max);
	double half_range = 0.5 * (wavelength_max - wavelength_min);

	if (n_samples < 2 || n_samples < degree + 1 || half_range <= 0.0)
		return 1;

	double *u = malloc(n_samples * sizeof(double));
	double *n_eff = malloc(n_samples * sizeof(double));

	struct Waveguide sample_waveguide = *waveguide;

	int status = 0;
	for (int i = 0; i < n_samples && status == 0; i++)
	{
		u[i] = -1.0 + 2.0 * i / (n_samples - 1);
		sample_waveguide.wavelength = wavelength_center + u[i] * half_range;

		status = solve_waveguide_mode(&sample_waveguide);
		n_eff[i] = sample_waveguide.n_eff;
	}

	if (status == 0)
		status = fit_polynomial(n_samples, u, n_eff, degree, coefficients, NULL);

	if (status == 0 && n_group != NULL)
		*n_group = coefficients[0] - wavelength_center * coefficients[1] / half_range;

	free(n_eff);
	free(u);

	return status;
}
