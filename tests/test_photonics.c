#include "test_photonics.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>

#include "defs.h"
#include "data_structures.h"
#include "photonics.h"
#include "ti_utils.h"
#include "traps.h"
#include "test_utils.h"

int test_all_photonics
(
	double atol
)
{
	printf("- Testing photonics module...\n");

	int result = 0;

	result += test_solve_slab_mode(atol);
	result += test_solve_waveguide_mode(atol);
	result += test_grating_coupler(atol);
	result += test_gaussian_beam(atol);
	result += test_multilayer_response(atol);
	result += test_fits(atol);
	result += test_detection(atol);
	result += test_trap_photonics(atol);

	return result;
}

int test_solve_slab_mode
(
	double atol
)
{
	double n_core = 2.0;
	double n_cladding = 1.45;
	double thickness = 0.3;
	double wavelength = 0.729;
	double k_0 = 2 * PI / wavelength;

	// Even modes of a symmetric slab satisfy tan(kappa d / 2) = gamma / kappa (TE) or (n_core / n_cladding)^2 gamma / kappa
	// (TM), and odd modes satisfy -cot(kappa d / 2) = gamma / kappa (TE)
	double error = 0.0;
	for (int polarization = TE; polarization <= TM; polarization++)
	{
		double n_eff;
		solve_slab_mode(n_core, n_cladding, n_cladding, thickness, wavelength, polarization, 0, &n_eff);

		double kappa = k_0 * sqrt(n_core*n_core - n_eff*n_eff);
		double gamma = k_0 * sqrt(n_eff*n_eff - n_cladding*n_cladding);
		double ratio = (polarization == TE) ? gamma / kappa : pow(n_core / n_cladding, 2) * gamma / kappa;

		error = MAX(error, fabs(tan(0.5 * kappa * thickness) - ratio) / ratio);
	}

	double n_eff_odd;
	solve_slab_mode(n_core, n_cladding, n_cladding, thickness, wavelength, TE, 1, &n_eff_odd);
	double kappa_odd = k_0 * sqrt(n_core*n_core - n_eff_odd*n_eff_odd);
	double gamma_odd = k_0 * sqrt(n_eff_odd*n_eff_odd - n_cladding*n_cladding);
	error = MAX(error, fabs(-1 / tan(0.5 * kappa_odd * thickness) - gamma_odd / kappa_odd) / (gamma_odd / kappa_odd));

	int result = check_error("slab modes satisfy symmetric dispersion relations", error, 1E3 * atol);

	// The first odd mode of a symmetric slab is cut off below d = wavelength / (2 sqrt(n_core^2 - n_cladding^2))
	double cutoff_thickness = wavelength / (2 * sqrt(n_core*n_core - n_cladding*n_cladding));
	double n_eff;
	int below_cutoff = solve_slab_mode(n_core, n_cladding, n_cladding, 0.99 * cutoff_thickness, wavelength, TE, 1, &n_eff);
	int above_cutoff = solve_slab_mode(n_core, n_cladding, n_cladding, 1.01 * cutoff_thickness, wavelength, TE, 1, &n_eff);

	result += check_error("slab mode cutoff", (below_cutoff == 1 && above_cutoff == 0) ? 0 : 1, 0);

	// An asymmetric slab (air above, oxide below) satisfies the general relation
	double n_cover = 1.0;
	solve_slab_mode(n_core, n_cover, n_cladding, thickness, wavelength, TE, 0, &n_eff);
	double kappa = k_0 * sqrt(n_core*n_core - n_eff*n_eff);
	double residual = kappa * thickness - atan(k_0 * sqrt(n_eff*n_eff - n_cover*n_cover) / kappa) - atan(k_0 * sqrt(n_eff*n_eff - n_cladding*n_cladding) / kappa);

	result += check_error("asymmetric slab mode satisfies dispersion relation", fabs(residual), 1E3 * atol);

	return result;
}

int test_solve_waveguide_mode
(
	double atol
)
{
	struct Waveguide waveguide = { 0.729, 0.6, 0.15, 2.0, 1.455, 0.0 };
	solve_waveguide_mode(&waveguide);

	double n_vertical;
	solve_slab_mode(waveguide.n_core, waveguide.n_cladding, waveguide.n_cladding, waveguide.thickness, waveguide.wavelength, TE, 0, &n_vertical);

	int result = check_error("waveguide effective index is bounded", (waveguide.n_cladding < waveguide.n_eff && waveguide.n_eff < n_vertical) ? 0 : 1, 0);

	// Wider waveguides confine the mode better, approaching the slab
	struct Waveguide wide_waveguide = waveguide;
	wide_waveguide.width = 1.2;
	solve_waveguide_mode(&wide_waveguide);

	struct Waveguide very_wide_waveguide = waveguide;
	very_wide_waveguide.width = 100.0;
	solve_waveguide_mode(&very_wide_waveguide);

	result += check_error("waveguide effective index increases with width", (wide_waveguide.n_eff > waveguide.n_eff) ? 0 : 1, 0);
	result += check_error("wide waveguide approaches slab", fabs(very_wide_waveguide.n_eff - n_vertical), 1E-4);

	return result;
}

static struct GratingCoupler make_grating
(
)
{
	struct GratingCoupler grating;

	grating.waveguide = (struct Waveguide) { 0.729, 0.6, 0.15, 2.0, 1.455, 0.0 };
	grating.position[0] = -0.75 * ION_HEIGHT;
	grating.position[1] = 10.0;
	grating.position[2] = 0.0;
	grating.azimuth = 0.0;
	grating.period = 0.0;
	grating.fill_factor = 0.5;
	grating.etch_depth = 0.15;
	grating.length = 20.0;
	grating.width = 8.0;
	grating.focal_distance = 0.0;
	grating.power = 1E-3;
	grating.efficiency = 0.3;

	return grating;
}

int test_grating_coupler
(
	double atol
)
{
	struct GratingCoupler grating = make_grating();

	double target[3] = { 0.0, 0.0, ION_HEIGHT };
	design_grating_coupler(&grating, target);

	// The emitted beam points from the grating to the target
	double theta;
	double k_hat[3];
	compute_grating_emission(&grating, &theta, &k_hat);

	double d[3] = { target[0] - grating.position[0], target[1] - grating.position[1], target[2] - grating.position[2] };
	double d_norm = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);

	double error = fabs(theta - acos(d[2] / d_norm));
	for (int k = 0; k < 3; k++)
		error = MAX(error, fabs(k_hat[k] - d[k] / d_norm));

	int result = check_error("designed grating coupler points at target", error, 1E3 * atol);

	// The beam carries the emitted power through any plane transverse to it
	double e_perp[3] = { -sin(grating.azimuth), cos(grating.azimuth), 0.0 };
	double e_par[3];
	cross(&e_perp, &k_hat, &e_par);

	double z = 0.6 * grating.focal_distance;
	double extent = 20.0;
	double step = 0.05;
	double power = 0.0;
	for (double rho_par = -extent; rho_par <= extent; rho_par += step)
		for (double rho_perp = -extent; rho_perp <= extent; rho_perp += step)
		{
			double position[3];
			for (int k = 0; k < 3; k++)
				position[k] = grating.position[k] + z * k_hat[k] + rho_par * e_par[k] + rho_perp * e_perp[k];

			double intensity;
			evaluate_grating_beam(&grating, position, &intensity, NULL);

			power += intensity * step*step * LENGTH_SCALE*LENGTH_SCALE;
		}

	result += check_error("grating beam conserves power", fabs(power - grating.power * grating.efficiency) / (grating.power * grating.efficiency), 1E-4);

	// Focusing increases the intensity at the target
	double intensity_target;
	evaluate_grating_beam(&grating, target, &intensity_target, NULL);

	struct GratingCoupler unfocused_grating = grating;
	unfocused_grating.focal_distance = 0.0;

	double intensity_unfocused;
	evaluate_grating_beam(&unfocused_grating, target, &intensity_unfocused, NULL);

	result += check_error("focusing grating increases intensity at target", (intensity_target > intensity_unfocused) ? 0 : 1, 0);

	// A grating whose aperture has a Rayleigh range much longer than the focal distance is brightest at the target
	struct GratingCoupler large_grating = make_grating();
	large_grating.length = 60.0;
	large_grating.width = 60.0;
	design_grating_coupler(&large_grating, target);

	evaluate_grating_beam(&large_grating, target, &intensity_target, NULL);

	double before[3];
	double after[3];
	for (int k = 0; k < 3; k++)
	{
		before[k] = target[k] - 5.0 * k_hat[k];
		after[k] = target[k] + 5.0 * k_hat[k];
	}

	double intensity_before;
	double intensity_after;
	evaluate_grating_beam(&large_grating, before, &intensity_before, NULL);
	evaluate_grating_beam(&large_grating, after, &intensity_after, NULL);

	result += check_error("large grating beam focuses at target", (intensity_target > intensity_before && intensity_target > intensity_after) ? 0 : 1, 0);

	return result;
}

int test_gaussian_beam
(
	double atol
)
{
	double wavelength = 0.729;

	// Unfocused beams keep their waist at the aperture
	double waist;
	double waist_distance;
	compute_gaussian_beam(5.0, 0.0, wavelength, &waist, &waist_distance);
	double error = fabs(waist - 5.0) + fabs(waist_distance);

	// Strongly focused beams (Rayleigh range much longer than the focal distance) reach waist wavelength f / (PI w)
	double waist_aperture = 50.0;
	double focal_distance = 100.0;
	compute_gaussian_beam(waist_aperture, focal_distance, wavelength, &waist, &waist_distance);

	double z_R = PI * waist_aperture*waist_aperture / wavelength;
	error += fabs(waist - wavelength * focal_distance / (PI * waist_aperture)) / waist - pow(focal_distance / z_R, 2);
	error += fabs(waist_distance - focal_distance) / focal_distance - pow(focal_distance / z_R, 2);

	return check_error("gaussian beam focusing limits", MAX(error, 0.0), 1E-6);
}

int test_multilayer_response
(
	double atol
)
{
	double wavelength = 0.633;

	// Single interface at normal incidence: r = (n_1 - n_2) / (n_1 + n_2)
	double n_real[4] = { 1.0, 1.5, 0.0, 0.0 };
	double n_imag[4] = { 0.0, 0.0, 0.0, 0.0 };
	double thicknesses[4] = { 0.0, 0.0, 0.0, 0.0 };
	double r[2];
	double t[2];
	double R;
	double T;
	compute_multilayer_response(2, n_real, n_imag, thicknesses, wavelength, 0.0, TE, &r, &t, &R, &T);

	double error = fabs(r[0] + 0.2) + fabs(r[1]) + fabs(R - 0.04) + fabs(T - 0.96);

	int result = check_error("multilayer response of single interface", error, atol);

	// Fresnel equations at oblique incidence
	double theta_i = 40 * PI / 180;
	double theta_t = asin(sin(theta_i) / 1.5);
	double R_s_fresnel = pow((cos(theta_i) - 1.5 * cos(theta_t)) / (cos(theta_i) + 1.5 * cos(theta_t)), 2);
	double R_p_fresnel = pow((1.5 * cos(theta_i) - cos(theta_t)) / (1.5 * cos(theta_i) + cos(theta_t)), 2);

	double R_s;
	double R_p;
	compute_multilayer_response(2, n_real, n_imag, thicknesses, wavelength, theta_i, TE, NULL, NULL, &R_s, &T);
	compute_multilayer_response(2, n_real, n_imag, thicknesses, wavelength, theta_i, TM, NULL, NULL, &R_p, &T);

	result += check_error("multilayer response matches Fresnel equations", fabs(R_s - R_s_fresnel) + fabs(R_p - R_p_fresnel), atol);

	// No TM reflection at Brewster's angle
	compute_multilayer_response(2, n_real, n_imag, thicknesses, wavelength, atan(1.5), TM, NULL, NULL, &R_p, &T);
	result += check_error("no reflection at Brewster's angle", R_p, atol);

	// A quarter-wave antireflection coating of index sqrt(n_substrate)
	double n_coated[3] = { 1.0, sqrt(1.5), 1.5 };
	double d_coated[3] = { 0.0, wavelength / (4 * sqrt(1.5)), 0.0 };
	compute_multilayer_response(3, n_coated, n_imag, d_coated, wavelength, 0.0, TE, NULL, NULL, &R, &T);
	result += check_error("quarter-wave antireflection coating", R, atol);

	// Lossless stacks conserve energy at any angle and polarization
	double n_stack[4] = { 1.0, 2.0, 1.45, 3.5 };
	double d_stack[4] = { 0.0, 0.17, 0.83, 0.0 };
	double energy_error = 0.0;
	for (int polarization = TE; polarization <= TM; polarization++)
		for (double theta = 0.0; theta < 1.5; theta += 0.1)
		{
			compute_multilayer_response(4, n_stack, n_imag, d_stack, wavelength, theta, polarization, NULL, NULL, &R, &T);
			energy_error = MAX(energy_error, fabs(R + T - 1));
		}

	result += check_error("lossless multilayer conserves energy", energy_error, atol);

	// Total internal reflection beyond the critical angle
	double n_tir[2] = { 1.5, 1.0 };
	compute_multilayer_response(2, n_tir, n_imag, thicknesses, wavelength, 60 * PI / 180, TE, NULL, NULL, &R, &T);
	result += check_error("total internal reflection", fabs(R - 1) + fabs(T), atol);

	// Absorbing substrates follow the Fresnel equations with complex indices
	double n_absorbing_real[2] = { 1.0, 3.7 };
	double n_absorbing_imag[2] = { 0.0, 0.01 };
	compute_multilayer_response(2, n_absorbing_real, n_absorbing_imag, thicknesses, wavelength, 0.0, TE, NULL, NULL, &R, &T);
	double complex n_s = 3.7 + 0.01 * I;
	result += check_error("reflection from absorbing substrate", fabs(R - pow(cabs((1 - n_s) / (1 + n_s)), 2)), atol);

	// The directionality of a grating ranges from (1 - |r|) / 2 to (1 + |r|) / 2 as the oxide thickness varies, and is
	// 1/2 without a reflecting substrate
	double theta_box = 0.4;
	double directionality = compute_grating_directionality(1.45, 2.0, 1.45, 0.0, wavelength, theta_box, TE);
	double directionality_error = fabs(directionality - 0.5);

	double n_box_si[2] = { 1.45, 3.7 };
	compute_multilayer_response(2, n_box_si, n_imag, thicknesses, wavelength, theta_box, TE, &r, &t, &R, &T);
	double r_abs = sqrt(R);
	double r_phase = atan2(r[1], r[0]);

	// Choose the oxide thickness so that the reflected beam returns in phase
	double round_trip_per_thickness = 4 * PI * 1.45 * cos(theta_box) / wavelength;
	double box_thickness = (2 * PI - r_phase) / round_trip_per_thickness;
	directionality = compute_grating_directionality(1.45, box_thickness, 3.7, 0.0, wavelength, theta_box, TE);
	directionality_error = MAX(directionality_error, fabs(directionality - 0.5 * (1 + r_abs)));

	directionality = compute_grating_directionality(1.45, box_thickness + PI / round_trip_per_thickness, 3.7, 0.0, wavelength, theta_box, TE);
	directionality_error = MAX(directionality_error, fabs(directionality - 0.5 * (1 - r_abs)));

	result += check_error("grating directionality from substrate reflection", directionality_error, atol);

	return result;
}

int test_fits
(
	double atol
)
{
	// Noise-free polynomial data are fit exactly
	int n_points = 20;
	double x[50];
	double y[50];
	double coefficients_analytical[4] = { 1.0, -2.0, 0.5, 0.25 };
	for (int i = 0; i < n_points; i++)
	{
		x[i] = -1.5 + 3.0 * i / (n_points - 1);
		y[i] = coefficients_analytical[0] + coefficients_analytical[1] * x[i] + coefficients_analytical[2] * x[i]*x[i] + coefficients_analytical[3] * x[i]*x[i]*x[i];
	}

	double coefficients[4];
	double chi_sq;
	fit_polynomial(n_points, x, y, 3, coefficients, &chi_sq);

	double error = 0.0;
	for (int k = 0; k < 4; k++)
		error = MAX(error, fabs(coefficients[k] - coefficients_analytical[k]));

	int result = check_error("fit_polynomial recovers coefficients", error, atol);

	// Noise-free Gaussian data are fit exactly
	n_points = 50;
	double parameters_analytical[4] = { 2.0, 0.3, 0.7, 0.1 };
	for (int i = 0; i < n_points; i++)
	{
		x[i] = -3.0 + 6.0 * i / (n_points - 1);
		y[i] = parameters_analytical[0] * exp(-0.5 * pow((x[i] - parameters_analytical[1]) / parameters_analytical[2], 2)) + parameters_analytical[3];
	}

	double parameters[4];
	fit_gaussian(n_points, x, y, &parameters);

	error = 0.0;
	for (int k = 0; k < 4; k++)
		error = MAX(error, fabs(parameters[k] - parameters_analytical[k]));

	result += check_error("fit_gaussian recovers parameters", error, 1E2 * atol);

	// The group index from the dispersion fit matches a finite difference of the effective index
	struct Waveguide waveguide = { 0.729, 0.6, 0.15, 2.0, 1.455, 0.0 };
	double dispersion_coefficients[5];
	double n_group;
	fit_waveguide_dispersion(&waveguide, 0.709, 0.749, 21, 4, dispersion_coefficients, &n_group);

	double h = 1E-4;
	struct Waveguide waveguide_plus = waveguide;
	struct Waveguide waveguide_minus = waveguide;
	waveguide_plus.wavelength += h;
	waveguide_minus.wavelength -= h;
	solve_waveguide_mode(&waveguide);
	solve_waveguide_mode(&waveguide_plus);
	solve_waveguide_mode(&waveguide_minus);

	double n_group_finite_differences = waveguide.n_eff - waveguide.wavelength * (waveguide_plus.n_eff - waveguide_minus.n_eff) / (2 * h);

	result += check_error("group index from dispersion fit", fabs(n_group - n_group_finite_differences), 1E-5);
	result += check_error("waveguide has normal group index", (n_group > waveguide.n_eff) ? 0 : 1, 0);

	return result;
}

int test_detection
(
	double atol
)
{
	int result = 0;

	// A detector much larger than the height of the ion above it collects half of the light
	struct Detector detector = { { 0.0, 0.0, 0.0 }, 1E8, 1E8, 0.397, 0.8, 0.0 };
	double position[3] = { 0.0, 0.0, 50.0 };
	result += check_error("compute_collection_efficiency of large detector", fabs(compute_collection_efficiency(&detector, position) - 0.4), 1E-5);

	// A small detector far below the ion collects its area over 4 PI r^2
	detector.width = 1.0;
	detector.length = 1.0;
	double collection = compute_collection_efficiency(&detector, position);
	double collection_expected = 0.8 * 1.0 / (4 * PI * 50.0 * 50.0);
	result += check_error("compute_collection_efficiency of small detector", fabs(collection - collection_expected) / collection_expected, 1E-3);

	// Scattering saturates at half the linewidth
	double gamma = 1.4E8;
	result += check_error("compute_scattering_rate saturates", fabs(compute_scattering_rate(gamma, 1E9, 0.0) / (0.5 * gamma) - 1), 1E-6);
	result += check_error("compute_scattering_rate at saturation", fabs(compute_scattering_rate(gamma, 1.0, 0.0) / (0.25 * gamma) - 1), atol);

	// With no dark counts and no decay, the only error is the bright state giving no counts, so the best threshold is a
	// single count and the error is half the probability of no counts
	double photon_rate = 1E5;
	double detection_time = 50E-6;
	int threshold;
	double fidelity = compute_detection_fidelity(photon_rate, 0.0, 0.0, detection_time, &threshold);
	result += check_error("compute_detection_fidelity without dark counts", fabs(fidelity - (1 - 0.5 * exp(-photon_rate * detection_time))), atol);
	result += check_error("compute_detection_fidelity threshold without dark counts", abs(threshold - 1), 0.0);

	// Without light the two states cannot be told apart
	result += check_error("compute_detection_fidelity without light", fabs(compute_detection_fidelity(0.0, 100.0, 0.0, detection_time, NULL) - 0.5), atol);

	// Decay of the dark state during detection lowers the fidelity, by at most half the probability of decay
	double decay_rate = 1E3;
	double fidelity_decay = compute_detection_fidelity(photon_rate, 0.0, decay_rate, detection_time, NULL);
	double decay_error = fidelity - fidelity_decay;
	result += check_error("compute_detection_fidelity with decay", (decay_error > 0.0 && decay_error <= 0.5 * (1 - exp(-decay_rate * detection_time))) ? 0.0 : 1.0, 0.0);

	return result;
}

int test_trap_photonics
(
	double atol
)
{
	struct Trap trap;
	int status = generate_trap_from_preset(LINEAR_SURFACE_TRAP_WITH_PHOTONICS, 7, 50.0, &trap);

	int result = check_error("trap preset with photonics designs every grating", status, 0.0);
	result += check_error("trap preset with photonics has every beam", abs(trap.n_grating_couplers - N_PHOTONICS_BEAMS) + abs(trap.n_detectors - 1), 0.0);

	// Every beam is focused on the zone it addresses, so its intensity peaks there
	double gate_position[3] = { 0.0, 0.0, ION_HEIGHT };
	double measure_position[3] = { 100.0, 0.0, ION_HEIGHT };
	int at_gate_zone[N_PHOTONICS_BEAMS] = { 1, 0, 0, 1 };

	double error = 0.0;
	for (int b = 0; b < trap.n_grating_couplers; b++)
	{
		double *target = at_gate_zone[b] ? gate_position : measure_position;

		double intensity;
		evaluate_grating_beam(&(*trap.grating_couplers)[b], target, &intensity, NULL);

		for (int k = 0; k < 2; k++)
		{
			double displaced[3] = { target[0], target[1], target[2] };
			displaced[k] += 2.0;

			double displaced_intensity;
			evaluate_grating_beam(&(*trap.grating_couplers)[b], displaced, &displaced_intensity, NULL);

			if (displaced_intensity >= intensity)
				error += 1.0;
		}
	}

	result += check_error("trap photonics beams peak at their zones", error, 0.0);

	free_trap(&trap);

	return result;
}
