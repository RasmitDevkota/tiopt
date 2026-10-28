"""Tests of the integrated photonics module through the Python bindings, against independent implementations."""

import numpy as np
import pytest
from scipy import optimize, signal

import tiopt


def slab_dispersion_te(n_eff, n_core, n_cover, n_substrate, thickness, wavelength, order):
    # Written independently of photonics.c, using arctan of ratios rather than atan2
    k_0 = 2 * np.pi / wavelength
    kappa = k_0 * np.sqrt(n_core ** 2 - n_eff ** 2)
    gamma_c = k_0 * np.sqrt(n_eff ** 2 - n_cover ** 2)
    gamma_s = k_0 * np.sqrt(n_eff ** 2 - n_substrate ** 2)
    return kappa * thickness - order * np.pi - np.arctan(gamma_c / kappa) - np.arctan(gamma_s / kappa)


@pytest.mark.parametrize("n_cover, n_substrate, thickness, order", [(1.45, 1.45, 0.3, 0), (1.0, 1.45, 0.25, 0), (1.45, 1.45, 0.8, 1), (1.0, 1.5, 1.2, 2)])
def test_slab_mode_matches_root_finding(n_cover, n_substrate, thickness, order):
    n_core = 2.0
    wavelength = 0.729
    expected = optimize.brentq(
        slab_dispersion_te, max(n_cover, n_substrate) + 1e-12, n_core - 1e-12,
        args=(n_core, n_cover, n_substrate, thickness, wavelength, order), xtol=1e-15
    )
    assert tiopt.slab_mode(n_core, n_cover, n_substrate, thickness, wavelength, "TE", order) == pytest.approx(expected, abs=1e-12)


def test_slab_mode_cutoff():
    # An asymmetric slab cannot guide even its fundamental mode if it is too thin
    assert tiopt.slab_mode(2.0, 1.0, 1.45, 0.02, 0.729) is None


def test_waveguide_effective_index_is_bounded():
    waveguide = tiopt.Waveguide(0.729, 0.6, 0.15, 2.0, 1.455)
    n_eff = waveguide.effective_index()
    n_slab = tiopt.slab_mode(2.0, 1.455, 1.455, 0.15, 0.729)
    assert 1.455 < n_eff < n_slab < 2.0
    _, n_group = waveguide.dispersion(0.709, 0.749)
    assert n_group > n_eff


def transfer_matrix_reflectance(indices, thicknesses, wavelength, theta, polarization):
    """Reflectance from interface and propagation matrices with Fresnel coefficients (a different formalism from the
    characteristic matrices used by photonics.c)."""
    indices = np.asarray(indices, dtype=complex)
    k_parallel = indices[0] * np.sin(theta)
    cos_theta = np.sqrt(1 - (k_parallel / indices) ** 2 + 0j)
    cos_theta = np.where(np.imag(indices * cos_theta) < 0, -cos_theta, cos_theta)
    M = np.eye(2, dtype=complex)
    for j in range(len(indices) - 1):
        n_1, n_2 = indices[j], indices[j + 1]
        c_1, c_2 = cos_theta[j], cos_theta[j + 1]
        if polarization == "TE":
            r = (n_1 * c_1 - n_2 * c_2) / (n_1 * c_1 + n_2 * c_2)
            t = 2 * n_1 * c_1 / (n_1 * c_1 + n_2 * c_2)
        else:
            r = (n_2 * c_1 - n_1 * c_2) / (n_2 * c_1 + n_1 * c_2)
            t = 2 * n_1 * c_1 / (n_2 * c_1 + n_1 * c_2)
        interface = np.array([[1, r], [r, 1]]) / t
        M = M @ interface
        if j + 1 < len(indices) - 1:
            delta = 2 * np.pi * n_2 * c_2 * thicknesses[j + 1] / wavelength
            M = M @ np.diag([np.exp(-1j * delta), np.exp(1j * delta)])
    return abs(M[1, 0] / M[0, 0]) ** 2


@pytest.mark.parametrize("polarization", ["TE", "TM"])
@pytest.mark.parametrize("theta", [0.0, 0.3, 0.9])
def test_multilayer_matches_interface_matrices(polarization, theta):
    rng = np.random.default_rng(3)
    for _ in range(5):
        n_layers = rng.integers(2, 6)
        indices = list(1.0 + 2.0 * rng.random(n_layers))
        indices[0] = 1.0
        indices[-1] = complex(3.5, 0.02)
        thicknesses = [0.0] + list(0.05 + rng.random(n_layers - 2)) + [0.0]
        _, _, R, T = tiopt.multilayer_response(indices, thicknesses, 0.633, theta, polarization)
        assert R == pytest.approx(transfer_matrix_reflectance(indices, thicknesses, 0.633, theta, polarization), abs=1e-12)
        # Part of the light is absorbed by the substrate, but none is created
        assert R + T <= 1 + 1e-12


def test_gaussian_beam_focus():
    waist_aperture, focal_distance, wavelength = 8.0, 68.75, 0.729
    z_R = np.pi * waist_aperture ** 2 / wavelength
    waist, distance = tiopt.gaussian_beam(waist_aperture, focal_distance, wavelength)
    assert waist == pytest.approx(waist_aperture / np.sqrt(1 + (z_R / focal_distance) ** 2))
    assert distance == pytest.approx(focal_distance / (1 + (focal_distance / z_R) ** 2))


def test_grating_beam_power(solved_trap):
    grating = solved_trap.grating_couplers()[0]
    theta, k_hat = grating.emission()
    e_perp = np.array([-np.sin(grating.azimuth), np.cos(grating.azimuth), 0.0])
    e_par = np.cross(e_perp, k_hat)
    # Integrate over a plane transverse to the beam halfway to the ion
    rho = np.linspace(-25, 25, 501)
    P, Q = np.meshgrid(rho, rho)
    points = grating.position + 0.5 * grating.focal_distance * k_hat + P[..., None] * e_par + Q[..., None] * e_perp
    intensity = grating.intensity(points)
    power = intensity.sum() * (rho[1] - rho[0]) ** 2 * tiopt.LENGTH_SCALE ** 2
    assert power == pytest.approx(grating.power * grating.efficiency, rel=1e-4)


def test_grating_points_at_target():
    grating = tiopt.Grating(position=(-30.0, 15.0, 0.0))
    target = np.array([0.0, 0.0, 55.0])
    grating.design(target)
    theta, k_hat = grating.emission()
    direction = (target - grating.position) / np.linalg.norm(target - grating.position)
    np.testing.assert_allclose(k_hat, direction, atol=1e-12)
    # First-order phase matching
    assert np.sin(theta) == pytest.approx(grating.effective_index() - grating.waveguide.wavelength / grating.period)


def test_fits_match_numpy_and_scipy():
    rng = np.random.default_rng(4)
    x = np.linspace(-2, 2, 40)
    y = 0.3 - 1.2 * x + 0.7 * x ** 2 + 0.1 * x ** 3 + 0.01 * rng.normal(size=x.size)
    coefficients, _ = tiopt.fit_polynomial(x, y, 3)
    np.testing.assert_allclose(coefficients, np.polynomial.polynomial.polyfit(x, y, 3), atol=1e-10)

    def gaussian(x, amplitude, center, sigma, offset):
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + offset

    y = gaussian(x, 1.5, 0.2, 0.6, 0.05) + 0.005 * rng.normal(size=x.size)
    expected, _ = optimize.curve_fit(gaussian, x, y, p0=[1, 0, 1, 0])
    np.testing.assert_allclose(tiopt.fit_gaussian(x, y), expected, rtol=1e-6, atol=1e-8)


def test_field_noise_psd_matches_scipy():
    # ion_heating.c computes a one-sided, Hann-windowed periodogram with the mean removed
    rng = np.random.default_rng(5)
    n, dt = 1024, 1e-8
    samples = 3.0 + rng.normal(size=n) + 2 * np.sin(2 * np.pi * 37 / (n * dt) * dt * np.arange(n))
    frequencies, psd = tiopt.field_noise_psd(samples, dt)
    expected_frequencies, expected_psd = signal.periodogram(samples, fs=1 / dt, window=signal.windows.hann(n, sym=True), detrend="constant", scaling="density")
    np.testing.assert_allclose(frequencies, expected_frequencies)
    np.testing.assert_allclose(psd[1:], expected_psd[1:], rtol=1e-10)


def test_collection_efficiency_matches_solid_angle():
    # Solid angle of a rectangle seen from a point above its center (A. Khadjavi, J. Opt. Soc. Am. 58, 1417 (1968))
    a, b, h = 30.0, 50.0, 40.0
    alpha = a / (2 * h)
    beta = b / (2 * h)
    solid_angle = 4 * np.arcsin(alpha * beta / np.sqrt((1 + alpha ** 2) * (1 + beta ** 2)))
    detector = tiopt.Detector(width=a, length=b, quantum_efficiency=0.7)
    assert detector.collection_efficiency([0.0, 0.0, h]) == pytest.approx(0.7 * solid_angle / (4 * np.pi), rel=1e-9)


def test_scattering_rate():
    gamma = 1.4e8
    assert tiopt.scattering_rate(gamma, 1.0) == pytest.approx(gamma / 4)
    # Detuning by half a linewidth halves the effective saturation
    assert tiopt.scattering_rate(gamma, 2.0, 0.5 * gamma) == pytest.approx(0.5 * gamma * 2 / 4)


def test_detection_fidelity_matches_poisson_thresholds():
    from scipy import stats

    rate, dark, time = 2e5, 500.0, 50e-6
    fidelity, threshold = tiopt.detection_fidelity(rate, dark, 0.0, time)
    # Without decay the fidelity is the best over thresholds of Poisson count distributions
    best = max(
        1 - 0.5 * (stats.poisson.cdf(n - 1, (rate + dark) * time) + stats.poisson.sf(n - 1, dark * time))
        for n in range(1, 40)
    )
    assert fidelity == pytest.approx(best, abs=1e-12)
    assert 1 <= threshold < (rate + dark) * time
