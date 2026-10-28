"""Tests of the quantum dynamics module through the Python bindings, against independent numerical models."""

import numpy as np
import pytest
from scipy import linalg

import tiopt

TWO_PI = 2 * np.pi


def ladder(n):
    return np.diag(np.sqrt(np.arange(1, n)), 1)


def displacement(alpha, n):
    a = ladder(n)
    return linalg.expm(alpha * a.T - np.conj(alpha) * a)


def test_displacement_matrix_elements_match_matrix_exponential():
    # Exponentiate in a much larger space so that truncation does not affect the low Fock states
    alpha = 0.4 - 0.9j
    D = displacement(alpha, 120)
    for m in range(8):
        for n in range(8):
            assert tiopt.displacement_matrix_element(m, n, alpha) == pytest.approx(D[m, n], abs=1e-12)


def sideband_populations(rabi, eta, sideband, detuning, n_bar, n_max, times):
    """Excitation from an independent model: the resolved sideband Hamiltonian in the frame rotating at the residual
    detuning, built with matrix exponentials and evolved exactly."""
    n_motion = n_max + 1
    D = displacement(1j * eta, n_motion + 60)[:n_motion, :n_motion]
    dim = 2 * n_motion
    H = np.zeros((dim, dim), dtype=complex)
    for n in range(n_motion):
        H[n, n] = detuning / 2
        H[n_motion + n, n_motion + n] = -detuning / 2
        if 0 <= n + sideband < n_motion:
            H[n_motion + n + sideband, n] = rabi / 2 * D[n + sideband, n]
            H[n, n_motion + n + sideband] = np.conj(H[n_motion + n + sideband, n])
    p = (n_bar / (n_bar + 1)) ** np.arange(n_motion) / (n_bar + 1)
    rho_0 = np.zeros((dim, dim), dtype=complex)
    rho_0[:n_motion, :n_motion] = np.diag(p / p.sum())
    populations = []
    for t in times:
        U = linalg.expm(-1j * H * t)
        rho = U @ rho_0 @ U.conj().T
        populations.append(np.real(np.trace(rho[n_motion:, n_motion:])))
    return np.array(populations)


@pytest.mark.parametrize("sideband, detuning_khz", [(0, 0.0), (0, 7.0), (1, 0.0), (-1, 2.0), (2, 0.0)])
def test_single_ion_dynamics_match_independent_model(ion, sideband, detuning_khz):
    nu = TWO_PI * 1e6
    rabi = TWO_PI * 120e3
    eta = 0.15
    n_bar = 0.8
    n_max = 25
    detuning = TWO_PI * detuning_khz * 1e3
    pulse = tiopt.Pulse(0.0, 40e-6, sideband * nu + detuning, rabi, targets=[1], lamb_dicke=eta)
    result = tiopt.simulate_pulses(ion, [pulse], 1, nu, n_max, n_bar, decay=False, n_samples=21)
    expected = sideband_populations(rabi, eta, sideband, detuning, n_bar, n_max, result.times)
    np.testing.assert_allclose(result.excited_populations[:, 0], expected, atol=1e-6)


def test_analytical_thermal_rabi_matches_simulation(ion):
    nu = TWO_PI * 1e6
    pulse = tiopt.Pulse(0.0, 30e-6, nu, TWO_PI * 100e3, lamb_dicke=0.1)
    result = tiopt.simulate_pulses(ion, [pulse], 1, nu, 30, 0.3, decay=False, n_samples=31)
    analytical = tiopt.thermal_rabi_excitation(TWO_PI * 100e3, 0.1, 1, 0.3, 0.0, result.times, 30)
    np.testing.assert_allclose(result.excited_populations[:, 0], analytical, atol=1e-6)


def test_heating_and_coherent_displacement(ion):
    # Without light, heating adds quanta linearly on top of a displaced thermal state with n_bar + |alpha|^2 quanta
    pulse = tiopt.Pulse(0.0, 50e-6, 0.0, 0.0, lamb_dicke=0.0)
    alpha = 0.7 + 0.2j
    result = tiopt.simulate_pulses(ion, [pulse], 1, TWO_PI * 1e6, 50, 0.4, alpha=alpha, heating_rate=2e4, decay=False, n_samples=6)
    np.testing.assert_allclose(result.mean_phonon_numbers, 0.4 + abs(alpha) ** 2 + 2e4 * result.times, atol=1e-5)


def test_molmer_sorensen_gate(ion):
    nu = TWO_PI * 1e6
    delta = TWO_PI * 20e3
    eta = 0.05
    rabi = delta / (2 * eta)
    pulses = [
        tiopt.Pulse(0.0, TWO_PI / delta, nu + delta, rabi, targets=[1, 2], lamb_dicke=eta),
        tiopt.Pulse(0.0, TWO_PI / delta, -nu - delta, rabi, targets=[1, 2], lamb_dicke=eta),
    ]
    n_max = 12
    result = tiopt.simulate_pulses(ion, pulses, 2, nu, n_max, 0.0, decay=False, n_samples=2, density_matrix=True)
    n_motion = n_max + 1
    rho = result.density_matrix
    reduced = np.zeros((4, 4), dtype=complex)
    for a in range(4):
        for b in range(4):
            reduced[a, b] = sum(rho[a * n_motion + k, b * n_motion + k] for k in range(n_motion))
    # exp(-i (PI/4) sigma_y sigma_y) takes |SS> to (|SS> + i |DD>) / sqrt(2) with this phase convention
    bell = np.array([1, 0, 0, 1j]) / np.sqrt(2)
    assert np.real(bell.conj() @ reduced @ bell) > 0.999
    assert np.trace(reduced).real == pytest.approx(1.0, abs=1e-6)


def test_laser_coupling(ion):
    k = TWO_PI / ion.qubit_wavelength
    eta = tiopt.lamb_dicke_parameter([k, 0, 0], [2, 0, 0], ion, TWO_PI * 1e6)
    assert eta == pytest.approx(k * np.sqrt(1.054571817e-34 / (2 * ion.mass * TWO_PI * 1e6)))
    I_sat = tiopt.saturation_intensity(1e8, 400e-9)
    assert tiopt.rabi_frequency_from_intensity(1e8, 400e-9, 8 * I_sat) == pytest.approx(2e8)


def test_invalid_target(ion):
    with pytest.raises(ValueError):
        tiopt.simulate_pulses(ion, [tiopt.Pulse(0.0, 1e-6, 0.0, 1e5, targets=[3])], 2)
