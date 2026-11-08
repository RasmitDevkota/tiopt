"""Tests of the molecular dynamics of crystals (tiopt.dynamics)."""

import numpy as np
import pytest

import tiopt
from tiopt import crystals, dynamics

OMEGA = 2 * np.pi * 1e6


@pytest.fixture(scope="module")
def model():
    return crystals.HarmonicPotential("40Ca+", (OMEGA, 4 * OMEGA, 4.4 * OMEGA))


def total_energy(potential, ions, positions, velocities):
    U, _, _, _ = potential.evaluate(ions, positions)
    coulomb, _ = crystals.coulomb(positions, [ion.charge for ion in ions])
    return U.sum() + coulomb + 0.5 * sum(ion.mass * np.sum(v ** 2) for ion, v in zip(ions, velocities))


def test_energy_is_conserved_without_lasers(model):
    ion = tiopt.Ion("40Ca+")
    start = dynamics.ion_cloud(5, model.center, 5.0, seed=3)
    assert np.min([np.linalg.norm(a - b) for i, a in enumerate(start) for b in start[i + 1:]]) >= 2.0
    result = dynamics.simulate(model, ion, start, duration=10e-6, dt=10e-9, temperature=1.0, n_samples=5)
    ions = [ion] * 5
    energies = [total_energy(model, ions, result.positions[k], result.velocities[k]) for k in range(5)]
    assert np.ptp(energies) < 0.01 * abs(energies[0])


def test_doppler_cooling_reaches_millikelvins(model):
    ion = tiopt.Ion("40Ca+")
    crystal = crystals.find_crystal(model, ion, n_ions=2)
    result = dynamics.simulate(model, ion, crystal.positions, duration=100e-6, dt=10e-9, beams=[dynamics.CoolingBeam()], temperature=0.1, n_samples=100)
    final = result.temperatures(0.6).mean()
    assert final < 0.01
    assert final > 0.5 * dynamics.doppler_limit(ion)
    assert result.scattered_photons.min() > 100 and not result.lost.any()


def test_field_noise_heats_at_the_expected_rate(model):
    ion = tiopt.Ion("40Ca+")
    S_E = 1e-12
    rates = []
    for seed in range(6):
        result = dynamics.simulate(model, ion, model.center[None, :], np.zeros((1, 3)), duration=500e-6, dt=20e-9, field_noise=S_E, n_samples=20,
                                   seed=seed, coulomb_interaction=False)
        rates.append(np.polyfit(result.times, result.kinetic_energy(), 1)[0])
    # Half of the energy of a harmonic oscillator is kinetic, and every axis gains q^2 S_E / (4 m) per second
    assert np.mean(rates) == pytest.approx(0.5 * 3 * ion.charge ** 2 * S_E / (4 * ion.mass), rel=0.35)


def test_full_rf_and_rf_phase_scans():
    ion = tiopt.Ion("40Ca+")
    trap = tiopt.Trap.from_preset(7, 50.0)
    trap.solve("analytical")
    a, b = trap.find_rf_null((-10.0, 0.0, None)), trap.find_rf_null((10.0, 0.0, None))
    waveform = tiopt.Waveform.transport(trap, ion, a, b, 2e-6, 51, OMEGA)
    phases, excitations = dynamics.rf_phase_scan(trap, ion, waveform, phases=[0.0, np.pi / 2, np.pi], settle=0.5e-6)
    assert len(excitations) == 3 and np.all(np.isfinite(excitations))
    pseudo = tiopt.simulate_trajectory(trap, ion, waveform, 2.5e-6, 5e-9, a)
    reference = tiopt.motional_quanta(ion, OMEGA, pseudo.positions[-1], pseudo.velocities[-1], b)
    assert np.mean(excitations) == pytest.approx(reference, rel=0.2, abs=0.05)
