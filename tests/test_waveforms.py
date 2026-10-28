"""Tests of waveforms, ion dynamics and heating through the Python bindings."""

import numpy as np
import pytest

import tiopt


def test_static_well(solved_trap, ion):
    null = solved_trap.find_rf_null((20.0, 0.0, None))
    waveform = tiopt.Waveform.transport(solved_trap, ion, null, null, 0.0, 1)
    max_force, max_frequency_error = waveform.errors(solved_trap, ion)
    assert max_force < 1e-20
    assert max_frequency_error < 1e-4
    # The ion sits at the target and is confined in every direction
    # (to well within the 11 nm spread of the motional ground state)
    minimum = solved_trap.find_minimum(ion, null + [0.5, 0.2, -0.3], waveform.voltages[0])
    np.testing.assert_allclose(minimum, null, atol=1e-3)
    omega, _ = solved_trap.normal_modes(ion, minimum, waveform.voltages[0])
    assert np.all(omega > 0)


def test_transport_is_adiabatic(solved_trap, ion):
    start = solved_trap.find_rf_null((-50.0, 0.0, None))
    end = solved_trap.find_rf_null((50.0, 0.0, None))
    waveform = tiopt.Waveform.transport(solved_trap, ion, start, end, 50e-6, 501)
    assert waveform.voltages.shape == (501, solved_trap.n_electrodes)
    np.testing.assert_allclose(waveform.well_positions[[0, -1], 0], [start, end])
    # RF electrodes carry no DC voltage
    assert np.all(waveform.voltages[:, : solved_trap.n_rf] == 0.0)

    trajectory = tiopt.simulate_trajectory(solved_trap, ion, waveform, 55e-6, 5e-9, start)
    quanta = tiopt.motional_quanta(ion, tiopt.AXIAL_FREQUENCY, trajectory.positions[-1], trajectory.velocities[-1], end)
    assert quanta < 0.05
    np.testing.assert_allclose(trajectory.positions[-1], end, atol=0.05)


def test_fast_transport_excites_ion(solved_trap, ion):
    # Moving the well 100 µm in 3 µs is far from adiabatic for a 1 MHz well
    start = solved_trap.find_rf_null((-50.0, 0.0, None))
    end = solved_trap.find_rf_null((50.0, 0.0, None))
    waveform = tiopt.Waveform.transport(solved_trap, ion, start, end, 3e-6, 301)
    trajectory = tiopt.simulate_trajectory(solved_trap, ion, waveform, 6e-6, 2e-9, start)
    quanta = tiopt.motional_quanta(ion, tiopt.AXIAL_FREQUENCY, trajectory.positions[-1], trajectory.velocities[-1], end)
    assert quanta > 100


def test_voltages_at_interpolates_linearly(solved_trap, ion):
    start = solved_trap.find_rf_null((-10.0, 0.0, None))
    end = solved_trap.find_rf_null((10.0, 0.0, None))
    waveform = tiopt.Waveform.transport(solved_trap, ion, start, end, 10e-6, 11)
    times = waveform.times
    voltages = waveform.voltages
    for t in [0.0, 2.5e-6, 7.3e-6, 10e-6, 20e-6]:
        expected = [np.interp(t, times, voltages[:, e]) for e in range(waveform.n_electrodes)]
        np.testing.assert_allclose(waveform.voltages_at(t), expected, atol=1e-12)


def test_resampled_waveform(solved_trap, ion):
    start = solved_trap.find_rf_null((-10.0, 0.0, None))
    end = solved_trap.find_rf_null((10.0, 0.0, None))
    coarse = tiopt.Waveform.transport(solved_trap, ion, start, end, 10e-6, 11)
    fine = coarse.resampled(solved_trap, 21)
    np.testing.assert_allclose(fine.voltages[::2], coarse.voltages, atol=1e-12)
    np.testing.assert_allclose(fine.times[::2], coarse.times, atol=1e-18)


def test_ideal_secular_frequencies_sum_rule(ion):
    # The DC potential is harmonic, so the sum of omega^2 only depends on the RF drive
    omega_rf_only = tiopt.ideal_secular_frequencies(ion, 0.0, 100.0, 2 * np.pi * 20e6, 500.0)
    omega = tiopt.ideal_secular_frequencies(ion, 2.0, 100.0, 2 * np.pi * 20e6, 500.0)
    assert np.sum(omega ** 2) == pytest.approx(np.sum(omega_rf_only ** 2))
    assert omega_rf_only[2] == 0.0


def test_heating_rates(solved_trap, ion):
    null = solved_trap.find_rf_null((0.0, 0.0, None))
    rate = solved_trap.voltage_noise_heating_rate(ion, null, tiopt.AXIAL_FREQUENCY, S_V=1e-18)
    assert rate > 0
    assert solved_trap.voltage_noise_heating_rate(ion, null, tiopt.AXIAL_FREQUENCY, S_V=4e-18) == pytest.approx(4 * rate)
    # Heating from RF amplitude noise vanishes on the RF null and grows away from it
    assert solved_trap.rf_noise_heating_rate(ion, null, 2 * np.pi * 4e6) < 1e-20
    assert solved_trap.rf_noise_heating_rate(ion, null + [0, 0, 1.0], 2 * np.pi * 4e6) > 0
    # The Turchette formula for field noise
    assert tiopt.heating_rate_field_noise(ion, 2 * np.pi * 1e6, 1e-12) == pytest.approx(ion.charge ** 2 * 1e-12 / (4 * ion.mass * 1.054571817e-34 * 2 * np.pi * 1e6))


def test_experiments(solved_trap, ion):
    idle_cost = solved_trap.run_experiment(ion, "idle")
    assert 0 <= idle_cost < 1.0
    back_and_forth_cost = solved_trap.run_experiment(ion, "back_and_forth")
    assert 0 <= back_and_forth_cost < 1.0
