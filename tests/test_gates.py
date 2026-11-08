"""Tests of laser-ion dynamics with several modes (tiopt.multimode), gate design (tiopt.gates), and analytic models
(tiopt.analytic).
"""

import numpy as np
import pytest

import tiopt
from tiopt import analytic, crystals, gates, multimode

OMEGA = 2 * np.pi * 1e6


@pytest.fixture(scope="module")
def chain():
    """Axial modes of two 40Ca+ ions with light along the axis at 729 nm."""
    potential = crystals.HarmonicPotential("40Ca+", (OMEGA, 5 * OMEGA, 5.5 * OMEGA))
    crystal = crystals.find_crystal(potential, "40Ca+", n_ions=2)
    return multimode.Modes.from_crystal(crystal, [2 * np.pi / 729e-9, 0.0, 0.0], crystal.modes_along("x"), cutoffs=5)


# The gates are simulated without the far off-resonant carrier terms, which only slow the integration down here
RWA = 2 * np.pi * 0.6e6


def test_carrier_and_sideband_flopping():
    eta = 0.1
    modes = multimode.Modes(np.array([OMEGA]), np.array([[eta]]), [10], [0.0])
    rabi = 2 * np.pi * 50e3
    result = multimode.simulate(1, modes, [multimode.Tone(rabi, 0.0, [0])], 20e-6, n_samples=11)
    assert result.excited_populations[:, 0] == pytest.approx(np.sin(rabi * np.exp(-eta ** 2 / 2) * result.times / 2) ** 2, abs=2e-3)
    assert result.excited_populations[:, 0] == pytest.approx(analytic.rabi_flop(result.times, rabi, eta), abs=2e-3)
    analytic_result = multimode.simulate(1, modes, [multimode.Tone(rabi, 0.0, [0])], 20e-6, n_samples=11, displacement="analytic")
    assert analytic_result.excited_populations == pytest.approx(result.excited_populations, abs=1e-3)


def test_analytic_models_match_the_c_library():
    times = np.linspace(0, 50e-6, 11)
    for sideband in (-1, 0, 1):
        assert analytic.rabi_flop(times, 2 * np.pi * 100e3, 0.1, 0.5, sideband) == pytest.approx(
            tiopt.thermal_rabi_excitation(2 * np.pi * 100e3, 0.1, sideband, 0.5, 0.0, times), abs=1e-9)
    rabi, eta = 2 * np.pi * 100e3, 0.1
    delta = 2 * eta * rabi
    gg, ee, mixed = analytic.ms_populations(np.array([2 * np.pi / delta]), rabi, eta, delta)
    assert gg[0] == pytest.approx(0.5) and ee[0] == pytest.approx(0.5) and mixed[0] == pytest.approx(0.0, abs=1e-12)
    assert analytic.sideband_n_bar(0.2, 0.5) == pytest.approx(2 / 3)


def test_fits_recover_the_temperature():
    rng = np.random.default_rng(0)
    times = np.linspace(0, 60e-6, 40)
    data = analytic.rabi_flop(times, 2 * np.pi * 100e3, 0.1, 0.8, 1) + 0.01 * rng.standard_normal(len(times))
    fit = analytic.fit_rabi_flop(times, data, sideband=1, eta=0.1, rabi_guess=2 * np.pi * 110e3, n_bar_guess=0.3)
    assert fit["rabi_frequency"] == pytest.approx(2 * np.pi * 100e3, rel=0.03)
    assert fit["n_bar"] == pytest.approx(0.8, abs=0.15)


def test_robust_amplitude_modulated_gates(chain):
    duration = 100e-6
    detuning = chain.frequencies.max() + 2 * np.pi * 20e3
    gate = gates.design_am_gate(chain.frequencies, chain.lamb_dicke, duration, detuning)
    assert abs(gate.phases[0, 1]) == pytest.approx(np.pi / 4, rel=1e-6)
    assert np.abs(gate.displacements).max() < 1e-9
    assert np.abs(gate.derivatives).max() < 1e-9
    assert gate.amplitudes == pytest.approx(gate.amplitudes[::-1])
    result = multimode.simulate(2, chain, gate.tones(), duration, n_samples=3, rwa_cutoff=RWA)
    assert result.bell_fidelity() > 0.999
    # Drifts of the mode frequencies spoil the gate much less when it is robust
    drifted = multimode.Modes(chain.frequencies + 2 * np.pi * 1e3, chain.lamb_dicke, chain.cutoffs, chain.n_bar)
    fragile = gates.design_am_gate(chain.frequencies, chain.lamb_dicke, duration, detuning, robust=False, n_segments=20)
    robust_fidelity = multimode.simulate(2, drifted, gate.tones(), duration, n_samples=3, rwa_cutoff=RWA).bell_fidelity()
    fragile_fidelity = multimode.simulate(2, drifted, fragile.tones(), duration, n_samples=3, rwa_cutoff=RWA).bell_fidelity()
    assert robust_fidelity > 0.99 and robust_fidelity > fragile_fidelity + 0.05
    with pytest.raises(ValueError):
        gates.design_am_gate(chain.frequencies, chain.lamb_dicke, duration, detuning, n_segments=6)


def test_frequency_modulated_gates_and_gate_times(chain):
    duration = 100e-6
    gate = gates.design_fm_gate(chain.frequencies, chain.lamb_dicke, duration, chain.frequencies.max() + 2 * np.pi * 20e3, n_segments=8)
    assert abs(gate.phases[0, 1]) == pytest.approx(np.pi / 4, rel=1e-3)
    assert np.abs(gate.displacements).max() < 1e-3
    assert multimode.simulate(2, chain, gate.tones(), duration, n_samples=3, rwa_cutoff=RWA).bell_fidelity() > 0.999
    # The gate times of Murali et al. (ISCA 2020)
    assert gates.gate_time_model("AM1", 10, 3) == 278
    assert gates.gate_time_model("AM2", 10, 3) == 124
    assert gates.gate_time_model("PM", 10, 3) == 175
    assert gates.gate_time_model("FM", 10, 3) == 100
    assert gates.gate_time_model("FM", 20, 3) == pytest.approx(13.33 * 20 - 54)
