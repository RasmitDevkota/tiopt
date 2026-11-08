"""Tests of shaped pulses (tiopt.pulses and the envelopes of the C library)."""

import numpy as np
import pytest

import tiopt
from tiopt import pulses


@pytest.mark.parametrize("shape, parameters", [("square", (0, 0)), ("trapezoid", (2e-6, 3e-6)), ("gaussian", (2e-6, 0)), ("cosine", (4e-6, 1e-6)), ("blackman", (0, 0))])
def test_envelopes_match_the_c_library(shape, parameters):
    t = np.linspace(-1e-6, 11e-6, 49)
    assert np.allclose(pulses.envelope(shape, parameters, 10e-6, t), tiopt.pulse_envelope(shape, parameters, 10e-6, t), atol=1e-12)
    inside = pulses.envelope(shape, parameters, 10e-6, np.linspace(0, 10e-6, 1001))
    assert inside.max() == pytest.approx(1.0, abs=1e-3)
    assert inside.min() >= -1e-12


@pytest.mark.parametrize("shape", ["trapezoid", "gaussian", "cosine", "blackman"])
def test_shaped_pi_pulses_invert_the_qubit(ion, shape):
    ramp = 0.2 if shape == "gaussian" else 2e-6
    duration, parameters = pulses.shaped_pulse(shape, ramp, np.pi, 2 * np.pi * 100e3)
    pulse = tiopt.Pulse(0.0, duration, 0.0, 2 * np.pi * 100e3, lamb_dicke=0.0, shape=shape, shape_parameters=parameters)
    assert pulse.area == pytest.approx(np.pi, rel=1e-4)
    result = tiopt.simulate_pulses(ion, [pulse], n_ions=1, n_max=3, decay=False, n_samples=3)
    assert result.excited_populations[-1, 0] == pytest.approx(1.0, abs=1e-4)


def test_ramps_suppress_off_resonant_excitation():
    duration = 100e-6
    detuning = 2 * np.pi * 1e6
    assert pulses.carrier_suppression("square", (0, 0), duration, detuning) == pytest.approx(1.0)
    trapezoid = pulses.carrier_suppression("trapezoid", (5e-6, 5e-6), duration, detuning)
    cosine = pulses.carrier_suppression("cosine", (5e-6, 5e-6), duration, detuning)
    # Smooth ramps suppress it far more than linear ramps, and Blackman pulses (smooth everywhere) the most
    assert trapezoid < 1e-3
    assert cosine < 0.2 * trapezoid
    assert pulses.carrier_suppression("blackman", (0, 0), duration, detuning) < 1e-3 * cosine


@pytest.mark.parametrize("shape, ramp", [("square", 0.0), ("trapezoid", 20e-6), ("cosine", 20e-6), ("gaussian", 0.2)])
def test_shaped_molmer_sorensen_gates_make_bell_states(ion, shape, ramp):
    eta, rabi, nu = 0.05, 2 * np.pi * 100e3, 2 * np.pi * 1e6
    duration, delta, _ = pulses.ms_gate(shape, ramp, eta, rabi)
    if shape == "square":
        assert delta == pytest.approx(2 * eta * rabi) and duration == pytest.approx(2 * np.pi / delta)
    gates = pulses.ms_pulses(shape, ramp, eta, rabi, nu)
    result = tiopt.simulate_pulses(ion, gates, n_ions=2, mode_frequency=nu, n_max=10, decay=False, n_samples=2, density_matrix=True)
    rho = result.density_matrix
    n_motion = 11
    populations = [np.real(sum(rho[c * n_motion + k, c * n_motion + k] for k in range(n_motion))) for c in (0, 3)]
    coherence = sum(rho[k, 3 * n_motion + k] for k in range(n_motion))
    assert 0.5 * sum(populations) + abs(coherence) > 0.9995
