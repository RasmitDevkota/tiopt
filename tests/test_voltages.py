"""Tests of voltages from objectives and constraints (tiopt.voltages) and of the hardware model (tiopt.hardware)."""

import numpy as np
import pytest

import tiopt
from tiopt import hardware, voltages

OMEGA = 2 * np.pi * 1e6


@pytest.fixture(scope="module")
def setup():
    ion = tiopt.Ion("40Ca+")
    trap = tiopt.Trap.from_preset(7, 50.0)
    trap.solve("analytical")
    return trap, ion


def test_static_wells_match_the_c_solver(setup):
    trap, ion = setup
    null = trap.find_rf_null((0.0, 0.0, None))
    solved, info = voltages.static_well(trap, ion, null, OMEGA)
    assert info.converged
    reference = tiopt.Waveform.transport(trap, ion, null, null, 0.0, 1).voltages[0]
    assert solved == pytest.approx(reference, rel=1e-4, abs=1e-6)
    # A limit which the well cannot meet still gives the closest well within the limit
    limited, info = voltages.static_well(trap, ion, null, OMEGA, max_voltage=1.0)
    assert np.abs(limited).max() <= 1.0 + 1e-6
    problem = voltages.Problem(trap, ion)
    problem.add(voltages.Gradient(0, null, 0.0, kind="=="), voltages.Hessian(0, null, voltages.frequency_curvature(ion, OMEGA), kind="=="), voltages.VoltageLimit(1.0))
    _, info = problem.solve(max_iterations=2000)
    assert not info.converged


def test_transports_with_limits(setup):
    trap, ion = setup
    start, end = np.array([-30.0, 0.0, 55.0]), np.array([30.0, 0.0, 55.0])
    waveform = voltages.transport(trap, ion, start, end, 20e-6, 51, OMEGA, max_voltage=5.0, max_slew_rate=2e6)
    assert waveform.info.converged
    V = waveform.voltages
    assert np.abs(V).max() <= 5.0 + 1e-6
    assert np.abs(np.diff(V, axis=0) / np.diff(waveform.times)[:, None]).max() <= 2e6 * (1 + 1e-6)
    for step in (0, 25, 50):
        well = waveform.well_positions[step, 0]
        assert trap.find_minimum(ion, well, V[step]) == pytest.approx(well, abs=0.05)
        assert trap.normal_modes(ion, well, V[step])[0][0] == pytest.approx(OMEGA, rel=1e-3)
    for profile in voltages.PROFILES.values():
        ramp = profile(0.0, 1.0, 11)
        assert ramp[0] == pytest.approx(0.0) and ramp[-1] == pytest.approx(1.0)


def test_filters_and_dacs():
    rc = hardware.Filter.rc(1e6)
    kernel = rc.kernel(1e-8, 2000)
    assert kernel.sum() == pytest.approx(1.0, abs=1e-6)
    assert kernel[0] == pytest.approx(1 - np.exp(-2 * np.pi * 1e6 * 1e-8))
    t = np.arange(400) * 2e-9
    measured = hardware.Filter.from_step_response(t, 1 - np.exp(-t * 2 * np.pi * 1e6))
    assert measured.kernel(1e-8, 50) == pytest.approx(kernel[:50], rel=0.05)
    dac = hardware.DAC(bits=12, v_min=-10, v_max=10)
    assert dac.lsb == pytest.approx(20 / 4095)
    assert np.abs(dac.quantize(np.linspace(-12, 12, 100)) - np.clip(np.linspace(-12, 12, 100), -10, 10)).max() <= dac.lsb / 2 + 1e-12


def test_precompensation(setup):
    trap, ion = setup
    a, b = trap.find_rf_null((-30.0, 0.0, None)), trap.find_rf_null((30.0, 0.0, None))
    waveform = tiopt.Waveform.transport(trap, ion, a, b, 5e-6, 201, OMEGA)
    rc = hardware.Filter.rc(1e6)
    plain = hardware.through_hardware(waveform, hardware.DAC(), rc, compensate=False)
    compensated = hardware.through_hardware(waveform, hardware.DAC(), rc, compensate=True)
    assert compensated.max_error < plain.max_error / 100
    assert compensated.waveform().n_steps == len(compensated.times)
