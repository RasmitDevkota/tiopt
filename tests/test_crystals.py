"""Tests of Coulomb crystals and their normal modes (tiopt.crystals)."""

import numpy as np
import pytest

import tiopt
from tiopt import crystals

OMEGA = 2 * np.pi * 1e6


@pytest.fixture(scope="module")
def well():
    """The linear trap preset with a static 1 MHz well at its center."""
    ion = tiopt.Ion("40Ca+")
    trap = tiopt.Trap.from_preset(7, 50.0)
    trap.solve("analytical")
    null = trap.find_rf_null((0.0, 0.0, None))
    waveform = tiopt.Waveform.transport(trap, ion, null, null, 0.0, 1)
    return trap, ion, waveform.voltages[0], null


def test_linear_chains_match_james():
    # Equilibrium positions of 2 and 3 ions in units of the length scale (D. F. V. James, 1998)
    assert crystals._chain_positions(2) == pytest.approx([-0.62996, 0.62996], abs=1e-5)
    assert crystals._chain_positions(3) == pytest.approx([-1.0772, 0.0, 1.0772], abs=1e-4)
    axial, radial = crystals.chain_frequencies(3, OMEGA, 5 * OMEGA)
    assert axial / OMEGA == pytest.approx([1.0, np.sqrt(3), np.sqrt(29 / 5)], rel=1e-9)
    assert radial / OMEGA == pytest.approx([5.0, np.sqrt(24), np.sqrt(25 - 2.4)], rel=1e-9)
    for n in (10, 40):
        assert np.all(np.diff(crystals._chain_positions(n)) > 0)


def test_crystals_in_a_model_trap():
    potential = crystals.HarmonicPotential("40Ca+", (OMEGA, 5 * OMEGA, 5.5 * OMEGA))
    crystal = crystals.find_crystal(potential, "40Ca+", n_ions=3)
    assert crystal.stable
    axial, radial = crystals.chain_frequencies(3, OMEGA, 5 * OMEGA)
    assert crystal.frequencies[crystal.modes_along("x")] == pytest.approx(axial, rel=1e-6)
    assert crystal.frequencies[crystal.modes_along("y")] == pytest.approx(np.sort(radial), rel=1e-6)
    assert np.sort(crystal.positions[:, 0]) == pytest.approx(crystals.linear_chain(3, OMEGA, "40Ca+") + potential.center[0], abs=1e-6)
    # The center-of-mass mode moves every ion equally, so each has a Lamb-Dicke parameter eta_0 / sqrt(N)
    k = 2 * np.pi / 729e-9
    eta = crystal.lamb_dicke([k, 0.0, 0.0], [crystal.modes_along("x")[0]])
    eta_0 = k * np.sqrt(tiopt.lib.HBAR / (2 * crystal.ions[0].mass * OMEGA))
    assert np.abs(eta[:, 0]) == pytest.approx(eta_0 / np.sqrt(3), rel=1e-6)


def test_zigzags_and_mixed_species():
    potential = crystals.HarmonicPotential("40Ca+", (OMEGA, 1.2 * OMEGA, 3 * OMEGA))
    zigzag = crystals.find_crystal(potential, "40Ca+", n_ions=6)
    assert zigzag.stable
    assert np.ptp(zigzag.positions[:, 1]) > 1.0
    assert np.ptp(zigzag.positions[:, 2]) < 1e-6
    mixed = crystals.find_crystal(crystals.HarmonicPotential("40Ca+", (OMEGA, 5 * OMEGA, 5.5 * OMEGA)), ["40Ca+", "9Be+", "40Ca+"])
    assert mixed.stable
    # The lighter beryllium ion sits in the middle, and its radial modes are much stiffer
    assert mixed.positions[1, 0] == pytest.approx(mixed.positions[:, 0].mean(), abs=1e-6)
    assert mixed.frequencies.max() > 4 * 5.5 * OMEGA


def test_crystals_in_a_trap(well):
    trap, ion, voltages, null = well
    crystal = crystals.find_crystal(trap, ion, n_ions=4, voltages=voltages, position=null)
    assert crystal.stable and crystal.n_outside == 0
    axial, _ = crystals.chain_frequencies(4, OMEGA, 4.5 * OMEGA)
    # The well of the trap is a little anharmonic, so the modes of the crystal are close to those of an ideal chain
    assert crystal.frequencies[crystal.modes_along("x")] == pytest.approx(axial, rel=5e-3)
    assert np.abs(crystal.positions[:, 1]).max() < 1e-3
    assert crystal.participation().sum(axis=1) == pytest.approx(np.ones(12))
    assert len(crystal.summary()) >= 4


def test_coulomb_derivatives():
    rng = np.random.default_rng(0)
    positions = rng.normal(size=(4, 3)) * 5
    charges = np.full(4, tiopt.lib.E_CHARGE)
    energy, gradient, H = crystals.coulomb(positions, charges, hessian=True)
    step = 1e-5
    for k in range(12):
        shifted = positions.copy().reshape(-1)
        shifted[k] += step
        plus = crystals.coulomb(shifted.reshape(4, 3), charges)
        shifted[k] -= 2 * step
        minus = crystals.coulomb(shifted.reshape(4, 3), charges)
        assert gradient.reshape(-1)[k] == pytest.approx((plus[0] - minus[0]) / (2 * step), rel=1e-6)
        assert H[k] == pytest.approx((plus[1] - minus[1]).reshape(-1) / (2 * step), rel=1e-4, abs=1e-6 * np.abs(H).max())
