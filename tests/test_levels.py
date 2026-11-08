"""Tests of the level structure of the species and the couplings of beams to it (tiopt.levels)."""

from fractions import Fraction

import numpy as np
import pytest

import tiopt
from tiopt import levels


def test_lande_factors_and_clebsch_gordan():
    assert levels.lande_g("S1/2") == pytest.approx(2.0023, abs=1e-4)
    assert levels.lande_g("D5/2") == pytest.approx(1.2005, abs=1e-4)
    assert levels.lande_g("P1/2") == pytest.approx(2 / 3, abs=1e-3)
    assert levels.clebsch_gordan(1, 0, 1, 0, 2, 0) == pytest.approx(np.sqrt(2 / 3))
    assert levels.clebsch_gordan(1, 0, 1, 0, 0, 0) == pytest.approx(-1 / np.sqrt(3))
    assert levels.clebsch_gordan(0.5, 0.5, 2, 2, 2.5, 2.5) == pytest.approx(1.0)


def test_selection_rules_of_the_quadrupole_transition():
    ion = tiopt.Ion("40Ca+")
    B = 4 * levels.GAUSS

    def driven(k, polarization):
        return sorted({line.delta_m for line in levels.qubit_lines(ion, B, k, polarization) if line.strength > 1e-9})

    assert driven([0, 0, 1], [1, 0, 0]) == [-1, 1]
    assert driven([1, 0, 0], [0, 0, 1]) == [-1, 1]
    assert driven([1, 0, 0], [0, 1, 0]) == [-2, 2]
    assert driven([1, 0, 1], [1, 0, -1]) == [-2, 0, 2]
    lines = levels.qubit_lines(ion, B, [1, 0, 1], levels.polarization_vector([1, 0, 1], 45), intensity=1e8)
    assert len(lines) == 10
    # The S-1/2 -> D-5/2 line is shifted by mu_B B (g_D (-5/2) - g_S (-1/2)) / h
    line = next(l for l in lines if l.m == Fraction(-1, 2) and l.m_prime == Fraction(-5, 2))
    assert line.shift == pytest.approx(levels.MU_B_OVER_H * B * (levels.lande_g("D5/2") * -2.5 + levels.lande_g("S1/2") * 0.5))
    assert all(l.rabi_frequency >= 0 for l in lines)
    # The same light with unit coupling has the Rabi frequency of the two-level formula of the C library within a factor sqrt(5)
    assert levels.quadrupole_rabi_frequency(ion, 1e8, 1.0) == pytest.approx(np.sqrt(5) * tiopt.rabi_frequency_from_intensity(ion.qubit_linewidth, ion.qubit_wavelength, 1e8), rel=1e-6)


def test_hyperfine_clock_transitions():
    expected = {"171Yb+": 12.642812118e9, "43Ca+": 3.2256082864e9, "9Be+": 1.250017e9, "133Ba+": 9.92545e9, "137Ba+": 8.03774e9, "25Mg+": 1.7887632e9}
    for element, frequency in expected.items():
        assert levels.hyperfine_transition(element, 0.0)[0] == pytest.approx(frequency, rel=1e-5)
    # The clock transition of 171Yb+ shifts quadratically by 310.8 Hz/G^2
    f0, sensitivity = levels.hyperfine_transition("171Yb+", 1 * levels.GAUSS)
    assert sensitivity * levels.GAUSS == pytest.approx(2 * 310.8, rel=0.01)
    assert len(levels.hyperfine_levels("43Ca+", 1e-4)) == 16
