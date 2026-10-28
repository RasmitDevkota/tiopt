"""Tests of the electrodynamics module through the Python bindings, against independent calculations."""

import ctypes
import pathlib

import numpy as np
import pytest
from scipy import integrate

import tiopt
from tiopt import lib

libname = pathlib.Path(__file__).resolve().parent / "../libtiopt.so"
libtiopt = ctypes.CDLL(str(libname))


def test_gapless_potential_matches_poisson_integral():
    # In the gapless plane approximation, phi(r) = (z / 2 pi) int dA' / |r - r'|^3 over the electrode
    rectangle = [(-15.0, -5.0), (25.0, -5.0), (25.0, 10.0), (-15.0, 10.0)]
    trap = tiopt.Trap.from_polygons([rectangle], n_rf=0)

    for point in [(0.0, 0.0, 10.0), (30.0, 20.0, 55.0), (-40.0, 3.0, 5.0)]:
        x, y, z = point
        integral, _ = integrate.dblquad(lambda yp, xp: z / ((x - xp) ** 2 + (y - yp) ** 2 + z ** 2) ** 1.5, -15.0, 25.0, -5.0, 10.0, epsabs=1e-13, epsrel=1e-12)
        assert trap.analytical_electrode_potential(0, [point])[0] == pytest.approx(integral / (2 * np.pi), rel=1e-8)


def test_infinite_electrode_limit():
    # A very large electrode is the whole plane, so the potential just above it is 1 V
    big = [(-1e7, -1e7), (1e7, -1e7), (1e7, 1e7), (-1e7, 1e7)]
    trap = tiopt.Trap.from_polygons([big], n_rf=0)
    assert trap.analytical_electrode_potential(0, [[0.0, 0.0, 50.0]])[0] == pytest.approx(1.0, abs=1e-5)


def test_expansion_matches_gapless_potential_near_ion_height(solved_trap):
    rng = np.random.default_rng(0)
    points = np.column_stack([rng.uniform(-100, 100, 300), rng.uniform(-30, 30, 300), rng.uniform(45, 62, 300)])
    for e in [0, 2, 5, 12]:
        weights = solved_trap.zero_voltages()
        weights[e] = 1.0
        expansion = solved_trap.potential(points, weights)
        exact = solved_trap.analytical_electrode_potential(e, points)
        np.testing.assert_allclose(expansion, exact, rtol=2e-4, atol=1e-7)


def test_gradient_matches_finite_differences(solved_trap):
    rng = np.random.default_rng(1)
    voltages = rng.normal(size=solved_trap.n_electrodes)
    points = np.column_stack([rng.uniform(-80, 80, 20), rng.uniform(-20, 20, 20), rng.uniform(48, 60, 20)])
    _, gradient = solved_trap.potential(points, voltages, gradient=True)
    h = 1e-3
    for k in range(3):
        step = np.zeros(3)
        step[k] = h
        numerical = (solved_trap.potential(points + step, voltages) - solved_trap.potential(points - step, voltages)) / (2 * h)
        np.testing.assert_allclose(gradient[:, k], numerical, rtol=1e-6, atol=1e-10)


def test_hessian_is_traceless_and_symmetric(solved_trap):
    rng = np.random.default_rng(2)
    voltages = rng.normal(size=solved_trap.n_electrodes)
    _, _, H = solved_trap.evaluate([12.0, -3.0, 54.0], voltages)
    scale = np.abs(H).max()
    assert abs(np.trace(H)) < 1e-12 * scale
    np.testing.assert_allclose(H, H.T, atol=1e-14 * scale)


def test_rf_null_at_ion_height(solved_trap):
    null = solved_trap.find_rf_null((0.0, 1.0, 50.0))
    np.testing.assert_allclose(null, [0.0, 0.0, tiopt.ION_HEIGHT], atol=1e-3)
    # The pseudopotential vanishes on the null
    assert solved_trap.pseudopotential(tiopt.Ion("40Ca+"), [null])[0] < 1e-30


def test_normal_modes_match_finite_difference_curvature(solved_trap, ion):
    null = solved_trap.find_rf_null((0.0, 0.0, None))
    waveform = tiopt.Waveform.transport(solved_trap, ion, null, null, 0.0, 1)
    voltages = waveform.voltages[0]
    omega, axes = solved_trap.normal_modes(ion, null, voltages)

    # Independent curvature from second differences of the potential energy
    h = 0.05
    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            e_i = np.eye(3)[i] * h
            e_j = np.eye(3)[j] * h
            points = np.array([null + e_i + e_j, null + e_i - e_j, null - e_i + e_j, null - e_i - e_j])
            U = solved_trap.potential_energy(ion, points, voltages)
            H[i, j] = (U[0] - U[1] - U[2] + U[3]) / (4 * h * h)
    k = np.linalg.eigvalsh(H / tiopt.LENGTH_SCALE ** 2)
    np.testing.assert_allclose(np.sort(omega), np.sort(np.sqrt(k / ion.mass)), rtol=1e-3)
    # The well has the default axial frequency along the trap axis
    axial = np.argmax(np.abs(axes[:, 0]))
    assert omega[axial] == pytest.approx(tiopt.AXIAL_FREQUENCY, rel=1e-4)


def test_relaxation_solver_through_ctypes():
    # Call the relaxation solver directly on a small grid owned by numpy
    NX, NY, NZ = 81, 81, 61
    trap = tiopt.Trap.from_polygons([[(-10, -10), (10, -10), (10, 10), (-10, 10)]], n_rf=0)
    V = np.zeros((NX, NY, NZ))
    position = np.zeros(3)
    electrode = ctypes.byref(trap.struct.electrodes[0])
    iterations = libtiopt.solver_relaxation(
        ctypes.c_int(NX), ctypes.c_int(NY), ctypes.c_int(NZ), electrode,
        position.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), V.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(1), ctypes.c_int(1), ctypes.c_int(1), ctypes.c_int(10000), ctypes.c_int(1),
    )
    lib.flush_c_output()
    assert 0 < iterations < 10000
    # The electrode is held at 1 V, half covered cells on its edges at 1/2 V, and the rest of the boundary at 0 V
    assert V[NX // 2, NY // 2, 0] == 1.0
    assert V[NX // 2 + 10, NY // 2, 0] == 0.5
    assert V[0, 0, 0] == 0.0
    # Close to the electrode, the grounded walls barely matter
    z = np.arange(2, 8)
    exact = trap.analytical_electrode_potential(0, np.column_stack([np.zeros_like(z), np.zeros_like(z), z]))
    np.testing.assert_allclose(V[NX // 2, NY // 2, 2:8], exact, rtol=2e-2)
