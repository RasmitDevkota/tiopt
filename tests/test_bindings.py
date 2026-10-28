"""Tests of the ctypes bindings themselves: structure layouts, constants, memory ownership and file round trips."""

import ctypes

import numpy as np
import pytest

import tiopt
from tiopt import lib


def test_library_loads():
    path = tiopt.library_path()
    assert path.exists()
    assert ctypes.CDLL(str(path)).tiopt_struct_size is not None


@pytest.mark.parametrize("name", sorted(lib.STRUCTURES))
def test_structure_sizes_match_library(name):
    assert ctypes.sizeof(lib.STRUCTURES[name]) == lib.struct_size(name)


def test_unknown_structure_and_constant():
    assert lib.struct_size("NotAStructure") == -1
    with pytest.raises(KeyError):
        lib.constant("NOT_A_CONSTANT")


def test_constants():
    assert tiopt.ION_HEIGHT == 55.0
    assert tiopt.LENGTH_SCALE == 1e-6
    assert tiopt.constant("LMAX") == 10
    # The sphere lattice covers the trap center at the height of the ions
    x_min = tiopt.constant("SPH_CENTER_X_MIN")
    x_max = x_min + (tiopt.constant("NSPH_X") - 1) * tiopt.constant("SPH_SPACING")
    assert x_min < 0 < x_max


@pytest.mark.parametrize("element, mass_u", [("40Ca+", 39.962591), ("88Sr+", 87.905612), ("171Yb+", 170.936323), ("9Be+", 9.012183)])
def test_ion_presets(element, mass_u):
    ion = tiopt.Ion(element)
    assert ion.mass == pytest.approx(mass_u * 1.66053906660e-27)
    assert ion.charge == pytest.approx(1.602176634e-19)
    assert ion.qubit_frequency > 0


def test_unknown_ion():
    with pytest.raises(ValueError):
        tiopt.Ion("12C+")


def test_trap_preset_fields():
    trap = tiopt.Trap.from_preset(5, 40.0, gratings=True)
    assert (trap.n_electrodes, trap.n_rf, trap.n_dc) == (13, 2, 11)
    assert trap.V_RF == 40.0
    assert trap.Omega_RF == pytest.approx(2 * np.pi * 40e6)
    polygons = trap.electrode_polygons()
    assert len(polygons) == 13
    assert all(polygon.shape == (4, 3) for polygon in polygons)
    # Every electrode lies in the trap surface
    assert all(np.all(polygon[:, 2] == 0.0) for polygon in polygons)
    gratings = trap.grating_couplers()
    assert len(gratings) == 1
    assert gratings[0].period > 0


def test_unsolved_trap_raises():
    trap = tiopt.Trap.from_preset(3, 50.0)
    with pytest.raises(tiopt.TiOptError):
        trap.potential([[0.0, 0.0, 55.0]])


def test_trap_file_round_trip(tmp_path):
    trap = tiopt.Trap.from_preset(4, 45.0, gratings=True)
    path = tmp_path / "trap.data"
    trap.save(path)
    loaded = tiopt.Trap.from_file(path)
    assert loaded.n_electrodes == trap.n_electrodes
    for a, b in zip(trap.electrode_polygons(), loaded.electrode_polygons()):
        np.testing.assert_array_equal(a, b)
    assert loaded.grating_couplers()[0].period == trap.grating_couplers()[0].period


def test_read_repository_trap_file(repository):
    trap = tiopt.Trap.from_file(repository / "trap.data")
    assert (trap.n_electrodes, trap.n_rf, trap.n_dc) == (2, 0, 2)
    np.testing.assert_array_equal(trap.electrode_positions(), [[25, 0, 0], [50, 0, 0]])


def test_invalid_trap_file(tmp_path):
    path = tmp_path / "bad.data"
    path.write_text("trap n_electrodes 1\nelectrode n_vertices 0 3\nelectrode v 0 5 (1 2 3)\n")
    with pytest.raises(tiopt.TiOptError):
        tiopt.Trap.from_file(path)


def test_trap_from_polygons():
    square = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
    trap = tiopt.Trap.from_polygons([square], n_rf=0)
    assert trap.n_electrodes == 1
    np.testing.assert_allclose(trap.electrode_polygons()[0][:, :2], square)
    # Directly above the center of a square of side a at height h, the solid angle is 4 asin(a^2 / (a^2 + 4 h^2))
    h = 30.0
    phi = trap.analytical_electrode_potential(0, [[0.0, 0.0, h]])[0]
    assert phi == pytest.approx(4 * np.arcsin(400 / (400 + 4 * h * h)) / (2 * np.pi), rel=1e-12)


def test_waveform_memory_is_released():
    waveform = tiopt.Waveform.empty(10, 3, 2)
    assert waveform.voltages.shape == (10, 3)
    assert waveform.well_positions.shape == (10, 2, 3)
    waveform.close()
    waveform.close()


def test_photonics_preset_and_detector_round_trip(tmp_path):
    trap = tiopt.Trap.from_preset(5, 40.0, photonics=True)
    gratings = trap.grating_couplers()
    detectors = trap.detectors()
    assert len(gratings) == len(tiopt.PHOTONICS_BEAMS)
    assert len(detectors) == 1
    wavelengths = sorted(g.struct.waveguide.wavelength for g in gratings)
    assert wavelengths == pytest.approx([0.396959, 0.729347, 0.854209, 0.866214])
    # The detector lies beneath the measurement zone, two segments from the center
    assert detectors[0].position == pytest.approx([80.0, 0.0, 0.0])

    detectors[0].dark_count_rate = 3.5
    path = tmp_path / "photonics.trap"
    trap.save(path)
    read = tiopt.Trap.from_file(path)
    assert read.detectors()[0].dark_count_rate == 3.5
    assert read.detectors()[0].position == pytest.approx(detectors[0].position)
    assert len(read.grating_couplers()) == len(gratings)


def test_add_photonics_replaces_gratings():
    trap = tiopt.Trap.from_preset(5, 40.0, gratings=True)
    assert len(trap.grating_couplers()) == 1
    trap.add_photonics((10.0, 0.0, tiopt.ION_HEIGHT), (-60.0, 0.0, tiopt.ION_HEIGHT))
    assert len(trap.grating_couplers()) == 4
    assert trap.detectors()[0].position == pytest.approx([-60.0, 0.0, 0.0])
