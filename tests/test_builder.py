"""Tests of the trap builder (tiopt.builder), which assembles traps from parts."""

import numpy as np
import pytest

import tiopt
from tiopt import builder
from tiopt.builder import Part


def test_overlap_area_of_known_shapes():
    square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    l_shape = np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]])
    assert builder.overlap_area(square, square) == pytest.approx(4.0)
    assert builder.overlap_area(square, l_shape) == pytest.approx(3.0)
    assert builder.overlap_area(l_shape, l_shape) == pytest.approx(5.0)
    assert builder.overlap_area(square, square + [2, 0]) == pytest.approx(0.0, abs=1e-12)


def test_overlaps_between_parts():
    # Sections which meet end to end only share edges, while sections which overlap by 5 µm or cross do not
    assert builder.find_overlaps([Part("linear_section", -125, 0), Part("linear_section", 125, 0)]) == []
    assert builder.find_overlaps([Part("linear_section", -120, 0), Part("linear_section", 125, 0)]) == [(0, 1)]
    assert builder.find_overlaps([Part("rf_rail"), Part("rf_rail", rotation=90)]) == [(0, 1)]
    assert builder.find_overlaps([Part("junction"), Part("point_trap", 600, 0)]) == []
    with pytest.raises(ValueError, match="overlap"):
        builder.build_trap([Part("dc_electrode"), Part("dc_electrode", 10, 0)])


def test_sections_make_a_long_trap():
    # Two 500 µm sections end to end hold the ion close to ION_HEIGHT above the middle, where it would be for infinitely
    # long rails (the ends of the rails pull the RF null down by about 1 µm)
    parts = [Part("linear_section", -250, 0, params=dict(n_segments=10)), Part("linear_section", 250, 0, params=dict(n_segments=10)), Part("zone", 0, 0)]
    trap = builder.build_trap(parts)
    trap.solve("analytical")
    assert (trap.n_rf, trap.n_dc) == (4, 42)
    null = trap.find_rf_null((0.0, 0.0, None))
    assert null[2] == pytest.approx(tiopt.ION_HEIGHT, abs=1.0)
    assert trap.zones()[0].types == "gate"


def test_rotation_and_splitting_up():
    part = Part("linear_section", 10, 20, 90)
    x_min, y_min, x_max, y_max = part.geometry().bounds()
    half_width = 0.5 * builder.CENTER_WIDTH + builder.RAIL_WIDTH + builder.SEGMENT_LENGTH
    assert (x_min, x_max) == pytest.approx((10 - half_width, 10 + half_width))
    assert (y_min, y_max) == pytest.approx((20 - 125, 20 + 125))
    pieces = part.explode()
    assert len(pieces) == len(part.geometry().electrodes) == 13
    combined = builder.combine(pieces).electrodes
    for (a, rf_a), (b, rf_b) in zip(part.geometry().electrodes, combined):
        assert rf_a == rf_b
        assert np.allclose(a, b)


def test_photonics_point_at_nearest_zones():
    parts = [Part("linear_section", params=dict(n_segments=8)), Part("zone", 0, 0, params=dict(types="gate")), Part("zone", 100, 0, params=dict(types="measure")),
             Part("grating", -40, 20, params=dict(beam="qubit")), Part("grating", 140, -20, params=dict(beam="detection")), Part("detector", 100, 0)]
    trap = builder.build_trap(parts, element="40Ca+")
    assert [(g.beam, g.zone) for g in trap.grating_couplers()] == [("qubit", 0), ("detection", 1)]
    assert trap.detectors()[0].zone == 1
    assert trap.detectors()[0].wavelength == pytest.approx(0.396959)
    # Every grating focuses on its zone at the ion height
    trap.solve("analytical")
    qubit = trap.grating("qubit")
    theta, k_hat = qubit.emission()
    direction = np.array([0.0, 0.0, tiopt.ION_HEIGHT]) - qubit.position
    assert np.allclose(k_hat, direction / np.linalg.norm(direction), atol=1e-6)
    with pytest.raises(ValueError, match="does not use"):
        builder.build_trap(parts[:2] + [Part("grating", 0, 30, params=dict(beam="quench"))], element="9Be+")
    with pytest.raises(ValueError, match="zone"):
        builder.build_trap([Part("linear_section"), Part("grating", 0, 30)])


def test_designs_round_trip(tmp_path):
    parts = [Part("junction", 1, 2, 30, dict(arms="0, 120, 240")), Part("zone", 5, 6, 0, dict(types="load+storage", capacity=4)), Part("grating", 7, 8, 0, dict(beam="repump"))]
    path = tmp_path / "design.json"
    builder.save_design(parts, path)
    assert builder.load_design(path) == parts


def test_parts_from_preset_rebuild_the_preset():
    trap = tiopt.Trap.from_preset(5, 50.0, preset="t_junction", photonics=True)
    parts = builder.parts_from_trap(trap)
    rebuilt = builder.build_trap(parts)
    assert (rebuilt.n_rf, rebuilt.n_dc) == (trap.n_rf, trap.n_dc)
    for a, b in zip(trap.electrode_polygons(), rebuilt.electrode_polygons()):
        assert np.allclose(a, b)
    assert [z.types for z in rebuilt.zones()] == [z.types for z in trap.zones()]
    assert len(rebuilt.grating_couplers()) == len(trap.grating_couplers())


def test_curved_sections_join_and_hold_ions(ion):
    bend = Part("curved_section", 0.0, 0.0, 0.0, dict(radius=200.0, angle=180.0, n_segments=6))
    ports = bend.geometry().ports
    assert [np.round(p, 6).tolist() for p, _ in ports] == [[200.0, 0.0], [-200.0, 0.0]]
    assert [round(d) for _, d in ports] == [270, 270]
    # A linear section turned upright joins onto either end without overlapping
    section = Part("linear_section", 200.0, -125.0, 90.0)
    assert builder.find_overlaps([bend, section]) == []
    assert any(np.allclose(p, (200.0, 0.0)) and round(d) == 90 for p, d in section.geometry().ports)
    trap = builder.build_trap([bend, section, Part("zone", 0.0, 200.0), Part("zone", 200.0, -125.0, params=dict(types="storage"))])
    trap.solve("analytical")
    (top, top_energy), (side, side_energy) = builder.zone_nulls(trap, ion)
    # The RF null on the curve is a little lower than on a straight section, and close to the axis of the curve
    assert 45.0 < top[2] < tiopt.ION_HEIGHT and np.hypot(*top[:2]) == pytest.approx(200.0, abs=3.0)
    # The zone on the straight section is 125 µm from the open end of its rails, which pull the null off the zone a little
    assert top_energy < 0.05 and side_energy < 2.0
    with pytest.raises(ValueError, match="radius"):
        Part("curved_section", params=dict(radius=100.0)).geometry()


@pytest.mark.parametrize("name", ["linear_two_sections.json", "curved_bend.json", "racetrack.json"])
def test_example_designs_build(repository, ion, name):
    parts = builder.load_design(repository / "examples" / "designs" / name)
    assert builder.find_overlaps(parts) == []
    trap = builder.build_trap(parts)
    trap.solve("analytical")
    nulls = builder.zone_nulls(trap, ion)
    assert all(null is not None for null in nulls)
    if name == "racetrack.json":
        # A closed loop has no rail ends, so every zone is on the RF null
        assert max(energy for _, energy in nulls) < 1e-3
