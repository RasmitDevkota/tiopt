"""Tests of trap design through the Python bindings: zone layouts, the preset library, photonics for every zone and
species, design reports, junction barriers, and layout files."""

import struct

import numpy as np
import pytest

import tiopt
from tiopt import export

LAYOUT = "load, storage, gate x2, measure"


@pytest.fixture(scope="module")
def layout_trap():
    trap = tiopt.Trap.from_layout(LAYOUT, zone_spacing=80.0)
    trap.solve()
    return trap


@pytest.fixture(scope="module")
def x_junction():
    trap = tiopt.Trap.from_preset(2, 40.0, preset="x_junction")
    trap.solve()
    return trap


def test_parse_layout():
    zones = tiopt.parse_layout("Load+Storage, empty, gate:3 x2, measure*2", n_ions=4)
    assert zones == [("load+storage", 4), ("empty", 1), ("gate", 3), ("gate", 3), ("measure", 4), ("measure", 4)]
    for layout in ("gate, warp", "gate x0", "", "load,,gate"):
        with pytest.raises(ValueError):
            tiopt.parse_layout(layout)


def test_trap_from_layout(layout_trap):
    zones = layout_trap.zones()
    assert [zone.types for zone in zones] == ["load", "storage", "gate", "gate", "measure"]
    np.testing.assert_allclose([zone.position[0] for zone in zones], [-160, -80, 0, 80, 160])
    # The loading zone gets the photoionization and cooling beams, each gate zone the qubit and quench beams, and the
    # measurement zone the detection and repump beams and a detector
    beams = sorted((grating.beam, grating.zone) for grating in layout_trap.grating_couplers())
    assert beams == sorted([
        ("detection", 0), ("repump", 0), ("photoionization", 0), ("ionization", 0),
        ("qubit", 2), ("quench", 2), ("qubit", 3), ("quench", 3),
        ("detection", 4), ("repump", 4),
    ])
    assert [detector.zone for detector in layout_trap.detectors()] == [4]
    assert layout_trap.grating("qubit", zone=3).zone == 3
    assert layout_trap.grating("qubit", zone=0) is None
    assert layout_trap.detector(4) is not None


def test_set_zones_and_layout():
    trap = tiopt.Trap.from_preset(7, 50.0)
    assert trap.zones() == []
    trap.set_zones([("gate", 0, (0.0, 0.0, None)), tiopt.Zone("measure", 3, np.array([100.0, 0.0, 60.0]))])
    zones = trap.zones()
    assert [(zone.types, zone.capacity) for zone in zones] == [("gate", 2), ("measure", 3)]
    assert zones[0].position[2] == tiopt.ION_HEIGHT and zones[1].position[2] == 60.0
    trap.add_zone_photonics("88Sr+")
    assert {grating.beam for grating in trap.grating_couplers()} == {"qubit", "quench", "detection", "repump"}
    assert trap.grating("qubit").struct.waveguide.wavelength == pytest.approx(0.674025)
    trap.set_layout("load+storage, gate, measure", zone_spacing=60.0)
    np.testing.assert_allclose([zone.position[0] for zone in trap.zones()], [-60, 0, 60])
    with pytest.raises(ValueError):
        trap.set_zones([("warp", 0, (0.0, 0.0, None))])


@pytest.mark.parametrize("element", tiopt.Ion.PRESETS)
def test_photonics_for_every_species(element):
    trap = tiopt.Trap.from_layout("load+gate+measure", element=element)
    beams = {grating.beam for grating in trap.grating_couplers()}
    expected = {beam for beam in tiopt.PHOTONICS_BEAMS if tiopt.photonics_beam(element, beam) is not None}
    assert beams == expected
    # 9Be+ has no metastable D states to quench, and photoionizes with two photons from one beam
    assert ("quench" in beams) == (element != "9Be+")
    for grating in trap.grating_couplers():
        waveguide, power = tiopt.photonics_beam(element, grating.beam)
        assert grating.struct.waveguide.wavelength == pytest.approx(waveguide.wavelength)
        assert grating.period > 0


@pytest.mark.parametrize("preset, n_rf, n_dc", [("x_junction", 4, 17), ("y_junction", 3, 13), ("t_junction", 3, 13), ("point", 1, 3)])
def test_presets(preset, n_rf, n_dc):
    trap = tiopt.Trap.from_preset(2, 40.0, preset=preset)
    assert (trap.n_rf, trap.n_dc) == (n_rf, n_dc)
    assert len(trap.zones()) == 1
    with pytest.raises(ValueError):
        tiopt.Trap.from_preset(2, 40.0, gratings=True, preset=preset)
    with pytest.raises(ValueError):
        tiopt.Trap.from_preset(2, 40.0, preset="spiral")


def test_point_trap_photonics_and_confinement(ion):
    trap = tiopt.Trap.from_preset(4, 40.0, preset="point", photonics=True)
    assert len(trap.grating_couplers()) == 4 and len(trap.detectors()) == 1
    trap.solve()
    null = trap.find_rf_null((0.0, 0.0, None))
    assert null[2] == pytest.approx(tiopt.ION_HEIGHT, abs=0.5)
    omega, _ = trap.normal_modes(ion, null)
    assert np.all(omega > 0)


def test_custom_junction(ion):
    # A five-way junction, with arms 72 degrees apart
    trap = tiopt.Trap.junction([0, 72, 144, 216, 288], n_segments=1, segment_width=40.0)
    assert (trap.n_rf, trap.n_dc) == (5, 11)
    trap.solve()
    direction = np.array([np.cos(np.radians(72)), np.sin(np.radians(72)), 0.0])
    null = trap.rf_null_across(150 * direction + [0, 0, tiopt.ION_HEIGHT], direction)
    assert null[2] == pytest.approx(tiopt.ION_HEIGHT, abs=5.0)
    with pytest.raises(tiopt.TiOptError):
        tiopt.Trap.junction([0, 90])


def test_junction_barrier(ion, x_junction, solved_trap):
    waypoints = [(150.0, 0.0, tiopt.ION_HEIGHT), (0.0, 0.0, tiopt.ION_HEIGHT), (0.0, 150.0, tiopt.ION_HEIGHT)]
    positions, energies, barrier = x_junction.rf_null_path(ion, waypoints, 61)
    assert positions.shape == (61, 3) and energies.shape == (61,)
    assert barrier == pytest.approx(energies.max() - min(energies[0], energies[-1]))
    assert barrier > 1e-3
    # Along the axis of a linear trap there is no barrier
    _, _, linear_barrier = solved_trap.rf_null_path(ion, [(-100.0, 0.0, tiopt.ION_HEIGHT), (100.0, 0.0, tiopt.ION_HEIGHT)], 41)
    assert abs(linear_barrier) < 1e-6


def test_design_report(ion, layout_trap):
    report = layout_trap.design_report(ion)
    assert [entry["types"] for entry in report] == ["load", "storage", "gate", "gate", "measure"]
    gate = report[2]
    assert gate["rf_null"][2] == pytest.approx(tiopt.ION_HEIGHT, abs=0.1)
    assert gate["pseudopotential"] < 1e-6
    assert 0 < gate["mathieu_q"] < 1 and gate["trap_depth"] > 0
    qubit = gate["beams"]["qubit"]
    assert qubit["rabi_frequency"] > 0 and 0 < qubit["lamb_dicke_parameter"] < 1 and qubit["crosstalk"] < 1e-3
    assert set(report[0]["beams"]) == {"detection", "repump", "photoionization", "ionization"}
    assert 0.5 < report[4]["detection_fidelity"] <= 1 and 0 < report[4]["collection_efficiency"] < 0.5
    text = tiopt.format_design_report(report)
    assert "Zone 2 (gate)" in text and "Rabi frequency" in text and "detection fidelity" in text


def test_design_report_outside_expansions(ion):
    trap = tiopt.Trap.from_layout("load, storage x2, gate x5, measure", photonics=False)
    trap.solve()
    report = trap.design_report(ion)
    outside = [entry for entry in report if entry.get("outside_expansions")]
    assert outside and all(entry["required_nsph_x"] > tiopt.constant("NSPH_X") for entry in outside)
    assert "rebuild with make clean" in tiopt.format_design_report(report)
    x_min, x_max, _, _ = tiopt.expansion_region()
    assert tiopt.required_nsph_x(x_min, x_max) == tiopt.constant("NSPH_X")


def test_trap_file_keeps_zones(layout_trap, tmp_path):
    path = tmp_path / "layout.trap"
    layout_trap.save(path)
    loaded = tiopt.Trap.from_file(path)
    assert [(zone.types, zone.capacity) for zone in loaded.zones()] == [(zone.types, zone.capacity) for zone in layout_trap.zones()]
    assert [(g.beam, g.zone) for g in loaded.grating_couplers()] == [(g.beam, g.zone) for g in layout_trap.grating_couplers()]


def test_gds_round_trip(layout_trap, tmp_path):
    path = tmp_path / "layout.gds"
    layout_trap.save_gds(path)
    polygons, texts = export.read_gds(path)
    assert len(polygons[export.LAYERS["rf"]]) == layout_trap.n_rf
    assert len(polygons[export.LAYERS["dc"]]) == layout_trap.n_dc
    assert len(polygons[export.GRATING_LAYERS["qubit"]]) == 2
    assert len(polygons[export.LAYERS["detector"]]) == 1
    assert [string for _, string, _ in texts] == ["load", "storage", "gate", "gate", "measure"]

    loaded = tiopt.Trap.from_gds(path)
    assert (loaded.n_rf, loaded.n_dc) == (layout_trap.n_rf, layout_trap.n_dc)
    for a, b in zip(layout_trap.electrode_polygons(), loaded.electrode_polygons()):
        np.testing.assert_allclose(a, b, atol=1e-3)
    np.testing.assert_allclose([z.position[0] for z in loaded.zones()], [z.position[0] for z in layout_trap.zones()], atol=1e-3)


def test_gds_reals_and_skipped_elements(tmp_path):
    for value in (1e-3, 1e-9, 0.5, -2.75, 123456.0):
        assert export._from_real8(export._real8(value)) == pytest.approx(value, rel=1e-14)
    # A reference to another cell is skipped with a warning
    data = export._record(export._HEADER, export._INT16, struct.pack(">h", 600))
    data += export._record(export._UNITS, export._REAL8, export._real8(1e-3) + export._real8(1e-9))
    data += export._boundary(1, [(0, 0), (10, 0), (10, 10)])
    data += export._record(export._SREF, 0) + export._record(export._ENDEL, 0)
    data += export._record(export._ENDLIB, 0)
    path = tmp_path / "hierarchy.gds"
    path.write_bytes(data)
    with pytest.warns(UserWarning, match="cell references"):
        polygons, _ = export.read_gds(path)
    np.testing.assert_allclose(polygons[1][0], [(0, 0), (10, 0), (10, 10)])
    with pytest.raises(tiopt.TiOptError), pytest.warns(UserWarning):
        tiopt.Trap.from_gds(path, rf_layers=(5,), dc_layers=(6,))


def test_svg(layout_trap, tmp_path):
    path = tmp_path / "layout.svg"
    layout_trap.save_svg(path)
    text = path.read_text()
    assert text.startswith("<svg") and text.count("<polygon") == layout_trap.n_electrodes + 10 + 1
    assert text.count("<circle") == 5


def test_trap_graphs(layout_trap):
    graph = tiopt.TrapGraph.from_layout("load+storage, gate x2, measure", zone_spacing=100.0, n_ions=3)
    np.testing.assert_allclose(graph.positions[:, 0], [-150, -50, 50, 150])
    assert graph.capacities == [3, 2, 2, 3]
    assert graph.zone_types == ["load+storage", "gate", "gate", "measure"]
    from_trap = layout_trap.zone_graph()
    assert from_trap.zone_types == ["load", "storage", "gate", "gate", "measure"]
    np.testing.assert_allclose(from_trap.positions[:, 0], [-160, -80, 0, 80, 160])


def test_parallel_gate_zones(repository):
    # The two gates of the second layer each run in a gate zone of their own
    circuit = tiopt.Circuit(repository / "tests" / "quantumcircuits" / "test_0000.qasm")
    graph = tiopt.TrapGraph.from_layout("load+storage, gate, gate, measure", zone_spacing=100.0, n_ions=4)
    program = tiopt.TransportProgram(graph, circuit)
    layers = [i.layer for i in circuit.instructions]
    second_layer = program.destinations[np.array(layers) == 2]
    assert len(second_layer) == 2 and second_layer[0, 0] != second_layer[1, 0]


def test_codesign_layout(repository):
    codesign = tiopt.Codesign(repository / "examples" / "circuits" / "bell.qasm")
    assert codesign.zone_layout == tiopt.lib.DEFAULT_LAYOUT
    codesign.close()
    codesign = tiopt.Codesign(repository / "examples" / "circuits" / "bell.qasm", layout="load+storage, gate+measure")
    assert codesign.zone_layout == "load+storage, gate+measure"
    assert [zone.types for zone in codesign.trap().zones()] == ["load+storage", "gate+measure"]
    assert 0 < codesign.metrics["circuit_fidelity"] <= 1
    codesign.close()
    with pytest.raises(tiopt.TiOptError):
        tiopt.Codesign(repository / "examples" / "circuits" / "bell.qasm", layout="load+storage, measure")
