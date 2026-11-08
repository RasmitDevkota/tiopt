"""Tests of design rule checks and gaps (tiopt.drc), and of the export of potentials to VTK files."""

import numpy as np
import pytest

import tiopt
from tiopt import drc, export


def test_rules_find_violations():
    polygons = [
        np.array([[0, 0], [100, 0], [100, 2], [0, 2]]),
        np.array([[0, 10], [50, 10], [50, 40], [30, 40], [30, 12], [28, 12], [28, 40], [0, 40]]),
        np.array([[0, 3], [100, 3.0], [0, 4.0]]),
    ]
    violations = drc.check_polygons(polygons)
    rules = {(v.rule, v.severity) for v in violations}
    assert ("width", "error") in rules
    assert ("notch", "error") in rules
    assert ("gap", "error") in rules
    assert ("angle", "warning") in rules
    gap = next(v for v in violations if v.rule == "gap")
    assert "1.00 µm" in gap.message
    assert drc.check_polygons([np.array([[0, 0], [50, 0], [50, 50], [0, 50]])]) == []


def test_gaps_for_fabrication():
    trap = tiopt.Trap.from_preset(7, 50.0, gratings=True)
    violations = drc.check_trap(trap)
    assert [v.severity for v in violations] == ["info"]
    gapped = drc.gapped_trap(trap, 5.0)
    assert drc.check_trap(gapped) == []
    square = drc.inset_polygon(np.array([[0, 0], [10, 0], [10, 10], [0, 10]]), 1.0)
    assert drc.polygon_area(square) == pytest.approx(64.0)
    for before, after in zip(trap.electrode_polygons(), gapped.electrode_polygons()):
        assert drc.polygon_area(after[:, :2]) < drc.polygon_area(before[:, :2])


def test_vtk_export(tmp_path):
    ion = tiopt.Ion("40Ca+")
    trap = tiopt.Trap.from_preset(7, 50.0)
    trap.solve("analytical")
    path = tmp_path / "trap.vtk"
    export.write_vtk(trap, path, ion=ion, bounds=(-20, 20, -20, 20, 40, 70), spacing=5.0, electrodes=True)
    lines = path.read_text().splitlines()
    assert lines[0].startswith("# vtk DataFile")
    assert "DIMENSIONS 9 9 7" in lines
    n = 9 * 9 * 7
    assert f"POINT_DATA {n}" in lines
    assert sum(line.startswith("SCALARS") for line in lines) == 2 + trap.n_electrodes
    start = lines.index("SCALARS pseudopotential_meV double 1") + 2
    values = np.array(" ".join(lines[start:start + (n + 7) // 8]).split(), dtype=float)
    assert len(values) == n and values.min() >= 0
