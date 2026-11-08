"""Tests of the boundary element method (tiopt.bem), the topology optimization of RF electrodes (tiopt.topology), and the
bindings which they use.
"""

import numpy as np
import pytest
from matplotlib.path import Path as MplPath

import tiopt
from tiopt import bem, lib, topology


def rectangle(x, y, z, x1, x2, y1, y2):
    """Gapless plane potential of a rectangle at 1 V (House 2008)."""
    total = 0.0
    for xi, sx in ((x1, -1), (x2, 1)):
        for yj, sy in ((y1, -1), (y2, 1)):
            R = np.sqrt((xi - x) ** 2 + (yj - y) ** 2 + z ** 2)
            total += sx * sy * np.arctan((xi - x) * (yj - y) / (z * R))
    return total / (2 * np.pi)


def test_panel_integrals():
    polygon = np.array([[0, 0], [3, 0], [4, 2], [1, 3], [-1, 1.5]], dtype=float)
    xs, ys = np.linspace(-1, 4, 1201), np.linspace(0, 3, 1201)
    X, Y = np.meshgrid(xs, ys)
    inside = MplPath(polygon).contains_points(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    dA = (xs[1] - xs[0]) * (ys[1] - ys[0])
    for point in ([1.5, 1.2, 0.7], [5.0, 5.0, 1.0], [10.0, -3.0, 0.0]):
        quadrature = np.sum(inside / np.sqrt((X - point[0]) ** 2 + (Y - point[1]) ** 2 + point[2] ** 2)) * dA
        assert bem.polygon_potential(polygon[None], np.array([point]))[0] == pytest.approx(quadrature, rel=2e-3)
        mesh = bem.mesh_trap([polygon], h_min=10.0, h_max=10.0)
        assert bem._kernel(mesh, np.array([point]), [(0.0, 1.0)], near=1e9)[0].sum() == pytest.approx(bem.polygon_potential(polygon[None], np.array([point]))[0], rel=1e-12)
    panels = bem.mesh_polygon(polygon, 0.5, 2.0)
    assert sum(bem._area_and_centroid(p)[0] for p in panels) == pytest.approx(10.25)


def test_electrodes_in_a_grounded_plane():
    # Electrodes which cover the plane (with a large grounded frame) converge to the gapless plane approximation
    left = np.array([[-100, -50], [0, -50], [0, 50], [-100, 50.0]])
    right = np.array([[0, -50], [100, -50], [100, 50], [0, 50.0]])
    solution = bem.solve_bem([left, right], h_min=2.0, frame=3000.0, frame_gap=0.0)
    points = np.array([[0, 0, 55.0], [-30, 0, 55.0], [0, 0, 20.0]])
    exact = [rectangle(*p, -100, 0, -50, 50) for p in points]
    assert solution.potential(points)[:, 0] == pytest.approx(exact, abs=2e-3)
    # A grounded plane beneath electrodes which cover the plane changes nothing above them, since the potential of the
    # plane is set, while beneath a gap it pulls the potential in the gap (and above it) towards 0 V
    shielded = bem.solve_bem([left, right], h_min=4.0, frame=3000.0, frame_gap=0.0, ground_depth=5.0)
    assert shielded.potential(points)[:, 0] == pytest.approx(solution.potential(points)[:, 0], abs=2e-3)
    gapped = [left - [5.0, 0.0] * (left[:, :1] > -1), right + [5.0, 0.0] * (right[:, :1] < 1)]
    above_gap = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 30.0]])
    open_gap = bem.solve_bem(gapped, h_min=3.0, frame=3000.0).potential(above_gap)[:, 0]
    grounded_gap = bem.solve_bem(gapped, h_min=3.0, frame=3000.0, ground_depth=3.0).potential(above_gap)[:, 0]
    assert np.all(grounded_gap < open_gap - 0.01)
    with pytest.raises(ValueError):
        bem.solve_bem([left, right], h_min=0.1, max_panels=1000)


def test_solving_traps_with_gaps():
    ion = tiopt.Ion("40Ca+")
    analytical = tiopt.Trap.from_preset(7, 50.0)
    analytical.solve("analytical")
    trap = tiopt.Trap.from_preset(7, 50.0)
    solution = bem.solve(trap, gap=5.0)
    assert trap.solved and trap.solver == "bem"
    assert solution.mesh.n_panels < bem.MAX_PANELS
    null = trap.find_rf_null((0.0, 0.0, None))
    reference = analytical.find_rf_null((0.0, 0.0, None))
    # The trap is symmetric, so the RF null stays on its axis, and the gaps move it by about a micrometer
    assert abs(null[1]) < 1e-3
    assert null[2] == pytest.approx(reference[2], abs=1.5)
    assert trap.normal_modes(ion, null)[0][1:] == pytest.approx(analytical.normal_modes(ion, reference)[0][1:], rel=0.06)


def test_expanding_external_samples():
    trap = tiopt.Trap.from_preset(5, 50.0)
    trap.solve("analytical")
    point = trap.find_rf_null((0.0, 0.0, None)) + np.array([3.0, 2.0, 1.0])
    before = trap.potential(point, trap.rf_weights())
    samples = lib.sphere_sample_positions()
    for e in range(trap.n_electrodes):
        trap.expand_samples(e, trap.analytical_electrode_potential(e, samples))
    assert trap.potential(point, trap.rf_weights()) == pytest.approx(before, rel=1e-9)
    U, grad, H, n_outside = trap.energy_derivatives_batch(tiopt.Ion("40Ca+"), np.array([point, point + 1.0]), hessian=True)
    U1, grad1, H1 = trap.energy_derivatives(tiopt.Ion("40Ca+"), point)
    assert n_outside == 0 and U[0] == pytest.approx(U1) and grad[0] == pytest.approx(grad1) and H[0] == pytest.approx(H1)


def test_topology_optimization():
    x0, x1, y0, y1 = np.array([-5.0]), np.array([5.0]), np.array([-3.0]), np.array([4.0])
    assert topology.rectangle_potential(x0, x1, y0, y1, np.array([[1.0, 2.0, 30.0]]))[0, 0] == pytest.approx(rectangle(1.0, 2.0, 30.0, -5, 5, -3, 4))
    result = topology.optimize_topology(half_width=150.0, pixel=5.0, layout="X junction")
    n = result.shape[0]
    pattern = result.pattern.reshape(n, n)
    # The pattern has the 4-fold symmetry of the X junction
    assert np.array_equal(pattern, pattern[::-1, :]) and np.array_equal(pattern, pattern.T)
    assert result.confinement.min() > 0
    trap = result.to_trap()
    for point in result.nulls[:3]:
        null = trap.rf_null_across(point, (1.0, 0.0, 0.0) if abs(point[1]) < 1e-9 else (0.0, 1.0, 0.0))
        assert null[2] == pytest.approx(point[2], abs=1.5)
    linear = topology.optimize_topology(half_width=150.0, pixel=10.0, layout="Linear")
    assert len(linear.polygons) >= 1


def test_tracing_pixels():
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:5, 1:5] = True
    mask[2:4, 2:4] = False
    edges = np.arange(7, dtype=float)
    polygons = topology.trace_pixels(mask, edges, edges)
    assert len(polygons) == 1
    x, y = polygons[0][:, 0], polygons[0][:, 1]
    assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) == pytest.approx(mask.sum())
