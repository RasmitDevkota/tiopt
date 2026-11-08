"""Tests of the free-form optimization of electrode shapes (tiopt.shapes)."""

import numpy as np
import pytest

import tiopt
from tiopt import builder, shapes


@pytest.fixture(scope="module")
def x_junction():
    trap = tiopt.Trap.junction((0, 90, 180, 270), 2, 40.0)
    trap.V_RF = 40.0
    trap.Omega_RF = 2 * np.pi * 40e6
    return trap


def test_refined_edges_stay_shared(x_junction):
    polygons = [shapes.refine_polygon(polygon[:, :2], np.zeros(2), 150.0, 4.0) for polygon in x_junction.electrode_polygons()]
    # Edges which two electrodes share get the same new vertices, so a deformation moves them together
    vertices = [set(map(tuple, np.round(polygon, 9))) for polygon in polygons]
    assert any(len(a & b) > 10 for i, a in enumerate(vertices) for b in vertices[i + 1:])
    deformation = shapes.Deformation((0.0, 0.0), 140.0, 40.0, 10.0, "4-fold")
    u = np.random.default_rng(0).uniform(-1, 1, deformation.n_variables)
    moved = shapes.deformed_polygons(polygons, deformation, u)
    # A deformation with a positive Jacobian can't fold electrodes over each other, and shared vertices stay shared
    assert deformation.min_jacobian(u) > 0
    moved_vertices = [set(map(tuple, np.round(polygon, 9))) for polygon in moved]
    assert sum(len(a & b) for i, a in enumerate(moved_vertices) for b in moved_vertices[i + 1:]) == sum(len(a & b) for i, a in enumerate(vertices) for b in vertices[i + 1:])
    assert all(np.sign(builder._signed_area(a)) == np.sign(builder._signed_area(b)) for a, b in zip(polygons, moved))
    # Nothing moves outside the region, and nothing moves more than max_shift
    shifts = [np.linalg.norm(b - a, axis=1) for a, b in zip(polygons, moved)]
    assert max(s.max() for s in shifts) <= 10.0
    far = np.concatenate([np.hypot(*p.T) > 140.0 for p in polygons])
    assert np.concatenate(shifts)[far].max() < 1e-12


def test_symmetric_deformations(x_junction):
    deformation = shapes.Deformation((0.0, 0.0), 140.0, 40.0, 10.0, "4-fold")
    u = np.random.default_rng(1).uniform(-1, 1, deformation.n_variables)
    point = np.array([[30.0, 12.0]])
    turned = np.array([[-12.0, 30.0]])
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    assert deformation.displacement(turned, u) == pytest.approx(deformation.displacement(point, u) @ rotation.T)


def test_shape_optimization_lowers_the_junction_barrier(x_junction, ion):
    result = shapes.optimize_shapes(x_junction, ion, symmetry="4-fold", objective="barrier", max_evaluations=60)
    assert result.final_path.barrier < result.initial_path.barrier
    assert 0 < result.max_displacement <= 8.0
    assert result.deformation.min_jacobian(result.u) > 0.3
    assert result.trap.solved and result.trap.n_electrodes == x_junction.n_electrodes
