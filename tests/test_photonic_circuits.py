"""Tests of photonic circuits, splitter trees, and waveguide routes (tiopt.photonic_circuits)."""

import numpy as np
import pytest

import tiopt
from tiopt import components, photonic_circuits as pc


def test_directional_couplers():
    stack = components.Stack()
    near = pc.analyze_coupler(stack, 0.6, 0.15, 0.2)
    far = pc.analyze_coupler(stack, 0.6, 0.15, 0.5)
    assert near.n_even > near.n_odd
    assert near.coupling_length < far.coupling_length
    assert near.cross_fraction(near.coupling_length) == pytest.approx(1.0)
    assert near.cross_fraction(near.length_for(0.3)) == pytest.approx(0.3)


def test_circuits_of_components():
    # A Mach-Zehnder interferometer switches all of the light between its outputs with a phase of pi
    for phase, through in ((0.0, 0.0), (np.pi, 1.0), (np.pi / 2, 0.5)):
        circuit = pc.Circuit().add("c1", pc.directional_coupler(0.5)).add("c2", pc.directional_coupler(0.5))
        circuit.add("p", pc.phase_shifter(phase)).add("w", pc.phase_shifter(0.0))
        circuit.connect("c1", "out1", "p", "in").connect("p", "out", "c2", "in1").connect("c1", "out2", "w", "in").connect("w", "out", "c2", "in2")
        circuit.expose("a", "c1", "in1").expose("x", "c2", "out1").expose("y", "c2", "out2")
        assert circuit.transmission("x", "a") == pytest.approx(through, abs=1e-12)
        assert circuit.transmission("x", "a") + circuit.transmission("y", "a") == pytest.approx(1.0)
    # Losses add up in decibels
    circuit = pc.Circuit().add("s", pc.straight(1e4, 1.6, 0.729, 2.0)).add("g", pc.grating(0.4))
    circuit.connect("s", "out", "g", "in").expose("fiber", "s", "in").expose("ion", "g", "beam")
    assert circuit.transmission("ion", "fiber") == pytest.approx(0.4 * 10 ** (-0.2))
    assert abs(pc.ring(10.0, 1.6, 0.729, 0.95, 0.0).S[1, 0]) == pytest.approx(1.0)


def test_splitter_trees():
    for weights in ([1, 1, 1, 1, 1], [3, 1, 1], [1, 2]):
        tree = pc.splitter_tree(weights, kind="y", excess_loss_db=0.0)
        assert tree.outputs == pytest.approx(np.array(weights) / np.sum(weights))
    lossy = pc.splitter_tree([1, 1, 1, 1], kind="coupler", excess_loss_db=0.5)
    assert lossy.outputs.sum() == pytest.approx(10 ** (-1.0 / 10))
    coupler = pc.analyze_coupler(components.Stack(), 0.6, 0.15, 0.3)
    tree = pc.splitter_tree([1, 1, 1], kind="coupler", coupler=coupler)
    assert all(length > 0 for _, _, length in tree.splits)


def test_routes_to_the_gratings_of_a_trap():
    trap = tiopt.Trap.from_layout("load, gate x2, measure")
    routes, crossings = pc.route_trap(trap, margin=200.0, radius=20.0)
    assert len(routes) == len(trap.grating_couplers())
    for route in routes:
        grating = trap.grating_couplers()[route.grating]
        entry = grating.position[:2] - 0.5 * grating.length * np.array([np.cos(grating.azimuth), np.sin(grating.azimuth)])
        assert route.points[-1] == pytest.approx(entry)
        assert route.length >= np.linalg.norm(route.points[-1] - route.points[0]) - 1e-9
        assert len(route.bends) <= 1
    qubit = [route for route in routes if trap.grating_couplers()[route.grating].beam == "qubit"]
    loss = qubit[0].loss(components.Stack(), 0.6, 0.15, 20.0)
    assert 0 < loss < 5
    assert crossings >= 0
