"""Tests of the integrated photonic components (tiopt.components) against analytical limits."""

import numpy as np
import pytest

import tiopt
from tiopt import components


@pytest.fixture
def stack():
    return components.Stack()


def test_slab_matches_analytical_mode(stack):
    # A wide waveguide on a thick oxide is a slab, whose effective index the analytical solver gives
    lossless = components.Stack(n_substrate=stack.n_cladding, box_thickness=3.0)
    result = components.analyze_waveguide(lossless, 20.0, 0.15)
    slab = tiopt.slab_mode(2.0, 1.455, 1.455, 0.15, 0.729, "TE")
    assert result.n_eff.real == pytest.approx(slab, abs=2e-3)


def test_single_mode_cutoff(stack):
    narrow = components.analyze_waveguide(stack, 0.3, 0.15)
    wide = components.analyze_waveguide(stack, 1.5, 0.15)
    assert narrow.n_modes == 1
    assert wide.n_modes > 1


def test_substrate_leakage_falls_with_oxide_thickness(stack):
    thin = components.analyze_waveguide(components.Stack(box_thickness=0.5), 0.6, 0.15)
    thick = components.analyze_waveguide(components.Stack(box_thickness=2.0), 0.6, 0.15)
    assert thin.losses["substrate leakage"] > 100 * thick.losses["substrate leakage"]


def test_sidewall_scattering_scales_with_roughness_squared(stack):
    rough = components.analyze_waveguide(components.Stack(sidewall_roughness=0.004), 0.6, 0.15)
    smooth = components.analyze_waveguide(components.Stack(sidewall_roughness=0.002), 0.6, 0.15)
    assert rough.losses["sidewall scattering"] == pytest.approx(4 * smooth.losses["sidewall scattering"], rel=1e-6)


def test_bend_loss_falls_with_radius_and_euler_sections(stack):
    tight = components.analyze_bend(stack, 0.6, 0.15, 10.0)
    loose = components.analyze_bend(stack, 0.6, 0.15, 30.0)
    euler = components.analyze_bend(stack, 0.6, 0.15, 10.0, euler_fraction=1.0)
    assert tight.loss > loose.loss
    assert euler.transition_loss < tight.transition_loss
    assert tight.effective_radius == pytest.approx(10.0, rel=1e-3)


def test_grating_points_at_target(stack):
    design = components.GratingDesign()
    design.period = components.phase_matched_period(stack, design)
    result = components.analyze_grating(stack, design)
    assert result.emission_angle == pytest.approx(result.target_angle, abs=1e-6)
    assert 0 < result.upward_fraction < 1
    assert 0 < result.efficiency < 1
    # Weaker gratings (shallower etches) radiate less per length
    shallow = components.analyze_grating(stack, components.GratingDesign(period=design.period, etch_depth=0.05))
    assert shallow.strength < result.strength


def test_optimizers_improve_designs(stack):
    start = components.analyze_waveguide(stack, 0.6, 0.15)
    waveguide = components.optimize_waveguide(stack, 0.6, 0.15, thickness_bounds=(0.1, 0.3), max_evaluations=30)
    assert waveguide.n_modes == 1 and waveguide.confinement >= 0.29
    design = components.GratingDesign()
    initial = components.analyze_grating(stack, design)
    optimized = components.optimize_grating(stack, design, max_evaluations=40)
    assert optimized.intensity > initial.intensity


def test_tapers_scatter_less_when_longer(stack):
    abrupt = components.analyze_bend(stack, 0.6, 0.15, 15.0, bend_width=1.0, euler_fraction=0.5, taper_length=0.5)
    gentle = components.analyze_bend(stack, 0.6, 0.15, 15.0, bend_width=1.0, euler_fraction=0.5, taper_length=20.0)
    assert gentle.taper_loss < 0.1 * abrupt.taper_loss
    # The tapers lengthen the path by twice their length, and the whole bend has the bend width
    plain = components.analyze_bend(stack, 0.6, 0.15, 15.0, bend_width=1.0, euler_fraction=0.5)
    assert gentle.length == pytest.approx(plain.length + 40.0, rel=1e-3)
    assert {width for _, (_, _, width) in gentle.sections if width > 0.99} == {1.0}
    assert components.taper_widths(0.5, 1.0, [0.0, 0.5, 1.0], "parabolic") == pytest.approx([0.5, np.sqrt(0.625), 1.0])


def test_non_uniform_gratings(stack):
    design = components.GratingDesign()
    design.period = components.phase_matched_period(stack, design)
    uniform = components.analyze_grating(stack, design)
    apodized = components.analyze_grating(stack, components.GratingDesign(period=design.period, fill_start=0.15))
    chirped = components.analyze_grating(stack, components.GratingDesign(period=design.period, chirp=True))
    assert apodized.fills[0] == pytest.approx(0.15) and apodized.fills[-1] == pytest.approx(0.5)
    # Chirped teeth all point at the target, so the periods shorten towards it and the light focuses along the grating
    assert np.all(np.diff(chirped.periods) > 0) or np.all(np.diff(chirped.periods) < 0)
    assert chirped.intensity > 2 * uniform.intensity
    layout = components.tooth_layout(stack, components.GratingDesign(period=design.period, fill_start=0.15, chirp=True))
    teeth, _ = components.grating_teeth(design, layout)
    assert len(teeth) == design.n_teeth
