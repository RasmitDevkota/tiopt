"""Tests of the full co-design of a system (tiopt.system)."""

import numpy as np
import pytest

import tiopt
from tiopt import circuits, system


@pytest.fixture(scope="module")
def layout_trap():
    trap = tiopt.Trap.from_layout(tiopt.lib.DEFAULT_LAYOUT, 100.0, 50.0, "40Ca+", True)
    trap.solve("analytical")
    return trap


def test_evaluation_follows_the_light_to_the_circuit(layout_trap, ion):
    inputs = system.SystemInputs(trap=layout_trap, ion=ion, circuit=circuits.parse_qasm(circuits.PRESETS["Bell pair"]))
    codesign = system.SystemCodesign(inputs)
    evaluation = codesign.evaluate(codesign.initial_parameters())
    m = evaluation.metrics
    # The analytical pseudopotential agrees with the normal modes of the C library
    null = layout_trap.find_rf_null((0.0, 0.0, None))
    layout_trap.V_RF, layout_trap.Omega_RF = inputs.V_RF, inputs.Omega_RF
    omega, _ = layout_trap.normal_modes(ion, null)
    assert m["radial_frequency"] == pytest.approx(np.sort(omega)[1], rel=0.01)
    # Light is lost on the way to the grating, and the gates take as long as the light allows
    assert m["grating_power"] == pytest.approx(inputs.qubit_power * 10 ** (-m["route_loss"] / 10))
    assert m["route_loss"] > inputs.coupling_loss
    assert m["two_qubit_time"] == pytest.approx(np.pi / (m["lamb_dicke_parameter"] * m["rabi_frequency"] / np.sqrt(2)), rel=1e-6)
    assert 0 < m["circuit_fidelity"] < 1
    assert sum(evaluation.budget.values()) == pytest.approx(-np.log(m["circuit_fidelity"]))
    # The trap of the inputs is left alone
    assert layout_trap.V_RF == inputs.V_RF
    codesign.close()


def test_codesign_improves_the_circuit(layout_trap, ion):
    inputs = system.SystemInputs(trap=layout_trap, ion=ion, circuit=circuits.parse_qasm(circuits.PRESETS["Bell pair"]), pulse_shape="cosine", ramp=2e-6)
    codesign = system.SystemCodesign(inputs, groups=("pulses", "waveforms", "gratings"))
    best = codesign.run(max_evaluations=60)
    assert best.objective < codesign.initial.objective
    assert best.metrics["circuit_fidelity"] > codesign.initial.metrics["circuit_fidelity"]
    assert {v.group for v in codesign.variables} == {"pulses", "waveforms", "gratings"}
    trap = codesign.best_trap()
    assert trap.solved and trap.n_electrodes == layout_trap.n_electrodes
    trap.close()
    codesign.close()


def test_codesign_needs_a_gate_zone(ion):
    trap = tiopt.Trap.from_preset(7, 50.0)
    trap.solve("analytical")
    with pytest.raises(ValueError, match="gate zone"):
        system.SystemCodesign(system.SystemInputs(trap=trap, ion=ion, circuit=circuits.parse_qasm(circuits.PRESETS["Bell pair"])))
