"""Smoke tests of the GUI, which drive every tab's main actions on a virtual display."""

import os
import shutil
import subprocess
import time

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture(scope="module")
def display():
    """A virtual X display from Xvfb if available (so that no windows appear), otherwise the current display."""
    process = None
    if shutil.which("Xvfb"):
        for number in range(90, 110):
            if os.path.exists(f"/tmp/.X{number}-lock"):
                continue
            process = subprocess.Popen(["Xvfb", f":{number}", "-screen", "0", "1600x1000x24", "-nolisten", "tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.5)
            if process.poll() is None:
                previous = os.environ.get("DISPLAY")
                os.environ["DISPLAY"] = f":{number}"
                break
            process = None
    if process is None:
        if not os.environ.get("DISPLAY"):
            pytest.skip("no X display available")
        previous = os.environ["DISPLAY"]
    yield os.environ["DISPLAY"]
    if process is not None:
        process.terminate()
        process.wait()
        if previous is None:
            del os.environ["DISPLAY"]
        else:
            os.environ["DISPLAY"] = previous


@pytest.fixture(scope="module")
def app(display):
    from tiopt.gui import TiOptApp

    app = TiOptApp(capture_output=False, synchronous=True)
    app.update()
    app.tabs["trap"].generate_preset()
    app.update()
    yield app
    app.quit_app()


def text_of(widget):
    return widget.get("1.0", "end").strip()


def test_trap_tab(app):
    assert app.trap is not None and app.trap.solved
    assert len(app.tabs["trap"].tree.get_children()) == app.trap.n_electrodes
    assert app.tabs["trap"].plot.figure.axes


def test_electrostatics_tab(app):
    tab = app.tabs["electrostatics"]
    assert "RF null" in text_of(tab.results)
    for quantity in tab.QUANTITIES[:2]:
        for plane in tab.PLANES:
            tab.view.set("quantity", quantity)
            tab.view.set("plane", plane)
            tab.view.set("offset", "55" if plane.startswith("xy") else "0")
            tab.view.set("resolution", "21")
            tab.plot_map()
            app.update()
            assert len(tab.plot.figure.axes) == 3


def test_waveform_and_dynamics_tabs(app):
    waveforms = app.tabs["waveforms"]
    waveforms.static_well()
    app.update()
    assert app.waveform.n_steps == 1
    assert "Secular frequencies" in text_of(waveforms.results)

    # The static well can be mapped in the Electrostatics tab
    electrostatics = app.tabs["electrostatics"]
    electrostatics.view.set("quantity", electrostatics.QUANTITIES[2])
    electrostatics.view.set("plane", electrostatics.PLANES[0])
    electrostatics.view.set("resolution", "21")
    electrostatics.plot_map()
    app.update()

    waveforms.transport_form.set("n_steps", "201")
    waveforms.transport_form.set("duration", "20")
    waveforms.transport()
    app.update()
    assert app.waveform.n_steps == 201

    dynamics = app.tabs["dynamics"]
    dynamics.simulation.set("hold", "2")
    dynamics.simulate()
    app.update()
    assert "Final axial excitation" in text_of(dynamics.results)
    assert len(dynamics.plot.figure.axes) == 3


def test_photonics_tab(app):
    tab = app.tabs["photonics"]
    tab.load_trap_grating()
    app.update()
    results = text_of(tab.results)
    assert "emission angle: 36.87°" in results
    assert len(tab.plot.figure.axes) >= 4


@pytest.mark.parametrize("preset", ["Carrier Rabi flopping", "Blue sideband", "Ramsey", "Mølmer-Sørensen gate"])
def test_quantum_tab(app, preset):
    tab = app.tabs["quantum"]
    tab.mode.set("n_samples", "60")
    tab.preset_var.set(preset)
    tab.load_preset()
    tab.simulate()
    app.update()
    results = text_of(tab.results)
    assert "Final D-state populations" in results
    if preset == "Mølmer-Sørensen gate":
        fidelity = float(results.split("Bell state fidelity: ")[1].split()[0])
        assert fidelity > 0.999


def test_quantum_pulse_editor(app):
    tab = app.tabs["quantum"]
    tab.clear_pulses()
    for key, value in zip([c[0] for c in tab.COLUMNS], ["0", "5", "0", "100", "0", "0.1", "1"]):
        tab.pulse_vars[key].set(value)
    tab.add_pulse()
    assert len(tab.tree.get_children()) == 1
    tab.pulse_vars["targets"].set("")
    with pytest.raises(ValueError):
        tab.add_pulse()


def test_scheduling_tab(app, repository):
    tab = app.tabs["scheduling"]
    tab.circuit_form.set("use_quantum", False)
    tab.qasm_var.set(str(repository / "tests" / "quantumcircuits" / "test_0000.qasm"))
    tab.actions.set("n_steps", "11")
    tab.schedule()
    app.update()
    results = text_of(tab.results)
    assert "on 4 qubits" in results and "two-qubit gate" in results and "Transports" in results
    assert len(tab.plot.figure.axes) == 3
    tab.build_waveform()
    app.update()
    assert app.waveform.n_wells == 4
    assert app.notebook.select() == str(app.tabs["waveforms"])
    # Every priority mode, side by side
    tab.compare()
    app.update()
    results = text_of(tab.results)
    assert all(description[:30] in results for description in __import__("tiopt").scheduling.MODES.values())
    tab.circuit_form.set("use_quantum", True)


def test_errors_are_reported(app):
    # Invalid input raises (and would be shown in a dialog outside of tests)
    form = app.tabs["waveforms"].transport_form
    form.set("duration", "fast")
    with pytest.raises(ValueError):
        app.tabs["waveforms"].transport()
    form.set("duration", "20")


def test_trap_tab_photonics_preset(app):
    tab = app.tabs["trap"]
    tab.preset.set("photonics", True)
    tab.generate_preset()
    app.update()
    assert len(app.trap.grating_couplers()) == 4 and len(app.trap.detectors()) == 1
    tab.preset.set("photonics", False)
    tab.generate_preset()
    app.update()


def test_acceleration_toolbar(app):
    app.threads_var.set("2")
    app.device_var.set("cpu")
    app.apply_acceleration()
    assert "2 threads" in app.device_label.cget("text")
    app.threads_var.set(str(os.cpu_count()))
    app.apply_acceleration()


def test_codesign_tab(app, repository):
    tab = app.tabs["codesign"]
    tab.circuit.set("use_quantum", False)
    tab.qasm_var.set(str(repository / "examples" / "circuits" / "bell.qasm"))
    tab.basic_design()
    app.update()
    assert tab.codesign is not None and len(tab.records) == 1
    tab.flow.set("n_passes", "1")
    tab.flow.set("max_evaluations", "2")
    tab.run_flow()
    app.update()
    assert len(tab.records) == 6
    assert "Circuit fidelity" in text_of(tab.results)
    assert len(tab.plot.figure.axes) == 4
    tab.use_trap()
    app.update()
    # The loading zone gets photoionization and cooling beams, the gate zone the qubit and quench beams, and the
    # measurement zone the detection and repump beams
    assert [zone.types for zone in app.trap.zones()] == ["load+storage", "empty", "gate", "empty", "measure"]
    assert len(app.trap.grating_couplers()) == 8 and len(app.trap.detectors()) == 1


def test_codesign_tab_layout(app, repository):
    tab = app.tabs["codesign"]
    tab.circuit.set("use_quantum", False)
    tab.qasm_var.set(str(repository / "examples" / "circuits" / "bell.qasm"))
    tab.circuit.set("layout", "load+storage, gate+measure")
    tab.basic_design()
    app.update()
    assert tab.codesign.zone_layout == "load+storage, gate+measure"
    tab.circuit.set("layout", tab.circuit.get_str("layout").replace("gate+measure", "gate, measure"))


@pytest.mark.parametrize("preset", ["Linear from zone layout", "X junction", "Y junction", "T junction", "Point trap"])
def test_trap_tab_presets(app, preset):
    tab = app.tabs["trap"]
    tab.preset.set("preset", preset)
    tab.preset.set("n_segments", "2")
    tab.preset.set("segment_width", "40")
    tab.preset.set("layout", "load, gate x2, measure")
    tab.preset.set("photonics", True)
    tab.generate_preset()
    app.update()
    assert app.trap is not None and app.trap.solved
    assert len(tab.zone_tree.get_children()) == len(app.trap.zones()) > 0
    tab.design_report()
    app.update()
    assert "Zone 0" in text_of(tab.report)
    tab.preset.set("preset", "Linear")
    tab.preset.set("n_segments", "7")
    tab.preset.set("segment_width", "50")
    tab.preset.set("photonics", False)


def test_trap_tab_layout_files(app, tmp_path, monkeypatch):
    tab = app.tabs["trap"]
    tab.preset.set("preset", "Linear from zone layout")
    tab.preset.set("layout", "load, gate, measure")
    tab.preset.set("photonics", True)
    tab.generate_preset()
    app.update()
    gds = tmp_path / "trap.gds"
    svg = tmp_path / "trap.svg"
    monkeypatch.setattr("tkinter.filedialog.asksaveasfilename", lambda **kwargs: str(gds if kwargs.get("defaultextension") == ".gds" else svg))
    tab.export_gds()
    tab.export_svg()
    assert gds.stat().st_size > 0 and svg.read_text().startswith("<svg")
    n_electrodes = app.trap.n_electrodes
    monkeypatch.setattr("tkinter.filedialog.askopenfilename", lambda **kwargs: str(gds))
    tab.import_gds()
    app.update()
    assert app.trap.n_electrodes == n_electrodes
    assert [zone.types for zone in app.trap.zones()] == ["load", "gate", "measure"]
    tab.preset.set("preset", "Linear")
    tab.preset.set("photonics", False)
    tab.generate_preset()
    app.update()


def test_scheduling_tab_trap_zones(app):
    tab = app.tabs["scheduling"]
    trap_tab = app.tabs["trap"]
    trap_tab.preset.set("preset", "Linear from zone layout")
    trap_tab.preset.set("layout", "load+storage, gate, gate, measure")
    trap_tab.preset.set("zone_spacing", "80")
    trap_tab.generate_preset()
    app.update()
    tab.zones.set("trap_zones", True)
    tab.schedule()
    app.update()
    assert "gate zone" in text_of(tab.results)
    tab.zones.set("trap_zones", False)
    trap_tab.preset.set("preset", "Linear")
    trap_tab.preset.set("zone_spacing", "100")
    trap_tab.generate_preset()
    app.update()


def test_photonics_tab_components(app):
    tab = app.tabs["photonics"]
    tab.component.set("Straight waveguide")
    app.update()
    tab.run_component(False)
    app.update()
    assert "Propagation loss" in text_of(tab.component_results)
    assert len(tab.plot.figure.axes) == 4
    tab.component.set("Grating coupler")
    tab.run_component(False)
    app.update()
    assert "Intensity at the ion" in text_of(tab.component_results)
    tab.component.set(tab.COMPONENTS[0])
    app.update()


def test_builder_tab(app):
    from types import SimpleNamespace

    tab = app.tabs["builder"]
    app.notebook.select(tab)
    app.update()
    tab.clear()

    # Drag a section in from the palette, then drop another next to it, which snaps onto its end
    canvas = tab.canvas
    X, Y = tab.to_screen((-130.0, 2.0))
    event = SimpleNamespace(x_root=canvas.winfo_rootx() + X, y_root=canvas.winfo_rooty() + Y)
    tab.palette_press(event, "linear_section", {})
    tab.palette_motion(event)
    tab.palette_release(event)
    assert len(tab.parts) == 1 and (tab.parts[0].x, tab.parts[0].y) == (-130.0, 0.0)
    second = tab.drop("linear_section", 123.0, 3.0)
    assert (second.x, second.y) == pytest.approx((120.0, 0.0))
    assert tab.overlaps == []
    gate = tab.drop("zone", 0.0, 3.0, types="gate")
    assert gate.y == pytest.approx(0.0)
    tab.drop("grating", -40.0, 30.0, beam="qubit")
    tab.fit_view()
    app.update()

    # Move the zone with the mouse, and undo the move
    index = tab.parts.index(gate)
    X, Y = tab.to_screen((gate.x, gate.y))
    canvas.event_generate("<ButtonPress-1>", x=int(X), y=int(Y))
    canvas.event_generate("<B1-Motion>", x=int(X) + 40, y=int(Y))
    canvas.event_generate("<ButtonRelease-1>", x=int(X) + 40, y=int(Y))
    assert tab.selected == index and tab.parts[index].x > 10
    tab.undo()
    assert tab.parts[index].x == pytest.approx(0.0)

    # Overlapping parts are outlined, and refused when building
    tab.drop("dc_electrode", -100.0, 0.0)
    assert tab.overlaps
    tab.delete_selected()
    assert not tab.overlaps

    tab.build()
    app.update()
    assert app.trap.n_rf == 4 and len(app.trap.grating_couplers()) == 1
    assert "RF null" in text_of(tab.results)
    # Short rails pull the RF null down a little from ION_HEIGHT
    assert tab.nulls and tab.nulls[0][0] == pytest.approx(0.0, abs=1e-6) and 0.9 * tiopt_ion_height() < tab.nulls[0][2] < tiopt_ion_height()

    # Properties of the selected part can be edited
    tab.select(index)
    tab.properties.set("types", "gate+cool")
    tab.apply_properties()
    assert tab.parts[index].params["types"] == "gate+cool"

    # The current trap can be split back up into parts
    tab.import_trap()
    assert sum(part.kind == "polygon" for part in tab.parts) == app.trap.n_electrodes


def tiopt_ion_height():
    import tiopt

    return tiopt.ION_HEIGHT


def test_quantum_tab_pulse_shapes(app):
    tab = app.tabs["quantum"]
    tab.simulation.set("Pulse program")
    tab.mode.set("n_samples", "40")
    tab.presets.set("shape", "trapezoid")
    tab.presets.set("ramp", "20")
    tab.preset_var.set("Mølmer-Sørensen gate")
    tab.load_preset()
    rows = [tab.tree.item(item, "values") for item in tab.tree.get_children()]
    assert all(row[7] == "trapezoid" and float(row[8]) == pytest.approx(20.0) for row in rows)
    # Ramped pulses are longer than the 50 µs square gate, and still make a Bell state
    assert float(rows[0][1]) > 50.0
    tab.simulate()
    app.update()
    fidelity = float(text_of(tab.results).split("Bell state fidelity: ")[1].split()[0])
    assert fidelity > 0.999
    assert len(tab.plot.figure.axes) == 4
    # Carrier pulses keep their area
    tab.presets.set("shape", "gaussian")
    tab.presets.set("gaussian_width", "0.2")
    tab.preset_var.set("Carrier Rabi flopping")
    tab.load_preset()
    assert tab.pulses()[0].area == pytest.approx(2 * np.pi * 100e3 * 30e-6, rel=1e-3)
    tab.presets.set("shape", "square")


def test_quantum_tab_circuits(app):
    tab = app.tabs["quantum"]
    tab.simulation.set("Circuit")
    app.update()
    tab.circuit_form.set("preset", "Bernstein-Vazirani (101)")
    tab.load_circuit_preset()
    tab.errors.set("two_qubit", "0")
    tab.errors.set("single_qubit", "0")
    tab.errors.set("measurement", "0")
    tab.simulate_circuit()
    app.update()
    results = text_of(tab.circuit_results)
    assert "      101   1.0000   1.0000" in results
    tab.errors.set("two_qubit", "0.02")
    tab.simulate_circuit()
    app.update()
    fidelity = float(text_of(tab.circuit_results).split("final state with errors: ")[1].split()[0])
    assert 0.9 < fidelity < 0.99
    tab.show_native_gates()
    assert "2 Mølmer-Sørensen gates" in text_of(tab.circuit_results)
    # Edited circuits are checked
    tab.set_circuit_text("qubit[2] q;\nfoo q[0];")
    with pytest.raises(ValueError, match="unknown gate"):
        tab.simulate_circuit()
    tab.circuit_form.set("preset", "Bell pair")
    tab.load_circuit_preset()
    tab.simulation.set("Pulse program")
    app.update()


def test_trap_tab_confinement_and_shapes(app):
    tab = app.tabs["trap"]
    tab.preset.set("preset", "X junction")
    tab.generate_preset()
    tab.apply_rf()
    app.update()
    tab.show_confinement()
    app.update()
    report = text_of(tab.report)
    assert "CTC path at" in report and len(tab.plot.figure.axes) == 5
    tab.shape_form.set("symmetry", "4-fold")
    tab.shape_form.set("evaluations", "40")
    tab.optimize_shapes()
    app.update()
    report = text_of(tab.report)
    before, after = (float(v) for v in report.split("Barrier on the RF null: ")[1].split(" meV")[0].split(" -> "))
    assert after < before
    tab.preset.set("preset", "Linear")
    tab.generate_preset()
    app.update()


def test_photonics_tab_tapers_and_gratings(app):
    tab = app.tabs["photonics"]
    tab.component.set("Waveguide bend")
    tab.bend.set("bend_width", "0.8")
    tab.bend.set("taper_length", "10")
    tab.bend.set("euler_fraction", "0.5")
    tab.run_component(False)
    app.update()
    assert "tapers (scattering and propagation)" in text_of(tab.component_results)
    tab.bend.set("bend_width", "0.6")
    tab.bend.set("taper_length", "0")
    tab.component.set("Grating coupler")
    tab.coupler.set("kind", tab.GRATING_KINDS[3])
    tab.run_component(False)
    app.update()
    assert "fill factors from 0.20" in text_of(tab.component_results)
    tab.coupler.set("kind", tab.GRATING_KINDS[0])
    tab.component.set(tab.COMPONENTS[0])
    app.update()


def test_codesign_tab_full(app):
    trap_tab = app.tabs["trap"]
    trap_tab.preset.set("preset", "Linear from zone layout")
    trap_tab.preset.set("layout", "load+storage, empty, gate, empty, measure")
    trap_tab.preset.set("photonics", True)
    trap_tab.generate_preset()
    app.update()
    app.tabs["quantum"].circuit_form.set("preset", "Bell pair")
    app.tabs["quantum"].load_circuit_preset()
    tab = app.tabs["codesign"]
    tab.flow_var.set(tab.FLOWS[0])
    tab.gather()
    assert "Circuit: 2 qubits" in text_of(tab.sources)
    tab.run_form.set("evaluations", "25")
    tab.run_system()
    app.update()
    best = tab.best_evaluation()
    assert best is not None and best.objective <= tab.system.initial.objective
    assert "Circuit fidelity" in text_of(tab.system_results) and len(tab.plot.figure.axes) == 4
    tab.send_to_tabs()
    assert app.trap.V_RF == pytest.approx(best.parameters["V_RF"])
    app.tabs["quantum"].use_codesign_errors()
    assert float(app.tabs["quantum"].errors.get("two_qubit")) == pytest.approx(1 - best.metrics["two_qubit_fidelity"], rel=1e-2)
    trap_tab.preset.set("photonics", False)
    trap_tab.preset.set("preset", "Linear")
    trap_tab.generate_preset()
    app.update()


def test_builder_tab_curved_sections(app):
    tab = app.tabs["builder"]
    app.notebook.select(tab)
    app.update()
    tab.clear()
    tab.drop("curved_section", 0.0, 0.0, angle=180.0, n_segments=6)
    # Its right end is at (200, 0), pointing down, so a vertical section below joins there
    section = tab.drop("linear_section", 202.0, -127.0)
    section.rotation = 90.0
    section.x, section.y = tab.snap(section, 202.0, -127.0, exclude=tab.parts.index(section))
    assert (section.x, section.y) == pytest.approx((200.0, -125.0))
    tab.drop("zone", 0.0, 200.0, types="gate")
    tab.build()
    app.update()
    assert tab.nulls and tab.nulls[0][2] == pytest.approx(tiopt_ion_height(), abs=6.0)
    tab.clear()



def test_trap_tab_topology_rules_and_vtk(app, tmp_path, monkeypatch):
    tab = app.tabs["trap"]
    tab.generate_preset()
    tab.topo.set("half_width", "120")
    tab.topo.set("pixel", "12")
    tab.optimize_topology()
    app.update()
    assert len(tab.plot.figure.axes) >= 3
    tab.use_topology()
    app.update()
    assert app.trap.n_rf == app.trap.n_electrodes
    tab.generate_preset()
    tab.check_design_rules()
    app.update()
    assert "touch" in text_of(tab.report)
    n_electrodes = app.trap.n_electrodes
    tab.open_gaps()
    app.update()
    assert app.trap.n_electrodes == n_electrodes
    path = tmp_path / "trap.vtk"
    monkeypatch.setattr("tkinter.filedialog.asksaveasfilename", lambda **kwargs: str(path))
    tab.generate_preset()
    tab.export_vtk()
    app.update()
    assert path.read_text().startswith("# vtk DataFile")


def test_electrostatics_tab_site_analysis_and_bem(app):
    app.tabs["trap"].generate_preset()
    app.tabs["waveforms"].static_well()
    tab = app.tabs["electrostatics"]
    tab.site.set("x", "-50")
    tab.analyze_site()
    app.update()
    assert "Floquet frequencies" in text_of(tab.site_results)
    tab.show_stability()
    app.update()
    assert "stable" in text_of(tab.site_results)
    tab.show_escape()
    app.update()
    assert "Depth" in text_of(tab.site_results)
    tab.show_multipoles()
    app.update()
    assert "U2" in text_of(tab.site_results)
    tab.solver.set("solver", "BEM (finite gaps)")
    tab.bem_form.set("h_min", "8")
    tab.solve()
    app.update()
    assert app.trap.solver == "bem"
    tab.solver.set("solver", "analytical")


def test_waveforms_crystals_and_dynamics_tabs_extras(app):
    app.tabs["trap"].generate_preset()
    waveforms = app.tabs["waveforms"]
    waveforms.solve_objectives(True)
    app.update()
    assert "Converged" in text_of(waveforms.results)
    waveforms.transport_form.set("duration", "5")
    waveforms.transport_form.set("n_steps", "101")
    waveforms.objectives.set("max_slew", "5")
    waveforms.solve_objectives(False)
    app.update()
    waveforms.apply_hardware()
    app.update()
    assert "Largest error" in text_of(waveforms.results)
    waveforms.use_hardware_output()
    crystals_tab = app.tabs["crystals"]
    crystals_tab.find()
    app.update()
    assert app.crystal is not None and app.crystal.stable
    waveforms.static_well()
    dynamics_tab = app.tabs["dynamics"]
    dynamics_tab.crystal_form.set("duration", "5")
    dynamics_tab.crystal_form.set("ions", "40Ca+ x2")
    dynamics_tab.simulate_crystal()
    app.update()
    assert "Kinetic temperature" in text_of(dynamics_tab.results)


def test_quantum_tab_gates_zeeman_and_fits(app):
    tab = app.tabs["quantum"]
    tab.simulation.set(tab.SIMULATIONS[2])
    app.update()
    tab.gate_form.set("cutoff", "4")
    tab.design_gate(True)
    app.update()
    assert "Bell state fidelity" in text_of(tab.gate_results)
    tab.simulation.set(tab.SIMULATIONS[3])
    tab.show_zeeman()
    app.update()
    assert "D-5/2" in text_of(tab.zeeman_results)
    tab.simulation.set(tab.SIMULATIONS[4])
    tab.example_data()
    tab.fit_flopping()
    app.update()
    assert "Mean phonon number" in text_of(tab.fit_results)
    tab.simulation.set(tab.SIMULATIONS[0])


def test_scheduling_tab_qccd_and_photonic_circuits(app):
    scheduling = app.tabs["scheduling"]
    scheduling.compile_qccd()
    app.update()
    assert "fidelity" in text_of(scheduling.results)
    scheduling.explore_qccd()
    app.update()
    photonics = app.tabs["photonics"]
    for component in ("Directional coupler", "Splitter tree"):
        photonics.component.set(component)
        app.update()
        photonics.run_circuit_component()
        app.update()
        assert text_of(photonics.circuit_results)
    app.tabs["trap"].preset.set("photonics", True)
    app.tabs["trap"].generate_preset()
    photonics.component.set("Waveguide routes")
    photonics.run_circuit_component()
    app.update()
    assert "routes" in text_of(photonics.circuit_results)
    app.tabs["trap"].preset.set("photonics", False)
    photonics.component.set(photonics.COMPONENTS[0])
