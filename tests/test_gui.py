"""Smoke tests of the GUI, which drive every tab's main actions on a virtual display."""

import os
import shutil
import subprocess
import time

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


def test_scheduling_tab(app):
    tab = app.tabs["scheduling"]
    tab.actions.set("n_steps", "11")
    tab.schedule()
    app.update()
    assert "Layer 5" in text_of(tab.results)
    tab.build_waveform()
    app.update()
    assert app.waveform.n_wells == 4
    assert app.notebook.select() == str(app.tabs["waveforms"])


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
