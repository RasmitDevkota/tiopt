"""Graphical interface to tiopt, built with Tkinter and matplotlib.

Each tab covers one part of the library, i.e. trap design, electrostatics, transport waveforms, Coulomb crystals, ion
dynamics, integrated photonics, quantum dynamics, circuit scheduling, and co-design. Computations run on a background
thread, and the output of the C library shows up in the log at the bottom of the window.
"""

import os
import queue
import sys
import tempfile
import threading
import traceback
import ctypes
import ctypes.util
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
from matplotlib import colormaps, colors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.patches import Rectangle

from . import (analytic, bem, builder, circuits, components, confinement, crystals, drc, dynamics, gates, hardware, levels, lib, multimode,
               multipoles, photonic_circuits, pulses, qccd, scheduling, shapes, stability, system, topology)
from . import export as layout_export
from . import voltages as voltage_objectives
from .lib import AXIAL_FREQUENCY, ION_HEIGHT, LENGTH_SCALE, Circuit, Grating, Ion, Pulse, TransportProgram, Trap, TrapGraph, Waveform, Waveguide

REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLES = REPOSITORY / "examples"
MEV = 1e3 / lib.E_CHARGE  # J to meV
TWO_PI = 2 * np.pi

# ---------------------------------------------------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------------------------------------------------


class OutputCapture:
    """Redirect the standard output of the process (where the C library prints) into a queue, while still echoing it to
    the original standard output.
    """

    def __init__(self):
        self.queue = queue.Queue()
        self._original_fd = None
        self._read_fd = None
        self._thread = None

    def start(self):
        sys.stdout.flush()
        lib.flush_c_output()
        self._original_fd = os.dup(1)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 1)
        os.close(write_fd)
        self._read_fd = read_fd

        # Line buffer the C library's output so that it appears as it is printed
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c"))
            stdout = ctypes.c_void_p.in_dll(libc, "stdout")
            libc.setvbuf(stdout, None, 1, 0)
        except (OSError, ValueError, AttributeError):
            pass

        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        while True:
            try:
                data = os.read(self._read_fd, 65536)
            except OSError:
                break
            if not data:
                break
            try:
                os.write(self._original_fd, data)
            except OSError:
                pass
            self.queue.put(data.decode(errors="replace"))

    def stop(self):
        if self._original_fd is None:
            return
        sys.stdout.flush()
        lib.flush_c_output()
        os.dup2(self._original_fd, 1)
        os.close(self._original_fd)
        self._original_fd = None


class Form(ttk.LabelFrame):
    """Group of labeled inputs laid out in a grid."""

    def __init__(self, parent, title):
        super().__init__(parent, text=title, padding=6)
        self.vars = {}
        self.row = 0
        self.columnconfigure(1, weight=1)

    def entry(self, key, label, default, unit=""):
        var = tk.StringVar(value=str(default))
        ttk.Label(self, text=label).grid(row=self.row, column=0, sticky="w", padx=(0, 6), pady=1)
        ttk.Entry(self, textvariable=var, width=10).grid(row=self.row, column=1, sticky="ew", pady=1)
        ttk.Label(self, text=unit).grid(row=self.row, column=2, sticky="w", padx=(4, 0))
        self.vars[key] = (var, label)
        self.row += 1
        return var

    def combo(self, key, label, values, default):
        var = tk.StringVar(value=default)
        ttk.Label(self, text=label).grid(row=self.row, column=0, sticky="w", padx=(0, 6), pady=1)
        ttk.Combobox(self, textvariable=var, values=values, state="readonly", width=18).grid(row=self.row, column=1, columnspan=2, sticky="ew", pady=1)
        self.vars[key] = (var, label)
        self.row += 1
        return var

    def check(self, key, label, default=False):
        var = tk.BooleanVar(value=default)
        ttk.Checkbutton(self, text=label, variable=var).grid(row=self.row, column=0, columnspan=3, sticky="w", pady=1)
        self.vars[key] = (var, label)
        self.row += 1
        return var

    def buttons(self, *buttons):
        frame = ttk.Frame(self)
        frame.grid(row=self.row, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        for text, command in buttons:
            ttk.Button(frame, text=text, command=command).pack(side="left", padx=(0, 4))
        self.row += 1
        return frame

    def text(self, height=4):
        widget = tk.Text(self, height=height, width=46, wrap="word", relief="flat", background=self.winfo_toplevel().cget("background"))
        widget.grid(row=self.row, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        widget.configure(state="disabled")
        self.row += 1
        return widget

    def get(self, key):
        var, label = self.vars[key]
        value = var.get()
        if isinstance(value, bool):
            return value
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"{label} must be a number, not {value!r}") from None

    def get_int(self, key):
        value = self.get(key)
        if value != int(value):
            raise ValueError(f"{self.vars[key][1]} must be an integer")
        return int(value)

    def get_str(self, key):
        return self.vars[key][0].get()

    def set(self, key, value):
        self.vars[key][0].set(value)


def set_text(widget, text):
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", text)
    widget.configure(state="disabled")


def parse_ions(text):
    """Species of every ion from text such as "40Ca+ x3, 9Be+"."""
    ions = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        element, _, count = item.partition(" x")
        count = int(count) if count.strip() else 1
        if element.strip() not in Ion.PRESETS:
            raise ValueError(f"Unknown species {element.strip()!r}; choose from {', '.join(Ion.PRESETS)}")
        ions += [element.strip()] * count
    if not ions:
        raise ValueError("Give at least one ion, e.g. 40Ca+ x3, 9Be+")
    return ions


def parse_optional(text):
    """Number in text, or None if it is blank or "none"."""
    text = str(text).strip()
    if text.lower() in ("", "none", "-"):
        return None
    return float(text)


def draw_electrodes(ax, trap, alpha=0.25, labels=False):
    """Draw the electrodes of a trap on xy axes (RF in orange and DC in blue)."""
    for e, polygon in enumerate(trap.electrode_polygons()):
        color = "tab:orange" if trap.is_rf(e) else "tab:blue"
        ax.add_patch(PolygonPatch(polygon[:, :2], closed=True, facecolor=colors.to_rgba(color, alpha), edgecolor=color, linewidth=0.6))
        if labels:
            center = polygon.mean(axis=0)
            ax.text(center[0], center[1], str(e), ha="center", va="center", fontsize=7)
    ax.autoscale_view()
    ax.set_aspect("equal")


def beam_wavevector(app, choice, position=None, wavelength=None):
    """Wavevector (1/m) of a beam: of the qubit grating of the trap at a position, or along a fixed direction."""
    ion = app.ion
    wavelength = ion.qubit_wavelength if wavelength is None else wavelength
    k = TWO_PI / wavelength
    if choice.startswith("Qubit grating"):
        trap = app.trap
        if trap is None:
            raise ValueError("Generate a trap with a qubit grating first")
        try:
            grating = trap.grating("qubit")
        except (KeyError, ValueError, StopIteration, lib.TiOptError):
            grating = trap.grating_couplers()[0] if trap.grating_couplers() else None
        if grating is None:
            raise ValueError("The trap has no grating couplers")
        target = grating.position + np.array([0.0, 0.0, ION_HEIGHT]) if position is None else np.asarray(position, dtype=float)
        _, k_vector = grating.beam_at(target)
        return np.asarray(k_vector, dtype=float)
    if choice.startswith("Raman"):
        return np.array([np.sqrt(2) * k, 0.0, 0.0])
    if "(1, 0, 1)" in choice:
        return k * np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
    return np.array([k, 0.0, 0.0])


class PlotPanel(ttk.Frame):
    """Matplotlib figure with its navigation toolbar."""

    def __init__(self, parent):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def axes(self, nrows=1, ncols=1, **kwargs):
        self.figure.clear()
        axes = self.figure.subplots(nrows, ncols, squeeze=False, **kwargs)
        return axes

    def draw(self):
        self.canvas.draw_idle()

    def message(self, text):
        self.figure.clear()
        ax = self.figure.add_subplot()
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_axis_off()
        self.draw()


class Tab(ttk.Frame):
    """Tab with a scrollable column of controls on the left and a plot on the right."""

    def __init__(self, app, title, plot=True):
        super().__init__(app.notebook, padding=4)
        self.app = app
        app.notebook.add(self, text=title)

        panes = ttk.PanedWindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True)

        outer = ttk.Frame(panes)
        canvas = tk.Canvas(outer, width=410, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        self.controls = ttk.Frame(canvas)
        self.controls.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        window = canvas.create_window((0, 0), window=self.controls, anchor="nw")
        # Fit the controls to the width of the column so that nothing is clipped on the right
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        panes.add(outer, weight=0)

        # Tabs without a plot get an empty frame to fill instead
        if plot:
            self.plot = PlotPanel(panes)
            panes.add(self.plot, weight=1)
        else:
            self.plot = None
            self.main = ttk.Frame(panes)
            panes.add(self.main, weight=1)

    def form(self, title):
        form = Form(self.controls, title)
        form.pack(fill="x", pady=(0, 6), padx=(0, 4))
        return form

    def require_trap(self):
        if self.app.trap is None or not self.app.trap.solved:
            raise ValueError("Generate or load a trap in the Trap tab first")
        return self.app.trap

    def refresh(self):
        """Update the tab when the trap or ion changes."""


# ---------------------------------------------------------------------------------------------------------------------
# Trap
# ---------------------------------------------------------------------------------------------------------------------


class TrapTab(Tab):
    # Presets of the preset library, with the name Trap.from_preset knows them by (None for a trap from a zone layout)
    PRESETS = {
        "Linear": "linear",
        "Linear from zone layout": None,
        "X junction": "x_junction",
        "Y junction": "y_junction",
        "T junction": "t_junction",
        "Point trap": "point",
    }

    def __init__(self, app):
        super().__init__(app, "Trap")

        preset = self.form("Trap preset")
        preset.combo("preset", "Preset", tuple(self.PRESETS), "Linear")
        preset.entry("n_segments", "Segments per side", 7)
        preset.entry("segment_width", "Segment width", 50, "µm")
        preset.entry("layout", "Zone layout", lib.DEFAULT_LAYOUT)
        preset.entry("zone_spacing", "Zone spacing", 100, "µm")
        preset.check("gratings", "Grating coupler under the trap center (linear)", True)
        preset.check("photonics", "Integrated photonics for every zone", False)
        preset.buttons(("Generate", self.generate_preset))
        self.preset = preset

        files = self.form("Trap and layout files")
        files.buttons(("Load...", self.load_file), ("Save...", self.save_file))
        files.buttons(("Import GDS...", self.import_gds), ("Export GDS...", self.export_gds), ("Export SVG...", self.export_svg))
        files.buttons(("Export potentials to VTK...", self.export_vtk))

        rf = self.form("RF drive")
        rf.entry("V_RF", "Amplitude", 40, "V")
        rf.entry("f_RF", "Frequency", 40, "MHz")
        rf.buttons(("Apply", self.apply_rf))
        self.rf = rf

        ion = self.form("Ion species")
        self.ion_var = ion.combo("species", "Species", Ion.PRESETS, "40Ca+")
        self.ion_var.trace_add("write", lambda *args: self.app.set_ion(self.ion_var.get()))

        ctc = self.form("Confinement and CTC paths")
        ctc.entry("waypoints", "Path (x, y; x, y; ...) or auto", "auto", "µm")
        ctc.entry("map_height", "Height of the map", ION_HEIGHT, "µm")
        ctc.buttons(("Show confinement", self.show_confinement))
        self.ctc = ctc

        shape = self.form("Electrode shape optimization")
        shape.entry("center", "Center (x, y) or auto", "auto", "µm")
        shape.entry("radius", "Radius of the region", 2.5 * ION_HEIGHT, "µm")
        shape.entry("max_shift", "Largest displacement", 8, "µm")
        shape.entry("spacing", "Spacing of the control points", 45, "µm")
        shape.combo("symmetry", "Symmetry", shapes.SYMMETRIES, "none")
        shape.combo("objective", "Objective", tuple(shapes.OBJECTIVES.values()), shapes.OBJECTIVES["both"])
        shape.entry("evaluations", "Evaluations", 200)
        shape.buttons(("Optimize shapes", self.optimize_shapes), ("Stop", self.stop_shapes))
        self.shape_form = shape
        self.shape_stop = threading.Event()

        topo = self.form("Pixel topology optimization")
        topo.combo("layout", "RF nulls along", topology.LAYOUTS, "X junction")
        topo.entry("half_width", "Half-width of the region", 200, "µm")
        topo.entry("pixel", "Pixel size", 10, "µm")
        topo.entry("height", "Height of the RF nulls", ION_HEIGHT, "µm")
        topo.entry("spacing", "Spacing of the RF nulls", 30, "µm")
        topo.combo("symmetry", "Symmetry", ("of the layout",) + topology.SYMMETRIES, "of the layout")
        topo.buttons(("Optimize topology", self.optimize_topology), ("Use as trap", self.use_topology))
        self.topo = topo
        self.topology_result = None

        rules = self.form("Design rules and gaps")
        rules.entry("min_width", "Narrowest electrode", 5, "µm")
        rules.entry("min_gap", "Narrowest gap", 3, "µm")
        rules.entry("min_angle", "Sharpest corner", 30, "°")
        rules.entry("min_feature", "Smallest grating tooth or groove", 150, "nm")
        rules.entry("gap", "Gap to open between electrodes", 5, "µm")
        rules.buttons(("Check design rules", self.check_design_rules), ("Open gaps", self.open_gaps))
        self.rules = rules

        report = self.form("Design report")
        report.buttons(("Evaluate zones", self.design_report))
        self.report = ScrolledText(report, height=10, width=46, wrap="none")
        self.report.grid(row=report.row, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        self.report.configure(state="disabled")
        report.row += 1

        zones = ttk.LabelFrame(self.controls, text="Zones", padding=6)
        zones.pack(fill="both", expand=True, padx=(0, 4), pady=(0, 6))
        self.zone_tree = ttk.Treeview(zones, columns=("types", "capacity", "x", "y"), height=5)
        self.zone_tree.heading("#0", text="#")
        self.zone_tree.column("#0", width=40)
        for column, text, width in (("types", "Types", 120), ("capacity", "Ions", 50), ("x", "x (µm)", 70), ("y", "y (µm)", 70)):
            self.zone_tree.heading(column, text=text)
            self.zone_tree.column(column, width=width, anchor="e")
        self.zone_tree.pack(fill="both", expand=True)

        table = ttk.LabelFrame(self.controls, text="Electrodes", padding=6)
        table.pack(fill="both", expand=True, padx=(0, 4))
        self.tree = ttk.Treeview(table, columns=("type", "x", "y", "vertices"), height=12)
        self.tree.heading("#0", text="#")
        self.tree.column("#0", width=40)
        for column, text, width in (("type", "Type", 50), ("x", "x (µm)", 80), ("y", "y (µm)", 80), ("vertices", "Vertices", 70)):
            self.tree.heading(column, text=text)
            self.tree.column(column, width=width, anchor="e")
        self.tree.pack(fill="both", expand=True)

        self.plot.message("Generate a preset or load a trap file")

    def generate_preset(self):
        name = self.preset.get_str("preset")
        if name not in self.PRESETS:
            raise ValueError(f"Choose a preset from {', '.join(self.PRESETS)}")
        preset = self.PRESETS[name]
        n_segments = self.preset.get_int("n_segments")
        segment_width = self.preset.get("segment_width")
        layout = self.preset.get_str("layout")
        zone_spacing = self.preset.get("zone_spacing")
        gratings = self.preset.get("gratings") and preset == "linear"
        photonics = self.preset.get("photonics")
        element = self.app.ion.element

        def work():
            if preset is None:
                trap = Trap.from_layout(layout, zone_spacing, segment_width, element, photonics)
            else:
                trap = Trap.from_preset(n_segments, segment_width, gratings and not photonics, False, preset)
                if photonics:
                    if preset == "linear":
                        # A gate zone at the center and a measurement zone two segments along
                        trap.set_zones([("gate", 0, (0.0, 0.0, None)), ("measure", 0, (2 * segment_width, 0.0, None))])
                    trap.add_zone_photonics(element)
            trap.solve("analytical")
            return trap

        self.app.run_task(f"Generating and solving the {name.lower()} trap", work, self.app.set_trap)

    def import_gds(self):
        path = filedialog.askopenfilename(parent=self, title="Import GDSII layout (RF electrodes on layer 1, DC electrodes on layer 2)", initialdir=REPOSITORY, filetypes=[("GDSII", "*.gds"), ("All files", "*")])
        if not path:
            return

        def work():
            trap = Trap.from_gds(path)
            trap.solve("analytical")
            return trap

        self.app.run_task(f"Importing {Path(path).name}", work, self.app.set_trap)

    def export_gds(self):
        if self.app.trap is None:
            raise ValueError("There is no trap to export")
        path = filedialog.asksaveasfilename(parent=self, title="Export GDSII layout", initialdir=REPOSITORY, defaultextension=".gds", filetypes=[("GDSII", "*.gds")])
        if path:
            self.app.trap.save_gds(path)
            self.app.set_status(f"Exported trap layout to {path}")

    def export_svg(self):
        if self.app.trap is None:
            raise ValueError("There is no trap to export")
        path = filedialog.asksaveasfilename(parent=self, title="Export SVG drawing", initialdir=REPOSITORY, defaultextension=".svg", filetypes=[("SVG", "*.svg")])
        if path:
            self.app.trap.save_svg(path)
            self.app.set_status(f"Exported trap drawing to {path}")

    def design_report(self):
        trap = self.require_trap()
        ion = self.app.ion

        def work():
            return lib.format_design_report(trap.design_report(ion))

        self.app.run_task("Evaluating the zones of the design", work, lambda text: set_text(self.report, text))

    def waypoints(self, trap):
        text = self.ctc.get_str("waypoints").strip()
        if text.lower() in ("", "auto"):
            return confinement.default_waypoints(trap)
        try:
            points = np.array([[float(v) for v in point.split(",")] for point in text.split(";") if point.strip()])
        except ValueError:
            raise ValueError("Write the path as x, y pairs separated by semicolons, e.g. -150, 0; 0, 0; 0, 150") from None
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("The path needs at least two x, y points")
        return points

    def _rf_trap(self):
        trap = self.require_trap()
        if trap.n_rf == 0:
            raise ValueError("The trap has no RF electrodes")
        if not trap.V_RF > 0 or not trap.Omega_RF > 0:
            raise ValueError("Set the RF drive of the trap first")
        return trap

    def show_confinement(self):
        trap = self._rf_trap()
        ion = self.app.ion
        waypoints = self.waypoints(trap)
        height = self.ctc.get("map_height")

        def work():
            path = confinement.confinement_path(trap, ion, waypoints, 121)
            vertices = np.vstack([polygon[:, :2] for polygon in trap.electrode_polygons()])
            low, high = vertices.min(axis=0), vertices.max(axis=0)
            x = np.linspace(low[0], high[0], 110)
            y = np.linspace(low[1], high[1], 110)
            return path, x, y, confinement.confinement_map(trap, ion, x, y, height)

        def done(result):
            path, x, y, values = result
            set_text(self.report, "\n".join([
                f"Path through {len(waypoints)} waypoints, {path.s[-1]:.0f} µm long",
                f"RF null from {path.null_positions[:, 2].min():.1f} to {path.null_positions[:, 2].max():.1f} µm up",
                f"Pseudopotential barrier on the null: {path.barrier * 1e3:.3f} meV",
                f"Total confinement on the null: {path.null_confinement.min() / 1e6:.2f} to {path.null_confinement.max() / 1e6:.2f} MHz",
                f"CTC path at {path.target / 1e6:.2f} MHz, up to {path.depth:.1f} µm below the null",
                f"Pseudopotential on the CTC path: up to {path.ctc_energy.max() * 1e3:.2f} meV",
                f"DC field which holds the ion on it: up to {path.holding_field.max():.0f} V/m",
                "(total confinement is sqrt(sum of the squared secular frequencies), which only the RF electrodes set)",
            ]))
            (ax_map, ax_height), (ax_confinement, ax_energy) = self.plot.axes(2, 2)
            mesh = ax_map.pcolormesh(x, y, values / 1e6, shading="auto", cmap="viridis", norm=colors.LogNorm(vmin=max(values.min() / 1e6, 1e-2), vmax=values.max() / 1e6))
            self.plot.figure.colorbar(mesh, ax=ax_map, label="Total confinement (MHz)")
            for e, polygon in enumerate(trap.electrode_polygons()):
                ax_map.add_patch(PolygonPatch(polygon[:, :2], closed=True, fill=False, edgecolor="tab:orange" if trap.is_rf(e) else "white", linewidth=0.5))
            ax_map.plot(path.null_positions[:, 0], path.null_positions[:, 1], color="tab:red", linewidth=1.5, label="Path (CTC beneath)")
            ax_map.plot(waypoints[:, 0], waypoints[:, 1], linestyle="none", marker="o", color="tab:red", markersize=3)
            ax_map.set_aspect("equal")
            ax_map.set_xlabel("x (µm)")
            ax_map.set_ylabel("y (µm)")
            ax_map.set_title(f"Total confinement at z = {self.ctc.get('map_height'):g} µm")
            ax_map.legend(fontsize=7, loc="lower left")

            ax_height.plot(path.s, path.null_positions[:, 2], label="RF null")
            ax_height.plot(path.s, path.ctc_positions[:, 2], label="CTC path")
            ax_height.set_xlabel("Distance along the path (µm)")
            ax_height.set_ylabel("Height (µm)")
            ax_height.legend(fontsize=7)
            ax_height.set_title("Height of the paths")

            ax_confinement.plot(path.s, path.null_confinement / 1e6, label="On the RF null")
            ax_confinement.plot(path.s, path.ctc_confinement / 1e6, label="On the CTC path")
            ax_confinement.axhline(path.target / 1e6, color="gray", linestyle=":")
            ax_confinement.set_xlabel("Distance along the path (µm)")
            ax_confinement.set_ylabel("Total confinement (MHz)")
            ax_confinement.legend(fontsize=7)
            ax_confinement.set_title("Total confinement")

            ax_energy.plot(path.s, path.null_energy * 1e3, label="On the RF null")
            ax_energy.plot(path.s, path.ctc_energy * 1e3, label="On the CTC path")
            ax_energy.set_xlabel("Distance along the path (µm)")
            ax_energy.set_ylabel("Pseudopotential (meV)")
            ax_energy.legend(fontsize=7)
            ax_energy.set_title("Pseudopotential")
            self.plot.draw()

        self.app.run_task("Following the RF null and the CTC path", work, done)

    def optimize_shapes(self):
        trap = self._rf_trap()
        ion = self.app.ion
        waypoints = self.waypoints(trap)
        f = self.shape_form
        center_text = f.get_str("center").strip()
        center = None
        if center_text.lower() not in ("", "auto"):
            try:
                center = [float(v) for v in center_text.split(",")]
            except ValueError:
                raise ValueError("Write the center as x, y") from None
        objectives = {description: key for key, description in shapes.OBJECTIVES.items()}
        objective = objectives.get(f.get_str("objective"), "both")
        settings = dict(center=center, radius=f.get("radius"), max_shift=f.get("max_shift"), spacing=f.get("spacing"), symmetry=f.get_str("symmetry"),
                        objective=objective, max_evaluations=f.get_int("evaluations"))
        self.shape_stop.clear()

        def progress(evaluation, value, metrics):
            if evaluation % 20 == 0:
                print(f"Shape evaluation {evaluation}: objective {value:.4f}, barrier {metrics.barrier:.3f} meV, confinement spread {metrics.confinement_spread:.4f}", flush=True)

        def work():
            return shapes.optimize_shapes(trap, ion, waypoints, callback=progress, stop=self.shape_stop.is_set, **settings)

        def done(result):
            before = [polygon[:, :2].copy() for polygon in trap.electrode_polygons()]
            rf = [trap.is_rf(e) for e in range(trap.n_electrodes)]
            self.app.set_trap(result.trap)
            set_text(self.report, "\n".join([
                f"Moved the electrodes by up to {result.max_displacement:.2f} µm ({len(result.history)} evaluations)",
                f"Barrier on the RF null: {result.initial_path.barrier * 1e3:.3f} -> {result.final_path.barrier * 1e3:.3f} meV",
                f"Lowest total confinement / CTC target: {result.initial_path.confinement_dip:.3f} -> {result.final_path.confinement_dip:.3f}",
                f"CTC path below the null: up to {result.initial_path.depth:.1f} -> {result.final_path.depth:.1f} µm",
                "The trap with the new shapes is now the current trap of every tab.",
            ]))
            (ax_shapes, ax_energy), (ax_history, ax_confinement) = self.plot.axes(2, 2)
            for polygon, is_rf in zip(before, rf):
                ax_shapes.add_patch(PolygonPatch(polygon, closed=True, fill=False, edgecolor="gray", linestyle="--", linewidth=0.6))
            for polygon, is_rf in zip(result.polygons, rf):
                color = "tab:orange" if is_rf else "tab:blue"
                ax_shapes.add_patch(PolygonPatch(polygon, closed=True, facecolor=colors.to_rgba(color, 0.3), edgecolor=color, linewidth=0.8))
            c = result.deformation.center
            r = result.deformation.radius
            ax_shapes.add_patch(PolygonPatch(np.column_stack([c[0] + r * np.cos(np.linspace(0, TWO_PI, 100)), c[1] + r * np.sin(np.linspace(0, TWO_PI, 100))]),
                                             closed=True, fill=False, edgecolor="tab:red", linestyle=":"))
            ax_shapes.set_xlim(c[0] - 1.2 * r, c[0] + 1.2 * r)
            ax_shapes.set_ylim(c[1] - 1.2 * r, c[1] + 1.2 * r)
            ax_shapes.set_aspect("equal")
            ax_shapes.set_title("Electrodes before (dashed) and after")
            ax_energy.plot(result.initial_path.s, result.initial_path.null_energy * 1e3, label="Before")
            ax_energy.plot(result.final_path.s, result.final_path.null_energy * 1e3, label="After")
            ax_energy.set_ylabel("Pseudopotential on the null (meV)")
            ax_energy.legend(fontsize=7)
            ax_history.plot(result.history, linewidth=0.8)
            ax_history.plot(np.minimum.accumulate(result.history), color="black")
            ax_history.set_xlabel("Evaluation")
            ax_history.set_ylabel("Objective")
            ax_confinement.plot(result.initial_path.s, result.initial_path.null_confinement / 1e6, label="Before")
            ax_confinement.plot(result.final_path.s, result.final_path.null_confinement / 1e6, label="After")
            ax_confinement.axhline(result.initial_path.target / 1e6, color="gray", linestyle=":")
            ax_confinement.set_xlabel("Distance along the path (µm)")
            ax_confinement.set_ylabel("Total confinement (MHz)")
            ax_confinement.legend(fontsize=7)
            self.plot.draw()

        self.app.run_task("Optimizing the shapes of the electrodes", work, done)

    def stop_shapes(self):
        self.shape_stop.set()

    def load_file(self):
        path = filedialog.askopenfilename(parent=self, title="Load trap file", initialdir=REPOSITORY, filetypes=[("Trap files", "*.data *.trap"), ("All files", "*")])
        if not path:
            return

        def work():
            trap = Trap.from_file(path)
            trap.solve("analytical")
            return trap

        self.app.run_task(f"Loading {Path(path).name}", work, self.app.set_trap)

    def save_file(self):
        if self.app.trap is None:
            raise ValueError("There is no trap to save")
        path = filedialog.asksaveasfilename(parent=self, title="Save trap file", initialdir=REPOSITORY, defaultextension=".data")
        if path:
            self.app.trap.save(path)
            self.app.set_status(f"Saved trap to {path}")

    def apply_rf(self):
        trap = self.require_trap()
        trap.V_RF = self.rf.get("V_RF")
        trap.Omega_RF = TWO_PI * self.rf.get("f_RF") * 1e6
        self.app.trap_changed()
        self.app.set_status(f"RF drive set to {trap.V_RF:g} V at {trap.Omega_RF / TWO_PI / 1e6:g} MHz")

    def export_vtk(self):
        trap = self.require_trap()
        path = filedialog.asksaveasfilename(parent=self, title="Export potentials to VTK", initialdir=REPOSITORY, defaultextension=".vtk", filetypes=[("VTK", "*.vtk")])
        if not path:
            return
        voltages = self.app.waveform.voltages[0] if self.app.waveform is not None else None
        ion = self.app.ion

        def work():
            layout_export.write_vtk(trap, path, ion=ion, voltages=voltages, spacing=4.0)
            return path

        self.app.run_task("Writing potentials to VTK", work, lambda path: self.app.set_status(f"Wrote potentials to {path}"))

    def optimize_topology(self):
        form = self.topo
        half_width = form.get("half_width")
        pixel = form.get("pixel")
        if pixel <= 0 or half_width < 4 * pixel:
            raise ValueError("The region must be at least 8 pixels across")
        if (2 * half_width / pixel) ** 2 > 40000:
            raise ValueError("Use fewer than 200 pixels across the region")
        layout = form.get_str("layout")
        symmetry = form.get_str("symmetry")
        kwargs = dict(half_width=half_width, pixel=pixel, layout=layout, height=form.get("height"), spacing=form.get("spacing"),
                      symmetry=None if symmetry.startswith("of the") else symmetry)
        V_RF, Omega_RF = self.rf.get("V_RF"), TWO_PI * self.rf.get("f_RF") * 1e6
        ion = self.app.ion

        def work():
            result = topology.optimize_topology(**kwargs)
            trap = result.to_trap(V_RF, Omega_RF)
            reach = 0.7 * half_width
            height = kwargs["height"]
            paths = {
                "X junction": [(-reach, 0.0), (0.0, 0.0), (0.0, reach)],
                "T junction": [(-reach, 0.0), (0.0, 0.0), (0.0, reach)],
                "Y junction": [(0.0, reach), (0.0, 0.0), (reach * np.cos(np.radians(330)), reach * np.sin(np.radians(330)))],
                "Linear": [(-reach, 0.0), (reach, 0.0)],
            }
            try:
                waypoints = [(x, y, height) for x, y in paths[layout]]
                positions, energies, barrier = trap.rf_null_path(ion, waypoints, 81)
                path = (positions, energies, barrier)
            except Exception:
                path = None
            return result, trap, path

        def done(output):
            result, trap, path = output
            self.topology_result = (result, trap)
            n = result.shape[0]
            edges = (-half_width, half_width, -half_width, half_width)
            ax_density, ax_pattern, ax_path = self.plot.axes(1, 3, width_ratios=[1, 1, 1.1])[0]
            image = ax_density.imshow(result.density.reshape(n, n).T, origin="lower", extent=edges, cmap="Oranges", vmin=0, vmax=1)
            self.plot.figure.colorbar(image, ax=ax_density, label="RF fraction of each pixel", shrink=0.7)
            ax_density.plot(result.nulls[:, 0], result.nulls[:, 1], "k.", markersize=4)
            ax_density.set_title("Linear program")
            for polygon in result.polygons:
                ax_pattern.add_patch(PolygonPatch(polygon, closed=True, facecolor=colors.to_rgba("tab:orange", 0.4), edgecolor="tab:orange", linewidth=0.8))
            ax_pattern.plot(result.nulls[:, 0], result.nulls[:, 1], "k.", markersize=4, label="Asked-for RF nulls")
            ax_pattern.set_xlim(-half_width, half_width)
            ax_pattern.set_ylim(-half_width, half_width)
            ax_pattern.set_aspect("equal")
            ax_pattern.set_title(f"{len(result.polygons)} RF electrodes")
            ax_pattern.legend(fontsize=7, loc="lower right")
            for ax in (ax_density, ax_pattern):
                ax.set_xlabel("x (µm)")
            ax_density.set_ylabel("y (µm)")
            if path is not None:
                positions, energies, barrier = path
                distance = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1))])
                ax_path.plot(distance, energies * 1e3, color="tab:red", label="Pseudopotential on the RF null")
                ax_twin = ax_path.twinx()
                ax_twin.plot(distance, positions[:, 2], color="gray", linestyle="--")
                ax_twin.set_ylabel("Height of the RF null (µm, dashed)")
                ax_path.set_xlabel("Distance along the arms (µm)")
                ax_path.set_ylabel("Pseudopotential (meV)")
                ax_path.set_title(f"Barrier {barrier * 1e3:.2f} meV")
                ax_path.legend(fontsize=7, loc="upper left")
            else:
                ax_path.set_axis_off()
            self.plot.draw()
            self.app.set_status(f"Weakest confinement {result.confinement.min():.3g} V/µm² per volt; largest residual field {result.residual:.2g} V/µm per volt at the nulls")

        self.app.run_task("Optimizing the topology of the RF electrodes", work, done)

    def use_topology(self):
        if self.topology_result is None:
            raise ValueError("Optimize a topology first")
        self.app.set_trap(self.topology_result[1])
        self.topology_result = None

    def design_rules(self):
        form = self.rules
        return drc.Rules(min_width=form.get("min_width"), min_notch=form.get("min_gap"), min_gap=form.get("min_gap"), min_angle=form.get("min_angle"),
                         min_feature=form.get("min_feature") * 1e-3)

    def check_design_rules(self):
        if self.app.trap is None:
            raise ValueError("Generate or load a trap first")
        trap = self.app.trap
        rules = self.design_rules()

        def work():
            return drc.check_trap(trap, rules)

        def done(violations):
            set_text(self.report, "\n".join(drc.summary(violations)))
            self.refresh()
            ax = self.plot.figure.axes[0]
            colors_by_severity = {"error": "tab:red", "warning": "gold", "info": "tab:gray"}
            for severity, color in colors_by_severity.items():
                points = [v.location for v in violations if v.severity == severity and v.location is not None]
                if points:
                    points = np.array(points)
                    ax.plot(points[:, 0], points[:, 1], marker="o", markersize=12, markerfacecolor="none", markeredgecolor=color, linestyle="none", label=f"{severity}s")
            ax.legend(loc="upper right", fontsize=8)
            self.plot.draw()

        self.app.run_task("Checking the design rules", work, done)

    def open_gaps(self):
        trap = self.require_trap()
        gap = self.rules.get("gap")
        if gap <= 0:
            raise ValueError("The gap must be positive")

        def work():
            return drc.gapped_trap(trap, gap, solve=True)

        self.app.run_task(f"Opening {gap:g} µm gaps between the electrodes", work, self.app.set_trap)

    def refresh(self):
        trap = self.app.trap
        self.tree.delete(*self.tree.get_children())
        self.zone_tree.delete(*self.zone_tree.get_children())
        if trap is None:
            return

        zones = trap.zones()
        for z, zone in enumerate(zones):
            self.zone_tree.insert("", "end", text=str(z), values=(zone.types, zone.capacity, f"{zone.position[0]:.1f}", f"{zone.position[1]:.1f}"))

        self.rf.set("V_RF", f"{trap.V_RF:g}")
        self.rf.set("f_RF", f"{trap.Omega_RF / TWO_PI / 1e6:g}")

        polygons = trap.electrode_polygons()
        for e, polygon in enumerate(polygons):
            center = polygon.mean(axis=0)
            self.tree.insert("", "end", text=str(e), values=("RF" if trap.is_rf(e) else "DC", f"{center[0]:.1f}", f"{center[1]:.1f}", len(polygon)))

        ax = self.plot.axes()[0, 0]
        for e, polygon in enumerate(polygons):
            color = "tab:orange" if trap.is_rf(e) else "tab:blue"
            ax.add_patch(PolygonPatch(polygon[:, :2], closed=True, facecolor=colors.to_rgba(color, 0.35), edgecolor=color, linewidth=1))
            center = polygon.mean(axis=0)
            ax.text(center[0], center[1], str(e), ha="center", va="center", fontsize=8)

        # Region covered by the spherical harmonics expansion
        x_min = lib.constant("SPH_CENTER_X_MIN") - lib.constant("SPH_R")
        y_min = lib.constant("SPH_CENTER_Y_MIN") - lib.constant("SPH_R")
        x_span = (lib.constant("NSPH_X") - 1) * lib.constant("SPH_SPACING") + 2 * lib.constant("SPH_R")
        y_span = (lib.constant("NSPH_Y") - 1) * lib.constant("SPH_SPACING") + 2 * lib.constant("SPH_R")
        ax.add_patch(Rectangle((x_min, y_min), x_span, y_span, fill=False, linestyle="--", edgecolor="gray", label="Expansion region"))

        draw_photonics(ax, trap)
        draw_zones(ax, zones)

        if trap.solved and trap.n_rf > 0:
            try:
                null = trap.find_rf_null((0.0, 0.0, None))
                ax.plot(null[0], null[1], marker="x", color="tab:red", markersize=10, linestyle="none", label=f"RF null (z = {null[2]:.2f} µm)")
            except lib.TiOptError:
                pass

        all_vertices = np.vstack(polygons) if polygons else np.zeros((1, 3))
        margin = 20
        ax.set_xlim(all_vertices[:, 0].min() - margin, all_vertices[:, 0].max() + margin)
        ax.set_ylim(all_vertices[:, 1].min() - margin, all_vertices[:, 1].max() + margin)
        ax.set_aspect("equal")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_title(f"{trap.n_rf} RF and {trap.n_dc} DC electrodes (orange: RF, blue: DC)")
        ax.legend(loc="upper right", fontsize=8)
        self.plot.draw()


# ---------------------------------------------------------------------------------------------------------------------
# Trap builder
# ---------------------------------------------------------------------------------------------------------------------


class BuilderTab(Tab):
    """Canvas where traps are put together by dragging parts from a palette onto the trap surface, moving, rotating, and
    editing them, and then building them into a trap which the other tabs use.
    """

    # Palette entries by category, i.e. (label, kind of part, parameters)
    PALETTE = (
        ("Sections", (
            ("Linear section", "linear_section", {}),
            ("Curved section", "curved_section", {}),
            ("U-turn", "curved_section", {"angle": 180.0, "n_segments": 6}),
            ("X junction", "junction", {"arms": "0, 90, 180, 270"}),
            ("T junction", "junction", {"arms": "0, 90, 180"}),
            ("Y junction", "junction", {"arms": "0, 120, 240"}),
            ("Point trap", "point_trap", {}),
        )),
        ("Electrodes", (
            ("RF rail", "rf_rail", {}),
            ("DC electrode", "dc_electrode", {}),
            ("DC segments", "dc_segments", {}),
        )),
        ("Zones", tuple((f"{types.capitalize()} zone", "zone", {"types": types}) for types in ("load", "storage", "gate", "measure", "cool", "junction"))),
        ("Photonics", tuple((f"{beam.capitalize()} grating", "grating", {"beam": beam}) for beam in lib.PHOTONICS_BEAMS) + (("Detector", "detector", {}),)),
    )
    PALETTE_COLORS = {"Sections": "#fde3c8", "Electrodes": "#d6e6f5", "Zones": "#e6e6e6", "Photonics": "#eadcf2"}
    RF_FILL, RF_OUTLINE = "#f6c48f", "#d9822b"
    DC_FILL, DC_OUTLINE = "#a9cbe8", "#2b6cb0"
    SELECTED = "#e0218a"
    OVERLAP = "#d62728"
    HIT_RADIUS = 8  # px
    SNAP_DISTANCE = 8  # px

    def __init__(self, app):
        super().__init__(app, "Trap builder", plot=False)
        self.parts = []
        self.selected = None
        self.overlaps = []
        self.history = []
        self.nulls = []
        self.built_trap = None
        self.scale = 0.8  # px per µm
        self.center = np.zeros(2)  # point of the trap surface at the center of the canvas (µm)
        self.drag = None
        self.palette_drag = None

        for category, entries in self.PALETTE:
            frame = ttk.LabelFrame(self.controls, text=category, padding=4)
            frame.pack(fill="x", padx=(0, 4), pady=(0, 6))
            frame.columnconfigure(0, weight=1)
            frame.columnconfigure(1, weight=1)
            for i, (label, kind, params) in enumerate(entries):
                item = tk.Label(frame, text=label, relief="raised", borderwidth=1, padx=6, pady=3, background=self.PALETTE_COLORS[category], cursor="hand2")
                item.grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
                item.bind("<ButtonPress-1>", lambda event, kind=kind, params=params: self.palette_press(event, kind, params))
                item.bind("<B1-Motion>", self.palette_motion)
                item.bind("<ButtonRelease-1>", self.palette_release)

        self.properties_holder = ttk.Frame(self.controls)
        self.properties_holder.pack(fill="x")
        self.properties = None

        build = self.form("Build")
        build.entry("V_RF", "RF amplitude", 40, "V")
        build.entry("f_RF", "RF frequency", 40, "MHz")
        build.entry("grid", "Snap grid", 5, "µm")
        build.check("snap", "Snap to the grid and to the edges of other parts", True)
        self.build_form = build

        results = self.form("Built trap")
        self.results = results.text(12)

        hint = ("Drag parts from the left onto the trap surface. Drag a part to move it and empty space to pan, scroll to "
                "zoom, and press R to rotate, Delete to remove, Ctrl+D to duplicate, and Ctrl+Z to undo.")
        toolbar = ttk.Frame(self.main)
        toolbar.pack(fill="x", pady=(0, 2))
        for text, command in (("Build trap", self.build), ("Fit view", self.fit_view), ("Undo", self.undo), ("Start from current trap", self.import_trap),
                              ("Open design...", self.open_design), ("Save design...", self.save_design), ("Clear", self.clear)):
            ttk.Button(toolbar, text=text, command=command).pack(side="left", padx=(0, 4))
        ttk.Label(self.main, text=hint, foreground="gray", wraplength=900, justify="left").pack(fill="x", padx=4, pady=(0, 2))
        self.canvas = tk.Canvas(self.main, background="white", highlightthickness=1, highlightbackground="#cccccc")
        self.canvas.pack(fill="both", expand=True)
        self.status_label = ttk.Label(self.main, text="", anchor="w")
        self.status_label.pack(fill="x", padx=4)

        c = self.canvas
        c.bind("<Configure>", lambda event: self.redraw())
        c.bind("<ButtonPress-1>", self.press)
        c.bind("<B1-Motion>", self.motion)
        c.bind("<ButtonRelease-1>", self.release)
        c.bind("<Motion>", self.hover)
        c.bind("<MouseWheel>", lambda event: self.zoom(event, 1.2 if event.delta > 0 else 1 / 1.2))
        c.bind("<Button-4>", lambda event: self.zoom(event, 1.2))
        c.bind("<Button-5>", lambda event: self.zoom(event, 1 / 1.2))
        for key in ("<Delete>", "<BackSpace>"):
            c.bind(key, lambda event: self.delete_selected())
        c.bind("<KeyPress-r>", lambda event: self.rotate_selected(90))
        c.bind("<KeyPress-R>", lambda event: self.rotate_selected(-90))
        c.bind("<Control-d>", lambda event: self.duplicate_selected())
        c.bind("<Control-z>", lambda event: self.undo())
        c.bind("<Escape>", lambda event: self.select(None))
        for key, (dx, dy) in (("<Left>", (-1, 0)), ("<Right>", (1, 0)), ("<Up>", (0, 1)), ("<Down>", (0, -1))):
            c.bind(key, lambda event, dx=dx, dy=dy: self.nudge(dx, dy))

        self.show_properties()

    # Coordinates ----------------------------------------------------------------------------------------------------

    def canvas_size(self):
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        return (width, height) if width > 10 and height > 10 else (900, 650)

    def to_screen(self, points):
        """Convert points of the trap surface (µm) with shape (..., 2) to canvas pixels."""
        width, height = self.canvas_size()
        points = np.asarray(points, dtype=float)
        return np.stack([0.5 * width + (points[..., 0] - self.center[0]) * self.scale, 0.5 * height - (points[..., 1] - self.center[1]) * self.scale], axis=-1)

    def to_world(self, X, Y):
        """Convert canvas pixels to a point of the trap surface (µm)."""
        width, height = self.canvas_size()
        return np.array([self.center[0] + (X - 0.5 * width) / self.scale, self.center[1] - (Y - 0.5 * height) / self.scale])

    # Drawing ----------------------------------------------------------------------------------------------------------

    def redraw(self):
        c = self.canvas
        c.delete("all")
        width, height = self.canvas_size()
        x_min, y_max = self.to_world(0, 0)
        x_max, y_min = self.to_world(width, height)

        # Grid lines at least 25 px apart, with the axes darker
        spacing = next((s for s in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000) if s * self.scale >= 25), 10000)
        for x in np.arange(np.floor(x_min / spacing) * spacing, x_max + spacing, spacing):
            X = self.to_screen((x, 0.0))[0]
            c.create_line(X, 0, X, height, fill="#cccccc" if abs(x) < 1e-9 else "#f0f0f0")
        for y in np.arange(np.floor(y_min / spacing) * spacing, y_max + spacing, spacing):
            Y = self.to_screen((0.0, y))[1]
            c.create_line(0, Y, width, Y, fill="#cccccc" if abs(y) < 1e-9 else "#f0f0f0")
        c.create_text(width - 6, height - 6, text=f"grid {spacing:g} µm", anchor="se", fill="gray")

        # Region covered by the spherical harmonics expansions, where zones can be evaluated
        e_x_min, e_x_max, e_y_min, e_y_max = lib.expansion_region()
        (X0, Y0), (X1, Y1) = self.to_screen([(e_x_min, e_y_max), (e_x_max, e_y_min)])
        c.create_rectangle(X0, Y0, X1, Y1, outline="gray", dash=(4, 3))
        c.create_text(X0 + 4, Y0 + 2, text="Expansion region", anchor="nw", fill="gray")

        overlapping = {i for pair in self.overlaps for i in pair}
        zones = []
        for i, part in enumerate(self.parts):
            geometry = part.geometry()
            tags = ("part", f"part{i}")
            for polygon, rf in geometry.electrodes:
                fill, outline = (self.RF_FILL, self.RF_OUTLINE) if rf else (self.DC_FILL, self.DC_OUTLINE)
                if i in overlapping:
                    outline = self.OVERLAP
                c.create_polygon(*self.to_screen(polygon).ravel(), fill=fill, outline=outline, width=2 if i in overlapping else 1, tags=tags)
            zones += [position for _, _, position in geometry.zones]
        for i, part in enumerate(self.parts):
            self.draw_markers(part.geometry(), ("part", f"part{i}"), zones)

        if self.selected is not None and self.selected < len(self.parts):
            x0, y0, x1, y1 = self.parts[self.selected].geometry().bounds()
            (X0, Y0), (X1, Y1) = self.to_screen([(x0, y1), (x1, y0)])
            c.create_rectangle(X0 - 5, Y0 - 5, X1 + 5, Y1 + 5, outline=self.SELECTED, dash=(5, 3), width=2)

        for x, y, z, energy in self.nulls:
            X, Y = self.to_screen((x, y))
            c.create_line(X - 6, Y - 6, X + 6, Y + 6, fill="red", width=2)
            c.create_line(X - 6, Y + 6, X + 6, Y - 6, fill="red", width=2)
            c.create_text(X, Y + 8, text=f"{z:.1f} µm", anchor="n", fill="red", font=("TkDefaultFont", 8))

        if not self.parts:
            c.create_text(0.5 * width, 0.5 * height, text="Drag parts here from the palette on the left", fill="gray", font=("TkDefaultFont", 14))

    def draw_markers(self, geometry, tags, zones, dash=None):
        """Draw the zones, grating couplers (with a line to the zone they focus on), and detectors of a geometry."""
        c = self.canvas
        for (x, y), detector_width, detector_length in geometry.detectors:
            (X0, Y0), (X1, Y1) = self.to_screen([(x - 0.5 * detector_width, y + 0.5 * detector_length), (x + 0.5 * detector_width, y - 0.5 * detector_length)])
            c.create_rectangle(X0, Y0, X1, Y1, outline="#7b3294", fill="#7b3294", stipple="gray25", dash=dash, tags=tags)
        for beam, position, zone in geometry.gratings:
            color = colors.to_hex(BEAM_COLORS.get(beam, "gray"))
            X, Y = self.to_screen(position)
            if zones:
                target = zones[zone] if 0 <= zone < len(zones) else min(zones, key=lambda z: np.hypot(z[0] - position[0], z[1] - position[1]))
                TX, TY = self.to_screen(target)
                c.create_line(X, Y, TX, TY, fill=color, dash=(2, 3), arrow="last", tags=tags)
            c.create_rectangle(X - 5, Y - 5, X + 5, Y + 5, fill=color, outline="black", dash=dash, tags=tags)
            c.create_text(X, Y - 8, text=beam, anchor="s", fill=color, font=("TkDefaultFont", 8), tags=tags)
        for types, _, position in geometry.zones:
            X, Y = self.to_screen(position)
            c.create_oval(X - 5, Y - 5, X + 5, Y + 5, fill="black", outline="white", dash=dash, tags=tags)
            c.create_text(X, Y - 8, text=types, anchor="s", font=("TkDefaultFont", 9, "bold"), tags=tags)

    def draw_ghost(self, part):
        """Draw the outline of a part which is being dragged in from the palette."""
        c = self.canvas
        c.delete("ghost")
        geometry = part.geometry()
        for polygon, rf in geometry.electrodes:
            c.create_polygon(*self.to_screen(polygon).ravel(), fill="", outline=self.RF_OUTLINE if rf else self.DC_OUTLINE, dash=(4, 3), tags="ghost")
        self.draw_markers(geometry, "ghost", [position for _, _, position in combine_zones(self.parts)], dash=(2, 2))

    # Selection and editing --------------------------------------------------------------------------------------------

    def hit(self, X, Y):
        """Find the part under a pixel of the canvas, preferring zones, grating couplers, and detectors to electrodes."""
        point = self.to_world(X, Y)
        radius = self.HIT_RADIUS / self.scale
        for i in reversed(range(len(self.parts))):
            geometry = self.parts[i].geometry()
            for _, _, position in geometry.zones:
                if np.hypot(*(point - position)) <= radius:
                    return i
            for _, position, _ in geometry.gratings:
                if np.hypot(*(point - position)) <= radius:
                    return i
            for (x, y), detector_width, detector_length in geometry.detectors:
                if abs(point[0] - x) <= 0.5 * detector_width and abs(point[1] - y) <= 0.5 * detector_length:
                    return i
        for i in reversed(range(len(self.parts))):
            for polygon, _ in self.parts[i].geometry().electrodes:
                if MplPath(polygon).contains_point(point):
                    return i
        return None

    def select(self, index):
        self.selected = index
        self.show_properties()
        self.redraw()

    def show_properties(self):
        if self.properties is not None:
            self.properties.destroy()
        form = Form(self.properties_holder, "Selected part")
        form.pack(fill="x", pady=(0, 6), padx=(0, 4))
        self.properties = form
        if self.selected is None or self.selected >= len(self.parts):
            ttk.Label(form, text="Click a part on the trap surface to edit it", foreground="gray").grid(row=0, column=0, columnspan=3, sticky="w")
            return
        part = self.parts[self.selected]
        kind = builder.CATALOG[part.kind]
        ttk.Label(form, text=part.name, font=("TkDefaultFont", 10, "bold")).grid(row=form.row, column=0, columnspan=3, sticky="w")
        form.row += 1
        ttk.Label(form, text=kind.description, wraplength=360, foreground="gray", justify="left").grid(row=form.row, column=0, columnspan=3, sticky="w", pady=(0, 4))
        form.row += 1
        form.entry("x", "x", f"{part.x:g}", "µm")
        form.entry("y", "y", f"{part.y:g}", "µm")
        form.entry("rotation", "Rotation", f"{part.rotation:g}", "°")
        for name, value in part.params.items():
            if isinstance(value, (list, tuple)):
                continue
            label = name.replace("_", " ").capitalize()
            if name in kind.choices:
                form.combo(name, label, kind.choices[name], str(value))
            else:
                form.entry(name, label, f"{value:g}" if isinstance(value, float) else str(value), "µm" if isinstance(value, float) and name != "rotation" else "")
        form.buttons(("Apply", self.apply_properties), ("Rotate 90°", lambda: self.rotate_selected(90)))
        form.buttons(("Duplicate", self.duplicate_selected), ("Split up", self.explode_selected), ("Delete", self.delete_selected))

    def apply_properties(self):
        if self.selected is None:
            return
        part = self.parts[self.selected]
        form = self.properties
        params = dict(part.params)
        for name, default in builder.CATALOG[part.kind].defaults.items():
            if isinstance(default, (list, tuple)):
                continue
            if isinstance(default, bool) or isinstance(default, int):
                params[name] = form.get_int(name)
            elif isinstance(default, float):
                params[name] = form.get(name)
            else:
                params[name] = form.get_str(name).strip()
        candidate = builder.Part(part.kind, form.get("x"), form.get("y"), form.get("rotation"), params)
        candidate.geometry()
        if part.kind == "zone":
            lib.parse_layout(candidate.params["types"])
        self.push_history()
        self.parts[self.selected] = candidate
        self.changed({self.selected})
        self.show_properties()

    def push_history(self):
        self.history.append([part.to_dict() for part in self.parts])
        del self.history[:-100]

    def undo(self):
        if not self.history:
            self.app.set_status("Nothing to undo")
            return
        self.parts = [builder.Part.from_dict(values) for values in self.history.pop()]
        self.selected = None
        self.changed()
        self.show_properties()

    def changed(self, indices=None):
        """Recheck overlapping electrodes (of the given parts, or of every part) and redraw."""
        if indices is None:
            self.overlaps = builder.find_overlaps(self.parts)
        else:
            self.overlaps = [pair for pair in self.overlaps if not set(pair) & set(indices)] + builder.find_overlaps(self.parts, set(indices))
        self.nulls = []
        self.redraw()
        if self.overlaps:
            self.set_status(f"{len(self.overlaps)} pair{'s' if len(self.overlaps) != 1 else ''} of parts with overlapping electrodes (outlined in red)")

    def add_part(self, part):
        self.push_history()
        self.parts.append(part)
        self.selected = len(self.parts) - 1
        self.changed({self.selected})
        self.show_properties()

    def delete_selected(self):
        if self.selected is None:
            return
        self.push_history()
        del self.parts[self.selected]
        self.selected = None
        self.changed()
        self.show_properties()

    def rotate_selected(self, angle):
        if self.selected is None:
            return
        self.push_history()
        part = self.parts[self.selected]
        part.rotation = (part.rotation + angle) % 360
        self.changed({self.selected})
        self.show_properties()

    def duplicate_selected(self):
        if self.selected is None:
            return
        copy = builder.Part.from_dict(self.parts[self.selected].to_dict())
        x0, _, x1, _ = copy.geometry().bounds()
        copy.x += max(x1 - x0, 20.0)
        self.add_part(copy)

    def explode_selected(self):
        if self.selected is None:
            return
        pieces = self.parts[self.selected].explode()
        if len(pieces) < 2:
            self.set_status("This part only has one piece")
            return
        self.push_history()
        self.parts[self.selected:self.selected + 1] = pieces
        self.selected = None
        self.changed()
        self.show_properties()
        self.set_status(f"Split the part into {len(pieces)} parts")

    def nudge(self, dx, dy):
        if self.selected is None:
            return
        step = self.grid() or 1.0
        self.push_history()
        part = self.parts[self.selected]
        part.x += dx * step
        part.y += dy * step
        self.changed({self.selected})
        self.show_properties()

    def grid(self):
        try:
            return max(self.build_form.get("grid"), 0.0) if self.build_form.get("snap") else 0.0
        except ValueError:
            return 0.0

    def snap(self, part, x, y, exclude=None):
        """Snap a part placed at (x, y) to the grid, and then to the edges and centers of nearby parts."""
        if not self.build_form.get("snap"):
            return x, y
        grid = self.grid()
        if grid > 0:
            x = round(x / grid) * grid
            y = round(y / grid) * grid
        geometry = part.geometry()
        reach = self.SNAP_DISTANCE / self.scale
        # Sections join end to end where the axis of one leaves and the axis of another comes in (e.g. a curved section
        # onto a linear one), which takes precedence over the grid and the edges
        if geometry.ports:
            offset = np.array([x - part.x, y - part.y])
            best_port = (3 * reach, None)
            for i, other in enumerate(self.parts):
                if i == exclude:
                    continue
                for position, direction in other.geometry().ports:
                    for own, own_direction in geometry.ports:
                        if abs((own_direction - direction) % 360 - 180) > 1.0:
                            continue
                        gap = np.asarray(position) - (np.asarray(own) + offset)
                        if np.linalg.norm(gap) < best_port[0]:
                            best_port = (np.linalg.norm(gap), gap)
            if best_port[1] is not None:
                return x + best_port[1][0], y + best_port[1][1]
        relative = geometry.bounds() - np.array([part.x, part.y, part.x, part.y])
        # Zones, gratings, and detectors snap to the centers of other parts (e.g. onto the axis of a section), and
        # electrodes also snap to their edges
        markers_only = not geometry.electrodes
        own = relative + np.array([x, y, x, y])
        near = 5 * reach
        best = [reach, reach]
        shift = [0.0, 0.0]
        for i, other in enumerate(self.parts):
            if i == exclude:
                continue
            bounds = other.geometry().bounds()
            for axis in (0, 1):
                across = 1 - axis
                # Only parts which are next to this one (across the axis being snapped) are snapped to
                if own[across] > bounds[across + 2] + near or bounds[across] > own[across + 2] + near:
                    continue
                edges = (own[axis], own[axis + 2], 0.5 * (own[axis] + own[axis + 2]))
                targets = (0.5 * (bounds[axis] + bounds[axis + 2]),) if markers_only else (bounds[axis], bounds[axis + 2], 0.5 * (bounds[axis] + bounds[axis + 2]))
                for edge in edges:
                    for target in targets:
                        if abs(target - edge) < best[axis]:
                            best[axis] = abs(target - edge)
                            shift[axis] = target - edge
        return x + shift[0], y + shift[1]

    # Mouse ------------------------------------------------------------------------------------------------------------

    def press(self, event):
        self.canvas.focus_set()
        index = self.hit(event.x, event.y)
        if index is not None:
            part = self.parts[index]
            self.drag = dict(mode="move", index=index, start=self.to_world(event.x, event.y), origin=(part.x, part.y), moved=False)
            if index != self.selected:
                self.select(index)
        else:
            self.drag = dict(mode="pan", last=(event.x, event.y))
            if self.selected is not None:
                self.select(None)

    def motion(self, event):
        drag = self.drag
        if drag is None:
            return
        if drag["mode"] == "pan":
            dx = event.x - drag["last"][0]
            dy = event.y - drag["last"][1]
            drag["last"] = (event.x, event.y)
            self.center = self.center - np.array([dx, -dy]) / self.scale
            self.redraw()
            return
        if not drag["moved"]:
            self.push_history()
            drag["moved"] = True
        part = self.parts[drag["index"]]
        point = self.to_world(event.x, event.y)
        x = drag["origin"][0] + point[0] - drag["start"][0]
        y = drag["origin"][1] + point[1] - drag["start"][1]
        part.x, part.y = self.snap(part, x, y, exclude=drag["index"])
        self.nulls = []
        self.redraw()
        self.set_status(f"{part.name} at ({part.x:.1f}, {part.y:.1f}) µm")

    def release(self, event):
        drag = self.drag
        self.drag = None
        if drag is not None and drag["mode"] == "move" and drag["moved"]:
            self.changed({drag["index"]})
            self.show_properties()

    def hover(self, event):
        x, y = self.to_world(event.x, event.y)
        index = self.hit(event.x, event.y)
        over = f" over {self.parts[index].name}" if index is not None else ""
        self.status_label.configure(text=f"({x:.1f}, {y:.1f}) µm{over}  |  {len(self.parts)} parts")

    def zoom(self, event, factor):
        point = self.to_world(event.x, event.y)
        self.scale = float(np.clip(self.scale * factor, 0.02, 50.0))
        # Keep the point under the cursor in place
        self.center = self.center + point - self.to_world(event.x, event.y)
        self.redraw()

    def fit_view(self):
        if not self.parts:
            self.center = np.zeros(2)
            self.scale = 0.8
        else:
            bounds = builder.combine(self.parts).bounds()
            width, height = self.canvas_size()
            span = np.maximum(bounds[2:] - bounds[:2], 20.0)
            self.scale = float(min(0.85 * width / span[0], 0.85 * height / span[1]))
            self.center = 0.5 * (bounds[:2] + bounds[2:])
        self.redraw()

    # Palette ----------------------------------------------------------------------------------------------------------

    def over_canvas(self, event):
        X = event.x_root - self.canvas.winfo_rootx()
        Y = event.y_root - self.canvas.winfo_rooty()
        return (0 <= X < self.canvas.winfo_width() and 0 <= Y < self.canvas.winfo_height()), X, Y

    def palette_press(self, event, kind, params):
        self.palette_drag = dict(kind=kind, params=dict(params), moved=False, part=None)
        self.canvas.configure(cursor="crosshair")

    def palette_motion(self, event):
        drag = self.palette_drag
        if drag is None:
            return
        drag["moved"] = True
        inside, X, Y = self.over_canvas(event)
        if not inside:
            drag["part"] = None
            self.canvas.delete("ghost")
            return
        part = builder.Part(drag["kind"], params=dict(drag["params"]))
        part.x, part.y = self.snap(part, *self.to_world(X, Y))
        drag["part"] = part
        self.draw_ghost(part)

    def palette_release(self, event):
        drag = self.palette_drag
        self.palette_drag = None
        self.canvas.configure(cursor="")
        self.canvas.delete("ghost")
        if drag is None:
            return
        if drag["part"] is not None:
            self.add_part(drag["part"])
        elif not drag["moved"]:
            # A click puts the part in the middle of the view
            part = builder.Part(drag["kind"], params=dict(drag["params"]))
            part.x, part.y = self.snap(part, *self.center)
            self.add_part(part)
        else:
            return
        self.set_status(f"Added {self.parts[-1].name} at ({self.parts[-1].x:.1f}, {self.parts[-1].y:.1f}) µm")

    def drop(self, kind, x, y, **params):
        """Drop a part from the palette at (x, y) (µm), snapping it like a drag would."""
        part = builder.Part(kind, params=params)
        part.x, part.y = self.snap(part, x, y)
        self.add_part(part)
        return part

    # Designs and building ---------------------------------------------------------------------------------------------

    def set_status(self, text):
        self.app.set_status(text)

    def clear(self):
        if not self.parts:
            return
        self.push_history()
        self.parts = []
        self.selected = None
        self.changed()
        self.show_properties()

    def import_trap(self):
        trap = self.app.trap
        if trap is None:
            raise ValueError("Generate or load a trap in the Trap tab first")
        self.push_history()
        self.parts = builder.parts_from_trap(trap)
        self.selected = None
        self.changed()
        self.show_properties()
        self.fit_view()
        self.set_status(f"Started from the current trap, with {len(self.parts)} parts")

    def open_design(self):
        path = filedialog.askopenfilename(parent=self, title="Open trap design", initialdir=REPOSITORY, filetypes=[("Trap designs", "*.json"), ("All files", "*")])
        if not path:
            return
        parts = builder.load_design(path)
        self.push_history()
        self.parts = parts
        self.selected = None
        self.changed()
        self.show_properties()
        self.fit_view()

    def save_design(self):
        path = filedialog.asksaveasfilename(parent=self, title="Save trap design", initialdir=REPOSITORY, defaultextension=".json", filetypes=[("Trap designs", "*.json")])
        if path:
            builder.save_design(self.parts, path)
            self.set_status(f"Saved the design to {path}")

    def build(self):
        if not self.parts:
            raise ValueError("Drag some parts onto the trap surface first")
        parts = [builder.Part.from_dict(part.to_dict()) for part in self.parts]
        element = self.app.ion.element
        ion = self.app.ion
        V_RF = self.build_form.get("V_RF")
        Omega_RF = TWO_PI * self.build_form.get("f_RF") * 1e6
        x_min, x_max, y_min, y_max = lib.expansion_region()

        def work():
            trap = builder.build_trap(parts, element, V_RF, Omega_RF)
            trap.solve("analytical")
            # The RF null nearest each zone, across the local axis of the trap there (along curved sections too)
            return trap, list(zip(trap.zones(), builder.zone_nulls(trap, ion)))

        def done(result):
            trap, nulls = result
            self.app.set_trap(trap)
            self.built_trap = trap
            self.nulls = [(point[0], point[1], point[2], energy) for _, best in nulls if best is not None for point, energy in [best]]
            lines = [
                f"{trap.n_rf} RF and {trap.n_dc} DC electrodes, {len(nulls)} zone{'s' if len(nulls) != 1 else ''}, {len(trap.grating_couplers())} grating coupler{'s' if len(trap.grating_couplers()) != 1 else ''}, and {len(trap.detectors())} detector{'s' if len(trap.detectors()) != 1 else ''}",
                "The trap is now the current trap of every tab.",
            ]
            outside = []
            for z, (zone, best) in enumerate(nulls):
                where = f"zone {z} ({zone.types}) at ({zone.position[0]:.0f}, {zone.position[1]:.0f})"
                inside = x_min <= zone.position[0] <= x_max and y_min <= zone.position[1] <= y_max
                if not inside:
                    outside.append(zone)
                if best is None:
                    lines.append(f"{where}: no RF null nearby")
                else:
                    point, energy = best
                    note = " (micromotion, e.g. near the end of a rail)" if energy > 0.1 else ""
                    lines.append(f"{where}: RF null {point[2]:.1f} µm up, {np.hypot(point[0] - zone.position[0], point[1] - zone.position[1]):.1f} µm across, pseudopotential {energy:.3f} meV{note}")
            if outside:
                xs = [zone.position[0] for zone, _ in nulls]
                ys = [zone.position[1] for zone, _ in nulls]
                lines.append(f"{len(outside)} zone{'s are' if len(outside) != 1 else ' is'} beyond the spherical harmonic expansions, so wells and waveforms can't be "
                             f"solved there (the RF nulls and CTC paths come from the gapless plane potentials, which reach everywhere); zones out to x = {min(xs):.0f} and "
                             f"{max(xs):.0f} µm need a build with NSPH_X = {lib.required_nsph_x(min(xs), max(xs))}"
                             + (f" and y out to {min(ys):.0f} and {max(ys):.0f} µm a larger NSPH_Y too" if min(ys) < y_min or max(ys) > y_max else "")
                             + f" (make clean && make DEFINES=\"-DNSPH_X={lib.required_nsph_x(min(xs), max(xs))}\").")
            set_text(self.results, "\n".join(lines))
            self.redraw()

        self.app.run_task("Building the trap", work, done)

    def refresh(self):
        # RF nulls only belong to the trap this tab built
        if self.app.trap is not self.built_trap and self.nulls:
            self.nulls = []
            self.redraw()


def combine_zones(parts):
    """Get the zones of every part of a design."""
    return builder.combine(parts).zones


# Colors of the beams of PHOTONICS_BEAMS
BEAM_COLORS = {"qubit": "tab:red", "detection": "tab:purple", "repump": "tab:brown", "quench": "tab:pink", "photoionization": "tab:blue", "ionization": "tab:cyan"}


def wavelength_color(wavelength):
    """Get the plot color for light of a wavelength (µm), distinguishing the 40Ca+ beams."""
    for limit, color in ((0.45, "tab:purple"), (0.8, "tab:red"), (0.86, "tab:pink")):
        if wavelength < limit:
            return color
    return "tab:brown"


def draw_zones(ax, zones):
    """Mark the zones of a trap on xy axes, labeled with their types."""
    for z, zone in enumerate(zones):
        ax.plot(zone.position[0], zone.position[1], marker="o", color="black", markersize=4, linestyle="none", label="Zones" if z == 0 else None)
        ax.annotate(zone.types, (zone.position[0], zone.position[1]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)


def draw_photonics(ax, trap):
    """Draw the grating couplers (with their beam directions) and detectors of a trap on xy axes."""
    labelled = set()
    for grating in trap.grating_couplers():
        position = grating.position
        wavelength = grating.struct.waveguide.wavelength
        color = BEAM_COLORS.get(grating.beam, wavelength_color(wavelength))
        label = f"{wavelength * 1e3:.0f} nm" + (f" {grating.beam}" if grating.beam else " grating")
        ax.plot(position[0], position[1], marker="s", color=color, linestyle="none", label=label if label not in labelled else None)
        labelled.add(label)
        emission = grating.emission()
        if emission is not None:
            theta, k_hat = emission
            ax.annotate("", xy=(position[0] + 25 * k_hat[0], position[1] + 25 * k_hat[1]), xytext=(position[0], position[1]), arrowprops=dict(arrowstyle="->", color=color))
    for d, detector in enumerate(trap.detectors()):
        position = detector.position
        ax.add_patch(Rectangle(
            (position[0] - 0.5 * detector.width, position[1] - 0.5 * detector.length), detector.width, detector.length,
            fill=False, hatch="//", edgecolor="tab:purple", linewidth=1, label="Detector" if d == 0 else None,
        ))


# ---------------------------------------------------------------------------------------------------------------------
# Electrostatics
# ---------------------------------------------------------------------------------------------------------------------


class ElectrostaticsTab(Tab):
    QUANTITIES = ("Pseudopotential", "Electrode potential at 1 V", "Static well (Waveforms tab)")
    PLANES = ("xz (at y)", "yz (at x)", "xy (at z)")

    def __init__(self, app):
        super().__init__(app, "Electrostatics")

        solver = self.form("Solver")
        solver.combo("solver", "Solver", ("analytical", "spectral", "relaxation", "BEM (finite gaps)"), "analytical")
        solver.buttons(("Solve", self.solve))
        self.solver = solver

        bem_form = self.form("Boundary element method")
        bem_form.entry("gap", "Gaps opened between electrodes", 5, "µm")
        bem_form.entry("h_min", "Panels at the edges", 4, "µm")
        bem_form.entry("ground_depth", "Grounded plane beneath (blank for none)", "", "µm")
        bem_form.entry("cover_height", "Grounded cover above (blank for none)", "", "µm")
        bem_form.entry("frame", "Grounded frame around the trap", 3000, "µm")
        self.bem_form = bem_form

        view = self.form("Map")
        view.combo("quantity", "Quantity", self.QUANTITIES, self.QUANTITIES[0])
        view.entry("electrode", "Electrode", 3)
        view.combo("plane", "Plane", self.PLANES, self.PLANES[0])
        view.entry("offset", "Plane offset", 0, "µm")
        view.entry("half_width", "Horizontal half range", 60, "µm")
        view.entry("z_min", "Lowest height", 40, "µm")
        view.entry("z_max", "Highest height", 70, "µm")
        view.entry("resolution", "Points per axis", 81)
        view.buttons(("Plot", self.plot_map))
        self.view = view

        results = self.form("RF null")
        self.results = results.text(6)

        site = self.form("Trapping site")
        site.entry("x", "x", 0, "µm")
        site.entry("y", "y", 0, "µm")
        site.combo("voltages", "Voltages", ("Static well (Waveforms tab)", "Pseudopotential only"), "Static well (Waveforms tab)")
        site.entry("n_ions", "Ions in the crystal of the report", 2)
        site.combo("order", "Highest multipole order", ("2", "3", "4"), "2")
        site.buttons(("Analyze", self.analyze_site), ("Stability diagram", self.show_stability))
        site.buttons(("Escape point", self.show_escape), ("Multipoles", self.show_multipoles), ("Save shims...", self.save_shims))
        self.site = site
        self.site_results = site.text(14)
        self.control = None

        self.plot.message("Generate a trap, then plot its potentials")

    def solve(self):
        trap = self.require_trap()
        solver = self.solver.get_str("solver")
        if solver == "relaxation" and not messagebox.askyesno(
            "Relaxation solver", "The relaxation solver solves Laplace's equation on a large grid for every electrode, which can take many minutes. Continue?", parent=self
        ):
            return
        if solver.startswith("BEM"):
            form = self.bem_form
            kwargs = dict(gap=form.get("gap"), h_min=form.get("h_min"), ground_depth=parse_optional(form.get_str("ground_depth")),
                          cover_height=parse_optional(form.get_str("cover_height")), frame=form.get("frame"))
            if kwargs["h_min"] <= 0.5:
                raise ValueError("Panels smaller than 0.5 µm would make a very large system")

            def work_bem():
                solution = bem.solve(trap, **kwargs)
                return solution

            def done_bem(solution):
                self.app.trap_changed()
                self.app.set_status(f"Solved with {solution.mesh.n_panels} boundary elements")

            self.app.run_task("Solving electrodynamics (boundary element method)", work_bem, done_bem)
            return

        def work():
            trap.solve(solver)
            return trap

        self.app.run_task(f"Solving electrodynamics ({solver})", work, lambda trap: self.app.trap_changed())

    def _site(self):
        """Trap, ion, voltages, and starting point of the trapping site in the form."""
        trap = self.require_trap()
        ion = self.app.ion
        x, y = self.site.get("x"), self.site.get("y")
        if self.site.get_str("voltages").startswith("Static"):
            if self.app.waveform is None:
                raise ValueError("Solve a static well in the Waveforms tab first, or analyze the pseudopotential only")
            voltages = self.app.waveform.voltages[0]
        else:
            voltages = trap.zero_voltages()
        if trap.n_rf > 0:
            position = trap.find_rf_null((x, y, None))
        else:
            position = np.array([x, y, ION_HEIGHT])
        return trap, ion, voltages, position

    def analyze_site(self):
        trap, ion, voltages, position = self._site()
        n_ions = self.site.get_int("n_ions")
        try:
            k_vector = beam_wavevector(self.app, "Qubit grating", position)
        except (ValueError, IndexError):
            k_vector = None

        def work():
            lines = stability.report(trap, ion, position, voltages, k_vector, n_ions)
            minimum = trap.find_minimum(ion, position, voltages)
            omega, axes = trap.normal_modes(ion, minimum, voltages)
            u = np.linspace(-15, 15, 121)
            cuts = [trap.potential_energy(ion, minimum + u[:, None] * axis[None, :], voltages) for axis in axes]
            return lines, minimum, axes, u, cuts

        def done(output):
            lines, minimum, axes, u, cuts = output
            set_text(self.site_results, "\n".join(lines))
            ax_cuts, ax_map = self.plot.axes(1, 2)[0]
            for axis, cut in zip(axes, cuts):
                label = f"along ({axis[0]:+.2f}, {axis[1]:+.2f}, {axis[2]:+.2f})"
                ax_cuts.plot(u, (cut - cut.min()) * MEV, label=label)
            ax_cuts.set_xlabel("Distance from the minimum along each principal axis (µm)")
            ax_cuts.set_ylabel("Potential energy (meV)")
            ax_cuts.legend(fontsize=7)
            ax_cuts.set_title("Potential along the principal axes")
            # Cross-section of the potential energy across the trap axis
            v = np.linspace(-25, 25, 61)
            Y, Z = np.meshgrid(minimum[1] + v, minimum[2] + v)
            points = np.stack([np.full_like(Y, minimum[0]), Y, Z], axis=-1)
            U = trap.potential_energy(ion, points, voltages) * MEV
            mesh = ax_map.pcolormesh(Y, Z, U - U.min(), shading="auto", cmap="viridis", norm=colors.PowerNorm(0.5))
            self.plot.figure.colorbar(mesh, ax=ax_map, label="Potential energy (meV)")
            for axis in axes:
                ax_map.annotate("", xy=(minimum[1] + 12 * axis[1], minimum[2] + 12 * axis[2]), xytext=(minimum[1], minimum[2]), arrowprops=dict(arrowstyle="->", color="white"))
            ax_map.set_xlabel("y (µm)")
            ax_map.set_ylabel("z (µm)")
            ax_map.set_aspect("equal")
            ax_map.set_title(f"Cross-section at x = {minimum[0]:.1f} µm (arrows: principal axes)")
            self.plot.draw()

        self.app.run_task("Analyzing the trapping site", work, done)

    def show_stability(self):
        trap, ion, voltages, position = self._site()
        if trap.n_rf == 0:
            raise ValueError("The trap has no RF electrodes")

        def work():
            minimum = trap.find_minimum(ion, position, voltages)
            return stability.stability_diagram(trap, ion, minimum, voltages, n_points=61), stability.floquet_at(trap, ion, minimum, voltages)

        def done(output):
            diagram, result = output
            ax = self.plot.axes()[0, 0]
            values = np.log10(np.maximum(diagram.max_multiplier, 1.0))
            mesh = ax.pcolormesh(diagram.V_RF, diagram.dc_scales, values, shading="auto", cmap="magma_r")
            self.plot.figure.colorbar(mesh, ax=ax, label="log10 of the largest Floquet multiplier (0 where stable)")
            ax.contour(diagram.V_RF, diagram.dc_scales, diagram.stable.astype(float), levels=[0.5], colors="tab:cyan", linewidths=1.5)
            ax.plot(diagram.operating_V_RF, 1.0, marker="*", color="tab:green", markersize=14, label="Operating point")
            ax.set_xlabel("RF amplitude (V)")
            ax.set_ylabel("Scale of the DC voltages")
            ax.set_title(f"Stability (cyan: edge of the stable region); |q| = {diagram.q_max:.3f}, |a| = {diagram.a_max:.2e} at the operating point")
            ax.legend(fontsize=8)
            self.plot.draw()
            set_text(self.site_results, "\n".join([
                "Floquet analysis at the operating point: " + ("stable" if result.stable else "UNSTABLE"),
                "Secular frequencies: " + ", ".join(f"{w / TWO_PI / 1e6:.4f}" for w in np.sort(result.frequencies)) + " MHz",
                "q along the axes of Q: " + ", ".join(f"{q:+.4f}" for q in result.q),
                "a along the axes of Q: " + ", ".join(f"{a:+.2e}" for a in result.a),
            ]))

        self.app.run_task("Mapping the stability of the site", work, done)

    def show_escape(self):
        trap, ion, voltages, position = self._site()

        def work():
            return stability.escape_point(trap, ion, position, voltages)

        def done(result):
            ax_top, ax_side, ax_energy = self.plot.axes(1, 3)[0]
            draw_electrodes(ax_top, trap)
            for ax, (i, j), names in ((ax_top, (0, 1), ("x", "y")), (ax_side, (0, 2), ("x", "z"))):
                ax.plot(result.path[:, i], result.path[:, j], color="tab:red", label="Path of least resistance")
                ax.plot(result.minimum[i], result.minimum[j], marker="o", color="black", label="Minimum")
                ax.plot(result.saddle[i], result.saddle[j], marker="*", markersize=12, color="tab:green", label="Saddle point")
                ax.set_xlabel(f"{names[0]} (µm)")
                ax.set_ylabel(f"{names[1]} (µm)")
            ax_side.legend(fontsize=7)
            span = np.ptp(result.path[:, :2], axis=0).max() + 60
            center = result.minimum[:2]
            ax_top.set_xlim(center[0] - span, center[0] + span)
            ax_top.set_ylim(center[1] - span, center[1] + span)
            energy = stability.AnalyticalEnergy(trap, ion, voltages)
            s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(result.path, axis=0), axis=1))])
            U = energy(result.path)
            ax_energy.plot(s, (U - U[0]) * MEV)
            ax_energy.axhline(result.depth * 1e3, color="tab:green", linestyle="--", label=f"Depth {'at least ' if result.lower_bound else ''}{result.depth * 1e3:.2f} meV")
            # Past the saddle, the path may fall far onto the electrodes, so the plot shows the barrier
            ax_energy.set_ylim(-0.5 * result.depth * 1e3, 1.6 * result.depth * 1e3)
            ax_energy.set_xlabel("Distance along the path (µm)")
            ax_energy.set_ylabel("Potential energy (meV)")
            ax_energy.legend(fontsize=8)
            self.plot.draw()
            set_text(self.site_results, "\n".join([
                (f"Depth: at least {result.depth * 1e3:.3f} meV (the potential still rises at the edge of the search)" if result.lower_bound
                 else f"Depth: {result.depth * 1e3:.3f} meV" + (" (saddle refined)" if result.refined else " (on the grid)")),
                "Saddle point: (" + ", ".join(f"{v:.2f}" for v in result.saddle) + ") µm",
                "Barrier along the vertical: " + (f"{result.vertical_depth * 1e3:.3f} meV" if np.isfinite(result.vertical_depth) else "none within the search"),
            ]))

        self.app.run_task("Finding the escape point", work, done)

    def show_multipoles(self):
        trap, ion, voltages, position = self._site()
        order = int(self.site.get_str("order"))

        def work():
            minimum = trap.find_minimum(ion, position, voltages) if np.any(voltages != 0) else position
            control = multipoles.control_matrix(trap, minimum, order)
            return control, control.multipoles(voltages)

        def done(output):
            control, values = output
            self.control = control
            ax_values, ax_matrix = self.plot.axes(1, 2, width_ratios=[1, 1.4])[0]
            names = control.names[1:]
            ax_values.barh(range(len(names)), [values[name] for name in names], color="tab:blue")
            ax_values.set_yticks(range(len(names)), [f"{name} ({multipoles.unit(name)})" for name in names], fontsize=7)
            ax_values.invert_yaxis()
            ax_values.set_title("Multipoles of the voltages")
            matrix = control.matrix[1:]
            limit = np.abs(matrix).max()
            image = ax_matrix.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
            self.plot.figure.colorbar(image, ax=ax_matrix, label="Multipole per volt")
            ax_matrix.set_yticks(range(len(names)), names, fontsize=7)
            ax_matrix.set_xlabel("Electrode")
            ax_matrix.set_title("Multipoles of every electrode at 1 V")
            self.plot.draw()
            lines = [f"Multipoles at (" + ", ".join(f"{v:.2f}" for v in control.position) + ") µm:"]
            lines += [f"  {name}: {values[name]:+.4e} {multipoles.unit(name)}" for name in control.names]
            set_text(self.site_results, "\n".join(lines))

        self.app.run_task("Expanding the potentials in multipoles", work, done)

    def save_shims(self):
        if self.control is None:
            raise ValueError("Show the multipoles of a site first")
        shims = multipoles.shim_matrix(self.control)
        path = filedialog.asksaveasfilename(parent=self, title="Save the control voltages of the multipoles", initialdir=REPOSITORY, defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if path:
            multipoles.save_shim_matrix(path, shims)
            self.app.set_status(f"Saved the control voltages of {', '.join(multipoles.SHIMS)} to {path}")

    def refresh(self):
        trap = self.app.trap
        if trap is None or not trap.solved:
            return

        lines = [f"Solver: {trap.solver}"]
        if trap.n_rf > 0:
            try:
                ion = self.app.ion
                null = trap.find_rf_null((0.0, 0.0, None))
                omega, axes = trap.normal_modes(ion, null)
                _, _, H_rf = trap.evaluate(null, trap.rf_weights())
                q_z = 2 * ion.charge * trap.V_RF * abs(H_rf[2, 2]) / LENGTH_SCALE ** 2 / (ion.mass * trap.Omega_RF ** 2)
                lines.append(f"RF null at x = 0: y = {null[1]:.3f} µm, z = {null[2]:.3f} µm")
                lines.append("RF-only secular frequencies: " + ", ".join(f"{w / TWO_PI / 1e6:.3f}" for w in omega) + " MHz")
                lines.append(f"Mathieu parameter q_z = {q_z:.3f}" + (" (unstable, q > 0.908)" if q_z > 0.908 else ""))
            except lib.TiOptError as error:
                lines.append(str(error))
        else:
            lines.append("The trap has no RF electrodes")
        set_text(self.results, "\n".join(lines))

    def plot_map(self):
        trap = self.require_trap()
        ion = self.app.ion
        quantity = self.view.get_str("quantity")
        plane = self.PLANES.index(self.view.get_str("plane"))
        offset = self.view.get("offset")
        half_width = self.view.get("half_width")
        z_min = self.view.get("z_min")
        z_max = self.view.get("z_max")
        n = self.view.get_int("resolution")
        electrode = self.view.get_int("electrode")

        if quantity == self.QUANTITIES[1] and not 0 <= electrode < trap.n_electrodes:
            raise ValueError(f"Electrode must be between 0 and {trap.n_electrodes - 1}")
        if quantity == self.QUANTITIES[2] and self.app.waveform is None:
            raise ValueError("Solve a static well or transport in the Waveforms tab first")

        voltages = None
        if quantity == self.QUANTITIES[2]:
            voltages = self.app.waveform.voltages[0]

        # Grid of points in the chosen plane
        u = np.linspace(-half_width, half_width, n)
        if plane == 2:
            v = np.linspace(-half_width, half_width, n)
            U_grid, V_grid = np.meshgrid(u, v)
            points = np.stack([U_grid, V_grid, np.full_like(U_grid, offset)], axis=-1)
            labels = ("x (µm)", "y (µm)")
        else:
            v = np.linspace(z_min, z_max, n)
            U_grid, V_grid = np.meshgrid(u, v)
            if plane == 0:
                points = np.stack([U_grid, np.full_like(U_grid, offset), V_grid], axis=-1)
                labels = ("x (µm)", "z (µm)")
            else:
                points = np.stack([np.full_like(U_grid, offset), U_grid, V_grid], axis=-1)
                labels = ("y (µm)", "z (µm)")

        line = np.stack([np.zeros(200), np.zeros(200), np.linspace(z_min, z_max, 200)], axis=-1)

        def evaluate(positions):
            if quantity == self.QUANTITIES[0]:
                return trap.pseudopotential(ion, positions) * MEV
            if quantity == self.QUANTITIES[1]:
                weights = trap.zero_voltages()
                weights[electrode] = 1.0
                return trap.potential(positions, weights)
            return trap.potential_energy(ion, positions, voltages) * MEV

        def work():
            return evaluate(points), evaluate(line)

        def done(result):
            values, line_values = result
            unit = "V" if quantity == self.QUANTITIES[1] else "meV"
            if quantity == self.QUANTITIES[2]:
                values = values - values.min()
                line_values = line_values - line_values.min()

            ax_map, ax_line = self.plot.axes(1, 2, width_ratios=[1.6, 1])[0]
            if quantity == self.QUANTITIES[0]:
                norm = colors.LogNorm(vmin=max(values.max() * 1e-4, values[values > 0].min() if np.any(values > 0) else 1e-12), vmax=values.max())
                mesh = ax_map.pcolormesh(U_grid, V_grid, np.maximum(values, norm.vmin), norm=norm, shading="auto", cmap="viridis")
            else:
                mesh = ax_map.pcolormesh(U_grid, V_grid, values, shading="auto", cmap="viridis")
                ax_map.contour(U_grid, V_grid, values, levels=12, colors="white", linewidths=0.5)
            self.plot.figure.colorbar(mesh, ax=ax_map, label=f"{quantity} ({unit})")
            ax_map.set_xlabel(labels[0])
            ax_map.set_ylabel(labels[1])
            ax_map.set_aspect("equal")
            ax_map.set_title(quantity)

            ax_line.plot(line_values, line[:, 2])
            ax_line.set_xlabel(f"{quantity} ({unit})")
            ax_line.set_ylabel("z (µm) at x = y = 0")
            if quantity == self.QUANTITIES[0] and trap.n_rf > 0:
                null = trap.find_rf_null((0.0, 0.0, None))
                ax_line.axhline(null[2], color="tab:red", linestyle="--", label="RF null")
                ax_line.legend(fontsize=8)
            ax_line.set_title("Along the z-axis")
            self.plot.draw()

        self.app.run_task(f"Evaluating {quantity.lower()}", work, done)


# ---------------------------------------------------------------------------------------------------------------------
# Waveforms
# ---------------------------------------------------------------------------------------------------------------------


class WaveformTab(Tab):
    def __init__(self, app):
        super().__init__(app, "Waveforms")

        transport = self.form("Transport")
        transport.entry("x_start", "Start", -50, "µm")
        transport.entry("x_end", "End", 50, "µm")
        transport.entry("duration", "Duration", 50, "µs")
        transport.entry("n_steps", "Steps", 501)
        transport.entry("f_axial", "Axial frequency", f"{AXIAL_FREQUENCY / TWO_PI / 1e6:g}", "MHz")
        transport.entry("max_voltage", "Voltage limit", 10, "V")
        transport.check("refine", "Enforce the limit with NLopt (slow)", False)
        transport.buttons(("Static well", self.static_well), ("Transport", self.transport))
        self.transport_form = transport

        objectives = self.form("Objectives and constraints (pytrans style)")
        objectives.combo("profile", "Transport profile", tuple(voltage_objectives.PROFILES), "sine squared")
        objectives.entry("max_slew", "Largest slew rate (0 for none)", 0, "V/µs")
        objectives.entry("tilt", "Tilt of the radial axes (blank for free)", "", "°")
        objectives.entry("roughness", "Weight on the roughness of the voltages", 1e-3)
        objectives.buttons(("Static well", lambda: self.solve_objectives(True)), ("Transport", lambda: self.solve_objectives(False)))
        self.objectives = objectives

        dac = self.form("DAC and filters")
        dac.entry("rate", "DAC update rate", 100, "MS/s")
        dac.entry("bits", "DAC resolution", 16, "bits")
        dac.entry("range", "DAC range", 10, "± V")
        dac.combo("filter", "Filter on every electrode", ("None", "RC", "Butterworth"), "RC")
        dac.entry("cutoff", "Cutoff frequency", 1, "MHz")
        dac.entry("order", "Order (Butterworth)", 2)
        dac.check("compensate", "Pre-compensate the waveform for the filter", True)
        dac.entry("smoothing", "Smoothing of the compensation", 1e-4)
        dac.buttons(("Apply", self.apply_hardware), ("Use the electrode voltages", self.use_hardware_output))
        self.dac = dac
        self.hardware_result = None
        self.hardware_source = None

        results = self.form("Results")
        self.results = results.text(8)

        self.plot.message("Solve a static well or a transport")

    def solve_objectives(self, static):
        trap = self.require_trap()
        ion = self.app.ion
        form = self.transport_form
        x_start = form.get("x_start")
        x_end = x_start if static else form.get("x_end")
        duration = form.get("duration") * 1e-6
        n_steps = form.get_int("n_steps")
        omega_axial = self.omega_axial()
        max_voltage = form.get("max_voltage")
        objectives = self.objectives
        tilt = parse_optional(objectives.get_str("tilt"))
        max_slew = objectives.get("max_slew") * 1e6
        profile = objectives.get_str("profile")
        roughness = objectives.get("roughness")
        if not static and n_steps < 2:
            raise ValueError("Transports need at least 2 steps")

        def work():
            start = trap.find_rf_null((x_start, 0.0, None)) if trap.n_rf > 0 else np.array([x_start, 0.0, ION_HEIGHT])
            if static:
                voltages, info = voltage_objectives.static_well(trap, ion, start, omega_axial, max_voltage=max_voltage, tilt=tilt)
                waveform = Waveform.from_voltages([0.0], voltages[None, :], start[None, None, :])
                return waveform, info
            end = trap.find_rf_null((x_end, 0.0, None)) if trap.n_rf > 0 else np.array([x_end, 0.0, ION_HEIGHT])
            waveform = voltage_objectives.transport(trap, ion, start, end, duration, n_steps, omega_axial, profile, max_voltage=max_voltage,
                                                    max_slew_rate=max_slew if max_slew > 0 else None, tilt=tilt, roughness=roughness)
            return waveform, waveform.info

        def done(output):
            waveform, info = output
            self.app.set_waveform(waveform, omega_axial)
            self.show(waveform, omega_axial)
            text = self.results.get("1.0", "end").strip()
            set_text(self.results, text + "\n" + "\n".join(info.summary()))

        self.app.run_task("Solving the objectives", work, done)

    def _filter(self):
        form = self.dac
        kind = form.get_str("filter")
        cutoff = form.get("cutoff") * 1e6
        if kind == "None":
            return None
        if cutoff <= 0:
            raise ValueError("The cutoff frequency must be positive")
        if kind == "RC":
            return hardware.Filter.rc(cutoff)
        return hardware.Filter.butterworth(cutoff, form.get_int("order"))

    def apply_hardware(self):
        self.require_trap()
        waveform = self.app.waveform
        if waveform is None or waveform.n_steps < 2:
            raise ValueError("Solve a transport first")
        form = self.dac
        dac = hardware.DAC(form.get("rate") * 1e6, form.get_int("bits"), -form.get("range"), form.get("range"))
        filter_ = self._filter()
        compensate = form.get("compensate") and filter_ is not None
        smoothing = form.get("smoothing")
        n_samples = waveform.duration * dac.sample_rate
        if n_samples > 5000:
            raise ValueError(f"The waveform takes {n_samples:.0f} DAC samples; use a shorter waveform or a slower DAC (at most 5000 samples)")

        def work():
            return hardware.through_hardware(waveform, dac, filter_, compensate, smoothing)

        def done(result):
            self.hardware_result = result
            self.hardware_source = waveform
            trap = self.app.trap
            times = result.times * 1e6
            swing = np.ptp(result.target[:, trap.n_rf:], axis=0)
            shown = trap.n_rf + np.argsort(swing)[::-1][:3]
            ax_voltages, ax_error = self.plot.axes(2, 1, sharex=True)[:, 0]
            for k, e in enumerate(shown):
                color = f"C{k}"
                ax_voltages.plot(times, result.target[:, e], color=color, linestyle=":", label=f"Electrode {e}: waveform")
                ax_voltages.step(times, result.samples[:, e], color=color, alpha=0.4, where="post", label=f"DAC")
                ax_voltages.plot(times, result.output[:, e], color=color, label="At the electrode")
            ax_voltages.set_ylabel("Voltage (V)")
            ax_voltages.legend(fontsize=6, ncols=3)
            ax_voltages.set_title(f"{filter_.name if filter_ else 'No filter'}" + (", pre-compensated" if compensate else ""))
            for e in range(trap.n_rf, trap.n_electrodes):
                ax_error.plot(times, (result.output[:, e] - result.target[:, e]) * 1e3, linewidth=0.7)
            ax_error.set_ylabel("Error at the electrodes (mV)")
            ax_error.set_xlabel("Time (µs)")
            self.plot.draw()
            set_text(self.results, "\n".join([
                f"{len(result.times)} samples at {dac.sample_rate / 1e6:g} MS/s ({result.pad_before} before and {result.pad_after} after the waveform)",
                f"Largest error at the electrodes: {result.max_error * 1e3:.3f} mV",
                f"Least significant bit: {dac.lsb * 1e3:.3f} mV",
                f"Largest DAC sample: {np.abs(result.samples).max():.3f} V",
            ]))

        self.app.run_task("Sending the waveform through the DAC and filters", work, done)

    def use_hardware_output(self):
        if self.hardware_result is None:
            raise ValueError("Apply the hardware to a waveform first")
        result = self.hardware_result
        source = self.hardware_source
        # The wells of the waveform, held at their ends over the padding
        wells = np.stack([np.interp(result.times, source.times, source.well_positions[:, 0, k]) for k in range(3)], axis=-1)[:, None, :]
        waveform = Waveform.from_voltages(result.times - result.times[0], result.output, wells)
        self.app.set_waveform(waveform, self.app.omega_axial)
        self.app.set_status("The Dynamics tab now uses the voltages at the electrodes")

    def omega_axial(self):
        return TWO_PI * self.transport_form.get("f_axial") * 1e6

    def static_well(self):
        self._solve(static=True)

    def transport(self):
        self._solve(static=False)

    def _solve(self, static):
        trap = self.require_trap()
        ion = self.app.ion
        form = self.transport_form
        x_start = form.get("x_start")
        x_end = x_start if static else form.get("x_end")
        duration = 0.0 if static else form.get("duration") * 1e-6
        n_steps = 1 if static else form.get_int("n_steps")
        omega_axial = self.omega_axial()
        max_voltage = form.get("max_voltage")
        refine = form.get("refine")

        if not static and n_steps < 2:
            raise ValueError("Transports need at least 2 steps")

        def work():
            start = trap.find_rf_null((x_start, 0.0, None)) if trap.n_rf > 0 else np.array([x_start, 0.0, ION_HEIGHT])
            end = trap.find_rf_null((x_end, 0.0, None)) if trap.n_rf > 0 else np.array([x_end, 0.0, ION_HEIGHT])
            waveform = Waveform.transport(trap, ion, start, end, duration, n_steps, omega_axial)
            refined = False
            if refine and np.abs(waveform.voltages).max() > max_voltage:
                waveform.optimize(trap, ion, omega_axial, max_voltage)
                refined = True
            return waveform, refined

        def done(result):
            waveform, refined = result
            self.app.set_waveform(waveform, omega_axial)
            self.show(waveform, omega_axial, refined)

        self.app.run_task("Solving for waveform" if not static else "Solving for static well", work, done)

    def show(self, waveform, omega_axial, refined=False):
        trap = self.app.trap
        ion = self.app.ion
        max_force, max_frequency_error = waveform.errors(trap, ion, omega_axial)
        voltages = waveform.voltages
        dc = slice(trap.n_rf, trap.n_electrodes)

        lines = [
            f"{waveform.n_steps} steps over {waveform.duration * 1e6:.2f} µs with {waveform.n_wells} well(s)",
            f"Peak DC voltage: {np.abs(voltages[:, dc]).max():.3f} V" + (" (refined with NLopt)" if refined else ""),
            f"Largest residual force: {max_force:.3e} N",
            f"Largest axial frequency error: {max_frequency_error:.3e}",
        ]
        if waveform.n_steps == 1:
            position = waveform.well_positions[0, 0]
            omega, axes = trap.normal_modes(ion, position, voltages[0])
            lines.append("Secular frequencies: " + ", ".join(f"{w / TWO_PI / 1e6:.3f}" for w in omega) + " MHz")
        set_text(self.results, "\n".join(lines))

        ax_voltages, ax_potential = self.plot.axes(2, 1)[:, 0]
        times = (waveform.times - waveform.times[0]) * 1e6
        color_map = colormaps["viridis"]
        for e in range(trap.n_rf, trap.n_electrodes):
            color = color_map((e - trap.n_rf) / max(trap.n_dc - 1, 1))
            if waveform.n_steps > 1:
                ax_voltages.plot(times, voltages[:, e], color=color, label=str(e))
            else:
                ax_voltages.bar(e, voltages[0, e], color=color)
        if waveform.n_steps > 1:
            ax_voltages.set_xlabel("Time (µs)")
            ax_voltages.legend(title="Electrode", ncols=6, fontsize=7, title_fontsize=7, loc="best")
        else:
            ax_voltages.set_xlabel("Electrode")
        ax_voltages.set_ylabel("Voltage (V)")
        ax_voltages.set_title("DC electrode voltages")

        # Potential energy along the trap axis at a few times
        wells = waveform.well_positions
        x_range = np.linspace(wells[..., 0].min() - 40, wells[..., 0].max() + 40, 400)
        height = wells[0, 0, 2]
        line = np.stack([x_range, np.zeros_like(x_range), np.full_like(x_range, height)], axis=-1)
        indices = np.unique(np.linspace(0, waveform.n_steps - 1, min(6, waveform.n_steps)).astype(int))
        for i in indices:
            U = trap.potential_energy(ion, line, voltages[i]) * MEV
            ax_potential.plot(x_range, U - U.min(), label=f"{times[i]:.1f} µs")
            for w in range(waveform.n_wells):
                ax_potential.axvline(wells[i, w, 0], color="gray", linewidth=0.5, linestyle=":")
        # A harmonic well of the target frequency rises to this energy 20 µm from its center
        ax_potential.set_ylim(0, 0.5 * ion.mass * (omega_axial * 20 * LENGTH_SCALE) ** 2 * MEV)
        ax_potential.set_xlabel("x (µm)")
        ax_potential.set_ylabel("Potential energy (meV)")
        ax_potential.set_title(f"Along the trap axis at z = {height:.2f} µm (dotted lines: wells)")
        ax_potential.legend(fontsize=8)
        self.plot.draw()


# ---------------------------------------------------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------------------------------------------------


class DynamicsTab(Tab):
    def __init__(self, app):
        super().__init__(app, "Dynamics")

        simulation = self.form("Simulation")
        simulation.entry("dt", "Timestep", 5, "ns")
        simulation.entry("hold", "Extra time after waveform", 10, "µs")
        simulation.entry("offset", "Initial axial offset", 0, "µm")
        simulation.combo("model", "RF force", ("Pseudopotential", "Full RF drive"), "Pseudopotential")
        simulation.buttons(("Simulate", self.simulate))
        self.simulation = simulation

        noise = self.form("Heating")
        noise.entry("S_V", "DC voltage noise", 1e-18, "V²/Hz")
        noise.entry("S_V_N", "RF amplitude noise", 1e-12, "V²/Hz")
        self.noise = noise

        crystal = self.form("Crystal molecular dynamics")
        crystal.entry("ions", "Ions", "40Ca+ x4")
        crystal.entry("duration", "Duration", 50, "µs")
        crystal.entry("dt", "Timestep", 10, "ns")
        crystal.check("from_crystal", "Start from the equilibrium crystal (else a cloud)", True)
        crystal.entry("cloud", "Size of the starting cloud", 5, "µm")
        crystal.entry("temperature", "Starting temperature", 0.1, "K")
        crystal.check("cooling", "Doppler cooling", True)
        crystal.entry("detuning", "Cooling detuning", -0.5, "Γ")
        crystal.entry("saturation", "Saturation parameter", 1.0)
        crystal.entry("field_noise", "Electric field noise", 0, "V²/m²/Hz")
        crystal.check("full_rf", "Full RF drive (micromotion)", False)
        crystal.check("follow", "Follow the waveform (else hold its first step)", False)
        crystal.buttons(("Simulate crystal", self.simulate_crystal), ("RF phase scan", self.rf_phase_scan))
        self.crystal_form = crystal

        results = self.form("Results")
        self.results = results.text(8)

        self.plot.message("Solve a waveform in the Waveforms tab, then simulate it")

    def simulate_crystal(self):
        trap = self.require_trap()
        if self.app.waveform is None:
            raise ValueError("Solve a static well or a transport in the Waveforms tab first, which holds the crystal")
        form = self.crystal_form
        ions = parse_ions(form.get_str("ions"))
        waveform = self.app.waveform
        duration = form.get("duration") * 1e-6
        dt = form.get("dt") * 1e-9
        full_rf = form.get("full_rf")
        if full_rf:
            dt = min(dt, TWO_PI / trap.Omega_RF / 40)
        if duration / dt > 2e5:
            raise ValueError(f"The simulation would take {duration / dt:.0f} steps; use a longer timestep or a shorter duration")
        beams = []
        if form.get("cooling"):
            species = {element: Ion(element) for element in set(ions)}
            # The cooling beam addresses the first species (the others are cooled sympathetically)
            element = ions[0]
            gamma = species[element].struct.gamma_SP
            beams.append(dynamics.CoolingBeam(detuning=form.get("detuning") * gamma, saturation=form.get("saturation"), element=element))
        follow = form.get("follow")
        center = waveform.well_positions[0, 0]
        kwargs = dict(duration=duration, dt=dt, waveform=waveform if follow else None, voltages=None if follow else waveform.voltages[0],
                      full_rf=full_rf, beams=beams, field_noise=form.get("field_noise"), temperature=form.get("temperature"), n_samples=300)
        size = form.get("cloud")
        from_crystal = form.get("from_crystal")

        def work():
            try:
                crystal = crystals.find_crystal(trap, ions, voltages=waveform.voltages[0], position=center)
            except Exception:
                crystal = None
            if from_crystal and crystal is not None:
                positions = crystal.positions
            else:
                positions = dynamics.ion_cloud(len(ions), center, size, min_distance=min(2.0, size))
            result = dynamics.simulate(trap, ions, positions, **kwargs)
            return result, crystal

        def done(output):
            result, crystal = output
            times = result.times * 1e6
            ax_x, ax_positions, ax_temperature = self.plot.axes(3, 1)[:, 0]
            for i in range(len(ions)):
                ax_x.plot(times, result.positions[:, i, 0], linewidth=0.8, label=f"{i}: {ions[i]}" if i < 8 else None)
            ax_x.set_ylabel("x (µm)")
            ax_x.set_xlabel("Time (µs)")
            ax_x.legend(fontsize=6, ncols=4)
            ax_x.set_title("Ions")
            markers = {element: m for element, m in zip(sorted(set(ions)), "osD^v")}
            for i, element in enumerate(ions):
                if not result.lost[i]:
                    ax_positions.plot(result.positions[-1, i, 0], result.positions[-1, i, 1], marker=markers[element], color="tab:blue", linestyle="none")
            if crystal is not None:
                ax_positions.plot(crystal.positions[:, 0], crystal.positions[:, 1], marker="x", color="tab:red", linestyle="none", label="Equilibrium crystal")
                ax_positions.legend(fontsize=7)
            ax_positions.set_xlabel("x (µm)")
            ax_positions.set_ylabel("y (µm)")
            ax_positions.set_title("Final positions")
            masses = np.array([Ion(element).mass for element in ions])
            kinetic = np.mean(masses[None, :, None] * result.velocities ** 2, axis=(1, 2)) / dynamics.K_B
            window = max(len(kinetic) // 30, 1)
            smooth = np.convolve(kinetic, np.ones(window) / window, mode="same")
            ax_temperature.semilogy(times, np.maximum(smooth, 1e-6))
            ax_temperature.axhline(dynamics.doppler_limit(Ion(ions[0])), color="gray", linestyle=":", label="Doppler limit")
            ax_temperature.set_ylabel("Kinetic temperature (K)")
            ax_temperature.set_xlabel("Time (µs)")
            ax_temperature.legend(fontsize=7)
            self.plot.draw()
            T = result.temperatures(0.7).mean(axis=0)
            set_text(self.results, "\n".join([
                f"{len(ions)} ions for {times[-1]:.1f} µs" + (" with the full RF drive" if full_rf else ""),
                "Kinetic temperature over the last 30%: " + ", ".join(f"{a} {t * 1e3:.3g} mK" for a, t in zip("xyz", T)),
                f"Ions lost: {int(result.lost.sum())}",
                f"Photons scattered per ion: {result.scattered_photons.mean():.0f}",
            ] + (crystal.summary()[1:] if crystal is not None else [])))

        self.app.run_task("Simulating the crystal", work, done)

    def rf_phase_scan(self):
        trap = self.require_trap()
        waveform = self.app.waveform
        if waveform is None or waveform.n_steps < 2:
            raise ValueError("Solve a transport in the Waveforms tab first")
        ion = self.app.ion
        omega_axial = self.app.omega_axial
        if waveform.duration * trap.Omega_RF / TWO_PI * 40 > 1e5:
            raise ValueError("The transport is too long to follow the full RF drive; use a shorter transport")

        def work():
            return dynamics.rf_phase_scan(trap, ion, waveform, omega_axial=omega_axial)

        def done(output):
            phases, excitations = output
            ax = self.plot.axes()[0, 0]
            ax.plot(np.degrees(phases), excitations, marker="o")
            ax.set_xlabel("Phase of the RF drive at the start (°)")
            ax.set_ylabel("Final axial excitation (quanta)")
            ax.set_title("Transport with the full RF drive at each phase")
            self.plot.draw()
            set_text(self.results, f"Excitation from {excitations.min():.4f} to {excitations.max():.4f} quanta over the RF phase; a large spread means the pseudopotential approximation misses part of the motion")

        self.app.run_task("Scanning the RF phase", work, done)

    def simulate(self):
        trap = self.require_trap()
        if self.app.waveform is None:
            raise ValueError("Solve a waveform in the Waveforms tab first")
        ion = self.app.ion
        waveform = self.app.waveform
        omega_axial = self.app.omega_axial
        dt = self.simulation.get("dt") * 1e-9
        hold = self.simulation.get("hold") * 1e-6
        offset = self.simulation.get("offset")
        full_rf = self.simulation.get_str("model") == "Full RF drive"
        S_V = self.noise.get("S_V")
        S_V_N = self.noise.get("S_V_N")

        if full_rf:
            dt = min(dt, TWO_PI / trap.Omega_RF / 50)

        duration = waveform.duration + hold
        n_it = int(np.ceil(duration / dt))
        if n_it > 2_000_000:
            raise ValueError(f"The simulation would take {n_it} steps; increase the timestep or shorten the waveform")

        def work():
            start = waveform.well_positions[0, 0] + np.array([offset, 0.0, 0.0])
            trajectory = lib.simulate_trajectory(trap, ion, waveform, duration, dt, start, full_rf=full_rf)
            # The voltages are interpolated linearly in time, and so (nearly) are the wells
            wells = np.column_stack([np.interp(trajectory.times, waveform.times, waveform.well_positions[:, 0, k]) for k in range(3)])
            final_well = waveform.well_positions[-1, 0]
            quanta = lib.motional_quanta(ion, omega_axial, trajectory.positions[-1], trajectory.velocities[-1], final_well)
            gamma_dc = trap.voltage_noise_heating_rate(ion, final_well, omega_axial, (1, 0, 0), S_V)
            omega, axes = trap.normal_modes(ion, final_well, waveform.voltages[-1])
            radial = int(np.argmax(np.abs(axes[:, 2])))
            gamma_rf = trap.rf_noise_heating_rate(ion, final_well + np.array([0, 0, 0.1]), abs(omega[radial]), (0, 0, 1), S_V_N)
            return trajectory, wells, quanta, gamma_dc, gamma_rf

        def done(result):
            trajectory, wells, quanta, gamma_dc, gamma_rf = result
            self.app.trajectory = trajectory
            times = (trajectory.times - trajectory.times[0]) * 1e6
            lag = trajectory.positions[:, 0] - wells[:, 0]

            set_text(self.results, "\n".join([
                f"{len(times) - 1} steps of {dt * 1e9:.3g} ns ({'full RF drive' if full_rf else 'pseudopotential'})",
                f"Final axial excitation: {quanta:.4f} quanta",
                f"Largest axial lag behind the well: {np.abs(lag).max() * 1e3:.2f} nm",
                f"Largest radial excursion: {np.abs(trajectory.positions[:, 1:] - wells[:, 1:]).max() * 1e3:.2f} nm",
                f"Axial heating from DC voltage noise: {gamma_dc:.3g} quanta/s",
                f"Radial heating from RF noise 100 nm off the null: {gamma_rf:.3g} quanta/s",
            ]))

            ax_x, ax_lag, ax_radial = self.plot.axes(3, 1, sharex=True)[:, 0]
            ax_x.plot(times, trajectory.positions[:, 0], label="Ion")
            ax_x.plot(times, wells[:, 0], linestyle="--", label="Well")
            ax_x.set_ylabel("x (µm)")
            ax_x.legend(fontsize=8)
            ax_x.set_title("Ion transport")
            ax_lag.plot(times, lag * 1e3)
            ax_lag.set_ylabel("x - x_well (nm)")
            ax_radial.plot(times, (trajectory.positions[:, 1] - wells[:, 1]) * 1e3, label="y")
            ax_radial.plot(times, (trajectory.positions[:, 2] - wells[:, 2]) * 1e3, label="z")
            ax_radial.set_ylabel("Radial offset (nm)")
            ax_radial.set_xlabel("Time (µs)")
            ax_radial.legend(fontsize=8)
            self.plot.draw()

        self.app.run_task(f"Simulating {n_it} steps", work, done)



# ---------------------------------------------------------------------------------------------------------------------
# Crystals
# ---------------------------------------------------------------------------------------------------------------------


class CrystalsTab(Tab):
    POTENTIALS = ("Static well solved here", "Static well (Waveforms tab)", "Model linear trap")
    BEAMS = ("Along x", "Along (1, 0, 1)", "Raman difference along x", "Qubit grating of the trap")

    def __init__(self, app):
        super().__init__(app, "Crystals")

        form = self.form("Coulomb crystal")
        form.entry("ions", "Ions", "40Ca+ x4")
        form.combo("potential", "Potential", self.POTENTIALS, self.POTENTIALS[0])
        form.entry("x", "Position along the trap", 0, "µm")
        form.entry("f_axial", "Axial frequency", 1.0, "MHz")
        form.entry("f_radial", "Radial frequencies (model)", "4.0, 4.4", "MHz")
        form.buttons(("Find crystal", self.find))
        self.form_ = form

        beam = self.form("Light")
        beam.combo("beam", "Wavevector", self.BEAMS, self.BEAMS[0])
        self.beam = beam

        results = self.form("Modes")
        self.results = results.text(14)
        self.plot.message("Find the equilibrium and normal modes of a crystal of ions")

    def find(self):
        form = self.form_
        ions = parse_ions(form.get_str("ions"))
        potential_choice = form.get_str("potential")
        omega = TWO_PI * form.get("f_axial") * 1e6
        x = form.get("x")
        app = self.app
        reference = Ion(ions[0])

        if potential_choice == "Model linear trap":
            radial = [float(v) for v in form.get_str("f_radial").split(",")]
            if len(radial) != 2:
                raise ValueError("Give two radial frequencies, e.g. 4.0, 4.4")
            potential = crystals.HarmonicPotential(reference, (omega, TWO_PI * radial[0] * 1e6, TWO_PI * radial[1] * 1e6))
            setup = None
        else:
            trap = self.require_trap()
            if potential_choice.startswith("Static well (W"):
                if app.waveform is None:
                    raise ValueError("Solve a static well in the Waveforms tab first")
                setup = (trap, app.waveform.voltages[0], app.waveform.well_positions[0, 0])
            else:
                setup = (trap, None, x)
            potential = None
        beam_choice = self.beam.get_str("beam")

        def work():
            if setup is None:
                crystal = crystals.find_crystal(potential, ions)
            else:
                trap, voltages, where = setup
                if voltages is None:
                    position = trap.find_rf_null((where, 0.0, None)) if trap.n_rf > 0 else np.array([where, 0.0, ION_HEIGHT])
                    voltages, _ = voltage_objectives.static_well(trap, reference, position, omega)
                else:
                    position = where
                crystal = crystals.find_crystal(trap, ions, voltages=voltages, position=position)
            k = beam_wavevector(app, beam_choice, crystal.positions.mean(axis=0), Ion(ions[0]).qubit_wavelength)
            return crystal, k

        def done(output):
            crystal, k = output
            app.crystal = crystal
            app.crystal_k = k
            eta = crystal.lamb_dicke(k)
            axes_of_modes = crystal.mode_axes()
            ax_positions, ax_spectrum, ax_participation, ax_eta = self.plot.axes(2, 2).ravel()
            species = sorted({ion.element for ion in crystal.ions})
            for s_index, element in enumerate(species):
                members = [i for i, ion in enumerate(crystal.ions) if ion.element == element]
                ax_positions.plot(crystal.positions[members, 0], crystal.positions[members, 1], marker="os^D"[s_index % 4], linestyle="none", label=element)
            ax_positions.set_xlabel("x (µm)")
            ax_positions.set_ylabel("y (µm)")
            ax_positions.set_aspect("equal", adjustable="datalim")
            ax_positions.legend(fontsize=7)
            ax_positions.set_title("Equilibrium" + ("" if crystal.stable else " (NOT stable)"))
            axis_colors = {"x": "tab:blue", "y": "tab:orange", "z": "tab:green"}
            for m, (w, a) in enumerate(zip(crystal.frequencies, axes_of_modes)):
                ax_spectrum.vlines(w / TWO_PI / 1e6, 0, 1, color=axis_colors[a])
            for a, color in axis_colors.items():
                ax_spectrum.plot([], [], color=color, label=f"Mostly along {a}")
            ax_spectrum.set_xlabel("Mode frequency (MHz)")
            ax_spectrum.set_yticks([])
            ax_spectrum.legend(fontsize=7)
            ax_spectrum.set_title("Normal modes")
            image = ax_participation.imshow(crystal.participation().T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            self.plot.figure.colorbar(image, ax=ax_participation, label="Participation")
            ax_participation.set_xlabel("Mode (by frequency)")
            ax_participation.set_ylabel("Ion")
            limit = max(np.abs(eta).max(), 1e-9)
            image = ax_eta.imshow(eta, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
            self.plot.figure.colorbar(image, ax=ax_eta, label="Lamb-Dicke parameter")
            ax_eta.set_xlabel("Mode (by frequency)")
            ax_eta.set_ylabel("Ion")
            ax_eta.set_title(f"Light along ({k[0] / np.linalg.norm(k):+.2f}, {k[1] / np.linalg.norm(k):+.2f}, {k[2] / np.linalg.norm(k):+.2f})")
            self.plot.draw()
            lines = crystal.summary()
            strongest = np.argsort(-np.abs(eta).max(axis=0))[:4]
            lines.append("Modes coupled most strongly to the light:")
            lines += [f"  {crystal.frequencies[m] / TWO_PI / 1e6:.4f} MHz ({axes_of_modes[m]}): η = " + ", ".join(f"{v:+.3f}" for v in eta[:, m]) for m in strongest]
            lines.append("The Quantum tab can design gates on these modes")
            set_text(self.results, "\n".join(lines))

        self.app.run_task("Finding the crystal", work, done)

# ---------------------------------------------------------------------------------------------------------------------
# Photonics
# ---------------------------------------------------------------------------------------------------------------------


class PhotonicsTab(Tab):
    COMPONENTS = ("Grating beam at the ion", "Straight waveguide", "Waveguide bend", "Grating coupler", "Directional coupler", "Splitter tree", "Waveguide routes")
    # Kinds of grating: (apodized, chirped)
    GRATING_KINDS = ("Uniform", "Apodized (fill factor changes)", "Chirped (focuses along the grating)", "Apodized and chirped")

    def __init__(self, app):
        super().__init__(app, "Photonics")

        selector = self.form("Component")
        self.component = selector.combo("component", "Component", self.COMPONENTS, self.COMPONENTS[0])
        self.component.trace_add("write", lambda *args: self.show_component())

        waveguide = self.form("Waveguide")
        waveguide.entry("wavelength", "Wavelength", 0.729, "µm")
        waveguide.entry("width", "Width", 0.6, "µm")
        waveguide.entry("thickness", "Thickness", 0.15, "µm")
        waveguide.entry("n_core", "Core index", 2.0)
        waveguide.entry("n_cladding", "Cladding index", 1.455)
        self.waveguide = waveguide

        grating = self.form("Grating coupler")
        grating.entry("x", "Exit point x", -41.25, "µm")
        grating.entry("y", "Exit point y", 0, "µm")
        grating.entry("target_x", "Target x", 0, "µm")
        grating.entry("target_y", "Target y", 0, "µm")
        grating.entry("target_z", "Target z", ION_HEIGHT, "µm")
        grating.entry("fill_factor", "Fill factor", 0.5)
        grating.entry("etch_depth", "Etch depth", 0.15, "µm")
        grating.entry("length", "Length", 20, "µm")
        grating.entry("grating_width", "Width", 8, "µm")
        grating.entry("power", "Power", 1, "mW")
        grating.entry("efficiency", "Efficiency", 0.3)
        grating.check("focus", "Focus on the target", True)
        self.grating = grating

        substrate = self.form("Substrate")
        substrate.entry("n_box", "Buried oxide index", 1.45)
        substrate.entry("n_substrate", "Substrate index", 3.7)
        substrate.entry("k_substrate", "Substrate extinction", 0.01)
        substrate.entry("box_max", "Largest oxide thickness", 3, "µm")
        self.substrate = substrate

        actions = self.form("Actions")
        actions.buttons(("Design and evaluate", self.evaluate), ("Use trap grating", self.load_trap_grating))

        results = self.form("Results")
        self.results = results.text(10)
        self.beam_forms = (waveguide, grating, substrate, actions, results)

        # Forms of the component designer, i.e. the chip stack shared by every component and one form per component
        stack = self.form("Chip stack")
        stack.entry("wavelength", "Wavelength", 0.729, "µm")
        stack.entry("n_core", "Core index", 2.0)
        stack.entry("n_cladding", "Cladding index", 1.455)
        stack.entry("n_substrate", "Substrate index", 3.71)
        stack.entry("k_substrate", "Substrate extinction", 0.008)
        stack.entry("box_thickness", "Buried oxide thickness", 3.0, "µm")
        stack.entry("top_cladding", "Top cladding thickness", 2.0, "µm")
        stack.entry("core_loss", "Core absorption", 0.5, "dB/cm")
        stack.entry("sidewall_roughness", "Sidewall roughness", 3.0, "nm")
        stack.entry("surface_roughness", "Surface roughness", 0.3, "nm")
        stack.entry("correlation_length", "Correlation length", 50, "nm")
        stack.combo("polarization", "Polarization", ("TE", "TM"), "TE")
        self.stack = stack

        straight = self.form("Straight waveguide")
        straight.entry("width", "Width", 0.6, "µm")
        straight.entry("thickness", "Thickness", 0.15, "µm")
        straight.entry("width_min", "Smallest width", 0.2, "µm")
        straight.entry("width_max", "Largest width", 3.0, "µm")
        straight.entry("thickness_min", "Smallest thickness", 0.05, "µm")
        straight.entry("thickness_max", "Largest thickness", 0.4, "µm")
        straight.entry("min_confinement", "Smallest confinement in the core", 0.3)
        straight.check("single_mode", "Keep the waveguide single-mode", True)
        self.straight = straight

        bend = self.form("Waveguide bend")
        bend.entry("radius", "Radius", 20, "µm")
        bend.entry("angle", "Angle", 90, "°")
        bend.entry("bend_width", "Width in the bend", 0.6, "µm")
        bend.entry("euler_fraction", "Euler fraction (taper)", 0.0)
        bend.entry("offset", "Offset of straight sections", 0.0, "µm")
        bend.entry("taper_length", "Straight tapers to the bend width", 0.0, "µm")
        bend.combo("taper_profile", "Taper profile", components.TAPER_PROFILES, "linear")
        bend.entry("max_radius", "Largest footprint", 50, "µm")
        bend.entry("taper_max", "Longest taper", 30, "µm")
        bend.check("optimize_taper", "Optimize the taper length too", False)
        self.bend = bend

        coupler = self.form("Grating coupler")
        coupler.entry("period", "Period", 0.78, "µm")
        coupler.entry("tooth_width", "Tooth width", 0.39, "µm")
        coupler.entry("n_teeth", "Teeth", 20)
        coupler.entry("etch_depth", "Etch depth", 0.15, "µm")
        coupler.entry("angular_width", "Angular width", 15, "°")
        coupler.entry("taper_length", "Taper length", 15, "µm")
        coupler.entry("target_height", "Ion height", ION_HEIGHT, "µm")
        coupler.entry("target_distance", "Ion distance along the beam", 41.25, "µm")
        coupler.entry("power", "Power", 1, "mW")
        coupler.combo("kind", "Teeth", self.GRATING_KINDS, self.GRATING_KINDS[0])
        coupler.entry("fill_start", "Fill factor of the first tooth", 0.2)
        self.coupler = coupler

        component_actions = self.form("Actions")
        component_actions.buttons(("Evaluate", lambda: self.run_component(False)), ("Optimize", lambda: self.run_component(True)))
        component_results = self.form("Results")
        self.component_results = component_results.text(12)

        directional = self.form("Directional coupler")
        directional.entry("gap", "Gap", 0.3, "µm")
        directional.entry("gap_min", "Smallest gap of the scan", 0.15, "µm")
        directional.entry("gap_max", "Largest gap of the scan", 0.8, "µm")
        directional.entry("ratio", "Fraction sent across", 0.5)
        self.directional = directional

        tree = self.form("Splitter tree")
        tree.entry("weights", "Weights of the outputs (or: gate zones)", "gate zones")
        tree.combo("kind", "Splitters", ("coupler", "y"), "coupler")
        tree.entry("excess_loss", "Excess loss of every splitter", 0.1, "dB")
        self.tree_form = tree

        routes = self.form("Waveguide routes")
        routes.entry("radius", "Bend radius", 20, "µm")
        routes.entry("margin", "Edge of the chip beyond the electrodes", 200, "µm")
        routes.entry("beams", "Beams (blank for every beam)", "")
        self.routes_form = routes

        circuit_actions = self.form("Actions")
        circuit_actions.buttons(("Evaluate", self.run_circuit_component))
        circuit_results = self.form("Results")
        self.circuit_results = circuit_results.text(12)
        self.component_forms = {
            "Straight waveguide": (stack, straight, component_actions, component_results),
            "Waveguide bend": (stack, straight, bend, component_actions, component_results),
            "Grating coupler": (stack, straight, coupler, component_actions, component_results),
            "Directional coupler": (stack, straight, directional, circuit_actions, circuit_results),
            "Splitter tree": (stack, straight, directional, tree, circuit_actions, circuit_results),
            "Waveguide routes": (stack, straight, routes, circuit_actions, circuit_results),
        }
        self.show_component()

        self.plot.message("Design a grating coupler which addresses the ion")

    def show_component(self):
        component = self.component.get()
        forms = self.beam_forms if component == self.COMPONENTS[0] else self.component_forms[component]
        every = set(self.beam_forms) | {form for group in self.component_forms.values() for form in group}
        for form in every:
            form.pack_forget()
        for form in forms:
            form.pack(fill="x", pady=(0, 6), padx=(0, 4))
        messages = {
            "Directional coupler": "Find the supermodes and coupling length of two waveguides side by side",
            "Splitter tree": "Share one input among several gratings (e.g. one per gate zone) with a tree of splitters",
            "Waveguide routes": "Route a waveguide from the edge of the chip to every grating of the trap",
            "Straight waveguide": "Evaluate or optimize the width and thickness of a straight waveguide",
            "Waveguide bend": "Evaluate or optimize the radius, width, Euler taper, straight tapers, and offset of a bend",
            "Grating coupler": "Evaluate or optimize the teeth of a uniform or non-uniform grating coupler which focuses on the ion",
        }
        self.plot.message(messages.get(component, "Design a grating coupler which addresses the ion"))

    def run_circuit_component(self):
        component = self.component.get()
        stack = self.chip_stack()
        width, thickness = self.straight.get("width"), self.straight.get("thickness")
        d = self.directional
        gap, ratio = d.get("gap"), d.get("ratio")

        if component == "Directional coupler":
            gaps = np.linspace(d.get("gap_min"), d.get("gap_max"), 8)

            def work():
                return [photonic_circuits.analyze_coupler(stack, width, thickness, g) for g in gaps], photonic_circuits.analyze_coupler(stack, width, thickness, gap)

            def done(output):
                scan, chosen = output
                ax_length, ax_power = self.plot.axes(1, 2)[0]
                ax_length.semilogy(gaps, [c.coupling_length for c in scan], marker="o")
                ax_length.set_xlabel("Gap (µm)")
                ax_length.set_ylabel("Coupling length (µm)")
                ax_length.set_title("Length over which all of the light crosses")
                lengths = np.linspace(0, 2 * chosen.coupling_length, 300)
                ax_power.plot(lengths, [chosen.cross_fraction(L) for L in lengths], label="Across")
                ax_power.plot(lengths, [1 - chosen.cross_fraction(L) for L in lengths], label="Through")
                ax_power.axvline(chosen.length_for(ratio), color="gray", linestyle=":", label=f"{ratio:g} across")
                ax_power.set_xlabel("Length of the coupler (µm)")
                ax_power.set_ylabel("Fraction of the power")
                ax_power.legend(fontsize=8)
                ax_power.set_title(f"Gap of {gap:g} µm")
                self.plot.draw()
                set_text(self.circuit_results, "\n".join([
                    f"Even and odd supermodes: n = {chosen.n_even:.5f} and {chosen.n_odd:.5f}",
                    f"Coupling length: {chosen.coupling_length:.2f} µm",
                    f"Length which sends {ratio:g} of the power across: {chosen.length_for(ratio):.2f} µm",
                ]))

            self.app.run_task("Finding the supermodes of the coupler", work, done)
            return

        if component == "Splitter tree":
            text = self.tree_form.get_str("weights").strip()
            if text.lower().startswith("gate"):
                trap = self.app.trap
                n = sum("gate" in zone.types for zone in trap.zones()) if trap is not None else 0
                weights = [1.0] * max(n, 2)
            else:
                weights = [float(v) for v in text.split(",") if v.strip()]
            if len(weights) < 2 or min(weights) <= 0:
                raise ValueError("Give at least two positive weights")
            kind = self.tree_form.get_str("kind")
            loss = self.tree_form.get("excess_loss")

            def work():
                coupler = photonic_circuits.analyze_coupler(stack, width, thickness, gap) if kind == "coupler" else None
                return photonic_circuits.splitter_tree(weights, kind, loss, coupler)

            def done(tree):
                ax = self.plot.axes()[0, 0]
                index = np.arange(len(weights))
                ax.bar(index - 0.2, tree.weights, width=0.4, label="Target")
                ax.bar(index + 0.2, tree.outputs, width=0.4, label="Through the tree")
                ax.set_xticks(index, [f"Output {k + 1}" for k in index])
                ax.set_ylabel("Fraction of the input power")
                ax.legend(fontsize=8)
                ax.set_title(f"Tree of {len(tree.splits)} {'directional couplers' if kind == 'coupler' else 'Y branches'}")
                self.plot.draw()
                lines = [f"Total transmission: {tree.outputs.sum():.4f}"]
                for name, fraction, length in tree.splits:
                    lines.append(f"  {name}: {fraction:.3f} across" + (f", {length:.2f} µm long" if length is not None else ""))
                set_text(self.circuit_results, "\n".join(lines))

            self.app.run_task("Designing the splitter tree", work, done)
            return

        trap = self.app.trap
        if trap is None or not trap.grating_couplers():
            raise ValueError("Generate a trap with grating couplers first")
        r = self.routes_form
        radius, margin = r.get("radius"), r.get("margin")
        beams = [b.strip() for b in r.get_str("beams").split(",") if b.strip()] or None

        def work():
            routes, crossings = photonic_circuits.route_trap(trap, margin, radius, beams)
            losses = [route.loss(stack, width, thickness, radius) for route in routes]
            return routes, crossings, losses

        def done(output):
            routes, crossings, losses = output
            ax = self.plot.axes()[0, 0]
            draw_electrodes(ax, trap, alpha=0.15)
            gratings = trap.grating_couplers()
            for route, loss in zip(routes, losses):
                grating = gratings[route.grating]
                color = BEAM_COLORS.get(grating.beam, "tab:gray")
                ax.plot(route.points[:, 0], route.points[:, 1], color=color, linewidth=1.2)
                ax.plot(*route.points[-1], marker="s", color=color, markersize=4)
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("y (µm)")
            ax.set_title(f"{len(routes)} routes from the edge of the chip, {crossings} crossings")
            ax.autoscale_view()
            self.plot.draw()
            lines = [f"{len(routes)} routes with {crossings} crossings"]
            for route, loss in zip(routes, losses):
                grating = gratings[route.grating]
                lines.append(f"  grating {route.grating} ({grating.beam or 'beam'}): {route.length:.0f} µm, bends " + ", ".join(f"{b:.0f}°" for b in route.bends) + f", {loss:.3f} dB")
            set_text(self.circuit_results, "\n".join(lines))

        self.app.run_task("Routing waveguides to the gratings", work, done)

    def chip_stack(self):
        s = self.stack
        return components.Stack(
            wavelength=s.get("wavelength"), n_core=s.get("n_core"), n_cladding=s.get("n_cladding"),
            n_substrate=complex(s.get("n_substrate"), s.get("k_substrate")), box_thickness=s.get("box_thickness"),
            top_cladding=s.get("top_cladding"), core_loss=s.get("core_loss"), sidewall_roughness=s.get("sidewall_roughness") * 1e-3,
            surface_roughness=s.get("surface_roughness") * 1e-3, correlation_length=s.get("correlation_length") * 1e-3,
            polarization=s.get_str("polarization"),
        )

    def grating_design(self, thickness=None):
        """Grating design of the Grating coupler form."""
        c = self.coupler
        period = c.get("period")
        kind = self.GRATING_KINDS.index(c.get_str("kind")) if c.get_str("kind") in self.GRATING_KINDS else 0
        return components.GratingDesign(
            period=period, fill_factor=c.get("tooth_width") / period, n_teeth=c.get_int("n_teeth"), etch_depth=c.get("etch_depth"),
            angular_width=c.get("angular_width"), taper_length=c.get("taper_length"), thickness=self.straight.get("thickness") if thickness is None else thickness,
            target_height=c.get("target_height"), target_distance=c.get("target_distance"), power=c.get("power") * 1e-3,
            fill_start=c.get("fill_start") if kind in (1, 3) else None, chirp=kind in (2, 3),
        )

    def bend_design(self):
        """Parameters of the Waveguide bend form, as keyword arguments of components.analyze_bend."""
        b = self.bend
        return dict(radius=b.get("radius"), angle=b.get("angle"), bend_width=b.get("bend_width"), euler_fraction=b.get("euler_fraction"), offset=b.get("offset"),
                    taper_length=b.get("taper_length"), taper_profile=b.get_str("taper_profile"))

    def run_component(self, optimize):
        component = self.component.get()
        stack = self.chip_stack()
        st = self.straight
        width, thickness = st.get("width"), st.get("thickness")
        if component == "Straight waveguide":
            bounds = ((st.get("width_min"), st.get("width_max")), (st.get("thickness_min"), st.get("thickness_max")))
            single_mode = st.get("single_mode")
            min_confinement = st.get("min_confinement")

            def work():
                if optimize:
                    result = components.optimize_waveguide(stack, width, thickness, *bounds, single_mode=single_mode, min_confinement=min_confinement)
                else:
                    result = components.analyze_waveguide(stack, width, thickness)
                if result is None:
                    raise ValueError("The waveguide does not guide a mode at this wavelength")
                widths = np.linspace(bounds[0][0], bounds[0][1], 40)
                sweep = [components.analyze_waveguide(stack, w, result.thickness) for w in widths]
                return result, widths, sweep

            self.app.run_task("Optimizing the waveguide" if optimize else "Evaluating the waveguide", work, self.show_waveguide)
        elif component == "Waveguide bend":
            b = self.bend
            values = self.bend_design()
            max_radius = b.get("max_radius")
            taper_bounds = (0.0, b.get("taper_max")) if b.get("optimize_taper") else None

            def work():
                if optimize:
                    result = components.optimize_bend(stack, width, thickness, values["radius"], values["angle"], max_radius, taper_length=values["taper_length"],
                                                      taper_profile=values["taper_profile"], taper_bounds=taper_bounds)
                else:
                    result = components.analyze_bend(stack, width, thickness, **values)
                if result is None:
                    raise ValueError("The waveguide does not guide a mode at this wavelength")
                radii = np.geomspace(max(2.0, 0.2 * result.radius), 5 * result.radius, 14)
                sweep = [components.analyze_bend(stack, width, thickness, r, result.angle, result.bend_width, result.euler_fraction, result.offset,
                                                 taper_length=result.taper_length, taper_profile=result.taper_profile, n_taper_steps=8) for r in radii]
                return result, radii, sweep

            self.app.run_task("Optimizing the bend" if optimize else "Evaluating the bend", work, self.show_bend)
        else:
            if stack.polarization != "TE":
                raise ValueError("The grating model only covers TE polarization")
            design = self.grating_design(thickness)

            def work():
                result = components.optimize_grating(stack, design) if optimize else components.analyze_grating(stack, design)
                if result is None:
                    raise ValueError("The grating region does not guide a mode at this wavelength")
                d = result.design
                layout = components.tooth_layout(stack, d)
                length = layout.length
                x = np.linspace(-d.target_distance - 0.5 * length - 10, 25, 110)
                z = np.linspace(1.0, 1.5 * d.target_height, 75)
                X, Z = np.meshgrid(x, z)
                side = np.abs(components.grating_field(stack, d, np.stack([X, np.zeros_like(X), Z], axis=-1), layout)) ** 2 * 1e12
                u = np.linspace(-20, 20, 81)
                spot_x = np.abs(components.grating_field(stack, d, np.stack([u, 0 * u, np.full_like(u, d.target_height)], axis=-1), layout)) ** 2 * 1e12
                spot_y = np.abs(components.grating_field(stack, d, np.stack([0 * u, u, np.full_like(u, d.target_height)], axis=-1), layout)) ** 2 * 1e12
                return result, X, Z, side, u, spot_x, spot_y, layout

            self.app.run_task("Optimizing the grating" if optimize else "Evaluating the grating", work, self.show_grating)

    def show_waveguide(self, r):
        result, widths, sweep = r
        self.straight.set("width", f"{result.width:.4g}")
        self.straight.set("thickness", f"{result.thickness:.4g}")
        set_text(self.component_results, "\n".join(
            [f"Width {result.width * 1e3:.0f} nm, thickness {result.thickness * 1e3:.0f} nm",
             f"Effective index {result.n_eff.real:.5f}, {result.n_modes} guided mode{'s' if result.n_modes != 1 else ''} of this polarization",
             f"Confinement in the core: {result.confinement:.3f}",
             f"Propagation loss: {result.loss:.3f} dB/cm"]
            + [f"  {name}: {value:.3f} dB/cm" for name, value in result.losses.items()]))

        (ax_mode, ax_losses), (ax_sweep, ax_modes) = self.plot.axes(2, 2)
        intensity = result.mode_intensity()
        ax_mode.pcolormesh(result.x, result.y, intensity / intensity.max(), shading="auto", cmap="inferno")
        ax_mode.add_patch(Rectangle((-0.5 * result.width, -result.thickness), result.width, result.thickness, fill=False, edgecolor="cyan"))
        ax_mode.set_xlim(-1.5 * result.width - 0.5, 1.5 * result.width + 0.5)
        ax_mode.set_ylim(-result.thickness - 1.0, 1.0)
        ax_mode.set_aspect("equal")
        ax_mode.set_xlabel("x (µm)")
        ax_mode.set_ylabel("y (µm)")
        ax_mode.set_title("Mode intensity")

        names = list(result.losses)
        ax_losses.barh(names, [result.losses[name] for name in names], color="tab:blue")
        ax_losses.set_xlabel("Loss (dB/cm)")
        ax_losses.set_title(f"Total {result.loss:.3f} dB/cm")

        guided = [(w, s) for w, s in zip(widths, sweep) if s is not None]
        ax_sweep.plot([w for w, _ in guided], [s.loss for _, s in guided], label="Total")
        for name in names:
            ax_sweep.plot([w for w, _ in guided], [s.losses[name] for _, s in guided], linestyle=":", label=name)
        ax_sweep.axvline(result.width, color="gray", linestyle="--")
        ax_sweep.set_yscale("log")
        ax_sweep.set_xlabel("Width (µm)")
        ax_sweep.set_ylabel("Loss (dB/cm)")
        ax_sweep.legend(fontsize=7)
        ax_sweep.set_title(f"Thickness {result.thickness * 1e3:.0f} nm")

        ax_modes.step([w for w, _ in guided], [s.n_modes for _, s in guided], where="mid")
        ax_modes.axvline(result.width, color="gray", linestyle="--")
        ax_modes.set_xlabel("Width (µm)")
        ax_modes.set_ylabel("Guided modes")
        ax_modes.set_title("Single-mode below the first step")
        self.plot.draw()

    def show_bend(self, r):
        result, radii, sweep = r
        for key, value in (("radius", result.radius), ("bend_width", result.bend_width), ("euler_fraction", result.euler_fraction), ("offset", result.offset), ("taper_length", result.taper_length)):
            self.bend.set(key, f"{value:.4g}")
        tapers = f", {result.taper_length:.1f} µm {result.taper_profile} tapers" if result.taper_length > 0 else ""
        set_text(self.component_results, "\n".join([
            f"Radius {result.radius:.2f} µm, width {result.bend_width * 1e3:.0f} nm in the bend, Euler fraction {result.euler_fraction:.2f}, offset {result.offset * 1e3:.0f} nm{tapers}",
            f"Loss of the {result.angle:g}° bend: {result.loss:.4f} dB",
            f"  radiation: {result.radiation_loss:.4f} dB",
            f"  transitions between sections: {result.transition_loss:.4f} dB",
            f"  propagation (scattering and absorption): {result.propagation_loss:.4f} dB",
            f"  tapers (scattering and propagation): {result.taper_loss:.4f} dB",
            f"Loss per length in the bend: {result.loss_per_cm:.3f} dB/cm",
            f"Path length {result.length:.1f} µm, footprint {result.effective_radius:.1f} µm",
        ]))

        (ax_top, ax_sweep), (ax_profiles, ax_curvature) = self.plot.axes(2, 2)
        path = result.path
        s_path = path[:, 3]
        # The intensity of the local mode of each section, drawn across the path
        start = 0.0
        for (kappa, length), (x, mode, local_width) in result.sections:
            s = np.linspace(start, start + length, 6)
            keep = np.abs(x) < local_width + 1.0
            u = x[keep]
            heading = np.interp(s, s_path, path[:, 2])
            cx = np.interp(s, s_path, path[:, 0])
            cy = np.interp(s, s_path, path[:, 1])
            # Positive x is towards the outside of the bend
            X = cx[:, None] + np.sin(heading)[:, None] * u[None]
            Y = cy[:, None] - np.cos(heading)[:, None] * u[None]
            intensity = np.abs(mode[keep]) ** 2
            ax_top.pcolormesh(X, Y, np.tile(intensity / intensity.max(), (len(s), 1)), shading="gouraud", cmap="inferno")
            start += length
        ax_top.plot(path[:, 0], path[:, 1], color="cyan", linewidth=0.5)
        ax_top.set_aspect("equal")
        ax_top.set_xlabel("x (µm)")
        ax_top.set_ylabel("y (µm)")
        ax_top.set_title("Mode along the bend")

        valid = [(radius, s) for radius, s in zip(radii, sweep) if s is not None]
        ax_sweep.plot([radius for radius, _ in valid], [s.loss for _, s in valid], marker="o", markersize=3, label="Total")
        ax_sweep.plot([radius for radius, _ in valid], [max(s.radiation_loss, 1e-6) for _, s in valid], linestyle=":", label="Radiation")
        ax_sweep.plot([radius for radius, _ in valid], [max(s.transition_loss, 1e-6) for _, s in valid], linestyle=":", label="Transitions")
        ax_sweep.plot([radius for radius, _ in valid], [max(s.propagation_loss, 1e-6) for _, s in valid], linestyle=":", label="Propagation")
        ax_sweep.axvline(result.radius, color="gray", linestyle="--")
        ax_sweep.set_xscale("log")
        ax_sweep.set_yscale("log")
        ax_sweep.set_xlabel("Radius (µm)")
        ax_sweep.set_ylabel(f"Loss per {result.angle:g}° bend (dB)")
        ax_sweep.legend(fontsize=7)

        for (kappa, length), (x, mode, local_width) in result.sections[len(result.sections) // 2:len(result.sections) // 2 + 1]:
            ax_profiles.plot(x, np.abs(mode) ** 2 / np.max(np.abs(mode) ** 2), label=f"Bend (R = {1 / kappa:.1f} µm)")
        straight = components.analyze_waveguide(self.chip_stack(), result.width, self.straight.get("thickness"))
        if straight is not None:
            ax_profiles.plot(straight.x + result.offset, np.abs(straight.lateral) ** 2 / np.max(np.abs(straight.lateral) ** 2), linestyle="--", label="Straight")
        ax_profiles.set_xlim(-2, 3)
        ax_profiles.set_xlabel("Distance towards the outside of the bend (µm)")
        ax_profiles.set_ylabel("Intensity")
        ax_profiles.legend(fontsize=7)
        ax_profiles.set_title("Mode profiles")

        ax_curvature.plot(path[:, 3], path[:, 4], drawstyle="steps-mid", color="tab:blue")
        ax_curvature.set_xlabel("Distance along the bend (µm)")
        ax_curvature.set_ylabel("Curvature (1/µm)", color="tab:blue")
        # Width of every section, including the straight tapers
        ends = np.cumsum([length for (_, length), _ in result.sections])
        widths = [local_width for _, (_, _, local_width) in result.sections]
        ax_width = ax_curvature.twinx()
        ax_width.step(np.concatenate(([0.0], ends)), np.concatenate(([result.width], widths)), where="pre", color="tab:orange")
        ax_width.set_ylabel("Width (µm)", color="tab:orange")
        ax_curvature.set_title("Curvature and width along the bend")
        self.plot.draw()

    def show_grating(self, r):
        result, X, Z, side, u, spot_x, spot_y, layout = r
        d = result.design
        c = self.coupler
        for key, value in (("period", d.period), ("tooth_width", d.fill_factor * d.period), ("n_teeth", d.n_teeth), ("etch_depth", d.etch_depth), ("angular_width", d.angular_width)):
            c.set(key, f"{value:.4g}")
        if d.fill_start is not None:
            c.set("fill_start", f"{d.fill_start:.4g}")
        ion = self.app.ion
        rabi = lib.rabi_frequency_from_intensity(ion.qubit_linewidth, ion.qubit_wavelength, result.intensity) if ion.qubit_linewidth > 0 else 0.0
        if d.uniform:
            teeth = f"{d.n_teeth} teeth of period {d.period * 1e3:.1f} nm and width {d.fill_factor * d.period * 1e3:.1f} nm, {d.etch_depth * 1e3:.0f} nm deep, over {d.angular_width:.1f}°"
        else:
            teeth = (f"{d.n_teeth} teeth with periods from {layout.periods[0] * 1e3:.1f} to {layout.periods[-1] * 1e3:.1f} nm and fill factors from "
                     f"{layout.fills[0]:.2f} to {layout.fills[-1]:.2f}, {d.etch_depth * 1e3:.0f} nm deep, over {d.angular_width:.1f}°")
        set_text(self.component_results, "\n".join([
            teeth,
            f"Emission angle {result.emission_angle:.2f}° (the ion is at {result.target_angle:.2f}°)",
            f"Grating strength {result.strength:.4f} /µm (decay length {result.decay_length:.1f} µm, grating length {result.length:.1f} µm)",
            f"Upward fraction {result.upward_fraction:.3f}, radiated fraction {result.radiated_fraction:.3f}, efficiency {result.efficiency:.3f}",
            f"Intensity at the ion: {result.intensity:.3e} W/m²",
            f"S-D Rabi frequency of {ion.element} at the ion: 2π × {rabi / TWO_PI / 1e3:.1f} kHz",
        ]))

        (ax_top, ax_side), (ax_spot, ax_angle) = self.plot.axes(2, 2)
        teeth, taper = components.grating_teeth(d, layout)
        ax_top.add_patch(PolygonPatch(taper, closed=True, color="tab:blue", alpha=0.3))
        for tooth in teeth:
            ax_top.add_patch(PolygonPatch(tooth, closed=True, color="tab:blue"))
        ax_top.plot(0, 0, marker="x", color="tab:red", markersize=8, label="Ion (above)")
        ax_top.autoscale_view()
        ax_top.set_aspect("equal")
        ax_top.set_xlabel("x (µm)")
        ax_top.set_ylabel("y (µm)")
        ax_top.legend(fontsize=7)
        ax_top.set_title("Teeth")

        positive = side[side > 0]
        norm = colors.LogNorm(vmin=max(side.max() * 1e-4, positive.min() if positive.size else 1.0), vmax=side.max())
        mesh = ax_side.pcolormesh(X, Z, np.maximum(side, norm.vmin), norm=norm, shading="auto", cmap="inferno")
        self.plot.figure.colorbar(mesh, ax=ax_side, label="Intensity (W/m²)")
        ax_side.plot(0, d.target_height, marker="x", color="cyan", markersize=8)
        ax_side.set_xlabel("x (µm)")
        ax_side.set_ylabel("Height (µm)")
        ax_side.set_title("Emitted beam")

        ax_spot.plot(u, spot_x, label="Along the beam (x)")
        ax_spot.plot(u, spot_y, label="Across the beam (y)")
        ax_spot.set_xlabel("Distance from the ion (µm)")
        ax_spot.set_ylabel("Intensity (W/m²)")
        ax_spot.legend(fontsize=7)
        ax_spot.set_title("Spot at the ion's height")

        if d.uniform:
            labels = ("Upward", "Radiated", "Efficiency")
            ax_angle.bar(labels, (result.upward_fraction, result.radiated_fraction, result.efficiency), color=("tab:blue", "tab:orange", "tab:green"))
            ax_angle.set_ylim(0, 1)
            ax_angle.set_title("Efficiency budget")
        else:
            # Period and fill factor of every tooth of a non-uniform grating
            teeth_index = np.arange(1, len(layout.periods) + 1)
            ax_angle.plot(teeth_index, layout.periods * 1e3, marker=".", color="tab:blue")
            ax_angle.set_xlabel("Tooth")
            ax_angle.set_ylabel("Period (nm)", color="tab:blue")
            ax_fill = ax_angle.twinx()
            ax_fill.plot(teeth_index, layout.fills, marker=".", color="tab:orange")
            ax_fill.set_ylabel("Fill factor", color="tab:orange")
            ax_angle.set_title(f"Teeth (efficiency {result.efficiency:.3f})")
        self.plot.draw()

    def load_trap_grating(self):
        trap = self.app.trap
        if trap is None or not trap.grating_couplers():
            raise ValueError("The current trap has no grating couplers (enable them in the Trap tab preset)")
        # The first qubit grating, if there is one
        grating = trap.grating("qubit") or trap.grating_couplers()[0]
        waveguide = grating.waveguide
        for key, value in (("wavelength", waveguide.wavelength), ("width", waveguide.width), ("thickness", waveguide.thickness), ("n_core", waveguide.n_core), ("n_cladding", waveguide.n_cladding)):
            self.waveguide.set(key, f"{value:g}")
        for key, value in (("x", grating.position[0]), ("y", grating.position[1]), ("fill_factor", grating.fill_factor), ("etch_depth", grating.etch_depth), ("length", grating.length), ("grating_width", grating.width), ("power", grating.power * 1e3), ("efficiency", grating.efficiency)):
            self.grating.set(key, f"{value:g}")
        self.evaluate()

    def evaluate(self):
        w = self.waveguide
        waveguide = Waveguide(w.get("wavelength"), w.get("width"), w.get("thickness"), w.get("n_core"), w.get("n_cladding"))
        g = self.grating
        target = np.array([g.get("target_x"), g.get("target_y"), g.get("target_z")])
        grating = Grating(
            waveguide=waveguide,
            position=(g.get("x"), g.get("y"), 0.0),
            fill_factor=g.get("fill_factor"),
            etch_depth=g.get("etch_depth"),
            length=g.get("length"),
            width=g.get("grating_width"),
            power=g.get("power") * 1e-3,
            efficiency=g.get("efficiency"),
        )
        focus = g.get("focus")
        s = self.substrate
        n_box = s.get("n_box")
        n_substrate = complex(s.get("n_substrate"), s.get("k_substrate"))
        box_max = s.get("box_max")
        ion = self.app.ion

        def work():
            n_eff = waveguide.effective_index()
            if n_eff is None:
                raise ValueError("The waveguide does not guide a mode at this wavelength")
            slab_te = lib.slab_mode(waveguide.n_core, waveguide.n_cladding, waveguide.n_cladding, waveguide.thickness, waveguide.wavelength, "TE")
            slab_tm = lib.slab_mode(waveguide.n_core, waveguide.n_cladding, waveguide.n_cladding, waveguide.thickness, waveguide.wavelength, "TM")
            _, n_group = waveguide.dispersion(waveguide.wavelength - 0.02, waveguide.wavelength + 0.02)

            widths = np.linspace(0.2, 2.0, 60)
            n_eff_widths = np.array([Waveguide(waveguide.wavelength, width, waveguide.thickness, waveguide.n_core, waveguide.n_cladding).effective_index() or np.nan for width in widths])

            grating.design(target)
            if not focus:
                grating.focal_distance = 0.0
            emission = grating.emission()
            if emission is None:
                raise ValueError("No diffraction order of the grating radiates")
            theta, k_hat = emission

            periods = np.linspace(0.5 * grating.period, 2.0 * grating.period, 200)
            n_grating = grating.effective_index()
            sin_theta = n_grating - waveguide.wavelength / periods
            angles = np.where(np.abs(sin_theta) < 1, np.degrees(np.arcsin(np.clip(sin_theta, -1, 1))), np.nan)

            intensity_target, k_vector = grating.beam_at(target)
            rabi_frequency = lib.rabi_frequency_from_intensity(ion.qubit_linewidth, ion.qubit_wavelength, intensity_target) if ion.qubit_linewidth > 0 else 0.0
            eta = lib.lamb_dicke_parameter(k_vector, (1, 0, 0), ion, AXIAL_FREQUENCY)

            # Intensity in the plane containing the beam axis and the surface normal
            span = np.linalg.norm(target[:2] - grating.position[:2])
            u = np.linspace(-0.5 * span - 30, 1.5 * span + 30, 220)
            z = np.linspace(0, 1.4 * target[2] + 10, 160)
            direction = (target[:2] - grating.position[:2]) / span if span > 0 else np.array([1.0, 0.0])
            U_grid, Z_grid = np.meshgrid(u, z)
            points = np.stack([grating.position[0] + U_grid * direction[0], grating.position[1] + U_grid * direction[1], Z_grid], axis=-1)
            intensity = grating.intensity(points)

            theta_box = np.arcsin(np.sin(theta) / n_box)
            thicknesses = np.linspace(0.05, box_max, 300)
            directionality = np.array([lib.grating_directionality(n_box, t, n_substrate, waveguide.wavelength, theta_box) for t in thicknesses])

            return dict(n_eff=n_eff, slab_te=slab_te, slab_tm=slab_tm, n_group=n_group, widths=widths, n_eff_widths=n_eff_widths,
                        theta=theta, periods=periods, angles=angles, intensity_target=intensity_target, rabi=rabi_frequency, eta=eta,
                        U_grid=U_grid, Z_grid=Z_grid, intensity=intensity, span=span, thicknesses=thicknesses, directionality=directionality,
                        period=grating.period, n_grating=n_grating, focal_distance=grating.focal_distance)

        def done(r):
            set_text(self.results, "\n".join([
                f"Waveguide effective index: {r['n_eff']:.5f} (slab TE {r['slab_te']:.5f}, TM {r['slab_tm']:.5f})",
                f"Group index: {r['n_group']:.4f}",
                f"Grating effective index: {r['n_grating']:.5f}",
                f"Period: {r['period'] * 1e3:.2f} nm, emission angle: {np.degrees(r['theta']):.2f}°",
                f"Focal distance: {r['focal_distance']:.2f} µm",
                f"Intensity at target: {r['intensity_target']:.3e} W/m²",
                f"S-D Rabi frequency of {ion.element}: 2π × {r['rabi'] / TWO_PI / 1e3:.1f} kHz",
                f"Lamb-Dicke parameter (axial, {AXIAL_FREQUENCY / TWO_PI / 1e6:g} MHz): {r['eta']:.4f}",
                f"Directionality range: {np.nanmin(r['directionality']):.3f} to {np.nanmax(r['directionality']):.3f}",
            ]))

            (ax_neff, ax_angle), (ax_beam, ax_dir) = self.plot.axes(2, 2)
            ax_neff.plot(r["widths"], r["n_eff_widths"])
            ax_neff.axvline(waveguide.width, color="gray", linestyle=":")
            ax_neff.set_xlabel("Waveguide width (µm)")
            ax_neff.set_ylabel("Effective index")
            ax_neff.set_title("Fundamental quasi-TE mode")

            ax_angle.plot(r["periods"] * 1e3, r["angles"])
            ax_angle.plot(r["period"] * 1e3, np.degrees(r["theta"]), marker="o", color="tab:red")
            ax_angle.set_xlabel("Grating period (nm)")
            ax_angle.set_ylabel("Emission angle (°)")
            ax_angle.set_title("First-order emission")

            intensity = r["intensity"]
            positive = intensity[intensity > 0]
            vmax = intensity.max() if intensity.max() > 0 else 1.0
            norm = colors.LogNorm(vmin=max(vmax * 1e-6, positive.min() if positive.size else vmax * 1e-6), vmax=vmax)
            mesh = ax_beam.pcolormesh(r["U_grid"], r["Z_grid"], np.maximum(intensity, norm.vmin), norm=norm, shading="auto", cmap="inferno")
            self.plot.figure.colorbar(mesh, ax=ax_beam, label="Intensity (W/m²)")
            ax_beam.plot(r["span"], target[2], marker="x", color="cyan", markersize=10, label="Target")
            ax_beam.set_xlabel("Distance along the trap surface from the grating (µm)")
            ax_beam.set_ylabel("Height (µm)")
            ax_beam.set_title("Emitted beam")
            ax_beam.legend(fontsize=8)

            ax_dir.plot(r["thicknesses"], r["directionality"])
            ax_dir.set_xlabel("Buried oxide thickness (µm)")
            ax_dir.set_ylabel("Upward fraction")
            ax_dir.set_ylim(0, 1)
            ax_dir.set_title("Directionality with substrate reflection")
            self.plot.draw()

        self.app.run_task("Evaluating grating coupler", work, done)


# ---------------------------------------------------------------------------------------------------------------------
# Quantum dynamics
# ---------------------------------------------------------------------------------------------------------------------


class QuantumTab(Tab):
    COLUMNS = (("start", "t₀ (µs)"), ("duration", "τ (µs)"), ("detuning", "Δ (kHz)"), ("rabi", "Ω (kHz)"), ("phase", "φ (π)"), ("eta", "η"), ("targets", "Ions"),
               ("shape", "Shape"), ("p1", "p₁ (µs)"), ("p2", "p₂ (µs)"))
    PRESETS = ("Carrier Rabi flopping", "Blue sideband", "Red sideband", "Ramsey", "Mølmer-Sørensen gate")
    SIMULATIONS = ("Pulse program", "Circuit", "Gate on several modes", "Zeeman structure", "Fit measured flopping")
    SHAPE_HELP = {
        "square": "Square pulses switch on and off at once.",
        "trapezoid": "Trapezoidal pulses ramp on over p₁ and off over p₂ (a triangle if they add up to τ).",
        "gaussian": "Gaussian pulses have width p₁ and are shifted down to start and end at zero.",
        "cosine": "Cosine pulses ramp on over p₁ and off over p₂ as sin² (a sin² pulse if they add up to τ).",
        "blackman": "Blackman pulses span the whole pulse and have no parameters.",
    }

    def __init__(self, app):
        super().__init__(app, "Quantum")

        selector = self.form("Simulate")
        self.simulation = selector.combo("simulation", "Simulate", self.SIMULATIONS, self.SIMULATIONS[0])
        self.simulation.trace_add("write", lambda *args: self.show_simulation())

        # Pulse programs
        mode = self.form("Ions and motional mode")
        mode.combo("n_ions", "Ions", ("1", "2", "3"), "1")
        mode.entry("f_mode", "Mode frequency", 1.0, "MHz")
        mode.entry("n_max", "Highest Fock state", 15)
        mode.entry("n_bar", "Thermal occupation", 0.05)
        mode.entry("alpha", "Coherent displacement", 0)
        mode.entry("heating", "Heating rate", 0, "quanta/s")
        mode.entry("dephasing", "Dephasing rate", 0, "1/s")
        mode.entry("n_samples", "Samples", 300)
        mode.check("decay", "Include D-state decay", True)
        self.mode = mode

        presets = self.form("Presets")
        self.preset_var = presets.combo("preset", "Sequence", self.PRESETS, self.PRESETS[0])
        presets.combo("shape", "Pulse shape", lib.PULSE_SHAPES, "square")
        presets.entry("ramp", "Ramp time (trapezoid, cosine)", 2.0, "µs")
        presets.entry("gaussian_width", "Gaussian width / duration", 0.2)
        presets.buttons(("Load", self.load_preset))
        note = presets.text(2)
        self.presets = presets
        set_text(note, "Presets keep the pulse area (and the Mølmer-Sørensen gate its closed loop and Bell state) for any shape, so shaped pulses are longer.")

        editor = ttk.LabelFrame(self.controls, text="Pulses", padding=6)
        self.tree = ttk.Treeview(editor, columns=[c[0] for c in self.COLUMNS], show="headings", height=6)
        for key, text in self.COLUMNS:
            self.tree.heading(key, text=text)
            self.tree.column(key, width=58 if key == "shape" else 38, minwidth=30, anchor="e")
        self.tree.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", self.select_pulse)

        entry_row = ttk.Frame(editor)
        entry_row.pack(fill="x", pady=(4, 0))
        self.pulse_vars = {}
        for i, (key, text) in enumerate(self.COLUMNS[:7]):
            ttk.Label(entry_row, text=text.split(" ")[0]).grid(row=0, column=i)
            var = tk.StringVar()
            ttk.Entry(entry_row, textvariable=var, width=5).grid(row=1, column=i, padx=1)
            self.pulse_vars[key] = var
        shape_row = ttk.Frame(editor)
        shape_row.pack(fill="x", pady=(4, 0))
        ttk.Label(shape_row, text="Shape").grid(row=0, column=0, sticky="w")
        self.pulse_vars["shape"] = tk.StringVar(value="square")
        ttk.Combobox(shape_row, textvariable=self.pulse_vars["shape"], values=lib.PULSE_SHAPES, state="readonly", width=10).grid(row=1, column=0, padx=(0, 4))
        for i, key in enumerate(("p1", "p2")):
            ttk.Label(shape_row, text=dict(self.COLUMNS)[key]).grid(row=0, column=1 + i)
            var = tk.StringVar(value="0")
            ttk.Entry(shape_row, textvariable=var, width=6).grid(row=1, column=1 + i, padx=1)
            self.pulse_vars[key] = var
        self.shape_help = ttk.Label(editor, text=self.SHAPE_HELP["square"], wraplength=380, foreground="gray")
        self.shape_help.pack(fill="x", pady=(2, 0))
        self.pulse_vars["shape"].trace_add("write", lambda *args: self.shape_help.configure(text=self.SHAPE_HELP.get(self.pulse_vars["shape"].get(), "")))
        buttons = ttk.Frame(editor)
        buttons.pack(fill="x", pady=(4, 0))
        for text, command in (("Add", self.add_pulse), ("Update", self.update_pulse), ("Remove", self.remove_pulse), ("Clear", self.clear_pulses)):
            ttk.Button(buttons, text=text, command=command).pack(side="left", padx=(0, 4))
        self.editor = editor

        run = self.form("Run")
        run.buttons(("Simulate", self.simulate))
        self.results = run.text(6)
        self.pulse_forms = (mode, presets, editor, run)

        # Circuits
        circuit = self.form("Circuit")
        circuit.combo("preset", "Example", tuple(circuits.PRESETS), "Bell pair")
        circuit.buttons(("Load example", self.load_circuit_preset), ("Open...", self.open_circuit), ("Save...", self.save_circuit))
        self.circuit_form = circuit

        qasm = ttk.LabelFrame(self.controls, text="OpenQASM (2 or 3)", padding=6)
        self.qasm = ScrolledText(qasm, height=16, width=44, wrap="none", font=("TkFixedFont", 9), undo=True)
        self.qasm.pack(fill="both", expand=True)
        self.qasm.insert("1.0", circuits.PRESETS["Bell pair"])
        self.qasm_frame = qasm

        errors = self.form("Gate errors")
        errors.entry("single_qubit", "Single-qubit gate infidelity", 1e-4)
        errors.entry("two_qubit", "Two-qubit gate infidelity", 5e-3)
        errors.entry("measurement", "Readout error", 1e-3)
        errors.entry("idle", "Dephasing of idle qubits per layer", 0.0)
        errors.buttons(("Use co-design estimates", self.use_codesign_errors))
        self.errors = errors

        circuit_run = self.form("Run circuit")
        circuit_run.buttons(("Simulate", self.simulate_circuit), ("Native gates", self.show_native_gates))
        self.circuit_results = ScrolledText(circuit_run, height=12, width=44, wrap="none")
        self.circuit_results.grid(row=circuit_run.row, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        circuit_run.row += 1
        self.circuit_forms = (circuit, qasm, errors, circuit_run)

        # Gates on several modes
        gate = self.form("Mølmer-Sørensen gate on several modes")
        gate.combo("source", "Ions and modes", ("Ideal chain", "Crystal of the Crystals tab"), "Ideal chain")
        gate.entry("n_ions", "Ions in the ideal chain", 2)
        gate.entry("f_axial", "Axial frequency of the ideal chain", 1.0, "MHz")
        gate.entry("eta", "Lamb-Dicke parameter of its COM mode", 0.1)
        gate.entry("targets", "Target ions (from 0)", "0, 1")
        gate.combo("kind", "Modulation", ("AM, robust to mode drifts", "AM", "FM"), "AM, robust to mode drifts")
        gate.entry("duration", "Duration", 100, "µs")
        gate.entry("offset", "Detuning above the highest mode", 20, "kHz")
        gate.entry("segments", "Segments (blank for the fewest)", "")
        gate.entry("cutoff", "Highest Fock state of each mode", 6)
        gate.entry("n_bar", "Thermal occupation", 0.0)
        gate.entry("mode_error", "Drift of the mode frequencies", 0, "kHz")
        gate.buttons(("Design", lambda: self.design_gate(False)), ("Design and simulate", lambda: self.design_gate(True)))
        self.gate_results = gate.text(8)
        self.gate_form = gate
        self.gate_forms = (gate,)

        zeeman = self.form("Zeeman structure")
        zeeman.entry("B", "Magnetic field", 4.0, "G")
        zeeman.entry("theta", "Polar angle of the field", 0, "°")
        zeeman.entry("phi", "Azimuth of the field", 0, "°")
        zeeman.combo("beam", "Beam", ("Along (1, 0, 1)", "Along x", "Qubit grating of the trap"), "Along (1, 0, 1)")
        zeeman.entry("polarization", "Polarization (0: in the plane of k and z)", 45, "°")
        zeeman.entry("intensity", "Intensity", 1e8, "W/m²")
        zeeman.buttons(("Show the lines", self.show_zeeman), ("Best polarization", self.best_polarization))
        self.zeeman_results = zeeman.text(10)
        self.zeeman_form = zeeman
        self.zeeman_forms = (zeeman,)

        fit = self.form("Fit measured flopping")
        fit.entry("path", "Data (two columns: µs, excited population)", "")
        fit.buttons(("Open...", self.open_data), ("Simulated example", self.example_data))
        fit.combo("sideband", "Driven on", ("carrier", "blue sideband", "red sideband"), "blue sideband")
        fit.entry("eta", "Lamb-Dicke parameter", 0.1)
        fit.check("fit_eta", "Fit the Lamb-Dicke parameter too", False)
        fit.buttons(("Fit", self.fit_flopping))
        self.fit_results = fit.text(6)
        self.fit_form = fit
        self.fit_forms = (fit,)
        self.data = None

        self.load_preset()
        self.show_simulation()

    def show_simulation(self):
        choice = self.simulation.get()
        groups = {
            self.SIMULATIONS[0]: (self.pulse_forms, "Load a preset or edit the pulses, then simulate"),
            self.SIMULATIONS[1]: (self.circuit_forms, "Edit the circuit, then simulate it with and without gate errors"),
            self.SIMULATIONS[2]: (self.gate_forms, "Design an amplitude- or frequency-modulated gate which closes every mode of a chain"),
            self.SIMULATIONS[3]: (self.zeeman_forms, "Show the Zeeman lines of the qubit and how strongly a beam drives each"),
            self.SIMULATIONS[4]: (self.fit_forms, "Fit Rabi flopping on the carrier or a sideband to find the Rabi frequency and temperature"),
        }
        for forms, _ in groups.values():
            for form in forms:
                form.pack_forget()
        forms, message = groups.get(choice, groups[self.SIMULATIONS[0]])
        for form in forms:
            form.pack(fill="both" if form in (self.qasm_frame,) else "x", expand=form is self.qasm_frame, pady=(0, 6), padx=(0, 4))
        self.plot.message(message)

    # Gates on several modes -----------------------------------------------------------------------------------

    def _gate_modes(self):
        """Mode frequencies (rad/s) and Lamb-Dicke matrix (ions x modes) of the axial modes of the chain in the form."""
        form = self.gate_form
        if form.get_str("source").startswith("Crystal"):
            crystal = self.app.crystal
            if crystal is None:
                raise ValueError("Find a crystal in the Crystals tab first")
            k = self.app.crystal_k
            modes = crystal.modes_along("x") or list(range(crystal.n_ions))
            return np.abs(crystal.frequencies[modes]), crystal.lamb_dicke(k, modes), crystal.n_ions
        n_ions = form.get_int("n_ions")
        if not 2 <= n_ions <= 6:
            raise ValueError("Use 2 to 6 ions")
        omega = TWO_PI * form.get("f_axial") * 1e6
        potential = crystals.HarmonicPotential(self.app.ion, (omega, 6 * omega, 6.5 * omega))
        crystal = crystals.find_crystal(potential, self.app.ion, n_ions=n_ions)
        modes = crystal.modes_along("x")
        eta = crystal.lamb_dicke([1.0, 0.0, 0.0], modes)
        # Scale the light so that the center-of-mass mode has the Lamb-Dicke parameter in the form
        eta *= form.get("eta") / np.abs(eta[:, 0]).max()
        return np.abs(crystal.frequencies[modes]), eta, n_ions

    def design_gate(self, simulate):
        form = self.gate_form
        frequencies, eta, n_ions = self._gate_modes()
        targets = tuple(int(t) for t in form.get_str("targets").split(","))
        if len(targets) != 2 or not all(0 <= t < n_ions for t in targets) or targets[0] == targets[1]:
            raise ValueError(f"Give two different target ions between 0 and {n_ions - 1}")
        duration = form.get("duration") * 1e-6
        detuning = frequencies.max() + TWO_PI * form.get("offset") * 1e3
        segments = parse_optional(form.get_str("segments"))
        kind = form.get_str("kind")
        cutoff = form.get_int("cutoff")
        n_bar = form.get("n_bar")
        drift = TWO_PI * form.get("mode_error") * 1e3
        if simulate and (cutoff + 1) ** len(frequencies) * 2 ** n_ions > 4000:
            raise ValueError("The simulation would be too large; use fewer ions, modes, or Fock states")

        def work():
            if kind.startswith("FM"):
                gate = gates.design_fm_gate(frequencies, eta, duration, detuning, targets=targets, n_segments=int(segments or 8))
            else:
                gate = gates.design_am_gate(frequencies, eta, duration, detuning, targets=targets, n_segments=None if segments is None else int(segments),
                                            robust=kind.startswith("AM, robust"))
            result = None
            if simulate:
                modes = multimode.Modes(frequencies + drift, eta, [cutoff] * len(frequencies), [n_bar] * len(frequencies))
                result = multimode.simulate(n_ions, modes, gate.tones(), duration, n_samples=101)
            return gate, result

        def done(output):
            gate, result = output
            lines = gate.summary()
            ax_amplitude, ax_populations = self.plot.axes(2, 1)[:, 0]
            edges = gate.edges * 1e6
            ax_amplitude.stairs(gate.amplitudes / TWO_PI / 1e3, edges, color="tab:purple", label="Rabi frequency")
            ax_amplitude.set_ylabel("Rabi frequency (kHz)")
            if not np.allclose(gate.detunings, gate.detunings[0]):
                ax_detuning = ax_amplitude.twinx()
                ax_detuning.stairs((gate.detunings - frequencies.max()) / TWO_PI / 1e3, edges, color="tab:green", label="Detuning above the highest mode")
                ax_detuning.set_ylabel("Detuning above the highest mode (kHz)")
            ax_amplitude.set_xlabel("Time (µs)")
            ax_amplitude.set_title(f"Gate ({kind}) on ions {targets[0]} and {targets[1]} with {len(frequencies)} modes")
            if result is not None:
                times = result.times * 1e6
                n = n_ions
                labels = {0: "g" * n, 2 ** n - 1: "e" * n}
                for state, label in labels.items():
                    ax_populations.plot(times, result.state_populations[:, state], label=f"P({label})")
                others = 1 - result.state_populations[:, 0] - result.state_populations[:, -1]
                ax_populations.plot(times, others, label="Other states")
                ax_populations.set_xlabel("Time (µs)")
                ax_populations.set_ylabel("Population")
                ax_populations.legend(fontsize=7)
                lines.append(f"Simulated Bell state fidelity: {result.bell_fidelity(targets):.5f}")
                lines.append("Mean phonon numbers at the end: " + ", ".join(f"{n_:.3f}" for n_ in result.mean_phonon_numbers[-1]))
            else:
                ax_populations.set_axis_off()
            self.plot.draw()
            set_text(self.gate_results, "\n".join(lines))

        self.app.run_task("Designing the gate" + (" and simulating it" if simulate else ""), work, done)

    # Zeeman structure ------------------------------------------------------------------------------------------

    def _zeeman_inputs(self):
        form = self.zeeman_form
        ion = self.app.ion
        B = form.get("B") * levels.GAUSS
        theta, phi = np.radians(form.get("theta")), np.radians(form.get("phi"))
        B_direction = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        k = beam_wavevector(self.app, form.get_str("beam"))
        return ion, B, B_direction, k

    def show_zeeman(self):
        ion, B, B_direction, k = self._zeeman_inputs()
        form = self.zeeman_form
        polarization = levels.polarization_vector(k, form.get("polarization"))
        intensity = form.get("intensity")
        ax = self.plot.axes()[0, 0]
        if ion.qubit_linewidth <= 0:
            if ion.element not in levels.HYPERFINE:
                raise ValueError(f"{ion.element} has no optical qubit or hyperfine data")
            fields = np.linspace(0, max(3 * B, 1e-4), 200)
            all_levels = [levels.hyperfine_levels(ion.element, b) for b in fields]
            for index in range(len(all_levels[0])):
                ax.plot(fields / levels.GAUSS, [lv[index].energy / 1e9 for lv in all_levels], linewidth=0.8)
            ax.axvline(B / levels.GAUSS, color="gray", linestyle=":")
            ax.set_xlabel("Magnetic field (G)")
            ax.set_ylabel("Energy (GHz)")
            ax.set_title(f"Hyperfine structure of the ground state of {ion.element} (Breit-Rabi)")
            self.plot.draw()
            set_text(self.zeeman_results, "\n".join(levels.zeeman_summary(ion, B)))
            return
        lines_ = levels.qubit_lines(ion, B, k, polarization, B_direction, intensity)
        colors_by_dm = {-2: "tab:blue", -1: "tab:cyan", 0: "tab:green", 1: "tab:orange", 2: "tab:red"}
        for line in lines_:
            ax.vlines(line.shift / 1e6, 0, line.rabi_frequency / TWO_PI / 1e3, color=colors_by_dm[line.delta_m], linewidth=2)
            if line.strength > 1e-3:
                ax.annotate(line.label().replace("S", "").replace(" -> D", "→"), (line.shift / 1e6, line.rabi_frequency / TWO_PI / 1e3), fontsize=6, ha="center", textcoords="offset points", xytext=(0, 3))
        for dm, color in colors_by_dm.items():
            ax.plot([], [], color=color, label=f"Δm = {dm:+d}")
        ax.set_xlabel("Shift from the line center (MHz)")
        ax.set_ylabel("Rabi frequency (kHz)")
        ax.set_title(f"Zeeman lines of the {ion.qubit_wavelength * 1e9:.0f} nm qubit of {ion.element} at {B / levels.GAUSS:g} G")
        ax.legend(fontsize=7)
        self.plot.draw()
        set_text(self.zeeman_results, "\n".join(levels.zeeman_summary(ion, B, k, polarization, B_direction, intensity)))

    def best_polarization(self):
        ion, B, B_direction, k = self._zeeman_inputs()
        if ion.qubit_linewidth <= 0:
            raise ValueError(f"{ion.element} uses a hyperfine qubit driven by Raman beams")
        from fractions import Fraction
        target = (Fraction(-1, 2), Fraction(-5, 2))
        angle, strength = levels.best_polarization(target, k, B_direction)
        self.zeeman_form.set("polarization", f"{angle:g}")
        self.show_zeeman()
        self.app.set_status(f"Polarization {angle:g}° drives S-1/2 -> D-5/2 most selectively (strength {strength:.3f})")

    # Fits ------------------------------------------------------------------------------------------------------

    def open_data(self):
        path = filedialog.askopenfilename(parent=self, title="Open measured flopping (two columns: time in µs, excited population)", filetypes=[("Text", "*.txt *.csv *.dat"), ("All files", "*")])
        if path:
            self.fit_form.set("path", path)
            data = np.loadtxt(path, delimiter="," if path.endswith(".csv") else None)
            self.data = (data[:, 0] * 1e-6, data[:, 1])
            self._plot_data()

    def example_data(self):
        rng = np.random.default_rng(1)
        times = np.linspace(0, 120e-6, 50)
        populations = analytic.rabi_flop(times, TWO_PI * 60e3, 0.1, 0.6, 1)
        populations = np.clip(populations + 0.02 * rng.standard_normal(len(times)), 0, 1)
        self.data = (times, populations)
        self.fit_form.set("sideband", "blue sideband")
        self.fit_form.set("eta", 0.1)
        self.fit_form.set("path", "(simulated: Ω = 2π × 60 kHz, n̄ = 0.6, η = 0.1)")
        self._plot_data()

    def _plot_data(self, fit=None, sideband=0, eta=0.1):
        times, populations = self.data
        ax = self.plot.axes()[0, 0]
        ax.plot(times * 1e6, populations, "o", markersize=4, label="Data")
        if fit is not None:
            t = np.linspace(0, times.max(), 400)
            ax.plot(t * 1e6, analytic.rabi_flop(t, fit["rabi_frequency"], fit["eta"], fit["n_bar"], sideband, fit["detuning"]), label="Fit")
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Excited population")
        ax.legend(fontsize=8)
        self.plot.draw()

    def fit_flopping(self):
        if self.data is None:
            raise ValueError("Open measured data or make a simulated example first")
        form = self.fit_form
        sideband = {"carrier": 0, "blue sideband": 1, "red sideband": -1}[form.get_str("sideband")]
        eta = form.get("eta")
        times, populations = self.data
        fit = analytic.fit_rabi_flop(times, populations, sideband, eta, fit_eta=form.get("fit_eta"))
        self._plot_data(fit, sideband)
        errors = fit["errors"]
        lines = [
            f"Rabi frequency: 2π × {fit['rabi_frequency'] / TWO_PI / 1e3:.3f} ± {errors['rabi_frequency'] / TWO_PI / 1e3:.3f} kHz",
            f"Mean phonon number: {fit['n_bar']:.3f} ± {errors['n_bar']:.3f}",
        ]
        if form.get("fit_eta"):
            lines.append(f"Lamb-Dicke parameter: {fit['eta']:.4f} ± {errors['eta']:.4f}")
        lines.append(f"Root mean square residual: {fit['rms']:.4f}")
        set_text(self.fit_results, "\n".join(lines))

    # Pulse table ----------------------------------------------------------------------------------------------

    def _row_from_vars(self):
        values = []
        for key, text in self.COLUMNS:
            value = self.pulse_vars[key].get().strip()
            if key == "targets":
                targets = [int(t) for t in value.replace(" ", "").split(",") if t]
                if not targets:
                    raise ValueError("A pulse needs at least one target ion")
                values.append(",".join(str(t) for t in targets))
            elif key == "shape":
                if value not in lib.PULSE_SHAPES:
                    raise ValueError(f"Choose a pulse shape from {', '.join(lib.PULSE_SHAPES)}")
                values.append(value)
            else:
                try:
                    float(value or "0")
                except ValueError:
                    raise ValueError(f"{text} must be a number") from None
                values.append(value or "0")
        return values

    def add_pulse(self):
        self.tree.insert("", "end", values=self._row_from_vars())

    def update_pulse(self):
        selection = self.tree.selection()
        if not selection:
            raise ValueError("Select a pulse to update")
        self.tree.item(selection[0], values=self._row_from_vars())

    def remove_pulse(self):
        for item in self.tree.selection():
            self.tree.delete(item)

    def clear_pulses(self):
        self.tree.delete(*self.tree.get_children())

    def select_pulse(self, event=None):
        selection = self.tree.selection()
        if selection:
            for (key, _), value in zip(self.COLUMNS, self.tree.item(selection[0], "values")):
                self.pulse_vars[key].set(value)

    def set_pulses(self, rows):
        self.clear_pulses()
        for row in rows:
            self.tree.insert("", "end", values=[f"{v:.6g}" if isinstance(v, float) else str(v) for v in row])
        if rows:
            for (key, _), value in zip(self.COLUMNS, rows[0]):
                self.pulse_vars[key].set(f"{value:.6g}" if isinstance(value, float) else str(value))

    def pulse_shape(self):
        """Pulse shape and ramp time (s) of the Presets form, which the co-design uses for gates too."""
        shape = self.presets.get_str("shape")
        if shape not in lib.PULSE_SHAPES:
            raise ValueError(f"Choose a pulse shape from {', '.join(lib.PULSE_SHAPES)}")
        if shape == "gaussian":
            width = self.presets.get("gaussian_width")
            if not 0 < width <= 1:
                raise ValueError("The width of a Gaussian pulse is a fraction of its duration between 0 and 1")
            return shape, width
        return shape, self.presets.get("ramp") * 1e-6

    def load_preset(self):
        preset = self.preset_var.get()
        f_mode_khz = self.mode.get("f_mode") * 1e3
        shape, ramp = self.pulse_shape()

        def pulse(start, area, detuning, rabi_khz, phase, eta, targets):
            # A pulse with the same area as a square pulse of the given duration
            duration, parameters = pulses.shaped_pulse(shape, ramp, area, TWO_PI * rabi_khz * 1e3)
            return (start, duration * 1e6, detuning, rabi_khz, phase, eta, targets, shape, parameters[0] * 1e6, parameters[1] * 1e6)

        if preset == "Carrier Rabi flopping":
            self.mode.set("n_ions", "1")
            self.set_pulses([pulse(0.0, TWO_PI * 100e3 * 30e-6, 0.0, 100.0, 0.0, 0.1, "1")])
        elif preset == "Blue sideband":
            self.mode.set("n_ions", "1")
            self.set_pulses([pulse(0.0, TWO_PI * 100e3 * 200e-6, f_mode_khz, 100.0, 0.0, 0.1, "1")])
        elif preset == "Red sideband":
            self.mode.set("n_ions", "1")
            self.mode.set("n_bar", "0.5")
            self.set_pulses([pulse(0.0, TWO_PI * 100e3 * 200e-6, -f_mode_khz, 100.0, 0.0, 0.1, "1")])
        elif preset == "Ramsey":
            self.mode.set("n_ions", "1")
            first = pulse(0.0, 0.5 * np.pi, 0.0, 200.0, 0.0, 0.0, "1")
            self.set_pulses([first, pulse(first[1] + 20.0, 0.5 * np.pi, 0.0, 200.0, 0.5, 0.0, "1")])
        elif preset == "Mølmer-Sørensen gate":
            self.mode.set("n_ions", "2")
            self.mode.set("n_bar", "0")
            self.mode.set("n_max", "12")
            eta = 0.05
            # Square pulses entangle after one loop with delta = 2 eta Omega, here 20 kHz
            rabi_khz = 20.0 / (2 * eta)
            gates = pulses.ms_pulses(shape, ramp, eta, TWO_PI * rabi_khz * 1e3, TWO_PI * f_mode_khz * 1e3)
            self.set_pulses([(g.start * 1e6, g.duration * 1e6, g.detuning / TWO_PI / 1e3, rabi_khz, 0.0, eta, "1,2", shape, g.shape_parameters[0] * 1e6, g.shape_parameters[1] * 1e6) for g in gates])

    def pulses(self):
        result = []
        for item in self.tree.get_children():
            start, duration, detuning, rabi, phase, eta, targets, shape, p1, p2 = self.tree.item(item, "values")
            result.append(Pulse(
                start=float(start) * 1e-6,
                duration=float(duration) * 1e-6,
                detuning=TWO_PI * float(detuning) * 1e3,
                rabi_frequency=TWO_PI * float(rabi) * 1e3,
                phase=np.pi * float(phase),
                targets=[int(t) for t in str(targets).split(",")],
                lamb_dicke=float(eta),
                shape=shape,
                shape_parameters=(float(p1) * 1e-6, float(p2) * 1e-6),
            ))
        if not result:
            raise ValueError("Add at least one pulse")
        return result

    # Simulation -----------------------------------------------------------------------------------------------

    def simulate(self):
        ion = self.app.ion
        program = self.pulses()
        m = self.mode
        n_ions = int(m.get_str("n_ions"))
        omega_mode = TWO_PI * m.get("f_mode") * 1e6
        kwargs = dict(
            n_ions=n_ions,
            mode_frequency=omega_mode,
            n_max=m.get_int("n_max"),
            n_bar=m.get("n_bar"),
            alpha=m.get("alpha"),
            heating_rate=m.get("heating"),
            dephasing_rate=m.get("dephasing"),
            decay=m.get("decay"),
            n_samples=m.get_int("n_samples"),
            density_matrix=n_ions == 2,
        )

        def work():
            return lib.simulate_pulses(ion, program, **kwargs)

        def done(result):
            times = result.times * 1e6
            has_bars = n_ions == 2
            axes = self.plot.axes(4 if has_bars else 3, 1, sharex=False)[:, 0]
            ax_envelope, ax_pop, ax_n = axes[0], axes[1], axes[2]

            # Rabi frequency of every pulse over time
            for p, pulse in enumerate(program):
                t = np.linspace(pulse.start, pulse.start + pulse.duration, 400)
                envelope = pulses.envelope(pulse.shape, pulse.shape_parameters, pulse.duration, t - pulse.start)
                ax_envelope.plot(np.concatenate(([pulse.start], t, [pulse.start + pulse.duration])) * 1e6,
                                 np.concatenate(([0.0], envelope, [0.0])) * pulse.rabi_frequency / TWO_PI / 1e3, label=f"Pulse {p + 1} ({pulse.shape})")
            ax_envelope.set_ylabel("Ω (kHz)")
            ax_envelope.set_xlim(times[0], times[-1])
            ax_envelope.legend(fontsize=7, ncols=2)
            ax_envelope.set_title(f"{len(program)} pulse(s) on {n_ions} {ion.element} ion(s)")

            for i in range(n_ions):
                ax_pop.plot(times, result.excited_populations[:, i], label=f"Ion {i + 1}")

            # Compare a single square pulse on a single ion with the analytical thermal result
            lines = []
            if n_ions == 1 and len(program) == 1 and program[0].shape == "square" and kwargs["heating_rate"] == 0 and kwargs["dephasing_rate"] == 0 and kwargs["alpha"] == 0:
                pulse = program[0]
                sideband = int(round(pulse.detuning / omega_mode))
                residual = pulse.detuning - sideband * omega_mode
                inside = (result.times >= pulse.start) & (result.times <= pulse.start + pulse.duration)
                analytic = np.zeros_like(result.times)
                analytic[inside] = lib.thermal_rabi_excitation(pulse.rabi_frequency, pulse.lamb_dicke, sideband, kwargs["n_bar"], residual, result.times[inside] - pulse.start, kwargs["n_max"])
                ax_pop.plot(times[inside], analytic[inside], color="black", linestyle=":", label="Analytical")
                lines.append(f"Largest deviation from analytical: {np.abs(analytic[inside] - result.excited_populations[inside, 0]).max():.2e}")

            ax_pop.set_ylabel("D-state population")
            ax_pop.set_ylim(-0.02, 1.02)
            ax_pop.set_xlim(times[0], times[-1])
            ax_pop.legend(fontsize=8)

            ax_n.plot(times, result.mean_phonon_numbers)
            ax_n.set_ylabel("Mean phonon number")
            ax_n.set_xlabel("Time (µs)")
            ax_n.set_xlim(times[0], times[-1])

            lines.insert(0, "Final D-state populations: " + ", ".join(f"{p:.4f}" for p in result.excited_populations[-1]))
            lines.insert(1, f"Final mean phonon number: {result.mean_phonon_numbers[-1]:.4f}")
            lines.insert(2, "Pulse areas: " + ", ".join(f"{pulse.area / np.pi:.3f}π" for pulse in program))

            if has_bars:
                rho = result.density_matrix
                n_motion = kwargs["n_max"] + 1
                populations = [np.real(sum(rho[c * n_motion + k, c * n_motion + k] for k in range(n_motion))) for c in range(4)]
                coherence = sum(rho[k, 3 * n_motion + k] for k in range(n_motion))
                fidelity = 0.5 * (populations[0] + populations[3]) + abs(coherence)
                ax_bars = axes[3]
                ax_bars.bar(["SS", "DS", "SD", "DD"], populations)
                ax_bars.set_ylabel("Final population")
                ax_bars.set_ylim(0, 1)
                lines.append(f"Bell state fidelity: {fidelity:.5f}")
            set_text(self.results, "\n".join(lines))
            self.plot.draw()

        self.app.run_task("Simulating quantum dynamics", work, done)

    # Circuits -------------------------------------------------------------------------------------------------

    def circuit_text(self):
        return self.qasm.get("1.0", "end-1c")

    def circuit(self):
        """Circuit in the editor, as a circuits.QuantumCircuit (which the Scheduling and Co-design tabs use too)."""
        return circuits.parse_qasm(self.circuit_text())

    def set_circuit_text(self, text):
        self.qasm.delete("1.0", "end")
        self.qasm.insert("1.0", text)

    def load_circuit_preset(self):
        name = self.circuit_form.get_str("preset")
        if name not in circuits.PRESETS:
            raise ValueError(f"Choose an example from {', '.join(circuits.PRESETS)}")
        self.set_circuit_text(circuits.PRESETS[name])

    def open_circuit(self):
        path = filedialog.askopenfilename(parent=self, title="Open OpenQASM file", initialdir=EXAMPLES / "circuits", filetypes=[("OpenQASM", "*.qasm"), ("All files", "*")])
        if path:
            self.set_circuit_text(Path(path).read_text())
            self.app.set_status(f"Opened {path}")

    def save_circuit(self):
        path = filedialog.asksaveasfilename(parent=self, title="Save OpenQASM file", initialdir=EXAMPLES / "circuits", defaultextension=".qasm", filetypes=[("OpenQASM", "*.qasm")])
        if path:
            Path(path).write_text(self.circuit_text())
            self.app.set_status(f"Saved the circuit to {path}")

    def gate_errors(self):
        e = self.errors
        values = [e.get(key) for key in ("single_qubit", "two_qubit", "measurement", "idle")]
        if any(not 0 <= v <= 1 for v in values):
            raise ValueError("Errors are probabilities between 0 and 1")
        return circuits.GateErrors(*values)

    def use_codesign_errors(self):
        evaluation = self.app.tabs["codesign"].best_evaluation()
        if evaluation is None or "single_qubit_fidelity" not in evaluation.metrics:
            raise ValueError("Run the full co-design in the Co-design tab first")
        m = evaluation.metrics
        self.errors.set("single_qubit", f"{1 - m['single_qubit_fidelity']:.3g}")
        self.errors.set("two_qubit", f"{1 - m['two_qubit_fidelity']:.3g}")
        self.errors.set("measurement", f"{1 - m['detection_fidelity']:.3g}")
        self.app.set_status("Gate errors set from the best co-designed system")

    def show_native_gates(self):
        circuit = self.circuit()
        counts = circuits.native_counts(circuit)
        lines = [f"{circuit.n_qubits} qubits, depth {circuit.depth}: {counts.ms_gates} Mølmer-Sørensen gates, {counts.rotations} rotations, "
                 f"{counts.virtual} virtual Z rotations, and {counts.measurements} measurements", ""]
        lines += circuits.native_gates(circuit)
        lines += ["", "For the scheduler:", circuits.to_scheduler_qasm(circuit)]
        self.circuit_results.delete("1.0", "end")
        self.circuit_results.insert("1.0", "\n".join(lines))

    def simulate_circuit(self):
        circuit = self.circuit()
        errors = self.gate_errors()

        def work():
            return circuits.simulate(circuit, errors)

        def done(result):
            counts = circuits.native_counts(circuit)
            lines = [
                f"{circuit.n_qubits} qubits, depth {circuit.depth}, {len(circuit.operations)} gates ({counts.ms_gates} MS gates and {counts.rotations} rotations natively)",
                f"Simulated with the {result.method}",
                f"Fidelity of the final state with errors: {result.state_fidelity:.5f}",
                f"Fidelity of the distribution of outcomes: {result.distribution_fidelity:.5f}",
                "",
                "Outcome    Ideal    With errors",
            ]
            lines += [f"{outcome:>9s}  {ideal:7.4f}  {noisy:7.4f}" for outcome, ideal, noisy in result.top(16)]
            self.circuit_results.delete("1.0", "end")
            self.circuit_results.insert("1.0", "\n".join(lines))

            ax_circuit, ax_outcomes = self.plot.axes(2, 1)[:, 0]
            draw_circuit(ax_circuit, circuit)
            top = result.top(16)
            x = np.arange(len(top))
            ax_outcomes.bar(x - 0.2, [i for _, i, _ in top], width=0.4, label="Ideal")
            ax_outcomes.bar(x + 0.2, [n for _, _, n in top], width=0.4, label="With errors")
            ax_outcomes.set_xticks(x)
            ax_outcomes.set_xticklabels([o for o, _, _ in top], rotation=60 if len(top) > 8 else 0, fontsize=7, family="monospace")
            ax_outcomes.set_ylabel("Probability")
            ax_outcomes.set_title("Outcomes (first classical bit on the right)")
            ax_outcomes.legend(fontsize=8)
            self.plot.draw()

        self.app.run_task("Simulating the circuit", work, done)


def draw_circuit(ax, circuit):
    """Draw the gates of a circuits.QuantumCircuit in its layers."""
    layers = circuit.layers()
    for q in range(circuit.n_qubits):
        ax.axhline(q, color="gray", linewidth=0.8, zorder=0)
    for x, layer in enumerate(layers, start=1):
        for gate in layer:
            if gate.name == "measure":
                ax.text(x, gate.qubits[0], "M", ha="center", va="center", fontsize=8, bbox=dict(boxstyle="square", facecolor="white", edgecolor="black"))
                continue
            if len(gate.qubits) > 1:
                ax.plot([x, x], [min(gate.qubits), max(gate.qubits)], color="tab:blue", linewidth=1.5, zorder=1)
                controlled = gate.name.startswith("c") and gate.name not in ("cswap",)
                for i, q in enumerate(gate.qubits):
                    if controlled and i < len(gate.qubits) - 1:
                        ax.plot(x, q, marker="o", color="tab:blue", markersize=5)
                    else:
                        label = "⊕" if gate.name in ("cx", "ccx") else (gate.name[1:] if controlled else gate.name)
                        ax.text(x, q, label, ha="center", va="center", fontsize=7, bbox=dict(boxstyle="round", facecolor="white", edgecolor="tab:blue"))
            else:
                label = gate.name if not gate.params else f"{gate.name}({', '.join(f'{p:.2g}' for p in gate.params)})"
                ax.text(x, gate.qubits[0], label, ha="center", va="center", fontsize=7, bbox=dict(boxstyle="round", facecolor="white", edgecolor="tab:blue"))
    ax.set_xlim(0.4, len(layers) + 0.6)
    ax.set_ylim(circuit.n_qubits - 0.5, -0.5)
    ax.set_yticks(range(circuit.n_qubits))
    ax.set_yticklabels(circuit.qubit_names or [f"q{q}" for q in range(circuit.n_qubits)], fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_title("Circuit")


# ---------------------------------------------------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------------------------------------------------


class SchedulingTab(Tab):
    # Colors of the kinds of segments of a plan
    SEGMENT_COLORS = {"transport": "tab:orange", "waiting": "0.85", "cooling": "tab:cyan", "single-qubit gate": "tab:blue", "two-qubit gate": "tab:purple", "measurement": "tab:green"}

    def __init__(self, app):
        super().__init__(app, "Scheduling")

        circuit = self.form("Circuit")
        circuit.check("use_quantum", "Use the circuit in the Quantum tab", True)
        self.qasm_var = circuit.entry("qasm", "Or an OpenQASM file", EXAMPLES / "circuits" / "ghz3.qasm")
        circuit.buttons(("Browse...", self.browse))
        self.circuit_form = circuit

        zones = self.form("Trap zones")
        # Wells closer than several ion heights need large voltages to be held at once, so the zones are 100 µm apart
        zones.entry("layout", "Zone layout", lib.DEFAULT_LAYOUT)
        zones.entry("zone_spacing", "Zone spacing", 100, "µm")
        zones.check("trap_zones", "Use the zones of the trap instead", False)
        self.zones = zones

        priority = self.form("Transport priority")
        self.mode_names = {description: mode for mode, description in scheduling.MODES.items()}
        priority.combo("mode", "Mode", tuple(self.mode_names), scheduling.MODES["eager"])
        priority.entry("max_speed", "Fastest transport", 20, "m/s")
        priority.entry("min_speed", "Slowest transport", 1, "m/s")
        priority.entry("min_duration", "Shortest transport", 5, "µs")
        priority.entry("single_qubit_time", "Single-qubit gate", 10, "µs")
        priority.entry("two_qubit_time", "Two-qubit gate", 200, "µs")
        priority.entry("detection_time", "Measurement", 200, "µs")
        priority.entry("heating", "Heating rate", 100, "quanta/s")
        priority.entry("n_bar_cooled", "Mean phonons after cooling", 0.05)
        priority.entry("cooling_time", "Time to cool a waiting ion", 100, "µs")
        priority.check("recool", "Cool ions waiting in load, storage, and cooling zones", True)
        self.priority = priority

        actions = self.form("Actions")
        actions.entry("n_steps", "Waveform steps per transport", 51)
        actions.buttons(("Schedule", self.schedule), ("Compare modes", self.compare), ("Build waveform", self.build_waveform))
        self.actions = actions

        architecture = self.form("QCCD architecture (chains of ions)")
        architecture.combo("topology", "Topology", ("Linear", "Grid", "Ring", "Zones of the trap"), "Linear")
        architecture.entry("n_traps", "Traps (linear, ring)", 4)
        architecture.entry("grid", "Junctions (rows, columns)", "2, 3")
        architecture.entry("capacity", "Ions per trap", 6)
        architecture.combo("gate_model", "Two-qubit gates", gates.GATE_MODELS, "FM")
        architecture.combo("swap", "Reordering", ("gate", "ion"), "gate")
        architecture.check("recool", "Recool every chain after a merge", False)
        architecture.buttons(("Compile", self.compile_qccd), ("Explore capacities", self.explore_qccd))
        self.architecture = architecture

        results = ttk.LabelFrame(self.controls, text="Schedule", padding=6)
        results.pack(fill="both", expand=True, padx=(0, 4))
        self.results = ScrolledText(results, height=16, width=40, wrap="none")
        self.results.pack(fill="both", expand=True)

        self.circuit = None
        self.program = None
        self.plan = None
        self.trap_graph = None
        self.plot.message("Schedule the transports of a circuit")

    def browse(self):
        path = filedialog.askopenfilename(parent=self, title="Open OpenQASM file", initialdir=EXAMPLES / "circuits", filetypes=[("OpenQASM", "*.qasm"), ("All files", "*")])
        if path:
            self.qasm_var.set(path)
            self.circuit_form.set("use_quantum", False)

    def source_circuit(self):
        """Circuit to schedule, as a circuits.QuantumCircuit."""
        if self.circuit_form.get("use_quantum"):
            return self.app.tabs["quantum"].circuit()
        return circuits.load(self.qasm_var.get())

    def transport_settings(self):
        """Mode and timing of the transports, which the co-design uses too."""
        p = self.priority
        description = p.get_str("mode")
        if description not in self.mode_names:
            raise ValueError("Choose a transport priority mode")
        return dict(mode=self.mode_names[description], max_speed=p.get("max_speed") * 1e6, min_speed=p.get("min_speed") * 1e6, min_duration=p.get("min_duration") * 1e-6,
                    heating_rate=p.get("heating"), n_bar_cooled=p.get("n_bar_cooled"), cooling_time=p.get("cooling_time") * 1e-6, recool=p.get("recool"))

    def timing(self):
        settings = self.transport_settings()
        p = self.priority
        if settings["min_speed"] <= 0 or settings["max_speed"] < settings["min_speed"]:
            raise ValueError("The fastest transport must be at least as fast as the slowest")
        return scheduling.Timing(
            single_qubit_time=p.get("single_qubit_time") * 1e-6, two_qubit_time=p.get("two_qubit_time") * 1e-6, detection_time=p.get("detection_time") * 1e-6,
            max_speed=settings["max_speed"], min_speed=settings["min_speed"], min_duration=settings["min_duration"], omega_axial=self.app.tabs["waveforms"].omega_axial(),
            mass=self.app.ion.mass, heating_rate=settings["heating_rate"], n_bar_cooled=settings["n_bar_cooled"], cooling_time=settings["cooling_time"], recool=settings["recool"],
        )

    def _trap_graph(self, n_ions):
        if self.zones.get("trap_zones"):
            if self.app.trap is None or not self.app.trap.zones():
                raise ValueError("The trap has no zones; generate a trap from a zone layout in the Trap tab or build one with zones")
            trap_graph = TrapGraph.from_trap(self.app.trap)
            return trap_graph, trap_graph.zone_types
        height = ION_HEIGHT
        if self.app.trap is not None and self.app.trap.solved and self.app.trap.n_rf > 0:
            try:
                height = self.app.trap.find_rf_null((0.0, 0.0, None))[2]
            except lib.TiOptError:
                pass
        trap_graph = TrapGraph.from_layout(self.zones.get_str("layout"), self.zones.get("zone_spacing"), n_ions, height)
        return trap_graph, trap_graph.zone_types

    def _prepare(self):
        """Parse the circuit and schedule its transports (on the worker thread)."""
        quantum_circuit = self.source_circuit()
        directory = Path(tempfile.mkdtemp())
        path = directory / "circuit.qasm"
        try:
            circuits.write_scheduler_qasm(quantum_circuit, path)
            circuit = Circuit(path)
        finally:
            path.unlink(missing_ok=True)
            directory.rmdir()
        trap_graph, types = self._trap_graph(circuit.n_ions)
        try:
            program = TransportProgram(trap_graph, circuit)
        except lib.TiOptError:
            raise ValueError("The circuit can't be scheduled on these zones (the loading and storage zones must hold every qubit)") from None
        return quantum_circuit, circuit, trap_graph, types, program

    def schedule(self):
        timing = self.timing()
        mode = self.transport_settings()["mode"]

        def work():
            quantum_circuit, circuit, trap_graph, types, program = self._prepare()
            plan = scheduling.plan(program, circuit, trap_graph.positions, types, timing, mode)
            return quantum_circuit, circuit, trap_graph, types, program, plan

        def done(result):
            quantum_circuit, self.circuit, self.trap_graph, types, self.program, self.plan = result
            self.show(quantum_circuit, self.trap_graph, types, self.plan)

        self.app.run_task("Scheduling transports", work, done)

    def compare(self):
        timing = self.timing()

        def work():
            quantum_circuit, circuit, trap_graph, types, program = self._prepare()
            return circuit, trap_graph, program, scheduling.compare(program, circuit, trap_graph.positions, types, timing)

        def done(result):
            self.circuit, self.trap_graph, self.program, plans = result
            self.plan = plans[self.transport_settings()["mode"]]
            lines = ["Mode                                        Duration   Moving   Excited   n̄ (2q)   n̄ (max)"]
            for mode, plan in plans.items():
                lines.append(f"{scheduling.MODES[mode][:42]:42s}  {plan.duration * 1e6:7.1f} µs  {plan.transport_time * 1e6:6.1f} µs  {plan.total_excitation:7.4f}  {plan.two_qubit_n_bar:7.4f}  {plan.max_n_bar:7.4f}")
            lines += ["", "Every mode runs the operations at the same times (set by the fastest transports), so they trade the motion of the ions at the gates "
                      "against time spent moving. Eager transports excite the least; transports just in time leave ions where they can be cooled."]
            self.results.delete("1.0", "end")
            self.results.insert("1.0", "\n".join(lines))

            ax_n, ax_time = self.plot.axes(1, 2)[0]
            names = [scheduling.MODES[m].split(" (")[0] for m in plans]
            x = np.arange(len(plans))
            ax_n.bar(x - 0.2, [p.two_qubit_n_bar for p in plans.values()], width=0.4, label="At two-qubit gates")
            ax_n.bar(x + 0.2, [p.max_n_bar for p in plans.values()], width=0.4, label="Largest")
            ax_n.set_xticks(x)
            ax_n.set_xticklabels(names, rotation=15, fontsize=7)
            ax_n.set_ylabel("Mean phonon number")
            ax_n.legend(fontsize=7)
            ax_n.set_title("Motion at the gates")
            ax_time.bar(x - 0.2, [p.transport_time * 1e6 for p in plans.values()], width=0.4, label="Time moving (all ions)")
            ax_time.bar(x + 0.2, [p.duration * 1e6 for p in plans.values()], width=0.4, label="Circuit")
            ax_time.set_xticks(x)
            ax_time.set_xticklabels(names, rotation=15, fontsize=7)
            ax_time.set_ylabel("Time (µs)")
            ax_time.legend(fontsize=7)
            ax_time.set_title("Time")
            self.plot.draw()

        self.app.run_task("Comparing transport priority modes", work, done)

    def show(self, quantum_circuit, trap_graph, types, plan):
        program = self.program
        zone_positions = trap_graph.positions
        layers = quantum_circuit.layers()
        text = [f"{len(quantum_circuit.operations)} gates on {quantum_circuit.n_qubits} qubits in {len(layers)} layers", plan.summary(), ""]
        text.append("Transports (ion: from -> to, leaving and arriving, quanta excited):")
        for t in plan.transports:
            start = int(np.argmin(np.linalg.norm(zone_positions - t.start, axis=1)))
            end = int(np.argmin(np.linalg.norm(zone_positions - t.destination, axis=1)))
            text.append(f"  ion {t.ion + 1}: {types[start]} -> {types[end]}, {t.depart * 1e6:.1f}-{t.arrive * 1e6:.1f} µs, {t.excitation:.2e}")
        text += ["", "Operations (start, ions, mean phonon number):"]
        for op in plan.operations:
            text.append(f"  {op.start * 1e6:7.1f} µs  {op.kind} on {', '.join(str(i + 1) for i in op.ions)} in {types[op.zone]} zone, n̄ = {op.n_bar:.3f}")
        self.results.delete("1.0", "end")
        self.results.insert("1.0", "\n".join(text))

        ax_gantt, ax_positions, ax_n = self.plot.axes(3, 1, sharex=True)[:, 0]
        for ion, start, end, kind in plan.segments:
            ax_gantt.barh(ion + 1, (end - start) * 1e6, left=start * 1e6, height=0.6, color=self.SEGMENT_COLORS[kind], edgecolor="none")
        for kind, color in self.SEGMENT_COLORS.items():
            ax_gantt.barh(0, 0, color=color, label=kind)
        ax_gantt.set_ylim(program.n_ions + 0.6, 0.4)
        ax_gantt.set_yticks(range(1, program.n_ions + 1))
        ax_gantt.set_ylabel("Ion")
        ax_gantt.legend(fontsize=6, ncols=3, loc="lower right")
        ax_gantt.set_title(scheduling.MODES[plan.mode])

        # Position of every ion along its transports
        for ion in range(program.n_ions):
            times, xs = [0.0], [program.initial_positions[ion, 0]]
            for t in sorted((t for t in plan.transports if t.ion == ion), key=lambda t: t.depart):
                u = np.linspace(0, 1, 20)
                times += list((t.depart + u * t.duration) * 1e6)
                xs += list(t.start[0] + scheduling.transport_profile(u) * (t.destination[0] - t.start[0]))
            times.append(plan.duration * 1e6)
            xs.append(xs[-1])
            ax_positions.plot(times, np.array(xs) + 2.0 * ion, label=f"Ion {ion + 1}")
        for z, position in enumerate(zone_positions):
            ax_positions.axhline(position[0], color="gray", linewidth=0.5, linestyle=":")
            ax_positions.text(plan.duration * 1e6, position[0], f" {types[z]}", va="center", fontsize=7, color="gray")
        ax_positions.set_ylabel("x (µm, offset per ion)")
        ax_positions.legend(fontsize=7, ncols=2)

        for ion in range(program.n_ions):
            ops = [op for op in plan.operations if ion in op.ions]
            ax_n.plot([op.start * 1e6 for op in ops], [op.n_bar for op in ops], marker="o", label=f"Ion {ion + 1}")
        ax_n.set_ylabel("n̄ at operations")
        ax_n.set_xlabel("Time (µs)")
        self.plot.draw()

    def _architecture(self, capacity=None):
        form = self.architecture
        topology = form.get_str("topology")
        capacity = form.get_int("capacity") if capacity is None else capacity
        if topology == "Linear":
            return qccd.Architecture.linear(form.get_int("n_traps"), capacity)
        if topology == "Ring":
            return qccd.Architecture.ring(form.get_int("n_traps"), capacity)
        if topology == "Grid":
            rows, cols = (int(v) for v in form.get_str("grid").split(","))
            return qccd.Architecture.grid(rows, cols, capacity)
        if self.app.trap is None or not self.app.trap.zones():
            raise ValueError("The trap has no zones; generate one from a zone layout first")
        return qccd.Architecture.from_trap(self.app.trap, capacity)

    def _qccd_models(self):
        form = self.architecture
        timing = qccd.TimingModel(gate_model=form.get_str("gate_model"), swap_type=form.get_str("swap"))
        noise = qccd.NoiseModel(recool=form.get("recool"))
        return timing, noise

    def compile_qccd(self):
        circuit = self.source_circuit()
        architecture = self._architecture()
        timing, noise = self._qccd_models()

        def work():
            return qccd.compile_circuit(circuit, architecture, timing, noise)

        def done(result):
            nodes = architecture.nodes
            text = result.summary() + ["", "Starting places: " + ", ".join(f"q{q}: {name}" for q, name in sorted(result.initial_mapping.items())), ""]
            for op in result.operations[:400]:
                text.append(f"{op.start:9.1f}-{op.end:9.1f} µs  {op.kind:18s} {op.label}" + (f"  F = {op.fidelity:.5f}" if op.fidelity < 1 else ""))
            if len(result.operations) > 400:
                text.append(f"... and {len(result.operations) - 400} more")
            self.results.delete("1.0", "end")
            self.results.insert("1.0", "\n".join(text))
            ax_graph, ax_gantt = self.plot.axes(1, 2, width_ratios=[1, 2])[0]
            for a, neighbors in architecture.neighbors.items():
                for b in neighbors:
                    if a < b:
                        pa, pb = nodes[a].position, nodes[b].position
                        ax_graph.plot([pa[0], pb[0]], [pa[1], pb[1]], color="gray", linewidth=1)
            for i, node in enumerate(nodes):
                trap_node = node.kind == "trap"
                ax_graph.plot(*node.position, marker="s" if trap_node else "o", markersize=9 if trap_node else 5,
                              color=("tab:purple" if node.gate_zone else "tab:blue") if trap_node else "black", linestyle="none")
                if trap_node:
                    ax_graph.annotate(node.name.split(" ")[0], node.position, fontsize=7, textcoords="offset points", xytext=(0, 7), ha="center")
            ax_graph.set_aspect("equal", adjustable="datalim")
            ax_graph.set_title("Traps (squares) and junctions")
            ax_graph.set_axis_off()
            kinds = {"two-qubit gate": "tab:purple", "single-qubit gate": "tab:blue", "measurement": "tab:green", "swap gate": "tab:pink", "ion swap": "tab:pink",
                     "split": "tab:orange", "merge": "tab:orange", "move": "tab:olive", "junction crossing": "tab:red", "cooling": "tab:cyan"}
            used = sorted({n for op in result.operations for n in op.nodes})
            rows = {n: k for k, n in enumerate(used)}
            for op in result.operations:
                for n in op.nodes:
                    ax_gantt.barh(rows[n], op.end - op.start, left=op.start, height=0.7, color=kinds.get(op.kind, "gray"), edgecolor="none")
            for kind, color in kinds.items():
                if any(op.kind == kind for op in result.operations):
                    ax_gantt.barh(0, 0, color=color, label=kind)
            ax_gantt.set_yticks(range(len(used)), [nodes[n].name.split(" ")[0] for n in used], fontsize=7)
            ax_gantt.invert_yaxis()
            ax_gantt.set_xlabel("Time (µs)")
            ax_gantt.legend(fontsize=6, ncols=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
            ax_gantt.set_title(f"{result.duration / 1e3:.2f} ms, fidelity {result.fidelity:.4f}")
            self.plot.draw()

        self.app.run_task("Compiling the circuit for the QCCD architecture", work, done)

    def explore_qccd(self):
        circuit = self.source_circuit()
        timing, noise = self._qccd_models()
        capacities = [c for c in (2, 3, 4, 6, 8, 12, 16, 24) if c <= max(circuit.n_qubits + 2, 4)]
        builders = {"chosen topology": lambda capacity: self._architecture(capacity)}

        def work():
            rows = []
            for model in gates.GATE_MODELS:
                for capacity in capacities:
                    row = dict(gate_model=model, capacity=capacity)
                    try:
                        result = qccd.compile_circuit(circuit, builders["chosen topology"](capacity), qccd.TimingModel(gate_model=model, swap_type=timing.swap_type), noise)
                        row.update(fidelity=result.fidelity, duration=result.duration, shuttles=result.counts.get("split", 0))
                    except ValueError:
                        row.update(fidelity=np.nan, duration=np.nan, shuttles=np.nan)
                    rows.append(row)
            return rows

        def done(rows):
            ax_fidelity, ax_duration = self.plot.axes(1, 2)[0]
            for model in gates.GATE_MODELS:
                subset = [r for r in rows if r["gate_model"] == model]
                ax_fidelity.plot([r["capacity"] for r in subset], [r["fidelity"] for r in subset], marker="o", label=model)
                ax_duration.plot([r["capacity"] for r in subset], [r["duration"] / 1e3 for r in subset], marker="o", label=model)
            ax_fidelity.set_xlabel("Ions per trap")
            ax_fidelity.set_ylabel("Estimated fidelity of the circuit")
            ax_fidelity.legend(fontsize=8)
            ax_duration.set_xlabel("Ions per trap")
            ax_duration.set_ylabel("Duration (ms)")
            self.plot.draw()
            text = ["capacity  gates  fidelity  duration (ms)  splits"]
            text += [f"{r['capacity']:8d}  {r['gate_model']:5s}  {r['fidelity']:.4f}  {r['duration'] / 1e3:12.3f}  {r['shuttles']}" for r in rows]
            self.results.delete("1.0", "end")
            self.results.insert("1.0", "\n".join(text))

        self.app.run_task("Exploring trap capacities and gate models", work, done)

    def build_waveform(self):
        trap = self.require_trap()
        if self.plan is None:
            raise ValueError("Schedule a circuit first")
        ion = self.app.ion
        n_steps = self.actions.get_int("n_steps")
        plan, program = self.plan, self.program
        omega_axial = self.app.tabs["waveforms"].omega_axial()

        def work():
            return scheduling.waveform_from_plan(trap, ion, plan, program, n_steps, omega_axial)

        def done(waveform):
            self.app.set_waveform(waveform, omega_axial)
            self.app.tabs["waveforms"].show(waveform, omega_axial)
            self.app.notebook.select(self.app.tabs["waveforms"])

        self.app.run_task("Building waveform for the plan", work, done)


# ---------------------------------------------------------------------------------------------------------------------
# Co-design
# ---------------------------------------------------------------------------------------------------------------------


class CodesignTab(Tab):
    """Tab for co-design, either of the whole system from the inputs of every tab at once (see system.py), or with the
    flow of the C library, which makes a basic design of a linear trap and then passes which optimize the trap,
    photonics, transport, and waveforms a little, first each based on its own figures of merit and then all based on the
    metrics of the whole circuit.
    """

    FLOWS = ("Full co-design (inputs from every tab)", "Linear trap flow (C library)")

    # Environment entries: (key, label, unit, scale from the displayed value to the library's units)
    ENVIRONMENT = (
        ("n_segments", "DC segments per side", "", 1),
        ("max_voltage", "DC voltage limit", "V", 1),
        ("qubit_power", "729 nm power", "mW", 1e-3),
        ("detection_power", "397 nm power", "µW", 1e-6),
        ("field_noise", "Field noise at 1 MHz", "V²/m²/Hz", 1),
        ("dephasing_rate", "Qubit dephasing rate", "1/s", 1),
        ("n_bar_cooled", "Mean phonons after cooling", "", 1),
        ("position_drift", "Ion position drift", "µm", 1),
        ("dark_count_density", "Dark counts per area", "1/s/µm²", 1),
    )

    # Groups which are optimized by default (shapes take longest, so they are left to the user)
    DEFAULT_GROUPS = ("trap", "waveguides", "bends", "gratings", "pulses", "transport", "waveforms", "detection")

    def __init__(self, app):
        super().__init__(app, "Co-design")

        selector = self.form("Co-design")
        self.flow_var = selector.combo("flow", "Flow", self.FLOWS, self.FLOWS[0])
        self.flow_var.trace_add("write", lambda *args: self.show_flow())

        # Full co-design
        sources = self.form("Inputs from the other tabs")
        sources.buttons(("Gather inputs", self.gather))
        self.sources = sources.text(9)
        set_text(self.sources, "The trap and ion (Trap and Trap builder tabs), the circuit and pulse shape (Quantum tab), the chip stack, waveguide, "
                               "bend, and qubit grating (Photonics tab), the transport priority (Scheduling tab), the axial frequency and voltage "
                               "limit (Waveforms tab), and the DC voltage noise (Dynamics tab) are gathered when the co-design starts.")

        groups = self.form("Optimize together")
        for key, label in system.GROUPS.items():
            groups.check(key, label, key in self.DEFAULT_GROUPS)
        self.groups = groups

        extra = self.form("System")
        extra.entry("route_length", "Waveguide from the chip edge to the grating", 5000, "µm")
        extra.entry("n_bends", "90° bends on the way", 2)
        extra.entry("coupling_loss", "Coupling loss at the chip edge", 3, "dB")
        extra.entry("qubit_power", "Qubit light at the chip edge", 1, "mW")
        extra.entry("detection_power", "Detection light at the grating", 10, "µW")
        extra.entry("detection_time", "Detection time", 50, "µs")
        extra.entry("detector_size", "Detector size", 50, "µm")
        extra.entry("detector_efficiency", "Detector efficiency", 0.8)
        extra.entry("dark_count_density", "Dark counts per area", 1e-3, "1/s/µm²")
        extra.entry("dephasing_rate", "Qubit dephasing rate", 100, "1/s")
        extra.entry("field_noise", "Field noise at 1 MHz", 1e-12, "V²/m²/Hz")
        extra.entry("field_noise_exponent", "Field noise exponent", 1.0)
        extra.entry("position_drift", "Ion position drift", 0.1, "µm")
        extra.entry("max_mathieu_q", "Largest Mathieu q", 0.4)
        extra.entry("min_trap_depth", "Smallest trap depth", 20, "meV")
        extra.entry("path_weight", "Weight on the junction barrier", 0.01, "per meV")
        extra.check("best_mode", "Try every transport priority mode", False)
        self.system_form = extra

        run = self.form("Run")
        run.entry("evaluations", "Evaluations", 300)
        run.combo("method", "Optimizer", ("Powell", "Nelder-Mead"), "Powell")
        run.check("simulate", "Simulate the two-qubit gates at every evaluation (slow)", False)
        run.buttons(("Run", self.run_system), ("Stop", self.stop_system), ("Send to tabs", self.send_to_tabs))
        self.run_form = run

        system_results = ttk.LabelFrame(self.controls, text="Best design", padding=6)
        self.system_results = ScrolledText(system_results, height=22, width=44, wrap="none")
        self.system_results.pack(fill="both", expand=True)
        self.system_frame = system_results
        self.full_forms = (sources, groups, extra, run, system_results)

        # Flow of the C library
        circuit = self.form("Circuit")
        circuit.check("use_quantum", "Use the circuit in the Quantum tab", True)
        self.qasm_var = circuit.entry("qasm", "Or an OpenQASM file", EXAMPLES / "circuits" / "bell.qasm")
        circuit.entry("layout", "Zone layout", lib.DEFAULT_LAYOUT)
        circuit.buttons(("Browse...", self.browse))
        self.circuit = circuit

        environment = self.form("Hardware and noise")
        defaults = lib.default_codesign_environment()
        for key, label, unit, scale in self.ENVIRONMENT:
            environment.entry(key, label, f"{defaults[key] / scale:g}", unit)
        self.environment = environment

        flow = self.form("Flow")
        flow.entry("n_passes", "Passes", 2)
        flow.entry("max_evaluations", "Evaluations per stage", 10)
        flow.buttons(("Basic design", self.basic_design), ("Run flow", self.run_flow), ("Use trap", self.use_trap))
        self.flow = flow

        results = ttk.LabelFrame(self.controls, text="Design", padding=6)
        self.results = ScrolledText(results, height=18, width=40, wrap="none")
        self.results.pack(fill="both", expand=True)
        self.c_forms = (circuit, environment, flow, results)

        self.codesign = None
        self.records = []
        self.system = None
        self.system_trap = None
        self.shown_evaluations = 0
        self.show_flow()

    def show_flow(self):
        full = self.flow_var.get() == self.FLOWS[0]
        for form in self.full_forms + self.c_forms:
            form.pack_forget()
        for form in self.full_forms if full else self.c_forms:
            expand = form in (self.system_frame, self.c_forms[-1])
            form.pack(fill="both" if expand else "x", expand=expand, pady=(0, 6), padx=(0, 4))
        if full:
            self.plot.message("Co-design the trap, photonics, pulses, transport, and wells together from the inputs of every tab")
        else:
            self.plot.message("Make a basic design for a circuit, then run the co-design flow")

    # Full co-design -----------------------------------------------------------------------------------------------

    def gather_inputs(self):
        """Collect SystemInputs from every tab (on the main thread), and describe where each came from."""
        app = self.app
        trap = app.trap
        if trap is None or not trap.solved:
            raise ValueError("Generate, build, or load a trap with zones first (Trap or Trap builder tab)")
        quantum = app.tabs["quantum"]
        photonics = app.tabs["photonics"]
        scheduling_tab = app.tabs["scheduling"]
        waveforms = app.tabs["waveforms"]
        dynamics = app.tabs["dynamics"]
        trap_tab = app.tabs["trap"]
        circuit = quantum.circuit()
        stack = photonics.chip_stack()
        width, thickness = photonics.straight.get("width"), photonics.straight.get("thickness")
        bend = photonics.bend_design()
        grating = photonics.grating_design(thickness)
        shape, ramp = quantum.pulse_shape()
        transport = scheduling_tab.transport_settings()
        f = self.system_form
        V_RF = trap.V_RF if trap.V_RF > 0 else trap_tab.rf.get("V_RF")
        Omega_RF = trap.Omega_RF if trap.Omega_RF > 0 else TWO_PI * trap_tab.rf.get("f_RF") * 1e6
        shape_form = trap_tab.shape_form
        center_text = shape_form.get_str("center").strip()
        center = None if center_text.lower() in ("", "auto") else tuple(float(v) for v in center_text.split(","))
        inputs = system.SystemInputs(
            trap=trap, ion=app.ion, circuit=circuit, V_RF=V_RF, Omega_RF=Omega_RF, stack=stack, waveguide_width=width, waveguide_thickness=thickness,
            route_length=f.get("route_length"), n_bends=f.get_int("n_bends"), bend_radius=bend["radius"], euler_fraction=bend["euler_fraction"], bend_width=bend["bend_width"],
            taper_length=bend["taper_length"], taper_profile=bend["taper_profile"], coupling_loss=f.get("coupling_loss"), grating=grating, qubit_power=f.get("qubit_power") * 1e-3,
            pulse_shape=shape, ramp=ramp, n_bar_cooled=transport["n_bar_cooled"], dephasing_rate=f.get("dephasing_rate"),
            transport_mode="best" if f.get("best_mode") else transport["mode"], max_speed=transport["max_speed"], min_speed=transport["min_speed"], recool=transport["recool"],
            cooling_time=transport["cooling_time"], omega_axial=waveforms.omega_axial(), max_voltage=waveforms.transport_form.get("max_voltage"),
            voltage_noise=dynamics.noise.get("S_V"), field_noise=f.get("field_noise"), field_noise_exponent=f.get("field_noise_exponent"),
            detection_power=f.get("detection_power") * 1e-6, detection_time=f.get("detection_time") * 1e-6, detector_size=f.get("detector_size"),
            detector_efficiency=f.get("detector_efficiency"), dark_count_density=f.get("dark_count_density"), position_drift=f.get("position_drift"),
            max_mathieu_q=f.get("max_mathieu_q"), min_trap_depth=f.get("min_trap_depth") * 1e-3, shape_center=center, shape_radius=shape_form.get("radius"),
            max_shift=shape_form.get("max_shift"), shape_symmetry=shape_form.get_str("symmetry"), path_weight=f.get("path_weight"),
        )
        zones = ", ".join(zone.types for zone in trap.zones()) or "none"
        kind = "uniform" if grating.uniform else ("apodized and chirped" if grating.fill_start is not None and grating.chirp else ("apodized" if grating.fill_start is not None else "chirped"))
        description = [
            f"Trap: {trap.n_rf} RF and {trap.n_dc} DC electrodes, zones {zones}, RF {V_RF:.3g} V at {Omega_RF / TWO_PI / 1e6:.3g} MHz, {app.ion.element}",
            f"Circuit: {circuit.n_qubits} qubits, {len(circuit.operations)} gates (Quantum tab)",
            f"Gate pulses: {shape}" + (f" with {ramp * 1e6:.3g} µs ramps" if shape in ("trapezoid", "cosine") else (f" {ramp:.3g} of their duration wide" if shape == "gaussian" else "")) + " (Quantum tab)",
            f"Waveguide: {width:.3g} x {thickness:.3g} µm at {stack.wavelength:.3g} µm, bends of {bend['radius']:.3g} µm (Photonics tab)",
            f"Qubit grating: {grating.n_teeth} {kind} teeth (Photonics tab)",
            f"Transport: {scheduling.MODES[transport['mode']] if not f.get('best_mode') else 'every mode'}, up to {transport['max_speed'] / 1e6:.3g} m/s (Scheduling tab)",
            f"Wells: {inputs.omega_axial / TWO_PI / 1e6:.3g} MHz, at most {inputs.max_voltage:.3g} V (Waveforms tab)",
            f"DC voltage noise: {inputs.voltage_noise:.3g} V²/Hz (Dynamics tab)",
        ]
        return inputs, description

    def gather(self):
        _, description = self.gather_inputs()
        set_text(self.sources, "\n".join(description))

    def selected_groups(self):
        groups = [key for key in system.GROUPS if self.groups.get(key)]
        if not groups:
            raise ValueError("Choose at least one group of variables to optimize")
        return groups

    def run_system(self):
        inputs, description = self.gather_inputs()
        set_text(self.sources, "\n".join(description))
        groups = self.selected_groups()
        max_evaluations = self.run_form.get_int("evaluations")
        method = self.run_form.get_str("method")
        simulate = self.run_form.get("simulate")
        if self.system is not None:
            self.system.close()
            self.system = None
        if self.system_trap is not None:
            self.system_trap.close()
            self.system_trap = None
        self.shown_evaluations = 0
        holder = {}

        def work():
            codesign = system.SystemCodesign(inputs, groups, simulate)
            holder["codesign"] = codesign
            best = codesign.run(max_evaluations, method)
            return codesign, best, codesign.best_trap()

        def done(result):
            self.system, best, self.system_trap = result
            self.show_system(final=True)

        self.app.run_task("Co-designing the system", work, done)
        self.holder = holder
        if not self.app.synchronous:
            self.after(1000, self._poll_system)

    def _poll_system(self):
        """Show the progress of the co-design while it runs on a background thread."""
        codesign = getattr(self, "holder", {}).get("codesign")
        if codesign is not None and len(codesign.history) > self.shown_evaluations + 4:
            self.system = codesign
            self.show_system(final=False)
        if self.app.busy:
            self.after(1000, self._poll_system)

    def stop_system(self):
        codesign = getattr(self, "holder", {}).get("codesign")
        if codesign is not None:
            codesign.stop()

    def best_evaluation(self):
        """Best evaluation of the full co-design, or None."""
        return self.system.best if self.system is not None else None

    def show_system(self, final):
        codesign = self.system
        history = list(codesign.history)
        if not history:
            return
        self.shown_evaluations = len(history)
        best = codesign.best
        initial = codesign.initial
        lines = system.format_evaluation(best, initial)
        lines += ["", "Variables (start -> best):"]
        for variable in codesign.variables:
            start = getattr(codesign.inputs, variable.name)
            value = best.parameters[variable.name]
            lines.append(f"  {variable.label:34s} {start:.4g} -> {value:.4g}")
        if codesign.n_shape:
            lines.append(f"  Electrode shapes                   {codesign.n_shape} control values, largest {np.max(np.abs(best.parameters['shape'])) * codesign.inputs.max_shift:.2f} µm")
        lines += ["", "Contributions to -log(circuit fidelity):"]
        lines += [f"  {kind:26s} {value:.4g}" for kind, value in best.budget.items()]
        if not final:
            lines.insert(0, f"Running: {len(history)} evaluations so far\n")
        self.system_results.delete("1.0", "end")
        self.system_results.insert("1.0", "\n".join(lines))

        (ax_history, ax_budget), (ax_trap, ax_variables) = self.plot.axes(2, 2)
        objectives = np.array([e.objective for e in history])
        infidelities = np.array([max(1 - e.metrics.get("circuit_fidelity", 0.0), 1e-9) for e in history])
        feasible = np.array([e.feasible for e in history])
        steps = np.arange(len(history))
        ax_history.semilogy(steps[feasible], infidelities[feasible], linestyle="none", marker=".", markersize=3, color="0.6", label="Each design")
        if (~feasible).any():
            ax_history.semilogy(steps[~feasible], infidelities[~feasible], linestyle="none", marker="x", markersize=3, color="tab:red", label="Beyond a limit")
        # The design with the lowest objective so far, which counts the limits too
        best_so_far = [int(np.argmin(objectives[:i + 1])) for i in steps]
        ax_history.semilogy(steps, infidelities[best_so_far], color="tab:blue", label="Best so far")
        ax_history.set_xlabel("Evaluation")
        ax_history.set_ylabel("1 - circuit fidelity")
        ax_history.legend(fontsize=7)
        ax_history.set_title("Circuit infidelity")

        kinds = list(best.budget) or list(initial.budget)
        x = np.arange(len(kinds))
        ax_budget.bar(x - 0.2, [initial.budget.get(k, 0.0) for k in kinds], width=0.4, label="Start")
        ax_budget.bar(x + 0.2, [best.budget.get(k, 0.0) for k in kinds], width=0.4, label="Best")
        ax_budget.set_xticks(x)
        ax_budget.set_xticklabels([k.replace(" ", "\n", 1) for k in kinds], fontsize=7)
        ax_budget.set_ylabel("-log(fidelity)")
        ax_budget.legend(fontsize=7)
        ax_budget.set_title("Error budget")

        trap = self.system_trap if final and self.system_trap is not None else codesign.inputs.trap
        for e, polygon in enumerate(trap.electrode_polygons()):
            color = "tab:orange" if trap.is_rf(e) else "tab:blue"
            ax_trap.add_patch(PolygonPatch(polygon[:, :2], closed=True, facecolor=colors.to_rgba(color, 0.25), edgecolor=color, linewidth=0.5))
        draw_photonics(ax_trap, trap)
        draw_zones(ax_trap, trap.zones())
        ax_trap.autoscale_view()
        ax_trap.set_aspect("equal")
        ax_trap.set_title("Co-designed trap" if final else "Trap")

        names = [v.label.split(" (")[0] for v in codesign.variables]
        starts = [v.to_unit(getattr(codesign.inputs, v.name)) for v in codesign.variables]
        values = [v.to_unit(best.parameters[v.name]) for v in codesign.variables]
        y = np.arange(len(names))
        ax_variables.barh(y + 0.2, starts, height=0.4, label="Start", color="0.7")
        ax_variables.barh(y - 0.2, values, height=0.4, label="Best", color="tab:green")
        ax_variables.set_yticks(y)
        ax_variables.set_yticklabels(names, fontsize=6)
        ax_variables.set_xlim(0, 1)
        ax_variables.set_xlabel("Position within the bounds")
        ax_variables.invert_yaxis()
        ax_variables.legend(fontsize=7)
        ax_variables.set_title("Variables")
        self.plot.draw()

    def send_to_tabs(self):
        """Put the best design into the tabs it came from."""
        if self.system is None or self.system.best is None:
            raise ValueError("Run the full co-design first")
        best = self.system.best.parameters
        app = self.app
        photonics = app.tabs["photonics"]
        names = {v.name for v in self.system.variables}
        if "waveguide_width" in names:
            photonics.straight.set("width", f"{best['waveguide_width']:.4g}")
            photonics.straight.set("thickness", f"{best['waveguide_thickness']:.4g}")
        if "bend_radius" in names:
            photonics.bend.set("radius", f"{best['bend_radius']:.4g}")
            photonics.bend.set("euler_fraction", f"{best['euler_fraction']:.4g}")
            photonics.bend.set("taper_length", f"{best['taper_length']:.4g}")
        if "grating_fill" in names:
            period = self.system.best.metrics.get("grating_period", photonics.coupler.get("period"))
            photonics.coupler.set("period", f"{period:.4g}")
            photonics.coupler.set("tooth_width", f"{best['grating_fill'] * period:.4g}")
            photonics.coupler.set("n_teeth", str(int(round(best["grating_teeth"]))))
            photonics.coupler.set("angular_width", f"{best['angular_width']:.4g}")
            if "fill_start" in names:
                photonics.coupler.set("fill_start", f"{best['fill_start']:.4g}")
        if "ramp" in names:
            app.tabs["quantum"].presets.set("ramp", f"{best['ramp'] * 1e6:.4g}")
        if "gaussian_width" in names:
            app.tabs["quantum"].presets.set("gaussian_width", f"{best['gaussian_width']:.4g}")
        if "max_speed" in names:
            app.tabs["scheduling"].priority.set("max_speed", f"{best['max_speed'] / 1e6:.4g}")
        if "omega_axial" in names:
            app.tabs["waveforms"].transport_form.set("f_axial", f"{best['omega_axial'] / TWO_PI / 1e6:.4g}")
        if "detection_time" in names:
            self.system_form.set("detection_time", f"{best['detection_time'] * 1e6:.4g}")
            self.system_form.set("detector_size", f"{best['detector_size']:.4g}")
        m = self.system.best.metrics
        if "single_qubit_time" in m:
            schedule = app.tabs["scheduling"].priority
            schedule.set("single_qubit_time", f"{m['single_qubit_time'] * 1e6:.4g}")
            schedule.set("two_qubit_time", f"{(m['two_qubit_time'] + 4 * m['single_qubit_time']) * 1e6:.4g}")
            schedule.set("detection_time", f"{best['detection_time'] * 1e6:.4g}")
            schedule.set("heating", f"{m['heating_rate']:.4g}")
        if self.system_trap is not None:
            trap = self.system_trap
            self.system_trap = None
            app.set_trap(trap)
            app.tabs["trap"].rf.set("V_RF", f"{trap.V_RF:.4g}")
            app.tabs["trap"].rf.set("f_RF", f"{trap.Omega_RF / TWO_PI / 1e6:.4g}")
        app.set_status("Sent the co-designed system to the Trap, Photonics, Quantum, Scheduling, and Waveforms tabs")

    # Flow of the C library ----------------------------------------------------------------------------------------

    def browse(self):
        path = filedialog.askopenfilename(parent=self, title="Open OpenQASM file", initialdir=EXAMPLES / "circuits", filetypes=[("OpenQASM", "*.qasm"), ("All files", "*")])
        if path:
            self.qasm_var.set(path)
            self.circuit.set("use_quantum", False)

    def _environment(self):
        values = {}
        for key, label, unit, scale in self.ENVIRONMENT:
            values[key] = self.environment.get_int(key) if key == "n_segments" else self.environment.get(key) * scale
        return values

    def basic_design(self):
        path = self.qasm_var.get()
        layout = self.circuit.get_str("layout")
        element = self.app.ion.element
        environment = self._environment()
        quantum_circuit = self.app.tabs["quantum"].circuit() if self.circuit.get("use_quantum") else None

        def work():
            if quantum_circuit is None:
                return lib.Codesign(path, element, environment, layout=layout)
            # The C library reads the circuit from a file when the co-design is made
            directory = Path(tempfile.mkdtemp())
            circuit_path = directory / "circuit.qasm"
            try:
                circuits.write_scheduler_qasm(quantum_circuit, circuit_path)
                return lib.Codesign(circuit_path, element, environment, layout=layout)
            finally:
                circuit_path.unlink(missing_ok=True)
                directory.rmdir()

        def done(codesign):
            if self.codesign is not None:
                self.codesign.close()
            self.codesign = codesign
            self.records = codesign.records()
            self.show()

        self.app.run_task("Making basic design", work, done)

    def run_flow(self):
        if self.codesign is None:
            raise ValueError("Make a basic design first")
        codesign = self.codesign
        codesign.environment = self._environment()
        n_passes = self.flow.get_int("n_passes")
        max_evaluations = self.flow.get_int("max_evaluations")

        def work():
            codesign.run(n_passes, max_evaluations)
            return codesign

        def done(codesign):
            self.records = codesign.records()
            self.show()

        self.app.run_task("Running co-design flow", work, done)
        if not self.app.synchronous:
            self.after(500, self._poll_records)

    def _poll_records(self):
        """Show the stages as they finish, while the flow runs on a background thread."""
        if self.codesign is None:
            return
        if self.codesign.n_records > len(self.records):
            self.records += self.codesign.records(len(self.records))
            self.show(live=True)
        if self.app.busy:
            self.after(500, self._poll_records)

    def use_trap(self):
        if self.codesign is None:
            raise ValueError("Make a basic design first")
        codesign = self.codesign

        def work():
            path = Path(tempfile.mkdtemp()) / "codesign.trap"
            codesign.save_trap(path)
            trap = Trap.from_file(path)
            trap.solve("analytical")
            path.unlink()
            path.parent.rmdir()
            return trap

        def done(trap):
            self.app.set_trap(trap)
            self.app.notebook.select(self.app.tabs["trap"])

        self.app.run_task("Loading co-designed trap", work, done)

    def show(self, live=False):
        records = self.records
        if not records:
            return
        last = records[-1]
        metrics = last.metrics
        parameters = last.parameters

        text = [f"After pass {last.pass_index}, stage '{last.stage}' ({len(records)} records)", ""]
        text += [
            f"Circuit fidelity      {metrics['circuit_fidelity']:.6f}",
            f"Circuit duration      {metrics['circuit_duration'] * 1e3:.3f} ms",
            f"Two-qubit gate        {metrics['two_qubit_time'] * 1e6:.1f} µs, F = {metrics['two_qubit_fidelity']:.6f}",
            f"Single-qubit gate     {metrics['single_qubit_time'] * 1e6:.2f} µs, F = {metrics['single_qubit_fidelity']:.6f}",
            f"Detection             F = {metrics['detection_fidelity']:.6f}",
            f"Radial frequency      {metrics['radial_frequency'] / TWO_PI / 1e6:.3f} MHz (q = {metrics['mathieu_q']:.3f})",
            f"Trap depth            {metrics['trap_depth'] * 1e3:.1f} meV",
            f"Heating rate          {metrics['heating_rate']:.1f} quanta/s",
            f"Lamb-Dicke parameter  {metrics['lamb_dicke_parameter']:.4f}",
            f"Shuttle excitation    {metrics['shuttle_excitation']:.4f} quanta",
            f"Peak voltage          {metrics['peak_voltage']:.2f} V",
            "",
            "Parameters:",
            f"  Segment width       {parameters['segment_width']:.1f} µm",
            f"  RF                  {parameters['V_RF']:.1f} V at {parameters['Omega_RF'] / TWO_PI / 1e6:.1f} MHz",
            f"  Qubit grating       {parameters['grating_length']:.1f} x {parameters['grating_width']:.1f} µm",
            f"  Rabi fraction       {parameters['rabi_fraction']:.3f}",
            f"  Detector            {parameters['detector_size']:.1f} µm, {parameters['detection_time'] * 1e6:.1f} µs",
            f"  Zone spacing        {parameters['zone_spacing']:.1f} µm",
            f"  Shuttle duration    {parameters['shuttle_duration'] * 1e6:.2f} µs",
            f"  Axial frequency     {parameters['omega_axial'] / TWO_PI / 1e6:.3f} MHz",
            f"  Roughness weight    1E{parameters['log_roughness_weight']:.2f}",
        ]
        self.results.delete("1.0", "end")
        self.results.insert("1.0", "\n".join(text))

        axes = self.plot.axes(2, 2)
        steps = np.arange(len(records))
        labels = [f"{r.pass_index}:{r.stage}" for r in records]
        infidelity = np.array([max(1 - r.metrics["circuit_fidelity"], 1e-12) for r in records])

        ax = axes[0, 0]
        ax.plot(steps, infidelity, marker="o")
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.set_ylabel("Circuit infidelity")
        ax.set_title("Circuit infidelity after each stage")

        ax = axes[0, 1]
        ax.plot(steps, [r.metrics["circuit_duration"] * 1e3 for r in records], marker="o", color="tab:orange")
        ax.set_ylabel("Circuit duration (ms)")
        ax.set_title("Circuit duration")

        ax = axes[1, 0]
        for key, label in (("two_qubit_fidelity", "Two-qubit gate"), ("single_qubit_fidelity", "Single-qubit gate"), ("detection_fidelity", "Detection")):
            ax.semilogy(steps, [max(1 - r.metrics[key], 1e-12) for r in records], marker="o", label=label)
        ax.set_ylabel("Infidelity")
        ax.set_title("Operation infidelities")
        ax.legend(fontsize=7)

        for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            # Shade the QC-aware passes
            for r, record in enumerate(records):
                if record.pass_index > 0:
                    ax.axvspan(r - 0.5, r + 0.5, color="tab:green", alpha=0.06, linewidth=0)
            ax.set_xticks(steps)
            ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)

        ax = axes[1, 1]
        if not live and self.codesign is not None:
            trap = self.codesign.trap()
            for e, polygon in enumerate(trap.electrode_polygons()):
                color = "tab:orange" if trap.is_rf(e) else "tab:blue"
                ax.add_patch(PolygonPatch(polygon[:, :2], closed=True, facecolor=colors.to_rgba(color, 0.25), edgecolor=color, linewidth=0.5))
            draw_photonics(ax, trap)
            zones = trap.zones()
            draw_zones(ax, zones)
            reach = max([260.0] + [abs(zone.position[0]) + 60.0 for zone in zones])
            ax.set_xlim(-reach, reach)
            ax.set_ylim(-200, 200)
            ax.set_aspect("equal")
            ax.legend(fontsize=6, loc="lower left")
            ax.set_title("Co-designed trap")
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("y (µm)")
        else:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Running...", ha="center", va="center", transform=ax.transAxes, color="gray")
        self.plot.draw()


# ---------------------------------------------------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------------------------------------------------


class TiOptApp(tk.Tk):
    """The tiopt GUI, where computations run on the calling thread and errors propagate if synchronous is True (which
    the tests use).
    """

    def __init__(self, capture_output=True, synchronous=False):
        super().__init__()
        self.title("tiopt")
        self.geometry("1400x900")
        self.minsize(1000, 650)
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        self.synchronous = synchronous
        self.busy = False
        self.results = queue.Queue()

        self.trap = None
        self.ion = Ion("40Ca+")
        self.waveform = None
        self.omega_axial = AXIAL_FREQUENCY
        self.trajectory = None
        self.crystal = None
        self.crystal_k = None

        toolbar = ttk.Frame(self, padding=(4, 2))
        toolbar.pack(fill="x", side="top")
        ttk.Label(toolbar, text="Threads").pack(side="left")
        self.threads_var = tk.StringVar(value=str(lib.get_num_threads()))
        ttk.Spinbox(toolbar, from_=1, to=max(os.cpu_count() or 1, 1), textvariable=self.threads_var, width=4).pack(side="left", padx=(4, 12))
        ttk.Label(toolbar, text="Device").pack(side="left")
        self.device_var = tk.StringVar(value=lib.get_device())
        ttk.Combobox(toolbar, textvariable=self.device_var, values=("cpu", "opencl"), state="readonly", width=8).pack(side="left", padx=(4, 4))
        ttk.Button(toolbar, text="Apply", command=self.apply_acceleration).pack(side="left", padx=(4, 12))
        self.device_label = ttk.Label(toolbar, text="", foreground="gray")
        self.device_label.pack(side="left")

        panes = ttk.PanedWindow(self, orient="vertical")
        panes.pack(fill="both", expand=True)

        self.notebook = ttk.Notebook(panes)
        panes.add(self.notebook, weight=4)

        log_frame = ttk.Frame(panes)
        panes.add(log_frame, weight=1)
        header = ttk.Frame(log_frame)
        header.pack(fill="x")
        ttk.Label(header, text="Log").pack(side="left")
        ttk.Button(header, text="Clear", command=lambda: self.log.delete("1.0", "end")).pack(side="right")
        self.log = ScrolledText(log_frame, height=8, font=("TkFixedFont", 9))
        self.log.pack(fill="both", expand=True)

        status = ttk.Frame(self)
        status.pack(fill="x", side="bottom")
        self.status = ttk.Label(status, text="Ready", anchor="w")
        self.status.pack(side="left", fill="x", expand=True, padx=4)
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=160)
        self.progress.pack(side="right", padx=4, pady=2)

        self.tabs = {
            "trap": TrapTab(self),
            "builder": BuilderTab(self),
            "electrostatics": ElectrostaticsTab(self),
            "waveforms": WaveformTab(self),
            "crystals": CrystalsTab(self),
            "dynamics": DynamicsTab(self),
            "photonics": PhotonicsTab(self),
            "quantum": QuantumTab(self),
            "scheduling": SchedulingTab(self),
            "codesign": CodesignTab(self),
        }

        self.capture = None
        if capture_output:
            self.capture = OutputCapture()
            self.capture.start()
            self.after(100, self._poll_output)

        self.report_callback_exception = self._report_exception
        self.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.after(50, self._poll_results)

    # Acceleration -----------------------------------------------------------------------------------------------

    def apply_acceleration(self):
        threads = int(self.threads_var.get())
        if threads < 1:
            raise ValueError("Use at least one thread")
        lib.set_num_threads(threads)
        device = self.device_var.get()
        try:
            lib.set_device(device)
        except lib.TiOptError:
            self.device_var.set("cpu")
            lib.set_device("cpu")
            raise ValueError("No OpenCL device with double precision was found, so computations stay on the CPU (see the README for setting up GPUs)") from None
        name = f" ({lib.opencl_device_name()})" if device == "opencl" else ""
        self.device_label.configure(text=f"{lib.get_num_threads()} threads, {device}{name}")
        self.set_status(f"Running on {lib.get_num_threads()} threads and the {device}{name}")

    # Tasks ------------------------------------------------------------------------------------------------------

    def run_task(self, description, work, on_done=None):
        """Run work() on a background thread, then on_done(result) on the main thread."""
        if self.busy:
            messagebox.showinfo("Busy", "Wait for the current computation to finish", parent=self)
            return
        self.busy = True
        self.set_status(f"{description}...")
        self.progress.start(10)

        def target():
            try:
                result, error = work(), None
            except Exception as exc:
                result, error = None, (exc, traceback.format_exc())
            self.results.put((description, on_done, result, error))

        if self.synchronous:
            target()
            self._handle_results(raise_errors=True)
        else:
            threading.Thread(target=target, daemon=True).start()

    def _handle_results(self, raise_errors=False):
        while True:
            try:
                description, on_done, result, error = self.results.get_nowait()
            except queue.Empty:
                break
            self.busy = False
            self.progress.stop()
            lib.flush_c_output()
            if error is not None:
                exc, trace = error
                self.set_status(f"{description} failed: {exc}")
                self.append_log(trace)
                if raise_errors:
                    raise exc
                messagebox.showerror("Error", str(exc), parent=self)
                continue
            self.set_status(f"{description}: done")
            if on_done is not None:
                try:
                    on_done(result)
                except Exception as exc:
                    if raise_errors:
                        raise
                    self._report_exception(type(exc), exc, exc.__traceback__)

    def _poll_results(self):
        self._handle_results()
        self.after(50, self._poll_results)

    def _poll_output(self):
        chunks = []
        while self.capture is not None:
            try:
                chunks.append(self.capture.queue.get_nowait())
            except queue.Empty:
                break
        if chunks:
            self.append_log("".join(chunks))
        self.after(100, self._poll_output)

    def append_log(self, text):
        self.log.insert("end", text)
        # Keep the log from growing without bound
        lines = int(self.log.index("end-1c").split(".")[0])
        if lines > 5000:
            self.log.delete("1.0", f"{lines - 5000}.0")
        self.log.see("end")

    def _report_exception(self, exc_type, exc, tb):
        if self.synchronous:
            raise exc
        self.append_log("".join(traceback.format_exception(exc_type, exc, tb)))
        messagebox.showerror("Error", str(exc), parent=self)

    def set_status(self, text):
        self.status.configure(text=text)

    # State ------------------------------------------------------------------------------------------------------

    def set_trap(self, trap):
        if self.trap is not None and self.trap is not trap:
            self.trap.close()
        self.trap = trap
        self.waveform = None
        self.trajectory = None
        self.trap_changed()

    def trap_changed(self):
        for tab in self.tabs.values():
            tab.refresh()

    def set_ion(self, element):
        self.ion = Ion(element)
        self.trap_changed()
        self.set_status(f"Ion species set to {element}")

    def set_waveform(self, waveform, omega_axial):
        self.waveform = waveform
        self.omega_axial = omega_axial

    def quit_app(self):
        if self.capture is not None:
            self.capture.stop()
            self.capture = None
        self.destroy()


def main():
    app = TiOptApp()
    # Start with the default trap so that every tab can be used straight away
    app.tabs["trap"].generate_preset()
    app.mainloop()


if __name__ == "__main__":
    main()
