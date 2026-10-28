"""Graphical interface to tiopt, built with Tkinter and matplotlib.

Each tab covers one part of the library, i.e. trap design, electrostatics, transport waveforms, ion dynamics, integrated
photonics, quantum dynamics, circuit scheduling, and co-design. Computations run on a background thread, and the output
of the C library shows up in the log at the bottom of the window.
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
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.patches import Rectangle

from . import lib
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

    def __init__(self, app, title):
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

        self.plot = PlotPanel(panes)
        panes.add(self.plot, weight=1)

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
    def __init__(self, app):
        super().__init__(app, "Trap")

        preset = self.form("Linear surface trap preset")
        preset.entry("n_segments", "Segments per side", 7)
        preset.entry("segment_width", "Segment width", 50, "µm")
        preset.check("gratings", "Grating coupler under the trap center", True)
        preset.check("photonics", "Full 40Ca+ photonics (4 gratings and a detector)", False)
        preset.buttons(("Generate", self.generate_preset))
        self.preset = preset

        files = self.form("Trap files")
        files.buttons(("Load...", self.load_file), ("Save...", self.save_file))

        rf = self.form("RF drive")
        rf.entry("V_RF", "Amplitude", 40, "V")
        rf.entry("f_RF", "Frequency", 40, "MHz")
        rf.buttons(("Apply", self.apply_rf))
        self.rf = rf

        ion = self.form("Ion species")
        self.ion_var = ion.combo("species", "Species", Ion.PRESETS, "40Ca+")
        self.ion_var.trace_add("write", lambda *args: self.app.set_ion(self.ion_var.get()))

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
        n_segments = self.preset.get_int("n_segments")
        segment_width = self.preset.get("segment_width")
        gratings = self.preset.get("gratings")
        photonics = self.preset.get("photonics")

        def work():
            trap = Trap.from_preset(n_segments, segment_width, gratings, photonics)
            trap.solve("analytical")
            return trap

        self.app.run_task("Generating and solving trap", work, self.app.set_trap)

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

    def refresh(self):
        trap = self.app.trap
        self.tree.delete(*self.tree.get_children())
        if trap is None:
            return

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


def wavelength_color(wavelength):
    """Get the plot color for light of a wavelength (µm), distinguishing the 40Ca+ beams."""
    for limit, color in ((0.45, "tab:purple"), (0.8, "tab:red"), (0.86, "tab:pink")):
        if wavelength < limit:
            return color
    return "tab:brown"


def draw_photonics(ax, trap):
    """Draw the grating couplers (with their beam directions) and detectors of a trap on xy axes."""
    labelled = set()
    for grating in trap.grating_couplers():
        position = grating.position
        wavelength = grating.struct.waveguide.wavelength
        color = wavelength_color(wavelength)
        label = f"{wavelength * 1e3:.0f} nm grating"
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
        solver.combo("solver", "Solver", ("analytical", "spectral", "relaxation"), "analytical")
        solver.buttons(("Solve", self.solve))
        self.solver = solver

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

        self.plot.message("Generate a trap, then plot its potentials")

    def solve(self):
        trap = self.require_trap()
        solver = self.solver.get_str("solver")
        if solver == "relaxation" and not messagebox.askyesno(
            "Relaxation solver", "The relaxation solver solves Laplace's equation on a large grid for every electrode, which can take many minutes. Continue?", parent=self
        ):
            return

        def work():
            trap.solve(solver)
            return trap

        self.app.run_task(f"Solving electrodynamics ({solver})", work, lambda trap: self.app.trap_changed())

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

        results = self.form("Results")
        self.results = results.text(6)

        self.plot.message("Solve a static well or a transport")

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

        results = self.form("Results")
        self.results = results.text(8)

        self.plot.message("Solve a waveform in the Waveforms tab, then simulate it")

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
# Photonics
# ---------------------------------------------------------------------------------------------------------------------


class PhotonicsTab(Tab):
    def __init__(self, app):
        super().__init__(app, "Photonics")

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

        self.plot.message("Design a grating coupler which addresses the ion")

    def load_trap_grating(self):
        trap = self.app.trap
        if trap is None or not trap.grating_couplers():
            raise ValueError("The current trap has no grating couplers (enable them in the Trap tab preset)")
        grating = trap.grating_couplers()[0]
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
    COLUMNS = (("start", "t₀ (µs)"), ("duration", "τ (µs)"), ("detuning", "Δ (kHz)"), ("rabi", "Ω (kHz)"), ("phase", "φ (π)"), ("eta", "η"), ("targets", "Ions"))
    PRESETS = ("Carrier Rabi flopping", "Blue sideband", "Red sideband", "Ramsey", "Mølmer-Sørensen gate")

    def __init__(self, app):
        super().__init__(app, "Quantum")

        mode = self.form("Ions and motional mode")
        mode.combo("n_ions", "Ions", ("1", "2"), "1")
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
        presets.buttons(("Load", self.load_preset))

        editor = ttk.LabelFrame(self.controls, text="Pulses", padding=6)
        editor.pack(fill="x", padx=(0, 4), pady=(0, 6))
        self.tree = ttk.Treeview(editor, columns=[c[0] for c in self.COLUMNS], show="headings", height=6)
        for key, text in self.COLUMNS:
            self.tree.heading(key, text=text)
            self.tree.column(key, width=46, minwidth=30, anchor="e")
        self.tree.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", self.select_pulse)

        entry_row = ttk.Frame(editor)
        entry_row.pack(fill="x", pady=(4, 0))
        self.pulse_vars = {}
        for i, (key, text) in enumerate(self.COLUMNS):
            ttk.Label(entry_row, text=text.split(" ")[0]).grid(row=0, column=i)
            var = tk.StringVar()
            ttk.Entry(entry_row, textvariable=var, width=5).grid(row=1, column=i, padx=1)
            self.pulse_vars[key] = var
        buttons = ttk.Frame(editor)
        buttons.pack(fill="x", pady=(4, 0))
        for text, command in (("Add", self.add_pulse), ("Update", self.update_pulse), ("Remove", self.remove_pulse), ("Clear", self.clear_pulses)):
            ttk.Button(buttons, text=text, command=command).pack(side="left", padx=(0, 4))

        run = self.form("Run")
        run.buttons(("Simulate", self.simulate))
        self.results = run.text(6)

        self.load_preset()
        self.plot.message("Load a preset or edit the pulses, then simulate")

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
            else:
                try:
                    float(value)
                except ValueError:
                    raise ValueError(f"{text} must be a number") from None
                values.append(value)
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
            self.tree.insert("", "end", values=[f"{v:g}" if isinstance(v, float) else str(v) for v in row])
        if rows:
            for (key, _), value in zip(self.COLUMNS, rows[0]):
                self.pulse_vars[key].set(f"{value:g}" if isinstance(value, float) else str(value))

    def load_preset(self):
        preset = self.preset_var.get()
        f_mode_khz = self.mode.get("f_mode") * 1e3
        if preset == "Carrier Rabi flopping":
            self.mode.set("n_ions", "1")
            self.set_pulses([(0.0, 30.0, 0.0, 100.0, 0.0, 0.1, "1")])
        elif preset == "Blue sideband":
            self.mode.set("n_ions", "1")
            self.set_pulses([(0.0, 200.0, f_mode_khz, 100.0, 0.0, 0.1, "1")])
        elif preset == "Red sideband":
            self.mode.set("n_ions", "1")
            self.mode.set("n_bar", "0.5")
            self.set_pulses([(0.0, 200.0, -f_mode_khz, 100.0, 0.0, 0.1, "1")])
        elif preset == "Ramsey":
            self.mode.set("n_ions", "1")
            self.set_pulses([(0.0, 1.25, 0.0, 200.0, 0.0, 0.0, "1"), (21.25, 1.25, 0.0, 200.0, 0.5, 0.0, "1")])
        elif preset == "Mølmer-Sørensen gate":
            self.mode.set("n_ions", "2")
            self.mode.set("n_bar", "0")
            self.mode.set("n_max", "12")
            delta_khz = 20.0
            eta = 0.05
            rabi_khz = delta_khz / (2 * eta)
            duration = 1e3 / delta_khz
            self.set_pulses([
                (0.0, duration, f_mode_khz + delta_khz, rabi_khz, 0.0, eta, "1,2"),
                (0.0, duration, -f_mode_khz - delta_khz, rabi_khz, 0.0, eta, "1,2"),
            ])

    def pulses(self):
        pulses = []
        for item in self.tree.get_children():
            start, duration, detuning, rabi, phase, eta, targets = self.tree.item(item, "values")
            pulses.append(Pulse(
                start=float(start) * 1e-6,
                duration=float(duration) * 1e-6,
                detuning=TWO_PI * float(detuning) * 1e3,
                rabi_frequency=TWO_PI * float(rabi) * 1e3,
                phase=np.pi * float(phase),
                targets=[int(t) for t in str(targets).split(",")],
                lamb_dicke=float(eta),
            ))
        if not pulses:
            raise ValueError("Add at least one pulse")
        return pulses

    # Simulation -----------------------------------------------------------------------------------------------

    def simulate(self):
        ion = self.app.ion
        pulses = self.pulses()
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
            return lib.simulate_pulses(ion, pulses, **kwargs)

        def done(result):
            times = result.times * 1e6
            has_bars = n_ions == 2
            axes = self.plot.axes(3 if has_bars else 2, 1)[:, 0]
            ax_pop, ax_n = axes[0], axes[1]

            for i in range(n_ions):
                ax_pop.plot(times, result.excited_populations[:, i], label=f"Ion {i + 1}")

            # Compare a single pulse on a single ion with the analytical thermal result
            lines = []
            if n_ions == 1 and len(pulses) == 1 and kwargs["heating_rate"] == 0 and kwargs["dephasing_rate"] == 0 and kwargs["alpha"] == 0:
                pulse = pulses[0]
                sideband = int(round(pulse.detuning / omega_mode))
                residual = pulse.detuning - sideband * omega_mode
                inside = (result.times >= pulse.start) & (result.times <= pulse.start + pulse.duration)
                analytic = np.zeros_like(result.times)
                analytic[inside] = lib.thermal_rabi_excitation(pulse.rabi_frequency, pulse.lamb_dicke, sideband, kwargs["n_bar"], residual, result.times[inside] - pulse.start, kwargs["n_max"])
                ax_pop.plot(times[inside], analytic[inside], color="black", linestyle=":", label="Analytical")
                lines.append(f"Largest deviation from analytical: {np.abs(analytic[inside] - result.excited_populations[inside, 0]).max():.2e}")

            ax_pop.set_ylabel("D-state population")
            ax_pop.set_ylim(-0.02, 1.02)
            ax_pop.legend(fontsize=8)
            ax_pop.set_title(f"{len(pulses)} pulse(s) on {n_ions} {ion.element} ion(s)")

            ax_n.plot(times, result.mean_phonon_numbers)
            ax_n.set_ylabel("Mean phonon number")
            ax_n.set_xlabel("Time (µs)")

            lines.insert(0, "Final D-state populations: " + ", ".join(f"{p:.4f}" for p in result.excited_populations[-1]))
            lines.insert(1, f"Final mean phonon number: {result.mean_phonon_numbers[-1]:.4f}")

            if has_bars:
                rho = result.density_matrix
                n_motion = kwargs["n_max"] + 1
                populations = [np.real(sum(rho[c * n_motion + k, c * n_motion + k] for k in range(n_motion))) for c in range(4)]
                coherence = sum(rho[k, 3 * n_motion + k] for k in range(n_motion))
                fidelity = 0.5 * (populations[0] + populations[3]) + abs(coherence)
                ax_bars = axes[2]
                ax_bars.bar(["SS", "DS", "SD", "DD"], populations)
                ax_bars.set_ylabel("Final population")
                ax_bars.set_ylim(0, 1)
                lines.append(f"Bell state fidelity: {fidelity:.5f}")
            set_text(self.results, "\n".join(lines))
            self.plot.draw()

        self.app.run_task("Simulating quantum dynamics", work, done)


# ---------------------------------------------------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------------------------------------------------


class SchedulingTab(Tab):
    def __init__(self, app):
        super().__init__(app, "Scheduling")

        circuit = self.form("Circuit")
        self.qasm_var = circuit.entry("qasm", "OpenQASM file", REPOSITORY / "tests" / "quantumcircuits" / "test_0000.qasm")
        circuit.buttons(("Browse...", self.browse))

        zones = self.form("Trap zones")
        # Wells closer than several ion heights need large voltages to be held at once, so the zones are 100 µm apart
        zones.entry("types", "Types", "load+storage,empty,gate,empty,measure")
        zones.entry("capacities", "Capacities", "4,1,2,1,4")
        zones.entry("x_min", "First zone", -200, "µm")
        zones.entry("x_max", "Last zone", 200, "µm")
        self.zones = zones

        actions = self.form("Actions")
        actions.entry("n_steps", "Waveform steps per transport", 101)
        actions.buttons(("Schedule", self.schedule), ("Build waveform", self.build_waveform))
        self.actions = actions

        results = ttk.LabelFrame(self.controls, text="Schedule", padding=6)
        results.pack(fill="both", expand=True, padx=(0, 4))
        self.results = ScrolledText(results, height=16, width=40, wrap="none")
        self.results.pack(fill="both", expand=True)

        self.circuit = None
        self.program = None
        self.plot.message("Schedule the transports of a circuit")

    def browse(self):
        path = filedialog.askopenfilename(parent=self, title="Open OpenQASM file", initialdir=REPOSITORY / "tests" / "quantumcircuits", filetypes=[("OpenQASM", "*.qasm"), ("All files", "*")])
        if path:
            self.qasm_var.set(path)

    def _trap_graph(self):
        types = [t.strip() for t in self.zones.get_str("types").split(",") if t.strip()]
        capacities = [int(c) for c in self.zones.get_str("capacities").split(",") if c.strip()]
        if len(types) != len(capacities):
            raise ValueError("Give one capacity for each zone")
        for t in types:
            lib._zone_type(t)
        height = ION_HEIGHT
        if self.app.trap is not None and self.app.trap.solved and self.app.trap.n_rf > 0:
            height = self.app.trap.find_rf_null((0.0, 0.0, None))[2]
        return TrapGraph(types, capacities, self.zones.get("x_min"), self.zones.get("x_max"), height), types

    def schedule(self):
        path = self.qasm_var.get()
        trap_graph, types = self._trap_graph()

        def work():
            circuit = Circuit(path)
            program = TransportProgram(trap_graph, circuit)
            return circuit, program

        def done(result):
            self.circuit, self.program = result
            self.show(trap_graph, types)

        self.app.run_task("Scheduling transports", work, done)

    def show(self, trap_graph, types):
        circuit = self.circuit
        program = self.program
        instructions = circuit.instructions
        zone_positions = trap_graph.positions

        text = [f"{len(instructions)} instructions on {circuit.n_ions} qubits in {max(i.layer for i in instructions)} layers", ""]
        for layer in range(1, max(i.layer for i in instructions) + 1):
            text.append(f"Layer {layer}: " + "; ".join(f"{i.operation} {i.targets}" for i in instructions if i.layer == layer))
        text += ["", "Transports (targets -> destination at arrival time):"]
        for t, (targets, destination, arrival) in enumerate(zip(program.targets, program.destinations, program.arrival_times)):
            zone = int(np.argmin(np.linalg.norm(zone_positions - destination, axis=1)))
            text.append(f"{t + 1:3d}: ions {targets} -> {types[zone]} zone (x = {destination[0]:.0f} µm) at {arrival * 1e6:.1f} µs")
        self.results.delete("1.0", "end")
        self.results.insert("1.0", "\n".join(text))

        ax_circuit, ax_positions = self.plot.axes(2, 1)[:, 0]
        for q in range(circuit.n_ions):
            ax_circuit.axhline(q + 1, color="gray", linewidth=0.8)
        for i in instructions:
            # Spread the gates of a layer out so that overlapping gates can be told apart
            same_layer = [j for j in instructions if j.layer == i.layer]
            x = i.layer + 0.2 * (same_layer.index(i) - 0.5 * (len(same_layer) - 1)) / max(len(same_layer) - 1, 1)
            if len(i.targets) > 1:
                ax_circuit.plot([x, x], [min(i.targets), max(i.targets)], color="tab:blue", linewidth=1.5)
            for target in i.targets:
                label = "M" if i.operation == "detect" else ("U" if len(i.targets) == 1 else "●")
                ax_circuit.text(x, target, label, ha="center", va="center", bbox=dict(boxstyle="round", facecolor="white", edgecolor="tab:blue"))
        ax_circuit.set_xlim(0.4, max(i.layer for i in instructions) + 0.6)
        ax_circuit.set_ylim(circuit.n_ions + 0.6, 0.4)
        ax_circuit.set_xlabel("Layer")
        ax_circuit.set_ylabel("Qubit")
        ax_circuit.set_title("Circuit layers")

        # Position of every ion over time, from the scheduled destinations
        positions = program.initial_positions[:, 0].copy()
        times = [0.0]
        history = [positions.copy()]
        for targets, destination, arrival in zip(program.targets, program.destinations, program.arrival_times):
            for target in targets:
                positions[target - 1] = destination[0]
            times.append(arrival * 1e6)
            history.append(positions.copy())
        history = np.array(history)
        for q in range(circuit.n_ions):
            ax_positions.step(times, history[:, q] + 2.0 * q, where="post", label=f"Ion {q + 1}")
        for z, position in enumerate(zone_positions):
            ax_positions.axhline(position[0], color="gray", linewidth=0.5, linestyle=":")
            ax_positions.text(times[-1], position[0], f" {types[z]}", va="center", fontsize=7, color="gray")
        ax_positions.set_xlabel("Arrival time (µs)")
        ax_positions.set_ylabel("Zone position (µm, offset per ion)")
        ax_positions.legend(fontsize=7, ncols=2)
        ax_positions.set_title("Ion positions")
        self.plot.draw()

    def build_waveform(self):
        trap = self.require_trap()
        if self.program is None:
            raise ValueError("Schedule a circuit first")
        ion = self.app.ion
        n_steps = self.actions.get_int("n_steps")
        program = self.program
        omega_axial = self.app.tabs["waveforms"].omega_axial()

        def work():
            return Waveform.from_program(trap, ion, program, n_steps, omega_axial)

        def done(waveform):
            self.app.set_waveform(waveform, omega_axial)
            self.app.tabs["waveforms"].show(waveform, omega_axial)
            self.app.notebook.select(self.app.tabs["waveforms"])

        self.app.run_task("Building waveform for schedule", work, done)


# ---------------------------------------------------------------------------------------------------------------------
# Co-design
# ---------------------------------------------------------------------------------------------------------------------


class CodesignTab(Tab):
    """Tab for the co-design flow, which makes a basic design of everything and then passes which optimize the trap,
    photonics, transport, and waveforms a little, first each based on its own figures of merit and then all based on the
    metrics of the whole circuit.
    """

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

    def __init__(self, app):
        super().__init__(app, "Co-design")

        circuit = self.form("Circuit")
        self.qasm_var = circuit.entry("qasm", "OpenQASM file", EXAMPLES / "circuits" / "bell.qasm")
        circuit.buttons(("Browse...", self.browse))

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
        results.pack(fill="both", expand=True, padx=(0, 4))
        self.results = ScrolledText(results, height=18, width=40, wrap="none")
        self.results.pack(fill="both", expand=True)

        self.codesign = None
        self.records = []
        self.plot.message("Make a basic design for a circuit, then run the co-design flow")

    def browse(self):
        path = filedialog.askopenfilename(parent=self, title="Open OpenQASM file", initialdir=EXAMPLES / "circuits", filetypes=[("OpenQASM", "*.qasm"), ("All files", "*")])
        if path:
            self.qasm_var.set(path)

    def _environment(self):
        values = {}
        for key, label, unit, scale in self.ENVIRONMENT:
            values[key] = self.environment.get_int(key) if key == "n_segments" else self.environment.get(key) * scale
        return values

    def basic_design(self):
        path = self.qasm_var.get()
        element = self.app.ion.element
        environment = self._environment()

        def work():
            return lib.Codesign(path, element, environment)

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
            layout = self.codesign.layout()
            for name, position in (("gate", layout["gate_position"]), ("measure", layout["measure_position"])):
                ax.plot(position[0], position[1], marker="x", color="black", linestyle="none")
                ax.annotate(name, (position[0], position[1]), textcoords="offset points", xytext=(4, 4), fontsize=7)
            ax.set_xlim(-260, 260)
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
            "electrostatics": ElectrostaticsTab(self),
            "waveforms": WaveformTab(self),
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
