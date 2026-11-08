"""Electrode voltages from objectives and constraints, in the style of pytrans, for static potentials and whole
waveforms.

A problem holds the voltages of every electrode at every step of a waveform (a static potential is a waveform with one
step) and a list of objectives, each of which is a linear function of the voltages (e.g. the gradient of the potential
at a well, an entry of its Hessian, a multipole, or a voltage). Each objective is either minimized as a weighted
quadratic penalty or enforced as a constraint ("==", "<=", or ">="), so that the whole problem is a convex quadratic
program, which tiopt solves with the alternating direction method of multipliers (as in OSQP) on sparse matrices. The
RF pseudopotential is included in the potential of every objective (unless pseudo is False), so that e.g. a gradient of
zero puts the ion where the DC and RF forces balance.

For example, a transport which holds a 1 MHz well that moves from -50 µm to 50 µm, with every voltage within 10 V and
slew rates below 10 V/µs:

    problem = Problem(trap, ion, n_steps=101, duration=20e-6)
    for step, x in enumerate(sinsquared(-50, 50, 101)):
        well = trap.find_rf_null((x, 0, None))
        problem.add(Gradient(step, well, 0.0), Hessian(step, well, frequency_curvature(ion, 2 * pi * 1e6), "xx"))
    problem.add(VoltageLimit(10.0), SlewRate(1e7))
    waveform = problem.waveform()
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse, special
from scipy.sparse import linalg as sparse_linalg

from .lib import LENGTH_SCALE, Waveform

KINDS = ("min", "==", "<=", ">=")
ENTRIES = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2), "xy": (0, 1), "yx": (0, 1), "xz": (0, 2), "zx": (0, 2), "yz": (1, 2), "zy": (1, 2)}


def frequency_curvature(ion, omega):
    """Curvature of the potential (V/µm^2) which confines an ion with secular frequency omega (rad/s)."""
    return ion.mass * omega ** 2 / ion.charge * LENGTH_SCALE ** 2


# Time profiles --------------------------------------------------------------------------------------------------------


def sinsquared(a, b, n):
    """n points from a to b which follow a sine squared ramp (slow at both ends)."""
    t = np.linspace(0, 1, n)
    return a + (b - a) * np.sin(np.pi * t / 2) ** 2


def erfspace(a, b, n, scaling=2.5):
    """n points from a to b which follow an error function ramp, which is steeper in the middle for larger scaling."""
    t = np.linspace(-1, 1, n)
    ramp = special.erf(scaling * t)
    ramp = (ramp - ramp[0]) / (ramp[-1] - ramp[0])
    return a + (b - a) * ramp


def minimum_jerk(a, b, n):
    """n points from a to b which follow the minimum-jerk profile 10 t^3 - 15 t^4 + 6 t^5."""
    t = np.linspace(0, 1, n)
    return a + (b - a) * (10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5)


PROFILES = {"sine squared": sinsquared, "error function": erfspace, "minimum jerk": minimum_jerk, "linear": np.linspace}


# Objectives ------------------------------------------------------------------------------------------------------------


@dataclass
class Objective:
    """Linear function of the voltages with a target, which is minimized as a penalty (kind "min", with a weight and a
    norm which sets the size of an acceptable error) or enforced as a constraint ("==", "<=", or ">=").
    """
    kind: str = field(default="min", kw_only=True)
    weight: float = field(default=1.0, kw_only=True)
    norm: float = field(default=1.0, kw_only=True)

    def rows(self, problem):
        """Rows of the linear function (in terms of the flattened voltages of every step) and their targets."""
        raise NotImplementedError

    def _check(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}; choose from {', '.join(KINDS)}")


@dataclass
class Potential(Objective):
    """Potential energy per charge (V) at points (µm) at a step, including the pseudopotential unless pseudo is False."""
    step: int
    points: np.ndarray
    value: object = 0.0
    pseudo: bool = True
    norm: float = field(default=1e-3, kw_only=True)

    def rows(self, problem):
        points = np.atleast_2d(np.asarray(self.points, dtype=float))
        A = np.zeros((len(points), problem.n_electrodes))
        offset = np.zeros(len(points))
        for k, point in enumerate(points):
            phi, _, _, pseudo, _, _ = problem.derivatives(point)
            A[k] = phi
            if self.pseudo:
                offset[k] = pseudo
        return problem.at_step(self.step, A), np.broadcast_to(np.asarray(self.value, dtype=float), (len(points),)) - offset


@dataclass
class Gradient(Objective):
    """Gradient of the potential energy per charge (V/µm) at a point at a step, for the given entries (e.g. "xyz" to
    put a well at the point).
    """
    step: int
    position: np.ndarray
    value: object = 0.0
    entries: str = "xyz"
    pseudo: bool = True
    norm: float = field(default=1e-5, kw_only=True)

    def rows(self, problem):
        _, grad, _, _, pseudo_grad, _ = problem.derivatives(self.position)
        indices = ["xyz".index(c) for c in self.entries]
        A = grad[:, indices].T
        offset = pseudo_grad[indices] if self.pseudo else np.zeros(len(indices))
        return problem.at_step(self.step, A), np.broadcast_to(np.asarray(self.value, dtype=float), (len(indices),)) - offset


@dataclass
class Hessian(Objective):
    """Entries of the Hessian of the potential energy per charge (V/µm^2) at a point at a step, e.g. "xx" with
    frequency_curvature for the axial frequency.
    """
    step: int
    position: np.ndarray
    value: object
    entries: object = "xx"
    pseudo: bool = True
    norm: float = field(default=1e-6, kw_only=True)

    def rows(self, problem):
        _, _, H, _, _, pseudo_H = problem.derivatives(self.position)
        entries = [self.entries] if isinstance(self.entries, str) else list(self.entries)
        A = np.array([H[:, ENTRIES[e][0], ENTRIES[e][1]] for e in entries])
        offset = np.array([pseudo_H[ENTRIES[e]] for e in entries]) if self.pseudo else np.zeros(len(entries))
        return problem.at_step(self.step, A), np.broadcast_to(np.asarray(self.value, dtype=float), (len(entries),)) - offset


@dataclass
class Tilt(Objective):
    """Rotation of the two principal axes in a plane (e.g. "yz" for the radial axes of a trap along x) by an angle
    (degrees), i.e. 2 H_yz = tan(2 angle) (H_yy - H_zz) for the total Hessian at a point at a step.
    """
    step: int
    position: np.ndarray
    angle: float
    plane: str = "yz"
    pseudo: bool = True
    kind: str = field(default="==", kw_only=True)
    norm: float = field(default=1e-6, kw_only=True)

    def rows(self, problem):
        _, _, H, _, _, pseudo_H = problem.derivatives(self.position)
        a, b = "xyz".index(self.plane[0]), "xyz".index(self.plane[1])
        t = np.tan(2 * np.radians(self.angle))
        A = (2 * H[:, a, b] - t * (H[:, a, a] - H[:, b, b]))[None, :]
        offset = 2 * pseudo_H[a, b] - t * (pseudo_H[a, a] - pseudo_H[b, b]) if self.pseudo else 0.0
        return problem.at_step(self.step, A), np.array([-offset])


@dataclass
class Multipole(Objective):
    """Multipole of the DC potential at a point at a step (see tiopt.multipoles), e.g. "Ex" to null a stray field or
    "R3,0" for a cubic term.
    """
    step: int
    position: np.ndarray
    name: str
    value: float = 0.0
    order: int = 2
    norm: float = field(default=1e-5, kw_only=True)

    def rows(self, problem):
        from . import multipoles

        control = problem.control(self.position, max(self.order, 2 if self.name in multipoles.BASIC else int(self.name[1:].split(",")[0])))
        return problem.at_step(self.step, control.rows([self.name])), np.array([self.value], dtype=float)


@dataclass
class Voltage(Objective):
    """Voltages (V) of electrodes at steps, which are all of the DC electrodes and every step by default."""
    value: object = 0.0
    electrodes: object = None
    steps: object = None
    norm: float = field(default=1.0, kw_only=True)

    def rows(self, problem):
        electrodes = problem.dc_electrodes if self.electrodes is None else list(np.atleast_1d(self.electrodes))
        steps = range(problem.n_steps) if self.steps is None else list(np.atleast_1d(self.steps))
        rows = []
        for step in steps:
            for e in electrodes:
                rows.append(step * problem.n_electrodes + e)
        A = sparse.csr_matrix((np.ones(len(rows)), (np.arange(len(rows)), rows)), shape=(len(rows), problem.n_variables))
        value = np.asarray(self.value, dtype=float)
        if value.ndim == 2:
            value = value[np.ix_(list(steps), electrodes)]
        return A, np.broadcast_to(value, (len(steps), len(electrodes))).reshape(-1)


def VoltageLimit(max_voltage, electrodes=None, steps=None):
    """Constraints which keep the magnitude of every voltage within max_voltage (V)."""
    return [Voltage(max_voltage, electrodes, steps, kind="<="), Voltage(-max_voltage, electrodes, steps, kind=">=")]


@dataclass
class SlewRate(Objective):
    """Rate of change of every DC voltage between steps (V/s), which is limited to max_rate as a constraint, or whose
    squares are minimized (as a roughness penalty) if kind is "min".
    """
    max_rate: float = 0.0
    electrodes: object = None
    kind: str = field(default="<=", kw_only=True)
    norm: float = field(default=1.0, kw_only=True)

    def rows(self, problem):
        if problem.n_steps < 2:
            return sparse.csr_matrix((0, problem.n_variables)), np.zeros(0)
        electrodes = problem.dc_electrodes if self.electrodes is None else list(np.atleast_1d(self.electrodes))
        dt = np.diff(problem.times)
        rows, cols, values = [], [], []
        r = 0
        for step in range(problem.n_steps - 1):
            for e in electrodes:
                rows += [r, r]
                cols += [(step + 1) * problem.n_electrodes + e, step * problem.n_electrodes + e]
                values += [1.0 / dt[step], -1.0 / dt[step]]
                r += 1
        A = sparse.csr_matrix((values, (rows, cols)), shape=(r, problem.n_variables))
        return A, np.full(r, self.max_rate)

    def expand(self):
        """A rate limit needs both a "<=" and a ">=" constraint."""
        if self.kind == "<=":
            return [self, _NegatedSlewRate(self.max_rate, self.electrodes)]
        return [self]


@dataclass
class _NegatedSlewRate(SlewRate):
    kind: str = field(default=">=", kw_only=True)

    def rows(self, problem):
        A, b = SlewRate.rows(self, problem)
        return A, -b


@dataclass
class Symmetry(Objective):
    """Voltages of two lists of electrodes which mirror each other (v_lhs = sign * v_rhs) at every step."""
    lhs: list = None
    rhs: list = None
    sign: float = 1.0
    kind: str = field(default="==", kw_only=True)

    def rows(self, problem):
        rows, cols, values = [], [], []
        r = 0
        for step in range(problem.n_steps):
            for a, b in zip(self.lhs, self.rhs):
                rows += [r, r]
                cols += [step * problem.n_electrodes + a, step * problem.n_electrodes + b]
                values += [1.0, -self.sign]
                r += 1
        return sparse.csr_matrix((values, (rows, cols)), shape=(r, problem.n_variables)), np.zeros(r)


# Problems --------------------------------------------------------------------------------------------------------------


class Problem:
    """Voltages of every electrode at every step of a waveform, with the objectives which they must meet."""

    def __init__(self, trap, ion, n_steps=1, duration=0.0, times=None):
        trap._require_solved()
        self.trap = trap
        self.ion = ion
        self.n_steps = int(n_steps)
        self.n_electrodes = trap.n_electrodes
        self.times = np.linspace(0.0, duration, self.n_steps) if times is None else np.asarray(times, dtype=float)
        self.dc_electrodes = list(range(trap.n_rf, trap.n_electrodes))
        self.objectives = []
        self.well_positions = np.zeros((self.n_steps, 1, 3))
        self._cache = {}
        self._controls = {}

    @property
    def n_variables(self):
        return self.n_steps * self.n_electrodes

    def add(self, *objectives):
        """Add objectives (or lists of them)."""
        for objective in objectives:
            if isinstance(objective, (list, tuple)):
                self.add(*objective)
                continue
            objective._check()
            expanded = objective.expand() if hasattr(objective, "expand") else [objective]
            self.objectives.extend(expanded)
            position = getattr(objective, "position", None)
            if position is not None and isinstance(getattr(objective, "step", None), (int, np.integer)):
                self.well_positions[objective.step, 0] = position
        return self

    def derivatives(self, position):
        """Potential (V), gradient (V/µm), and Hessian (V/µm^2) of every electrode at 1 V at a position, and the
        pseudopotential per charge with its gradient and Hessian.
        """
        key = tuple(np.round(np.asarray(position, dtype=float), 9))
        if key not in self._cache:
            n = self.n_electrodes
            phi = np.zeros(n)
            grad = np.zeros((n, 3))
            H = np.zeros((n, 3, 3))
            for e in range(n):
                voltages = self.trap.zero_voltages()
                voltages[e] = 1.0
                phi[e], grad[e], H[e] = self.trap.evaluate(position, voltages)
            U, U_grad, U_H = self.trap.energy_derivatives(self.ion, position, None)
            q = self.ion.charge
            self._cache[key] = (phi, grad, H, U / q, U_grad / q, U_H / q)
        return self._cache[key]

    def control(self, position, order):
        from . import multipoles

        key = (tuple(np.round(np.asarray(position, dtype=float), 9)), order)
        if key not in self._controls:
            self._controls[key] = multipoles.control_matrix(self.trap, position, order)
        return self._controls[key]

    def at_step(self, step, A):
        """Place rows over the electrodes at a step into rows over every variable."""
        A = np.atleast_2d(A)
        full = sparse.lil_matrix((A.shape[0], self.n_variables))
        full[:, step * self.n_electrodes:(step + 1) * self.n_electrodes] = A
        return full.tocsr()

    def _assemble(self):
        n = self.n_variables
        P = sparse.csr_matrix((n, n))
        q = np.zeros(n)
        constraints, lower, upper = [], [], []
        for objective in self.objectives:
            A, b = objective.rows(self)
            A = sparse.csr_matrix(A)
            b = np.asarray(b, dtype=float)
            if A.shape[0] == 0:
                continue
            if objective.kind == "min":
                w = objective.weight / objective.norm ** 2
                P = P + 2 * w * (A.T @ A)
                q = q - 2 * w * (A.T @ b)
            else:
                # Scale every constraint row to unit length, which helps the ADMM converge
                norms = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).reshape(-1))
                norms[norms == 0] = 1.0
                A = sparse.diags(1 / norms) @ A
                b = b / norms
                constraints.append(A)
                lower.append(b if objective.kind in ("==", ">=") else np.full(len(b), -np.inf))
                upper.append(b if objective.kind in ("==", "<=") else np.full(len(b), np.inf))
        # The RF electrodes carry no DC voltage
        fixed = [step * self.n_electrodes + e for step in range(self.n_steps) for e in range(self.trap.n_rf)]
        if fixed:
            constraints.append(sparse.csr_matrix((np.ones(len(fixed)), (np.arange(len(fixed)), fixed)), shape=(len(fixed), n)))
            lower.append(np.zeros(len(fixed)))
            upper.append(np.zeros(len(fixed)))
        C = sparse.vstack(constraints).tocsc() if constraints else sparse.csc_matrix((0, n))
        return P.tocsc(), q, C, np.concatenate(lower) if lower else np.zeros(0), np.concatenate(upper) if upper else np.zeros(0)

    def solve(self, max_iterations=20000, tolerance=1e-7, rho=1.0, sigma=1e-9, alpha=1.6):
        """Solve the problem.

        Returns:
            Voltages with shape (n_steps, n_electrodes), and a SolveInfo with the residuals and the objective of each
            objective
        """
        P, q, C, lower, upper = self._assemble()
        n = P.shape[0]
        # A small multiple of the identity keeps P definite for electrodes which no objective touches
        P = P + 1e-12 * max(abs(P).max() if P.nnz else 1.0, 1.0) * sparse.identity(n, format="csc")
        x, iterations, primal, dual, converged = admm(P, q, C, lower, upper, max_iterations, tolerance, rho, sigma, alpha)
        voltages = x.reshape(self.n_steps, self.n_electrodes)
        return voltages, SolveInfo(iterations, primal, dual, converged, self.residuals(voltages))

    def residuals(self, voltages):
        """Largest error of each objective, as a list of (label, error), where penalties give their error relative to
        their norm and constraints how far they are violated relative to their targets.
        """
        x = np.asarray(voltages, dtype=float).reshape(-1)
        out = []
        for objective in self.objectives:
            A, b = objective.rows(self)
            if A.shape[0] == 0:
                continue
            r = sparse.csr_matrix(A) @ x - b
            scale = max(np.abs(b).max(), objective.norm)
            if objective.kind == "==":
                error = np.abs(r).max() / scale
            elif objective.kind == "<=":
                error = max(r.max(), 0.0) / scale
            elif objective.kind == ">=":
                error = max(-r.min(), 0.0) / scale
            else:
                error = np.abs(r).max() / objective.norm
            name = type(objective).__name__.lstrip("_").replace("Negated", "")
            out.append((f"{name} ({'penalty' if objective.kind == 'min' else objective.kind})", float(error)))
        return out

    def waveform(self, **kwargs):
        """Solve the problem and make a waveform of its voltages."""
        voltages, info = self.solve(**kwargs)
        waveform = Waveform.from_voltages(self.times, voltages, self.well_positions)
        waveform.info = info
        return waveform


@dataclass
class SolveInfo:
    """Number of iterations of the solver, its final primal and dual residuals, whether it converged (which it does not
    if the constraints cannot all be met), and the error of every objective.
    """
    iterations: int
    primal_residual: float
    dual_residual: float
    converged: bool
    residuals: list

    def summary(self):
        """Lines of text which describe the solution."""
        lines = [f"{'Converged' if self.converged else 'Did NOT converge (the constraints may be infeasible)'} after {self.iterations} iterations"]
        worst = {}
        for name, error in self.residuals:
            worst[name] = max(worst.get(name, 0.0), error)
        lines.extend(f"  {name}: largest {'error relative to its norm' if 'penalty' in name else 'relative violation'} {error:.3g}" for name, error in worst.items())
        return lines


def admm(P, q, C, lower, upper, max_iterations=20000, tolerance=1e-7, rho=1.0, sigma=1e-9, alpha=1.6):
    """Solve the quadratic program min 1/2 x^T P x + q^T x subject to lower <= C x <= upper with the alternating
    direction method of multipliers of OSQP (B. Stellato et al., Math. Program. Comput. 12, 637 (2020)).

    Returns:
        Solution, number of iterations, the final primal and dual residuals, and whether it converged
    """
    n = P.shape[0]
    m = C.shape[0]
    if m == 0:
        x = sparse_linalg.spsolve(P.tocsc(), -q)
        return np.asarray(x).reshape(-1), 0, 0.0, 0.0, True
    # Equality constraints get a larger step size, as in OSQP
    equality = np.isclose(lower, upper)
    rho_vector = np.where(equality, 1e3 * rho, rho)

    def factor(rho_vector):
        K = (P + sigma * sparse.identity(n, format="csc") + C.T @ sparse.diags(rho_vector) @ C).tocsc()
        return sparse_linalg.factorized(K)

    solve = factor(rho_vector)
    x = np.zeros(n)
    z = np.clip(np.zeros(m), lower, upper)
    y = np.zeros(m)
    scale = max(np.abs(q).max() if len(q) else 0.0, 1e-300)
    primal = dual = np.inf
    converged = False
    for iteration in range(1, max_iterations + 1):
        x_tilde = solve(sigma * x - q + C.T @ (rho_vector * z - y))
        z_tilde = C @ x_tilde
        x_next = alpha * x_tilde + (1 - alpha) * x
        z_relaxed = alpha * z_tilde + (1 - alpha) * z
        z_next = np.clip(z_relaxed + y / rho_vector, lower, upper)
        y = y + rho_vector * (z_relaxed - z_next)
        x, z = x_next, z_next
        if iteration % 25 == 0 or iteration == max_iterations:
            Cx = C @ x
            primal = np.abs(Cx - z).max()
            dual = np.abs(P @ x + q + C.T @ y).max()
            primal_scale = max(np.abs(Cx).max(), np.abs(z).max(), 1e-300)
            dual_scale = max(np.abs(P @ x).max(), np.abs(C.T @ y).max(), scale)
            if primal <= tolerance * max(primal_scale, 1.0) and dual <= tolerance * dual_scale:
                converged = True
                break
            # Rebalance the step size when the residuals drift apart, as in OSQP
            ratio = np.sqrt((primal / max(primal_scale, 1e-300)) / max(dual / dual_scale, 1e-300))
            if ratio > 5 or ratio < 0.2:
                rho_vector = np.clip(rho_vector * ratio, 1e-6, 1e6)
                solve = factor(rho_vector)
    return x, iteration, float(primal), float(dual), converged


# Convenience functions -------------------------------------------------------------------------------------------------


def static_well(trap, ion, position, omega_axial, axis="xx", max_voltage=None, tilt=None, weights=None):
    """Find the voltages of a static well at a position with an axial frequency (rad/s), with the radial axes rotated
    by tilt degrees (if given) and every voltage within max_voltage (V, if given).

    Returns:
        Voltages (V) and the SolveInfo
    """
    position = np.asarray(position, dtype=float)
    problem = Problem(trap, ion)
    weights = weights or {}
    # With a voltage limit, the well may not be reachable exactly, so it is then met as closely as possible
    kind = "==" if max_voltage is None else "min"
    problem.add(Gradient(0, position, 0.0, kind=kind, weight=weights.get("gradient", 1.0)))
    problem.add(Hessian(0, position, frequency_curvature(ion, omega_axial), axis, kind=kind, weight=weights.get("hessian", 1.0)))
    if tilt is not None:
        problem.add(Tilt(0, position, tilt, kind=kind))
    # The smallest voltages which do so
    problem.add(Voltage(0.0, weight=weights.get("voltage", 1e-6)))
    if max_voltage is not None:
        problem.add(VoltageLimit(max_voltage))
    voltages, info = problem.solve()
    return voltages[0], info


def transport(trap, ion, start, end, duration, n_steps, omega_axial, profile="sine squared", max_voltage=None,
              max_slew_rate=None, tilt=None, roughness=1e-3, follow_null=True):
    """Find a transport which holds a well with an axial frequency (rad/s) as it moves from start to end (µm) along a
    profile (see PROFILES), with optional limits on the voltages (V) and their slew rates (V/s) and a tilt of the radial
    axes (degrees) at every step.

    Returns:
        Waveform, whose info attribute holds the SolveInfo
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    fraction = PROFILES[profile](0.0, 1.0, n_steps)
    problem = Problem(trap, ion, n_steps, duration)
    curvature = frequency_curvature(ion, omega_axial)
    # With limits, the wells may not be reachable exactly, so they are then met as closely as possible
    kind = "==" if max_voltage is None and max_slew_rate is None else "min"
    for step, u in enumerate(fraction):
        position = start + u * (end - start)
        if follow_null and trap.n_rf > 0:
            position = trap.find_rf_null(position)
        problem.add(Gradient(step, position, 0.0, kind=kind), Hessian(step, position, curvature, "xx", kind=kind))
        if tilt is not None:
            problem.add(Tilt(step, position, tilt, kind=kind))
    problem.add(Voltage(0.0, weight=1e-6))
    if roughness:
        problem.add(SlewRate(0.0, kind="min", weight=roughness, norm=1e7))
    if max_voltage is not None:
        problem.add(VoltageLimit(max_voltage))
    if max_slew_rate is not None:
        problem.add(SlewRate(max_slew_rate))
    return problem.waveform()
