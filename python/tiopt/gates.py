"""Design of Mølmer-Sørensen gates for chains of ions with several motional modes, like the gate optimizers of Q-CTRL's
Boulder Opal: amplitude-modulated (AM) gates whose Rabi frequency changes in segments, and frequency-modulated (FM)
gates whose detuning does.

With tones at detunings of +mu and -mu from the qubit transition and a Rabi frequency Omega(t) on the target ions, the
gate displaces every mode m by
    alpha_im = i eta_im integral of Omega(t) cos(mu t) exp(i omega_m t) dt
and entangles ions i and j with the phase
    Phi_ij = 2 sum_m eta_im eta_jm double integral over t1 < t2 of Omega(t2) Omega(t1) cos(mu t2) cos(mu t1)
             sin(omega_m (t2 - t1)),
so that the gate is exp(i Phi_ij sigma_y^i sigma_y^j) once every alpha is zero, and maximally entangling at
|Phi_ij| = pi/4. For piecewise constant amplitudes, every alpha is linear in the amplitudes of the segments and Phi is
quadratic, so the AM gate which closes every mode (and, to be robust, every derivative of alpha with respect to the
mode frequencies, which makes it insensitive to their drifts to first order) with the least power is the eigenvector
of the largest eigenvalue of Phi on the null space of the closure constraints (as in T. Choi et al., PRL 112, 190502
(2014), and P. H. Leung et al., PRL 120, 020501 (2018)). FM gates are found by nonlinear least squares on the
detunings of their segments.
"""

from dataclasses import dataclass

import numpy as np
from scipy import linalg, optimize

from .lib import Pulse

TARGET_PHASE = np.pi / 4


class _Grid:
    """Fine grid of times over the gate, which the integrals over the segments are evaluated on."""

    def __init__(self, duration, n_segments, n_fine):
        self.edges = np.linspace(0.0, duration, n_segments + 1)
        self.dt = duration / (n_segments * n_fine)
        self.t = (np.arange(n_segments * n_fine) + 0.5) * self.dt
        self.segment = np.repeat(np.arange(n_segments), n_fine)
        self.n_segments = n_segments


def _carrier(grid, detunings):
    """cos(mu t) on the fine grid, with the detuning of each segment (whose phase is continuous across segments)."""
    detunings = np.broadcast_to(np.asarray(detunings, dtype=float), (grid.n_segments,))
    rate = detunings[grid.segment]
    phase = np.cumsum(rate * grid.dt) - 0.5 * rate * grid.dt
    return np.cos(phase)


def _displacements(grid, carrier, frequencies):
    """Integrals of cos(mu t) exp(i omega_m t) over each segment (shape (n_modes, n_segments)), and of i t cos(mu t)
    exp(i omega_m t), which is their derivative with respect to omega_m.
    """
    A = np.zeros((len(frequencies), grid.n_segments), dtype=complex)
    dA = np.zeros_like(A)
    for m, omega in enumerate(frequencies):
        integrand = carrier * np.exp(1j * omega * grid.t) * grid.dt
        A[m] = np.bincount(grid.segment, integrand.real, grid.n_segments) + 1j * np.bincount(grid.segment, integrand.imag, grid.n_segments)
        derivative = 1j * grid.t * integrand
        dA[m] = np.bincount(grid.segment, derivative.real, grid.n_segments) + 1j * np.bincount(grid.segment, derivative.imag, grid.n_segments)
    return A, dA


def _phase_matrices(grid, carrier, frequencies):
    """Matrices B_m with Phi_ij = sum_m eta_im eta_jm Omega^T B_m Omega for piecewise constant amplitudes Omega."""
    K = grid.n_segments
    B = np.zeros((len(frequencies), K, K))
    for m, omega in enumerate(frequencies):
        c = np.cos(omega * grid.t)
        s = np.sin(omega * grid.t)
        g = carrier * grid.dt
        for l in range(K):
            mask = grid.segment == l
            u = np.where(mask, g * c, 0.0)
            v = np.where(mask, g * s, 0.0)
            # Sums over earlier times only (with half of the current point, the midpoint rule for t1 = t2)
            U = np.cumsum(u) - 0.5 * u
            V = np.cumsum(v) - 0.5 * v
            values = 2 * g * (s * U - c * V)
            B[m, :, l] = np.bincount(grid.segment, values, K)
        B[m] = 0.5 * (B[m] + B[m].T)
    return B


@dataclass
class MSGate:
    """Mølmer-Sørensen gate: its duration (s), the Rabi frequency (rad/s) and detuning (rad/s) of each segment, the
    modes (rad/s) and Lamb-Dicke parameters (n_ions, n_modes) it was designed for, the ions it entangles, the phase
    Phi between every pair of ions, the displacement left in every mode by every ion, and its derivative with respect to
    the mode frequency (s).
    """
    duration: float
    amplitudes: np.ndarray
    detunings: np.ndarray
    frequencies: np.ndarray
    lamb_dicke: np.ndarray
    targets: tuple
    phases: np.ndarray
    displacements: np.ndarray
    derivatives: np.ndarray

    @property
    def n_segments(self):
        return len(self.amplitudes)

    @property
    def edges(self):
        return np.linspace(0.0, self.duration, self.n_segments + 1)

    @property
    def peak_rabi_frequency(self):
        return float(np.abs(self.amplitudes).max())

    def amplitude(self, t):
        """Rabi frequency (rad/s) at a time t (s) since the start of the gate."""
        k = int(np.clip(t / self.duration * self.n_segments, 0, self.n_segments - 1))
        return self.amplitudes[k]

    def tones(self, start=0.0):
        """Blue and red tones of tiopt.multimode which carry out the gate (with the Rabi frequency as the envelope of
        unit tones, and a detuning which changes in FM gates through the phase).
        """
        from .multimode import Tone

        peak = self.peak_rabi_frequency
        edges = self.edges
        if np.allclose(self.detunings, self.detunings[0]):
            mu = self.detunings[0]
            envelope = lambda t: self.amplitude(t) / peak
            return [
                Tone(peak, mu, list(self.targets), start, self.duration, 0.0, envelope, breakpoints=list(edges[1:-1])),
                Tone(peak, -mu, list(self.targets), start, self.duration, 0.0, envelope, breakpoints=list(edges[1:-1])),
            ]
        # For FM gates, each segment is its own pair of tones, whose phases keep the carrier continuous
        tones = []
        phase = 0.0
        for k in range(self.n_segments):
            mu = self.detunings[k]
            a, b = edges[k], edges[k + 1]
            # The phase of exp(-i mu t) at the start of the segment matches the accumulated phase
            offset = phase - mu * a
            tones.append(Tone(self.amplitudes[k], mu, list(self.targets), start + a, b - a, -offset))
            tones.append(Tone(self.amplitudes[k], -mu, list(self.targets), start + a, b - a, offset))
            phase += mu * (b - a)
        return tones

    def pulses(self, lamb_dicke=0.1, start=0.0):
        """Pulses of tiopt.Pulse (for simulate_pulses with one shared mode) which carry out an AM gate."""
        out = []
        edges = self.edges
        targets = [i + 1 for i in self.targets]
        for k in range(self.n_segments):
            for sign in (1, -1):
                out.append(Pulse(start + edges[k], edges[k + 1] - edges[k], sign * self.detunings[k], abs(self.amplitudes[k]),
                                 0.0 if self.amplitudes[k] >= 0 else np.pi, targets, lamb_dicke))
        return out

    def summary(self):
        """Lines of text which describe the gate."""
        i, j = self.targets
        lines = [
            f"{self.n_segments} segments over {self.duration * 1e6:.2f} µs, peak Rabi frequency 2π × {self.peak_rabi_frequency / (2 * np.pi * 1e3):.2f} kHz",
            f"Phase between ions {i} and {j}: {self.phases[i, j] / np.pi:.4f} π (target ±0.25 π)",
            f"Largest displacement left in any mode: {np.abs(self.displacements).max():.2e}",
            f"Largest derivative of the displacements by the mode frequencies: {np.abs(self.derivatives).max() * 2 * np.pi * 1e3:.2e} per kHz",
        ]
        spectators = [(a, b) for a in range(len(self.phases)) for b in range(a + 1, len(self.phases)) if (a, b) != (min(i, j), max(i, j))]
        if spectators:
            lines.append("Phases with spectator ions: " + ", ".join(f"({a},{b}) {self.phases[a, b] / np.pi:+.3f} π" for a, b in spectators))
        return lines

    def infidelity(self, n_bar=0.0):
        """Estimate of the infidelity of the gate from the displacements left in the modes (for thermal modes with a
        mean phonon number) and from the error of the entangling phase.
        """
        i, j = self.targets
        residual = np.sum(np.abs(self.displacements[[i, j]]) ** 2 * (2 * np.asarray(n_bar) + 1))
        phase_error = abs(abs(self.phases[i, j]) - TARGET_PHASE)
        return float(min(residual + phase_error ** 2, 1.0))


def _evaluate(grid, amplitudes, detunings, frequencies, eta, targets):
    carrier = _carrier(grid, detunings)
    A, dA = _displacements(grid, carrier, frequencies)
    B = _phase_matrices(grid, carrier, frequencies)
    displacements = 1j * eta * (A @ amplitudes)[None, :]
    derivatives = 1j * eta * (dA @ amplitudes)[None, :]
    quadratic = np.einsum("k,mkl,l->m", amplitudes, B, amplitudes)
    phases = np.einsum("im,jm,m->ij", eta, eta, quadratic)
    return displacements, derivatives, phases


def design_am_gate(frequencies, lamb_dicke, duration, detuning, targets=(0, 1), n_segments=None, robust=True, symmetric=True,
                   n_fine=400):
    """Design an amplitude-modulated Mølmer-Sørensen gate which closes every mode and has the least power.

    Args:
        frequencies: Angular frequencies of the modes (rad/s).
        lamb_dicke: Lamb-Dicke parameter of every ion in every mode, with shape (n_ions, n_modes).
        duration: Duration of the gate (s).
        detuning: Detuning of the tones from the qubit transition (rad/s).
        targets: The two ions which the gate entangles.
        n_segments: Number of segments of constant amplitude, which is the fewest that can close every condition (plus
            some freedom to lower the power) by default.
        robust: Specifies whether to also close the derivatives of the displacements with respect to the mode
            frequencies, which makes the gate insensitive to drifts of the mode frequencies to first order.
        symmetric: Specifies whether the amplitudes are symmetric in time.
        n_fine: Number of points of the integrals in each segment.

    Returns:
        MSGate
    """
    frequencies = np.asarray(frequencies, dtype=float)
    eta = np.atleast_2d(np.asarray(lamb_dicke, dtype=float))
    i, j = targets
    involved = [m for m in range(len(frequencies)) if abs(eta[i, m]) + abs(eta[j, m]) >= 1e-12]
    n_conditions = (4 if robust else 2) * len(involved)
    # Every condition takes one free amplitude, and symmetric gates have half as many
    needed = 2 * (n_conditions + 1) + 2 if symmetric else n_conditions + 2
    n_segments = needed if n_segments is None else int(n_segments)
    grid = _Grid(duration, n_segments, n_fine)
    carrier = _carrier(grid, detuning)
    A, dA = _displacements(grid, carrier, frequencies)
    B = _phase_matrices(grid, carrier, frequencies)
    # Closure of every mode which either target ion moves, as real equations in the amplitudes
    rows = []
    for m in involved:
        rows += [A[m].real, A[m].imag]
        if robust:
            rows += [dA[m].real, dA[m].imag]
    constraints = np.array(rows) if rows else np.zeros((0, n_segments))
    if symmetric:
        mirror = np.zeros((n_segments // 2, n_segments))
        for k in range(n_segments // 2):
            mirror[k, k] = 1.0
            mirror[k, n_segments - 1 - k] = -1.0
        constraints = np.vstack([constraints, mirror]) if len(constraints) else mirror
    # Scale the rows so that the null space is found reliably
    norms = np.linalg.norm(constraints, axis=1)
    constraints = constraints[norms > 0] / norms[norms > 0, None]
    null = linalg.null_space(constraints, rcond=1e-10) if len(constraints) else np.eye(n_segments)
    if null.shape[1] == 0:
        raise ValueError(f"{n_segments} segments are too few to close the {n_conditions} conditions of {len(involved)} modes; use at least {needed}")
    G = np.einsum("m,mkl->kl", eta[i] * eta[j], B)
    S = null.T @ G @ null
    values, vectors = np.linalg.eigh(0.5 * (S + S.T))
    k = int(np.argmax(np.abs(values)))
    if abs(values[k]) < 1e-300:
        raise ValueError("the gate cannot entangle the target ions with these modes and detuning")
    amplitudes = null @ vectors[:, k] * np.sqrt(TARGET_PHASE / abs(values[k]))
    displacements, derivatives, phases = _evaluate(grid, amplitudes, detuning, frequencies, eta, targets)
    return MSGate(duration, amplitudes, np.full(n_segments, float(detuning)), frequencies, eta, tuple(targets), phases, displacements, derivatives)


def design_fm_gate(frequencies, lamb_dicke, duration, detuning, rabi_frequency=None, targets=(0, 1), n_segments=10,
                   robust=False, n_fine=200, seed=0):
    """Design a frequency-modulated Mølmer-Sørensen gate (P. H. Leung et al., PRL 120, 020501 (2018)), whose detuning
    is constant in each of its symmetric segments and whose Rabi frequency is constant, by nonlinear least squares on
    the closure of every mode and the entangling phase.

    Args:
        frequencies: Angular frequencies of the modes (rad/s).
        lamb_dicke: Lamb-Dicke parameter of every ion in every mode, with shape (n_ions, n_modes).
        duration: Duration of the gate (s).
        detuning: Starting detuning of every segment (rad/s).
        rabi_frequency: Rabi frequency (rad/s), which is optimized too if None.
        targets: The two ions which the gate entangles.
        n_segments: Number of segments.
        robust: Specifies whether to also close the derivatives with respect to the mode frequencies.
        n_fine: Number of points of the integrals in each segment.
        seed: Seed of the small random start of the detunings.

    Returns:
        MSGate
    """
    frequencies = np.asarray(frequencies, dtype=float)
    eta = np.atleast_2d(np.asarray(lamb_dicke, dtype=float))
    i, j = targets
    grid = _Grid(duration, n_segments, n_fine)
    half = (n_segments + 1) // 2
    rng = np.random.default_rng(seed)
    scale = 2 * np.pi / duration

    def unpack(x):
        mus = detuning + scale * x[:half]
        full = np.concatenate([mus, mus[: n_segments - half][::-1]])
        omega = rabi_frequency if rabi_frequency is not None else abs(x[half]) * scale * 10
        return full, omega

    def residuals(x):
        mus, omega = unpack(x)
        amplitudes = np.full(n_segments, omega)
        displacements, derivatives, phases = _evaluate(grid, amplitudes, mus, frequencies, eta, targets)
        r = [displacements[[i, j]].real.ravel(), displacements[[i, j]].imag.ravel()]
        if robust:
            r += [derivatives[[i, j]].real.ravel() * scale, derivatives[[i, j]].imag.ravel() * scale]
        r.append([(abs(phases[i, j]) - TARGET_PHASE) * 10])
        return np.concatenate(r)

    x0 = np.concatenate([0.01 * rng.standard_normal(half), [] if rabi_frequency is not None else [0.1]])
    result = optimize.least_squares(residuals, x0, xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=2000)
    mus, omega = unpack(result.x)
    amplitudes = np.full(n_segments, omega)
    displacements, derivatives, phases = _evaluate(grid, amplitudes, mus, frequencies, eta, targets)
    return MSGate(duration, amplitudes, mus, frequencies, eta, tuple(targets), phases, displacements, derivatives)


def gate_time_model(model, n_ions, separation=1):
    """Duration (µs) of a two-qubit gate in a chain of n_ions with the target ions separation apart, for the gate models
    of the QCCD architecture study of P. Murali et al. (ISCA 2020): "AM1" (100 d - 22), "AM2" (38 d + 10), "PM"
    (5 d + 160), or "FM" (max(13.33 N - 54, 100)).
    """
    d = max(int(separation), 1)
    if model == "AM1":
        return max(100 * d - 22, 1.0)
    if model == "AM2":
        return max(38 * d + 10, 1.0)
    if model == "PM":
        return max(5 * d + 160, 1.0)
    if model == "FM":
        return max(13.33 * n_ions - 54, 100.0)
    raise ValueError(f"unknown gate model {model!r}; choose from AM1, AM2, PM, FM")


GATE_MODELS = ("AM1", "AM2", "PM", "FM")
