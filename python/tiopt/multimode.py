"""Laser-ion dynamics of several ions sharing several motional modes, like the Hamiltonians of IonSim.jl, which the
master equation of quantum.c (one shared mode) cannot follow, e.g. Mølmer-Sørensen gates in longer chains where every
mode is displaced, or the spectator modes of a gate.

Each tone of the light drives its target ions with a Rabi frequency, an envelope, a detuning from the qubit transition,
and a phase, and couples to every mode through the Lamb-Dicke parameter of each ion in that mode (e.g. from
tiopt.crystals). In the interaction picture, ion i sees
    H = (hbar Omega f(t) / 2) sigma_i^+ exp(-i (delta t - phi)) prod_m D_m(i eta_im exp(i omega_m t)) + h.c.,
and the matrix elements of each displacement D(i eta) between Fock states n and n + s turn with exp(i s omega_m t), so
the Hamiltonian is a sum of constant operators with known frequencies. Terms are kept up to a Lamb-Dicke order (the
largest change of any phonon number) and, optionally, up to a rotating wave cutoff on their frequencies, and the
displacements are either the matrix exponentials in the truncated Fock spaces ("truncated", which keeps them unitary)
or the analytic matrix elements of the infinite space ("analytic"), as in IonSim.jl.

The modes start in thermal states, which are sampled by every product of Fock states with a probability above a
cutoff, and the Schrödinger equation is integrated for all of them at once. Quasi-static noise on the qubit frequency
(e.g. from the magnetic field) and on the mode frequencies is averaged over random draws.
"""

from dataclasses import dataclass, field
from itertools import product
from math import factorial

import numpy as np
from scipy import integrate, sparse, special
from scipy.linalg import expm


@dataclass
class Tone:
    """One frequency component of the light, which drives its target ions (starting at 0) with a peak Rabi frequency
    (rad/s), detuned from the qubit transition by detuning (rad/s), with a phase (rad), from start for a duration (s)
    under an envelope (a function of the time since start which peaks at 1, or None for a square pulse). Its Lamb-Dicke
    parameters (with shape (n_ions, n_modes)) default to those of the modes.
    """
    rabi_frequency: float
    detuning: float
    targets: list
    start: float = 0.0
    duration: float = None
    phase: float = 0.0
    envelope: object = None
    lamb_dicke: np.ndarray = None
    breakpoints: list = field(default_factory=list)

    def amplitude(self, t):
        """Envelope times the Rabi frequency at time t (s)."""
        if t < self.start or (self.duration is not None and t > self.start + self.duration):
            return 0.0
        return self.rabi_frequency * (1.0 if self.envelope is None else self.envelope(t - self.start))


@dataclass
class Modes:
    """Motional modes in a simulation: their angular frequencies (rad/s), the Lamb-Dicke parameter of every ion in each
    (shape (n_ions, n_modes)), the highest Fock state kept in each, and the mean phonon number of their thermal states.
    """
    frequencies: np.ndarray
    lamb_dicke: np.ndarray
    cutoffs: list
    n_bar: list

    @classmethod
    def from_crystal(cls, crystal, k_vector, modes, cutoffs=8, n_bar=0.0):
        """Modes of a tiopt.crystals.Crystal for light with a wavevector (1/m), keeping the modes with the given
        indices.
        """
        modes = list(modes)
        cutoffs = [cutoffs] * len(modes) if np.isscalar(cutoffs) else list(cutoffs)
        n_bar = [n_bar] * len(modes) if np.isscalar(n_bar) else list(n_bar)
        return cls(np.abs(crystal.frequencies[modes]), crystal.lamb_dicke(k_vector, modes), cutoffs, n_bar)

    @property
    def n_modes(self):
        return len(self.frequencies)


def displacement_matrix(eta, cutoff, method="truncated"):
    """Matrix of D(i eta) between the Fock states 0 to cutoff."""
    n = cutoff + 1
    if method == "truncated":
        a = np.diag(np.sqrt(np.arange(1, n)), 1)
        return expm(1j * eta * (a + a.T))
    M = np.zeros((n, n), dtype=complex)
    x = eta ** 2
    for m in range(n):
        for k in range(n):
            low, high = min(m, k), max(m, k)
            s = high - low
            value = np.exp(-x / 2) * eta ** s * np.sqrt(factorial(low) / factorial(high)) * special.eval_genlaguerre(low, s, x)
            M[m, k] = value * (1j) ** s
    return M


def thermal_distribution(n_bar, cutoff):
    """Probabilities of the Fock states 0 to cutoff of a thermal state (renormalized within the cutoff)."""
    if n_bar <= 0:
        p = np.zeros(cutoff + 1)
        p[0] = 1.0
        return p
    n = np.arange(cutoff + 1)
    p = n_bar ** n / (n_bar + 1) ** (n + 1)
    return p / p.sum()


@dataclass
class MultimodeResult:
    """Result of simulate: the sample times (s), the excited population of each ion and the mean phonon number of each
    mode at every sample, the population of every qubit basis state at every sample (with ion 0 as the most significant
    bit and 1 meaning excited), and the final density matrix of the qubits.
    """
    times: np.ndarray
    excited_populations: np.ndarray
    mean_phonon_numbers: np.ndarray
    state_populations: np.ndarray
    density_matrix: np.ndarray

    def bell_fidelity(self, pair=(0, 1)):
        """Fidelity of the final state of two ions with the closest Bell state (|gg> + e^{i phi} |ee>) / sqrt(2)."""
        rho = reduced_density_matrix(self.density_matrix, pair)
        return float(0.5 * (rho[0, 0].real + rho[3, 3].real) + abs(rho[0, 3]))


def reduced_density_matrix(rho, keep):
    """Reduce a density matrix of n qubits to the qubits in keep (in their order)."""
    n = int(round(np.log2(rho.shape[0])))
    keep = list(keep)
    rho = rho.reshape([2] * (2 * n))
    others = [q for q in range(n) if q not in keep]
    # Trace out every other qubit, from the last one so that the axes keep their places
    for q in sorted(others, reverse=True):
        rho = np.trace(rho, axis1=q, axis2=q + rho.ndim // 2)
    k = len(keep)
    return rho.reshape(2 ** k, 2 ** k)


class _Hamiltonian:
    """Constant operators and frequencies of the Hamiltonian, grouped by tone and frequency."""

    def __init__(self, n_ions, modes, tones, lamb_dicke_order, rwa_cutoff, method, mode_frequencies):
        self.dims_modes = [c + 1 for c in modes.cutoffs]
        self.n_qubit = 2 ** n_ions
        self.n_motion = int(np.prod(self.dims_modes)) if self.dims_modes else 1
        self.dim = self.n_qubit * self.n_motion
        self.terms = []
        for tone in tones:
            eta = modes.lamb_dicke if tone.lamb_dicke is None else np.asarray(tone.lamb_dicke)
            grouped = {}
            for ion in tone.targets:
                # Raising operator of the qubit of this ion (ion 0 is the most significant bit)
                sigma = sparse.csr_matrix(([1.0], ([1], [0])), shape=(2, 2))
                qubit = sparse.kron(sparse.kron(sparse.identity(2 ** ion), sigma), sparse.identity(2 ** (n_ions - 1 - ion)))
                diagonals = []
                for m in range(modes.n_modes):
                    D = displacement_matrix(eta[ion, m], modes.cutoffs[m], method)
                    order = modes.cutoffs[m] if lamb_dicke_order is None else lamb_dicke_order
                    diagonals.append({s: np.diag(np.diag(D, -s), -s) for s in range(-order, order + 1) if abs(s) <= modes.cutoffs[m]})
                for shifts in product(*[sorted(d) for d in diagonals]):
                    frequency = sum(s * w for s, w in zip(shifts, mode_frequencies)) - tone.detuning
                    if rwa_cutoff is not None and abs(frequency) > rwa_cutoff:
                        continue
                    motion = sparse.identity(1, format="csr")
                    for m, s in enumerate(shifts):
                        motion = sparse.kron(motion, sparse.csr_matrix(diagonals[m][s]))
                    operator = sparse.kron(qubit, motion, format="csr")
                    if operator.nnz == 0:
                        continue
                    key = round(frequency, 3)
                    if key in grouped:
                        grouped[key] = (grouped[key][0] + operator, frequency)
                    else:
                        grouped[key] = (operator, frequency)
            for operator, frequency in grouped.values():
                self.terms.append((tone, frequency, operator.tocsr(), operator.conj().T.tocsr()))

    def apply(self, t, psi):
        out = np.zeros_like(psi)
        for tone, frequency, operator, adjoint in self.terms:
            amplitude = tone.amplitude(t)
            if amplitude == 0.0:
                continue
            phase = np.exp(1j * (frequency * t + tone.phase))
            out += (0.5 * amplitude * phase) * (operator @ psi) + (0.5 * amplitude * np.conj(phase)) * (adjoint @ psi)
        return -1j * out


def _initial_columns(n_ions, modes, initial, thermal_cutoff):
    """Initial states (as columns) and their probabilities: the qubits in a basis state and every product of Fock
    states whose thermal probability is above the cutoff.
    """
    distributions = [thermal_distribution(n_bar, cutoff) for n_bar, cutoff in zip(modes.n_bar, modes.cutoffs)]
    choices = [[n for n in range(len(p)) if p[n] > thermal_cutoff] or [0] for p in distributions]
    q = int(initial, 2) if isinstance(initial, str) else int(initial)
    dims = [c + 1 for c in modes.cutoffs]
    n_motion = int(np.prod(dims)) if dims else 1
    columns, weights = [], []
    for fock in product(*choices):
        index = 0
        for n, d in zip(fock, dims):
            index = index * d + n
        column = np.zeros((2 ** n_ions) * n_motion, dtype=complex)
        column[q * n_motion + index] = 1.0
        columns.append(column)
        weights.append(np.prod([p[n] for p, n in zip(distributions, fock)]) if fock else 1.0)
    weights = np.array(weights)
    return np.column_stack(columns), weights / weights.sum()


def simulate(n_ions, modes, tones, duration, n_samples=100, lamb_dicke_order=2, rwa_cutoff=None, displacement="truncated",
             initial="0", qubit_noise=0.0, mode_noise=0.0, n_noise=1, seed=0, thermal_cutoff=1e-4, rtol=1e-8, atol=1e-10):
    """Simulate tones of light on ions sharing several motional modes.

    Args:
        n_ions: Number of ions.
        modes: Modes.
        tones: List of Tone.
        duration: Duration of the simulation (s).
        n_samples: Number of times at which the populations are sampled.
        lamb_dicke_order: Largest change of any phonon number which is kept, or None for every order.
        rwa_cutoff: Largest frequency of a term which is kept (rad/s), or None for every term.
        displacement: "truncated" or "analytic" matrix elements of the displacements.
        initial: Initial state of the qubits, as a bit string (e.g. "00") or an integer (0 means every ion is in g).
        qubit_noise: Standard deviation of a quasi-static shift of the qubit frequency (rad/s).
        mode_noise: Standard deviation of quasi-static shifts of the mode frequencies (rad/s).
        n_noise: Number of random draws of the noise which are averaged.
        seed: Seed of the random draws.
        thermal_cutoff: Smallest probability of a product of Fock states which is simulated.
        rtol: Relative tolerance of the integrator.
        atol: Absolute tolerance of the integrator.

    Returns:
        MultimodeResult
    """
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, duration, n_samples)
    psi0, weights = _initial_columns(n_ions, modes, initial, thermal_cutoff)
    dims = [c + 1 for c in modes.cutoffs]
    n_motion = int(np.prod(dims)) if dims else 1
    n_qubit = 2 ** n_ions
    excited = np.zeros((n_samples, n_ions))
    phonons = np.zeros((n_samples, modes.n_modes))
    populations = np.zeros((n_samples, n_qubit))
    rho = np.zeros((n_qubit, n_qubit), dtype=complex)
    bits = np.array([[(q >> (n_ions - 1 - i)) & 1 for i in range(n_ions)] for q in range(n_qubit)])
    number = []
    for m, d in enumerate(dims):
        before = int(np.prod(dims[:m])) if m else 1
        after = int(np.prod(dims[m + 1:])) if m + 1 < len(dims) else 1
        number.append(np.tile(np.repeat(np.arange(d), after), before))

    draws = max(int(n_noise), 1)
    for draw in range(draws):
        qubit_shift = rng.normal(0.0, qubit_noise) if qubit_noise > 0 else 0.0
        mode_shifts = rng.normal(0.0, mode_noise, modes.n_modes) if mode_noise > 0 else np.zeros(modes.n_modes)
        shifted = [Tone(**{**tone.__dict__, "detuning": tone.detuning + qubit_shift}) for tone in tones]
        H = _Hamiltonian(n_ions, modes, shifted, lamb_dicke_order, rwa_cutoff, displacement, modes.frequencies + mode_shifts)
        shape = psi0.shape

        def rhs(t, y):
            return H.apply(t, y.reshape(shape)).reshape(-1)

        # Integrate between the edges of the tones, where envelopes may have kinks
        edges = {0.0, duration}
        for tone in tones:
            for t in [tone.start, tone.start + (tone.duration or 0.0)] + [tone.start + b for b in tone.breakpoints]:
                if 0 < t < duration:
                    edges.add(t)
        edges = sorted(edges)
        states = np.zeros((n_samples,) + shape, dtype=complex)
        states[times <= 0.0] = psi0
        y = psi0.reshape(-1).copy()
        for a, b in zip(edges[:-1], edges[1:]):
            samples = np.flatnonzero((times > a) & (times <= b))
            t_eval = np.union1d(times[samples], [b])
            solution = integrate.solve_ivp(rhs, (a, b), y, method="DOP853", t_eval=t_eval, rtol=rtol, atol=atol)
            for k in samples:
                states[k] = solution.y[:, int(np.searchsorted(t_eval, times[k]))].reshape(shape)
            y = solution.y[:, -1]

        probabilities = np.abs(states) ** 2  # (samples, dim, columns)
        probabilities = np.einsum("sdc,c->sd", probabilities, weights).reshape(n_samples, n_qubit, n_motion)
        qubit_populations = probabilities.sum(axis=2)
        populations += qubit_populations / draws
        excited += qubit_populations @ bits / draws
        motion = probabilities.sum(axis=1)
        for m in range(modes.n_modes):
            phonons[:, m] += motion @ number[m] / draws
        final = states[-1].reshape(n_qubit, n_motion, -1)
        rho += np.einsum("qmc,pmc,c->qp", final, final.conj(), weights) / draws
    return MultimodeResult(times, excited, phonons, populations, rho)
