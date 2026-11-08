"""Level structure of the trapped ion species in a magnetic field, and how strongly a beam drives each Zeeman line,
like the ion and laser descriptions of IonSim.jl.

The S1/2, P1/2, P3/2, D3/2, and D5/2 levels split into Zeeman sublevels with the Landé g-factors of LS coupling, so the
S1/2-D5/2 quadrupole (E2) transition of the qubit splits into 10 lines, and the S-P dipole (E1) transitions into 6 or
8. Which of them a beam drives, and how strongly, depends on its wavevector k and polarization epsilon relative to the
magnetic field: an E1 line with m' = m + q is driven in proportion to the Clebsch-Gordan coefficient times the
spherical component epsilon_q, and an E2 line in proportion to the Clebsch-Gordan coefficient times
epsilon . c^(q) . k (D. F. V. James, Appl. Phys. B 66, 181 (1998); C. F. Roos, PhD thesis, Innsbruck (2000)). For the
odd isotopes, the hyperfine structure of the ground state is found by diagonalizing A I.J plus the Zeeman Hamiltonian,
which gives the Breit-Rabi levels and the field sensitivity of the hyperfine qubit (e.g. its clock transition).
"""

from dataclasses import dataclass
from fractions import Fraction
from math import factorial, sqrt

import numpy as np

from .lib import C_LIGHT, HBAR, Ion

MU_B_OVER_H = 1.39962449361e10  # Bohr magneton over Planck's constant (Hz/T)
G_S = 2.00231930436
GAUSS = 1e-4  # T
EPSILON_0 = 8.8541878128e-12

# Orbital angular momentum, spin, and total angular momentum of each level
LEVELS = {
    "S1/2": (0, Fraction(1, 2), Fraction(1, 2)),
    "P1/2": (1, Fraction(1, 2), Fraction(1, 2)),
    "P3/2": (1, Fraction(1, 2), Fraction(3, 2)),
    "D3/2": (2, Fraction(1, 2), Fraction(3, 2)),
    "D5/2": (2, Fraction(1, 2), Fraction(5, 2)),
}

# Nuclear spin, ground state hyperfine constant A (Hz), and nuclear magnetic moment (in nuclear magnetons) of the odd
# isotopes
HYPERFINE = {
    "43Ca+": (Fraction(7, 2), -806.4020716e6, -1.3176),
    "9Be+": (Fraction(3, 2), -625.008837e6, -1.1778),
    "25Mg+": (Fraction(5, 2), -596.254376e6, -0.85545),
    "171Yb+": (Fraction(1, 2), 12642.812118e6, 0.4919),
    "133Ba+": (Fraction(1, 2), 9925.45e6, -0.7717),
    "137Ba+": (Fraction(3, 2), 4018.87e6, 0.9374),
}
PROTON_ELECTRON_MASS_RATIO = 1836.15267343


def lande_g(level):
    """Landé g-factor of a level (e.g. "D5/2") in LS coupling."""
    L, S, J = LEVELS[level]
    L, S, J = float(L), float(S), float(J)
    return (J * (J + 1) - S * (S + 1) + L * (L + 1)) / (2 * J * (J + 1)) + G_S * (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))


def sublevels(level):
    """Magnetic quantum numbers of a level, from -J to J."""
    J = LEVELS[level][2]
    return [Fraction(int(2 * m), 2) for m in np.arange(-float(J), float(J) + 0.5, 1.0)]


def clebsch_gordan(j1, m1, j2, m2, J, M):
    """Clebsch-Gordan coefficient <j1 m1; j2 m2 | J M> by the Racah formula (arguments may be half-integers)."""
    j1, m1, j2, m2, J, M = (Fraction(x).limit_denominator(2) for x in (j1, m1, j2, m2, J, M))
    if m1 + m2 != M or not abs(j1 - j2) <= J <= j1 + j2 or abs(m1) > j1 or abs(m2) > j2 or abs(M) > J:
        return 0.0
    def f(x):
        return factorial(int(x))
    prefactor = sqrt((2 * J + 1) * f(J + j1 - j2) * f(J - j1 + j2) * f(j1 + j2 - J) / f(j1 + j2 + J + 1))
    prefactor *= sqrt(f(J + M) * f(J - M) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2))
    total = 0.0
    for k in range(0, int(j1 + j2 - J) + 1):
        terms = (k, j1 + j2 - J - k, j1 - m1 - k, j2 + m2 - k, J - j2 + m1 + k, J - j1 - m2 + k)
        if any(t < 0 for t in terms):
            continue
        denominator = 1
        for t in terms:
            denominator *= f(t)
        total += (-1) ** k / denominator
    return prefactor * total


def _frame(B_direction):
    """Rotation whose rows are the axes of the frame with z along the magnetic field."""
    z = np.asarray(B_direction, dtype=float)
    z = z / np.linalg.norm(z)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.array([x, y, z])


def _quadrupole_tensors():
    """Normalized spherical tensors c^(q) of the quadrupole coupling, for q from -2 to 2 (James 1998)."""
    c = {}
    c[0] = np.diag([-1.0, -1.0, 2.0]).astype(complex) / sqrt(6)
    for sign in (1, -1):
        c[sign] = 0.5 * np.array([[0, 0, -sign], [0, 0, -1j], [-sign, -1j, 0]], dtype=complex)
        c[2 * sign] = np.array([[1, sign * 1j, 0], [sign * 1j, -1, 0], [0, 0, 0]], dtype=complex) / sqrt(6)
    return c


QUADRUPOLE_TENSORS = _quadrupole_tensors()


def polarization_vector(k_direction, angle=0.0, reference=(0.0, 0.0, 1.0)):
    """Linear polarization perpendicular to a wavevector, rotated by angle (degrees) about it from the direction in the
    plane of k and reference (by default the vertical, i.e. TM for a beam from a grating; 90 degrees gives TE).
    """
    k = np.asarray(k_direction, dtype=float)
    k = k / np.linalg.norm(k)
    reference = np.asarray(reference, dtype=float)
    first = reference - (reference @ k) * k
    if np.linalg.norm(first) < 1e-9:
        first = np.cross(k, [1.0, 0.0, 0.0])
    first /= np.linalg.norm(first)
    second = np.cross(k, first)
    a = np.radians(angle)
    return np.cos(a) * first + np.sin(a) * second


def e2_coupling(m, m_prime, k_direction, polarization, B_direction, J=Fraction(1, 2), J_prime=Fraction(5, 2)):
    """Geometric strength (between 0 and 1) with which a beam drives the E2 line from |J m> to |J' m'>, i.e. the
    Clebsch-Gordan coefficient times |epsilon . c^(q) . k|, with the vectors in the frame of the magnetic field.
    """
    q = int(Fraction(m_prime) - Fraction(m))
    if abs(q) > 2:
        return 0.0
    R = _frame(B_direction)
    k = R @ (np.asarray(k_direction, dtype=float) / np.linalg.norm(k_direction))
    e = R @ (np.asarray(polarization, dtype=complex) / np.linalg.norm(polarization))
    geometric = abs(e @ QUADRUPOLE_TENSORS[q] @ k)
    return abs(clebsch_gordan(J, m, 2, q, J_prime, m_prime)) * geometric


def e1_coupling(m, m_prime, polarization, B_direction, J=Fraction(1, 2), J_prime=Fraction(1, 2)):
    """Geometric strength (between 0 and 1) with which light of a polarization drives the E1 line from |J m> to
    |J' m'>, i.e. the Clebsch-Gordan coefficient times |epsilon_q| in the frame of the magnetic field.
    """
    q = int(Fraction(m_prime) - Fraction(m))
    if abs(q) > 1:
        return 0.0
    R = _frame(B_direction)
    e = R @ (np.asarray(polarization, dtype=complex) / np.linalg.norm(polarization))
    components = {0: e[2], 1: -(e[0] + 1j * e[1]) / sqrt(2), -1: (e[0] - 1j * e[1]) / sqrt(2)}
    return abs(clebsch_gordan(J, m, 1, q, J_prime, m_prime)) * abs(components[q])


@dataclass
class Line:
    """Zeeman line of a transition: the magnetic quantum numbers of its lower and upper sublevels, its shift from the
    unperturbed transition (Hz), the geometric strength with which the beam drives it, and its Rabi frequency (rad/s)
    if the intensity was given.
    """
    m: Fraction
    m_prime: Fraction
    shift: float
    strength: float
    rabi_frequency: float = None

    @property
    def delta_m(self):
        return int(self.m_prime - self.m)

    def label(self):
        return f"S{_half(self.m)} -> D{_half(self.m_prime)}"


def _half(m):
    m = Fraction(m)
    return f"{'+' if m >= 0 else '-'}{abs(m.numerator)}/{m.denominator}" if m.denominator != 1 else f"{int(m):+d}"


def quadrupole_rabi_frequency(ion, intensity, strength):
    """Rabi frequency (rad/s) of an E2 line of the S-D transition at an intensity (W/m^2) with a geometric strength
    (see e2_coupling), Omega = E sqrt(15 pi epsilon_0 Gamma / (hbar k^3)) times the strength (James 1998), where Gamma is
    the decay rate of the D state.
    """
    k = ion.struct.omega_SD / C_LIGHT
    E = np.sqrt(2 * intensity / (C_LIGHT * EPSILON_0))
    return E * np.sqrt(15 * np.pi * EPSILON_0 * ion.qubit_linewidth / (HBAR * k ** 3)) * strength


def qubit_lines(ion, B, k_direction, polarization, B_direction=(0.0, 0.0, 1.0), intensity=None):
    """Zeeman lines of the S1/2-D5/2 qubit transition of a species in a magnetic field B (T), with the strength (and
    Rabi frequency, at an intensity in W/m^2) with which a beam drives each.

    Returns:
        List of Line
    """
    if ion.qubit_linewidth <= 0:
        raise ValueError(f"{ion.element} uses a hyperfine qubit, which is driven by Raman beams; see hyperfine_levels")
    g_S = lande_g("S1/2")
    g_D = lande_g("D5/2")
    lines = []
    for m in sublevels("S1/2"):
        for m_prime in sublevels("D5/2"):
            if abs(m_prime - m) > 2:
                continue
            shift = MU_B_OVER_H * B * (g_D * float(m_prime) - g_S * float(m))
            strength = e2_coupling(m, m_prime, k_direction, polarization, B_direction)
            rabi = quadrupole_rabi_frequency(ion, intensity, strength) if intensity is not None else None
            lines.append(Line(m, m_prime, shift, strength, rabi))
    return sorted(lines, key=lambda line: line.shift)


def detection_lines(ion, B, polarization, B_direction=(0.0, 0.0, 1.0), upper="P1/2"):
    """Zeeman lines of the S1/2-P dipole transition used for cooling and detection, with the strength with which light
    of a polarization drives each.
    """
    g_S = lande_g("S1/2")
    g_P = lande_g(upper)
    J_prime = LEVELS[upper][2]
    lines = []
    for m in sublevels("S1/2"):
        for m_prime in sublevels(upper):
            if abs(m_prime - m) > 1:
                continue
            shift = MU_B_OVER_H * B * (g_P * float(m_prime) - g_S * float(m))
            lines.append(Line(m, m_prime, shift, e1_coupling(m, m_prime, polarization, B_direction, Fraction(1, 2), J_prime)))
    return sorted(lines, key=lambda line: line.shift)


def best_polarization(target, k_direction, B_direction=(0.0, 0.0, 1.0), n_angles=181):
    """Find the linear polarization angle (degrees, see polarization_vector) which drives a target E2 line (a pair
    (m, m')) most strongly relative to every other line, along with its strength.
    """
    best = (0.0, -1.0, 0.0)
    for angle in np.linspace(0, 180, n_angles):
        e = polarization_vector(k_direction, angle)
        strength = e2_coupling(target[0], target[1], k_direction, e, B_direction)
        others = max(e2_coupling(m, mp, k_direction, e, B_direction) for m in sublevels("S1/2") for mp in sublevels("D5/2")
                     if abs(mp - m) <= 2 and (m, mp) != (Fraction(target[0]), Fraction(target[1])))
        score = strength / max(others, 1e-9)
        if score > best[1]:
            best = (angle, score, strength)
    return best[0], best[2]


@dataclass
class HyperfineLevel:
    """Sublevel of the hyperfine structure of the ground state: its total angular momentum F (at zero field), m_F, and
    energy (Hz) relative to the center of gravity.
    """
    F: Fraction
    m: Fraction
    energy: float


def hyperfine_levels(element, B):
    """Hyperfine sublevels of the ground state of an odd isotope in a magnetic field B (T), by diagonalizing
    A I.J + mu_B B (g_J J_z + g_I I_z) in each subspace of total m (which gives the Breit-Rabi levels).

    Returns:
        List of HyperfineLevel, sorted by energy
    """
    element = element if isinstance(element, str) else element.element
    if element not in HYPERFINE:
        raise ValueError(f"{element} has no hyperfine structure; choose from {', '.join(HYPERFINE)}")
    I, A, mu = HYPERFINE[element]
    J = Fraction(1, 2)
    g_J = lande_g("S1/2")
    g_I = -mu / (float(I) * PROTON_ELECTRON_MASS_RATIO)
    levels = []
    m_values = [Fraction(int(2 * m), 2) for m in np.arange(-float(I + J), float(I + J) + 0.5, 1.0)]
    for m in m_values:
        basis = [(mJ, m - mJ) for mJ in (Fraction(-1, 2), Fraction(1, 2)) if abs(m - mJ) <= I]
        n = len(basis)
        H = np.zeros((n, n))
        for a, (mJ, mI) in enumerate(basis):
            H[a, a] = A * float(mJ * mI) + MU_B_OVER_H * B * (g_J * float(mJ) + g_I * float(mI))
            for b, (mJ2, mI2) in enumerate(basis):
                # The raising and lowering parts of A I.J = A (I_z J_z + (I_+ J_- + I_- J_+) / 2)
                if mJ2 == mJ + 1 and mI2 == mI - 1:
                    value = 0.5 * A * sqrt(float((J - mJ) * (J + mJ + 1) * (I + mI) * (I - mI + 1)))
                    H[b, a] = H[a, b] = value
        energies = np.linalg.eigvalsh(H)
        # At low field, the upper state of each pair belongs to F = I + 1/2 if A > 0 (and to I - 1/2 if A < 0)
        F_values = [I - J, I + J] if A > 0 else [I + J, I - J]
        if n == 1:
            levels.append(HyperfineLevel(I + J, m, energies[0]))
        else:
            for F, energy in zip(F_values, np.sort(energies)):
                levels.append(HyperfineLevel(F, m, energy))
    return sorted(levels, key=lambda level: level.energy)


def hyperfine_transition(element, B, lower=(None, 0), upper=(None, 0)):
    """Frequency (Hz) of a transition between ground hyperfine sublevels (F, m) at a field B (T), where F of None picks
    the lower (for lower) or upper (for upper) hyperfine manifold, which by default gives the m = 0 to m = 0 clock
    transition, along with its sensitivity to the field (Hz/T).
    """
    def frequency(B):
        levels = hyperfine_levels(element, B)
        F_values = sorted({level.F for level in levels})
        def find(F, m, default):
            F = default if F is None else Fraction(F)
            return next(level.energy for level in levels if level.F == F and level.m == Fraction(m))
        A = HYPERFINE[element if isinstance(element, str) else element.element][1]
        # The manifold with the larger F lies higher if A > 0 and lower if A < 0
        low_F, high_F = (F_values[0], F_values[-1]) if A > 0 else (F_values[-1], F_values[0])
        return abs(find(upper[0], upper[1], high_F) - find(lower[0], lower[1], low_F))
    h = max(abs(B) * 1e-4, 1e-9)
    return frequency(B), (frequency(B + h) - frequency(B - h)) / (2 * h)


def zeeman_summary(ion, B, k_direction=None, polarization=None, B_direction=(0.0, 0.0, 1.0), intensity=None):
    """Lines of text which describe the Zeeman structure of a species in a field B (T) and how a beam drives it."""
    ion = ion if isinstance(ion, Ion) else Ion(ion)
    lines = [f"{ion.element} in {B / GAUSS:.3g} G: g(S1/2) = {lande_g('S1/2'):.5f}, g(D5/2) = {lande_g('D5/2'):.5f}"]
    if ion.element in HYPERFINE:
        f0, sensitivity = hyperfine_transition(ion.element, B)
        lines.append(f"Hyperfine clock transition (m = 0 to 0): {f0 / 1e9:.6f} GHz, sensitivity {sensitivity * GAUSS / 1e3:.3f} kHz/G")
    if ion.qubit_linewidth > 0 and k_direction is not None and polarization is not None:
        for line in qubit_lines(ion, B, k_direction, polarization, B_direction, intensity):
            text = f"  {line.label()}: {line.shift / 1e6:+8.3f} MHz, strength {line.strength:.3f}"
            if line.rabi_frequency is not None:
                text += f", Rabi frequency 2π × {line.rabi_frequency / (2 * np.pi * 1e3):.2f} kHz"
            lines.append(text)
    return lines
