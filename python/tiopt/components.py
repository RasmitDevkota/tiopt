"""Simulation and optimization of integrated photonic components, i.e. straight waveguides, waveguide bends, and grating
couplers which focus light onto an ion.

The modes are found with a finite-difference mode solver on a one-dimensional grid (with perfectly matched layers at the
edges, so that light which leaks into the substrate or radiates from a bend leaves the grid), and rectangular waveguides
are reduced to one dimension with the effective index method. Bends are mapped onto straight waveguides with the
conformal transformation n(x) -> n(x) exp(x / R), and the strength and directionality of a grating are found with
first-order perturbation theory (the volume current method), which is accurate for shallow etches and low index
contrast and overestimates the strength of strong gratings. Lengths are in µm and losses are in dB unless stated
otherwise.
"""

from dataclasses import astuple, dataclass, field, replace

import numpy as np
from scipy import optimize, sparse
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import eigs, spsolve

DB = 10 / np.log(10)  # dB per neper of power
CM = 1e4  # µm per cm
PML_SIGMA = 4.0

# ---------------------------------------------------------------------------------------------------------------------
# One-dimensional mode solver
# ---------------------------------------------------------------------------------------------------------------------


def _grid(x_min, x_max, h, layers, background):
    """Sample the permittivity of a stack of layers (start, end, index) on a grid, averaging over each cell."""
    x = np.arange(x_min, x_max + 0.5 * h, h)
    eps = np.full(len(x), complex(background) ** 2)
    for start, end, n in layers:
        overlap = np.clip(np.minimum(x + 0.5 * h, end) - np.maximum(x - 0.5 * h, start), 0.0, h) / h
        eps += overlap * (complex(n) ** 2 - complex(background) ** 2)
    return x, eps


def _stretch(n_points, n_pml):
    """Complex coordinate stretching of the perfectly matched layers at nodes and half nodes."""
    def profile(index):
        depth = np.maximum(np.maximum(n_pml[0] - index, index - (n_points - 1 - n_pml[1])), 0.0)
        width = np.where(index < n_points / 2, max(n_pml[0], 1), max(n_pml[1], 1))
        return 1 + 1j * PML_SIGMA * (depth / width) ** 2

    nodes = profile(np.arange(n_points, dtype=float))
    halves = profile(np.arange(n_points - 1, dtype=float) + 0.5)
    return nodes, halves


def _operator(eps, h, k_0, polarization, n_pml=(0, 0)):
    """Assemble the operator whose eigenvalues are beta^2, i.e. d^2/dx^2 + k_0^2 eps for TE (the electric field parallel
    to the layers) and eps d/dx (1/eps d/dx) + k_0^2 eps for TM (the magnetic field parallel to the layers).
    """
    s, s_half = _stretch(len(eps), n_pml)
    if polarization == "TE":
        p_half = np.ones(len(eps) - 1)
        scale = np.ones(len(eps))
    else:
        p_half = 0.5 * (1 / eps[:-1] + 1 / eps[1:])
        scale = eps
    coupling = p_half / s_half / h ** 2
    upper = scale[:-1] * coupling / s[:-1]
    lower = scale[1:] * coupling / s[1:]
    diagonal = k_0 ** 2 * eps + 0j
    diagonal[:-1] -= scale[:-1] * coupling / s[:-1]
    diagonal[1:] -= scale[1:] * coupling / s[1:]
    return sparse.diags([lower, diagonal, upper], [-1, 0, 1], format="csc")


def _count_modes(eps, h, k_0, polarization, n_cladding, second=False):
    """Count the guided modes of a lossless profile, i.e. the eigenvalues above (k_0 n_cladding)^2."""
    eps = np.real(eps)
    if polarization == "TE":
        diagonal = k_0 ** 2 * eps - 2 / h ** 2
        off = np.full(len(eps) - 1, 1 / h ** 2)
    else:
        # eps d/dx (1/eps d/dx) is similar to the symmetric matrix n M n
        p_half = 0.5 * (1 / eps[:-1] + 1 / eps[1:])
        p_left = np.concatenate(([p_half[0]], p_half))
        p_right = np.concatenate((p_half, [p_half[-1]]))
        diagonal = eps * (k_0 ** 2 - (p_left + p_right) / h ** 2)
        off = np.sqrt(eps[:-1] * eps[1:]) * p_half / h ** 2
    upper = k_0 ** 2 * np.max(eps) * 1.0001
    values = eigh_tridiagonal(diagonal, off, eigvals_only=True, select="v", select_range=((k_0 * n_cladding) ** 2, upper))
    if second:
        # Effective index of the second mode, or the cladding index if there is none
        return len(values), (np.sqrt(np.sort(values)[-2]) / k_0 if len(values) > 1 else n_cladding)
    return len(values)


def _solve_mode(eps, h, k_0, polarization, n_guess, n_pml=(0, 0), core=None):
    """Find the guided mode nearest n_guess, returning its (complex) effective index and field. Of the modes nearest
    n_guess, the one with the most power away from the perfectly matched layers is chosen, since radiation modes of
    the layers can lie closer to n_guess than the guided mode.
    """
    A = _operator(eps, h, k_0, polarization, n_pml)
    values, vectors = eigs(A, k=min(6, len(eps) - 2), sigma=(k_0 * n_guess) ** 2, v0=np.real(eps) - np.min(np.real(eps)) + 1e-3)
    interior = np.zeros(len(eps), dtype=bool)
    interior[n_pml[0] + 1:len(eps) - n_pml[1] - 1] = True
    peak = core if core is not None else np.real(eps) >= 0.5 * (np.max(np.real(eps)) + np.min(np.real(eps[interior])))
    scores = []
    for j in range(len(values)):
        power = np.abs(vectors[:, j]) ** 2
        scores.append(np.sum(power[peak]) / np.sum(power) - np.sum(power[~interior]) / np.sum(power))
    best = int(np.argmax(scores))
    n_eff = np.sqrt(values[best] + 0j) / k_0
    if n_eff.real < 0:
        n_eff = -n_eff
    field_ = vectors[:, best]
    field_ = field_ / field_[np.argmax(np.abs(field_))]
    return n_eff, field_


def _overlap(a, b):
    """Power coupling between two transverse fields on the same grid."""
    return np.abs(np.sum(a * b)) ** 2 / (np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2))


def _loss_db_per_cm(n_eff_imag, wavelength):
    return DB * 2 * (2 * np.pi / wavelength) * n_eff_imag * CM


# ---------------------------------------------------------------------------------------------------------------------
# Straight waveguides
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Stack:
    """Layers of a photonic chip, i.e. a core in silica cladding above a buried oxide on a substrate.

    The defaults are silicon nitride waveguides for 729 nm light on a silicon substrate. Material losses are in dB/cm
    and roughnesses are root mean square values (µm).
    """

    wavelength: float = 0.729
    n_core: float = 2.0
    n_cladding: float = 1.455
    n_substrate: complex = 3.71 + 0.008j
    box_thickness: float = 3.0
    top_cladding: float = 2.0
    core_loss: float = 0.5
    cladding_loss: float = 0.0
    sidewall_roughness: float = 0.003
    surface_roughness: float = 0.0003
    correlation_length: float = 0.05
    polarization: str = "TE"

    @property
    def k_0(self):
        return 2 * np.pi / self.wavelength


@dataclass
class WaveguideResult:
    """Modes and losses (dB/cm) of a straight waveguide."""

    width: float
    thickness: float
    n_eff: complex
    n_modes: int
    confinement: float
    losses: dict
    x: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    lateral: np.ndarray = field(repr=False)
    vertical: np.ndarray = field(repr=False)
    n_second: float = 0.0

    @property
    def loss(self):
        """Total propagation loss (dB/cm)."""
        return sum(self.losses.values())

    @property
    def guided(self):
        return self.n_modes > 0

    def mode_intensity(self):
        """Intensity of the mode over the cross-section, with shape (len(y), len(x))."""
        return np.abs(np.outer(self.vertical, self.lateral)) ** 2


def _payne_lacey(sigma, correlation_length, half_width, n_1, n_2, n_eff, k_0):
    """Scattering loss (1/µm, of power) from the roughness of the two walls of a slab (F. P. Payne and J. P. R. Lacey,
    Opt. Quantum Electron. 26, 977 (1994)).
    """
    if sigma <= 0.0 or n_eff <= n_2 or n_eff >= n_1:
        return 0.0
    U = half_width * k_0 * np.sqrt(n_1 ** 2 - n_eff ** 2)
    W = half_width * k_0 * np.sqrt(n_eff ** 2 - n_2 ** 2)
    V = half_width * k_0 * np.sqrt(n_1 ** 2 - n_2 ** 2)
    delta = (n_1 ** 2 - n_2 ** 2) / (2 * n_1 ** 2)
    x = W * correlation_length / half_width
    gamma = n_2 * V / (n_1 * W * np.sqrt(delta))
    root = np.sqrt((1 + x ** 2) ** 2 + 2 * x ** 2 * gamma ** 2)
    f_e = x * np.sqrt(root + 1 - x ** 2) / root
    g = U ** 2 * V ** 2 / (1 + W)
    return sigma ** 2 / (np.sqrt(2) * k_0 * half_width ** 4 * n_1) * g * f_e


def _vertical(stack, thickness, etch=0.0, eps_etched=None, vacuum_above=False):
    """Grid and permittivity of the stack beneath the center of the core (the top of the core is at y = 0)."""
    n_max = max(abs(stack.n_core), stack.n_cladding)
    h = min(stack.wavelength / (n_max * 40), thickness / 20)
    margin = 1.2
    top = (stack.top_cladding + margin) if vacuum_above else 2.5
    layers = [(-thickness, -etch, stack.n_core), (-thickness - stack.box_thickness - margin, -thickness - stack.box_thickness, stack.n_substrate)]
    if etch > 0.0:
        layers.append((-etch, 0.0, np.sqrt(eps_etched)))
    if vacuum_above:
        layers.append((stack.top_cladding, top, 1.0))
    y, eps = _grid(-thickness - stack.box_thickness - margin, top, h, layers, stack.n_cladding)
    n_pml = int(round(0.8 / h))
    return y, eps, h, (n_pml, n_pml)


def _vertical_mode(stack, thickness):
    """Fundamental vertical mode of the core region (with leakage into the substrate), the effective index without the
    substrate, and the number of vertical modes, or None if there are none.
    """
    k_0 = stack.k_0
    y, eps_lossless, h, n_pml = _vertical(replace(stack, n_substrate=stack.n_cladding), thickness)
    n_modes = _count_modes(eps_lossless, h, k_0, stack.polarization, stack.n_cladding)
    if n_modes == 0:
        return None
    # The fundamental mode is the one closest to the core index
    n_lossless, _ = _solve_mode(eps_lossless, h, k_0, stack.polarization, stack.n_core, n_pml)
    _, eps, _, _ = _vertical(stack, thickness)
    n_eff, mode = _solve_mode(eps, h, k_0, stack.polarization, n_lossless.real, n_pml)
    return n_eff, n_lossless.real, mode, y, n_modes


def _lateral_grid(stack, width, n_core_eff, extra_right=0.0):
    h = min(stack.wavelength / (abs(n_core_eff) * 30), width / 25)
    left = -0.5 * width - 2.5
    right = 0.5 * width + 2.5 + extra_right
    x, eps = _grid(left, right, h, [(-0.5 * width, 0.5 * width, n_core_eff)], stack.n_cladding)
    n_pml = int(round(0.8 / h))
    return x, eps, h, (n_pml, n_pml)


# Results of analyze_waveguide, which bends and tapers ask for over and over
_WAVEGUIDE_CACHE = {}


def analyze_waveguide(stack, width, thickness):
    """Find the fundamental mode, the number of guided modes, and the losses (dB/cm) of a straight waveguide (see
    _analyze_waveguide), remembering recent results.
    """
    key = (astuple(stack), round(float(width), 6), round(float(thickness), 6))
    if key not in _WAVEGUIDE_CACHE:
        if len(_WAVEGUIDE_CACHE) > 2000:
            _WAVEGUIDE_CACHE.clear()
        _WAVEGUIDE_CACHE[key] = _analyze_waveguide(stack, width, thickness)
    return _WAVEGUIDE_CACHE[key]


def _analyze_waveguide(stack, width, thickness):
    """Find the fundamental mode, the number of guided modes, and the losses (dB/cm) of a straight waveguide.

    Args:
        stack: Stack of the chip.
        width: Width of the core.
        thickness: Thickness of the core.

    Returns:
        WaveguideResult, or None if the waveguide does not guide light
    """
    k_0 = stack.k_0
    vertical_result = _vertical_mode(stack, thickness)
    if vertical_result is None:
        return None
    n_vertical, n_vertical_lossless, vertical, y, n_vertical_modes = vertical_result

    # Quasi-TE modes have their electric field normal to the side walls, i.e. they are TM in the lateral direction
    lateral_polarization = "TM" if stack.polarization == "TE" else "TE"
    x, eps, h, n_pml = _lateral_grid(stack, width, n_vertical)
    n_lateral_modes, n_second = _count_modes(_lateral_grid(stack, width, n_vertical_lossless)[1], h, k_0, lateral_polarization, stack.n_cladding, True)
    if n_lateral_modes == 0:
        return None
    n_eff, lateral = _solve_mode(eps, h, k_0, lateral_polarization, n_vertical_lossless, n_pml)

    # A second vertical mode adds its own lateral modes
    n_modes = n_lateral_modes
    if n_vertical_modes > 1:
        n_modes += max(n_vertical_modes - 1, 0)

    in_core_x = np.abs(x) <= 0.5 * width
    in_core_y = (y <= 0.0) & (y >= -thickness)
    confinement = (np.sum(np.abs(lateral[in_core_x]) ** 2) / np.sum(np.abs(lateral) ** 2)) * (np.sum(np.abs(vertical[in_core_y]) ** 2) / np.sum(np.abs(vertical) ** 2))

    n_real = n_eff.real
    losses = {
        "substrate leakage": _loss_db_per_cm(max(n_eff.imag, 0.0), stack.wavelength),
        "sidewall scattering": DB * CM * _payne_lacey(stack.sidewall_roughness, stack.correlation_length, 0.5 * width, n_vertical_lossless, stack.n_cladding, n_real, k_0),
        "surface scattering": DB * CM * _payne_lacey(stack.surface_roughness, stack.correlation_length, 0.5 * thickness, stack.n_core, stack.n_cladding, n_vertical_lossless, k_0),
        "absorption": confinement * stack.core_loss + (1 - confinement) * stack.cladding_loss,
    }
    if n_vertical_modes > 1:
        n_second = max(n_second, stack.n_cladding + 1e-3)
    return WaveguideResult(width, thickness, n_eff, n_modes, confinement, losses, x, y, lateral, vertical, n_second)


def optimize_waveguide(stack, width=0.6, thickness=0.15, width_bounds=(0.2, 3.0), thickness_bounds=(0.05, 0.5), single_mode=True, min_confinement=0.3, max_evaluations=80):
    """Choose the width and thickness of a straight waveguide which minimize its propagation loss.

    Args:
        stack: Stack of the chip.
        width: Initial width.
        thickness: Initial thickness.
        width_bounds: Smallest and largest widths.
        thickness_bounds: Smallest and largest thicknesses.
        single_mode: Whether the waveguide has to guide a single mode (of its polarization).
        min_confinement: Smallest fraction of the power in the core, which keeps the mode tight enough to bend (the
            loss falls as the mode spreads into the cladding, so without it the optimizer finds barely-guided modes).
        max_evaluations: Largest number of designs to evaluate.

    Returns:
        WaveguideResult of the best design
    """
    def objective(u):
        result = analyze_waveguide(stack, *u)
        if result is None:
            return 1e3
        # Multimode designs are penalized by how far their second mode is above cutoff, so that the optimizer is led
        # back to single-mode designs
        penalty = 0.0
        if single_mode and result.n_modes > 1:
            penalty = 1.0 + 100 * (result.n_second - stack.n_cladding) / (stack.n_core - stack.n_cladding) + (result.n_modes - 2)
        penalty += 100 * max(min_confinement - result.confinement, 0.0)
        return np.log10(result.loss) + penalty

    best = optimize.minimize(objective, (width, thickness), method="Nelder-Mead", bounds=(width_bounds, thickness_bounds),
                             options=dict(maxfev=max_evaluations, xatol=1e-3, fatol=1e-4, initial_simplex=[(width, thickness), (1.3 * width, thickness), (width, 1.3 * thickness)]))
    return analyze_waveguide(stack, *np.clip(best.x, [width_bounds[0], thickness_bounds[0]], [width_bounds[1], thickness_bounds[1]]))


# ---------------------------------------------------------------------------------------------------------------------
# Bends
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class BendResult:
    """Losses and geometry of a waveguide bend, whose curvature rises linearly from zero over its Euler sections."""

    radius: float
    width: float
    bend_width: float
    euler_fraction: float
    offset: float
    angle: float
    loss: float
    radiation_loss: float
    propagation_loss: float
    transition_loss: float
    length: float
    effective_radius: float
    loss_per_cm: float
    path: np.ndarray = field(repr=False)
    sections: list = field(repr=False)
    taper_length: float = 0.0
    taper_profile: str = "linear"
    taper_loss: float = 0.0


def _bend_mode(stack, n_vertical, n_vertical_lossless, width, radius, polarization, n_guess):
    """Mode of a bend (or of a straight waveguide if radius is infinite) on a lateral grid shared by every section."""
    k_0 = stack.k_0
    caustic = 0.0 if not np.isfinite(radius) else radius * max(n_vertical_lossless / stack.n_cladding - 1, 0.0)
    extra = min(caustic + 2.0, 60.0) if np.isfinite(radius) else 0.0
    x, eps, h, n_pml = _lateral_grid(stack, width, n_vertical, extra_right=extra)
    if np.isfinite(radius):
        eps = eps * np.exp(2 * x / radius)
        guess = n_guess * (1 + 0.5 * width / radius)
    else:
        guess = n_guess
    n_eff, mode = _solve_mode(eps, h, k_0, polarization, guess, n_pml, core=np.abs(x) <= 0.5 * width + 0.1)
    return x, n_eff, mode


def _bend_curvature(radius, angle, euler_fraction, n_euler_steps):
    """Curvature and length of each section of a bend, with Euler sections whose curvature rises linearly."""
    sections = []
    euler_angle = 0.5 * euler_fraction * angle
    circle_angle = (1 - euler_fraction) * angle
    euler_length = 2 * euler_angle * radius
    steps = [(i + 0.5) / n_euler_steps for i in range(n_euler_steps)] if euler_angle > 0 else []
    for f in steps:
        sections.append((f / radius, euler_length / n_euler_steps))
    if circle_angle > 0:
        sections.append((1 / radius, circle_angle * radius))
    for f in reversed(steps):
        sections.append((f / radius, euler_length / n_euler_steps))
    return sections


def _bend_path(sections, n_points=400):
    s_total = sum(length for _, length in sections)
    s = np.linspace(0, s_total, n_points)
    curvature = np.zeros(n_points)
    start = 0.0
    for kappa, length in sections:
        curvature[(s >= start) & (s <= start + length + 1e-12)] = kappa
        start += length
    heading = np.concatenate(([0.0], np.cumsum(0.5 * (curvature[1:] + curvature[:-1]) * np.diff(s))))
    x = np.concatenate(([0.0], np.cumsum(0.5 * (np.cos(heading[1:]) + np.cos(heading[:-1])) * np.diff(s))))
    y = np.concatenate(([0.0], np.cumsum(0.5 * (np.sin(heading[1:]) + np.sin(heading[:-1])) * np.diff(s))))
    return np.stack([x, y, heading, s, curvature], axis=-1)


TAPER_PROFILES = ("linear", "exponential", "parabolic")


def taper_widths(width_from, width_to, u, profile="linear"):
    """Widths of a taper at fractions u of its length, where the width changes linearly, exponentially, or so that its
    square changes linearly (parabolic, which changes the width slowly where the waveguide is narrow and the mode is most
    sensitive to it).
    """
    u = np.asarray(u, dtype=float)
    if profile == "exponential":
        return width_from * (width_to / width_from) ** u
    if profile == "parabolic":
        return np.sqrt(width_from ** 2 + (width_to ** 2 - width_from ** 2) * u)
    if profile != "linear":
        raise ValueError(f"unknown taper profile {profile!r}; choose from {', '.join(TAPER_PROFILES)}")
    return width_from + (width_to - width_from) * u


def _taper(stack, thickness, n_vertical, n_vertical_lossless, polarization, width_from, width_to, length, profile, n_steps):
    """Loss of a straight taper between two widths, as the light it scatters out of the guided mode (the mismatch of
    the local modes of consecutive steps, added up with the phase of radiation at the cladding light line so that long
    tapers scatter little) plus its propagation loss, and the local mode of every step.
    """
    k_0 = stack.k_0
    widths = taper_widths(width_from, width_to, np.linspace(0, 1, n_steps + 1), profile)
    step = length / n_steps
    modes = []
    losses = []
    for w in widths:
        local = analyze_waveguide(stack, w, thickness)
        if local is None:
            return None
        x, n_eff, mode = _bend_mode(stack, n_vertical, n_vertical_lossless, w, np.inf, polarization, local.n_eff.real)
        modes.append((x, mode, w, local.n_eff.real))
        losses.append(local.loss)
    propagation = np.trapezoid(losses, dx=step) / CM
    radiated = 0.0j
    phase = 0.0
    for (x_a, mode_a, _, n_a), (x_b, mode_b, _, n_b) in zip(modes[:-1], modes[1:]):
        previous = np.interp(x_b, x_a, mode_a.real) + 1j * np.interp(x_b, x_a, mode_a.imag)
        mismatch = np.sqrt(max(1 - _overlap(previous, mode_b), 0.0))
        radiated += mismatch * np.exp(1j * phase)
        phase += k_0 * (0.5 * (n_a + n_b) - stack.n_cladding) * step
    scattered = min(abs(radiated) ** 2, 0.999)
    sections = [((0.0, step), (x, mode, w)) for x, mode, w, _ in modes[1:]]
    return -DB * np.log(1 - scattered) + propagation, sections, modes[-1]


def analyze_bend(stack, width, thickness, radius, angle=90.0, bend_width=None, euler_fraction=0.0, offset=0.0, n_euler_steps=4,
                 taper_length=0.0, taper_profile="linear", n_taper_steps=16):
    """Find the losses of a bend between two straight waveguides.

    Args:
        stack: Stack of the chip.
        width: Width of the straight waveguides.
        thickness: Thickness of the core.
        radius: Smallest radius of curvature, i.e. that of the circular section.
        angle: Angle the bend turns through (degrees).
        bend_width: Width of the circular section, which the Euler sections taper to (the straight width by default).
        euler_fraction: Fraction of the angle turned in the Euler sections (0 for a circular bend, 1 for a full Euler bend).
        offset: Lateral offset of the straight waveguides towards the outside of the bend, which matches them to the bent
            mode (µm).
        n_euler_steps: Number of steps of constant curvature which each Euler section is divided into.
        taper_length: Length of straight tapers before and after the bend, which change the width between the straight
            waveguides and the bend (so that the whole bend has bend_width); 0 tapers the width in the Euler sections
            with the curvature instead.
        taper_profile: Profile of the tapers (see TAPER_PROFILES).
        n_taper_steps: Number of steps which each taper is divided into.

    Returns:
        BendResult
    """
    bend_width = width if bend_width is None else bend_width
    k_0 = stack.k_0
    vertical_result = _vertical_mode(stack, thickness)
    if vertical_result is None:
        return None
    n_vertical, n_vertical_lossless = vertical_result[:2]
    polarization = "TM" if stack.polarization == "TE" else "TE"
    straight = analyze_waveguide(stack, width, thickness)
    if straight is None:
        return None

    sections = _bend_curvature(radius, np.radians(angle), euler_fraction, n_euler_steps)
    max_curvature = 1 / radius
    previous_x, _, previous_mode = _bend_mode(stack, n_vertical, n_vertical_lossless, width, np.inf, polarization, straight.n_eff.real)
    tapered = taper_length > 0 and abs(bend_width - width) > 1e-9
    taper_loss = 0.0
    taper_sections = []
    if tapered:
        taper = _taper(stack, thickness, n_vertical, n_vertical_lossless, polarization, width, bend_width, taper_length, taper_profile, n_taper_steps)
        if taper is None:
            return None
        # The taper out of the bend is the mirror image of the taper into it, so it loses as much
        taper_loss, taper_sections, (previous_x, previous_mode, _, _) = taper
        taper_loss *= 2
    previous_shift = offset
    transition = 0.0
    radiation = 0.0
    propagation = 0.0
    profiles = []
    for kappa, length in sections:
        local_width = bend_width if tapered else width + (bend_width - width) * kappa / max_curvature
        local = analyze_waveguide(stack, local_width, thickness)
        x, n_eff, mode = _bend_mode(stack, n_vertical, n_vertical_lossless, local_width, 1 / kappa, polarization, local.n_eff.real)
        # Fields are compared on the grid of this section, with the previous one shifted by any offset
        previous = np.interp(x - previous_shift, previous_x, previous_mode.real) + 1j * np.interp(x - previous_shift, previous_x, previous_mode.imag)
        transition += -DB * np.log(max(_overlap(previous, mode), 1e-30))
        radiation += max(_loss_db_per_cm(n_eff.imag, stack.wavelength) - local.losses["substrate leakage"], 0.0) * length / CM
        propagation += local.loss * length / CM
        profiles.append((x, mode, local_width))
        previous_x, previous_mode, previous_shift = x, mode, 0.0
    if tapered:
        straight_x, straight_mode = taper_sections[-1][1][:2]
    else:
        straight_x, _, straight_mode = _bend_mode(stack, n_vertical, n_vertical_lossless, width, np.inf, polarization, straight.n_eff.real)
    exit_mode = np.interp(straight_x + offset, previous_x, previous_mode.real) + 1j * np.interp(straight_x + offset, previous_x, previous_mode.imag)
    transition += -DB * np.log(max(_overlap(exit_mode, straight_mode), 1e-30))

    bend_sections = list(zip(sections, profiles))
    all_sections = taper_sections + bend_sections + taper_sections[::-1]
    path = _bend_path([section for section, _ in all_sections])
    bend_length = sum(length for _, length in sections)
    effective_radius = max(np.ptp(path[:, 0]), np.ptp(path[:, 1]))
    loss = transition + radiation + propagation + taper_loss
    return BendResult(radius, width, bend_width, euler_fraction, offset, angle, loss, radiation, propagation, transition, path[-1, 3], effective_radius,
                      (radiation + propagation) / bend_length * CM, path, all_sections, taper_length if tapered else 0.0, taper_profile, taper_loss)


def optimize_bend(stack, width, thickness, radius=20.0, angle=90.0, max_effective_radius=50.0, radius_bounds=(2.0, 200.0), bend_width_bounds=(0.3, 2.0), max_evaluations=60,
                  taper_length=0.0, taper_profile="linear", taper_bounds=None):
    """Choose the radius, width, Euler fraction, offset, and optionally the taper length of a bend which minimize its
    loss within a footprint.

    Args:
        stack: Stack of the chip.
        width: Width of the straight waveguides.
        thickness: Thickness of the core.
        radius: Initial radius.
        angle: Angle the bend turns through (degrees).
        max_effective_radius: Largest extent of the bend (with its tapers) along either axis.
        radius_bounds: Smallest and largest radii.
        bend_width_bounds: Smallest and largest widths of the bend.
        max_evaluations: Largest number of designs to evaluate.
        taper_length: Length of the straight tapers before and after the bend (see analyze_bend), which is the initial
            length if taper_bounds is given.
        taper_profile: Profile of the tapers (see TAPER_PROFILES).
        taper_bounds: Shortest and longest tapers, if the taper length is optimized too.

    Returns:
        BendResult of the best design
    """
    bounds = [radius_bounds, bend_width_bounds, (0.0, 1.0), (0.0, 0.3)]
    start = [min(radius, 0.9 * max_effective_radius), width, 0.5, 0.0]
    deltas = [0.3 * start[0], 0.2 * width, 0.3, 0.05]
    if taper_bounds is not None:
        bounds.append(tuple(taper_bounds))
        start.append(float(np.clip(taper_length if taper_length > 0 else 0.5 * (taper_bounds[0] + taper_bounds[1]), *taper_bounds)))
        deltas.append(0.3 * (taper_bounds[1] - taper_bounds[0]))

    def evaluate(u):
        values = np.clip(u, [b[0] for b in bounds], [b[1] for b in bounds])
        taper = values[4] if taper_bounds is not None else taper_length
        return analyze_bend(stack, width, thickness, values[0], angle, values[1], values[2], values[3], taper_length=taper, taper_profile=taper_profile, n_taper_steps=10)

    def objective(u):
        result = evaluate(u)
        if result is None:
            return 1e3
        return result.loss + 10 * max(result.effective_radius / max_effective_radius - 1, 0.0)

    simplex = [start] + [[s + (d if i == j else 0.0) for j, s in enumerate(start)] for i, d in enumerate(deltas)]
    best = optimize.minimize(objective, start, method="Nelder-Mead", bounds=bounds, options=dict(maxfev=max_evaluations, xatol=1e-3, fatol=1e-4, initial_simplex=simplex))
    values = np.clip(best.x, [b[0] for b in bounds], [b[1] for b in bounds])
    return analyze_bend(stack, width, thickness, values[0], angle, values[1], values[2], values[3], taper_length=values[4] if taper_bounds is not None else taper_length,
                        taper_profile=taper_profile)


# ---------------------------------------------------------------------------------------------------------------------
# Grating couplers
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class GratingDesign:
    """Grating coupler at the end of a waveguide, which emits light up through a window in the electrodes onto a target
    (e.g. an ion) at a height above the trap surface and a distance along the direction of emission from the grating.

    The teeth are arcs in a fan of the given angular width which starts taper_length before the first tooth, curved so
    that they focus the light across the grating onto the target. Gratings can also be non-uniform: the fill factor can
    change linearly from fill_start at the first tooth to fill_factor at the last (apodization, which spreads the light
    more evenly over the grating than the exponential decay of a uniform grating), and the period of every tooth can be
    chirped so that every tooth emits towards the target (which focuses the light along the grating too).
    """

    period: float = 0.6
    fill_factor: float = 0.5
    n_teeth: int = 20
    etch_depth: float = 0.15
    angular_width: float = 15.0
    taper_length: float = 15.0
    thickness: float = 0.15
    target_height: float = 55.0
    target_distance: float = 41.25
    power: float = 1e-3
    fill_start: float = None
    chirp: bool = False

    @property
    def uniform(self):
        return self.fill_start is None and not self.chirp


@dataclass
class GratingResult:
    """Performance of a grating coupler, with the period and fill factor of every tooth."""

    design: GratingDesign
    n_eff: float
    emission_angle: float
    target_angle: float
    strength: float
    upward_fraction: float
    radiated_fraction: float
    efficiency: float
    intensity: float
    length: float
    periods: np.ndarray = field(default=None, repr=False)
    fills: np.ndarray = field(default=None, repr=False)

    @property
    def decay_length(self):
        """Length over which the guided power falls by 1/e (µm)."""
        return 1 / (2 * self.strength) if self.strength > 0 else np.inf


@dataclass
class ToothLayout:
    """Teeth of a grating along its waveguide: the start (µm, from the first tooth), period (µm), fill factor, effective
    index, tangential wavevector of the emitted light (1/µm), strength (1/µm), and upward fraction of every tooth.
    """

    starts: np.ndarray
    periods: np.ndarray
    fills: np.ndarray
    n_eff: np.ndarray
    k_t: np.ndarray
    strength: np.ndarray
    upward: np.ndarray

    @property
    def length(self):
        return float(self.starts[-1] + self.periods[-1])


def _grating_strength(stack, design):
    """Effective index of the grating region, the strength alpha (1/µm, of the field) of the first diffraction order,
    and the fraction of it which radiates up into vacuum.
    """
    k_0 = stack.k_0
    etch = min(design.etch_depth, design.thickness)
    eps_core = stack.n_core ** 2
    eps_gap = stack.n_cladding ** 2
    eps_average = design.fill_factor * eps_core + (1 - design.fill_factor) * eps_gap
    y, eps, h, n_pml = _vertical(stack, design.thickness, etch, eps_average, vacuum_above=True)
    lossless = np.where(np.abs(np.imag(eps)) > 0, eps_gap, eps)
    if _count_modes(lossless, h, k_0, "TE", stack.n_cladding) == 0:
        return None
    n_lossless, _ = _solve_mode(lossless, h, k_0, "TE", stack.n_core, n_pml)
    n_eff, mode = _solve_mode(eps, h, k_0, "TE", n_lossless.real, n_pml)
    beta = k_0 * n_eff.real

    # First-order perturbation of the average stack, radiating with the tangential wavevector beta - 2 pi / period
    k_t = beta - 2 * np.pi / design.period
    delta_eps = (eps_core - eps_gap) * np.sin(np.pi * design.fill_factor) / np.pi
    source = np.where((y > -etch) & (y < 0.0), -k_0 ** 2 * delta_eps * mode, 0.0)
    A = _operator(eps, h, k_0, "TE", n_pml) - k_t ** 2 * sparse.identity(len(y), format="csc")
    scattered = spsolve(A, source)

    def flux(index):
        derivative = (scattered[index + 1] - scattered[index - 1]) / (2 * h)
        return np.imag(np.conj(scattered[index]) * derivative)

    top = len(y) - n_pml[1] - 3
    bottom = np.searchsorted(y, -design.thickness - stack.box_thickness - 0.1)
    up = max(flux(top), 0.0) if abs(k_t) < k_0 else 0.0
    down = max(-flux(bottom), 0.0)
    guided = beta * np.sum(np.abs(mode) ** 2) * h
    strength = 0.5 * (up + down) / guided
    return n_eff.real, k_t, strength, up / (up + down) if up + down > 0 else 0.0


def tooth_layout(stack, design, n_samples=5):
    """Lay out the teeth of a grating (see ToothLayout), with the fill factor of every tooth, and its period chirped if
    the design asks for it, where the strength of teeth of each fill factor is interpolated between n_samples fill
    factors.

    Returns:
        ToothLayout, or None if the grating region does not guide light
    """
    n = max(int(design.n_teeth), 1)
    fills = np.full(n, design.fill_factor) if design.fill_start is None else np.linspace(design.fill_start, design.fill_factor, n)
    k_0 = stack.k_0
    if np.ptp(fills) < 1e-9:
        sample_fills = np.array([fills[0]])
    else:
        sample_fills = np.linspace(fills.min(), fills.max(), n_samples)
    samples = []
    for f in sample_fills:
        result = _grating_strength(stack, replace(design, fill_factor=float(f)))
        if result is None:
            return None
        samples.append(result)
    samples = np.array(samples)

    def interpolate(column):
        return np.interp(fills, sample_fills, samples[:, column]) if len(sample_fills) > 1 else np.full(n, samples[0, column])

    n_eff = interpolate(0)
    strength = interpolate(2)
    upward = interpolate(3)
    periods = np.full(n, float(design.period))
    if design.chirp:
        # Every tooth emits towards the target, from where it sits (with the grating centered on -target_distance)
        for _ in range(6):
            starts = np.concatenate(([0.0], np.cumsum(periods[:-1])))
            x = starts + 0.5 * periods - 0.5 * periods.sum() - design.target_distance
            sin_theta = -x / np.hypot(x, design.target_height)
            periods = stack.wavelength / np.maximum(n_eff - sin_theta, 0.05)
    starts = np.concatenate(([0.0], np.cumsum(periods[:-1])))
    k_t = k_0 * n_eff - 2 * np.pi / periods
    return ToothLayout(starts, periods, fills, n_eff, k_t, strength, upward)


def _aperture(design, layout, k_0, spacing):
    """Field emitted at the trap surface over the grating, which is centered at x = -target_distance, sampled finely
    enough along the grating to resolve the phase of the emission (so that no spurious beams appear).
    """
    length = layout.length
    fan = np.radians(design.angular_width)
    spacing_s = min(spacing, np.pi / (np.abs(layout.k_t).max() + 0.5 * k_0))
    s = np.arange(0.5 * spacing_s, length, spacing_s)
    tooth = np.clip(np.searchsorted(layout.starts, s, side="right") - 1, 0, len(layout.starts) - 1)
    strength = layout.strength[tooth]
    # Power left in the waveguide, and the phase of the emitted light, accumulated tooth by tooth
    decay = np.exp(-2 * np.cumsum(strength) * spacing_s + strength * spacing_s)
    phase_s = np.cumsum(layout.k_t[tooth]) * spacing_s - 0.5 * layout.k_t[tooth] * spacing_s
    max_width = (design.taper_length + length) * fan
    v = np.arange(-0.5 * max_width + 0.5 * spacing, 0.5 * max_width, spacing)
    S, V = np.meshgrid(s, v, indexing="ij")
    r = design.taper_length + S
    inside = np.abs(V) < 0.5 * r * fan
    emitted = (layout.upward[tooth] * 2 * strength * design.power * decay)[:, None] * 2 / (r * fan) * np.cos(np.pi * V / (r * fan)) ** 2
    amplitude = np.sqrt(np.where(inside, emitted, 0.0))
    distance = np.hypot(design.target_height, design.target_distance)
    phase = phase_s[:, None] - k_0 * (np.sqrt(V ** 2 + distance ** 2) - distance)
    x = S - 0.5 * length - design.target_distance
    return x, V, amplitude * np.exp(1j * phase), spacing * spacing_s


def grating_field(stack, design, points, layout=None, spacing=None):
    """Field (sqrt(W/µm^2)) emitted by a grating at points (µm) with shape (..., 3), by Rayleigh-Sommerfeld diffraction
    from the trap surface, where the target is at (0, 0, target_height).
    """
    k_0 = stack.k_0
    if layout is None:
        layout = tooth_layout(stack, design)
    spacing = spacing or stack.wavelength / 2
    x, v, aperture, area = _aperture(design, layout, k_0, spacing)
    x, v, aperture = x.ravel(), v.ravel(), aperture.ravel()
    keep = np.abs(aperture) > 0
    x, v, aperture = x[keep], v[keep], aperture[keep]
    points = np.asarray(points, dtype=float)
    flat = points.reshape(-1, 3)
    out = np.zeros(len(flat), dtype=complex)
    for start in range(0, len(flat), 256):
        p = flat[start:start + 256]
        rho = np.sqrt((p[:, None, 0] - x[None]) ** 2 + (p[:, None, 1] - v[None]) ** 2 + p[:, None, 2] ** 2)
        out[start:start + 256] = np.sum(aperture[None] * np.exp(1j * k_0 * rho) * p[:, None, 2] / rho ** 2, axis=1) * area / (1j * stack.wavelength)
    return out.reshape(points.shape[:-1])


def analyze_grating(stack, design):
    """Find the emission, efficiency, and intensity at the target (W/m^2) of a grating coupler (TE only).

    Args:
        stack: Stack of the chip (only TE polarization is modelled).
        design: GratingDesign.

    Returns:
        GratingResult, or None if the grating region does not guide light
    """
    layout = tooth_layout(stack, design)
    if layout is None:
        return None
    k_0 = stack.k_0
    # Light left in the waveguide at the start of each tooth, and the light each tooth radiates
    remaining = np.exp(-2 * np.concatenate(([0.0], np.cumsum(layout.strength * layout.periods))))
    radiated_teeth = remaining[:-1] - remaining[1:]
    radiated = 1 - remaining[-1]
    upward = float(np.sum(radiated_teeth * layout.upward) / radiated) if radiated > 0 else 0.0
    center = len(layout.k_t) // 2
    k_t = layout.k_t[center]
    emission = np.degrees(np.arcsin(k_t / k_0)) if abs(k_t) < k_0 else np.nan
    target = np.array([0.0, 0.0, design.target_height])
    intensity = np.abs(grating_field(stack, design, target, layout)) ** 2 * 1e12
    return GratingResult(design, float(np.mean(layout.n_eff)), emission, np.degrees(np.arctan2(design.target_distance, design.target_height)),
                         float(np.sum(layout.strength * layout.periods) / layout.length), upward, radiated, upward * radiated, float(intensity), layout.length,
                         layout.periods, layout.fills)


def phase_matched_period(stack, design):
    """Period which points the first diffraction order of a grating at its target."""
    result = _grating_strength(stack, design)
    if result is None:
        return design.period
    n_eff = result[0]
    sin_theta = design.target_distance / np.hypot(design.target_distance, design.target_height)
    return stack.wavelength / (n_eff - sin_theta)


def optimize_grating(stack, design, period_bounds=None, fill_bounds=(0.2, 0.8), teeth_bounds=(5, 80), angular_width_bounds=(4.0, 40.0), etch_bounds=None, max_evaluations=150,
                     fill_start_bounds=(0.05, 0.95)):
    """Choose the period, fill factor, number of teeth, angular width, and etch depth of a grating which maximize the
    intensity at its target, and the fill factor of its first tooth if it is apodized (fill_start is not None). The
    periods of chirped gratings follow from where their teeth are, so their period is not optimized.

    Args:
        stack: Stack of the chip.
        design: Initial GratingDesign, whose other fields are kept.
        period_bounds: Smallest and largest periods (around the phase-matched period by default).
        fill_bounds: Smallest and largest fill factors.
        teeth_bounds: Smallest and largest numbers of teeth.
        angular_width_bounds: Smallest and largest angular widths (degrees).
        etch_bounds: Smallest and largest etch depths (up to the thickness by default).
        max_evaluations: Largest number of designs to evaluate.
        fill_start_bounds: Smallest and largest fill factors of the first tooth of an apodized grating.

    Returns:
        GratingResult of the best design
    """
    period = phase_matched_period(stack, design) if not design.chirp else design.period
    period_bounds = period_bounds or (0.8 * period, 1.2 * period)
    etch_bounds = etch_bounds or (0.2 * design.thickness, design.thickness)
    names = ["fill_factor", "n_teeth", "angular_width", "etch_depth"]
    bounds = [fill_bounds, teeth_bounds, angular_width_bounds, etch_bounds]
    start = [design.fill_factor, design.n_teeth, design.angular_width, min(design.etch_depth, design.thickness)]
    if not design.chirp:
        names.insert(0, "period")
        bounds.insert(0, period_bounds)
        start.insert(0, period)
    if design.fill_start is not None:
        names.append("fill_start")
        bounds.append(fill_start_bounds)
        start.append(design.fill_start)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    def make(u):
        values = dict(zip(names, np.clip(u, lower, upper)))
        values["n_teeth"] = int(round(values["n_teeth"]))
        return replace(design, **values)

    def objective(u):
        result = analyze_grating(stack, make(u))
        if result is None or not result.intensity > 0:
            return 1e3
        return -np.log10(result.intensity)

    start = np.clip(start, lower, upper)
    steps = 0.25 * (upper - lower)
    n = len(start)
    simplex = [start] + [np.clip(start + np.where(np.arange(n) == i, steps[i] * (1 if start[i] + steps[i] <= upper[i] else -1), 0.0), lower, upper) for i in range(n)]
    best = optimize.minimize(objective, start, method="Nelder-Mead", bounds=list(zip(lower, upper)), options=dict(maxfev=max_evaluations, xatol=1e-4, fatol=1e-4, initial_simplex=simplex))
    return analyze_grating(stack, make(best.x))


def grating_teeth(design, layout=None):
    """Outlines of the teeth of a grating in the trap surface (µm), with the grating centered at x = -target_distance,
    as arcs of the fan which starts taper_length before the first tooth (using the period and fill factor of every
    tooth from a ToothLayout if one is given).
    """
    fan = np.radians(design.angular_width)
    n = int(design.n_teeth)
    if layout is None:
        starts = np.arange(n) * design.period
        periods = np.full(n, design.period)
        fills = np.full(n, design.fill_factor)
    else:
        starts, periods, fills = layout.starts, layout.periods, layout.fills
    length = float(starts[-1] + periods[-1])
    apex = -design.target_distance - 0.5 * length - design.taper_length
    angles = np.linspace(-0.5 * fan, 0.5 * fan, 40)
    teeth = []
    for q in range(n):
        r_in = design.taper_length + starts[q]
        r_out = r_in + fills[q] * periods[q]
        outer = np.stack([apex + r_out * np.cos(angles), r_out * np.sin(angles)], axis=-1)
        inner = np.stack([apex + r_in * np.cos(angles[::-1]), r_in * np.sin(angles[::-1])], axis=-1)
        teeth.append(np.concatenate([outer, inner]))
    taper = np.array([[apex, 0.0], [apex + design.taper_length * np.cos(0.5 * fan), design.taper_length * np.sin(0.5 * fan)],
                      [apex + design.taper_length * np.cos(0.5 * fan), -design.taper_length * np.sin(0.5 * fan)]])
    return teeth, taper
