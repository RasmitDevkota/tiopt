"""Shaped laser pulses, i.e. their envelopes, areas, and off-resonant excitation, and Mølmer-Sørensen gates made with
shaped pulses.

The envelopes are the same as those of the C library (see enum PulseShape in data_structures.h and PULSE_SHAPES), but
they are evaluated with numpy here so that whole pulses can be integrated at once. Ramped pulses take longer than square
pulses with the same peak Rabi frequency, but they excite far less off-resonant light (e.g. the carrier during a
Mølmer-Sørensen gate), which is what pulse shaping trades.
"""

import numpy as np
from scipy import optimize

from .lib import PULSE_SHAPES

# Number of samples which pulses are integrated over
N_SAMPLES = 4001


def shape_parameters(shape, ramp, duration=None):
    """Parameters (s) of a pulse shape from a single number: the rise and fall times (s) of trapezoidal and cosine
    pulses, or the width of a Gaussian pulse as a fraction of its duration (s), which Gaussian pulses need.
    """
    if shape in ("trapezoid", "cosine"):
        return (ramp, ramp)
    if shape == "gaussian":
        if duration is None:
            raise ValueError("Gaussian pulses need their duration, since their width is a fraction of it")
        return (ramp * duration, 0.0)
    return (0.0, 0.0)


def envelope(shape, parameters, duration, t):
    """Envelope (which peaks at 1) of a pulse with a shape in PULSE_SHAPES, its parameters (s), and duration (s) at times
    t (s) since it starts, the same as pulse_envelope in quantum.c.
    """
    if shape not in PULSE_SHAPES:
        raise ValueError(f"unknown pulse shape {shape!r}; choose from {', '.join(PULSE_SHAPES)}")
    t = np.asarray(t, dtype=float)
    parameters = tuple(parameters) + (0.0, 0.0)
    inside = (t >= 0) & (t <= duration)
    rise = min(max(parameters[0], 0.0), duration)
    fall = min(max(parameters[1], 0.0), duration - rise)
    f = np.ones_like(t)
    if shape in ("trapezoid", "cosine"):
        with np.errstate(divide="ignore", invalid="ignore"):
            up = np.clip(t / rise, 0, 1) if rise > 0 else np.ones_like(t)
            down = np.clip((duration - t) / fall, 0, 1) if fall > 0 else np.ones_like(t)
        if shape == "cosine":
            up = np.sin(0.5 * np.pi * up) ** 2
            down = np.sin(0.5 * np.pi * down) ** 2
        f = np.minimum(up, down)
    elif shape == "gaussian":
        sigma = parameters[0]
        edge = np.exp(-0.125 * duration ** 2 / sigma ** 2) if sigma > 0 else 1.0
        if edge <= 1 - 1e-12:
            f = (np.exp(-0.5 * ((t - 0.5 * duration) / sigma) ** 2) - edge) / (1 - edge)
    elif shape == "blackman":
        u = t / duration
        f = 0.42 - 0.5 * np.cos(2 * np.pi * u) + 0.08 * np.cos(4 * np.pi * u)
    return np.where(inside, f, 0.0)


def _samples(shape, parameters, duration):
    t = np.linspace(0.0, duration, N_SAMPLES)
    return t, envelope(shape, parameters, duration, t)


def area_fraction(shape, parameters, duration):
    """Area of a pulse relative to a square pulse with the same peak and duration."""
    t, f = _samples(shape, parameters, duration)
    return np.trapezoid(f, t) / duration


def shaped_pulse(shape, ramp, area, rabi_frequency):
    """Duration (s) and parameters (s) of a pulse with a peak Rabi frequency (rad/s) whose pulse area is area (rad), e.g.
    pi / 2 for a single-qubit gate, where ramp is the ramp time (s) of trapezoidal and cosine pulses (which are
    lengthened until they have room for both ramps if they need to be) or the width of a Gaussian pulse as a fraction of
    its duration (see shape_parameters).
    """
    square = area / rabi_frequency
    if shape == "square":
        return square, (0.0, 0.0)
    if shape in ("gaussian", "blackman"):
        # Their area is a fixed fraction of a square pulse's
        unit = shape_parameters(shape, ramp, 1.0)
        duration = square / area_fraction(shape, unit, 1.0)
        return duration, shape_parameters(shape, ramp, duration)
    parameters = shape_parameters(shape, ramp)

    def residual(duration):
        return rabi_frequency * duration * area_fraction(shape, parameters, duration) - area

    upper = square
    while residual(upper) < 0:
        upper *= 2
    duration = optimize.brentq(residual, square, upper, xtol=1e-6 * square) if residual(square) < 0 else square
    return duration, parameters


def duration_for_area(shape, ramp, area, rabi_frequency):
    """Duration (s) of a pulse whose pulse area is area (rad), see shaped_pulse."""
    return shaped_pulse(shape, ramp, area, rabi_frequency)[0]


def spectrum(shape, parameters, duration, detuning):
    """Fourier transform (s) of the envelope of a pulse at detunings (rad/s), i.e. the integral of f(t) e^(i detuning t).
    """
    t, f = _samples(shape, parameters, duration)
    detuning = np.atleast_1d(np.asarray(detuning, dtype=float))
    values = np.trapezoid(f[None, :] * np.exp(1j * detuning[:, None] * t[None, :]), t, axis=1)
    return values


def off_resonant_excitation(shape, parameters, duration, rabi_frequency, detuning):
    """Probability (to lowest order) that a pulse with a peak Rabi frequency (rad/s) leaves a transition detuned by
    detuning (rad/s) excited at its end, averaged over detunings spanning ten oscillations of the spectrum of the pulse
    (whose zeros a square pulse would otherwise land on or between).
    """
    detunings = detuning + 2 * np.pi / duration * np.linspace(-5, 5, 201)
    return float(np.mean(np.abs(0.5 * rabi_frequency * spectrum(shape, parameters, duration, detunings)) ** 2))


def carrier_suppression(shape, parameters, duration, detuning):
    """Off-resonant excitation of a pulse relative to that of a square pulse of the same peak and duration."""
    square = off_resonant_excitation("square", (0.0, 0.0), duration, 1.0, detuning)
    return off_resonant_excitation(shape, parameters, duration, 1.0, detuning) / square


def _ms_closure(shape, parameters, duration):
    """Detuning (rad/s) from the sidebands which closes the loop of the motion in phase space by the end of the pulse."""
    guess = 2 * np.pi / duration

    def residual(delta):
        return float(np.abs(spectrum(shape, parameters, duration, delta)[0]) / duration)

    deltas = np.linspace(0.5 * guess, 4 * guess, 141)
    values = np.array([residual(d) for d in deltas])
    minima = [i for i in range(1, len(values) - 1) if values[i] <= values[i - 1] and values[i] <= values[i + 1]]
    i = minima[0] if minima else int(np.argmin(values))
    result = optimize.minimize_scalar(residual, bounds=(deltas[max(i - 1, 0)], deltas[min(i + 1, len(deltas) - 1)]), method="bounded", options=dict(xatol=1e-10 * guess))
    return result.x, result.fun


def _ms_phase(shape, parameters, duration, delta):
    """Geometric phase of a Mølmer-Sørensen gate per (eta Omega)^2 (s^2), i.e. the integral over t > s of
    f(t) f(s) sin(delta (t - s)).
    """
    t, f = _samples(shape, parameters, duration)
    dt = t[1] - t[0]
    inner = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] * np.exp(-1j * delta * t[1:]) + f[:-1] * np.exp(-1j * delta * t[:-1])) * dt)))
    return float(np.imag(np.trapezoid(f * np.exp(1j * delta * t) * inner, t)))


def ms_gate(shape, ramp, eta, rabi_frequency):
    """Duration and detuning of a Mølmer-Sørensen gate made with two tones of the given pulse shape.

    The tones are detuned by delta from the red and blue sidebands, and the gate entangles two ions into a Bell state
    when the motion returns to its start (so that the spectrum of the envelope vanishes at delta) with a geometric phase
    of pi / 2, i.e. (eta Omega)^2 times the integral over t > s of f(t) f(s) sin(delta (t - s)). For square pulses this
    gives delta = 2 eta Omega and a duration of 2 pi / delta.

    Args:
        shape: Shape of the pulses (see PULSE_SHAPES).
        ramp: Ramp time (s) of trapezoidal and cosine pulses, which is kept as the gate gets longer or shorter, or the
            width of a Gaussian pulse as a fraction of its duration (see shape_parameters).
        eta: Lamb-Dicke parameter of each ion.
        rabi_frequency: Peak carrier Rabi frequency of each tone (rad/s).

    Returns:
        Duration (s), detuning delta from the sidebands (rad/s), and the parameters of the pulse shape (s)
    """
    coupling = eta * rabi_frequency
    square = np.pi / coupling
    if shape == "square" or (shape in ("trapezoid", "cosine", "gaussian") and ramp <= 0):
        return square, 2 * coupling, (0.0, 0.0)

    def parameters(duration):
        return shape_parameters(shape, ramp, duration)

    def residual(duration):
        delta, _ = _ms_closure(shape, parameters(duration), duration)
        return coupling ** 2 * _ms_phase(shape, parameters(duration), duration, delta) - 0.5 * np.pi

    lower = max(square, 2.0 * ramp if shape in ("trapezoid", "cosine") else 0.0) * 1.0001
    upper = 2 * lower
    while residual(upper) < 0:
        upper *= 1.5
        if upper > 1e3 * square:
            raise ValueError(f"no {shape} Mølmer-Sørensen gate with these pulses reaches a Bell state")
    if residual(lower) > 0:
        duration = lower
    else:
        duration = optimize.brentq(residual, lower, upper, xtol=1e-6 * square)
    return duration, _ms_closure(shape, parameters(duration), duration)[0], parameters(duration)


def ms_pulses(shape, ramp, eta, rabi_frequency, mode_frequency, start=0.0, targets=(1, 2), phase=0.0):
    """Two Pulse objects (see lib.Pulse) which make a Mølmer-Sørensen gate with shaped pulses (see ms_gate) on the
    sidebands of a mode of angular frequency mode_frequency (rad/s).
    """
    from .lib import Pulse

    duration, delta, parameters = ms_gate(shape, ramp, eta, rabi_frequency)
    return [
        Pulse(start, duration, mode_frequency + delta, rabi_frequency, phase, list(targets), eta, shape, parameters),
        Pulse(start, duration, -mode_frequency - delta, rabi_frequency, phase, list(targets), eta, shape, parameters),
    ]
