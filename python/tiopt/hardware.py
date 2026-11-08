"""Hardware between a waveform and the electrodes, i.e. the digital-to-analog converters (DACs), which update at a
sample rate with a finite resolution and range, and the low-pass filters on every electrode, along with the
pre-compensation of waveforms for those filters (like the filter transforms of pytrans and the post-processing of the
shuttling toolchain of Conta et al., arXiv:2601.08495).

The filters are linear and time-invariant, so the voltage an electrode sees is the convolution of the DAC output with
the impulse response of its filter. Pre-compensation inverts that convolution: it finds the DAC samples whose filtered
output follows the waveform as closely as possible, while keeping the samples smooth (a small penalty on their
differences, since an exact inverse of a low-pass filter amplifies noise) and within the range of the DAC. The waveform
is padded with copies of its first and last samples, so that the compensated samples can lead the waveform and the
output can settle.
"""

from dataclasses import dataclass

import numpy as np
from scipy import linalg, optimize, signal

from .lib import Waveform


@dataclass
class DAC:
    """Digital-to-analog converter which updates at a sample rate (samples/s) with a number of bits over a range of
    voltages (V), holding each sample until the next.
    """
    sample_rate: float = 100e6
    bits: int = 16
    v_min: float = -10.0
    v_max: float = 10.0

    @property
    def lsb(self):
        """Voltage of the least significant bit (V)."""
        return (self.v_max - self.v_min) / (2 ** self.bits - 1)

    def quantize(self, voltages):
        """Clip voltages to the range of the DAC and round them to its nearest level."""
        clipped = np.clip(np.asarray(voltages, dtype=float), self.v_min, self.v_max)
        return self.v_min + np.round((clipped - self.v_min) / self.lsb) * self.lsb

    def sample(self, times, voltages, duration=None):
        """Sample voltages given at times (s) at the update rate of the DAC (interpolating linearly between them).

        Returns:
            Times of the samples (s) and the sampled voltages with shape (n_samples, n_electrodes)
        """
        times = np.asarray(times, dtype=float)
        voltages = np.asarray(voltages, dtype=float).reshape(len(times), -1)
        duration = times[-1] - times[0] if duration is None else duration
        n = max(int(np.floor(duration * self.sample_rate + 1e-9)) + 1, 1)
        sample_times = times[0] + np.arange(n) / self.sample_rate
        sampled = np.column_stack([np.interp(sample_times, times, voltages[:, e]) for e in range(voltages.shape[1])])
        return sample_times, sampled


class Filter:
    """Linear low-pass filter on an electrode, given by its transfer function (in s) or by its impulse response."""

    def __init__(self, numerator=None, denominator=None, impulse_response=None, dt=None, name="filter"):
        self.numerator = numerator
        self.denominator = denominator
        self.response = None if impulse_response is None else np.asarray(impulse_response, dtype=float)
        self.response_dt = dt
        self.name = name

    @classmethod
    def rc(cls, cutoff):
        """First-order RC low-pass filter with a cutoff frequency (Hz)."""
        omega = 2 * np.pi * cutoff
        return cls([omega], [1.0, omega], name=f"RC ({cutoff / 1e6:g} MHz)")

    @classmethod
    def butterworth(cls, cutoff, order=2):
        """Butterworth low-pass filter of an order with a cutoff frequency (Hz)."""
        b, a = signal.butter(order, 2 * np.pi * cutoff, analog=True)
        return cls(b, a, name=f"Butterworth order {order} ({cutoff / 1e6:g} MHz)")

    @classmethod
    def from_step_response(cls, times, step):
        """Filter whose response to a unit step was measured (e.g. on an oscilloscope) at evenly spaced times (s), whose
        differences give the impulse response (normalized so that the filter passes DC unchanged).
        """
        times = np.asarray(times, dtype=float)
        step = np.asarray(step, dtype=float)
        response = np.diff(np.concatenate([[0.0], step]))
        response /= response.sum()
        return cls(impulse_response=response, dt=times[1] - times[0], name="measured")

    def kernel(self, dt, n):
        """Discrete impulse response for samples held for dt (s) each, i.e. the n weights h_j by which sample k
        contributes to the output at the end of sample k + j (which sum to nearly 1 for long kernels).
        """
        edges = np.arange(n + 1) * dt
        if self.response is not None:
            # The step response of the measured filter, interpolated onto the sample period
            step_times = np.arange(len(self.response)) * self.response_dt
            step = np.cumsum(self.response)
            values = np.interp(edges, step_times, step, left=0.0, right=step[-1])
        else:
            _, values = signal.step(signal.lti(self.numerator, self.denominator), T=edges)
        return np.diff(values)

    def apply(self, voltages, dt):
        """Filter samples which are each held for dt (s), starting from the steady state of the first sample.

        Args:
            voltages: Samples with shape (n_samples, n_electrodes).
            dt: Sample period (s).

        Returns:
            Filtered voltages at the ends of the samples, with the same shape
        """
        voltages = np.asarray(voltages, dtype=float)
        n = len(voltages)
        kernel = self.kernel(dt, n)
        out = np.empty_like(voltages)
        for e in range(voltages.shape[1]):
            v = voltages[:, e] - voltages[0, e]
            out[:, e] = voltages[0, e] + np.convolve(v, kernel)[:n]
        return out


def convolution_matrix(kernel, n):
    """Lower triangular matrix which convolves n samples with a kernel."""
    column = np.zeros(n)
    m = min(len(kernel), n)
    column[:m] = kernel[:m]
    return linalg.toeplitz(column, np.zeros(n))


def precompensate(voltages, dt, filter, v_min=-10.0, v_max=10.0, smoothing=1e-4, pad_before=0, pad_after=None):
    """Find DAC samples whose filtered output follows a waveform (sampled at the DAC period dt) as closely as possible.

    Each electrode is compensated separately by minimizing ||K u - v||^2 + smoothing ||D u||^2 subject to
    v_min <= u <= v_max, where K convolves with the impulse response of the filter, D takes differences of neighboring
    samples, and v is the waveform padded with copies of its first and last samples.

    Args:
        voltages: Waveform sampled at the DAC period, with shape (n_samples, n_electrodes).
        dt: Sample period (s).
        filter: Filter on the electrodes.
        v_min: Lowest voltage of the DAC (V).
        v_max: Highest voltage of the DAC (V).
        smoothing: Weight of the differences of the samples.
        pad_before: Number of samples before the waveform, which let the compensation start early.
        pad_after: Number of samples after the waveform, which let the output settle (five time constants of the
            filter by default).

    Returns:
        Compensated samples with shape (n_samples + pad_before + pad_after, n_electrodes), and the number of padding
        samples before and after
    """
    voltages = np.asarray(voltages, dtype=float)
    if pad_after is None:
        kernel = filter.kernel(dt, 100000)
        settle = np.searchsorted(np.cumsum(kernel), 1 - 1e-3)
        pad_after = int(min(settle + 1, 2000))
    padded = np.vstack([np.repeat(voltages[:1], pad_before, axis=0), voltages, np.repeat(voltages[-1:], pad_after, axis=0)])
    n = len(padded)
    K = convolution_matrix(filter.kernel(dt, n), n)
    D = np.diff(np.eye(n), axis=0)
    # The filter starts in the steady state of the first sample, so every electrode is compensated relative to it
    offsets = padded[0]
    targets = padded - offsets
    normal = K.T @ K + smoothing * (D.T @ D)
    factor = linalg.cho_factor(normal)
    out = linalg.cho_solve(factor, K.T @ targets) + offsets
    # Electrodes whose compensated samples leave the range of the DAC are solved again with bounds
    weight = np.sqrt(smoothing)
    for e in range(padded.shape[1]):
        if out[:, e].min() < v_min - 1e-12 or out[:, e].max() > v_max + 1e-12:
            system = np.vstack([K, weight * D])
            rhs = np.concatenate([targets[:, e], np.zeros(n - 1)])
            result = optimize.lsq_linear(system, rhs, bounds=(v_min - offsets[e], v_max - offsets[e]))
            out[:, e] = result.x + offsets[e]
    return out, pad_before, pad_after


@dataclass
class HardwareResult:
    """Waveform through the hardware: the DAC sample times (s), the samples the DAC outputs (V), the voltages the
    electrodes see after the filters (V), the target waveform at the sample times (V), and the number of padding samples
    before and after it.
    """
    times: np.ndarray
    samples: np.ndarray
    output: np.ndarray
    target: np.ndarray
    pad_before: int
    pad_after: int

    @property
    def max_error(self):
        """Largest difference between the output and the target (V)."""
        return float(np.abs(self.output - self.target).max())

    def waveform(self, use_output=True):
        """Waveform of the voltages which the electrodes see (or of the DAC samples), e.g. to simulate the ion."""
        return Waveform.from_voltages(self.times, self.output if use_output else self.samples)


def through_hardware(waveform, dac=None, filter=None, compensate=False, smoothing=1e-4, pad_before=None, pad_after=None):
    """Send a waveform through a DAC and filter, optionally pre-compensating it for the filter.

    Args:
        waveform: Waveform.
        dac: DAC (a 100 MS/s, 16 bit, ±10 V DAC by default).
        filter: Filter on every electrode, or None for none.
        compensate: Specifies whether to pre-compensate the waveform for the filter.
        smoothing: Weight of the differences of the compensated samples.
        pad_before: Number of samples before the waveform, which let a compensated waveform lead the filter (as many as
            the filter takes to settle by default when compensating, and none otherwise).
        pad_after: Number of samples after the waveform (enough for the filter to settle by default).

    Returns:
        HardwareResult
    """
    dac = DAC() if dac is None else dac
    dt = 1.0 / dac.sample_rate
    times, sampled = dac.sample(waveform.times, waveform.voltages)
    settle = 0 if filter is None else int(min(np.searchsorted(np.cumsum(filter.kernel(dt, 100000)), 1 - 1e-3) + 1, 2000))
    if pad_before is None:
        pad_before = settle if (filter is not None and compensate) else 0
    if filter is not None and compensate:
        samples, pad_before, pad_after = precompensate(sampled, dt, filter, dac.v_min, dac.v_max, smoothing, pad_before, pad_after)
    else:
        if pad_after is None:
            pad_after = settle
        samples = np.vstack([np.repeat(sampled[:1], pad_before, axis=0), sampled, np.repeat(sampled[-1:], pad_after, axis=0)])
    samples = dac.quantize(samples)
    target = np.vstack([np.repeat(sampled[:1], pad_before, axis=0), sampled, np.repeat(sampled[-1:], pad_after, axis=0)])
    output = filter.apply(samples, dt) if filter is not None else samples.copy()
    all_times = times[0] + (np.arange(len(samples)) - pad_before) * dt
    return HardwareResult(all_times, samples, output, target, pad_before, pad_after)
