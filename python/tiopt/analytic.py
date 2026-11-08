"""Analytic models of laser-ion interactions (like the analytic functions of IonSim.jl), and fits of them to measured
data, e.g. to find the temperature of an ion from its sideband flopping.

Rabi flopping on the carrier or the s-th sideband couples |g, n> to |e, n + s> with the Rabi frequency
    Omega_{n,n+s} = Omega exp(-eta^2 / 2) eta^|s| sqrt(n_<! / n_>!) L_{n_<}^{|s|}(eta^2),
which includes every order in the Lamb-Dicke parameter (D. J. Wineland et al., J. Res. NIST 103, 259 (1998)), and a
thermal state averages over n. A Mølmer-Sørensen interaction in the Lamb-Dicke and resolved sideband regime is
U = D(alpha(t) S) exp(i chi(t) S^2) with S = sigma_y^1 + sigma_y^2 (as in tiopt.gates), and the motion of a thermal
state averages each coherence between eigenstates s and s' of S by exp(-|alpha|^2 (s - s')^2 (n_bar + 1/2)).
"""

import numpy as np
from scipy import optimize, special


def rabi_frequencies(rabi_frequency, eta, n, sideband=0):
    """Rabi frequencies (rad/s) between |g, n> and |e, n + sideband> for Fock states n (array)."""
    n = np.asarray(n)
    s = abs(int(sideband))
    low = n if sideband >= 0 else n - s
    high = low + s
    valid = low >= 0
    low_safe = np.where(valid, low, 0)
    x = eta ** 2
    log_ratio = 0.5 * (special.gammaln(low_safe + 1) - special.gammaln(low_safe + s + 1))
    values = rabi_frequency * np.exp(-x / 2) * eta ** s * np.exp(log_ratio) * special.eval_genlaguerre(low_safe, s, x)
    return np.where(valid, np.abs(values), 0.0)


def thermal_distribution(n_bar, n_max):
    """Probabilities of the Fock states 0 to n_max of a thermal state with mean phonon number n_bar."""
    n = np.arange(n_max + 1)
    if n_bar <= 0:
        return (n == 0).astype(float)
    return (n_bar / (n_bar + 1)) ** n / (n_bar + 1)


def rabi_flop(times, rabi_frequency, eta=0.0, n_bar=0.0, sideband=0, detuning=0.0, n_max=None):
    """Excited population of an ion in a thermal state driven on its carrier (sideband 0) or a sideband, to every order
    in the Lamb-Dicke parameter.

    Args:
        times: Times (s).
        rabi_frequency: Rabi frequency of the carrier at eta = 0 (rad/s).
        eta: Lamb-Dicke parameter.
        n_bar: Mean phonon number of the thermal state.
        sideband: Order of the sideband (e.g. 1 for the blue and -1 for the red sideband).
        detuning: Detuning from the sideband (rad/s).
        n_max: Highest Fock state included (enough for 1e-6 of the distribution by default).

    Returns:
        Excited population at each time
    """
    times = np.asarray(times, dtype=float)
    if n_max is None:
        n_max = int(max(20, np.ceil(np.log(1e-6) / np.log(n_bar / (n_bar + 1)))) if n_bar > 0 else 20)
    n = np.arange(n_max + 1)
    p = thermal_distribution(n_bar, n_max)
    omega = rabi_frequencies(rabi_frequency, eta, n, sideband)
    generalized = np.sqrt(omega ** 2 + detuning ** 2)
    amplitude = np.divide(omega ** 2, generalized ** 2, out=np.zeros_like(omega), where=generalized > 0)
    return np.sum(p[None, :] * amplitude[None, :] * np.sin(generalized[None, :] * times[:, None] / 2) ** 2, axis=1)


def ms_populations(times, rabi_frequency, eta, detuning, n_bar=0.0):
    """Populations of |gg>, of |ee>, and of |ge> plus |eg> during a Mølmer-Sørensen interaction of constant Rabi
    frequency on two ions starting in |gg>, with tones detuned by detuning (rad/s) from the sidebands of a thermal mode
    (in the Lamb-Dicke and resolved sideband regime).

    Returns:
        Arrays of P_gg, P_ee, and P_ge + P_eg at each time
    """
    times = np.asarray(times, dtype=float)
    force = eta * rabi_frequency / 2
    alpha = force / detuning * (np.exp(1j * detuning * times) - 1)
    chi = force ** 2 * (times / detuning - np.sin(detuning * times) / detuning ** 2)
    # Eigenstates of S = sigma_y1 + sigma_y2 with eigenvalues 2, 0, 0, -2 in the basis gg, ge, eg, ee
    plus = np.array([1, 1j]) / np.sqrt(2)
    minus = np.array([1, -1j]) / np.sqrt(2)
    states = [np.kron(a, b) for a in (plus, minus) for b in (plus, minus)]
    values = np.array([2, 0, 0, -2])
    initial = np.array([1, 0, 0, 0], dtype=complex)
    c = np.array([np.vdot(state, initial) for state in states])
    out = np.zeros((len(times), 3))
    for k, (a, x) in enumerate(zip(alpha, chi)):
        rho = np.zeros((4, 4), dtype=complex)
        for p in range(4):
            for q in range(4):
                coherence = c[p] * np.conj(c[q]) * np.exp(1j * x * (values[p] ** 2 - values[q] ** 2))
                coherence *= np.exp(-abs(a) ** 2 * (values[p] - values[q]) ** 2 * (n_bar + 0.5))
                rho += coherence * np.outer(states[p], np.conj(states[q]))
        out[k] = [rho[0, 0].real, rho[3, 3].real, rho[1, 1].real + rho[2, 2].real]
    return out[:, 0], out[:, 1], out[:, 2]


def sideband_n_bar(red, blue):
    """Mean phonon number of a thermal state from the excitations on the red and blue sidebands (at the same short
    probe time), n_bar = r / (1 - r) with r = red / blue.
    """
    r = np.asarray(red, dtype=float) / np.asarray(blue, dtype=float)
    return r / (1 - r)


def fit_rabi_flop(times, populations, sideband=0, eta=0.1, rabi_guess=None, n_bar_guess=0.5, fit_eta=False, fit_detuning=False):
    """Fit measured Rabi flopping on the carrier or a sideband with rabi_flop.

    Args:
        times: Times (s).
        populations: Measured excited populations.
        sideband: Order of the sideband which was driven.
        eta: Lamb-Dicke parameter (or its starting guess if fit_eta).
        rabi_guess: Starting guess of the Rabi frequency (rad/s), from the first maximum by default.
        n_bar_guess: Starting guess of the mean phonon number.
        fit_eta: Specifies whether to fit the Lamb-Dicke parameter too.
        fit_detuning: Specifies whether to fit a detuning too.

    Returns:
        Dict of the fitted rabi_frequency (rad/s), n_bar, eta, and detuning (rad/s), with their standard errors under
        "errors", and the residual root mean square under "rms"
    """
    times = np.asarray(times, dtype=float)
    populations = np.asarray(populations, dtype=float)
    if rabi_guess is None:
        # The first time the population reaches its largest value is about half of a Rabi period
        peak = times[int(np.argmax(populations))]
        rabi_guess = np.pi / max(peak, times[-1] / 100)
        if sideband != 0:
            rabi_guess /= max(eta, 1e-3) * np.sqrt(max(n_bar_guess, 0) + 1)
    names = ["rabi_frequency", "n_bar"] + (["eta"] if fit_eta else []) + (["detuning"] if fit_detuning else [])
    x0 = [rabi_guess, n_bar_guess] + ([eta] if fit_eta else []) + ([0.0] if fit_detuning else [])
    lower = [0.0, 0.0] + ([1e-4] if fit_eta else []) + ([-np.inf] if fit_detuning else [])
    upper = [np.inf, 100.0] + ([2.0] if fit_eta else []) + ([np.inf] if fit_detuning else [])
    scale = np.array([max(rabi_guess, 1.0), 1.0] + ([0.1] if fit_eta else []) + ([max(rabi_guess, 1.0)] if fit_detuning else []))

    def model(x):
        values = dict(zip(names, x * scale))
        return rabi_flop(times, values["rabi_frequency"], values.get("eta", eta), values["n_bar"], sideband, values.get("detuning", 0.0))

    result = optimize.least_squares(lambda x: model(x) - populations, np.array(x0) / scale,
                                    bounds=(np.array(lower) / scale, np.array(upper) / scale))
    fitted = dict(zip(names, result.x * scale))
    residual = model(result.x) - populations
    dof = max(len(times) - len(names), 1)
    try:
        covariance = np.linalg.inv(result.jac.T @ result.jac) * np.sum(residual ** 2) / dof
        errors = dict(zip(names, np.sqrt(np.diag(covariance)) * scale))
    except np.linalg.LinAlgError:
        errors = {name: np.nan for name in names}
    fitted.setdefault("eta", eta)
    fitted.setdefault("detuning", 0.0)
    fitted["errors"] = errors
    fitted["rms"] = float(np.sqrt(np.mean(residual ** 2)))
    return fitted
