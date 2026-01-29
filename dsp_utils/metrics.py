"""Common metric computations for DSP experiments.

The functions in this module provide straightforward implementations of
signal‑to‑noise ratio, root mean square error and bandpower estimation.
These metrics are used across the weekly demonstrations to quantify
improvements and diagnose failures.
"""

from typing import Tuple

import numpy as np
from scipy.signal import welch


def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Compute the signal‑to‑noise ratio in decibels.

    Parameters
    ----------
    signal : numpy.ndarray
        The clean or reference signal.
    noise : numpy.ndarray
        The noise component (e.g. signal − reference).

    Returns
    -------
    snr : float
        Signal‑to‑noise ratio in dB.
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return np.inf
    return 10.0 * np.log10(signal_power / noise_power)


def rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute root mean square error between a reference and an estimate.

    Parameters
    ----------
    reference : numpy.ndarray
        Ground‑truth or reference signal.
    estimate : numpy.ndarray
        Estimated or reconstructed signal.

    Returns
    -------
    error : float
        Root mean square error.
    """
    return np.sqrt(np.mean((reference - estimate) ** 2))


def bandpower(signal: np.ndarray, fs: float, fmin: float, fmax: float,
              window: str = 'hanning', nperseg: Optional[int] = None) -> float:
    """Compute the average power of a signal in a specific frequency band.

    This function uses Welch’s method to estimate the power spectral density
    and then integrates it over the desired band.

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal.
    fs : float
        Sampling frequency in Hz.
    fmin : float
        Lower bound of the frequency band.
    fmax : float
        Upper bound of the frequency band.
    window : str
        Window type for Welch’s method.
    nperseg : int, optional
        Length of each segment for Welch’s method.  Defaults to 256.

    Returns
    -------
    power : float
        Average power in the specified band.
    """
    if nperseg is None:
        nperseg = 256
    f, psd = welch(signal, fs=fs, window=window, nperseg=nperseg, noverlap=nperseg // 2)
    # Select band indices
    idx = np.logical_and(f >= fmin, f <= fmax)
    # Integrate PSD over the band
    return np.trapz(psd[idx], f[idx])