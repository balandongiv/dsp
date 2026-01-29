"""Week 5 demonstration: IIR filter design and stability.

This script designs a Chebyshev Type I band‑pass IIR filter to isolate a
low‑frequency oscillation (around 5 Hz) in a synthetic thermocouple signal.  It
computes the SNR before and after filtering and prints the filter’s pole
locations to verify stability (all poles should lie inside the unit circle).
"""

from typing import Dict, Tuple

import numpy as np
from scipy.signal import cheby1, lfilter, tf2zpk

import os
import sys
# Allow imports of dsp_utils when running this script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dsp_utils import generate_week_data
from dsp_utils.metrics import snr_db

# IIR filter specifications
ORDER = 4
PASSBAND = [3.0, 7.0]  # Hz
RIPPLE_DB = 1.0


def design_cheby_bandpass(fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Design a Chebyshev Type I band‑pass filter using bilinear transform.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    b, a : ndarray
        Numerator and denominator polynomials of the digital filter.
    """
    nyq = fs / 2.0
    wp = [PASSBAND[0] / nyq, PASSBAND[1] / nyq]
    b, a = cheby1(ORDER, RIPPLE_DB, wp, btype='bandpass')
    return b, a


def run_demo() -> Dict[str, float]:
    # Set random seed for reproducibility of synthetic data
    import numpy as np
    np.random.seed(0)
    """Run the Week 5 demonstration.

    Returns a dictionary with SNR before/after filtering and a boolean
    indicating stability.
    """
    fs = 100.0
    data = generate_week_data('week_05', {'fs': fs, 'duration': 10.0})
    x_clean = data['x_clean']
    x_noisy = data['x']
    # Design Chebyshev band‑pass filter
    b, a = design_cheby_bandpass(fs)
    # Apply filter
    x_filt = lfilter(b, a, x_noisy)
    # Also filter the clean signal to obtain a proper reference for SNR computation
    x_clean_filt = lfilter(b, a, x_clean)
    # Compute SNR (skip initial transient)
    skip = 100  # discard first samples to avoid filter transient
    # Baseline SNR relative to original clean signal
    snr_before = snr_db(x_clean[skip:], x_noisy[skip:] - x_clean[skip:])
    # After filtering, compute SNR relative to the band‑passed clean signal
    snr_after = snr_db(x_clean_filt[skip:], x_filt[skip:] - x_clean_filt[skip:])
    improvement = snr_after - snr_before
    # Check stability by examining poles
    z, p, k = tf2zpk(b, a)
    # Convert numpy boolean to Python bool for predictable type
    stable = bool(np.all(np.abs(p) < 1.0))
    print(f"SNR before filtering: {snr_before:.2f} dB")
    print(f"SNR after  filtering: {snr_after:.2f} dB")
    print(f"Improvement: {improvement:.2f} dB")
    print(f"Filter stability: {'stable' if stable else 'unstable'}")
    return {
        'snr_before': snr_before,
        'snr_after': snr_after,
        'improvement': improvement,
        'stable': stable,
    }


if __name__ == '__main__':
    run_demo()