"""Week 6 demonstration: PSD estimation and bandpower.

This script computes the power spectral density (PSD) of a multi‑tone signal
using Welch’s method, calculates the variance of the PSD estimate and
computes bandpower in three bands corresponding to the synthetic components.
Learners can adjust the segment length and window type to explore the
variance–resolution trade‑off.
"""

from typing import Dict

import numpy as np
from scipy.signal import welch

import os
import sys
# Add the package root so dsp_utils can be resolved when this script is executed directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dsp_utils import generate_week_data

# Welch parameters
N_PER_SEG = 512
OVERLAP = N_PER_SEG // 2
WINDOW = 'hann'


def run_demo() -> Dict[str, float]:
    # Seed numpy RNG for deterministic PSD computation
    import numpy as np
    np.random.seed(0)
    fs = 8000.0
    data = generate_week_data('week_06', {'fs': fs, 'duration': 2.0})
    x = data['x']
    # Compute PSD using Welch’s method
    f, psd = welch(x, fs=fs, window=WINDOW, nperseg=N_PER_SEG, noverlap=OVERLAP)
    # Variance of PSD estimate as a proxy for stability
    variance_psd = np.var(psd)
    # Compute bandpowers in three frequency bands
    def compute_bandpower(fmin: float, fmax: float) -> float:
        """Compute bandpower even when only a single PSD bin falls in the band.

        Welch’s method returns PSD estimates on a discrete frequency grid.  If
        the band defined by (fmin, fmax) contains only a single frequency bin,
        trapezoidal integration yields zero.  In that case, approximate the
        bandpower as the PSD value times the frequency resolution.
        """
        idx = np.logical_and(f >= fmin, f <= fmax)
        if not np.any(idx):
            return 0.0
        # If only one bin is present, multiply by the bin width
        if np.sum(idx) < 2:
            # Frequency resolution from adjacent bins
            df = f[1] - f[0] if len(f) > 1 else 0.0
            return psd[idx][0] * df
        return np.trapz(psd[idx], f[idx])
    band1 = compute_bandpower(50.0, 70.0)
    band2 = compute_bandpower(140.0, 160.0)
    band3 = compute_bandpower(240.0, 260.0)
    print(f"PSD variance: {variance_psd:.4e}")
    print(f"Bandpower 50–70 Hz: {band1:.4e}")
    print(f"Bandpower 140–160 Hz: {band2:.4e}")
    print(f"Bandpower 240–260 Hz: {band3:.4e}")
    return {
        'variance_of_psd': float(variance_psd),
        'bandpower_50_70': float(band1),
        'bandpower_140_160': float(band2),
        'bandpower_240_260': float(band3),
    }


if __name__ == '__main__':
    run_demo()