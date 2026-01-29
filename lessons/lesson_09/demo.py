"""Week 9 demonstration: multichannel cross‑correlation and coherence.

This script generates two correlated signals with shared and unique components,
computes the cross‑correlation to estimate time delay and uses
`scipy.signal.coherence` to compute magnitude‑squared coherence as a function
of frequency.  It reports the lag at which cross‑correlation peaks and the
maximum coherence value.
"""

from typing import Dict

import numpy as np
from scipy.signal import coherence, correlation_lags, correlate

import os
import sys
# Ensure dsp_utils module is importable when running this script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dsp_utils import generate_week_data


def run_demo() -> Dict[str, float]:
    # Seed RNG for reproducible correlated signals
    import numpy as np
    np.random.seed(0)
    fs = 5000.0
    data = generate_week_data('week_09', {'fs': fs, 'duration': 1.0})
    x1 = data['x1']
    x2 = data['x2']
    # Compute cross‑correlation and find lag of maximum correlation
    corr = correlate(x1, x2, mode='full')
    lags = correlation_lags(len(x1), len(x2), mode='full')
    max_idx = np.argmax(corr)
    peak_lag_samples = lags[max_idx]
    peak_lag_sec = peak_lag_samples / fs
    # Compute magnitude‑squared coherence
    f, coh = coherence(x1, x2, fs=fs, nperseg=1024)
    max_coh = np.max(coh)
    print(f"Peak cross‑correlation lag: {peak_lag_sec:.6f} s")
    print(f"Maximum coherence: {max_coh:.3f}")
    return {
        'peak_lag_s': float(peak_lag_sec),
        'max_coherence': float(max_coh),
    }


if __name__ == '__main__':
    run_demo()