"""Week 3 demonstration: FFT and spectral leakage.

This script computes the FFT of a multi‑tone signal and estimates spectral
leakage as the fraction of spectral energy that lies outside a small set of
dominant frequency bins.  Learners can experiment with different window
functions and lengths by modifying the `WINDOW_TYPE` and `WINDOW_LENGTH`
variables.
"""

from typing import Dict

import numpy as np
from scipy.signal import get_window

import os
import sys
# Add package root to sys.path so dsp_utils can be imported when running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dsp_utils import generate_week_data

# FFT parameters
WINDOW_TYPE = 'hann'
WINDOW_LENGTH = 1024


def run_demo() -> Dict[str, float]:
    # Use a fixed seed to make the synthetic signal and window noise reproducible
    import numpy as np
    np.random.seed(0)
    """Run the Week 3 demonstration.

    Returns a dictionary with the leakage ratio.  A lower leakage ratio
    indicates that most of the spectral energy is concentrated in the
    main bins.
    """
    # Generate a synthetic multi‑tone signal
    data = generate_week_data('week_03', {'fs': 5000.0, 'duration': 0.5})
    x = data['x']
    fs = 5000.0
    # Apply a window function to the signal
    window = get_window(WINDOW_TYPE, WINDOW_LENGTH)
    # Use the first WINDOW_LENGTH samples (zero pad if needed)
    if len(x) < WINDOW_LENGTH:
        x_pad = np.pad(x, (0, WINDOW_LENGTH - len(x)))
    else:
        x_pad = x[:WINDOW_LENGTH]
    x_win = x_pad * window
    # Compute the one‑sided FFT magnitude
    spectrum = np.abs(np.fft.rfft(x_win))
    power = spectrum ** 2
    total_power = power.sum()
    # Find the indices of the largest peaks
    peak_indices = np.argsort(power)[-3:]
    peak_power = power[peak_indices].sum()
    leakage_ratio = 1.0 - (peak_power / total_power)
    print(f"Leakage ratio with {WINDOW_TYPE} window: {leakage_ratio:.3f}")
    return {'leakage_ratio': float(leakage_ratio)}


if __name__ == '__main__':
    run_demo()