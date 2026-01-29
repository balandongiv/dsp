"""Completion checks for Week 3.

The check verifies that the leakage ratio is computed and is a valid
floating‑point value between 0 and 1.  Lower values indicate less
spectral leakage.
"""

import os
import sys
# Ensure the package root is importable when this test is run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from demo import run_demo
from dsp_utils import generate_week_data
import numpy as np
from scipy.signal import get_window


def test_leakage_ratio() -> None:
    results = run_demo()
    assert 'leakage_ratio' in results, "Result must include 'leakage_ratio'"
    ratio = results['leakage_ratio']
    assert isinstance(ratio, float), "Leakage ratio must be a float"
    assert 0.0 <= ratio <= 1.0, "Leakage ratio must be between 0 and 1"


def test_rectangular_window_leakage() -> None:
    """Compare leakage ratio using a rectangular window against the baseline Hann window.

    A rectangular (boxcar) window has poorer spectral concentration than a Hann
    window, so the leakage ratio should be higher.
    """
    baseline = run_demo()
    baseline_ratio = baseline['leakage_ratio']
    # Generate the same synthetic signal as the demo uses
    fs = 5000.0
    data = generate_week_data('week_03', {'fs': fs, 'duration': 0.5})
    x = data['x']
    # Use the same length as in the demo for comparability
    window_length = 1024
    window = get_window('boxcar', window_length)
    # Pad or truncate the signal
    if len(x) < window_length:
        x_pad = np.pad(x, (0, window_length - len(x)))
    else:
        x_pad = x[:window_length]
    x_win = x_pad * window
    spectrum = np.abs(np.fft.rfft(x_win))
    power = spectrum ** 2
    total_power = power.sum()
    # Sum the power of the three largest bins
    peak_indices = np.argsort(power)[-3:]
    peak_power = power[peak_indices].sum()
    leakage_ratio = 1.0 - (peak_power / total_power)
    assert leakage_ratio > baseline_ratio, \
        "Rectangular window should produce a larger leakage ratio than the Hann window baseline"


if __name__ == '__main__':
    test_leakage_ratio()
    test_rectangular_window_leakage()
    print('All Week 3 checks passed.')