"""Completion checks for Week 6.

The check verifies that PSD variance and bandpower values are computed.  It
ensures that bandpower in the target bands is positive and finite.
"""

import os
import sys
# Ensure dsp_utils can be imported when running this file directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from demo import run_demo
from dsp_utils import generate_week_data
import numpy as np
from scipy.signal import welch


def test_psd_bandpower() -> None:
    """Verify that PSD variance and bandpowers are computed and positive."""
    results = run_demo()
    assert 'variance_of_psd' in results, "Result must include variance_of_psd"
    assert results['variance_of_psd'] >= 0.0, "PSD variance should be non‑negative"
    for key in ['bandpower_50_70', 'bandpower_140_160', 'bandpower_240_260']:
        assert key in results, f"Missing {key} in results"
        assert results[key] > 0.0, f"{key} should be positive"

def test_low_resolution_variance() -> None:
    """Compute PSD with a longer segment length; variance should increase."""
    baseline = run_demo()
    baseline_var = baseline['variance_of_psd']
    fs = 8000.0
    data = generate_week_data('week_06', {'fs': fs, 'duration': 2.0})
    x = data['x']
    # Use a longer segment length than in the demo (fewer averages)
    _, psd_long = welch(x, fs=fs, window='hann', nperseg=1024, noverlap=512)
    variance_long = np.var(psd_long)
    assert variance_long > baseline_var, \
        "Using longer Welch segments should yield a larger PSD variance"


if __name__ == '__main__':
    test_psd_bandpower()
    test_low_resolution_variance()
    print('All Week 6 checks passed.')