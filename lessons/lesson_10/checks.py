"""Completion checks for Week 10.

The check verifies that all five statistical features are computed and that
their values are finite.  Crest factor must be greater than or equal to 1
for any non‑zero signal.
"""

import os
import sys
# Make dsp_utils available when running tests directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from demo import run_demo
import numpy as np


def test_statistical_features() -> None:
    """Ensure that time-domain features are computed and finite."""
    results = run_demo()
    for key in ['RMS', 'peak', 'crest_factor', 'kurtosis', 'skewness']:
        assert key in results, f"Missing {key} in results"
        value = results[key]
        assert np.isfinite(value), f"{key} should be finite"
    # Crest factor should be ≥1 for any non-zero signal
    assert results['crest_factor'] >= 1.0, "Crest factor must be at least 1"

def test_feature_sensitivity() -> None:
    """Compare crest factor of the impulsive signal to that of a pure sine wave."""
    baseline = run_demo()
    baseline_cf = baseline['crest_factor']
    fs = 2000.0
    duration = 5.0
    t = np.arange(0, duration, 1.0 / fs)
    # Pure sine wave with amplitude 1 at 100 Hz
    sine = np.sin(2.0 * np.pi * 100.0 * t)
    rms = np.sqrt(np.mean(sine ** 2))
    peak = np.max(np.abs(sine))
    crest = peak / rms if rms > 0 else np.inf
    assert baseline_cf > crest * 1.2, \
        "Impulsive vibration should yield a crest factor significantly higher than that of a pure tone"


if __name__ == '__main__':
    test_statistical_features()
    test_feature_sensitivity()
    print('All Week 10 checks passed.')