"""Completion checks for Week 9.

The check verifies that the peak lag and maximum coherence values are computed
and lie within reasonable bounds.  The coherence should be between 0 and 1.
"""

import os
import sys
# Allow imports when running tests directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from demo import run_demo
from dsp_utils import generate_week_data
import numpy as np
from scipy.signal import correlation_lags, correlate


def test_multichannel_metrics() -> None:
    """Check basic properties of the multichannel cross-correlation and coherence demo."""
    results = run_demo()
    assert 'peak_lag_s' in results and 'max_coherence' in results, \
        "Missing peak_lag_s or max_coherence in results"
    assert abs(results['peak_lag_s']) < 0.01, \
        "Peak lag should be within ±10 ms for synthetic data"
    assert 0.0 <= results['max_coherence'] <= 1.0, \
        "Coherence must be between 0 and 1"

def test_time_shift_lag() -> None:
    """Introduce a time shift between the two channels and verify that the peak lag increases."""
    baseline = run_demo()
    baseline_lag = abs(baseline['peak_lag_s'])
    fs = 5000.0
    data = generate_week_data('week_09', {'fs': fs, 'duration': 1.0})
    x1 = data['x1']
    x2 = data['x2']
    # Introduce a 2 ms shift to x2
    shift_samples = int(0.002 * fs)
    x2_shifted = np.roll(x2, shift_samples)
    corr = correlate(x1, x2_shifted, mode='full')
    lags = correlation_lags(len(x1), len(x2_shifted), mode='full')
    peak_idx = np.argmax(corr)
    lag_samples = lags[peak_idx]
    lag_sec = lag_samples / fs
    assert abs(lag_sec) >= 0.002 - 1e-4, \
        "The introduced time shift should produce a peak lag close to the shift duration"
    assert abs(lag_sec) > baseline_lag, \
        "Shifted channels should yield a larger peak lag than the baseline"


if __name__ == '__main__':
    test_multichannel_metrics()
    test_time_shift_lag()
    print('All Week 9 checks passed.')